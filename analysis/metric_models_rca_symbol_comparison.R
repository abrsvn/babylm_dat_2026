#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(stringr)
})

options(width = 120, contrasts = c("contr.sum", "contr.poly"))
emm_options(lmer.df = "satterthwaite", lmerTest.limit = 200000, pbkrtest.limit = 0)

script_dir <- dirname(sub("--file=", "", commandArgs()[grep("--file=", commandArgs())]))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."
source(file.path(script_dir, "mixed_effects_utils.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7) {
  cat("Usage: Rscript metric_models_rca_symbol_comparison.R <long_scores.csv> <classification_pseudo_items.csv> <zero_shot_submeasures.csv> <output_dir> <run_tag> <expected_seeds> <symbol_modes_csv>\n")
  quit(status = 1)
}

long_scores_path <- args[1]
classification_pseudo_items_path <- args[2]
zero_shot_submeasures_path <- args[3]
output_dir <- args[4]
run_tag <- args[5]
expected_seeds <- as.integer(args[6])
symbol_levels <- strsplit(args[7], ",", fixed = TRUE)[[1]]

if (is.na(expected_seeds) || expected_seeds < 2) {
  stop("expected_seeds must be at least 2 for repeated-seed statistics")
}

if (!"relative" %in% symbol_levels) {
  stop("symbol_modes_csv must include the relative baseline")
}
if (length(unique(symbol_levels)) != length(symbol_levels)) {
  stop("symbol_modes_csv contains duplicate modes")
}
rca_groups <- paste(run_tag, symbol_levels, sep = "_")
aggregate_metric_levels <- c(
  "blimp_full",
  "comps_full",
  "entity_tracking_full",
  "reading_full",
  "eye_tracking_full",
  "self_paced_full"
)

metric_labels <- c(
  blimp_full = "BLiMP",
  comps_full = "COMPS",
  entity_tracking_full = "Entity",
  reading_full = "Reading",
  eye_tracking_full = "Eye",
  self_paced_full = "SPR"
)

long_scores <- read.csv(long_scores_path, stringsAsFactors = FALSE)
classification_pseudo_items <- read.csv(classification_pseudo_items_path, stringsAsFactors = FALSE)
submeasures <- read.csv(zero_shot_submeasures_path, stringsAsFactors = FALSE)

required_long_columns <- c(
  "experiment_name",
  "experiment_group",
  "seed_name",
  "metric",
  "score",
  "symbol_retrieval"
)
missing_long_columns <- setdiff(required_long_columns, names(long_scores))
if (length(missing_long_columns) > 0) {
  stop(sprintf("long_scores is missing required columns: %s", paste(missing_long_columns, collapse = ", ")))
}

required_subtask_columns <- c(
  "experiment_name",
  "seed_name",
  "task",
  "split",
  "subtest",
  "ling_term",
  "pseudo_item_id",
  "successes",
  "n_items",
  "accuracy"
)
missing_subtask_columns <- setdiff(required_subtask_columns, names(classification_pseudo_items))
if (length(missing_subtask_columns) > 0) {
  stop(sprintf("classification pseudo-item CSV is missing required columns: %s", paste(missing_subtask_columns, collapse = ", ")))
}

required_submeasure_columns <- c(
  "experiment_name",
  "experiment_group",
  "task",
  "submeasure_type",
  "submeasure",
  "score"
)
missing_submeasure_columns <- setdiff(required_submeasure_columns, names(submeasures))
if (length(missing_submeasure_columns) > 0) {
  stop(sprintf("zero_shot_submeasures is missing required columns: %s", paste(missing_submeasure_columns, collapse = ", ")))
}

missing_groups <- setdiff(rca_groups, unique(long_scores$experiment_group))
if (length(missing_groups) > 0) {
  stop(sprintf("Missing RCA symbol groups in long_scores: %s", paste(missing_groups, collapse = ", ")))
}

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

validate_blimp_trial_counts <- function(df, label) {
  invalid_rows <- df %>%
    filter(
      is.na(successes) |
        is.na(n_items) |
        n_items <= 0 |
        successes < 0 |
        successes > n_items
    )
  if (nrow(invalid_rows) > 0) {
    stop(sprintf("%s contains invalid BLiMP trial counts", label))
  }
  if (any(is.na(df$accuracy))) {
    stop(sprintf("%s contains missing BLiMP trial accuracies", label))
  }
}

validate_symbol_seed_coverage <- function(df, label, strata_columns = character()) {
  expected <- if (length(strata_columns) == 0) {
    crossing(symbol_retrieval = factor(symbol_levels, levels = symbol_levels))
  } else {
    df %>%
      select(all_of(strata_columns)) %>%
      distinct() %>%
      crossing(symbol_retrieval = factor(symbol_levels, levels = symbol_levels))
  }
  group_columns <- c(strata_columns, "symbol_retrieval")
  coverage <- df %>%
    select(all_of(c(strata_columns, "symbol_retrieval", "seed_name"))) %>%
    distinct() %>%
    count(across(all_of(group_columns)), name = "n_seeds") %>%
    right_join(expected, by = group_columns) %>%
    mutate(n_seeds = replace_na(n_seeds, 0))
  incomplete <- coverage %>% filter(n_seeds != expected_seeds)
  if (nrow(incomplete) > 0) {
    cell_columns <- c(strata_columns, "symbol_retrieval", "n_seeds")
    shown_cells <- head(incomplete %>% select(all_of(cell_columns)), 12)
    cell_labels <- apply(as.data.frame(shown_cells), 1, function(values) {
      paste(sprintf("%s=%s", names(values), values), collapse = "/")
    })
    stop(sprintf(
      "%s has incomplete symbol x seed coverage: %s",
      label,
      paste(cell_labels, collapse = ", ")
    ))
  }
}

single_effect_row <- function(anova_df, effect_name, label) {
  effect_rows <- anova_df %>% filter(effect == effect_name)
  if (nrow(effect_rows) != 1) {
    stop(sprintf("%s must contain exactly one %s effect row", label, effect_name))
  }
  effect_rows
}

fit_or_load_ultra_fast_glm <- function(cache_file, df, fe_str, family) {
  family <- normalize_family(family)
  validate_fixed_effect_design(df, fe_str)
  cache_file <- cache_file_for_selection_mode(cache_file, "env_ultra_fast")
  signature <- model_cache_signature(
    df,
    fe_str,
    character(),
    family,
    fast_fallback = FALSE,
    selection_mode = "env_ultra_fast_glm"
  )

  if (file.exists(cache_file)) {
    cached <- readRDS(cache_file)
    if (
      is.list(cached) &&
      identical(names(cached), c("signature", "model")) &&
      identical(cached$signature, signature)
    ) {
      cat(sprintf("  Loading cached GLM from %s\n", cache_file))
      return(cached$model)
    }
    cat(sprintf("  Cached GLM metadata missing or stale at %s; refitting.\n", cache_file))
  }

  cat("  Fitting binomial GLM without random effects for ultra-fast aggregate path...\n")
  model <- glm(as.formula(fe_str), data = df, family = family)
  dir.create(dirname(cache_file), showWarnings = FALSE, recursive = TRUE)
  cat(sprintf("  Saving fitted GLM to %s\n", cache_file))
  saveRDS(list(signature = signature, model = model), cache_file, compress = "xz")
  model
}

type3_chisq_anova <- function(model, label) {
  if (!requireNamespace("car", quietly = TRUE)) {
    stop(sprintf("car is required for %s type-III Chi-square tests", label))
  }
  res <- as.data.frame(car::Anova(model, type = 3, test.statistic = "Wald"))
  chisq_columns <- intersect(c("Chisq", "Wald Chisq", "LR Chisq"), names(res))
  if (!"Chisq" %in% names(res) && length(chisq_columns) == 1) {
    names(res)[names(res) == chisq_columns[[1]]] <- "Chisq"
  }
  if (!"Chisq" %in% names(res) || !"Pr(>Chisq)" %in% names(res)) {
    stop(sprintf("%s did not return complete Chi-square ANOVA columns", label))
  }
  if (any(is.na(res$`Pr(>Chisq)`))) {
    stop(sprintf("%s returned missing Chi-square p-values", label))
  }
  res
}

rca_long <- long_scores %>%
  filter(experiment_group %in% rca_groups, metric %in% aggregate_metric_levels) %>%
  mutate(
    score = as.numeric(score),
    symbol_retrieval = factor(
      substr(experiment_group, nchar(run_tag) + 2, nchar(experiment_group)),
      levels = symbol_levels
    ),
    seed_name = factor(seed_name),
    metric = factor(metric, levels = aggregate_metric_levels)
  ) %>%
  filter(!is.na(score))

mapping <- rca_long %>%
  mutate(seed_name = as.character(seed_name), symbol_retrieval = as.character(symbol_retrieval)) %>%
  select(experiment_name, experiment_group, seed_name, symbol_retrieval) %>%
  distinct()
if (anyDuplicated(mapping[c("experiment_name", "seed_name")]) > 0) {
  stop("RCA metadata mapping is not unique by experiment_name and seed_name.")
}

rca_subtests_all <- classification_pseudo_items %>%
  filter(task == "blimp") %>%
  select(-any_of("experiment_group")) %>%
  inner_join(mapping, by = c("experiment_name", "seed_name")) %>%
  mutate(
    successes = as.numeric(successes),
    n_items = as.numeric(n_items),
    accuracy = as.numeric(accuracy),
    experiment_name = factor(experiment_name),
    experiment_group = factor(experiment_group),
    seed_name = factor(seed_name),
    subtest = factor(subtest),
    pseudo_item_id = factor(pseudo_item_id),
    symbol_retrieval = factor(symbol_retrieval, levels = symbol_levels)
  )
validate_blimp_trial_counts(rca_subtests_all, "rca_subtests_all")
validate_symbol_seed_coverage(
  rca_subtests_all,
  "rca_subtests_all",
  c("subtest", "pseudo_item_id")
)

cat("Fitting GLMM for BLiMP Subtests...\n")
fe_str_sub <- "cbind(successes, n_items - successes) ~ symbol_retrieval"
# Note: Because pseudo_item_id is globally unique (e.g., 'subtest_p0'),
# crossing (1 | subtest) + (1 | pseudo_item_id) is mathematically identical
# to explicit nesting (1 | subtest / pseudo_item_id) in lme4.
# Similarly, experiment_name uniquely identifies the random seed for a specific architecture.
re_cands_sub <- c("(1 | experiment_name)", "(1 | subtest)", "(1 | pseudo_item_id)")

cache_file_sub <- file.path(output_dir, "cache_rca_subtests.rds")
m_sub <- fit_or_load_model(cache_file_sub, rca_subtests_all, fe_str_sub, re_cands_sub, family = "binomial")

anova_sub <- as.data.frame(anova(m_sub))
anova_sub$effect <- rownames(anova_sub)
single_effect_row(anova_sub, "symbol_retrieval", "rca_subtest_anova")
write.csv(anova_sub, file.path(output_dir, "rca_subtest_anova.csv"), row.names = FALSE)

write_emmeans_contrasts(
  m_sub,
  pairwise ~ symbol_retrieval,
  "symbol_retrieval",
  file.path(output_dir, "rca_subtest_contrasts.csv"),
  "RCA Subtest"
)

generate_diagnostics(m_sub, file.path(output_dir, "rca_subtest"))

if (any(is.na(rca_long$symbol_retrieval))) {
  stop("RCA rows contain symbol_retrieval values outside the expected symbol levels")
}

metric_counts <- rca_long %>%
  count(metric, symbol_retrieval, name = "n_rows") %>%
  complete(metric, symbol_retrieval, fill = list(n_rows = 0))

complete_metrics <- metric_counts %>%
  group_by(metric) %>%
  summarize(complete = all(n_rows == expected_seeds), .groups = "drop") %>%
  filter(complete) %>%
  pull(metric) %>%
  as.character()

excluded_metrics <- metric_counts %>%
  group_by(metric) %>%
  summarize(
    min_rows_per_symbol = min(n_rows),
    max_rows_per_symbol = max(n_rows),
    expected_rows_per_symbol = expected_seeds,
    complete = all(n_rows == expected_seeds),
    .groups = "drop"
  ) %>%
  filter(!complete)
write.csv(excluded_metrics, file.path(output_dir, "rca_symbol_metric_exclusions.csv"), row.names = FALSE)

if (length(complete_metrics) == 0) {
  stop("No aggregate metrics have complete RCA symbol data")
}

rca_model_data <- rca_long %>% filter(metric %in% complete_metrics)

write.csv(
  rca_model_data %>%
    group_by(metric, symbol_retrieval) %>%
    summarize(
      n = n(),
      mean = mean(score),
      sd = sd(score),
      se = sd(score) / sqrt(n()),
      .groups = "drop"
    ),
  file.path(output_dir, "rca_symbol_metric_means.csv"),
  row.names = FALSE
)

metric_anova_rows <- list()
metric_coefficient_rows <- list()
metric_pairwise_rows <- list()
metric_vs_relative_rows <- list()

for (metric_name in complete_metrics) {
  metric_data <- rca_model_data %>%
    filter(metric == metric_name) %>%
    mutate(symbol_retrieval = relevel(droplevels(symbol_retrieval), ref = "relative"))

  model <- lmer(score ~ symbol_retrieval + (1 | seed_name), data = metric_data, REML = TRUE)

  anova_row <- as.data.frame(anova(model))
  anova_row$effect <- rownames(anova_row)
  anova_row$metric <- metric_name
  metric_anova_rows[[metric_name]] <- anova_row

  coefficient_row <- as.data.frame(coef(summary(model)))
  coefficient_row$term <- rownames(coefficient_row)
  coefficient_row$metric <- metric_name
  metric_coefficient_rows[[metric_name]] <- coefficient_row

  emmeans_symbol <- emmeans(model, ~ symbol_retrieval)

  pairwise_row <- as.data.frame(pairs(emmeans_symbol))
  pairwise_row$metric <- metric_name
  metric_pairwise_rows[[metric_name]] <- pairwise_row

  vs_relative_row <- as.data.frame(contrast(emmeans_symbol, method = "trt.vs.ctrl", ref = 1))
  vs_relative_row$metric <- metric_name
  metric_vs_relative_rows[[metric_name]] <- vs_relative_row
}

metric_anova <- bind_rows(metric_anova_rows)
metric_coefficients <- bind_rows(metric_coefficient_rows)
metric_pairwise <- bind_rows(metric_pairwise_rows)
metric_vs_relative <- bind_rows(metric_vs_relative_rows)

write.csv(metric_anova, file.path(output_dir, "rca_symbol_metric_anova.csv"), row.names = FALSE)
write.csv(metric_coefficients, file.path(output_dir, "rca_symbol_metric_coefficients.csv"), row.names = FALSE)
write.csv(metric_pairwise, file.path(output_dir, "rca_symbol_metric_pairwise.csv"), row.names = FALSE)
write.csv(metric_vs_relative, file.path(output_dir, "rca_symbol_metric_vs_relative.csv"), row.names = FALSE)

ling_terms <- c(
  "filler_gap_dependency",
  "determiner_noun_agreement",
  "binding",
  "island_effects",
  "subject_verb_agreement",
  "ellipsis",
  "npi_licensing",
  "quantifiers",
  "anaphor_agreement",
  "argument_structure",
  "control_raising",
  "s-selection",
  "irregular_forms"
)

mapping <- rca_long %>%
  mutate(seed_name = as.character(seed_name), symbol_retrieval = as.character(symbol_retrieval)) %>%
  select(experiment_name, experiment_group, seed_name, symbol_retrieval) %>%
  distinct()
if (anyDuplicated(mapping[c("experiment_name", "experiment_group")]) > 0) {
  stop("RCA metadata mapping is not unique by experiment_name and experiment_group.")
}

rca_submeasures_all <- submeasures %>%
  filter(
    experiment_group %in% rca_groups,
    submeasure_type %in% c("uid_accuracy", "eye_tracking_dv", "self_paced")
  ) %>%
  inner_join(mapping, by = c("experiment_name", "experiment_group")) %>%
  mutate(
    score = as.numeric(score),
    symbol_retrieval = factor(symbol_retrieval, levels = symbol_levels),
    seed_name = factor(seed_name),
    task = factor(task),
    submeasure_id = factor(paste(task, submeasure_type, submeasure, sep = "::"))
  )

submeasure_expected_grid <- rca_submeasures_all %>%
  select(task, submeasure_id) %>%
  distinct() %>%
  crossing(symbol_retrieval = factor(symbol_levels, levels = symbol_levels))
submeasure_counts <- rca_submeasures_all %>%
  filter(!is.na(score)) %>%
  select(task, submeasure_id, symbol_retrieval, seed_name) %>%
  distinct() %>%
  count(task, submeasure_id, symbol_retrieval, name = "n_seeds") %>%
  right_join(submeasure_expected_grid, by = c("task", "submeasure_id", "symbol_retrieval")) %>%
  mutate(n_seeds = replace_na(n_seeds, 0))

complete_submeasure_tasks <- submeasure_counts %>%
  group_by(task) %>%
  summarize(complete = all(n_seeds == expected_seeds), .groups = "drop") %>%
  filter(complete) %>%
  pull(task) %>%
  as.character()

excluded_submeasure_tasks <- submeasure_counts %>%
  group_by(task) %>%
  summarize(
    n_submeasures = n_distinct(submeasure_id),
    min_seeds_per_symbol = min(n_seeds),
    max_seeds_per_symbol = max(n_seeds),
    expected_seeds_per_symbol = expected_seeds,
    complete = all(n_seeds == expected_seeds),
    .groups = "drop"
  ) %>%
  filter(!complete)
write.csv(
  excluded_submeasure_tasks,
  file.path(output_dir, "rca_symbol_submeasure_exclusions.csv"),
  row.names = FALSE
)

if (length(complete_submeasure_tasks) == 0) {
  stop("No zero-shot submeasure tasks have complete RCA symbol data")
}

rca_submeasures <- rca_submeasures_all %>% filter(!is.na(score))
rca_submeasure_model_data <- rca_submeasures %>% filter(task %in% complete_submeasure_tasks)

write.csv(
  rca_submeasure_model_data %>%
    group_by(task, symbol_retrieval) %>%
    summarize(
      n = n(),
      mean = mean(score),
      sd = sd(score),
      se = sd(score) / sqrt(n()),
      .groups = "drop"
    ),
  file.path(output_dir, "rca_symbol_submeasure_task_means.csv"),
  row.names = FALSE
)

submeasure_anova_rows <- list()
submeasure_coefficient_rows <- list()
submeasure_pairwise_rows <- list()
submeasure_vs_relative_rows <- list()

for (task_name in complete_submeasure_tasks) {
  task_data <- rca_submeasure_model_data %>%
    filter(task == task_name) %>%
    mutate(symbol_retrieval = relevel(droplevels(symbol_retrieval), ref = "relative"))

  model <- lmer(score ~ symbol_retrieval + (1 | seed_name) + (1 | submeasure_id), data = task_data, REML = TRUE)

  anova_row <- as.data.frame(anova(model))
  anova_row$effect <- rownames(anova_row)
  anova_row$task <- task_name
  submeasure_anova_rows[[task_name]] <- anova_row

  coefficient_row <- as.data.frame(coef(summary(model)))
  coefficient_row$term <- rownames(coefficient_row)
  coefficient_row$task <- task_name
  submeasure_coefficient_rows[[task_name]] <- coefficient_row

  emmeans_symbol <- emmeans(model, ~ symbol_retrieval)

  pairwise_row <- as.data.frame(pairs(emmeans_symbol))
  pairwise_row$task <- task_name
  submeasure_pairwise_rows[[task_name]] <- pairwise_row

  vs_relative_row <- as.data.frame(contrast(emmeans_symbol, method = "trt.vs.ctrl", ref = 1))
  vs_relative_row$task <- task_name
  submeasure_vs_relative_rows[[task_name]] <- vs_relative_row
}

submeasure_anova <- bind_rows(submeasure_anova_rows)
submeasure_coefficients <- bind_rows(submeasure_coefficient_rows)
submeasure_pairwise <- bind_rows(submeasure_pairwise_rows)
submeasure_vs_relative <- bind_rows(submeasure_vs_relative_rows)

write.csv(submeasure_anova, file.path(output_dir, "rca_symbol_submeasure_anova.csv"), row.names = FALSE)
write.csv(submeasure_coefficients, file.path(output_dir, "rca_symbol_submeasure_coefficients.csv"), row.names = FALSE)
write.csv(submeasure_pairwise, file.path(output_dir, "rca_symbol_submeasure_pairwise.csv"), row.names = FALSE)
write.csv(submeasure_vs_relative, file.path(output_dir, "rca_symbol_submeasure_vs_relative.csv"), row.names = FALSE)

if (!"Pr(>F)" %in% names(metric_anova)) {
  stop("Metric ANOVA did not include Pr(>F)")
}
if (!"F value" %in% names(metric_anova)) {
  stop("Metric ANOVA did not include F value")
}
if (!"Pr(>Chisq)" %in% names(anova_sub)) {
  stop("Missing GLMM p-value column in BLiMP ANOVA results")
}
if (!"Chisq" %in% names(anova_sub)) {
  stop("Missing Chi-square column in BLiMP ANOVA results")
}
if (!"Pr(>F)" %in% names(submeasure_anova)) {
  stop("Submeasure ANOVA did not include Pr(>F)")
}
if (!"F value" %in% names(submeasure_anova)) {
  stop("Submeasure ANOVA did not include F value")
}

format_p <- function(value) {
  if (is.na(value)) {
    return("--")
  }
  if (value < 0.001) {
    return("$<0.001$")
  }
  sprintf("%.3f", value)
}

metric_label <- function(metric_name) {
  metric_name <- as.character(metric_name)
  label <- metric_labels[[metric_name]]
  if (is.null(label)) {
    return(metric_name)
  }
  label
}

metric_effects <- metric_anova %>%
  filter(effect == "symbol_retrieval") %>%
  mutate(metric_label = vapply(as.character(metric), metric_label, character(1))) %>%
  arrange(match(metric, aggregate_metric_levels))

metric_tex_lines <- c(
  "\\begin{tabular}{lrrr}",
  "\\toprule",
  "Metric & $F$ & DenDF & $p$ \\\\",
  "\\midrule"
)
for (row_index in seq_len(nrow(metric_effects))) {
  row <- metric_effects[row_index, ]
  metric_tex_lines <- c(
    metric_tex_lines,
    sprintf(
      "%s & %.2f & %.1f & %s \\\\",
      row$metric_label,
      row[["F value"]],
      row[["DenDF"]],
      format_p(row[["Pr(>F)"]])
    )
  )
}
metric_tex_lines <- c(metric_tex_lines, "\\bottomrule", "\\end{tabular}")
writeLines(metric_tex_lines, file.path(output_dir, "metric_model_summary.tex"))

blimp_effect <- single_effect_row(anova_sub, "symbol_retrieval", "rca_subtest_anova")
blimp_tex_lines <- c(
  "\\begin{tabular}{lrrr}",
  "\\toprule",
  "Effect & $\\chi^2$ & Df & $p$ \\\\",
  "\\midrule",
  sprintf(
    "Symbol retrieval & %.2f & %.1f & %s \\\\",
    blimp_effect[["Chisq"]],
    blimp_effect[["Df"]],
    format_p(blimp_effect[["Pr(>Chisq)"]])
  ),
  "\\bottomrule",
  "\\end{tabular}"
)
writeLines(blimp_tex_lines, file.path(output_dir, "blimp_subtest_model_summary.tex"))

submeasure_effects <- submeasure_anova %>%
  filter(effect == "symbol_retrieval") %>%
  arrange(task)

submeasure_tex_lines <- c(
  "\\begin{tabular}{lrrr}",
  "\\toprule",
  "Task & $F$ & DenDF & $p$ \\\\",
  "\\midrule"
)
for (row_index in seq_len(nrow(submeasure_effects))) {
  row <- submeasure_effects[row_index, ]
  submeasure_tex_lines <- c(
    submeasure_tex_lines,
    sprintf(
      "%s & %.2f & %.1f & %s \\\\",
      gsub("_", "\\_", row$task, fixed = TRUE),
      row[["F value"]],
      row[["DenDF"]],
      format_p(row[["Pr(>F)"]])
    )
  )
}
submeasure_tex_lines <- c(submeasure_tex_lines, "\\bottomrule", "\\end{tabular}")
writeLines(submeasure_tex_lines, file.path(output_dir, "submeasure_model_summary.tex"))

contrast_tex_lines <- c(
  "\\begin{tabular}{llrr}",
  "\\toprule",
  "Metric & Contrast & Estimate & $p$ \\\\",
  "\\midrule"
)
for (row_index in seq_len(nrow(metric_vs_relative))) {
  row <- metric_vs_relative[row_index, ]
  contrast_tex_lines <- c(
    contrast_tex_lines,
    sprintf(
      "%s & %s & %.3f & %s \\\\",
      metric_label(row$metric),
      gsub("_", "\\_", row$contrast, fixed = TRUE),
      row$estimate,
      format_p(row$p.value)
    )
  )
}
contrast_tex_lines <- c(contrast_tex_lines, "\\bottomrule", "\\end{tabular}")
writeLines(contrast_tex_lines, file.path(output_dir, "metric_vs_relative.tex"))

plot_metric_data <- rca_model_data %>%
  mutate(
    metric_label = factor(
      vapply(as.character(metric), metric_label, character(1)),
      levels = vapply(complete_metrics, metric_label, character(1))
    )
  ) %>%
  group_by(metric_label, symbol_retrieval) %>%
  summarize(mean_score = mean(score), se_score = sd(score) / sqrt(n()), .groups = "drop")

metric_plot <- ggplot(plot_metric_data, aes(x = symbol_retrieval, y = mean_score, color = symbol_retrieval)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean_score - se_score, ymax = mean_score + se_score), width = 0.15) +
  facet_wrap(~ metric_label, scales = "free_y") +
  theme_minimal() +
  labs(
    title = "RCA/SwiGLU 264-token symbol retrieval",
    x = "Symbol retrieval",
    y = "Score",
    color = "Symbols"
  ) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1), legend.position = "none")

ggsave(file.path(output_dir, "rca_symbol_metric_means.png"), metric_plot, width = 10, height = 7, dpi = 300)

submeasure_plot_data <- rca_submeasure_model_data %>%
  group_by(task, symbol_retrieval) %>%
  summarize(mean_score = mean(score), se_score = sd(score) / sqrt(n()), .groups = "drop")

submeasure_plot <- ggplot(submeasure_plot_data, aes(x = symbol_retrieval, y = mean_score, color = symbol_retrieval)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean_score - se_score, ymax = mean_score + se_score), width = 0.15) +
  facet_wrap(~ task, scales = "free_y") +
  theme_minimal() +
  labs(
    title = "RCA/SwiGLU zero-shot submeasure task families",
    x = "Symbol retrieval",
    y = "Submeasure score",
    color = "Symbols"
  ) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1), legend.position = "none")

ggsave(
  file.path(output_dir, "rca_symbol_submeasure_task_means.png"),
  submeasure_plot,
  width = 11,
  height = 7,
  dpi = 300
)

individual_submeasure_plot_data <- rca_submeasure_model_data %>%
  group_by(task, submeasure, symbol_retrieval) %>%
  summarize(mean_score = mean(score), se_score = sd(score) / sqrt(n()), .groups = "drop")

write.csv(
  individual_submeasure_plot_data,
  file.path(output_dir, "rca_symbol_individual_submeasure_means.csv"),
  row.names = FALSE
)

for (task_name in unique(individual_submeasure_plot_data$task)) {
  task_subset <- individual_submeasure_plot_data %>% filter(task == task_name)
  # Replace underscores with spaces for readability in plot labels
  task_subset$submeasure_label <- gsub("_", " ", task_subset$submeasure)

  task_plot <- ggplot(task_subset, aes(x = submeasure_label, y = mean_score, color = symbol_retrieval)) +
    geom_point(position = position_dodge(width = 0.6), size = 2.5) +
    geom_errorbar(
      aes(ymin = mean_score - se_score, ymax = mean_score + se_score),
      position = position_dodge(width = 0.6),
      width = 0.15
    ) +
    coord_flip() +
    theme_minimal() +
    labs(
      title = sprintf("RCA/SwiGLU %s zero-shot submeasures", task_name),
      x = "Submeasure",
      y = "Score",
      color = "Symbols"
    )

  plot_height <- max(5, length(unique(task_subset$submeasure)) * 0.4)
  ggsave(
    file.path(output_dir, sprintf("rca_symbol_%s_submeasures.png", task_name)),
    task_plot,
    width = 11,
    height = plot_height,
    dpi = 300
  )
}

rca_ling <- rca_subtests_all %>%
  filter(!is.na(ling_term), ling_term != "unknown") %>%
  mutate(
    ling_term = factor(gsub("_", " ", ling_term)),
    symbol_retrieval = factor(symbol_retrieval, levels = symbol_levels)
  )
expected_ling_terms <- gsub("_", " ", ling_terms, fixed = TRUE)
missing_ling_terms <- setdiff(expected_ling_terms, unique(as.character(rca_ling$ling_term)))
if (length(missing_ling_terms) > 0) {
  stop(sprintf(
    "rca_ling is missing canonical BLiMP linguistic domains: %s",
    paste(missing_ling_terms, collapse = ", ")
  ))
}
validate_blimp_trial_counts(rca_ling, "rca_ling")
validate_symbol_seed_coverage(
  rca_ling,
  "rca_ling",
  c("ling_term", "subtest", "pseudo_item_id")
)

if (mixed_model_ultra_fast_mode()) {
  cat("Fitting binomial GLM for Linguistic Sub-domains...\n")
  rca_ling_model_data <- rca_ling %>%
    group_by(experiment_name, seed_name, symbol_retrieval, ling_term) %>%
    summarize(
      successes = sum(successes),
      n_items = sum(n_items),
      .groups = "drop"
    ) %>%
    mutate(
      experiment_name = factor(experiment_name),
      seed_name = factor(seed_name),
      symbol_retrieval = factor(symbol_retrieval, levels = symbol_levels),
      ling_term = factor(ling_term)
    )
  fe_str_ling <- "cbind(successes, n_items - successes) ~ symbol_retrieval + ling_term"
  ling_contrast_specs <- pairwise ~ symbol_retrieval | ling_term
  ling_contrast_required <- c("symbol_retrieval", "ling_term")
  cache_file_ling <- file.path(output_dir, "cache_rca_ling.rds")
  m_ling <- fit_or_load_ultra_fast_glm(cache_file_ling, rca_ling_model_data, fe_str_ling, binomial())
  anova_ling <- type3_chisq_anova(m_ling, "rca_ling_anova")
} else {
  cat("Fitting GLMM for Linguistic Sub-domains...\n")
  rca_ling_model_data <- rca_ling
  fe_str_ling <- "cbind(successes, n_items - successes) ~ symbol_retrieval * ling_term"
  re_cands_ling <- c("(1 | experiment_name)", "(1 | subtest)", "(1 | pseudo_item_id)")
  ling_contrast_specs <- pairwise ~ symbol_retrieval | ling_term
  ling_contrast_required <- c("symbol_retrieval", "ling_term")
  cache_file_ling <- file.path(output_dir, "cache_rca_ling.rds")
  m_ling <- fit_or_load_model(cache_file_ling, rca_ling_model_data, fe_str_ling, re_cands_ling, family = "binomial", fast_fallback = TRUE)
  anova_ling <- as.data.frame(anova(m_ling))
}

anova_ling$effect <- rownames(anova_ling)
write.csv(anova_ling, file.path(output_dir, "rca_ling_anova.csv"), row.names = FALSE)

write_emmeans_contrasts(
  m_ling,
  ling_contrast_specs,
  ling_contrast_required,
  file.path(output_dir, "rca_ling_contrasts.csv"),
  "RCA Ling"
)

generate_diagnostics(m_ling, file.path(output_dir, "rca_ling"))

if (nrow(rca_ling) > 0) {
  ling_plot_data <- rca_ling %>%
    group_by(ling_term, symbol_retrieval) %>%
    summarize(
      successes = sum(successes),
      n_items = sum(n_items),
      mean_score = successes / n_items,
      se_score = sqrt(mean_score * (1 - mean_score) / n_items),
      .groups = "drop"
    )

  ling_plot <- ggplot(ling_plot_data, aes(x = ling_term, y = mean_score, color = symbol_retrieval)) +
    geom_point(position = position_dodge(width = 0.6), size = 2.5) +
    geom_errorbar(
      aes(ymin = mean_score - se_score, ymax = mean_score + se_score),
      position = position_dodge(width = 0.6),
      width = 0.15
    ) +
    coord_flip() +
    theme_minimal() +
    labs(
      title = "RCA/SwiGLU BLiMP linguistic domains",
      x = "Linguistic domain",
      y = "Accuracy",
      color = "Symbols"
    )

  ggsave(file.path(output_dir, "rca_symbol_blimp_ling_terms.png"), ling_plot, width = 11, height = 8, dpi = 300)
}

cat(sprintf("Wrote RCA symbol model outputs to %s\n", output_dir))
