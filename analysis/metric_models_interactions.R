#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(dplyr)
  library(tidyr)
})

options(width = 120, contrasts = c("contr.sum", "contr.poly"))
emm_options(lmer.df = "satterthwaite", lmerTest.limit = 200000, pbkrtest.limit = 0)

script_dir <- dirname(sub("--file=", "", commandArgs()[grep("--file=", commandArgs())]))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."
source(file.path(script_dir, "mixed_effects_utils.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript metric_models_interactions.R <classification_pseudo_items.csv> <long_scores.csv> <output_dir>\n")
  quit(status = 1)
}

trials_path <- args[1]
long_scores_path <- args[2]
output_dir <- args[3]

data <- read.csv(trials_path, stringsAsFactors = FALSE)
if (!"task" %in% names(data)) {
  stop("classification pseudo-item CSV is missing required task column.")
}
data <- data %>% filter(task == "blimp")
long_scores <- read.csv(long_scores_path, stringsAsFactors = FALSE)
if (!"analysis_family" %in% names(long_scores)) {
  stop("long_scores.csv is missing required analysis_family column.")
}
long_scores <- long_scores %>%
  filter(analysis_family %in% c("architecture_comparison", "rca_symbol_retrieval_comparison"))
if (nrow(long_scores) == 0) {
  stop("long_scores.csv has no rows for the selected analysis families.")
}

mapping <- long_scores %>% 
  select(experiment_name, experiment_group, architecture, training_condition) %>% 
  distinct()
missing_trials <- mapping %>%
  distinct(experiment_name, experiment_group) %>%
  anti_join(data %>% distinct(experiment_name, experiment_group), by = c("experiment_name", "experiment_group"))
if (nrow(missing_trials) > 0) {
  stop(sprintf(
    "classification pseudo-item CSV is missing architecture-comparison BLiMP rows for: %s",
    paste(
      paste(missing_trials$experiment_name, missing_trials$experiment_group, sep = " / "),
      collapse = ", "
    )
  ))
}

# Join with mapping and ling mapping
data <- data %>%
  inner_join(mapping, by = c("experiment_name", "experiment_group")) %>%
  filter(!is.na(successes))

# ---------------------------------------------------------
# MODEL 1: Architecture x Ling Term (using Ling Terms)
# ---------------------------------------------------------
data_ling <- data %>%
  filter(architecture %in% c("lm", "dat_9sa_3ra_relative_swiglu_rca")) %>%
  filter(training_condition %in% c("tied_ntp", "untied_ntp")) %>%
  filter(!is.na(ling_term), ling_term != "unknown")

validate_complete_crossed_cells(
  data_ling,
  "data_ling",
  "architecture",
  c("lm", "dat_9sa_3ra_relative_swiglu_rca"),
  "training_condition",
  c("tied_ntp", "untied_ntp")
)

data_ling <- data_ling %>%
  mutate(
    architecture = factor(architecture),
    training_condition = factor(training_condition),
    experiment_name = factor(experiment_name),
    ling_term = factor(ling_term),
    subtest = factor(subtest),
    pseudo_item_id = factor(pseudo_item_id)
  )

cat("Fitting Ling Term Interaction Model (N =", nrow(data_ling), ")...\n")
fe_str_ling <- "cbind(successes, n_items - successes) ~ architecture * training_condition + architecture * ling_term"
re_cands_ling <- c("(1 | experiment_name)", "(1 | subtest)", "(1 | pseudo_item_id)")

cache_file_ling <- file.path(output_dir, "cache_interaction_ling.rds")
model_ling <- fit_or_load_model(cache_file_ling, data_ling, fe_str_ling, re_cands_ling, family = "binomial", fast_fallback = TRUE)

anova_ling <- as.data.frame(anova(model_ling))
anova_ling$effect <- rownames(anova_ling)
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
write.csv(anova_ling, file.path(output_dir, "interaction_model_ling_anova.csv"), row.names = FALSE)
cat(sprintf("Wrote Ling Term ANOVA to %s/interaction_model_ling_anova.csv\n", output_dir))

cat("  --- Post-hoc Contrasts ---\n")
write_emmeans_contrasts(
  model_ling,
  pairwise ~ architecture | ling_term,
  c("architecture", "ling_term"),
  file.path(output_dir, "interaction_model_ling_contrasts.csv"),
  "Ling Term"
)

cat("  --- Diagnostic Plots ---\n")
generate_diagnostics(model_ling, file.path(output_dir, "interaction_model_ling"))
cat(sprintf("Wrote Ling Term Diagnostics to %s/interaction_model_ling_*.png\n\n", output_dir))
