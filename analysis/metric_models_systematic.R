#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(emmeans)
})

options(width = 120, contrasts = c("contr.sum", "contr.poly"))

script_dir <- dirname(sub("--file=", "", commandArgs()[grep("--file=", commandArgs())]))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."
source(file.path(script_dir, "mixed_effects_utils.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript metric_models_systematic.R <classification_pseudo_items.csv> <long_scores.csv> <output_dir>\n")
  quit(status = 1)
}

trials_path <- args[1]
long_scores_path <- args[2]
output_dir <- args[3]

trials <- read.csv(trials_path, stringsAsFactors = FALSE)
if (!"task" %in% names(trials)) {
  stop("classification pseudo-item CSV is missing required task column.")
}
trials <- trials %>% filter(task == "blimp")
long_scores <- read.csv(long_scores_path, stringsAsFactors = FALSE)
if (!"analysis_family" %in% names(long_scores)) {
  stop("long_scores.csv is missing required analysis_family column.")
}
long_scores <- long_scores %>%
  filter(analysis_family %in% c("architecture_comparison", "rca_symbol_retrieval_comparison"))
if (nrow(long_scores) == 0) {
  stop("long_scores.csv has no rows for the selected analysis families.")
}

metadata <- long_scores %>%
  select(
    experiment_name,
    experiment_group,
    model_type,
    objective,
    tie_lm_head,
    n_heads_sa,
    n_heads_ra,
    datapoint_length,
    symbol_retrieval,
    ra_type,
    ffn_activation
  ) %>%
  distinct()

if (anyDuplicated(metadata[c("experiment_name", "experiment_group")]) > 0) {
  stop("long_scores.csv contains inconsistent metadata for at least one experiment.")
}

required_base_metadata <- c(
  "model_type",
  "objective",
  "tie_lm_head",
  "datapoint_length"
)
required_dat_metadata <- c(
  "n_heads_sa",
  "n_heads_ra",
  "symbol_retrieval",
  "ra_type",
  "ffn_activation"
)
missing_base_metadata <- metadata %>%
  filter(if_any(all_of(required_base_metadata), ~ is.na(.x) | .x == ""))
missing_dat_metadata <- metadata %>%
  filter(model_type == "dat") %>%
  filter(if_any(all_of(required_dat_metadata), ~ is.na(.x) | .x == ""))
missing_metadata <- bind_rows(missing_base_metadata, missing_dat_metadata) %>%
  distinct(experiment_name, .keep_all = TRUE)
if (nrow(missing_metadata) > 0) {
  stop(sprintf(
    "long_scores.csv has missing required model metadata for: %s",
    paste(unique(missing_metadata$experiment_name), collapse = ", ")
  ))
}

missing_trials <- metadata %>%
  distinct(experiment_name, experiment_group) %>%
  anti_join(trials %>% distinct(experiment_name, experiment_group), by = c("experiment_name", "experiment_group"))
if (nrow(missing_trials) > 0) {
  stop(sprintf(
    "classification pseudo-item CSV is missing architecture-comparison BLiMP rows for: %s",
    paste(
      paste(missing_trials$experiment_name, missing_trials$experiment_group, sep = " / "),
      collapse = ", "
    )
  ))
}

validate_allowed_values <- function(df, column, allowed_values, label) {
  observed <- unique(as.character(df[[column]][!is.na(df[[column]]) & df[[column]] != ""]))
  unexpected <- setdiff(observed, allowed_values)
  if (length(unexpected) > 0) {
    stop(sprintf(
      "%s contains unexpected %s value(s): %s",
      label,
      column,
      paste(sort(unexpected), collapse = ", ")
    ))
  }
}

validate_allowed_values(metadata, "objective", c("NTP", "NextLat"), "long_scores.csv")
validate_allowed_values(metadata, "tie_lm_head", c("true", "false"), "long_scores.csv")
validate_allowed_values(metadata, "model_type", c("dat", "self_attention", "causal"), "long_scores.csv")
validate_allowed_values(
  metadata %>% filter(model_type == "dat"),
  "symbol_retrieval",
  c("relative", "symbolic", "positional", "relsymbolic"),
  "long_scores.csv DAT rows"
)

data <- trials %>%
  inner_join(metadata, by = c("experiment_name", "experiment_group")) %>%
  filter(!is.na(successes)) %>%
  mutate(
    # 1. Training Conditions
    is_tied = factor(ifelse(tie_lm_head == "true", "tied", "untied")),
    objective = factor(ifelse(objective == "NextLat", "nextlat", "ntp")),

    # 2. Base Architecture
    base_arch = factor(ifelse(model_type == "dat", "dat", "baseline")),

    # 3. DAT Sub-factors (NA for baseline)
    attn_type = ifelse(model_type == "dat", ra_type, NA_character_),
    layers = ifelse(model_type == "dat", paste0(n_heads_sa, "sa", n_heads_ra, "ra"), NA_character_),
    activation = ifelse(model_type == "dat", ffn_activation, NA_character_),
    position = ifelse(model_type == "dat" & symbol_retrieval == "relative", "relative", "symbolic"),
    context_length = factor(datapoint_length),

    # Clean up factors
    experiment_name = factor(experiment_name),
    subtest = factor(subtest),
    pseudo_item_id = factor(pseudo_item_id)
  )

# Ensure factors for DATs
data$attn_type[data$base_arch == "baseline"] <- NA
data$layers[data$base_arch == "baseline"] <- NA
data$activation[data$base_arch == "baseline"] <- NA
data$position[data$base_arch == "baseline"] <- NA

# -------------------------------------------------------------------------
# MODEL 1: Systematic Baseline vs Optimal DAT Comparison
# -------------------------------------------------------------------------
# Filter to Baseline OR Optimal DAT backbone
data_m1 <- data %>%
  filter(context_length == "512") %>%
  filter(
    base_arch == "baseline" |
    (base_arch == "dat" & attn_type == "rca" & layers == "9sa3ra" & activation == "swiglu" & position == "relative")
  ) %>%
  mutate(base_arch = droplevels(base_arch))

cat("Fitting Model 1: Baseline vs Optimal DAT (Fully Crossed)...\n")
# We cross base_arch * is_tied * objective
fe_str_m1 <- "cbind(successes, n_items - successes) ~ base_arch * is_tied + base_arch * objective"
# Note: Because pseudo_item_id is globally unique (e.g., 'subtest_p0'),
# crossing (1 | subtest) + (1 | pseudo_item_id) is mathematically identical
# to explicit nesting (1 | subtest / pseudo_item_id) in lme4.
# Similarly, experiment_name uniquely identifies the random seed for a specific architecture.
re_cands_m1 <- c("(1 | experiment_name)", "(1 | subtest)", "(1 | pseudo_item_id)")

cache_file_m1 <- file.path(output_dir, "cache_systematic_m1.rds")
m1 <- fit_or_load_model(cache_file_m1, data_m1, fe_str_m1, re_cands_m1, family = "binomial")

anova_m1 <- as.data.frame(anova(m1))
anova_m1$effect <- rownames(anova_m1)
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
write.csv(anova_m1, file.path(output_dir, "systematic_baseline_vs_dat_anova.csv"), row.names = FALSE)
cat(sprintf("Wrote M1 ANOVA to %s/systematic_baseline_vs_dat_anova.csv\n", output_dir))

cat("  --- Post-hoc Contrasts ---\n")
write_emmeans_contrasts(
  m1,
  pairwise ~ base_arch | is_tied + objective,
  c("base_arch", "is_tied", "objective"),
  file.path(output_dir, "systematic_baseline_vs_dat_contrasts.csv"),
  "M1"
)

cat("  --- Diagnostic Plots ---\n")
generate_diagnostics(m1, file.path(output_dir, "systematic_baseline_vs_dat"))
cat(sprintf("Wrote M1 Diagnostics to %s/systematic_baseline_vs_dat_*.png\n\n", output_dir))

# -------------------------------------------------------------------------
# MODEL 2: Systematic DAT Internal Mechanics
# -------------------------------------------------------------------------
data_dat <- data %>%
  filter(base_arch == "dat") %>%
  filter(!is.na(attn_type), !is.na(layers), !is.na(activation), !is.na(position)) %>%
  mutate(
    attn_type = factor(attn_type),
    layers = factor(layers),
    activation = factor(activation),
    position = factor(position),
    activation_position = factor(paste(activation, position, sep = "_")),
    context_length = factor(context_length)
  )

cat("Fitting Model 2: Internal DAT Mechanics...\n")

#cat("\n--- Debug: Factor Levels ---\n")
#print(table(data_dat$attn_type, useNA="always"))
#print(table(data_dat$layers, useNA="always"))
#print(table(data_dat$activation_position, useNA="always"))
#print(table(data_dat$is_tied, useNA="always"))
#print(table(data_dat$objective, useNA="always"))
#print(table(data_dat$context_length, useNA="always"))

fe_str_m2 <- "cbind(successes, n_items - successes) ~ attn_type + layers + activation_position + is_tied + objective + context_length"
re_cands_m2 <- c("(1 | experiment_name)", "(1 | subtest)", "(1 | pseudo_item_id)")

cache_file_m2 <- file.path(output_dir, "cache_systematic_m2.rds")
m2 <- fit_or_load_model(cache_file_m2, data_dat, fe_str_m2, re_cands_m2, family = "binomial")

anova_m2 <- as.data.frame(anova(m2))
anova_m2$effect <- rownames(anova_m2)
write.csv(anova_m2, file.path(output_dir, "systematic_dat_internal_anova.csv"), row.names = FALSE)
cat(sprintf("Wrote M2 ANOVA to %s/systematic_dat_internal_anova.csv\n", output_dir))

cat("  --- Post-hoc Contrasts ---\n")
write_emmeans_contrasts(
  m2,
  pairwise ~ attn_type,
  "attn_type",
  file.path(output_dir, "systematic_dat_internal_contrasts.csv"),
  "M2"
)

cat("  --- Diagnostic Plots ---\n")
generate_diagnostics(m2, file.path(output_dir, "systematic_dat_internal"))
cat(sprintf("Wrote M2 Diagnostics to %s/systematic_dat_internal_*.png\n\n", output_dir))
