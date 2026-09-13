#!/usr/bin/env Rscript
# Stage 2 cross-experiment reading model.
#
# The reading eval already computes per-experiment OLS regressions (Stage 1):
#   for each reading DV, fit baseline (no pred) vs full (with pred),
#   store R2, delta-R2, coefficient, t, p in predictive_power.jsonl.
#
# This script reads those already-computed per-experiment reading scores,
# joins them with the experiment matrix for architecture/config factors,
# and fits three separate cross-experiment models:
#
#   1. Architecture-level (all experiments):
#      delta_r2 ~ model_type + objective + tie_lm_head + datapoint_length +
#                 model_variant + (1 | experiment_name) + (1 | dv_name)
#
#   2. DAT-internal (DAT experiments only):
#      delta_r2 ~ ra_type + ffn_activation + symbol_retrieval + objective +
#                 tie_lm_head + datapoint_length + model_variant +
#                 (1 | experiment_name) + (1 | dv_name)
#
#   3. RCA 264 symbol subset (RCA symbol experiments only):
#      delta_r2 ~ symbol_retrieval + model_variant +
#                 (1 | seed_name) + (1 | dv_name)
#
# The dependent variable is the per-experiment per-DV delta-R2 (how much
# model surprisal improves prediction of human reading). Random effects
# account for clustering by experiment and by reading DV.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(dplyr)
})

options(width = 120, contrasts = c("contr.sum", "contr.poly"))

script_dir <- dirname(sub("--file=", "", commandArgs()[grep("--file=", commandArgs())]))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."
source(file.path(script_dir, "mixed_effects_utils.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript metric_models_reading.R <reading_scores.csv.gz> <experiment_matrix.csv> <output_dir>\n")
  quit(status = 1)
}

reading_scores_path <- args[1]
experiment_matrix_path <- args[2]
output_dir <- args[3]
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

reading_scores <- read.csv(reading_scores_path, stringsAsFactors = FALSE)
experiment_matrix <- read.csv(experiment_matrix_path, stringsAsFactors = FALSE)

required_reading_cols <- c("experiment_name", "dv_name", "model_variant", "delta_r2")
missing_reading_cols <- setdiff(required_reading_cols, names(reading_scores))
if (length(missing_reading_cols) > 0) {
  stop(sprintf("reading_scores.csv is missing required columns: %s",
               paste(missing_reading_cols, collapse = ", ")))
}

required_matrix_cols <- c("experiment_name", "model_type", "objective",
                          "tie_lm_head", "datapoint_length")
missing_matrix_cols <- setdiff(required_matrix_cols, names(experiment_matrix))
if (length(missing_matrix_cols) > 0) {
  stop(sprintf("experiment_matrix.csv is missing required columns: %s",
               paste(missing_matrix_cols, collapse = ", ")))
}

# Join reading scores with experiment matrix on experiment_name.
data <- reading_scores %>%
  inner_join(
    experiment_matrix %>%
      select(experiment_name, model_type, objective, tie_lm_head,
             datapoint_length, ra_type, ffn_activation, symbol_retrieval,
             seed_name, analysis_family),
    by = "experiment_name"
  )

if (nrow(data) == 0) {
  stop("No rows after joining reading scores with experiment matrix")
}

# Convert factors for modeling.
factor_cols <- c("model_type", "objective", "tie_lm_head", "datapoint_length",
                 "ra_type", "ffn_activation", "symbol_retrieval",
                 "model_variant", "dv_name", "experiment_name", "seed_name",
                 "analysis_family")
for (col in factor_cols) {
  if (col %in% names(data)) {
    data[[col]] <- factor(data[[col]])
  }
}

data$delta_r2 <- as.numeric(data$delta_r2)

cat(sprintf("Joined data: %d rows, %d experiments, %d DVs\n",
            nrow(data), n_distinct(data$experiment_name),
            n_distinct(data$dv_name)))

# ---------------------------------------------------------------------------
# Model 1: Architecture-level (all experiments)
# delta_r2 ~ model_type + objective + tie_lm_head + datapoint_length +
#            model_variant + (1 | experiment_name) + (1 | dv_name)
# ---------------------------------------------------------------------------
cat("\n=== Fitting architecture-level reading model ===\n")
fe_str_m1 <- paste(
  "delta_r2 ~ model_type + objective + tie_lm_head +",
  "datapoint_length + model_variant"
)
re_cands_m1 <- c("(1 | experiment_name)", "(1 | dv_name)")

cache_file_m1 <- file.path(output_dir, "cache_reading_arch.rds")
model_m1 <- fit_or_load_model(cache_file_m1, data, fe_str_m1, re_cands_m1)

anova_m1 <- as.data.frame(anova(model_m1))
anova_m1$effect <- rownames(anova_m1)
write.csv(anova_m1, file.path(output_dir, "reading_arch_anova.csv"), row.names = FALSE)
cat(sprintf("Wrote architecture-level reading ANOVA to %s/reading_arch_anova.csv\n", output_dir))

coef_m1 <- as.data.frame(summary(model_m1)$coefficients)
coef_m1$term <- rownames(coef_m1)
write.csv(coef_m1, file.path(output_dir, "reading_arch_coefficients.csv"), row.names = FALSE)
cat(sprintf("Wrote architecture-level reading coefficients to %s/reading_arch_coefficients.csv\n", output_dir))

generate_diagnostics(model_m1, file.path(output_dir, "reading_arch"))

# ---------------------------------------------------------------------------
# Model 2: DAT-internal (DAT experiments only)
# delta_r2 ~ ra_type + ffn_activation + symbol_retrieval + model_variant +
#            (1 | experiment_name) + (1 | dv_name)
#
# Note: ra_type, ffn_activation, and symbol_retrieval are confounded in the
# available data (certain combinations never co-occur), so including all three
# as separate factors produces a rank-deficient design. We catch this and skip
# the model with a message rather than crashing.
# ---------------------------------------------------------------------------
data_dat <- data %>% filter(model_type == "dat")
if (nrow(data_dat) > 0 && n_distinct(data_dat$experiment_name) > 1) {
  cat("\n=== Fitting DAT-internal reading model ===\n")
  fe_str_m2 <- paste(
    "delta_r2 ~ ra_type + ffn_activation + symbol_retrieval +",
    "model_variant"
  )
  re_cands_m2 <- c("(1 | experiment_name)", "(1 | dv_name)")

  cache_file_m2 <- file.path(output_dir, "cache_reading_dat.rds")
  model_m2 <- tryCatch(
    fit_or_load_model(cache_file_m2, data_dat, fe_str_m2, re_cands_m2),
    error = function(e) {
      cat(sprintf("  Skipping DAT-internal model: %s\n", conditionMessage(e)))
      NULL
    }
  )

  if (!is.null(model_m2)) {
    anova_m2 <- as.data.frame(anova(model_m2))
    anova_m2$effect <- rownames(anova_m2)
    write.csv(anova_m2, file.path(output_dir, "reading_dat_anova.csv"), row.names = FALSE)
    cat(sprintf("Wrote DAT-internal reading ANOVA to %s/reading_dat_anova.csv\n", output_dir))

    coef_m2 <- as.data.frame(summary(model_m2)$coefficients)
    coef_m2$term <- rownames(coef_m2)
    write.csv(coef_m2, file.path(output_dir, "reading_dat_coefficients.csv"), row.names = FALSE)
    cat(sprintf("Wrote DAT-internal reading coefficients to %s/reading_dat_coefficients.csv\n", output_dir))

    generate_diagnostics(model_m2, file.path(output_dir, "reading_dat"))
  }
} else {
  cat("\nSkipping DAT-internal reading model: not enough DAT experiments\n")
}

# ---------------------------------------------------------------------------
# Model 3: RCA 264 symbol subset (RCA symbol experiments only)
# delta_r2 ~ symbol_retrieval + model_variant +
#            (1 | seed_name) + (1 | dv_name)
# ---------------------------------------------------------------------------
data_rca <- data %>% filter(analysis_family == "rca_symbol_retrieval_comparison")
if (nrow(data_rca) > 0 && n_distinct(data_rca$experiment_name) > 1) {
  cat("\n=== Fitting RCA 264 symbol subset reading model ===\n")
  fe_str_m3 <- "delta_r2 ~ symbol_retrieval + model_variant"
  re_cands_m3 <- c("(1 | seed_name)", "(1 | dv_name)")

  cache_file_m3 <- file.path(output_dir, "cache_reading_rca.rds")
  model_m3 <- tryCatch(
    fit_or_load_model(cache_file_m3, data_rca, fe_str_m3, re_cands_m3),
    error = function(e) {
      cat(sprintf("  Skipping RCA 264 model: %s\n", conditionMessage(e)))
      NULL
    }
  )

  if (!is.null(model_m3)) {
    anova_m3 <- as.data.frame(anova(model_m3))
    anova_m3$effect <- rownames(anova_m3)
    write.csv(anova_m3, file.path(output_dir, "reading_rca_anova.csv"), row.names = FALSE)
    cat(sprintf("Wrote RCA 264 symbol subset reading ANOVA to %s/reading_rca_anova.csv\n", output_dir))

    coef_m3 <- as.data.frame(summary(model_m3)$coefficients)
    coef_m3$term <- rownames(coef_m3)
    write.csv(coef_m3, file.path(output_dir, "reading_rca_coefficients.csv"), row.names = FALSE)
    cat(sprintf("Wrote RCA 264 symbol subset reading coefficients to %s/reading_rca_coefficients.csv\n", output_dir))

    generate_diagnostics(model_m3, file.path(output_dir, "reading_rca"))
  }
} else {
  cat("\nSkipping RCA 264 symbol subset reading model: not enough RCA experiments\n")
}

cat("\nDone.\n")
