#!/usr/bin/env Rscript
# Mixed-effects models for the SA/RA head-ratio sweep at 512-token context.
# Uses 3 seeds per ratio. Seed is a random intercept.
# Models BLiMP, Reading, COMPS, and EWoK. Entity Tracking is not modeled.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(lme4)
  library(lmerTest)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript metric_models_ra_sa_ratio_lmer.R <output_dir>")
}
output_dir <- args[1]

scores_path <- file.path(output_dir, "ra_sa_ratio_scores.csv")
submeasures_path <- file.path(output_dir, "ra_sa_ratio_submeasures.csv")

if (!file.exists(scores_path) || !file.exists(submeasures_path)) {
  stop("Missing CSV files. Run analyze_ra_sa_ratio_results.py first.")
}

scores <- read_csv(scores_path, show_col_types = FALSE) %>%
  mutate(
    ra_head_fraction = as.numeric(ra_head_fraction),
    context_length = as.integer(context_length)
  )

submeasures <- read_csv(submeasures_path, show_col_types = FALSE) %>%
  mutate(
    ra_head_fraction = as.numeric(ra_head_fraction),
    context_length = as.integer(context_length)
  )

# Seed is encoded in the run name as "_seed<N>"; the sweep's first seed
# carries no token, so a name without one is seed 0.
seed_from_name <- function(experiment_name) {
  m <- regexpr("_seed[0-9]+(_|$)", experiment_name)
  malformed <- grepl("_seed", experiment_name, fixed = TRUE) & m == -1L
  if (any(malformed)) {
    stop(sprintf(
      "Cannot parse a seed from run name(s): %s",
      paste(unique(experiment_name[malformed]), collapse = ", ")
    ))
  }
  out <- rep(0L, length(experiment_name))
  out[m != -1L] <- as.integer(gsub("[^0-9]", "", regmatches(experiment_name, m)))
  out
}

scores <- scores %>% mutate(seed = seed_from_name(experiment_name))
submeasures <- submeasures %>% mutate(seed = seed_from_name(experiment_name))

# Filter to 512 context only (3 seeds per ratio)
scores_512 <- scores %>% filter(context_length == 512)
submeasures_512 <- submeasures %>% filter(context_length == 512)

cat("Scores at 512:", nrow(scores_512), "rows\n")
cat("Submeasures at 512:", nrow(submeasures_512), "rows\n")
cat("Seeds:", paste(sort(unique(scores_512$seed)), collapse=", "), "\n")

if (nrow(scores_512) == 0 || nrow(submeasures_512) == 0) {
  stop("No rows at 512-token context; check --input-root and the context filter.")
}

seed_levels <- sort(unique(scores_512$seed))
if (length(seed_levels) < 3) {
  stop(sprintf(
    paste0(
      "The (1 | seed) random intercept needs at least 3 seeds; found %d (%s). ",
      "Fewer levels give a singular fit and unusable standard errors."
    ),
    length(seed_levels), paste(seed_levels, collapse = ", ")
  ))
}

# Prepare submeasure data with submeasure_id factor
prepare_submeasure_data <- function(submeasures_data, task_name) {
  task_data <- submeasures_data %>% filter(task == task_name)
  if (nrow(task_data) == 0) {
    stop(sprintf("No rows for task: %s", task_name))
  }
  task_data %>%
    mutate(submeasure_id = factor(paste(task, submeasure_type, submeasure, sep = "::")))
}

# Fit mixed-effects models
# Aggregate score models: lmer(score ~ ra_head_fraction + (1|seed))
# Submeasure models: lmer(score ~ ra_head_fraction + submeasure_id + (1|seed))

extract_coefs <- function(model, term = "ra_head_fraction") {
  cfs <- summary(model)$coefficients[term, ]
  est <- cfs["Estimate"]
  se <- cfs["Std. Error"]
  tv <- cfs["t value"]
  pv <- cfs["Pr(>|t|)"]
  c(est, se, tv, pv)
}

format_pval <- function(p) {
  if (is.na(p)) return("N/A")
  if (p < 0.001) return("<0.001***")
  if (p < 0.01) return(sprintf("%.3f**", p))
  if (p < 0.05) return(sprintf("%.3f*", p))
  sprintf("%.3f", p)
}

add_row <- function(name, coefs) {
  if (any(is.na(coefs))) {
    stop(sprintf("Model for %s did not produce a complete ra_head_fraction coefficient", name))
  }
  sprintf("%s & %.3f & %.3f & %.3f & %s \\\\", name, coefs[1], coefs[2], coefs[3], format_pval(coefs[4]))
}

# 1. BLiMP submeasure model
blimp_data <- prepare_submeasure_data(submeasures_512, "blimp")
m_blimp <- lmer(score ~ ra_head_fraction + submeasure_id + (1 | seed), data = blimp_data)
s_blimp <- extract_coefs(m_blimp)
cat("BLiMP model fitted\n")

# 2. Reading model
m_reading <- lmer(reading_mean ~ ra_head_fraction + (1 | seed), data = scores_512)
s_reading <- extract_coefs(m_reading)
cat("Reading model fitted\n")

# 3. COMPS submeasure model
comps_data <- prepare_submeasure_data(submeasures_512, "comps")
m_comps <- lmer(score ~ ra_head_fraction + submeasure_id + (1 | seed), data = comps_data)
s_comps <- extract_coefs(m_comps)
cat("COMPS model fitted\n")

# 4. EWoK submeasure model
ewok_data <- prepare_submeasure_data(submeasures_512, "ewok")
m_ewok <- lmer(score ~ ra_head_fraction + submeasure_id + (1 | seed), data = ewok_data)
s_ewok <- extract_coefs(m_ewok)
cat("EWoK model fitted\n")

# Write LaTeX table
tex_lines <- c(
  "\\begin{tabular}{lrrrr}",
  "\\toprule",
  "Metric & Estimate (Slope) & Std. Error & $t$ & $p$ \\\\",
  "\\midrule",
  add_row("BLiMP", s_blimp),
  add_row("Reading", s_reading),
  add_row("COMPS", s_comps),
  add_row("EWoK", s_ewok),
  "\\bottomrule",
  "\\end{tabular}"
)

writeLines(tex_lines, file.path(output_dir, "ra_sa_ratio_lmer_stats.tex"))
cat("Successfully generated SA/RA ratio LME statistical models.\n")
cat("Output:", file.path(output_dir, "ra_sa_ratio_lmer_stats.tex"), "\n")
