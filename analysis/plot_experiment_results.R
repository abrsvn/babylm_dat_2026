#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(stringr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript plot_experiment_results.R <long_scores.csv> <output_dir>\n")
  quit(status = 1)
}

long_scores_path <- args[1]
output_dir <- args[2]
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

data <- read.csv(long_scores_path, stringsAsFactors = FALSE)
if (!"analysis_family" %in% names(data)) {
  stop("long_scores.csv is missing required analysis_family column.")
}
data <- data %>%
  filter(analysis_family %in% c("architecture_comparison", "rca_symbol_retrieval_comparison"))
if (nrow(data) == 0) {
  stop("long_scores.csv has no rows for the selected analysis families.")
}

# Filter to BLiMP
blimp_data <- data %>% filter(metric == "blimp_full")

# Clean up architecture labels for plotting
blimp_data <- blimp_data %>%
  mutate(
    architecture = str_replace(architecture, "dat_", ""),
    architecture = ifelse(architecture == "lm", "Standard Transformer", architecture),
    training_condition = case_when(
      training_condition == "tied_ntp" ~ "Tied Head (NTP)",
      training_condition == "untied_ntp" ~ "Untied Head (NTP)",
      training_condition == "untied_nextlat" ~ "Untied Head (NextLat)",
      TRUE ~ training_condition
    )
  )

# 1. Boxplot of BLiMP scores across architectures and training conditions
p1 <- ggplot(blimp_data, aes(x = architecture, y = score, fill = training_condition)) +
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.7, outlier.shape = NA) +
  geom_point(position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.8), size = 2, alpha = 0.8) +
  theme_minimal() +
  coord_flip() +
  labs(
    title = "BLiMP Performance across Architectures and Objectives",
    subtitle = paste0("N = ", nrow(blimp_data), " experiment runs across ", length(unique(blimp_data$seed_name)), " seeds"),
    x = "Architecture",
    y = "BLiMP Zero-Shot Accuracy",
    fill = "Objective & Head"
  ) +
  theme(
    legend.position = "bottom",
    text = element_text(size = 12),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(output_dir, "blimp_boxplot.png"), p1, width = 10, height = 6, dpi = 300)

# 2. Main effect plot (Average across seeds)
summary_data <- blimp_data %>%
  group_by(architecture, training_condition) %>%
  summarize(
    mean_score = mean(score),
    sd_score = sd(score),
    n = n(),
    se_score = sd_score / sqrt(n),
    .groups = "drop"
  )

p2 <- ggplot(summary_data, aes(x = mean_score, y = architecture, color = training_condition)) +
  geom_point(position = position_dodge(width = 0.5), size = 3) +
  geom_errorbarh(aes(xmin = mean_score - se_score, xmax = mean_score + se_score), 
                 position = position_dodge(width = 0.5), height = 0.2) +
  theme_minimal() +
  labs(
    title = "Mean BLiMP Accuracy (± 1 Standard Error)",
    x = "Mean BLiMP Accuracy",
    y = "Architecture",
    color = "Objective & Head"
  ) +
  theme(
    legend.position = "bottom",
    text = element_text(size = 12),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(output_dir, "blimp_means.png"), p2, width = 10, height = 6, dpi = 300)

cat(sprintf("Saved plots to %s\n", output_dir))
