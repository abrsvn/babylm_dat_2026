#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(stringr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript plot_subtasks.R <blimp_subtasks.csv> <long_scores.csv> <output_dir>\n")
  quit(status = 1)
}

subtasks_path <- args[1]
long_scores_path <- args[2]
output_dir <- args[3]

subtasks <- read.csv(subtasks_path, stringsAsFactors = FALSE)
long_scores <- read.csv(long_scores_path, stringsAsFactors = FALSE)

# We just need the architecture and training_condition mapping from long_scores
mapping <- long_scores %>% 
  select(experiment_name, experiment_group, architecture, training_condition) %>% 
  distinct()

# Join subtasks with mapping
data <- subtasks %>%
  inner_join(mapping, by = c("experiment_name", "experiment_group"))

# Ling terms
ling_terms <- c("filler_gap_dependency", "determiner_noun_agreement", "binding", 
                "island_effects", "subject_verb_agreement", "ellipsis", 
                "npi_licensing", "quantifiers", "anaphor_agreement", 
                "argument_structure", "control_raising", "s-selection", "irregular_forms")

# Format for plotting
data_long <- data %>%
  pivot_longer(
    cols = any_of(ling_terms),
    names_to = "ling_term",
    values_to = "score"
  ) %>%
  filter(!is.na(score))

data_long <- data_long %>%
  mutate(
    architecture = str_replace(architecture, "dat_", ""),
    architecture = ifelse(architecture == "lm", "Standard Transformer", architecture),
    training_condition = case_when(
      training_condition == "tied_ntp" ~ "Tied Head (NTP)",
      training_condition == "untied_ntp" ~ "Untied Head (NTP)",
      training_condition == "untied_nextlat" ~ "Untied Head (NextLat)",
      TRUE ~ training_condition
    ),
    ling_term = str_replace_all(ling_term, "_", " ")
  )

# Calculate means
summary_data <- data_long %>%
  group_by(architecture, training_condition, ling_term) %>%
  summarize(
    mean_score = mean(score),
    se_score = sd(score) / sqrt(n()),
    .groups = "drop"
  )

# Filter to just the top architectures to keep the plot readable
top_archs <- c("Standard Transformer", "9sa_3ra_relative_swiglu_rca", "9sa_3ra_relative_swiglu_disrca", "9sa_3ra_relative_swiglu_ra")
summary_data <- summary_data %>% filter(architecture %in% top_archs)

p <- ggplot(summary_data, aes(x = ling_term, y = mean_score, color = architecture, shape = training_condition)) +
  geom_point(position = position_dodge(width = 0.6), size = 3) +
  geom_errorbar(aes(ymin = mean_score - se_score, ymax = mean_score + se_score), 
                 position = position_dodge(width = 0.6), width = 0.2) +
  theme_minimal() +
  coord_flip() +
  labs(
    title = "BLiMP Linguistic Category Accuracies",
    subtitle = "Comparing Top DAT Architectures vs Standard Transformer Baseline",
    x = "Linguistics Category",
    y = "Accuracy",
    color = "Architecture",
    shape = "Objective & Head"
  ) +
  theme(
    legend.position = "right",
    text = element_text(size = 11),
    plot.title = element_text(face = "bold")
  )

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
output_path <- file.path(output_dir, "blimp_ling_terms.png")
ggsave(output_path, p, width = 12, height = 8, dpi = 300)

cat(sprintf("Saved subtask plot to %s\n", output_path))
