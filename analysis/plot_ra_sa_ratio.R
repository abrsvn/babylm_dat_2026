# This script generates plots for the SA/RA head ratio sweep, replacing the manual .Rmd process.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript plot_ra_sa_ratio.R <output_dir>")
}
output_dir <- args[1]

library(ggplot2)
library(dplyr)
library(readr)
library(tidyr)

scores_path <- file.path(output_dir, "ra_sa_ratio_scores.csv")
submeasures_path <- file.path(output_dir, "ra_sa_ratio_submeasures.csv")

if (!file.exists(scores_path) || !file.exists(submeasures_path)) {
  stop("Missing CSV files. Run analyze_ra_sa_ratio_results.py first.")
}

scores <- read_csv(scores_path, show_col_types = FALSE) %>%
  mutate(
    architecture = factor(paste0(n_heads_sa, "sa_", n_heads_ra, "ra")),
    context_length = factor(context_length)
  )

submeasures <- read_csv(submeasures_path, show_col_types = FALSE) %>%
  mutate(
    architecture = factor(paste0(n_heads_sa, "sa_", n_heads_ra, "ra")),
    context_length = factor(context_length)
  )

# Reorder architecture factor: by number of SA heads, then NA. But we want 9sa_3ra at top.
order_architectures <- function(arch) {
  arch_chr <- as.character(arch)
  sa_count <- as.integer(sub("sa.*", "", arch_chr))
  ordered_levels <- unique(arch_chr)[order(sa_count)]
  if ("9sa_3ra" %in% ordered_levels) {
    ordered_levels <- c(setdiff(ordered_levels, "9sa_3ra"), "9sa_3ra")
  }
  ordered_levels
}
scores$architecture <- factor(scores$architecture, levels = order_architectures(scores$architecture))
submeasures$architecture <- factor(submeasures$architecture, levels = order_architectures(submeasures$architecture))

task_labels <- c(
  comps = "COMPS",
  comps_full = "COMPS",
  ewok = "EWoK",
  ewok_full = "EWoK",
  entity_tracking = "Entity Tracking",
  entity_tracking_full = "Entity Tracking"
)

display_task_label <- function(task_name) {
  task_chr <- as.character(task_name)
  labels <- unname(task_labels[task_chr])
  ifelse(is.na(labels), task_chr, labels)
}

# 1. BLiMP Average Accuracy
p_blimp <- ggplot(scores, aes(x = blimp_full, y = architecture, colour = context_length, shape = context_length)) +
  geom_point(size = 3) +
  labs(x = "Average accuracy", y = "Architecture", title = "BLiMP-filtered average accuracy by ratio") +
  theme_bw(base_size = 13)

ggsave(file.path(output_dir, "blimp_average.png"), p_blimp, width = 7, height = 5)

# 2. BLiMP Category Accuracy
# We need to extract the BLiMP linguistic categories from the submeasures.
# Submeasures contains 'split', 'category' (linguistics term), etc.
cat_results <- submeasures %>%
  filter(task == "blimp") %>%
  rename(category = submeasure_type) %>%
  group_by(context_length, architecture, category) %>%
  summarise(accuracy = mean(score), .groups = "drop")

cat_means <- cat_results %>%
  group_by(category) %>%
  summarise(mean_acc = mean(accuracy)) %>%
  arrange(mean_acc)

cat_results$category <- factor(cat_results$category, levels = cat_means$category)

p_blimp_cat <- ggplot(cat_results, aes(x = accuracy, y = category, colour = architecture, shape = context_length)) +
  geom_point(size = 2.5, position = position_dodge(width = 0.6)) +
  labs(x = "Accuracy", y = "Linguistics category", colour = "Architecture", title = "BLiMP-filtered accuracy by linguistics category") +
  theme_bw(base_size = 13)

ggsave(file.path(output_dir, "blimp_category.png"), p_blimp_cat, width = 11, height = 6)

# 3. Supplement Average Accuracy
p_supp <- ggplot(scores, aes(x = blimp_supplement, y = architecture, colour = context_length, shape = context_length)) +
  geom_point(size = 3) +
  labs(x = "Average accuracy", y = "Architecture", title = "Supplement-filtered average accuracy by ratio") +
  theme_bw(base_size = 13)

ggsave(file.path(output_dir, "supplement_average.png"), p_supp, width = 7, height = 5)

# 4. Other zero-shot task averages
other_task_averages <- scores %>%
  select(context_length, architecture, comps_full, ewok_full, entity_tracking_full) %>%
  pivot_longer(
    cols = c(comps_full, ewok_full, entity_tracking_full),
    names_to = "task",
    values_to = "score"
  ) %>%
  mutate(
    task = display_task_label(task)
  )

p_other_avg <- ggplot(other_task_averages, aes(x = score, y = architecture, colour = context_length, shape = context_length)) +
  geom_point(size = 3) +
  facet_wrap(~ task, scales = "free_x") +
  labs(x = "Average accuracy", y = "Architecture", title = "Non-BLiMP zero-shot task accuracy by ratio") +
  theme_bw(base_size = 13)

ggsave(file.path(output_dir, "other_zero_shot_average.png"), p_other_avg, width = 10, height = 5)

# 5. Other zero-shot submeasure scores
other_submeasures <- submeasures %>%
  filter(task %in% c("comps", "ewok", "entity_tracking"), submeasure_type != "average") %>%
  mutate(
    task = display_task_label(task),
    submeasure_label = case_when(
      submeasure_type == "uid_accuracy" ~ submeasure,
      TRUE ~ paste(submeasure_type, submeasure, sep = ": ")
    )
  )

p_other_sub <- ggplot(
  other_submeasures,
  aes(x = score, y = reorder(submeasure_label, score, FUN = mean), colour = architecture, shape = context_length)
) +
  geom_point(size = 1.8, position = position_dodge(width = 0.55)) +
  facet_wrap(~ task, scales = "free_y", ncol = 1) +
  labs(x = "Accuracy", y = "Submeasure", colour = "Architecture", title = "Non-BLiMP zero-shot submeasure accuracy by ratio") +
  theme_bw(base_size = 12)

ggsave(file.path(output_dir, "other_zero_shot_submeasures.png"), p_other_sub, width = 11, height = 9)

# 6. Reading Scores
p_reading <- ggplot(scores, aes(x = reading_mean, y = architecture, colour = context_length, shape = context_length)) +
  geom_point(size = 3) +
  labs(x = "Mean reading score (eye tracking + self-paced)", y = "Architecture", title = "Reading score by architecture") +
  theme_bw(base_size = 13)

ggsave(file.path(output_dir, "reading_average.png"), p_reading, width = 7, height = 5)

cat("Successfully generated RCA SA/RA ratio plots.\n")
