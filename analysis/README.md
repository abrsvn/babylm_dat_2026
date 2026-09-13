# Analysis Pipeline

Python and R scripts that produced the paper's tables, figures, and
mixed-effects models, run against user-supplied experiment trees and
official BabyLM evaluation outputs; neither is redistributed.
`regenerate_experiment_matrix_analysis.sh` orchestrates the full
pipeline; `EXPERIMENT_ROOTS` (or `EXPERIMENT_ROOT`) is required and
selects the experiment tree, `ANALYSIS_DIR` is required when
`EXPERIMENT_ROOTS` is set, `RA_SA_RATIO_INPUT_ROOT` is required when
the head-ratio stage runs, and `RCA_SYMBOL_RUN_TAG` is required when
the symbol-retrieval stages run; the remaining environment variables
(`EVAL_DATA_ROOT` and the `RUN_*` toggles) select the other inputs and
stages.

## Requirements

Python dependencies are covered by the repository's `requirements.txt`
(numpy, pandas, statsmodels, PyYAML). The R scripts additionally
require: lme4, lmerTest, dplyr, tidyr, stringr, emmeans, readr,
ggplot2, and car (loaded via `requireNamespace`).

## Inputs

- **Experiment configurations**: user-supplied experiment trees
  (`EXPERIMENT_ROOTS` / `--experiment_root`; `--input-root` for the
  SA/RA head-ratio summary). The pipeline identifies the analysis
  families by directory name, so the supplied trees must contain
  directories named exactly `architecture_comparison`,
  `rca_symbol_retrieval_comparison`, and
  `rca_sa_ra_head_ratio_comparison`. Every run in the three comparison
  families is tabulated in the paper's configuration appendix.
- **Evaluation data**: not redistributed. The scripts that read it
  require `--eval_data_root` / `--eval-data-root` explicitly (the
  official BabyLM full_eval data directory).
- **Evaluation outputs**: not redistributed. The pipeline's first
  stage, `summarize_experiment_matrix.py`, locates each experiment's
  evaluation outputs via `resolve_eval_root` and fails with a
  `FileNotFoundError` naming the five candidate roots when none is
  complete; every downstream R model consumes that CSV. To run the
  pipeline, regenerate the official evaluation and place the outputs at
  `<experiment_dir>/official_reports/` so that each sentence task has
  `best_temperature_report.txt` (or `temperature_summary.json`) and
  `reading/report.txt` exists; `has_eval_outputs` in
  `summarize_experiment_matrix.py` checks for exactly this layout.

## Notes

- The head-ratio stage (`RUN_RA_SA_RATIO_ANALYSIS=1`) runs
  `analyze_ra_sa_ratio_results.py` first, which writes
  `ra_sa_ratio_scores.csv` and `ra_sa_ratio_submeasures.csv` into the
  head-ratio output directory, then `plot_ra_sa_ratio.R` and
  `metric_models_ra_sa_ratio_lmer.R`. The latter fits the mixed-effects
  models behind the paper's head-ratio statistics -- BLiMP, Reading,
  COMPS, and EWoK, each with seed as a random intercept, at 512-token
  context only -- and writes `ra_sa_ratio_lmer_stats.tex`. It expects
  three seeds per ratio.
- Every run used a 16,384-token tokenizer with pad_token_id 3;
  max_seq_len = datapoint_length + 2 when nextlat_enabled is set
  (datapoint_length + 1 otherwise), and tie_lm_head = not nextlat_enabled
  unless set explicitly. The trainer derives the last two in
  `train/runner.py`.
