suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
})

# Helper functions for formatting
sig_stars <- function(p) {
  ifelse(is.na(p), "NA",
    ifelse(p < 0.001, "***",
      ifelse(p < 0.01, "**",
        ifelse(p < 0.05, "*", "ns"))))
}

format_p <- function(p) {
  ifelse(is.na(p), "NA",
    ifelse(p < 0.001, "<.001", sprintf("%.3f", p)))
}

validate_complete_crossed_cells <- function(df, label, first_factor, first_levels, second_factor, second_levels) {
  if (nrow(df) == 0) {
    stop(sprintf("Dataset '%s' is empty after filtering.", label))
  }
  required_columns <- c(first_factor, second_factor)
  missing_columns <- setdiff(required_columns, names(df))
  if (length(missing_columns) > 0) {
    stop(sprintf(
      "Dataset '%s' is missing required cell columns: %s",
      label,
      paste(missing_columns, collapse = ", ")
    ))
  }

  expected <- expand.grid(
    first = first_levels,
    second = second_levels,
    stringsAsFactors = FALSE
  )
  expected_keys <- paste(expected$first, expected$second, sep = "\t")
  observed_keys <- unique(paste(
    as.character(df[[first_factor]]),
    as.character(df[[second_factor]]),
    sep = "\t"
  ))
  missing_keys <- setdiff(expected_keys, observed_keys)
  if (length(missing_keys) > 0) {
    missing_labels <- vapply(strsplit(missing_keys, "\t", fixed = TRUE), function(parts) {
      sprintf("%s=%s, %s=%s", first_factor, parts[[1]], second_factor, parts[[2]])
    }, character(1))
    stop(sprintf(
      "Dataset '%s' is missing required cells: %s",
      label,
      paste(missing_labels, collapse = "; ")
    ))
  }
}

p_value_column <- function(df) {
  candidates <- c("Pr(Chi)", "Pr(>Chi)", "Pr(>Chisq)", "Pr(>F)")
  matches <- candidates[candidates %in% names(df)]
  if (length(matches) != 1) {
    stop(sprintf(
      "Expected exactly one p-value column, found %d among available columns: %s",
      length(matches),
      paste(names(df), collapse = ", ")
    ))
  }
  matches
}

model_missing_variables <- function(model, variables) {
  model_variables <- all.vars(delete.response(terms(model)))
  setdiff(variables, model_variables)
}

write_emmeans_contrasts <- function(model, specs, required_variables, output_path, label, adjust = "tukey") {
  missing_variables <- model_missing_variables(model, required_variables)
  if (length(missing_variables) > 0) {
    reason <- sprintf(
      "Skipped because fixed-effect selection removed required variable(s): %s",
      paste(missing_variables, collapse = ", ")
    )
    write.csv(data.frame(status = "skipped", reason = reason), output_path, row.names = FALSE)
    cat(sprintf("Skipped %s Contrasts: %s\n", label, reason))
    return(invisible(FALSE))
  }
  emmeans_result <- emmeans(model, specs = specs, adjust = adjust)
  contrast_rows <- as.data.frame(emmeans_result$contrasts)
  write.csv(contrast_rows, output_path, row.names = FALSE)
  cat(sprintf("Wrote %s Contrasts to %s\n", label, output_path))
  invisible(TRUE)
}

normalize_family <- function(family) {
  if (is.null(family)) {
    return(NULL)
  }
  if (is.character(family)) {
    if (length(family) != 1 || family != "binomial") {
      stop(sprintf("Unsupported family string: %s", paste(family, collapse = ", ")))
    }
    return(binomial())
  }
  if (is.function(family)) {
    return(family())
  }
  family
}

mixed_model_fast_mode <- function() {
  value <- Sys.getenv("FAST_MIXED_MODELS", "0")
  if (!value %in% c("0", "1")) {
    stop(sprintf("FAST_MIXED_MODELS must be 0 or 1, got %s", value))
  }
  value == "1"
}

mixed_model_ultra_fast_mode <- function() {
  value <- Sys.getenv("ULTRA_FAST_MIXED_MODELS", "0")
  if (!value %in% c("0", "1")) {
    stop(sprintf("ULTRA_FAST_MIXED_MODELS must be 0 or 1, got %s", value))
  }
  value == "1"
}

cache_file_for_selection_mode <- function(cache_file, selection_mode) {
  if (selection_mode == "env_fast") {
    return(sub("\\.rds$", "_fast.rds", cache_file))
  }
  if (selection_mode == "env_ultra_fast") {
    return(sub("\\.rds$", "_ultrafast.rds", cache_file))
  }
  cache_file
}

intercept_only_re_candidates <- function(re_candidates) {
  re_candidates[grepl("^\\(\\s*1\\s*\\|\\s*[^)]+\\s*\\)$", re_candidates)]
}

model_cache_signature <- function(df, fe_str, re_candidates, family, fast_fallback, selection_mode) {
  family <- normalize_family(family)
  family_name <- if (is.null(family)) "NULL" else family$family
  payload <- list(
    data = df,
    fe_str = fe_str,
    re_candidates = re_candidates,
    family = family_name,
    fast_fallback = fast_fallback,
    selection_mode = selection_mode
  )
  temp_path <- tempfile(fileext = ".rds")
  on.exit(unlink(temp_path), add = TRUE)
  saveRDS(payload, temp_path)
  unname(tools::md5sum(temp_path))
}

validate_fixed_effect_design <- function(df, fe_str) {
  design <- model.matrix(as.formula(fe_str), data = df)
  rank <- qr(design)$rank
  column_count <- ncol(design)
  if (rank < column_count) {
    stop(sprintf(
      "Fixed-effect design is rank deficient before model fitting: rank %d < %d columns for formula %s",
      rank,
      column_count,
      fe_str
    ))
  }
}

# ---------------------------------------------------------------------------
# Random Effects Backward Elimination
# ---------------------------------------------------------------------------
# Evaluates all combinations of the provided RE candidates, finds the richest
# non-singular model, and then performs backward elimination via LRTs.
backward_eliminate_re <- function(df, fe_str, re_candidates, family = NULL) {

  build_re <- function(active) {
    if (length(active) == 0) return("1") # Fallback to no RE
    paste(active, collapse = " + ")
  }

  try_fit <- function(re) {
    if (re == "1") {
       if (is.null(family)) {
         return(tryCatch(lm(as.formula(fe_str), data = df), error = function(e) NULL))
       } else {
         return(tryCatch(glm(as.formula(fe_str), data = df, family = family), error = function(e) NULL))
       }
    }
    if (is.null(family)) {
      tryCatch(
        lmer(as.formula(paste0(fe_str, " + ", re)), data = df, REML = TRUE),
        error = function(e) NULL
      )
    } else {
      tryCatch(
        glmer(as.formula(paste0(fe_str, " + ", re)), data = df, family = family, control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))),
        error = function(e) NULL
      )
    }
  }

  all_combos <- list()
  for (k in seq(length(re_candidates), 1)) {
    all_combos <- c(all_combos, combn(re_candidates, k, simplify = FALSE))
  }

  fits <- list()
  info <- list()

  for (combo in all_combos) {
    key <- paste(sort(combo), collapse = "+")
    re <- build_re(combo)
    cat(sprintf("  %-30s", key))
    fit <- try_fit(re)
    fits[[key]] <- fit

    if (!is.null(fit)) {
      if (inherits(fit, "lm") || inherits(fit, "glm")) {
         sing <- FALSE
         aic_val <- AIC(fit)
      } else {
         sing <- isSingular(fit)
         aic_val <- AIC(fit)
      }
      cat(sprintf(" AIC=%.1f%s\n", aic_val, if (sing) " [SINGULAR]" else ""))
      info[[key]] <- list(terms = combo, re = re, converged = TRUE, singular = sing, aic = aic_val)
    } else {
      cat(" FAILED\n")
      info[[key]] <- list(terms = combo, re = re, converged = FALSE, singular = NA, aic = NA)
    }
  }

  valid_keys <- names(info)[sapply(info, function(x) x$converged && !x$singular)]
  if (length(valid_keys) == 0) {
    cat("  No non-singular alternatives. Falling back to simple lm.\n")
    return(list(selected_re = "1", model = try_fit("1")))
  }

  n_terms <- sapply(info[valid_keys], function(x) length(x$terms))
  max_n <- max(n_terms)
  richest <- valid_keys[n_terms == max_n]
  richest_aics <- sapply(info[richest], function(x) x$aic)
  start_key <- richest[which.min(richest_aics)]

  current_key <- start_key
  current_terms <- info[[current_key]]$terms
  cat(sprintf("\n  Backward elimination from: %s\n", current_key))

  repeat {
    if (length(current_terms) <= 1) break

    drop_candidates <- list()
    for (term in current_terms) {
      simpler <- setdiff(current_terms, term)
      skey <- paste(sort(simpler), collapse = "+")
      if (is.null(fits[[skey]]) || !info[[skey]]$converged) next

      comp <- tryCatch(anova(fits[[skey]], fits[[current_key]]), error = function(e) NULL)
      if (is.null(comp)) next

      p_col <- p_value_column(comp)
      pv <- comp[2, p_col]
      cat(sprintf("    drop %-15s: Chisq=%.2f p=%s %s%s\n",
                  term, comp[2, "Chisq"], format_p(pv), sig_stars(pv),
                  if (info[[skey]]$singular) " [SINGULAR]" else ""))

      drop_candidates[[term]] <- list(term = term, key = skey, p = pv, singular = info[[skey]]$singular)
    }

    if (length(drop_candidates) == 0) break

    ok <- Filter(function(d) d$p >= 0.05 && !d$singular, drop_candidates)
    if (length(ok) == 0) {
      cat("    All removals significant or singular -> stop\n")
      break
    }

    best <- ok[[which.max(sapply(ok, function(d) d$p))]]
    cat(sprintf("    Dropping '%s' (p=%.3f) -> %s\n", best$term, best$p, best$key))
    current_key <- best$key
    current_terms <- info[[current_key]]$terms
  }

  cat(sprintf("\n  SELECTED RE: %s\n", info[[current_key]]$re))
  list(selected_re = info[[current_key]]$re, model = fits[[current_key]])
}

fit_fast_random_intercept_model <- function(df, fe_str, re_candidates, family = NULL) {
  hierarchy <- character()
  add_re <- function(re) {
    if (!re %in% hierarchy) {
      hierarchy <<- c(hierarchy, re)
    }
  }
  if (length(re_candidates) > 0) {
    for (k in length(re_candidates):1) {
      add_re(paste(re_candidates[1:k], collapse = " + "))
    }
    for (candidate in re_candidates) {
      add_re(candidate)
    }
  }
  add_re("1")

  for (re in hierarchy) {
    cat(sprintf("  Trying RE: %s\n", re))
    form_str <- if (re == "1") fe_str else paste0(fe_str, " + ", re)

    if (is.null(family)) {
      if (re == "1") {
        model <- tryCatch(lm(as.formula(form_str), data = df), error = function(e) NULL)
      } else {
        model <- tryCatch(lmer(as.formula(form_str), data = df, REML = TRUE), error = function(e) NULL)
      }
    } else {
      if (re == "1") {
        model <- tryCatch(glm(as.formula(form_str), data = df, family = family), error = function(e) NULL)
      } else {
        model <- tryCatch(glmer(as.formula(form_str), data = df, family = family, control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))), error = function(e) NULL)
      }
    }

    if (is.null(model)) {
      cat("    [FAILED] falling back...\n")
      next
    }
    if (inherits(model, "lm") || inherits(model, "glm") || !isSingular(model)) {
      cat(sprintf("  SELECTED RE: %s\n", re))
      return(model)
    }
    cat("    [SINGULAR] falling back...\n")
  }

  stop("All random-intercept fast-track models failed.")
}

fit_ultra_fast_model <- function(df, fe_str, re_candidates, family = NULL) {
  re <- if (length(re_candidates) == 0) "1" else paste(re_candidates, collapse = " + ")
  form_str <- if (re == "1") fe_str else paste0(fe_str, " + ", re)
  cat(sprintf("  Fitting fixed structure: %s\n", form_str))

  if (is.null(family)) {
    if (re == "1") {
      model <- lm(as.formula(form_str), data = df)
    } else {
      model <- lmer(as.formula(form_str), data = df, REML = TRUE)
    }
  } else {
    if (re == "1") {
      model <- glm(as.formula(form_str), data = df, family = family)
    } else {
      model <- glmer(as.formula(form_str), data = df, family = family, control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000)))
    }
  }

  if (!inherits(model, "lm") && !inherits(model, "glm") && isSingular(model)) {
    warning(sprintf("Ultra-fast mixed model fit is singular for formula: %s", form_str))
  }
  model
}

# ---------------------------------------------------------------------------
# Fixed Effects Backward Elimination
# ---------------------------------------------------------------------------
# Takes a fitted model and eliminates non-significant highest-order interaction terms
backward_eliminate_fe <- function(model) {
  current_model <- model

  repeat {
    if (inherits(current_model, "lm") && !inherits(current_model, "lmerModLmerTest") && !inherits(current_model, "lmerMod")) {
       break
    }

    drop_tests <- tryCatch(drop1(current_model, test = "Chisq"), error = function(e) NULL)
    if (is.null(drop_tests) || nrow(drop_tests) <= 1) break

    candidates <- drop_tests[-1, ]
    p_col <- p_value_column(candidates)
    p_values <- candidates[[p_col]]
    non_sig <- candidates[!is.na(p_values) & p_values >= 0.05, ]

    if (nrow(non_sig) == 0) {
      cat("  All droppable FE terms significant -> stop\n")
      break
    }

    non_sig_p_values <- non_sig[[p_col]]
    term_to_drop <- rownames(non_sig)[which.max(non_sig_p_values)]
    p_val <- max(non_sig_p_values)

    cat(sprintf("  Dropping FE term: %s (p = %.3f)\n", term_to_drop, p_val))

    new_formula <- update(formula(current_model), paste(". ~ . -", term_to_drop))
    current_model <- update(current_model, new_formula)
  }

  current_model
}

# ---------------------------------------------------------------------------
# Diagnostics Generation
# ---------------------------------------------------------------------------
generate_diagnostics <- function(model, prefix) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("ggplot2 not available. Skipping diagnostics.")
    return()
  }
  library(ggplot2)

  resids <- residuals(model)
  fitted_vals <- fitted(model)

  df <- data.frame(fitted = fitted_vals, residuals = resids)

  # Sample if data is too large to prevent huge PDFs/PNGs and slow plotting
  if (nrow(df) > 50000) {
    df <- df[sample(nrow(df), 50000), ]
  }

  p1 <- ggplot(df, aes(x = fitted, y = residuals)) +
    geom_point(alpha = 0.1) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    geom_smooth(method = "loess", se = FALSE, color = "blue") +
    theme_minimal() +
    labs(x = "Fitted Values", y = "Residuals", title = "Residuals vs Fitted")

  ggsave(paste0(prefix, "_resid_vs_fitted.png"), plot = p1, width = 6, height = 5, bg="white")

  p2 <- ggplot(df, aes(sample = residuals)) +
    stat_qq(alpha = 0.1) +
    stat_qq_line(color = "red") +
    theme_minimal() +
    labs(x = "Theoretical Quantiles", y = "Sample Quantiles", title = "Normal Q-Q")

  ggsave(paste0(prefix, "_qq.png"), plot = p2, width = 6, height = 5, bg="white")
}

# Override anova to provide type-III Wald Chi-square tables for binomial GLMMs.
anova <- function(object, ...) {
  if (inherits(object, "glmerMod")) {
    if (length(list(...)) > 0) {
      orig_anova <- getS3method("anova", "merMod")
      return(orig_anova(object, ...))
    }
    if (!requireNamespace("car", quietly = TRUE)) {
      stop("car is required for GLMM type-III ANOVA p-values")
    }
    res <- as.data.frame(car::Anova(object, type = 3))
    if (!"Pr(>Chisq)" %in% names(res) || any(is.na(res$`Pr(>Chisq)`))) {
      stop("car::Anova did not return complete GLMM p-values")
    }
    return(res)
  } else {
    return(stats::anova(object, ...))
  }
}

# ---------------------------------------------------------------------------
# Caching Model Fits
# ---------------------------------------------------------------------------
fit_or_load_model <- function(cache_file, df, fe_str, re_candidates, family = NULL, fast_fallback = FALSE) {
  family <- normalize_family(family)
  validate_fixed_effect_design(df, fe_str)
  env_fast <- mixed_model_fast_mode()
  env_ultra_fast <- mixed_model_ultra_fast_mode()
  if (env_fast && env_ultra_fast) {
    stop("FAST_MIXED_MODELS=1 and ULTRA_FAST_MIXED_MODELS=1 are mutually exclusive")
  }
  selection_mode <- if (env_ultra_fast) {
    "env_ultra_fast"
  } else if (env_fast) {
    "env_fast"
  } else if (fast_fallback) {
    "fast_fallback"
  } else {
    "full"
  }
  active_re_candidates <- if (env_fast || env_ultra_fast) {
    intercept_only_re_candidates(re_candidates)
  } else {
    re_candidates
  }
  if ((env_fast || env_ultra_fast) && length(active_re_candidates) == 0) {
    stop(sprintf("%s requires at least one random-intercept candidate", selection_mode))
  }
  cache_file <- cache_file_for_selection_mode(cache_file, selection_mode)
  signature <- model_cache_signature(
    df,
    fe_str,
    active_re_candidates,
    family,
    fast_fallback,
    selection_mode
  )
  if (file.exists(cache_file)) {
    cached <- readRDS(cache_file)
    if (
      is.list(cached) &&
      identical(names(cached), c("signature", "model")) &&
      identical(cached$signature, signature)
    ) {
      cat(sprintf("  Loading cached model from %s\n", cache_file))
      return(cached$model)
    }
    cat(sprintf("  Cached model metadata missing or stale at %s; refitting.\n", cache_file))
  }

  if (env_ultra_fast) {
    cat("  Fitting fixed mixed-model structure without selection...\n")
    cat("  --- Ultra-Fast Fixed Structure ---\n")
    m_final <- fit_ultra_fast_model(df, fe_str, active_re_candidates, family = family)
  } else if (env_fast) {
    cat("  Fitting and selecting model (this may take a while)...\n")
    cat("  --- Random Effects Selection (Random-Intercept Fast Track) ---\n")
    m_res <- backward_eliminate_re(df, fe_str, active_re_candidates, family = family)

    cat("  --- Fixed Effects Selection ---\n")
    m_final <- backward_eliminate_fe(m_res$model)
  } else if (fast_fallback) {
    cat("  Fitting and selecting model (this may take a while)...\n")
    cat("  --- Random Effects Fallback Hierarchy (Fast) ---\n")
    m_initial <- fit_fast_random_intercept_model(df, fe_str, active_re_candidates, family = family)

    cat("  --- Fixed Effects Selection ---\n")
    m_final <- backward_eliminate_fe(m_initial)
  } else {
    cat("  Fitting and selecting model (this may take a while)...\n")
    cat("  --- Random Effects Selection ---\n")
    m_res <- backward_eliminate_re(df, fe_str, active_re_candidates, family = family)

    cat("  --- Fixed Effects Selection ---\n")
    m_final <- backward_eliminate_fe(m_res$model)
  }

  cat(sprintf("  Saving fitted model to %s\n", cache_file))
  dir.create(dirname(cache_file), showWarnings = FALSE, recursive = TRUE)
  # Cache files are committed with the report artifacts, so prefer smaller
  # xz-compressed RDS files over faster gzip writes.
  saveRDS(list(signature = signature, model = m_final), cache_file, compress = "xz")

  return(m_final)
}
