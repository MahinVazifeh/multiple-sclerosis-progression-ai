# =========================
# Missing data imputation benchmark
# =========================

# Libraries
library(dplyr)
library(nlme)
library(mitml)
library(imputeTS)
library(mice)
library(jomo)
library(emmeans)

# -------------------------
# Utility functions
# -------------------------

RMSE_calculation <- function(original_data, imputed_data, cols = 1:8) {
  original_mat <- as.matrix(original_data[, cols])
  imputed_mat  <- as.matrix(imputed_data[, cols])

  mse  <- mean((original_mat - imputed_mat)^2, na.rm = TRUE)
  rmse <- sqrt(mse)
  rmse
}

store_rmse <- function(results_list, method_name, missing_rate, rmse_value) {
  append(results_list, list(list(
    Method = method_name,
    MissingRate = missing_rate,
    RMSE = rmse_value
  )))
}

fill_group_means <- function(df, group_col = "Patient_number") {
  missing_cols <- names(which(colSums(is.na(df)) > 0))

  if (length(missing_cols) == 0) return(df)

  df %>%
    group_by(.data[[group_col]]) %>%
    mutate(
      across(
        all_of(missing_cols),
        ~ {
          group_mean <- mean(., na.rm = TRUE)
          if (is.nan(group_mean)) {
            .
          } else {
            ifelse(is.na(.), round(group_mean), .)
          }
        }
      )
    ) %>%
    ungroup()
}

smooth_with_ma <- function(df) {
  na_ma(df, k = 4, weighting = "exponential", maxgap = Inf)
}

prepare_lmm_data <- function(data, target_col) {
  complete_rows <- !is.na(data[[target_col]])
  completed_data <- data[complete_rows, , drop = FALSE]
  missing_rows   <- data[!complete_rows, , drop = FALSE]

  if (nrow(missing_rows) == 0) {
    return(list(completed_data = completed_data, missing_rows = missing_rows))
  }

  missing_rows_no_target <- missing_rows[, !(names(missing_rows) %in% target_col), drop = FALSE]

  completed_data <- completed_data %>%
    fill_group_means() %>%
    smooth_with_ma()

  missing_rows_no_target <- missing_rows_no_target %>%
    fill_group_means() %>%
    smooth_with_ma()

  list(
    completed_data = completed_data,
    missing_rows = missing_rows_no_target
  )
}

build_imputed_template <- function(random_NA) {
  random_NA
}

# -------------------------
# Mixed-effects model imputation
# -------------------------

Mean_MixedEffectModel <- function(random_NA) {
  model_specs <- list(
    Pyramidal = list(
      formula = Pyramidal ~ -1 + real_time + EDSS + Cerebellar + Deambulation,
      weights = TRUE
    ),
    Cerebellar = list(
      formula = Cerebellar ~ -1 + real_time + EDSS + Pyramidal + Sphincteric,
      weights = FALSE
    ),
    Thronchioencephalic = list(
      formula = Thronchioencephalic ~ -1 + real_time + EDSS + Deambulation + Cerebellar,
      weights = TRUE
    ),
    Sensitive = list(
      formula = Sensitive ~ -1 + real_time + EDSS,
      weights = FALSE
    ),
    Sphincteric = list(
      formula = Sphincteric ~ -1 + real_time + EDSS + Deambulation,
      weights = FALSE
    ),
    Visual = list(
      formula = Visual ~ -1 + real_time + EDSS + Mental,
      weights = FALSE
    ),
    Mental = list(
      formula = Mental ~ -1 + real_time + EDSS + Visual,
      weights = FALSE
    ),
    Deambulation = list(
      formula = Deambulation ~ -1 + real_time + EDSS,
      weights = FALSE
    )
  )

  target_cols <- names(model_specs)
  imputed_data <- build_imputed_template(random_NA)

  for (col in target_cols) {
    cat("Imputing:", col, "\n")

    prep <- prepare_lmm_data(random_NA, col)
    completed_data <- prep$completed_data
    missing_rows   <- prep$missing_rows

    if (nrow(missing_rows) == 0) {
      next
    }

    spec <- model_specs[[col]]

    lme_args <- list(
      fixed = spec$formula,
      random = ~ real_time | Patient_number,
      data = completed_data,
      method = "REML",
      control = lmeControl(msMaxIter = 1000, msMaxEval = 1000, opt = "optim"),
      na.action = na.omit
    )

    if (isTRUE(spec$weights)) {
      lme_args$weights <- varIdent(form = ~ 1 | as.numeric(real_time))
    }

    model <- do.call(lme, lme_args)
    predictions <- predict(model, newdata = missing_rows)

    imputed_data[[col]][is.na(random_NA[[col]])] <- predictions
  }

  imputed_data <- smooth_with_ma(imputed_data)
  imputed_data
}

# -------------------------
# Data preprocessing
# -------------------------

preprocess_data <- function(data) {
  data %>%
    group_by(Patient_ID) %>%
    mutate(Patient_number = cur_group_id()) %>%
    ungroup() %>%
    group_by(Patient_number) %>%
    arrange(Patient_number, Date_of_visit, .by_group = TRUE) %>%
    filter(n() > 1) %>%
    mutate(real_time = row_number()) %>%
    ungroup() %>%
    select(-Date_of_visit, -Patient_ID)
}

introduce_missingness <- function(data, missing_rate) {
  data %>%
    mutate(
      across(
        Pyramidal:Deambulation,
        ~ ifelse(runif(length(.)) < missing_rate, NA, .)
      )
    )
}

# -------------------------
# Main workflow
# -------------------------

data <- read.csv(file.choose(), header = TRUE)
data_selected <- preprocess_data(data)

rmse_results <- list()
missing_rates <- seq(0.1, 0.5, by = 0.1)
mice_methods <- c("pmm", "cart", "2l.lmer", "norm", "norm.boot", "rf", "norm.predict", "norm.nob")

# This will store the final imputed dataset from the last iteration/method
final_imputed_data <- NULL
final_method_name <- NULL
final_missing_rate <- NULL

for (missing_rate in missing_rates) {
  cat("Processing missing rate:", missing_rate, "\n")

  set.seed(456)
  random_NA <- introduce_missingness(data_selected, missing_rate)

  # Mean + Mixed Effect Model
  mean_mixed_imputed <- Mean_MixedEffectModel(random_NA)
  rmse_val <- RMSE_calculation(data_selected, mean_mixed_imputed)
  rmse_results <- store_rmse(rmse_results, "Mean+MixedEffectModel", missing_rate, rmse_val)

  # Save latest imputed data
  final_imputed_data <- mean_mixed_imputed
  final_method_name <- "Mean+MixedEffectModel"
  final_missing_rate <- missing_rate

  # Linear interpolation
  linear_imputed <- na_interpolation(random_NA)
  rmse_val <- RMSE_calculation(data_selected, linear_imputed)
  rmse_results <- store_rmse(rmse_results, "Linear Interpolation", missing_rate, rmse_val)

  final_imputed_data <- linear_imputed
  final_method_name <- "Linear Interpolation"
  final_missing_rate <- missing_rate

  # Spline interpolation
  spline_imputed <- na_interpolation(random_NA, option = "spline")
  rmse_val <- RMSE_calculation(data_selected, spline_imputed)
  rmse_results <- store_rmse(rmse_results, "Spline Interpolation", missing_rate, rmse_val)

  final_imputed_data <- spline_imputed
  final_method_name <- "Spline Interpolation"
  final_missing_rate <- missing_rate

  # LOCF
  locf_imputed <- na_locf(random_NA, option = "locf", na_remaining = "rev", maxgap = Inf)
  rmse_val <- RMSE_calculation(data_selected, locf_imputed)
  rmse_results <- store_rmse(rmse_results, "LOCF", missing_rate, rmse_val)

  final_imputed_data <- locf_imputed
  final_method_name <- "LOCF"
  final_missing_rate <- missing_rate

  # Weighted moving average
  weighted_imputed <- na_ma(random_NA, k = 4, weighting = "exponential", maxgap = Inf)
  rmse_val <- RMSE_calculation(data_selected, weighted_imputed)
  rmse_results <- store_rmse(rmse_results, "Weighted Moving Average", missing_rate, rmse_val)

  final_imputed_data <- weighted_imputed
  final_method_name <- "Weighted Moving Average"
  final_missing_rate <- missing_rate

  # MICE methods
  for (method in mice_methods) {
    cat("  MICE method:", method, "\n")

    imp <- mice(random_NA, m = 3, maxit = 20, meth = method, printFlag = FALSE)
    imputed_data <- complete(imp)

    rmse_val <- RMSE_calculation(data_selected, imputed_data)
    rmse_results <- store_rmse(rmse_results, paste("MICE -", method), missing_rate, rmse_val)

    final_imputed_data <- imputed_data
    final_method_name <- paste("MICE -", method)
    final_missing_rate <- missing_rate
  }
}

# -------------------------
# Save outputs
# -------------------------

rmse_df <- do.call(rbind, lapply(rmse_results, as.data.frame))
print(rmse_df)
write.csv(rmse_df, "rmse_results_1.csv", row.names = FALSE)

# Final imputed dataset from the last method in the loop
print(paste("Final imputed dataset saved from method:", final_method_name,
            "at missing rate:", final_missing_rate))

print(head(final_imputed_data))
write.csv(final_imputed_data, "final_imputed_data.csv", row.names = FALSE)