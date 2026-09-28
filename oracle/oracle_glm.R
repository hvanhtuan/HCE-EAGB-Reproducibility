options(stringsAsFactors = FALSE, digits = 17)
suppressPackageStartupMessages(library(Matrix))

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "/workspace"
work <- file.path(root, "oracle", "work")
outdir <- file.path(root, "oracle", "results")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

train <- read.csv(file.path(work, "glm_train.csv"), check.names = FALSE)
test <- read.csv(file.path(work, "glm_test.csv"), check.names = FALSE)
pred <- read.csv(file.path(work, "test_predictions.csv"), check.names = FALSE)
model_map <- read.csv(file.path(work, "model_map.csv"), check.names = FALSE)

categorical <- c("zone", "occ", "prim", "elev", "pfirm", "floors", "ded", "cov", "state", "county")
numeric <- c("logcov", "cyear", "crs")

one_hot_pair <- function(train_values, test_values, min_frequency = 20L) {
  train_values <- as.character(train_values)
  test_values <- as.character(test_values)
  counts <- table(train_values)
  frequent <- sort(names(counts[counts >= min_frequency]))
  rare <- names(counts[counts < min_frequency])
  columns <- frequent
  if (length(rare) > 0L) columns <- c(columns, "__infrequent__")

  map_values <- function(values, is_train) {
    mapped <- values
    if (length(rare) > 0L) mapped[values %in% rare] <- "__infrequent__"
    if (!is_train) mapped[!(values %in% names(counts))] <- NA_character_
    match(mapped, columns)
  }
  j_train <- map_values(train_values, TRUE)
  j_test <- map_values(test_values, FALSE)
  x_train <- sparseMatrix(
    i = which(!is.na(j_train)), j = j_train[!is.na(j_train)], x = 1,
    dims = c(length(train_values), length(columns)),
    dimnames = list(NULL, columns)
  )
  x_test <- sparseMatrix(
    i = which(!is.na(j_test)), j = j_test[!is.na(j_test)], x = 1,
    dims = c(length(test_values), length(columns)),
    dimnames = list(NULL, columns)
  )
  list(train = x_train, test = x_test, columns = columns)
}

train_parts <- list()
test_parts <- list()
feature_names <- character()
for (column in categorical) {
  pair <- one_hot_pair(train[[column]], test[[column]], 20L)
  train_parts[[length(train_parts) + 1L]] <- pair$train
  test_parts[[length(test_parts) + 1L]] <- pair$test
  feature_names <- c(feature_names, paste0(column, "=", pair$columns))
}

numeric_train <- matrix(0, nrow(train), length(numeric))
numeric_test <- matrix(0, nrow(test), length(numeric))
for (j in seq_along(numeric)) {
  column <- numeric[[j]]
  center <- mean(train[[column]])
  scale <- sqrt(mean((train[[column]] - center)^2))
  numeric_train[, j] <- (train[[column]] - center) / scale
  numeric_test[, j] <- (test[[column]] - center) / scale
}
train_parts[[length(train_parts) + 1L]] <- Matrix(numeric_train, sparse = TRUE)
test_parts[[length(test_parts) + 1L]] <- Matrix(numeric_test, sparse = TRUE)
feature_names <- c(feature_names, numeric)
X <- do.call(cbind, train_parts)
Xtest <- do.call(cbind, test_parts)
colnames(X) <- feature_names
colnames(Xtest) <- feature_names
rm(train_parts, test_parts, numeric_train, numeric_test)
gc()

fit_tweedie_newton <- function(X, y, weights, power = 1.3, alpha = 1e-4, max_iter = 100L, tol = 1e-8) {
  weights <- weights / sum(weights)
  beta <- rep(0, ncol(X))
  intercept <- log(sum(weights * y))

  objective <- function(intercept, beta) {
    eta <- as.numeric(intercept + X %*% beta)
    mu <- exp(pmin(pmax(eta, -40), 40))
    loss <- mu^(2 - power) / (2 - power) - y * mu^(1 - power) / (1 - power)
    sum(weights * loss) + alpha * sum(beta^2) / 2
  }

  current <- objective(intercept, beta)
  history <- data.frame()
  converged <- FALSE
  for (iteration in seq_len(max_iter)) {
    eta <- as.numeric(intercept + X %*% beta)
    mu <- exp(pmin(pmax(eta, -40), 40))
    grad_eta <- mu^(2 - power) - y * mu^(1 - power)
    hess_eta <- (2 - power) * mu^(2 - power) - (1 - power) * y * mu^(1 - power)
    wh <- weights * hess_eta
    wg <- weights * grad_eta

    gradient <- c(sum(wg), as.numeric(crossprod(X, wg)) + alpha * beta)
    Xw <- X
    Xw@x <- Xw@x * sqrt(wh[Xw@i + 1L])
    h00 <- sum(wh)
    h0b <- as.numeric(crossprod(X, wh))
    hbb <- as.matrix(crossprod(Xw))
    diag(hbb) <- diag(hbb) + alpha
    hessian <- rbind(c(h00, h0b), cbind(h0b, hbb))
    step <- as.numeric(solve(hessian, gradient))

    step_scale <- 1
    accepted <- FALSE
    for (line_search in 0:30) {
      new_intercept <- intercept - step_scale * step[[1]]
      new_beta <- beta - step_scale * step[-1]
      candidate <- objective(new_intercept, new_beta)
      if (is.finite(candidate) && candidate <= current) {
        accepted <- TRUE
        break
      }
      step_scale <- step_scale / 2
    }
    if (!accepted) stop("Newton line search failed")
    intercept <- new_intercept
    beta <- new_beta
    max_step <- max(abs(step_scale * step))
    history <- rbind(history, data.frame(iteration = iteration, objective = candidate, max_step = max_step, step_scale = step_scale))
    current <- candidate
    cat(sprintf("iteration=%d objective=%.12g max_step=%.6g scale=%.6g\n", iteration, current, max_step, step_scale))
    if (max_step < tol) {
      converged <- TRUE
      break
    }
  }
  list(intercept = intercept, beta = beta, history = history, converged = converged)
}

y_train <- train$S / train$E
fit <- fit_tweedie_newton(X, y_train, train$E, power = 1.3, alpha = 1e-4, max_iter = 100L, tol = 1e-8)
write.csv(fit$history, file.path(outdir, "glm_newton_history.csv"), row.names = FALSE)
coef_table <- data.frame(feature = c("(Intercept)", feature_names), coefficient = c(fit$intercept, fit$beta))
write.csv(coef_table, file.path(outdir, "glm_oracle_coefficients.csv"), row.names = FALSE)

oracle_glm <- exp(as.numeric(fit$intercept + Xtest %*% fit$beta))
glm_row <- model_map[grepl("GLM Tweedie \\(đối chứng chính\\)", model_map$model), ]
if (nrow(glm_row) != 1L) stop("Cannot identify locked GLM prediction column")
locked_glm <- pred[[glm_row$column[[1]]]]

null_expected <- sum(train$S) / sum(train$E)
null_row <- model_map[model_map$model == "Mô hình rỗng", ]
if (nrow(null_row) != 1L) stop("Cannot identify null-model prediction column")
locked_null <- pred[[null_row$column[[1]]]]

tweedie_dev <- function(y, mu, power) {
  2 * (pmax(y, 0)^(2 - power) / ((1 - power) * (2 - power)) - y * mu^(1 - power) / (1 - power) + mu^(2 - power) / (2 - power))
}
metric <- function(mu) {
  y <- test$S / test$E
  c(
    deviance = sum(test$E * tweedie_dev(y, mu, 1.5)) / sum(test$E),
    OE = sum(test$S) / sum(test$E * mu),
    mean_prediction = sum(test$E * mu) / sum(test$E)
  )
}

comparison <- data.frame(
  check = c(
    "glm_prediction_rmse", "glm_prediction_max_abs", "glm_prediction_max_relative",
    "glm_prediction_correlation", "null_expected_rate", "null_locked_mean",
    "null_max_abs", "glm_converged", "glm_iterations"
  ),
  value = c(
    sqrt(mean((oracle_glm - locked_glm)^2)),
    max(abs(oracle_glm - locked_glm)),
    max(abs(oracle_glm - locked_glm) / pmax(abs(locked_glm), 1e-12)),
    cor(oracle_glm, locked_glm),
    null_expected,
    mean(locked_null),
    max(abs(locked_null - null_expected)),
    fit$converged,
    nrow(fit$history)
  )
)
write.csv(comparison, file.path(outdir, "baseline_oracle_comparison.csv"), row.names = FALSE)

metrics <- rbind(
  data.frame(model = "GLM oracle R", t(metric(oracle_glm))),
  data.frame(model = "GLM locked Python", t(metric(locked_glm))),
  data.frame(model = "Null oracle formula", t(metric(rep(null_expected, nrow(test))))),
  data.frame(model = "Null locked Python", t(metric(locked_null)))
)
write.csv(metrics, file.path(outdir, "baseline_oracle_metrics.csv"), row.names = FALSE)
write.csv(data.frame(oracle_glm = oracle_glm, locked_glm = locked_glm), file.path(outdir, "glm_predictions_comparison.csv"), row.names = FALSE)
cat(sprintf("GLM oracle completed: converged=%s iterations=%d correlation=%.12f\n", fit$converged, nrow(fit$history), cor(oracle_glm, locked_glm)))
