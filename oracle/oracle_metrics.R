options(stringsAsFactors = FALSE, digits = 17)

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "/workspace"
work <- file.path(root, "oracle", "work")
outdir <- file.path(root, "oracle", "results")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

pred <- read.csv(file.path(work, "test_predictions.csv"), check.names = FALSE)
model_map <- read.csv(file.path(work, "model_map.csv"), check.names = FALSE)
reference <- read.csv(file.path(work, "reference_metrics.csv"), check.names = FALSE)

tweedie_unit_deviance <- function(y, mu, power) {
  if (any(y < 0) || any(mu <= 0)) stop("Tweedie domain violation")
  2 * (
    pmax(y, 0)^(2 - power) / ((1 - power) * (2 - power)) -
      y * mu^(1 - power) / (1 - power) +
      mu^(2 - power) / (2 - power)
  )
}

weighted_gini <- function(mu, exposure, loss) {
  ord <- order(-mu, seq_along(mu), method = "radix")
  mu_o <- mu[ord]
  e_o <- exposure[ord]
  s_o <- loss[ord]
  runs <- c(TRUE, mu_o[-1] != mu_o[-length(mu_o)])
  group <- cumsum(runs)
  eg <- as.numeric(rowsum(e_o, group, reorder = FALSE))
  sg <- as.numeric(rowsum(s_o, group, reorder = FALSE))
  x <- c(0, cumsum(eg) / sum(eg))
  y <- c(0, cumsum(sg) / sum(sg))
  2 * sum((y[-1] + y[-length(y)]) * diff(x) / 2) - 1
}

calibration <- function(mu, exposure, loss, n_bins = 10L) {
  fitted_loss <- exposure * mu
  oe <- sum(loss) / sum(fitted_loss)
  ord <- order(mu, seq_along(mu), method = "radix")
  cum_exposure <- cumsum(exposure[ord]) / sum(exposure)
  bin <- pmin(floor(n_bins * cum_exposure), n_bins - 1L) + 1L
  observed <- as.numeric(rowsum(loss[ord], bin, reorder = FALSE))
  expected <- as.numeric(rowsum(fitted_loss[ord], bin, reorder = FALSE))
  exp_e <- as.numeric(rowsum(exposure[ord], bin, reorder = FALSE))
  dec_oe <- observed / expected
  x <- log(expected / exp_e)
  y <- log(pmax(observed / exp_e, 1e-12))
  w <- expected
  xbar <- sum(w * x) / sum(w)
  ybar <- sum(w * y) / sum(w)
  slope <- if (sd(x) < 1e-9) NaN else sum(w * (x - xbar) * (y - ybar)) / sum(w * (x - xbar)^2)
  c(OE = oe, slope = slope, max_dec_dev = max(abs(dec_oe - 1)))
}

evaluate <- function(mu, exposure, loss, power) {
  rate <- loss / exposure
  dev <- tweedie_unit_deviance(rate, mu, power)
  cal <- calibration(mu, exposure, loss)
  c(
    tweedie_dev = sum(exposure * dev) / sum(exposure),
    gini = weighted_gini(mu, exposure, loss),
    rmse_rate = sqrt(sum(exposure * (rate - mu)^2) / sum(exposure)),
    mae_rate = sum(exposure * abs(rate - mu)) / sum(exposure),
    cal
  )
}

metric_rows <- list()
for (i in seq_len(nrow(model_map))) {
  column <- model_map$column[[i]]
  model <- model_map$model[[i]]
  values <- evaluate(pmax(pred[[column]], 1e-6), pred$E, pred$S, 1.5)
  metric_rows[[i]] <- data.frame(model = model, t(values), check.names = FALSE)
}
metrics <- do.call(rbind, metric_rows)
write.csv(metrics, file.path(outdir, "metric_oracle.csv"), row.names = FALSE)

comparison <- merge(reference, metrics, by = "model", suffixes = c("_reference", "_oracle"), sort = FALSE)
fields <- c("tweedie_dev", "gini", "rmse_rate", "mae_rate", "OE", "slope", "max_dec_dev")
for (field in fields) {
  comparison[[paste0(field, "_abs_diff")]] <- abs(
    comparison[[paste0(field, "_oracle")]] - comparison[[paste0(field, "_reference")]]
  )
}
write.csv(comparison, file.path(outdir, "metric_comparison.csv"), row.names = FALSE)

powers <- c(1.1, 1.3, 1.5, 1.7, 1.9)
xi_rows <- list()
k <- 1L
for (power in powers) {
  for (i in seq_len(nrow(model_map))) {
    column <- model_map$column[[i]]
    dev <- tweedie_unit_deviance(pred$S / pred$E, pmax(pred[[column]], 1e-6), power)
    xi_rows[[k]] <- data.frame(
      power = power,
      model = model_map$model[[i]],
      tweedie_dev = sum(pred$E * dev) / sum(pred$E)
    )
    k <- k + 1L
  }
}
xi <- do.call(rbind, xi_rows)
write.csv(xi, file.path(outdir, "xi_sensitivity.csv"), row.names = FALSE)

max_diff <- max(unlist(comparison[paste0(fields, "_abs_diff")]), na.rm = TRUE)
summary <- data.frame(
  check = c("rows", "models", "max_absolute_metric_difference", "all_metrics_within_1e-9"),
  value = c(nrow(pred), nrow(model_map), format(max_diff, scientific = TRUE), max_diff < 1e-9)
)
write.csv(summary, file.path(outdir, "metric_oracle_summary.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(outdir, "sessionInfo.txt"))
cat(sprintf("Metric oracle completed: %d rows, %d models, max abs diff %.17g\n", nrow(pred), nrow(model_map), max_diff))
