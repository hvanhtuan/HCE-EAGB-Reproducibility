options(stringsAsFactors = FALSE, digits = 17)

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "/workspace"
oracle <- file.path(root, "oracle")
outdir <- file.path(oracle, "results")
fixture <- read.csv(file.path(oracle, "synthetic_fixture.csv"))

tweedie_dev <- function(y, mu, power) {
  2 * (pmax(y, 0)^(2 - power) / ((1 - power) * (2 - power)) -
         y * mu^(1 - power) / (1 - power) +
         mu^(2 - power) / (2 - power))
}

values <- c(
  rows = nrow(fixture),
  sum_exposure = sum(fixture$E),
  sum_loss = sum(fixture$S),
  sum_predicted_loss = sum(fixture$E * fixture$mu),
  OE = sum(fixture$S) / sum(fixture$E * fixture$mu),
  mean_prediction = sum(fixture$E * fixture$mu) / sum(fixture$E),
  tweedie_dev_1_3 = sum(fixture$E * tweedie_dev(fixture$S / fixture$E, fixture$mu, 1.3)) / sum(fixture$E),
  tweedie_dev_1_5 = sum(fixture$E * tweedie_dev(fixture$S / fixture$E, fixture$mu, 1.5)) / sum(fixture$E)
)

expected_text <- paste(readLines(file.path(oracle, "expected_results.json")), collapse = "")
extract_number <- function(name) {
  pattern <- paste0('"', name, '"\\s*:\\s*([-+0-9.eE]+)')
  hit <- regmatches(expected_text, regexec(pattern, expected_text, perl = TRUE))[[1]]
  if (length(hit) != 2L) stop(paste("Missing expected value", name))
  as.numeric(hit[[2]])
}
expected <- vapply(names(values), extract_number, numeric(1))
diff <- abs(values - expected)
result <- data.frame(metric = names(values), expected = expected, observed = values, abs_diff = diff, pass = diff < 1e-12)
write.csv(result, file.path(outdir, "synthetic_oracle_check.csv"), row.names = FALSE)
if (!all(result$pass)) stop("Synthetic oracle fixture failed")
cat(sprintf("Synthetic oracle completed: %d checks passed\n", nrow(result)))
