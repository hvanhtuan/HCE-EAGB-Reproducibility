# Independent implementation oracle

This directory verifies the locked NFIP baseline and evaluation path through a second implementation that does not import the study's Python metric or model-fitting modules.

The oracle has two parts:

1. `oracle_metrics.R` recomputes Tweedie deviance, weighted Gini, RMSE, MAE, O/E, grouped calibration slope and maximum grouped calibration error from the released 399,128 cell-level predictions.
2. `oracle_glm.R` constructs the design independently and refits the exposure-weighted penalized Tweedie GLM with a separate Newton solver in R. It also checks the null-model rate against the exposure-weighted training loss rate.

`prepare_inputs.py` is restricted to Parquet-to-CSV conversion, checksum checking and extraction of reference values. It contains no evaluation or model-fitting formulas. The R scripts read plain CSV files and do not call any code under `code/` or `scripts/`.

## Run

Requirements are Python with pandas/pyarrow and Docker.

```powershell
pwsh oracle/run_oracle.ps1
```

The Docker image is pinned to `rocker/r-ver:4.5.1`. Outputs are written to `oracle/results/`; temporary CSV files under `oracle/work/` are not release artifacts.

## Interpretation

This package is an independent implementation oracle. It can detect inconsistencies in exposure weighting, response scale, link inversion, Tweedie power, grouped calibration and the fitted GLM path. It is not an external independent replication because it is shipped with the authors' reproducibility package and uses the same released data snapshot.

The principal machine-readable result is `oracle/results/oracle_report.json`. `SHA256SUMS.txt` covers the oracle source and result files, excluding temporary converted inputs.
