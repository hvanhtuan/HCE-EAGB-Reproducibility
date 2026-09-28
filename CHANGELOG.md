# Changelog

## Unreleased — post-full-replay audit synchronization

- Replaced the single encoder timing with three fresh-process repetitions, machine metadata, incremental working-set measurements, and a cached-tree 2024 refresh benchmark.
- Added a dated targeted literature-search audit with exact query families and explicit inclusion/exclusion rules; it is not represented as a systematic review.
- Added an explicitly unexecuted prospective multi-event confirmation protocol covering historical snapshots, equal-budget nested tuning, five seeds, and event-level uncertainty.
- Added five-seed convergence-controlled refits for the five selected geography representations, with a shared 12,000-round cap and stopping rule.
- Added machine-readable per-seed results, rank summaries, encoder resource measurements, and deviance/decision-sensitivity figures.
- Recorded explicitly that the multi-seed run is post-hoc, reuses selected configurations, and does not replace independent-event evaluation.
- Added exposure-balanced H1b calibration tables and a log-scale calibration figure, with the unbounded maximum O/E distance stated explicitly.
- Added fixed-prediction partial-identification bounds and tipping fractions for unmatched and missing-deductible claims at tract-year and county-year scope.
- Added a complete inventory of the 12 released models and a consolidated audit of locked and post-hoc actuarial baselines.
- Renamed the R check as an implementation-independent oracle on the same snapshot and specification; it is not described as third-party replication.
- Added an independent R implementation oracle for the released cell-level metrics, null model, penalized Tweedie GLM, and Tweedie-power sensitivity.
- Added a synthetic fixture with frozen expected results and machine-readable oracle comparisons.
- Recomputed the H1b distance estimands and paired county-cluster bootstrap directly from the released cell-level predictions.
- Preserved the superseded pre-full-replay H1b audit under `results/audit/history/`.
- Added H1b provenance and point-estimand checks to release verification.
- Clarified that checksum counts are taken from the final verifier run.
- Added a 25-candidate same-cap ablation across five geography representations and retained two historical cap-extension fits as explicitly flagged audit rows.
- Added annual 2021-2024 results and paired county-cluster intervals, labelled post-review exploratory rather than confirmatory.
- Added tract-year and county-year sensitivity analyses that include the 1,123 positive claims excluded for missing deductible group.
- Published the complete Table 10 candidate log and a recorded post-review analysis protocol.

## v1.0.0 — 2026-09-27

- Consolidated the archived HCE/EAGB analysis code into one release structure.
- Preserved original, supplementary, and reviewer-audit results in separate namespaces.
- Added a central seed registry and locked workflow configuration.
- Added a machine-readable model-selection log builder.
- Added deterministic export of cell-level NFIP test predictions.
- Added cross-platform full-replay and verification entry points.
- Added machine-readable table builders and the complete existing figure build system.
- Added Docker configuration and a GitHub Actions workflow template (`github-actions-verify.yml.example`).
- Added release-wide SHA-256 verification.
- Added a seeded, post-review numerical-stability stress test and its machine-readable 60-configuration output.
- Completed the clean Docker full replay, added replay outputs and 399,128 cell-level predictions, and documented comparison with the archived reference results.
- Added freMTPL2freq to the raw-data manifest and aligned the prediction exporter with the eight-state study filter.
