# Changelog

## Unreleased — post-full-replay audit synchronization

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
