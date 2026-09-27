# HCE/EAGB reproducibility package

This repository contains the reference code, locked analysis configuration, machine-readable results, reviewer audit, and figure-building scripts for the HCE/EAGB study on NFIP and freMTPL2 data.

## Reproducibility status

The repository supports two levels of verification:

1. **Artifact verification** checks the repository structure, JSON files, checksums, locked model-selection record, and any generated cell-level prediction file.
2. **Full replay** rebuilds the NFIP risk-cell table from the public raw files, fits the locked models, exports cell-level predictions, and rebuilds derived tables and figures.

Raw NFIP files are not included because of their size and their original distribution terms. Their expected names, sizes, and SHA-256 values are recorded in `data/raw_manifest.json`. A fresh download from OpenFEMA may not be byte-identical to the September 2026 snapshot used in the study.

A clean Docker replay was completed on 2026-09-27 with Python 3.11.15 on Linux/amd64. It rebuilt the 2,453,910-cell study table, reproduced all locked NFIP point metrics exactly, reran robustness and spatial analyses, reran freMTPL2, exported 399,128 test-cell predictions, rebuilt seven figures, and passed release verification. The machine-readable comparison is in `results/replay_verification.json` and the run manifest is in `artifacts/logs/run_manifest_20260927_full_replay.json`.

The committed JSON files in `results/reference/` are the original reference results. Results from a new replay are written directly under `results/`; they do not overwrite `results/reference/`.

## Repository layout

```text
code/                    Original analysis code
code/supplementary/      Supplementary numerical checks and modern baseline
configs/                 Locked seeds and paper workflow configuration
data/                    Data manifest and data dictionary; raw data excluded
results/reference/       Original machine-readable paper results
results/supplementary/   Supplementary analyses
results/audit/           Reviewer-driven matching/equation audit
artifacts/predictions/   Cell-level predictions produced by a full replay
artifacts/model_selection/ Machine-readable selection log
figures/                 Figure scripts, source data, and generated figures
scripts/                 Cross-platform orchestration and release checks
```

## Quick verification

Python 3.11.15 is the reference interpreter.

```bash
python -m pip install -r requirements-lock.txt
python scripts/verify_release.py
```

The repository also includes a container:

```bash
docker build -t hce-repro:v1.0.0 .
docker run --rm hce-repro:v1.0.0
```

## Full replay

Place all files described by `data/raw_manifest.json`, including OpenML freMTPL2freq version 1, under `data/raw/`. Then run:

```bash
python scripts/run_all.py --mode full --rebuild-cells
```

The full workflow preserves the committed preregistration/configuration file `results/prereg_H1.json` and therefore does not reselect hyperparameters on the test period. It runs the locked primary test, robustness analysis, spatial out-of-sample analysis, freMTPL2 comparison, paired comparison, exports cell-level predictions, recomputes the post-review H1b distance bootstrap, builds tabular summaries, rebuilds figures, and verifies the release.

Useful partial commands:

```bash
python scripts/run_all.py --mode verify
python scripts/build_model_selection_log.py
python scripts/export_cell_predictions.py
python scripts/recompute_h1b.py
python scripts/build_tables.py
python figures/build_all.py
python code/numerical_stability.py
```

On Windows, execute these commands from PowerShell. The orchestrator changes into the required working directories automatically.

## Cell-level predictions

After `kb2_h1.py test` creates `results/kb2_test_preds.npz`, the exporter writes:

- `artifacts/predictions/nfip_test_predictions.parquet`
- `artifacts/predictions/nfip_test_predictions.schema.json`

The Parquet file contains one row per 2021–2023 risk cell, observed exposure/loss/count fields, geography at the public aggregate level, the split label, and one prediction column per model. It contains no person or policy identifier.

The replay-generated prediction artifact and schema are committed to this release. They can also be deterministically regenerated from the risk-cell table and saved NPZ output using `scripts/export_cell_predictions.py`.

## Post-review H1b audit

`scripts/recompute_h1b.py` reads the committed cell-level predictions and recomputes the three H1b distance estimands with paired county-cluster bootstrap resampling. This uncertainty analysis is explicitly post-review and exploratory. It does not replace the locked simultaneous point rule, and H1b remains not met because the maximum-decile calibration component exceeds its margin. The pre-full-replay audit is retained under `results/audit/history/` for provenance.

## Model selection and seeds

`configs/seeds.json` is the central seed registry for the release. The historical core seed remains `20260918`, as used in `code/nfip_lib.py` and the preregistered analysis.

`artifacts/model_selection/model_selection.csv` records every tested HCE shrinkage candidate, Tweedie variance-power candidate, selected value, validation score, and locked iteration count recoverable from the pilot and preregistration files. Regenerate it with:

```bash
python scripts/build_model_selection_log.py
```

## Tables and figures

`scripts/build_tables.py` converts the reference or replay JSON into CSV tables under `results/tables/`. The complete figure set is rebuilt by `figures/build_all.py` from the machine-readable data in `figures/data/`. `configs/paper_outputs.json` maps each delivered output to its inputs and builder.

## Post-review numerical stability check

`code/numerical_stability.py` runs the documented post-review stress test for the tree solver over depth, exposure imbalance, ridge penalty, and zero-exposure leaves. It writes all 60 configurations and their condition numbers, forward errors, relative residuals, and normalized backward errors to `results/supplementary/numerical_stability.json`. This check is explicitly exploratory and separate from the preregistered results.

## Integrity and versioning

`SHA256SUMS.txt` records repository artifact hashes. The number of entries can change when release artefacts are added; the authoritative count is the final `verified N release checksums` line from `scripts/verify_release.py`. Regenerate it only when intentionally preparing a new release:

```bash
python scripts/generate_checksums.py
python scripts/verify_release.py
```

For publication, create a Git tag such as `v1.0.0`, create a GitHub Release from that tag, and archive the same release on Zenodo/OSF. Add the resulting repository URL and DOI to `CITATION.cff`; do not claim a DOI before one is issued.

## Limitations

- The raw NFIP snapshot is not redistributed.
- The completed replay took approximately 66 minutes across the NFIP, spatial, freMTPL2, export, and verification stages on the documented two-thread container; runtime is hardware-dependent.
- Historical per-cell prediction arrays were absent from the archived material. The committed cell-level artifact is therefore a deterministic replay output, not a recovered historical file.
- The local preregistration timestamp and hash are preserved, but they are not evidence of an independently timestamped registry deposit.
- Small numerical variation may occur across CPU/BLAS implementations even with fixed seeds. The environment, thread counts, and release hashes should therefore be reported with any replay.

## License

The source code is released under the MIT license in `LICENSE`. OpenFEMA/OpenML data remain subject to their original terms. Committed numerical outputs are provided for scholarly verification and should be cited with the associated study.
