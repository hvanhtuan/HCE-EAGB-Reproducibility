# Cell-level predictions

This directory is populated by `scripts/export_cell_predictions.py` after the locked test workflow creates `results/kb2_test_preds.npz`.

The default Parquet output is ignored by Git because it may be large. For a GitHub Release or archival deposit, attach the Parquet file as a release asset and keep the generated `.schema.json` and SHA-256 value with it.
