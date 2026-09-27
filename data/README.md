# Data placement

Raw data are intentionally excluded from Git. Put OpenFEMA files under `data/raw/` using the names and directory layout recorded in `raw_manifest.json`.

The analysis creates `data/nfip_cells.parquet`. This file is also excluded because it can be rebuilt with:

```bash
cd code
python prep_nfip.py
```

The source data retain their original licenses and terms. Do not commit policy-level or claim-level files to the public repository.
