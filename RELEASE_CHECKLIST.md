# GitHub release checklist

- [ ] Review author and study metadata in `CITATION.cff`.
- [ ] Confirm no raw NFIP policy/claim files are present.
- [ ] Run `python scripts/generate_checksums.py`.
- [ ] Run `python scripts/verify_release.py` on a clean machine/container.
- [ ] Run `pwsh oracle/run_oracle.ps1` or `bash oracle/run_oracle.sh` and confirm `oracle_report.json` has status `PASS`.
- [ ] Run `python code/v18_audits.py` and confirm the summary has status `PASS`, 12 models, 399,128 H1b rows, and 18 missing-label bounds.
- [ ] Run the full workflow if the raw snapshot and compute budget are available.
- [ ] Attach cell-level prediction Parquet and its schema to the release, or document why it cannot be distributed.
- [ ] Commit the release and create the annotated tag `v1.0.0`.
- [ ] Create a GitHub Release from the tag.
- [ ] Archive the same tagged release on Zenodo/OSF.
- [ ] Add the issued repository URL and DOI to `CITATION.cff` and the manuscript.
- [ ] Report the tag, commit SHA, data manifest hash, and DOI to the reviewer.
