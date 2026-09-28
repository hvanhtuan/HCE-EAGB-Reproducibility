"""Prepare plain CSV inputs for the independent R oracle.

This script performs file-format conversion and reference-result extraction only.
It deliberately contains no metric, calibration, or model-fitting formulas.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ORACLE = Path(__file__).resolve().parent
WORK = ORACLE / "work"

STUDY_STATES = ["09", "10", "23", "24", "25", "33", "34", "44"]
CATEGORICAL = ["zone", "occ", "prim", "elev", "pfirm", "floors", "ded", "cov", "state", "county"]
NUMERIC = ["logcov", "cyear", "crs"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    pred_path = ROOT / "artifacts" / "predictions" / "nfip_test_predictions.parquet"
    schema_path = pred_path.with_suffix(".schema.json")
    cells_path = ROOT / "data" / "nfip_cells.parquet"
    result_path = ROOT / "results" / "kb2_test.json"
    for path in (pred_path, schema_path, cells_path, result_path):
        if not path.exists():
            raise SystemExit(f"Missing required artifact: {path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    actual_hash = sha256(pred_path)
    if actual_hash != schema["sha256"]:
        raise SystemExit(f"Prediction Parquet checksum mismatch: {actual_hash} != {schema['sha256']}")

    predictions = pd.read_parquet(pred_path)
    if len(predictions) != schema["rows"]:
        raise SystemExit(f"Prediction row mismatch: {len(predictions)} != {schema['rows']}")
    predictions.to_csv(WORK / "test_predictions.csv", index=False, float_format="%.17g")

    mapping_rows = []
    for column, metadata in schema["prediction_columns"].items():
        mapping_rows.append({"column": column, "model": metadata["model"]})
    pd.DataFrame(mapping_rows).to_csv(WORK / "model_map.csv", index=False)

    cells = pd.read_parquet(cells_path)
    cells["state"] = cells["state"].astype(str).str.zfill(2)
    cells["county"] = cells["county"].astype(str)
    study = cells[cells["year"].between(2010, 2024) & cells["state"].isin(STUDY_STATES)].copy()
    for column in CATEGORICAL:
        study[column] = study[column].astype(str)
    glm_columns = ["year", "E", "S", *CATEGORICAL, *NUMERIC]
    train = study.loc[study["year"].between(2012, 2018), glm_columns]
    test = study.loc[study["year"].between(2021, 2023), glm_columns]
    if len(test) != len(predictions):
        raise SystemExit(f"GLM test row mismatch: {len(test)} != {len(predictions)}")
    train.to_csv(WORK / "glm_train.csv", index=False, float_format="%.17g")
    test.to_csv(WORK / "glm_test.csv", index=False, float_format="%.17g")

    reference = json.loads(result_path.read_text(encoding="utf-8"))
    reference_rows = []
    for model, values in reference["eval"].items():
        reference_rows.append(
            {
                "model": model,
                "tweedie_dev": values["tweedie_dev"],
                "gini": values["gini"],
                "rmse_rate": values["rmse_rate"],
                "mae_rate": values["mae_rate"],
                "OE": values["OE"],
                "slope": values["slope"],
                "max_dec_dev": values["max_dec_dev"],
            }
        )
    pd.DataFrame(reference_rows).to_csv(WORK / "reference_metrics.csv", index=False, float_format="%.17g")

    manifest = {
        "prediction_rows": len(predictions),
        "train_rows": len(train),
        "test_rows": len(test),
        "prediction_parquet_sha256": actual_hash,
        "source_cells_sha256": sha256(cells_path),
        "categorical_columns": CATEGORICAL,
        "numeric_columns": NUMERIC,
        "train_years": [2012, 2018],
        "test_years": [2021, 2023],
    }
    (WORK / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
