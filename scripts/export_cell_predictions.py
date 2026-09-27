"""Export the locked NFIP test predictions as a documented cell-level Parquet file."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, default=ROOT / "data" / "nfip_cells.parquet")
    parser.add_argument("--predictions", type=Path, default=ROOT / "results" / "kb2_test_preds.npz")
    parser.add_argument("--metrics", type=Path, default=ROOT / "results" / "kb2_test.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "predictions" / "nfip_test_predictions.parquet")
    args = parser.parse_args()

    for path in (args.cells, args.predictions, args.metrics):
        if not path.exists():
            raise SystemExit(f"Missing required replay artifact: {path}")

    cells = pd.read_parquet(args.cells)
    mask = cells["year"].between(2021, 2023)
    test = cells.loc[mask].reset_index(drop=True)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    model_names = list(metrics["eval"].keys())

    archive = np.load(args.predictions)
    prediction_keys = list(archive.files)
    if len(model_names) != len(prediction_keys):
        raise SystemExit(
            f"Prediction/model count mismatch: {len(prediction_keys)} arrays, {len(model_names)} models"
        )

    key_candidates = [
        "state", "county", "tract", "year", "zone", "occ", "prim",
        "elev", "pfirm", "floors", "ded", "cov"
    ]
    key_columns = [column for column in key_candidates if column in test.columns]
    hashes = pd.util.hash_pandas_object(test[key_columns].astype(str), index=False).to_numpy(dtype="uint64")

    base_columns = [column for column in ["state", "county", "tract", "year", "E", "S", "N"] if column in test]
    output = test[base_columns].copy()
    output.insert(0, "cell_id", [f"cell-{value:016x}" for value in hashes])
    output["split"] = "test_2021_2023"

    mapping: dict[str, dict[str, str]] = {}
    for index, (array_key, model_name) in enumerate(zip(prediction_keys, model_names)):
        values = archive[array_key]
        if len(values) != len(output):
            raise SystemExit(f"Length mismatch for {model_name}: {len(values)} != {len(output)}")
        column = f"prediction_{index:02d}"
        output[column] = values
        mapping[column] = {"model": model_name, "npz_key": array_key}

    if output["cell_id"].duplicated().any():
        raise SystemExit("cell_id is not unique; revise the declared risk-cell key")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    schema_path = args.output.with_suffix(".schema.json")
    schema = {
        "release": "1.0.0",
        "rows": len(output),
        "columns": {column: str(dtype) for column, dtype in output.dtypes.items()},
        "risk_cell_key": key_columns,
        "prediction_columns": mapping,
        "source_cells": str(args.cells),
        "source_predictions": str(args.predictions),
        "sha256": sha256(args.output),
        "contains_person_or_policy_identifier": False,
    }
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(output):,} cells to {args.output.relative_to(ROOT)}")
    print(f"SHA-256: {schema['sha256']}")


if __name__ == "__main__":
    main()
