"""Recompute the H1b estimands from the replay-generated cell predictions.

This is a post-review, exploratory uncertainty analysis.  The locked H1b
decision remains the simultaneous point rule recorded in results/prereg_H1.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from metrics import calib  # noqa: E402

PREDICTIONS = ROOT / "artifacts" / "predictions" / "nfip_test_predictions.parquet"
SCHEMA = PREDICTIONS.with_suffix(".schema.json")
OUTPUT = ROOT / "results" / "audit" / "h1b_estimands.json"
MODEL_NAMES = {
    "EAGB": "EAGB (HCE đồng thời)",
    "GLM": "GLM Tweedie (đối chứng chính)",
}
MARGINS = {"OE": 0.05, "slope": 0.10, "max_dec_dev": 0.10}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prediction_columns(schema: dict[str, object]) -> dict[str, str]:
    mapping = schema.get("prediction_columns", {})
    found: dict[str, str] = {}
    for column, metadata in mapping.items():
        model = metadata.get("model") if isinstance(metadata, dict) else None
        for short, expected in MODEL_NAMES.items():
            if model == expected:
                found[short] = column
    missing = sorted(set(MODEL_NAMES) - set(found))
    if missing:
        raise ValueError(f"Missing model mappings in prediction schema: {missing}")
    return found


def metric_vector(prediction: np.ndarray, exposure: np.ndarray, loss: np.ndarray) -> np.ndarray:
    try:
        result = calib(np.maximum(prediction, 1e-6), exposure, loss)
        return np.array([result["OE"], result["slope"], result["max_dec_dev"]], dtype=float)
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError):
        return np.full(3, np.nan)


def distance_vector(values: np.ndarray) -> np.ndarray:
    return np.array([abs(values[0] - 1.0), abs(values[1] - 1.0), values[2]], dtype=float)


def interval(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("No finite bootstrap values")
    return {
        "lo95": float(np.quantile(finite, 0.025)),
        "hi95": float(np.quantile(finite, 0.975)),
        "finite_replicates": int(len(finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-replicates", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()

    if not PREDICTIONS.exists() or not SCHEMA.exists():
        raise FileNotFoundError("Run scripts/export_cell_predictions.py before recomputing H1b")

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    source_hash = sha256(PREDICTIONS)
    if schema.get("sha256") != source_hash:
        raise ValueError("Prediction Parquet SHA-256 does not match its schema")

    columns = prediction_columns(schema)
    frame = pd.read_parquet(
        PREDICTIONS,
        columns=["county", "E", "S", columns["EAGB"], columns["GLM"]],
    )
    if len(frame) != schema.get("rows"):
        raise ValueError(f"Prediction row count mismatch: {len(frame)} != {schema.get('rows')}")

    exposure = frame["E"].to_numpy(float)
    loss = frame["S"].to_numpy(float)
    predictions = {
        model: frame[column].to_numpy(float) for model, column in columns.items()
    }
    point = {
        model: metric_vector(prediction, exposure, loss)
        for model, prediction in predictions.items()
    }

    counties, inverse = np.unique(frame["county"].astype(str).to_numpy(), return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(counties))]
    rng = np.random.default_rng(args.seed)
    samples = {
        model: np.empty((args.bootstrap_replicates, 3), dtype=float)
        for model in MODEL_NAMES
    }
    for replicate in range(args.bootstrap_replicates):
        selected = rng.integers(0, len(counties), len(counties))
        indices = np.concatenate([members[index] for index in selected])
        for model, prediction in predictions.items():
            samples[model][replicate] = metric_vector(
                prediction[indices], exposure[indices], loss[indices]
            )

    names = ["OE", "slope", "max_dec_dev"]
    components = []
    for index, name in enumerate(names):
        raw_difference = point["EAGB"][index] - point["GLM"][index]
        distance_eagb = distance_vector(point["EAGB"])[index]
        distance_glm = distance_vector(point["GLM"])[index]
        distance_difference = distance_eagb - distance_glm
        raw_bootstrap = samples["EAGB"][:, index] - samples["GLM"][:, index]
        distance_bootstrap = np.array([
            distance_vector(samples["EAGB"][row])[index]
            - distance_vector(samples["GLM"][row])[index]
            for row in range(args.bootstrap_replicates)
        ])
        components.append({
            "metric": name,
            "raw_EAGB": float(point["EAGB"][index]),
            "raw_GLM": float(point["GLM"][index]),
            "raw_difference": float(raw_difference),
            "distance_EAGB": float(distance_eagb),
            "distance_GLM": float(distance_glm),
            "distance_difference": float(distance_difference),
            "margin": MARGINS[name],
            "point_rule_met": bool(distance_difference <= MARGINS[name]),
            "bootstrap_raw_difference": interval(raw_bootstrap),
            "bootstrap_distance_difference": interval(distance_bootstrap),
        })

    output = {
        "analysis_status": "post_review_exploratory",
        "locked_H1b_rule": "simultaneous point rule; unchanged by this bootstrap",
        "source": PREDICTIONS.relative_to(ROOT).as_posix(),
        "source_schema": SCHEMA.relative_to(ROOT).as_posix(),
        "source_sha256": source_hash,
        "source_rows": int(len(frame)),
        "prediction_columns": columns,
        "bootstrap_distances_recomputed": True,
        "bootstrap_unit": "county",
        "county_clusters": int(len(counties)),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
        "paired_resampling": True,
        "components": components,
        "overall_point_rule_met": bool(all(item["point_rule_met"] for item in components)),
        "supersedes": "results/audit/history/h1b_estimands_pre_full_replay.json",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
