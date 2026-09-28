"""Summarize oracle outputs and generate checksums without importing study code."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pandas as pd


ORACLE = Path(__file__).resolve().parent
RESULTS = ORACLE / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    metric_summary = pd.read_csv(RESULTS / "metric_oracle_summary.csv")
    baseline = pd.read_csv(RESULTS / "baseline_oracle_comparison.csv")
    baseline_values = dict(zip(baseline["check"], baseline["value"]))
    baseline_metrics = pd.read_csv(RESULTS / "baseline_oracle_metrics.csv")
    synthetic = pd.read_csv(RESULTS / "synthetic_oracle_check.csv")
    xi_ranking = pd.read_csv(RESULTS / "xi_ranking_summary.csv")
    metric_pass = str(metric_summary.loc[metric_summary.check == "all_metrics_within_1e-9", "value"].iloc[0]).lower() == "true"
    synthetic_pass = bool(synthetic["pass"].all())
    baseline_pass = (
        float(baseline_values["glm_prediction_max_relative"]) < 1e-6
        and float(baseline_values["null_max_abs"]) < 1e-9
        and float(baseline_values["glm_converged"]) == 1.0
    )
    report = {
        "status": "PASS" if metric_pass and synthetic_pass and baseline_pass else "FAIL",
        "metric_oracle": dict(zip(metric_summary["check"], metric_summary["value"])),
        "synthetic_oracle": {
            "checks": int(len(synthetic)),
            "max_absolute_difference": float(synthetic["abs_diff"].max()),
            "all_pass": synthetic_pass,
        },
        "baseline_oracle": baseline_values,
        "baseline_metrics": baseline_metrics.to_dict(orient="records"),
        "xi_eval_sensitivity": xi_ranking.to_dict(orient="records"),
        "interpretation": {
            "metric_path": "Independent R formulas reproduce the locked metrics from released cell-level predictions.",
            "null_model": "The null prediction is checked against the exposure-weighted training loss rate.",
            "glm_model": "A separate R Newton implementation refits the penalized Tweedie GLM from the released cell table.",
            "scope": "This is an independent implementation oracle, not an independent external replication.",
        },
        "python_platform": platform.platform(),
    }
    (RESULTS / "oracle_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    checksum_targets = sorted(
        [p for p in ORACLE.rglob("*") if p.is_file() and "work" not in p.parts and p.name != "SHA256SUMS.txt"],
        key=lambda p: p.as_posix(),
    )
    lines = [f"{sha256(path)}  {path.relative_to(ORACLE).as_posix()}" for path in checksum_targets]
    (ORACLE / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
