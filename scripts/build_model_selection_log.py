"""Build a machine-readable model-selection log from pilot and locked files."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results" / "supplementary" / "kb2_pilot.json"
PREREG = ROOT / "results" / "prereg_H1.json"
OUTPUT = ROOT / "artifacts" / "model_selection" / "model_selection.csv"

FIELDS = [
    "stage", "parameter_family", "candidate", "validation_metric",
    "validation_value", "selected", "selection_reason", "train_years",
    "validation_years", "test_years", "test_set_accessed", "seed",
    "locked_at_utc", "source_file"
]


def main() -> None:
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    common = {
        "train_years": "2012-2018",
        "validation_years": "2019-2020",
        "test_years": "2021-2023",
        "test_set_accessed": "false",
        "seed": 20260918,
        "locked_at_utc": prereg.get("locked_at_utc", ""),
    }

    for family in ("simul", "seq", "flat", "par"):
        scores = pilot["meta"][f"kgrid_{family}"]
        chosen = str(prereg["k_selected"][family])
        for candidate, score in scores.items():
            rows.append({
                **common,
                "stage": "pilot",
                "parameter_family": f"hce_k_{family}",
                "candidate": candidate,
                "validation_metric": "training-source temporal encoding deviance",
                "validation_value": score,
                "selected": str(candidate == chosen).lower(),
                "selection_reason": "minimum recorded deviance" if candidate == chosen else "not minimum",
                "source_file": "results/supplementary/kb2_pilot.json",
            })

    chosen_xi = str(prereg["xi_selected"])
    for candidate, score in pilot["meta"]["xi_pilot"].items():
        rows.append({
            **common,
            "stage": "pilot",
            "parameter_family": "tweedie_variance_power",
            "candidate": candidate,
            "validation_metric": "weighted Tweedie deviance",
            "validation_value": score,
            "selected": str(candidate == chosen_xi).lower(),
            "selection_reason": "minimum pilot deviance" if candidate == chosen_xi else "not minimum",
            "source_file": "results/supplementary/kb2_pilot.json",
        })

    for model, rounds in prereg["rounds"].items():
        rows.append({
            **common,
            "stage": "locked_configuration",
            "parameter_family": f"iterations::{model}",
            "candidate": rounds,
            "validation_metric": "pilot early-stopping iteration",
            "validation_value": rounds,
            "selected": "true",
            "selection_reason": "locked before final test",
            "source_file": "results/prereg_H1.json",
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
