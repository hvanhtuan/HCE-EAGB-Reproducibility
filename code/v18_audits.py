"""Post-hoc audits added for the v18 manuscript revision.

The analyses in this module do not alter the locked H1 gate sequence. They:
1. expose the ten-bin calibration components used by H1b;
2. inventory the twelve released prediction columns and actuarial baselines; and
3. partially identify the effect of excluded positive claims on fixed predictions.

The missing-label bounds are exact for fixed predictions under the stated
tract-year or county-year feasibility sets. They do not include retraining.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # Numeric replay remains available in a minimal environment.
    plt = None


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "supplementary"
TABLES = ROOT / "results" / "tables" / "sources" / "supplementary"
FIGURES = ROOT / "figures" / "generated"
for directory in (RESULTS, TABLES, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)

PREDICTIONS = ROOT / "artifacts" / "predictions" / "nfip_test_predictions.parquet"
SCHEMA = ROOT / "artifacts" / "predictions" / "nfip_test_predictions.schema.json"
XI_EVAL = 1.5
PRED_FLOOR = 1e-6
STATES = {"09", "10", "23", "24", "25", "33", "34", "44"}


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def calibration_groups(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    d = frame[["E", "S", column]].copy()
    d[column] = np.maximum(d[column].to_numpy(float), PRED_FLOOR)
    d = d.sort_values(column, kind="mergesort").reset_index(drop=True)
    cumulative = d.E.cumsum().to_numpy() / float(d.E.sum())
    d["decile"] = np.minimum(np.floor(10 * cumulative).astype(int), 9) + 1
    d["predicted_loss"] = d.E * d[column]
    grouped = d.groupby("decile", observed=True).agg(
        cells=("E", "size"),
        exposure=("E", "sum"),
        observed_loss=("S", "sum"),
        predicted_loss=("predicted_loss", "sum"),
    ).reset_index()
    grouped["exposure_share"] = grouped.exposure / d.E.sum()
    grouped["observed_rate"] = grouped.observed_loss / grouped.exposure
    grouped["predicted_rate"] = grouped.predicted_loss / grouped.exposure
    grouped["OE"] = grouped.observed_loss / grouped.predicted_loss
    grouped["abs_OE_minus_1"] = np.abs(grouped.OE - 1.0)
    grouped.insert(0, "model", label)
    return grouped


def write_calibration(frame: pd.DataFrame) -> dict:
    models = {
        "EAGB": "prediction_00",
        "GLM Tweedie": "prediction_08",
    }
    parts = [calibration_groups(frame, column, label) for label, column in models.items()]
    out = pd.concat(parts, ignore_index=True)
    for path in (
        RESULTS / "v18_h1b_decile_calibration.csv",
        TABLES / "v18_h1b_decile_calibration.csv",
    ):
        out.to_csv(path, index=False, encoding="utf-8-sig")

    if plt is not None:
        fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
        for label, color, marker in (("EAGB", "#1f4e79", "o"), ("GLM Tweedie", "#a61c00", "s")):
            g = out[out.model == label]
            ax.plot(g.decile, g.OE, marker=marker, linewidth=2, color=color, label=label)
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Nhóm thập phân vị phơi nhiễm theo dự báo của từng mô hình")
        ax.set_ylabel("O/E (thang log)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(FIGURES / "v18_h1b_calibration.png", dpi=240)
        plt.close(fig)

    summary = {}
    for label in models:
        g = out[out.model == label]
        idx = g.abs_OE_minus_1.idxmax()
        row = out.loc[idx]
        summary[label] = {
            "max_abs_OE_minus_1": float(row.abs_OE_minus_1),
            "max_decile": int(row.decile),
            "OE_at_max": float(row.OE),
            "cells_at_max": int(row.cells),
            "exposure_at_max": float(row.exposure),
            "observed_loss_at_max": float(row.observed_loss),
            "predicted_loss_at_max": float(row.predicted_loss),
        }
    payload = {
        "status": "post-hoc calibration audit; does not change the locked H1b point rule",
        "source_rows": int(len(frame)),
        "grouping": "ten exposure-balanced groups formed separately for each model after stable sorting by prediction",
        "M_definition": "max_d abs(OE_d - 1); unbounded above when predicted loss is small",
        "models": summary,
    }
    json_dump(RESULTS / "v18_h1b_decile_calibration.json", payload)
    return payload


def write_prediction_inventory() -> dict:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    rows = []
    for column, item in schema["prediction_columns"].items():
        rows.append({"prediction_column": column, "model": item["model"], "npz_key": item["npz_key"]})
    inventory = pd.DataFrame(rows)
    for path in (
        RESULTS / "v18_prediction_inventory.csv",
        TABLES / "v18_prediction_inventory.csv",
    ):
        inventory.to_csv(path, index=False, encoding="utf-8-sig")
    payload = {"models": len(rows), "prediction_columns": rows}
    json_dump(RESULTS / "v18_prediction_inventory.json", payload)
    return payload


def write_baseline_audit(frame: pd.DataFrame) -> dict:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    mapping = {v["model"]: k for k, v in schema["prediction_columns"].items()}
    released = [
        "Mô hình rỗng",
        "GLM Tweedie (đối chứng chính)",
        "GLM Tweedie + spline (xấp xỉ GAM)",
        "EAGB-PG (Poisson–Gamma)",
    ]

    def tweedie_unit(y: np.ndarray, mu: np.ndarray, p: float) -> np.ndarray:
        mu = np.maximum(mu, PRED_FLOOR)
        return 2 * (
            y ** (2 - p) / ((1 - p) * (2 - p))
            - y * mu ** (1 - p) / (1 - p)
            + mu ** (2 - p) / (2 - p)
        )

    y = frame.S.to_numpy(float) / frame.E.to_numpy(float)
    e = frame.E.to_numpy(float)
    rows = []
    for model in released:
        pred = np.maximum(frame[mapping[model]].to_numpy(float), PRED_FLOOR)
        rows.append({
            "model": model,
            "source": "released cell-level predictions",
            "tweedie_deviance": float(np.sum(e * tweedie_unit(y, pred, XI_EVAL)) / e.sum()),
            "mean_prediction": float(np.sum(e * pred) / e.sum()),
            "OE": float(frame.S.sum() / np.sum(e * pred)),
        })

    extra = json.loads((ROOT / "results" / "supplementary" / "extra_baselines.json").read_text(encoding="utf-8"))
    for model in ("GLMM lồng nhau (PQL)", "LocalGLMnet", "CANN"):
        metrics = extra["eval_test"][model]
        rows.append({
            "model": model,
            "source": "post-hoc actuarial baseline audit",
            "tweedie_deviance": float(metrics["tweedie_dev"]),
            "mean_prediction": None,
            "OE": float(metrics["OE"]),
        })
    out = pd.DataFrame(rows)
    for path in (
        RESULTS / "v18_baseline_audit.csv",
        TABLES / "v18_baseline_audit.csv",
    ):
        out.to_csv(path, index=False, encoding="utf-8-sig")
    payload = {
        "status": "diagnostic only; does not replace the locked GLM in H1a",
        "xi_eval": XI_EVAL,
        "rows": rows,
        "interpretation": "implementation agreement does not establish baseline adequacy; all listed actuarial alternatives remain above the null deviance on this test window",
    }
    json_dump(RESULTS / "v18_baseline_audit.json", payload)
    return payload


DED = {
    "0": 500, "1": 1000, "2": 2000, "3": 3000, "4": 4000,
    "5": 5000, "9": 750, "A": 10000, "B": 15000, "C": 20000,
    "D": 25000, "E": 50000, "F": 1250, "G": 1500, "H": 200,
}


def raw_claims() -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "NfipClaims_*.parquet")))
    files += sorted(glob.glob(str(ROOT / "data" / "raw" / "part" / "NfipClaims_*.parquet")))
    if not files:
        raise FileNotFoundError("NFIP claim parquet files are required for the missing-label bounds")
    claims = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True).drop_duplicates("id")
    paid = claims.amountPaidOnBuildingClaim.fillna(0) + claims.amountPaidOnContentsClaim.fillna(0)
    return claims.assign(paid=paid.clip(lower=0))


def missing_ded_amounts(claims: pd.DataFrame, scope: str) -> pd.DataFrame:
    code = claims.buildingDeductibleCode.astype("string")
    mask = (
        claims.censusGeoid.notna()
        & claims.yearOfLoss.between(2021, 2023)
        & claims.censusGeoid.astype(str).str[:2].isin(STATES)
        & (~code.isin(DED))
        & (claims.paid > 0)
    )
    out = claims.loc[mask, ["yearOfLoss", "censusGeoid", "paid"]].copy()
    out["year"] = out.yearOfLoss.astype(int)
    out["tract"] = out.censusGeoid.astype(str).str[:11]
    out["county"] = out.tract.str[:5]
    return out.groupby(["year", scope], as_index=False).paid.sum().rename(columns={"paid": "excluded_loss"})


def normalized_unmatched_amounts(claims: pd.DataFrame, scope: str) -> pd.DataFrame:
    source = (ROOT / "code" / "prep_nfip.py").read_text(encoding="utf-8")
    namespace = {"np": np, "pd": pd}
    block = re.search(r"^OCC = .*?^KEY = \[.*?\]$", source, re.M | re.S)
    years = re.search(r"^YEARS = .*?$", source, re.M)
    if block is None or years is None:
        raise RuntimeError("Could not locate NFIP key normalization definitions")
    exec(block.group(0), namespace)
    exec(years.group(0), namespace)
    keys, key = namespace["keys"], namespace["KEY"]

    eligible = claims[
        claims.yearOfLoss.between(2021, 2023)
        & claims.censusGeoid.notna()
        & claims.censusGeoid.astype(str).str[:2].isin(STATES)
        & (claims.paid > 0)
    ].copy()
    normalized = keys(eligible, "numberOfFloorsInTheInsuredBuilding")
    normalized["year"] = eligible.yearOfLoss.astype(int).to_numpy()
    normalized["S"] = eligible.paid.to_numpy(float)
    grouped = normalized.groupby(["year"] + key, observed=True, as_index=False).S.sum()
    cells = pd.read_parquet(ROOT / "data" / "nfip_cells.parquet", columns=["year"] + key)
    cells = cells[cells.year.between(2021, 2023)].drop_duplicates(["year"] + key)
    cells["_matched"] = 1
    joined = grouped.merge(cells, on=["year"] + key, how="left")
    unmatched = joined[joined._matched.isna()].copy()
    unmatched["county"] = unmatched.tract.astype(str).str[:5]
    return unmatched.groupby(["year", scope], as_index=False).S.sum().rename(columns={"S": "excluded_loss"})


def tweedie_unit(y: np.ndarray, mu: np.ndarray, p: float) -> np.ndarray:
    mu = np.maximum(mu, PRED_FLOOR)
    return 2 * (
        y ** (2 - p) / ((1 - p) * (2 - p))
        - y * mu ** (1 - p) / (1 - p)
        + mu ** (2 - p) / (2 - p)
    )


def bound_pair(frame: pd.DataFrame, amounts: pd.DataFrame, scope: str, comparator: str, family: str) -> dict:
    e = frame.E.to_numpy(float)
    y = frame.S.to_numpy(float) / e
    a = np.maximum(frame.prediction_00.to_numpy(float), PRED_FLOOR)
    b = np.maximum(frame[comparator].to_numpy(float), PRED_FLOOR)
    base_total = float(np.sum(e * (tweedie_unit(y, a, XI_EVAL) - tweedie_unit(y, b, XI_EVAL))))
    coefficient = 2 * (a ** (1 - XI_EVAL) - b ** (1 - XI_EVAL)) / (XI_EVAL - 1)
    cells = frame[["year", scope]].copy()
    cells["coefficient"] = coefficient
    extremes = cells.groupby(["year", scope], as_index=False).coefficient.agg(["min", "max"]).reset_index()
    joined = amounts.merge(extremes, on=["year", scope], how="left")
    covered = joined["min"].notna()
    allocated = joined.loc[covered, "excluded_loss"]
    min_effect = float(np.sum(allocated * joined.loc[covered, "min"]))
    max_effect = float(np.sum(allocated * joined.loc[covered, "max"]))
    total_exposure = float(frame.E.sum())

    def crossing_fraction(effect: float) -> float | None:
        if effect == 0 or base_total * effect >= 0:
            return None
        value = -base_total / effect
        return float(value) if 0 <= value <= 1 else None

    return {
        "label_family": family,
        "scope": scope,
        "comparator_column": comparator,
        "input_excluded_loss": float(joined.excluded_loss.sum()),
        "allocated_excluded_loss": float(allocated.sum()),
        "unallocated_excluded_loss": float(joined.loc[~covered, "excluded_loss"].sum()),
        "groups": int(len(joined)),
        "covered_groups": int(covered.sum()),
        "base_deviance_difference": base_total / total_exposure,
        "minimum_deviance_difference_at_full_allocation": (base_total + min_effect) / total_exposure,
        "maximum_deviance_difference_at_full_allocation": (base_total + max_effect) / total_exposure,
        "fraction_to_zero_under_minimizing_allocation": crossing_fraction(min_effect),
        "fraction_to_zero_under_maximizing_allocation": crossing_fraction(max_effect),
    }


def write_missing_label_bounds(frame: pd.DataFrame) -> dict:
    claims = raw_claims()
    families = {}
    for scope in ("tract", "county"):
        missing = missing_ded_amounts(claims, scope)
        unmatched = normalized_unmatched_amounts(claims, scope)
        combined = pd.concat([missing, unmatched]).groupby(["year", scope], as_index=False).excluded_loss.sum()
        families[("missing_ded", scope)] = missing
        families[("unmatched", scope)] = unmatched
        families[("combined", scope)] = combined

    comparisons = {
        "GLM Tweedie": "prediction_08",
        "GBDT mã hóa phẳng": "prediction_02",
        "GBDT không địa lý": "prediction_05",
    }
    rows = []
    for (family, scope), amounts in families.items():
        for label, column in comparisons.items():
            row = bound_pair(frame, amounts, scope, column, family)
            row["comparator"] = label
            rows.append(row)
    out = pd.DataFrame(rows)
    order = [
        "label_family", "scope", "comparator", "input_excluded_loss", "allocated_excluded_loss",
        "unallocated_excluded_loss", "groups", "covered_groups", "base_deviance_difference",
        "minimum_deviance_difference_at_full_allocation", "maximum_deviance_difference_at_full_allocation",
        "fraction_to_zero_under_minimizing_allocation", "fraction_to_zero_under_maximizing_allocation",
    ]
    out = out[order]
    for path in (
        RESULTS / "v18_missing_label_bounds.csv",
        TABLES / "v18_missing_label_bounds.csv",
    ):
        out.to_csv(path, index=False, encoding="utf-8-sig")
    payload = {
        "status": "post-hoc partial-identification bounds on fixed predictions; no retraining",
        "xi_eval": XI_EVAL,
        "test_window": "2021-2023",
        "method": "within each feasible year-scope group, allocate its excluded loss to the cell with the minimum or maximum coefficient of the EAGB-minus-comparator Tweedie deviance difference",
        "rows": rows,
    }
    json_dump(RESULTS / "v18_missing_label_bounds.json", payload)
    return payload


def update_table_manifest() -> None:
    """Register the four row-oriented audit tables after the generic table build."""
    manifest = ROOT / "results" / "tables" / "all_result_sources_manifest.csv"
    entries = {
        "v18_prediction_inventory": 3,
        "v18_baseline_audit": 5,
        "v18_h1b_decile_calibration": 10,
        "v18_missing_label_bounds": 13,
    }
    if manifest.exists():
        rows = pd.read_csv(manifest, encoding="utf-8-sig")
        rows = rows[~rows.source_json.isin([
            f"results/supplementary/{stem}.json" for stem in entries
        ])]
    else:
        rows = pd.DataFrame(columns=["namespace", "source_json", "table_csv", "field_count"])
    additions = pd.DataFrame([
        {
            "namespace": "supplementary",
            "source_json": f"results/supplementary/{stem}.json",
            "table_csv": f"results/tables/sources/supplementary/{stem}.csv",
            "field_count": fields,
        }
        for stem, fields in entries.items()
    ])
    pd.concat([rows, additions], ignore_index=True).to_csv(
        manifest, index=False, encoding="utf-8-sig"
    )


def main() -> None:
    frame = pd.read_parquet(PREDICTIONS)
    outputs = {
        "prediction_inventory": write_prediction_inventory(),
        "baseline_audit": write_baseline_audit(frame),
        "h1b_calibration": write_calibration(frame),
        "missing_label_bounds": write_missing_label_bounds(frame),
    }
    json_dump(RESULTS / "v18_audit_summary.json", {
        "status": "PASS",
        "analyses_are_post_hoc": True,
        "locked_H1_sequence_changed": False,
        "outputs": {
            "prediction_models": outputs["prediction_inventory"]["models"],
            "h1b_rows": outputs["h1b_calibration"]["source_rows"],
            "missing_label_bound_rows": len(outputs["missing_label_bounds"]["rows"]),
        },
    })
    update_table_manifest()
    print(json.dumps({
        "h1b": outputs["h1b_calibration"],
        "bounds": outputs["missing_label_bounds"]["rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
