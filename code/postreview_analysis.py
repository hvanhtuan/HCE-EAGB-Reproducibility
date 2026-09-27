"""Post-review exploratory analyses requested in REVIEWEDv8.

This script deliberately labels every output exploratory.  The current snapshot and
some 2024 aggregates were available before this run, so 2024 is a distribution-shift
check, not a newly sealed confirmatory holdout.

Outputs
-------
artifacts/model_selection/table10_candidates.csv
results/supplementary/postreview_ablation_windows.json
results/supplementary/missing_ded_sensitivity.json
results/tables/sources/supplementary/postreview_ablation_windows.csv
results/tables/sources/supplementary/missing_ded_sensitivity.csv
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from hce import hce_sequential, hce_solve, node_stats
from metrics import tweedie_dev
from nfip_lib import (
    CAT,
    LAG,
    LGB_BASE,
    align_cats,
    build_tree,
    cluster_boot_rel,
    features,
    load,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUPP = RESULTS / "supplementary"
TABLES = RESULTS / "tables" / "sources" / "supplementary"
MODELSEL = ROOT / "artifacts" / "model_selection"
for directory in (SUPP, TABLES, MODELSEL):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 20260927
B = 600
XI_EVAL = 1.5
GRID = [(31, 500), (15, 500), (63, 500), (31, 100), (31, 2000)]
MODEL_NAMES = {
    "EAGB": "EAGB (HCE đồng thời)",
    "TuanTu": "GBDT + HCE tuần tự",
    "Phang": "GBDT + mã hóa phẳng",
    "KhongDiaLy": "GBDT không dùng địa lý",
    "DiaLyPhanLoai": "GBDT địa lý phân loại nội tại",
}


def weighted_dev(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(weight * tweedie_dev(y, np.maximum(pred, 1e-6), XI_EVAL)) / weight.sum())


def encode_all(d: pd.DataFrame, pre: dict) -> dict[str, np.ndarray]:
    years = list(range(2010, 2025))
    e, s = d.E.to_numpy(), d.S.to_numpy()
    tree, node = build_tree(d)
    flat_tree, flat_node = build_tree(d, levels=("tract",))

    def enc(tree_, node_, k: float, levels: int, method: str) -> np.ndarray:
        z = np.full(len(d), np.nan)
        for year in years:
            src = (d.year.to_numpy() + LAG) <= year
            block = d.year.to_numpy() == year
            if not src.any():
                continue
            ev, sv = node_stats(tree_, node_[src], e[src], s[src])
            mu = sv.sum() / ev.sum()
            penalties = [0] + [k] * levels
            if method == "simul":
                theta = hce_solve(tree_, ev, sv, penalties, 1.0, mu)
            else:
                theta = hce_sequential(tree_, ev, sv, penalties, 1.0, mu)
            z[block] = theta[node_[block]]
        return z

    return {
        "EAGB": enc(tree, node, pre["k_selected"]["simul"], 3, "simul"),
        "TuanTu": enc(tree, node, pre["k_selected"]["seq"], 3, "seq"),
        "Phang": enc(flat_tree, flat_node, pre["k_selected"]["flat"], 1, "simul"),
    }


def feature_frame(d: pd.DataFrame, key: str, z: dict[str, np.ndarray]) -> pd.DataFrame:
    if key in z:
        return features(d, z[key])
    if key == "DiaLyPhanLoai":
        return features(d, None, geo_cat=True)
    return features(d, None)


def tune_sequential(d: pd.DataFrame, z: dict[str, np.ndarray], xi: float) -> list[dict]:
    tr = d.year.between(2012, 2018).to_numpy()
    va = d.year.between(2019, 2020).to_numpy()
    x = feature_frame(d, "TuanTu", z)
    xtr = x[tr]
    xva = align_cats(xtr, x[va].copy())
    rows = []
    for leaves, minimum in GRID:
        params = dict(
            LGB_BASE,
            objective="tweedie",
            tweedie_variance_power=xi,
            num_leaves=leaves,
            min_data_in_leaf=minimum,
            num_threads=4,
        )
        started = time.time()
        model = lgb.train(
            params,
            lgb.Dataset(xtr, d.rate.to_numpy()[tr], weight=d.E.to_numpy()[tr]),
            4000,
            valid_sets=[lgb.Dataset(xva, d.rate.to_numpy()[va], weight=d.E.to_numpy()[va])],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        pred = model.predict(xva, num_iteration=model.best_iteration)
        rows.append(
            {
                "model": "TuanTu",
                "num_leaves": leaves,
                "min_data": minimum,
                "best_iter": int(model.best_iteration),
                "pilot_dev": weighted_dev(d.rate.to_numpy()[va], pred, d.E.to_numpy()[va]),
                "sec": time.time() - started,
                "source": "postreview_same_grid",
                "cap": 4000,
                "cap_extension": False,
            }
        )
        print("tune", rows[-1], flush=True)
    del x, xtr, xva
    return rows


def historical_table10_rows() -> list[dict]:
    raw = json.loads((RESULTS / "reference" / "v4_tune.json").read_text(encoding="utf-8"))
    rows = []
    for model, candidates in raw["grid"].items():
        seen = set()
        for item in candidates:
            pair = (int(item["num_leaves"]), int(item["min_data"]))
            extended = int(item["best_iter"]) > 4000
            # Keep both the original capped fit and the explicit extension when present.
            key = pair + (extended,)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "model": model,
                    "num_leaves": pair[0],
                    "min_data": pair[1],
                    "best_iter": int(item["best_iter"]),
                    "pilot_dev": float(item["pilot_dev"]),
                    "sec": float(item["sec"]),
                    "source": "historical_v4_tune",
                    "cap": 12000 if extended else 4000,
                    "cap_extension": extended,
                }
            )
    return rows


def select_same_cap(rows: list[dict]) -> dict[str, dict]:
    selected = {}
    for model in MODEL_NAMES:
        candidates = [r for r in rows if r["model"] == model and not r["cap_extension"]]
        selected[model] = min(candidates, key=lambda r: r["pilot_dev"])
    return selected


def fit_fixed_predict(
    d: pd.DataFrame,
    z: dict[str, np.ndarray],
    key: str,
    config: dict,
) -> np.ndarray:
    tr = d.year.between(2012, 2018).to_numpy()
    x = feature_frame(d, key, z)
    xtr = x[tr]
    params = dict(
        LGB_BASE,
        objective="tweedie",
        tweedie_variance_power=1.3,
        num_leaves=int(config["num_leaves"]),
        min_data_in_leaf=int(config["min_data"]),
        num_threads=4,
    )
    model = lgb.train(
        params,
        lgb.Dataset(xtr, d.rate.to_numpy()[tr], weight=d.E.to_numpy()[tr]),
        num_boost_round=int(config["best_iter"]),
    )
    pred = model.predict(align_cats(xtr, x.copy()))
    del x, xtr, model
    return pred


def summarize_predictions(d: pd.DataFrame, predictions: dict[str, np.ndarray]) -> dict:
    years = [2021, 2022, 2023, 2024]
    out = {"by_year": {}, "windows": {}}
    for year in years:
        mask = d.year.to_numpy() == year
        out["by_year"][str(year)] = {
            MODEL_NAMES[key]: weighted_dev(d.rate.to_numpy()[mask], pred[mask], d.E.to_numpy()[mask])
            for key, pred in predictions.items()
        }
    for label, lo, hi in [("2021-2023", 2021, 2023), ("2022-2023", 2022, 2023), ("2024", 2024, 2024)]:
        mask = d.year.between(lo, hi).to_numpy()
        devs = {
            key: weighted_dev(d.rate.to_numpy()[mask], pred[mask], d.E.to_numpy()[mask])
            for key, pred in predictions.items()
        }
        base = devs["EAGB"]
        out["windows"][label] = {
            "deviance": {MODEL_NAMES[k]: v for k, v in devs.items()},
            "relative_EAGB_vs_comparator": {
                MODEL_NAMES[k]: (base - value) / value for k, value in devs.items() if k != "EAGB"
            },
        }
    return out


def bootstrap_comparisons(d: pd.DataFrame, predictions: dict[str, np.ndarray], lo: int, hi: int) -> dict:
    mask = d.year.between(lo, hi).to_numpy()
    e = d.E.to_numpy()[mask]
    y = d.rate.to_numpy()[mask]
    county = d.county.to_numpy()[mask]
    de = tweedie_dev(y, np.maximum(predictions["EAGB"][mask], 1e-6), XI_EVAL)
    out = {}
    for key in ("TuanTu", "Phang", "KhongDiaLy", "DiaLyPhanLoai"):
        dc = tweedie_dev(y, np.maximum(predictions[key][mask], 1e-6), XI_EVAL)
        bs, point = cluster_boot_rel(de, dc, e, county, B=B, seed=SEED)
        out[MODEL_NAMES[key]] = {
            "relative": point,
            "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
        }
    return out


DED = {
    "0": 500,
    "1": 1000,
    "2": 2000,
    "3": 3000,
    "4": 4000,
    "5": 5000,
    "9": 750,
    "A": 10000,
    "B": 15000,
    "C": 20000,
    "D": 25000,
    "E": 50000,
    "F": 1250,
    "G": 1500,
    "H": 200,
}


def missing_ded_claims() -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "NfipClaims_*.parquet")))
    files += sorted(glob.glob(str(ROOT / "data" / "raw" / "part" / "NfipClaims_*.parquet")))
    claims = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True).drop_duplicates("id")
    paid = claims.amountPaidOnBuildingClaim.fillna(0) + claims.amountPaidOnContentsClaim.fillna(0)
    claims = claims.assign(paid=paid.clip(lower=0))
    code = claims.buildingDeductibleCode.astype("string")
    mask = (
        claims.censusGeoid.notna()
        & claims.yearOfLoss.between(2010, 2024)
        & claims.censusGeoid.astype(str).str[:2].isin(["09", "10", "23", "24", "25", "33", "34", "44"])
        & (~code.isin(DED))
        & (claims.paid > 0)
    )
    out = claims.loc[mask, ["yearOfLoss", "censusGeoid", "paid"]].copy()
    out["year"] = out.yearOfLoss.astype(int)
    out["tract"] = out.censusGeoid.astype(str).str[:11]
    out["county"] = out.tract.str[:5]
    out["N_add"] = 1.0
    out["S_add"] = out.paid.astype(float)
    return out[["year", "tract", "county", "N_add", "S_add"]]


def allocate_missing(d: pd.DataFrame, claims: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, dict]:
    keys = ["year", scope]
    amounts = claims.groupby(keys, as_index=False)[["N_add", "S_add"]].sum()
    exposure = d.groupby(keys, as_index=False).E.sum().rename(columns={"E": "E_group"})
    alloc = d[keys + ["E"]].merge(amounts, on=keys, how="left").merge(exposure, on=keys, how="left")
    alloc[["N_add", "S_add"]] = alloc[["N_add", "S_add"]].fillna(0.0)
    weight = np.where(alloc.E_group.to_numpy() > 0, alloc.E.to_numpy() / alloc.E_group.to_numpy(), 0.0)
    n_add = alloc.N_add.to_numpy() * weight
    s_add = alloc.S_add.to_numpy() * weight
    out = d.copy()
    out["N"] = out.N.to_numpy() + n_add
    out["S"] = out.S.to_numpy() + s_add
    out["rate"] = out.S / out.E
    available = amounts.merge(exposure, on=keys, how="left")
    covered = available.E_group.notna()
    audit = {
        "scope": scope,
        "input_positive_claims": float(claims.N_add.sum()),
        "input_amount": float(claims.S_add.sum()),
        "allocated_positive_claims": float(n_add.sum()),
        "allocated_amount": float(s_add.sum()),
        "unallocated_positive_claims": float(available.loc[~covered, "N_add"].sum()),
        "unallocated_amount": float(available.loc[~covered, "S_add"].sum()),
    }
    return out, audit


def run_sensitivity(d: pd.DataFrame, pre: dict, selected: dict, baseline: dict[str, np.ndarray]) -> dict:
    claims = missing_ded_claims()
    scenarios = {"retained": (d, None)}
    for scope in ("tract", "county"):
        scenarios[f"missing_ded_{scope}_year"] = allocate_missing(d, claims, scope)
    output = {
        "status": "post-review exploratory; configurations fixed from the baseline same-grid analysis",
        "missing_claims": {"positive_claims": int(claims.N_add.sum()), "amount": float(claims.S_add.sum())},
        "scenarios": {},
    }
    for scenario, payload in scenarios.items():
        ds, audit = payload if isinstance(payload, tuple) else (payload, None)
        if scenario == "retained":
            predictions = baseline
        else:
            z = encode_all(ds, pre)
            predictions = {key: fit_fixed_predict(ds, z, key, selected[key]) for key in MODEL_NAMES}
        summary = summarize_predictions(ds, predictions)
        output["scenarios"][scenario] = {
            "allocation_audit": audit,
            "test_2021_2023": summary["windows"]["2021-2023"],
            "bootstrap_EAGB_vs_comparators": bootstrap_comparisons(ds, predictions, 2021, 2023),
        }
        print("sensitivity", scenario, output["scenarios"][scenario]["test_2021_2023"], flush=True)
    return output


def write_flat_csvs(ablation: dict, sensitivity: dict) -> None:
    rows = []
    for year, values in ablation["metrics"]["by_year"].items():
        for model, value in values.items():
            rows.append({"window": year, "model": model, "tweedie_deviance": value})
    for window, payload in ablation["metrics"]["windows"].items():
        for model, value in payload["deviance"].items():
            rows.append({"window": window, "model": model, "tweedie_deviance": value})
    pd.DataFrame(rows).to_csv(TABLES / "postreview_ablation_windows.csv", index=False, encoding="utf-8-sig")

    rows = []
    for scenario, payload in sensitivity["scenarios"].items():
        for model, value in payload["test_2021_2023"]["deviance"].items():
            rows.append({"scenario": scenario, "model": model, "tweedie_deviance": value})
    pd.DataFrame(rows).to_csv(TABLES / "missing_ded_sensitivity.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    protocol = json.loads((ROOT / "configs" / "postreview_exploratory.json").read_text(encoding="utf-8"))
    pre = json.loads((RESULTS / "reference" / "prereg_H1.json").read_text(encoding="utf-8"))
    d = load()
    z = encode_all(d, pre)
    rows = historical_table10_rows()
    rows.extend(tune_sequential(d, z, pre["xi_selected"]))
    selected = select_same_cap(rows)
    for row in rows:
        row["selected_same_cap"] = bool(
            row["model"] in selected
            and row["num_leaves"] == selected[row["model"]]["num_leaves"]
            and row["min_data"] == selected[row["model"]]["min_data"]
            and row["best_iter"] == selected[row["model"]]["best_iter"]
            and not row["cap_extension"]
        )
    pd.DataFrame(rows).to_csv(MODELSEL / "table10_candidates.csv", index=False, encoding="utf-8-sig")

    baseline_predictions = {
        key: fit_fixed_predict(d, z, key, selected[key]) for key in MODEL_NAMES
    }
    ablation = {
        "protocol": protocol,
        "selected": selected,
        "candidate_count": len(rows),
        "same_cap_candidate_count": sum(not row["cap_extension"] for row in rows),
        "metrics": summarize_predictions(d, baseline_predictions),
        "bootstrap_2021_2023": bootstrap_comparisons(d, baseline_predictions, 2021, 2023),
        "bootstrap_2024": bootstrap_comparisons(d, baseline_predictions, 2024, 2024),
        "interpretation": "Post-review exploratory. The current snapshot contains already-accessible outcomes, so these results cannot convert the original gatekeeping sequence into confirmation.",
    }
    (SUPP / "postreview_ablation_windows.json").write_text(
        json.dumps(ablation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sensitivity = run_sensitivity(d, pre, selected, baseline_predictions)
    (SUPP / "missing_ded_sensitivity.json").write_text(
        json.dumps(sensitivity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_flat_csvs(ablation, sensitivity)
    print(json.dumps({"selected": selected, "ablation": ablation["metrics"], "missing": sensitivity}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
