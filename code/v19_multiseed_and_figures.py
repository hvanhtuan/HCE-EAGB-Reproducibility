"""Post-hoc multi-seed refits, practical encoder benchmarks, and compact figures.

The run is checkpointed after each fit. It deliberately distinguishes algorithmic
seed variation from independent event replication and uses the previously selected
Table 14a hyperparameter pairs rather than silently claiming a repeated grid search.
"""
from __future__ import annotations

import ctypes
import gc
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import tweedie_dev
from nfip_lib import LGB_BASE, align_cats, load
from postreview_analysis import MODEL_NAMES, encode_all, feature_frame

ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "results" / "supplementary"
TABLES = ROOT / "results" / "tables" / "sources" / "supplementary"
FIGURES = ROOT / "figures" / "generated"
for directory in (SUPP, TABLES, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)

PROTOCOL = json.loads((ROOT / "configs" / "v19_multiseed_exploratory.json").read_text(encoding="utf-8"))
SEEDS = PROTOCOL["seeds"]
MAX_ROUNDS = PROTOCOL["common_training_budget"]["maximum_boosting_rounds"]
PATIENCE = PROTOCOL["common_training_budget"]["early_stopping_rounds"]
MIN_DELTA = PROTOCOL["common_training_budget"]["minimum_improvement"]
XI_EVAL = PROTOCOL["common_training_budget"]["evaluation_power"]
MODEL_FILTER = [x for x in os.environ.get("V19_MODEL_KEYS", "").split(",") if x]
CSV_PATH = TABLES / ("v19_multiseed_runs.csv" if not MODEL_FILTER else f"v19_multiseed_runs_{'_'.join(MODEL_FILTER)}.csv")


def weighted_dev(y: np.ndarray, pred: np.ndarray, weight: np.ndarray, xi: float = XI_EVAL) -> float:
    return float(np.sum(weight * tweedie_dev(y, np.maximum(pred, 1e-6), xi)) / weight.sum())


def peak_working_set_mb() -> float | None:
    if os.name != "nt":
        return None
    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    counters = PMC(); counters.cb = ctypes.sizeof(counters)
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    query.restype = ctypes.c_int
    ok = query(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return float(counters.PeakWorkingSetSize / 2**20) if ok else None


def load_checkpoint() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    return pd.read_csv(CSV_PATH).to_dict("records")


def save_checkpoint(rows: list[dict]) -> None:
    pd.DataFrame(rows).sort_values(["model_key", "seed"]).to_csv(CSV_PATH, index=False, encoding="utf-8-sig")


def fit_one(d: pd.DataFrame, x: pd.DataFrame, key: str, cfg: dict, seed: int) -> dict:
    tr = d.year.between(2012, 2018).to_numpy()
    va = d.year.between(2019, 2020).to_numpy()
    te = d.year.between(2021, 2023).to_numpy()
    t24 = (d.year.to_numpy() == 2024)
    xtr = x.loc[tr]
    xva = align_cats(xtr, x.loc[va].copy())
    params = dict(
        LGB_BASE,
        objective="tweedie",
        tweedie_variance_power=1.3,
        num_leaves=int(cfg["num_leaves"]),
        min_data_in_leaf=int(cfg["min_data"]),
        num_threads=8,
        seed=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
        deterministic=True,
    )
    started = time.perf_counter()
    train_set = lgb.Dataset(xtr, d.rate.to_numpy()[tr], weight=d.E.to_numpy()[tr])
    valid_set = lgb.Dataset(xva, d.rate.to_numpy()[va], weight=d.E.to_numpy()[va], reference=train_set)
    callbacks = [lgb.early_stopping(PATIENCE, first_metric_only=True, verbose=False, min_delta=MIN_DELTA)]
    model = lgb.train(params, train_set, MAX_ROUNDS, valid_sets=[valid_set], callbacks=callbacks)
    xeval = align_cats(xtr, x.loc[te | t24].copy())
    pred = model.predict(xeval, num_iteration=model.best_iteration)
    nte = int(te.sum())
    row = {
        "model_key": key,
        "model": MODEL_NAMES[key],
        "seed": seed,
        "num_leaves": int(cfg["num_leaves"]),
        "min_data_in_leaf": int(cfg["min_data"]),
        "best_iteration": int(model.best_iteration),
        "reached_cap": bool(model.best_iteration >= MAX_ROUNDS),
        "training_seconds": time.perf_counter() - started,
        "pilot_deviance": weighted_dev(d.rate.to_numpy()[va], model.predict(xva, num_iteration=model.best_iteration), d.E.to_numpy()[va]),
        "deviance_2021_2023": weighted_dev(d.rate.to_numpy()[te], pred[:nte], d.E.to_numpy()[te]),
        "deviance_2024": weighted_dev(d.rate.to_numpy()[t24], pred[nte:], d.E.to_numpy()[t24]),
    }
    del model, train_set, valid_set, xtr, xva, xeval, pred
    gc.collect()
    return row


def summarize(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    frame["rank_2021_2023"] = frame.groupby("seed")["deviance_2021_2023"].rank(method="min")
    frame.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    summary = []
    for key, g in frame.groupby("model_key", sort=False):
        summary.append({
            "model_key": key, "model": g.model.iloc[0], "n_seeds": int(len(g)),
            "deviance_median": float(g.deviance_2021_2023.median()),
            "deviance_q1": float(g.deviance_2021_2023.quantile(.25)),
            "deviance_q3": float(g.deviance_2021_2023.quantile(.75)),
            "rank_median": float(g.rank_2021_2023.median()),
            "rank_q1": float(g.rank_2021_2023.quantile(.25)),
            "rank_q3": float(g.rank_2021_2023.quantile(.75)),
            "rank1_count": int((g.rank_2021_2023 == 1).sum()),
            "iteration_min": int(g.best_iteration.min()), "iteration_max": int(g.best_iteration.max()),
            "cap_hits": int(g.reached_cap.sum()),
        })
    pd.DataFrame(summary).to_csv(TABLES / "v19_multiseed_summary.csv", index=False, encoding="utf-8-sig")
    return {"protocol": PROTOCOL, "runs": frame.to_dict("records"), "summary": summary}


def plot_multiseed(payload: dict) -> None:
    frame = pd.DataFrame(payload["runs"])
    order = frame.groupby("model_key").deviance_2021_2023.median().sort_values().index.tolist()
    short = {"EAGB": "HCE đồng thời", "KhongDiaLy": "Không địa lý", "Phang": "Mã hóa phẳng", "TuanTu": "HCE tuần tự", "DiaLyPhanLoai": "Địa lý nội tại"}
    labels = [short[k] for k in order]
    vals = [frame.loc[frame.model_key == k, "rank_2021_2023"].to_numpy() for k in order]
    fig, ax = plt.subplots(figsize=(8.1, 4.5))
    ax.boxplot(vals, tick_labels=labels, showmeans=True)
    for i, values in enumerate(vals, 1):
        ax.scatter(np.full(len(values), i), values, s=24, color="#176B87", alpha=.75, zorder=3)
    ax.set_ylabel("Thứ hạng deviance (1 = thấp nhất)")
    ax.set_title("Ổn định thứ hạng qua 5 seed thuật toán (hậu kiểm)")
    ax.tick_params(axis="x", labelrotation=12, labelsize=9)
    ax.set_ylim(5.35, .65); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIGURES / "v19_multiseed_ranks.png", dpi=220); plt.close(fig)


def plot_year_event() -> None:
    diag = json.loads((ROOT / "results" / "reference" / "review_diag.json").read_text(encoding="utf-8"))
    yearly = diag["by_year"]
    event = diag["event_dev_share"]
    keep = ["EAGB", "Tuần tự", "Phẳng", "Không địa lý", "Rỗng"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    years = ["2021", "2022", "2023"]
    for model in keep:
        axes[0].plot(years, [yearly[model][y] for y in years], marker="o", label=model)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Tweedie deviance (thang log)")
    axes[0].set_title("Deviance theo năm"); axes[0].grid(alpha=.25)
    events = list(next(iter(event.values())).keys())
    x = np.arange(len(events)); width = .15
    for j, model in enumerate(keep):
        axes[1].bar(x + (j-2)*width, [event[model][e] for e in events], width, label=model)
    axes[1].set_xticks(x, events, rotation=25, ha="right")
    axes[1].set_ylabel("Tỷ trọng tổng deviance")
    axes[1].set_title("Đóng góp deviance theo nhóm sự kiện"); axes[1].grid(axis="y", alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, .1, 1, 1)); fig.savefig(FIGURES / "v19_deviance_year_event.png", dpi=220); plt.close(fig)


def plot_xi_sensitivity(d: pd.DataFrame) -> None:
    test = d.year.between(2021, 2023).to_numpy()
    dt = d.loc[test]
    preds = np.load(ROOT / "results" / "kb2_test_preds.npz")
    keys = preds.files
    labels = ["EAGB", "HCE tuần tự", "Phẳng", "Cấp quận", "Nội tại", "Không địa lý", "EAGB-PG", "HCE thuần", "GLM", "GLM spline", "CatBoost", "Rỗng"]
    xis = np.arange(1.1, 1.91, .1)
    values = {label: [weighted_dev(dt.rate.to_numpy(), preds[key], dt.E.to_numpy(), float(xi)) for xi in xis] for key, label in zip(keys, labels)}
    ranks = {label: [1 + sum(values[other][j] < values[label][j] for other in labels) for j in range(len(xis))] for label in labels}
    out = {"xi": xis.tolist(), "deviance": values, "rank": ranks, "note": "Post-hoc decision sensitivity on the released 2021-2023 predictions."}
    (SUPP / "v19_xi_sensitivity.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8.3, 4.7))
    for label in ["EAGB", "Không địa lý", "EAGB-PG", "HCE thuần", "GLM", "CatBoost", "Rỗng"]:
        ax.plot(xis, ranks[label], marker="o", label=label)
    ax.axvline(1.5, color="black", linestyle="--", linewidth=1, label="ξ_eval = 1,5")
    ax.set_xlabel("ξ_eval"); ax.set_ylabel("Thứ hạng deviance (1 = thấp nhất)")
    ax.set_title("Độ nhạy thứ hạng theo ξ_eval (hậu kiểm)")
    ax.set_ylim(12.5, .5); ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "v19_xi_sensitivity.png", dpi=220); plt.close(fig)


def main() -> None:
    os.chdir(ROOT / "code")
    d = load()
    pre = json.loads((ROOT / "results" / "reference" / "prereg_H1.json").read_text(encoding="utf-8"))
    selected = json.loads((SUPP / "postreview_ablation_windows.json").read_text(encoding="utf-8"))["selected"]
    before = peak_working_set_mb(); started = time.perf_counter(); z = encode_all(d, pre); encode_seconds = time.perf_counter() - started
    bench = {
        "rows": int(len(d)), "hierarchical_blocks": 15, "encoding_seconds": encode_seconds,
        "peak_working_set_mb_after_encoding": peak_working_set_mb(), "peak_working_set_mb_before_encoding": before,
        "encoded_array_bytes": int(sum(v.nbytes for v in z.values())),
        "note": "Observed process-level benchmark on the manuscript build machine; not a cross-platform performance claim."
    }
    (SUPP / "v19_encoder_benchmark.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")
    if os.environ.get("V19_MERGE_CHECKPOINTS") == "1":
        frames = [pd.read_csv(path) for path in TABLES.glob("v19_multiseed_runs*.csv")]
        merged = pd.concat(frames, ignore_index=True).drop_duplicates(["model_key", "seed"], keep="last")
        rows = merged.to_dict("records")
        if len(rows) != 25:
            raise RuntimeError(f"Expected 25 unique model-seed rows, found {len(rows)}")
        global CSV_PATH
        CSV_PATH = TABLES / "v19_multiseed_runs.csv"
        payload = summarize(rows)
        (SUPP / "v19_multiseed_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        plot_multiseed(payload); plot_year_event(); plot_xi_sensitivity(d)
        print(json.dumps({"summary": payload["summary"], "benchmark": bench}, ensure_ascii=False, indent=2), flush=True)
        return
    rows = load_checkpoint(); done = {(str(r["model_key"]), int(r["seed"])) for r in rows}
    for key in (MODEL_FILTER or list(MODEL_NAMES)):
        x = feature_frame(d, key, z)
        for seed in SEEDS:
            if (key, seed) in done:
                continue
            row = fit_one(d, x, key, selected[key], seed)
            rows.append(row); save_checkpoint(rows)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        del x; gc.collect()
    payload = summarize(rows)
    (SUPP / "v19_multiseed_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_multiseed(payload); plot_year_event(); plot_xi_sensitivity(d)
    print(json.dumps({"summary": payload["summary"], "benchmark": bench}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
