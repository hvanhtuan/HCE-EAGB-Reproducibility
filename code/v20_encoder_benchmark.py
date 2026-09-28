"""Repeated fresh-process benchmark for the three released geographic encoders.

This is an observed implementation benchmark, not a cross-platform complexity
claim.  Each repetition loads the released cell table in a fresh process, times
the full 15-block rebuild, then times a cached-tree refresh for the 2024 block.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from hce import hce_sequential, hce_solve, node_stats
from nfip_lib import LAG, build_tree, load
from postreview_analysis import encode_all

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "supplementary" / "v20_encoder_benchmark.json"
CSV = ROOT / "results" / "tables" / "sources" / "supplementary" / "v20_encoder_benchmark_runs.csv"


def memory_mib() -> tuple[float | None, float | None]:
    if os.name != "nt":
        return None, None

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PMC()
    counters.cb = ctypes.sizeof(counters)
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    query.restype = ctypes.c_int
    ok = query(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        return None, None
    return counters.WorkingSetSize / 2**20, counters.PeakWorkingSetSize / 2**20


def refresh_2024(d, pre: dict) -> dict[str, np.ndarray]:
    e, s = d.E.to_numpy(), d.S.to_numpy()
    years = d.year.to_numpy()
    src = (years + LAG) <= 2024
    block = years == 2024
    tree, node = build_tree(d)
    flat_tree, flat_node = build_tree(d, levels=("tract",))

    def solve(tree_, node_, k: float, levels: int, method: str) -> np.ndarray:
        ev, sv = node_stats(tree_, node_[src], e[src], s[src])
        mu = sv.sum() / ev.sum()
        penalties = [0] + [k] * levels
        theta = (hce_solve(tree_, ev, sv, penalties, 1.0, mu)
                 if method == "simul" else hce_sequential(tree_, ev, sv, penalties, 1.0, mu))
        return np.asarray(theta)[node_[block]]

    # Tree construction is deliberately outside the timed refresh interval.
    started = time.perf_counter()
    result = {
        "simultaneous": solve(tree, node, pre["k_selected"]["simul"], 3, "simul"),
        "sequential": solve(tree, node, pre["k_selected"]["seq"], 3, "seq"),
        "flat": solve(flat_tree, flat_node, pre["k_selected"]["flat"], 1, "simul"),
    }
    result["seconds"] = time.perf_counter() - started
    return result


def worker(run: int, path: Path) -> None:
    os.chdir(ROOT / "code")
    d = load()
    pre = json.loads((ROOT / "results" / "reference" / "prereg_H1.json").read_text(encoding="utf-8"))
    current_before, peak_before = memory_mib()
    started = time.perf_counter()
    encoded = encode_all(d, pre)
    full_seconds = time.perf_counter() - started
    current_after, peak_after = memory_mib()
    refresh = refresh_2024(d, pre)
    current_refresh, peak_refresh = memory_mib()
    payload = {
        "run": run,
        "rows": int(len(d)),
        "full_rebuild_seconds": full_seconds,
        "refresh_2024_seconds": float(refresh.pop("seconds")),
        "working_set_before_mib": current_before,
        "working_set_after_full_mib": current_after,
        "working_set_after_refresh_mib": current_refresh,
        "peak_before_mib": peak_before,
        "peak_after_full_mib": peak_after,
        "peak_after_refresh_mib": peak_refresh,
        "encoded_array_mib": sum(x.nbytes for x in encoded.values()) / 2**20,
        "refresh_array_mib": sum(x.nbytes for x in refresh.values()) / 2**20,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def q(values: list[float], p: float) -> float:
    return float(np.quantile(np.asarray(values, float), p))


def parent(repetitions: int) -> None:
    temp_paths = []
    runs = []
    for run in range(1, repetitions + 1):
        path = OUT.with_name(f".v20_encoder_benchmark_run_{run}.json")
        temp_paths.append(path)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(run), "--worker-output", str(path)], check=True)
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    for path in temp_paths:
        path.unlink(missing_ok=True)

    full = [r["full_rebuild_seconds"] for r in runs]
    refresh = [r["refresh_2024_seconds"] for r in runs]
    deltas = [r["working_set_after_full_mib"] - r["working_set_before_mib"] for r in runs]
    machine = {
        "os": platform.platform(),
        "processor": platform.processor() or platform.uname().processor,
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    payload = {
        "scope": "observed implementation benchmark; not a portable or asymptotic performance claim",
        "repetitions": repetitions,
        "fresh_process_per_repetition": True,
        "rows": runs[0]["rows"],
        "hierarchical_blocks": 15,
        "machine": machine,
        "summary": {
            "full_rebuild_seconds_median": statistics.median(full),
            "full_rebuild_seconds_q1": q(full, .25),
            "full_rebuild_seconds_q3": q(full, .75),
            "refresh_2024_seconds_median": statistics.median(refresh),
            "refresh_2024_seconds_q1": q(refresh, .25),
            "refresh_2024_seconds_q3": q(refresh, .75),
            "incremental_working_set_mib_median": statistics.median(deltas),
            "encoded_array_mib": runs[0]["encoded_array_mib"],
            "refresh_array_mib": runs[0]["refresh_array_mib"],
        },
        "runs": runs,
        "definitions": {
            "MiB": "2^20 bytes",
            "working_set": "physical memory resident in the benchmark process at the observation point",
            "refresh_2024": "statistics, solve, and assignment for the three encoders with cached trees; excludes data loading and tree construction",
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    CSV.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(runs).to_csv(CSV, index=False, encoding="utf-8-sig")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker, args.worker_output)
    else:
        parent(args.repetitions)
