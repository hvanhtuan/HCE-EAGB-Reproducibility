"""Exploratory floating-point stability stress test for HCE-SOLVE.

This post-review diagnostic is not part of the preregistered predictive tests.
It varies tree depth, exposure imbalance and penalty scale, then compares the
two-pass solution with a dense linear solve and reports residual/backward error.
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np

from hce import Tree, hce_dense, hce_solve, node_stats

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "supplementary" / "numerical_stability.json"
SEED = 20260927


def full_binary_paths(depth: int) -> list[tuple[str, ...]]:
    paths = [()]
    for level in range(depth):
        paths = [path + (f"L{level}_{side}",) for path in paths for side in (0, 1)]
    return paths


def rhs(tree: Tree, Sv: np.ndarray, k0: float, mu0: float) -> np.ndarray:
    value = np.asarray(Sv, dtype=float).copy()
    value[0] += k0 * mu0
    return value


def rel_error(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), np.finfo(float).tiny))


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for depth in (2, 4, 6, 8):
        paths = full_binary_paths(depth)
        tree = Tree(paths)
        leaves = np.array([tree.index[path] for path in paths])
        for exposure_ratio in (1e2, 1e6, 1e10):
            for penalty_scale in (1e-8, 1e-4, 1.0, 1e4, 1e8):
                log_e = rng.uniform(0.0, np.log(exposure_ratio), len(leaves))
                exposure = np.exp(log_e)
                exposure[rng.random(len(leaves)) < 0.20] = 0.0
                if not np.any(exposure > 0):
                    exposure[0] = 1.0
                rate = np.exp(rng.normal(0.0, 2.0, len(leaves)))
                loss = exposure * rate
                Ev, Sv = node_stats(tree, leaves, exposure, loss)
                k = np.zeros(depth + 1)
                k[1:] = penalty_scale * np.logspace(-2, 2, depth)
                k0 = max(penalty_scale, 1e-12)
                mu0 = 1.0

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dense, H = hce_dense(tree, Ev, Sv, k, k0, mu0)
                two_pass = np.asarray(hce_solve(tree, Ev, Sv, k, k0, mu0), dtype=float)
                b = rhs(tree, Sv, k0, mu0)
                residual = H @ two_pass - b
                denom = np.linalg.norm(H, 2) * np.linalg.norm(two_pass) + np.linalg.norm(b)
                backward = float(np.linalg.norm(residual) / max(denom, np.finfo(float).tiny))
                condition = float(np.linalg.cond(H, 2))
                rows.append({
                    "depth": depth,
                    "nodes": int(tree.n),
                    "exposure_ratio_design": exposure_ratio,
                    "penalty_scale": penalty_scale,
                    "condition_2": condition,
                    "forward_error_vs_dense": rel_error(two_pass, dense),
                    "relative_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(b), np.finfo(float).tiny)),
                    "normwise_backward_error": backward,
                    "finite": bool(np.isfinite(two_pass).all() and np.isfinite(dense).all()),
                })

    finite_rows = [row for row in rows if row["finite"]]
    summary = {
        "analysis_status": "post-review exploratory numerical diagnostic",
        "seed": SEED,
        "n_configurations": len(rows),
        "n_nonfinite": len(rows) - len(finite_rows),
        "design": {
            "depths": [2, 4, 6, 8],
            "maximum_nodes": max(row["nodes"] for row in rows),
            "exposure_ratio_design": [1e2, 1e6, 1e10],
            "penalty_scales": [1e-8, 1e-4, 1.0, 1e4, 1e8],
            "zero_exposure_leaf_probability": 0.20,
        },
        "condition_2_max": max(row["condition_2"] for row in finite_rows),
        "forward_error_max": max(row["forward_error_vs_dense"] for row in finite_rows),
        "relative_residual_max": max(row["relative_residual"] for row in finite_rows),
        "backward_error_max": max(row["normwise_backward_error"] for row in finite_rows),
        "forward_error_median": float(np.median([row["forward_error_vs_dense"] for row in finite_rows])),
        "backward_error_median": float(np.median([row["normwise_backward_error"] for row in finite_rows])),
        "rows": rows,
        "interpretation": (
            "Residual and backward error assess numerical solution of each generated linear system; "
            "forward error may grow with condition number. This finite stress test is not a universal "
            "stability proof and is not part of the preregistered predictive evaluation."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
