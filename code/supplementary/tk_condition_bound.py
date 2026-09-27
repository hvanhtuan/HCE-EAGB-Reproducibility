"""TK-bo sung (khong thuoc bo TK1-TK7 tien dang ky cua CDTS1): kiem chung so cho Menh de moi
(chan can duoi/tren so dieu kien cua ma tran Hessian H trong HCE-SOLVE, xem 04_ch4_part1.md).
Day la phan tich THAM DO sau khi da xem ket qua TK1 (khong tien dang ky) — duoc bao cao trung
thuc la nhu vay trong ban thao, khong duoc gan nhan "kiem chung xac nhan".

Dung lai chinh xac tap 21 cay ngau nhien cua TK1 (cung rng seed, cung vong lap sinh cay trong
tk_tests.py) de doi chieu true eigenvalues (np.linalg.eigvalsh) voi hai can duoi (LONG, CHAT) va
mot can tren (Gershgorin) cho lambda_min(H), lambda_max(H).
"""
import json
import heapq
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hce import Tree, node_stats, hce_dense


def nearest_ground(T, k, k0, Ev):
    """Cho moi nut v: (R*_v = dien tro toi nut neo gan nhat, g_v = trong so neo tai nut do)."""
    n = T.n
    edge_w = np.zeros(n)
    for v in range(1, n):
        edge_w[v] = 1.0 / k[T.depth[v]]
    grounded = []
    if k0 > 0:
        grounded.append((0, k0))
    for v in range(n):
        if Ev[v] > 0:
            grounded.append((v, Ev[v]))
    dist = np.full(n, np.inf)
    gval = np.zeros(n)
    heap = []
    seen_src = set()
    for gnode, gw in grounded:
        if gnode not in seen_src:
            dist[gnode] = 0.0
            gval[gnode] = gw
            heap.append((0.0, gnode))
            seen_src.add(gnode)
    heapq.heapify(heap)
    adj = [[] for _ in range(n)]
    for v in range(1, n):
        p = T.parent[v]
        w = edge_w[v]
        adj[v].append((p, w))
        adj[p].append((v, w))
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                gval[v] = gval[u]
                heapq.heappush(heap, (nd, v))
    return dist, gval


def gershgorin_lam_max(T, k, k0, Ev):
    n = T.n
    diagH = Ev.copy()
    diagH[0] += k0
    for v in range(1, n):
        kv = k[T.depth[v]]
        diagH[v] += kv
        diagH[T.parent[v]] += kv
    offabs = diagH.copy()
    offabs[0] -= (Ev[0] + k0)
    for v in range(1, n):
        offabs[v] -= Ev[v]
    gersh = diagH + offabs
    return float(gersh.max())


def main():
    rng = np.random.default_rng(20260918)

    def rand_tree(L, branch=(2, 4)):
        paths = [()]
        for l in range(L):
            paths = [p + (f"{l}_{i}",) for p in paths for i in range(rng.integers(*branch))]
        return paths

    rows = []
    for L in range(2, 9):
        for rep in range(3):
            paths = rand_tree(L, (2, 3) if L > 5 else (2, 4))
            T = Tree(paths)
            leaves = np.array([T.index[p] for p in paths])
            E = rng.gamma(0.5, 5, len(leaves)) * (rng.random(len(leaves)) > 0.2)
            S = E * rng.gamma(1.0, 1.0, len(leaves)) * (rng.random(len(leaves)) > 0.5)
            Ev, Sv = node_stats(T, leaves, E, S)
            k = np.r_[0, rng.uniform(0.1, 10, L)]
            k0 = rng.choice([0.0, 1.0])
            mu0 = 1.0
            if k0 == 0 and Ev.sum() == 0:
                continue
            thd, H = hce_dense(T, Ev, Sv, k, k0, mu0)
            eigs = np.linalg.eigvalsh(H)
            lam_min_actual = float(eigs.min())
            lam_max_actual = float(eigs.max())
            n = T.n
            Rstar, gval = nearest_ground(T, k, k0, Ev)
            Rmax = float(Rstar.max())
            Gmin = float(gval[gval > 0].min())
            lam_min_loose = 1.0 / (2 * n * (Rmax + 1.0 / Gmin))
            sumR = float(Rstar.sum())
            sumInvG = float(np.sum(1.0 / gval))
            lam_min_tight = 1.0 / (2 * (sumR + sumInvG))
            lam_max_bound = gershgorin_lam_max(T, k, k0, Ev)
            rows.append(dict(
                n=int(n), lam_min_actual=lam_min_actual, lam_min_loose=lam_min_loose,
                lam_min_tight=lam_min_tight, lam_max_actual=lam_max_actual,
                lam_max_gershgorin=lam_max_bound,
                ok_loose=bool(lam_min_loose <= lam_min_actual + 1e-9),
                ok_tight=bool(lam_min_tight <= lam_min_actual + 1e-9),
                ok_gersh=bool(lam_max_actual <= lam_max_bound + 1e-9),
            ))

    fail_loose = sum(1 for r in rows if not r["ok_loose"])
    fail_tight = sum(1 for r in rows if not r["ok_tight"])
    fail_gersh = sum(1 for r in rows if not r["ok_gersh"])
    ratio_loose = [r["lam_min_actual"] / r["lam_min_loose"] for r in rows]
    ratio_tight = [r["lam_min_actual"] / r["lam_min_tight"] for r in rows]
    ratio_gersh = [r["lam_max_gershgorin"] / r["lam_max_actual"] for r in rows]
    improve = [r["lam_min_tight"] / r["lam_min_loose"] for r in rows]
    cond_actual = [r["lam_max_actual"] / r["lam_min_actual"] for r in rows]
    cond_bound = [r["lam_max_gershgorin"] / r["lam_min_tight"] for r in rows]

    summary = dict(
        n_configs=len(rows),
        violations_loose=fail_loose,
        violations_tight=fail_tight,
        violations_gershgorin=fail_gersh,
        looseness_loose_min=min(ratio_loose), looseness_loose_median=float(np.median(ratio_loose)), looseness_loose_max=max(ratio_loose),
        looseness_tight_min=min(ratio_tight), looseness_tight_median=float(np.median(ratio_tight)), looseness_tight_max=max(ratio_tight),
        gershgorin_ratio_min=min(ratio_gersh), gershgorin_ratio_median=float(np.median(ratio_gersh)), gershgorin_ratio_max=max(ratio_gersh),
        improve_tight_over_loose_min=min(improve), improve_tight_over_loose_median=float(np.median(improve)), improve_tight_over_loose_max=max(improve),
        cond_actual_max=max(cond_actual),
        cond_bound_max=max(cond_bound),
        rows=rows,
    )
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tk_condition_bound.json")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Vi phạm: LỎNG {fail_loose}/{len(rows)}, CHẶT {fail_tight}/{len(rows)}, Gershgorin {fail_gersh}/{len(rows)}")
    print(f"Độ lỏng CHẶT: min={summary['looseness_tight_min']:.2f} median={summary['looseness_tight_median']:.2f} max={summary['looseness_tight_max']:.1f}")
    print(f"Độ lỏng LỎNG: min={summary['looseness_loose_min']:.1f} median={summary['looseness_loose_median']:.1f} max={summary['looseness_loose_max']:.1f}")
    print(f"Gershgorin (bound/actual): min={summary['gershgorin_ratio_min']:.3f} median={summary['gershgorin_ratio_median']:.3f} max={summary['gershgorin_ratio_max']:.3f}")
    print(f"Cải thiện CHẶT/LỎNG: min={summary['improve_tight_over_loose_min']:.1f} median={summary['improve_tight_over_loose_median']:.1f} max={summary['improve_tight_over_loose_max']:.1f}")
    print("saved", out_path)


if __name__ == "__main__":
    main()
