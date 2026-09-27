"""Kiểm chứng số TK1–TK7 (thiết kế ở Bảng 4.1 Chuyên đề Tiến sĩ 1). Kết quả ghi vào results/tk_results.json."""
import numpy as np, json, time
from fractions import Fraction as Fr
from hce import Tree, node_stats, hce_solve, hce_dense, hce_sequential, hce_objective, hce_gradient
from hshap import owen_from_table, shapley_from_table
from amoro import Problem, nsga2, nds

rng = np.random.default_rng(20260918)
R = {}


def rand_tree(L, branch=(2, 4)):
    paths = [()]
    for l in range(L):
        paths = [p + (f"{l}_{i}",) for p in paths for i in range(rng.integers(*branch))]
    return paths


# TK1: tính đúng HCE-SOLVE so với giải hệ đặc và số học hữu tỷ
tk1 = []
for L in range(2, 9):
    for rep in range(3):
        paths = rand_tree(L, (2, 3) if L > 5 else (2, 4)); T = Tree(paths)
        leaves = np.array([T.index[p] for p in paths])
        E = rng.gamma(0.5, 5, len(leaves)) * (rng.random(len(leaves)) > 0.2)
        S = E * rng.gamma(1.0, 1.0, len(leaves)) * (rng.random(len(leaves)) > 0.5)
        Ev, Sv = node_stats(T, leaves, E, S)
        k = np.r_[0, rng.uniform(0.1, 10, L)]; k0 = rng.choice([0.0, 1.0]); mu0 = 1.0
        if k0 == 0 and Ev.sum() == 0: continue
        th = hce_solve(T, Ev, Sv, k, k0, mu0); thd, H = hce_dense(T, Ev, Sv, k, k0, mu0)
        g = hce_gradient(T, Ev, Sv, th, k, k0, mu0)
        rec = dict(L=L, V=T.n, maxdiff_dense=float(np.abs(th - thd).max()),
                   rel_resid=float(np.linalg.norm(H @ th - (Sv + np.eye(T.n)[0] * k0 * mu0)) / np.linalg.norm(Sv + np.eye(T.n)[0] * k0 * mu0)),
                   grad_inf=float(np.abs(g).max()), cond=float(np.linalg.cond(H)))
        if T.n <= 400:  # đối chiếu số học hữu tỷ
            Evf = [Fr(x).limit_denominator(10**6) for x in Ev]; Svf = [Fr(x).limit_denominator(10**6) for x in Sv]
            kf = [Fr(x).limit_denominator(10**6) for x in k]
            the = hce_solve(T, Evf, Svf, kf, Fr(k0), Fr(1), exact=True)
            thf = hce_solve(T, np.array([float(x) for x in Evf]), np.array([float(x) for x in Svf]), np.array([float(x) for x in kf]), k0, 1.0)
            rec["maxerr_vs_rational"] = float(max(abs(float(a) - b) for a, b in zip(the, thf)))
            rec["max_bits"] = int(max(max(x.numerator.bit_length(), x.denominator.bit_length()) for x in the))
        tk1.append(rec)
R["TK1"] = tk1

# TK2: bậc chi phí — thời gian/nút khi |V| tăng
tk2 = []
for L, br in [(3, 10), (3, 20), (4, 12), (4, 18), (5, 12)]:
    paths = [()]
    for l in range(L):
        paths = [p + (i,) for p in paths for i in range(br)]
    T = Tree(paths); leaves = np.array([T.index[p] for p in paths])
    E = rng.gamma(1, 1, len(leaves)); S = E * rng.gamma(1, 1, len(leaves))
    Ev, Sv = node_stats(T, leaves, E, S); k = np.r_[0, np.ones(L)]
    t0 = time.perf_counter(); [hce_solve(T, Ev, Sv, k, 1, 1) for _ in range(3)]; dt = (time.perf_counter() - t0) / 3
    tk2.append(dict(V=T.n, sec=dt, us_per_node=dt / T.n * 1e6))
R["TK2"] = tk2

# TK3: trường hợp suy biến hai cấp, gốc cố định so với Bühlmann–Straub
tk3 = []
for rep in range(200):
    m = rng.integers(2, 30); paths = [(i,) for i in range(m)]; T = Tree(paths)
    E = rng.gamma(1, 3, m) + 1e-3; S = E * rng.gamma(1, 1, m); k1 = rng.uniform(0.01, 50); mu = rng.uniform(0.5, 2)
    Ev, Sv = node_stats(T, np.arange(1, m + 1), E, S)
    th = hce_solve(T, Ev, Sv, [0, k1], 0, mu, fixed_root=True)
    Z = E / (E + k1); bs = Z * S / E + (1 - Z) * mu
    tk3.append(float(np.abs(th[1:] - bs).max()))
R["TK3"] = dict(n=len(tk3), max_abs_diff=max(tk3))
# kiểm tra hữu tỷ một cấu hình
T = Tree([(0,), (1,), (2,)]); Ev = [Fr(0), Fr(2), Fr(3), Fr(5)]; Sv = [Fr(0), Fr(1), Fr(9), Fr(0)]
th = hce_solve(T, Ev, Sv, [0, Fr(4)], 0, Fr(3, 2), fixed_root=True, exact=True)
bs = [Ev[v] / (Ev[v] + 4) * Sv[v] / Ev[v] + 4 / (Ev[v] + 4) * Fr(3, 2) for v in (1, 2, 3)]
R["TK3"]["rational_equal"] = all(a == b for a, b in zip(th[1:], bs))

# TK4: phản ví dụ Mệnh đề 3.1 và họ cây độ sâu 2–6
T = Tree([("a", "b"), ("c",)]); ids = np.array([T.index[("a", "b")], T.index[("c",)]])
Ev, Sv = node_stats(T, ids, np.array([1., 1.]), np.array([0., 4.]))
ths = hce_sequential(T, Ev, Sv, [0, 1, 1], 1, 2); the = hce_solve(T, [Fr(x) for x in Ev], [Fr(x) for x in Sv], [0, 1, 1], 1, 2, exact=True)
Js = hce_objective(T, [Fr(x) for x in Ev], [Fr(x) for x in Sv], [Fr(x).limit_denominator() for x in ths], [0, 1, 1], 1, 2, 16)
Je = hce_objective(T, [Fr(x) for x in Ev], [Fr(x) for x in Sv], the, [0, 1, 1], 1, 2, 16)
R["TK4"] = dict(grad_a=float(hce_gradient(T, Ev, Sv, ths, [0, 1, 1], 1, 2)[1]), gap=str(Js - Je), theta_star=[str(x) for x in the])
fam = []
for L in range(2, 7):
    gaps = []
    for rep in range(30):
        paths = rand_tree(L, (2, 3)); T = Tree(paths); leaves = np.array([T.index[p] for p in paths])
        E = rng.gamma(0.3, 10, len(leaves)); S = E * rng.gamma(1, 1, len(leaves))
        Ev, Sv = node_stats(T, leaves, E, S); k = np.r_[0, rng.uniform(0.5, 5, L)]
        a = hce_objective(T, Ev, Sv, hce_sequential(T, Ev, Sv, k, 1, 1), k, 1, 1); b = hce_objective(T, Ev, Sv, hce_solve(T, Ev, Sv, k, 1, 1), k, 1, 1)
        gaps.append(a - b)
    fam.append(dict(L=L, min_gap=float(min(gaps)), median_gap=float(np.median(gaps)), share_positive=float(np.mean(np.array(gaps) > 1e-9))))
R["TK4"]["family"] = fam

# TK5: hiệu suất Owen trên trò chơi ngẫu nhiên p<=8
res5 = []
for rep in range(300):
    p = rng.integers(2, 9); nu = rng.normal(size=1 << p); nu[0] = 0
    perm = rng.permutation(p); cuts = sorted(rng.choice(np.arange(1, p), size=rng.integers(0, p - 1), replace=False)) if p > 1 else []
    groups = [list(x) for x in np.split(perm, cuts)]
    psi, Phi = owen_from_table(nu, p, groups)
    res5.append(max(max(abs(psi[g].sum() - Phi[i]) for i, g in enumerate(groups)), abs(Phi.sum() - nu[-1])))
R["TK5"] = dict(n=300, max_violation=float(max(res5)))

# TK6: cộng tính hai tầng trên thang log, thất bại trên thang gốc (kỳ vọng biên, lưới rời rạc)
res6 = []
for rep in range(100):
    p = 4; levels = [rng.integers(2, 4) for _ in range(p)]
    Xbg = np.stack([rng.integers(0, l, 200) for l in levels], 1); x = np.array([rng.integers(0, l) for l in levels])
    a = rng.normal(0, .5, (p, 4)); b = rng.normal(0, .5, (p, 4))
    lam = lambda X: np.exp(sum(a[j, X[:, j]] for j in range(p)) + 0.3 * a[0, X[:, 0]] * a[1, X[:, 1]])
    mm = lambda X: np.exp(sum(b[j, X[:, j]] for j in range(p)) - 0.2 * b[2, X[:, 2]] * b[3, X[:, 3]])
    def table(fn):
        out = np.zeros(1 << p)
        for m in range(1 << p):
            X = Xbg.copy(); keep = [(m >> j) & 1 for j in range(p)]
            for j in range(p):
                if keep[j]: X[:, j] = x[j]
            out[m] = fn(X).mean()
        return out - out[0]
    tl, tm = table(lambda X: np.log(lam(X))), table(lambda X: np.log(mm(X)))
    tp = table(lambda X: np.log(lam(X) * mm(X)))
    g = [[0, 1], [2, 3]]
    log_viol = np.abs(owen_from_table(tp, p, g)[0] - owen_from_table(tl, p, g)[0] - owen_from_table(tm, p, g)[0]).max()
    ol, om, op = table(lam), table(mm), table(lambda X: lam(X) * mm(X))
    raw_viol = np.abs(owen_from_table(op, p, g)[0] - owen_from_table(ol, p, g)[0] - owen_from_table(om, p, g)[0]).max()
    res6.append((log_viol, raw_viol))
res6 = np.array(res6)
R["TK6"] = dict(n=100, max_log_violation=float(res6[:, 0].max()), min_raw_violation=float(res6[:, 1].min()), median_raw_violation=float(np.median(res6[:, 1])))

# TK7: bài toán AMORO nhỏ (q<=4), lưới dày làm tập tham chiếu
tk7 = []
for rep in range(5):
    n, q = 300, 3; A = rng.integers(0, q, n); s = rng.gamma(2, 50, n); Pi0 = s * np.exp(rng.normal(0.1, 0.05, n))
    prob = Problem(s, Pi0, A, q, 0.2 * s, np.ones(n), 1.7, 2.0, Pi0, -0.3 * np.ones(q), 0.3 * np.ones(q), 0.35, 0.9)
    sU = s * 1.08; probU = Problem(s, Pi0, A, q, 0.2 * s, np.ones(n), 1.7, 2.0, Pi0, -0.3 * np.ones(q), 0.3 * np.ones(q), 0.35, 0.9, s_obj=sU)
    g1 = np.linspace(-0.3, 0.3, 31); G = np.array(np.meshgrid(g1, g1, g1)).reshape(3, -1).T
    FG, fg, _ = prob.evaluate(G); FGU, fgU, _ = probU.evaluate(G)
    ref = FG[fg][nds(FG[fg])[0]]
    beta0 = np.array([np.median(prob.lr0[A == l]) for l in range(q)])
    out = nsga2(prob, beta0, Npop=40, Tgen=250, seed=rep)
    F = out["F"]
    dominated_by_grid = int(sum(((ref <= f).all(1) & (ref < f).any(1)).any() for f in F))
    tk7.append(dict(front=len(F), dominated_by_grid=dominated_by_grid,
                    FU_subset_F=bool(np.all(fg[fgU])), F1U_ge_F1=bool(np.all(FGU[:, 0] >= FG[:, 0] - 1e-9)),
                    n_feas=int(fg.sum()), n_feasU=int(fgU.sum()), rejected=out["log"]["rejected"], generated=out["log"]["generated"]))
R["TK7"] = tk7
json.dump(R, open("../results/tk_results.json", "w"), indent=1, default=float)
print(json.dumps({k: (v if k not in ("TK1",) else v[:3]) for k, v in R.items()}, indent=1, default=float)[:4000])
print("TK1 max dense diff", max(r["maxdiff_dense"] for r in tk1), "max grad", max(r["grad_inf"] for r in tk1),
      "max rational err", max(r.get("maxerr_vs_rational", 0) for r in tk1), "max cond", max(r["cond"] for r in tk1), "n", len(tk1))
