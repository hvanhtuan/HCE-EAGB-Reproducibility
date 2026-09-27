"""Bổ sung theo REVIEWEDv3 §10 (thăm dò, không tiền đăng ký). Chế độ:
  tree : dựng cây riêng cho từng năm chỉ từ bản ghi nguồn (năm + LAG <= t); bản ghi khối t lùi về tổ tiên gần nhất có mặt. So sánh z với cây dựng từ mọi ô.
  tune : điều chỉnh ngang ngân sách (cùng lưới num_leaves x min_data_in_leaf, dừng sớm trên 2019-2020) cho 4 mô hình GBDT; chọn theo độ lệch thí điểm; huấn luyện lại và đánh giá 2021-2023 (một lần).
  counts: số vụ ghép/không ghép theo bang và năm"""
import sys, os, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from nfip_lib import *
from hce import Tree, node_stats, hce_solve, hce_sequential
from metrics import tweedie_dev
MODE = sys.argv[1]; XI_EVAL = 1.5
pre = json.load(open("../results/prereg_H1.json")); ksel = pre["k_selected"]; xi = pre["xi_selected"]
d = load(); years_all = list(range(2010, 2025))
TR = (d.year >= 2012) & (d.year <= 2018); VA = (d.year >= 2019) & (d.year <= 2020); TE = (d.year >= 2021) & (d.year <= 2023)
E, S, rate = d.E.values, d.S.values, d.rate.values
os.makedirs("../results/tune_cache", exist_ok=True)

if MODE == "tree":
    out = {}
    for name, (meth, levels, k) in {"simul": ("simul", ("state", "county", "tract"), ksel["simul"]), "seq": ("seq", ("state", "county", "tract"), ksel["seq"]),
                                    "flat": ("simul", ("tract",), ksel["flat"])}.items():
        Tf, nf = build_tree(d, levels); L = len(levels)
        zfull, _ = temporal_encoder(d, Tf, nf, years_all, [0] + [k] * L, k0=1.0, method=meth)
        zt = np.full(len(d), np.nan); info = []
        for t in years_all:
            src = (d.year.values + LAG) <= t; blk = d.year.values == t
            if not src.any(): continue
            Ts = Tree(sorted(d.loc[src, list(levels)].drop_duplicates().itertuples(index=False, name=None)))
            nsrc = np.array([Ts.index[p] for p in d.loc[src, list(levels)].itertuples(index=False, name=None)])
            Ev, Sv = node_stats(Ts, nsrc, E[src], S[src]); mu0 = Sv.sum() / Ev.sum()
            th = (hce_solve if meth == "simul" else hce_sequential)(Ts, Ev, Sv, [0] + [k] * L, 1.0, mu0) if meth == "simul" else hce_sequential(Ts, Ev, Sv, [0] + [k] * L, 1.0, mu0)
            bp = d.loc[blk, list(levels)].drop_duplicates()
            mp = {p: Ts.node_of(p) for p in bp.itertuples(index=False, name=None)}
            nb = np.array([mp[p] for p in d.loc[blk, list(levels)].itertuples(index=False, name=None)])
            zt[blk] = th[nb]
            info.append(dict(year=t, nodes_src=int(Ts.n), nodes_full=int(Tf.n), frac_blk_unseen_leaf=float(np.mean(Ts.depth[nb] < L))))
        m = ~np.isnan(zfull)
        out[name] = dict(max_abs_diff=float(np.nanmax(np.abs(zfull[m] - zt[m]))), n=int(m.sum()), years=info)
        print(name, out[name]["max_abs_diff"], flush=True)
    json.dump(out, open("../results/v4_tree.json", "w"), indent=1)

elif MODE == "counts":
    c = pd.read_parquet("../data/claims_by_cell_event.parquet"); print(c.columns.tolist(), len(c))
    print(d[["N", "S"]].describe())

elif MODE == "tune":
    Tt, node = build_tree(d); Tflat, nodeflat = build_tree(d, levels=("tract",))
    Zs = temporal_encoder(d, Tt, node, years_all, [0] + [ksel["simul"]] * 3, k0=1.0)[0]
    Zf = temporal_encoder(d, Tflat, nodeflat, years_all, [0, ksel["flat"]], k0=1.0)[0]
    FE = {"EAGB": features(d, Zs), "Phang": features(d, Zf), "KhongDiaLy": features(d, None), "DiaLyPhanLoai": features(d, None, geo_cat=True)}
    GRID = [(31, 500), (15, 500), (63, 500), (31, 100), (31, 2000)]
    which = sys.argv[2].split(",") if len(sys.argv) > 2 else list(FE)
    Ee_va = E[VA.values]
    for nm in which:
        X = FE[nm]; Xtr = X[TR.values]; Xva = align_cats(Xtr, X[VA.values].copy())
        for nl, md in GRID:
            f = f"../results/tune_cache/{nm}_{nl}_{md}.json"
            if os.path.exists(f): continue
            t0 = time.time()
            m, _ = fit_lgb(Xtr, rate[TR], E[TR], Xva, rate[VA], E[VA], "tweedie", dict(tweedie_variance_power=xi, num_leaves=nl, min_data_in_leaf=md), rounds=4000)
            p = m.predict(Xva, num_iteration=m.best_iteration)
            dv = float(np.sum(Ee_va * tweedie_dev(rate[VA], np.maximum(p, 1e-6), XI_EVAL)) / Ee_va.sum())
            json.dump(dict(model=nm, num_leaves=nl, min_data=md, best_iter=int(m.best_iteration), pilot_dev=dv, sec=time.time() - t0), open(f, "w"))
            print(nm, nl, md, m.best_iteration, dv, time.time() - t0, flush=True)

if MODE == "lag":
    from metrics import tweedie_dev as _td
    LAGV = int(sys.argv[2]); Tt, node = build_tree(d); Tflat, nodeflat = build_tree(d, levels=("tract",))
    Zs = temporal_encoder(d, Tt, node, years_all, [0] + [ksel["simul"]] * 3, k0=1.0, lag=LAGV)[0]
    Zq = temporal_encoder(d, Tt, node, years_all, [0] + [ksel["seq"]] * 3, k0=1.0, method="seq", lag=LAGV)[0]
    Zf = temporal_encoder(d, Tflat, nodeflat, years_all, [0, ksel["flat"]], k0=1.0, lag=LAGV)[0]
    Ee, Se, cl = E[TE.values], S[TE.values], d.county.values[TE.values]
    dv = lambda p: tweedie_dev(Se / Ee, np.maximum(p, 1e-6), XI_EVAL)
    P = {}
    for nm, Z, key in [("simul", Zs, "EAGB (HCE đồng thời)"), ("seq", Zq, "GBDT + HCE tuần tự (GT2)"), ("flat", Zf, "GBDT + mã hóa phẳng co ngót (H1c)")]:
        X = features(d, Z); Xtr = X[TR.values]; Xev = align_cats(Xtr, X[TE.values].copy())
        m = lgb.train(dict(LGB_BASE, objective="tweedie", tweedie_variance_power=xi), lgb.Dataset(Xtr, rate[TR], weight=E[TR]), pre["rounds"][key]); P[nm] = m.predict(Xev)
    cats = CAT + ["state", "county"]
    oh = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False, dtype=np.float64).fit(d.loc[TR, cats]); sc = StandardScaler().fit(d.loc[TR, NUM])
    mk = lambda m: np.hstack([oh.transform(d.loc[m, cats]), sc.transform(d.loc[m, NUM])])
    P["glm"] = TweedieRegressor(power=xi, link="log", alpha=1e-4, solver="newton-cholesky", max_iter=100, tol=1e-8).fit(mk(TR), rate[TR], sample_weight=E[TR]).predict(mk(TE))
    D = {k: dv(v) for k, v in P.items()}; out = dict(lag=LAGV, dev={k: float(np.sum(Ee * v) / Ee.sum()) for k, v in D.items()})
    def boot(a, b):
        bs, pt = cluster_boot_rel(a, b, Ee, cl, B=2000); return dict(rel=pt, lo95=float(np.quantile(bs, .025)), hi95=float(np.quantile(bs, .975)), hi95_1s=float(np.quantile(bs, .95)))
    out["H1c"] = boot(D["simul"], D["flat"]); out["GT2"] = boot(D["simul"], D["seq"]); out["H1a"] = boot(D["simul"], D["glm"])
    print(json.dumps(out, indent=1)); json.dump(out, open(f"../results/v4_lag{LAGV}.json", "w"), indent=1)
