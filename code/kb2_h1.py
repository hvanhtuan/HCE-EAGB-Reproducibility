"""KB2 (kiểm định H1) và KB3 (độ nhạy cỡ mẫu, thay đổi lược đồ) trên Tầng 1c.
Chế độ 'pilot': huấn luyện 2012–2018, đánh giá trên 2019–2020 (thí điểm, chọn siêu tham số).
Chế độ 'test' : đọc tệp tiền đăng ký đã khóa, huấn luyện lại với cấu hình đã khóa, mở tập kiểm định 2021–2023 đúng một lần."""
import sys, json, time, hashlib, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from catboost import CatBoostRegressor, Pool
from nfip_lib import *
from metrics import tweedie_dev

MODE = sys.argv[1]
XI_EVAL = 1.5  # tham số phương sai dùng cho độ đo chính, cố định trước thí điểm
d = load(); T, node = build_tree(d)
TR = (d.year >= 2012) & (d.year <= 2018); VA = (d.year >= 2019) & (d.year <= 2020); TE = (d.year >= 2021) & (d.year <= 2023)
EVAL = VA if MODE == "pilot" else TE
if MODE == "test":
    pre = json.load(open("../results/prereg_H1.json"))
out = {"mode": MODE, "n_train": int(TR.sum()), "n_val": int(VA.sum()), "n_eval": int(EVAL.sum())}
years_all = list(range(2010, 2025))
Tflat, nodeflat = build_tree(d, levels=("tract",))
Tpar, nodepar = build_tree(d, levels=("state", "county"))


def encoders(kmul, method="simul", tree="full"):
    L = {"full": 3, "flat": 1, "par": 2}[tree]
    TT, nd = {"full": (T, node), "flat": (Tflat, nodeflat), "par": (Tpar, nodepar)}[tree]
    return temporal_encoder(d, TT, nd, years_all, [0] + [kmul] * L, k0=1.0, method=method)


def enc_dev(z, mask):
    return float(np.sum(d.E[mask] * tweedie_dev(d.rate[mask], np.maximum(z[mask], 1e-3), XI_EVAL)) / d.E[mask].sum())


E, S, rate = d.E.values, d.S.values, d.rate.values
preds = {}; meta = {}
# ---------- chọn k (thí điểm) hoặc đọc k đã khóa
KGRID = [100, 300, 1000, 3000, 10000, 30000, 100000]
if MODE == "pilot":
    ksel = {}
    for name, (meth, tr) in {"simul": ("simul", "full"), "seq": ("seq", "full"), "flat": ("simul", "flat"), "par": ("simul", "par")}.items():
        devs = {}
        for km in KGRID:
            z, _ = encoders(km, meth, tr); devs[km] = enc_dev(z, TR)  # chọn k chỉ trên phần huấn luyện (nhãn ngoài mẫu)
        ksel[name] = min(devs, key=devs.get); meta["kgrid_" + name] = devs
    meta["k_selected"] = ksel
else:
    ksel = pre["k_selected"]
Z = {}
for name, (meth, tr) in {"simul": ("simul", "full"), "seq": ("seq", "full"), "flat": ("simul", "flat"), "par": ("simul", "par")}.items():
    Z[name], logs = encoders(ksel[name], meth, tr)
    if name == "simul":
        meta["leak_log"] = logs
meta["I2_violations"] = int(sum(l.get("overlap_block", 0) for l in meta["leak_log"]))
meta["I2_max_src_avail_ok"] = all((not l["valid"]) or l["max_src_label_avail"] <= l["year"] for l in meta["leak_log"])

# ---------- boosting (cùng siêu tham số, cùng dừng sớm trên VA; ở chế độ test, số vòng lấy từ thí điểm)
XIS = [1.3, 1.5, 1.7]


def gbm(Xall, name, xi=None, obj="tweedie"):
    Xtr, Xva = Xall[TR.values], align_cats(Xall[TR.values], Xall[VA.values].copy())
    Xev = align_cats(Xall[TR.values], Xall[EVAL.values].copy())
    extra = {"tweedie_variance_power": xi} if obj == "tweedie" else {}
    if MODE == "pilot":
        m, sec = fit_lgb(Xtr, rate[TR], E[TR], Xva, rate[VA], E[VA], obj, extra)
        meta.setdefault("rounds", {})[name] = m.best_iteration
    else:
        p = dict(LGB_BASE, objective=obj, **extra); t0 = time.time()
        m = lgb.train(p, lgb.Dataset(Xtr, rate[TR], weight=E[TR]), pre["rounds"][name]); sec = time.time() - t0
    t0 = time.time(); pr = m.predict(Xev); meta.setdefault("infer_sec", {})[name] = time.time() - t0
    meta.setdefault("train_sec", {})[name] = sec
    return pr, m


if MODE == "pilot":
    xi_dev = {}
    for xi in XIS:
        pr, _ = gbm(features(d, Z["simul"]), f"EAGB_xi{xi}", xi)
        xi_dev[xi] = float(np.sum(E[EVAL] * tweedie_dev(rate[EVAL], pr, XI_EVAL)) / E[EVAL].sum())
    xi_sel = min(xi_dev, key=xi_dev.get); meta["xi_pilot"] = xi_dev; meta["xi_selected"] = xi_sel
else:
    xi_sel = pre["xi_selected"]
    for xi in XIS:  # giữ khóa tên số vòng
        pass
feat_models = {
    "EAGB (HCE đồng thời)": features(d, Z["simul"]),
    "GBDT + HCE tuần tự (GT2)": features(d, Z["seq"]),
    "GBDT + mã hóa phẳng co ngót (H1c)": features(d, Z["flat"]),
    "GBDT + mã hóa cấp quận (lùi về nút cha)": features(d, Z["par"]),
    "GBDT phân loại nội tại (bang/quận/tract)": features(d, None, geo_cat=True),
    "GBDT không biến địa lý": features(d, None),
}
models = {}
for name, X in feat_models.items():
    preds[name], models[name] = gbm(X, name, xi_sel)
# EAGB cấu hình Poisson–Gamma
Xs = features(d, Z["simul"])
prf, mf = gbm(Xs, "EAGB-PG tần suất", obj="poisson") if False else (None, None)


def fit_pg():
    Xtr, Xva = Xs[TR.values], align_cats(Xs[TR.values], Xs[VA.values].copy()); Xev = align_cats(Xs[TR.values], Xs[EVAL.values].copy())
    N = d.N.values; sev_tr = TR.values & (N > 0); sev_va = VA.values & (N > 0)
    res = {}
    for nm, obj, ytr, wtr, yva, wva, mtr, mva in [("freq", "poisson", N / E, E, N / E, E, TR.values, VA.values),
                                                  ("sev", "gamma", np.where(N > 0, S / np.maximum(N, 1), 1), N, np.where(N > 0, S / np.maximum(N, 1), 1), N, sev_tr, sev_va)]:
        key = "EAGB-PG " + nm
        if MODE == "pilot":
            m, sec = fit_lgb(Xs[mtr], ytr[mtr], wtr[mtr], align_cats(Xs[mtr], Xs[mva].copy()), yva[mva], wva[mva], obj)
            meta.setdefault("rounds", {})[key] = m.best_iteration
        else:
            t0 = time.time(); m = lgb.train(dict(LGB_BASE, objective=obj), lgb.Dataset(Xs[mtr], ytr[mtr], weight=wtr[mtr]), pre["rounds"][key]); sec = time.time() - t0
        meta.setdefault("train_sec", {})[key] = sec; res[nm] = m
    return res
pg = fit_pg()
Xev = align_cats(Xs[TR.values], Xs[EVAL.values].copy())
preds["EAGB-PG (Poisson–Gamma)"] = pg["freq"].predict(Xev) * pg["sev"].predict(Xev)
models["EAGB-PG freq"], models["EAGB-PG sev"] = pg["freq"], pg["sev"]

# HCE thuần (tín nhiệm phân cấp, không bộ học) — chuẩn hóa về mức danh mục huấn luyện
preds["Tín nhiệm phân cấp thuần (HCE)"] = np.maximum(Z["simul"][EVAL.values], 1e-3)

# GLM Tweedie (đối chứng chính) và GLM có spline (xấp xỉ GAM)
def glm_design(spline=False):
    cats = CAT + ["state", "county"]
    oh = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False, dtype=np.float64).fit(d.loc[TR, cats])
    sc = StandardScaler().fit(d.loc[TR, NUM])
    if spline:
        spl = SplineTransformer(n_knots=6, degree=3).fit(sc.transform(d.loc[TR, NUM]))
    def mk(m):
        parts = [oh.transform(d.loc[m, cats]), sc.transform(d.loc[m, NUM])]
        if spline:
            parts.append(spl.transform(sc.transform(d.loc[m, NUM])))
        return np.hstack(parts)
    return mk(TR), mk(EVAL)
for nm, sp in [("GLM Tweedie (đối chứng chính)", False), ("GLM Tweedie + spline (xấp xỉ GAM)", True)]:
    A, B = glm_design(sp); t0 = time.time()
    g = TweedieRegressor(power=xi_sel, link="log", alpha=1e-4, solver="newton-cholesky", max_iter=100, tol=1e-8).fit(A, rate[TR], sample_weight=E[TR])
    meta.setdefault("train_sec", {})[nm] = time.time() - t0; meta.setdefault("glm_iter", {})[nm] = int(g.n_iter_); preds[nm] = g.predict(B)
    del A, B
# CatBoost (thống kê mục tiêu có thứ tự theo thời gian)
cols = CAT + NUM + ["state", "county", "tract"]; cats = CAT + ["state", "county", "tract"]
ordtr = np.argsort(d.year.values[TR.values], kind="stable")
dtr = d[TR].iloc[ordtr]
t0 = time.time()
cb = CatBoostRegressor(loss_function=f"Tweedie:variance_power={xi_sel}", iterations=(1500 if MODE == "pilot" else pre["rounds"]["CatBoost"]), learning_rate=0.05, depth=6,
                       random_seed=SEED, verbose=0, thread_count=2, has_time=True, **({"od_type": "Iter", "od_wait": 150} if MODE == "pilot" else {}))
if MODE == "pilot":
    cb.fit(Pool(dtr[cols], dtr.rate, cat_features=cats, weight=dtr.E), eval_set=Pool(d.loc[VA, cols], d.rate[VA], cat_features=cats, weight=d.E[VA]))
    meta.setdefault("rounds", {})["CatBoost"] = int(cb.get_best_iteration() + 1)
else:
    cb.fit(Pool(dtr[cols], dtr.rate, cat_features=cats, weight=dtr.E))
meta.setdefault("train_sec", {})["CatBoost"] = time.time() - t0
preds["CatBoost OTS (có thứ tự thời gian)"] = cb.predict(d.loc[EVAL, cols])
preds["Mô hình rỗng"] = np.full(EVAL.sum(), S[TR].sum() / E[TR].sum())

# ---------- đánh giá
ev = {}
Ee, Se, cl = E[EVAL.values], S[EVAL.values], d.county.values[EVAL.values]
for nm, p in preds.items():
    ev[nm] = evaluate(np.maximum(p, 1e-6), Ee, Se, XI_EVAL)
    ev[nm]["OE_by_year"] = {int(y): float(Se[d.year.values[EVAL.values] == y].sum() / (p * Ee)[d.year.values[EVAL.values] == y].sum()) for y in np.unique(d.year.values[EVAL.values])}
if MODE == "pilot":
    meta["level_factor"] = {nm: float(Se.sum() / (p * Ee).sum()) for nm, p in preds.items()}
    LF = meta["level_factor"]
else:
    LF = pre["level_factor"]
ev_adj = {nm: evaluate(np.maximum(p * LF[nm], 1e-6), Ee, Se, XI_EVAL) for nm, p in preds.items()}
ref = "GLM Tweedie (đối chứng chính)"
devs = {nm: tweedie_dev(Se / Ee, np.maximum(p, 1e-6), XI_EVAL) for nm, p in preds.items()}
boot = {}
for nm in preds:
    for base in [ref, "GBDT + mã hóa phẳng co ngót (H1c)", "GBDT + HCE tuần tự (GT2)", "GBDT phân loại nội tại (bang/quận/tract)"]:
        if nm == base: continue
        bs, pt = cluster_boot_rel(devs[nm], devs[base], Ee, cl, B=2000)
        boot[f"{nm} || {base}"] = dict(rel=pt, lo95_2s=float(np.quantile(bs, .025)), hi95_2s=float(np.quantile(bs, .975)),
                                     hi95_1s=float(np.quantile(bs, .95)), lo95_1s=float(np.quantile(bs, .05)))
devs_adj = {nm: tweedie_dev(Se / Ee, np.maximum(p * LF[nm], 1e-6), XI_EVAL) for nm, p in preds.items()}
boot_adj = {}
for nm in preds:
    for base in [ref, "GBDT + mã hóa phẳng co ngót (H1c)", "GBDT + HCE tuần tự (GT2)"]:
        if nm == base: continue
        bs, pt = cluster_boot_rel(devs_adj[nm], devs_adj[base], Ee, cl, B=2000)
        boot_adj[f"{nm} || {base}"] = dict(rel=pt, lo95_2s=float(np.quantile(bs, .025)), hi95_2s=float(np.quantile(bs, .975)), hi95_1s=float(np.quantile(bs, .95)), lo95_1s=float(np.quantile(bs, .05)))
out.update(meta=meta, eval=ev, boot=boot, eval_level_adjusted=ev_adj, boot_level_adjusted=boot_adj)
json.dump(out, open(f"../results/kb2_{MODE}.json", "w"), indent=1, ensure_ascii=False, default=float)
np.savez_compressed(f"../results/kb2_{MODE}_preds.npz", **{k.encode('ascii', 'ignore').decode()[:40] + str(i): v for i, (k, v) in enumerate(preds.items())})
import pickle; pickle.dump({"Z": Z, "xi": xi_sel, "k": ksel}, open(f"../results/kb2_{MODE}_enc.pkl", "wb"))
for nm, m in models.items():
    if hasattr(m, "save_model"): m.save_model(f"../results/models/{MODE}_{abs(hash(nm)) % 10**8}.txt")
json.dump({nm: f"{MODE}_{abs(hash(nm)) % 10**8}.txt" for nm in models}, open(f"../results/models/{MODE}_index.json", "w"), ensure_ascii=False)
for nm, r in ev.items():
    print(f"{nm:45s} dev={r['tweedie_dev']:.4f} gini={r['gini']:.3f} OE={r['OE']:.3f} slope={r['slope']:.3f} maxdec={r['max_dec_dev']:.3f}")
print(json.dumps({k: v for k, v in meta.items() if k not in ('leak_log',)}, indent=1, default=float)[:3000])
