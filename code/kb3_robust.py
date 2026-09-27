"""KB3 (độ nhạy cỡ mẫu, thay đổi lược đồ, không gian) và một phần KB9 (năm 2024, tái trọng số vùng lũ, năm có sự kiện).
Phân tích mô tả; dùng cấu hình đã khóa ở prereg_H1.json."""
import json, time, pickle, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse
from nfip_lib import *
from hce import Tree, node_stats, hce_solve
from metrics import tweedie_dev

pre = json.load(open("../results/prereg_H1.json")); XIE = 1.5
d = load(); T, node = build_tree(d)
xi = pre["xi_selected"]; kS, kF = pre["k_selected"]["simul"], pre["k_selected"]["flat"]
TR = ((d.year >= 2012) & (d.year <= 2018)).values; VA = ((d.year >= 2019) & (d.year <= 2020)).values
TE = ((d.year >= 2021) & (d.year <= 2023)).values; Y24 = (d.year == 2024).values
E, S, rate = d.E.values, d.S.values, d.rate.values
Tflat, nodeflat = build_tree(d, levels=("tract",))
years_all = list(range(2010, 2025))
res = {}


def dev(p, m, w=None):
    w = E[m] if w is None else w
    return float(np.sum(w * tweedie_dev(rate[m], np.maximum(p, 1e-6), XIE)) / w.sum())


def train_eval(keep, Zs, Zf, eval_masks, tag):
    """keep: mặt nạ ô được phép dùng (huấn luyện + nguồn); trả về độ lệch trên các tập đánh giá."""
    out = {}
    tr, va = TR & keep, VA
    for nm, X in [("EAGB", features(d, Zs)), ("Mã hóa phẳng", features(d, Zf)), ("GBDT phân loại nội tại", features(d, None, geo_cat=True))]:
        Xtr = X[tr]; Xva = align_cats(Xtr, X[va].copy())
        m, _ = fit_lgb(Xtr, rate[tr], E[tr], Xva, rate[va], E[va], "tweedie", {"tweedie_variance_power": xi})
        for en, em in eval_masks.items():
            out[(nm, en)] = m.predict(align_cats(Xtr, X[em].copy()))
    return out


# ---------- (a) cỡ mẫu: lấy mẫu con theo quận cho mọi năm <= 2020 (cả nguồn mã hóa lẫn huấn luyện)
counties = np.unique(d.county.values)
res["sample_size"] = []
for frac in [0.1, 0.25, 0.5, 1.0]:
    for rep in range(2 if frac < 1 else 1):
        g = np.random.default_rng(700 + rep + int(frac * 100))
        sel = set(g.choice(counties, int(round(frac * len(counties))), replace=False)) if frac < 1 else set(counties)
        keep = d.county.isin(sel).values | (d.year.values >= 2021)
        dd_keep = keep | TE
        # bộ mã hóa chỉ dùng nguồn trong keep
        def enc(TT, nd, k, L):
            z = np.full(len(d), np.nan)
            for t in years_all:
                src = ((d.year.values + LAG) <= t) & keep
                if not src.any(): continue
                Ev, Sv = node_stats(TT, nd[src], E[src], S[src]); th = hce_solve(TT, Ev, Sv, [0] + [k] * L, 1.0, Sv.sum() / Ev.sum())
                blk = d.year.values == t; z[blk] = th[nd[blk]]
            return z
        Zs, Zf = enc(T, node, kS, 3), enc(Tflat, nodeflat, kF, 1)
        o = train_eval(keep, Zs, Zf, {"test": TE}, f"f{frac}")
        rec = dict(frac=frac, rep=rep, n_train=int((TR & keep).sum()))
        for (nm, en), p in o.items():
            rec[nm] = dev(p, TE)
        res["sample_size"].append(rec); print(rec, flush=True)
# ---------- (b) thay đổi lược đồ: ở các năm kiểm định, 50% quận được "phân định lại" (mã tract mới, không có lịch sử)
g = np.random.default_rng(801); relab = set(g.choice(counties, len(counties) // 2, replace=False))
d2 = d.copy(); m = TE & d.county.isin(relab).values
d2.loc[m, "tract"] = d2.loc[m, "tract"] + "N"  # lá mới của phiên bản cây mới
T2, node2 = build_tree(d2)
Tf2, nodef2 = build_tree(d2, levels=("tract",))
def enc2(TT, nd, k, L):
    z = np.full(len(d2), np.nan)
    for t in years_all:
        src = (d2.year.values + LAG) <= t
        Ev, Sv = node_stats(TT, nd[src], E[src], S[src]); th = hce_solve(TT, Ev, Sv, [0] + [k] * L, 1.0, Sv.sum() / Ev.sum())
        blk = d2.year.values == t; z[blk] = th[nd[blk]]
    return z
Zs2, Zf2 = enc2(T2, node2, kS, 3), enc2(Tf2, nodef2, kF, 1)
sch = {}
for nm, X in [("EAGB", features(d2, Zs2)), ("Mã hóa phẳng", features(d2, Zf2)), ("GBDT phân loại nội tại", features(d2, None, geo_cat=True))]:
    Xtr = X[TR]; Xva = align_cats(Xtr, X[VA].copy())
    mdl, _ = fit_lgb(Xtr, rate[TR], E[TR], Xva, rate[VA], E[VA], "tweedie", {"tweedie_variance_power": xi})
    p = mdl.predict(align_cats(Xtr, X[TE].copy()))
    mm = m[TE]
    sch[nm] = dict(dev_relabelled=dev(p[mm], TE & m), dev_unchanged=dev(p[~mm], TE & ~m), dev_all=dev(p, TE),
                   gini_relabelled=gini_conc(p[mm], E[TE & m], S[TE & m]))
res["schema_change"] = dict(share_cells_relabelled=float(m[TE].mean()), models=sch)
print(res["schema_change"], flush=True)
# ---------- (c) theo bang, (d) năm 2024, (e) tái trọng số vùng lũ, (f) năm có sự kiện — cho EAGB và GLM Tweedie đã khóa
enc = pickle.load(open("../results/kb2_test_enc.pkl", "rb")); z = enc["Z"]["simul"]
idx = json.load(open("../results/models/test_index.json")); main = lgb.Booster(model_file="../results/models/" + idx["EAGB (HCE đồng thời)"])
X = features(d, z); Xtr = X[TR]
pE = main.predict(align_cats(Xtr, X.copy()))
cats = CAT + ["state", "county"]
oh = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False, dtype=np.float64).fit(d.loc[TR, cats]); sc = StandardScaler().fit(d.loc[TR, NUM])
glm = TweedieRegressor(power=xi, link="log", alpha=1e-4, solver="newton-cholesky", max_iter=100, tol=1e-8).fit(np.hstack([oh.transform(d.loc[TR, cats]), sc.transform(d.loc[TR, NUM])]), rate[TR], sample_weight=E[TR])
pG = np.concatenate([glm.predict(np.hstack([oh.transform(d.iloc[i:i + 400000][cats]), sc.transform(d.iloc[i:i + 400000][NUM])])) for i in range(0, len(d), 400000)])
def cmp(mask, w=None):
    ww = E[mask] if w is None else w
    dE, dG = dev(pE[mask], mask, ww), dev(pG[mask], mask, ww)
    return dict(dev_EAGB=dE, dev_GLM=dG, rel=(dE - dG) / dG, OE_EAGB=float(S[mask] @ (ww / E[mask]) / (pE[mask] * ww).sum()), OE_GLM=float(S[mask] @ (ww / E[mask]) / (pG[mask] * ww).sum()),
                gini_EAGB=gini_conc(pE[mask], ww, S[mask] * ww / E[mask]), gini_GLM=gini_conc(pG[mask], ww, S[mask] * ww / E[mask]), n=int(mask.sum()))
res["by_state_test"] = {s: cmp(TE & (d.state.values == s)) for s in sorted(d.state.unique())}
res["year_2024"] = cmp(Y24)
res["by_year"] = {int(y): cmp(d.year.values == y) for y in range(2019, 2025)}
w = E[TE] * np.where(np.isin(d.zone.values[TE], ["A", "V"]), 2.0, 1.0)
res["reweight_AV_x2_test"] = cmp(TE, w)
json.dump(res, open("../results/kb3_robust.json", "w"), indent=1, default=float, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k != "sample_size"}, indent=1, default=float, ensure_ascii=False)[:4000])
