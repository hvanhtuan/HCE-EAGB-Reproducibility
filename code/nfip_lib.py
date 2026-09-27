"""Thư viện dùng chung cho các kịch bản trên Tầng 1c (NFIP công khai)."""
import numpy as np, pandas as pd, lightgbm as lgb, time
from hce import Tree, node_stats, hce_solve, hce_sequential
from metrics import tweedie_dev, poisson_dev, gini_conc, calib

SEED = 20260918
CAT = ["zone", "occ", "prim", "elev", "pfirm", "floors", "ded", "cov"]
NUM = ["logcov", "cyear", "crs"]
LAG = 2  # nhãn năm t khả dụng từ đầu năm t+2 (độ trưởng thành 12 tháng sau cuối năm) — giả định khai báo


def load():
    d = pd.read_parquet("../data/nfip_cells.parquet")
    d = d[(d.year >= 2010) & (d.year <= 2024) & d.state.isin(["09", "10", "23", "24", "25", "33", "34", "44"])].reset_index(drop=True)
    for c in CAT:
        d[c] = d[c].astype(str)
    d["rate"] = d.S / d.E
    return d


def build_tree(d, levels=("state", "county", "tract")):
    paths = d[list(levels)].drop_duplicates().itertuples(index=False, name=None)
    T = Tree(sorted(paths))
    node = np.array([T.index[p] for p in d[list(levels)].itertuples(index=False, name=None)])
    return T, node


def temporal_encoder(d, T, node, years, k, k0=1.0, method="simul", lag=LAG, mu_mode="source"):
    """Giao thức Định nghĩa 3.5 với khối = năm: nguồn hợp lệ cho năm t = mọi bản ghi có a_j = year_j + lag <= t.
    Trả về z (NaN nếu khối không hợp lệ) và nhật ký kiểm soát truy cập nhãn."""
    z = np.full(len(d), np.nan); logs = []
    for t in years:
        src = (d.year.values + lag) <= t
        blk = d.year.values == t
        if not src.any():
            logs.append(dict(year=t, valid=False)); continue
        Ev, Sv = node_stats(T, node[src], d.E.values[src], d.S.values[src])
        mu0 = Sv.sum() / Ev.sum()
        if method == "simul":
            th = hce_solve(T, Ev, Sv, k, k0, mu0)
        elif method == "seq":
            th = hce_sequential(T, Ev, Sv, k, k0, mu0)
        z[blk] = th[node[blk]]
        # bất biến I2 kiểm tra được: không bản ghi nguồn nào thuộc khối hoặc có nhãn khả dụng sau đầu khối
        logs.append(dict(year=t, valid=True, n_src=int(src.sum()), overlap_block=int((src & blk).sum()),
                         max_src_label_avail=int(d.year.values[src].max() + lag)))
    return z, logs


def features(d, z=None, geo_cat=False):
    X = d[CAT + NUM].copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    if z is not None:
        X["z"] = z
    if geo_cat:
        for c in ["state", "county", "tract"]:
            X[c] = d[c].astype("category")
    return X


def align_cats(Xtr, Xte):
    for c in Xtr.columns:
        if str(Xtr[c].dtype) == "category":
            Xte[c] = pd.Categorical(Xte[c].astype(str), categories=Xtr[c].cat.categories)
    return Xte


LGB_BASE = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=2, max_cat_to_onehot=8, cat_smooth=10)


def fit_lgb(Xtr, y, w, Xva, yva, wva, objective, extra=None, rounds=4000):
    p = dict(LGB_BASE, objective=objective, **(extra or {}))
    t0 = time.time()
    dtr = lgb.Dataset(Xtr, y, weight=w); dva = lgb.Dataset(Xva, yva, weight=wva, reference=dtr)
    m = lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
    return m, time.time() - t0


def evaluate(pred_rate, E, S, xi, N=None):
    r = dict(tweedie_dev=float(np.sum(E * tweedie_dev(S / E, pred_rate, xi)) / E.sum()),
             gini=gini_conc(pred_rate, E, S), rmse_rate=float(np.sqrt(np.sum(E * (S / E - pred_rate) ** 2) / E.sum())),
             mae_rate=float(np.sum(E * np.abs(S / E - pred_rate)) / E.sum()))
    r.update({k: v for k, v in calib(pred_rate, E, S).items()})
    return r


def cluster_boot_rel(devA, devB, E, clusters, B=2000, seed=SEED):
    """Bootstrap theo cụm cho chênh lệch tương đối độ lệch trung bình có trọng số: (D_A − D_B)/D_B."""
    uc, inv = np.unique(clusters, return_inverse=True)
    a = np.bincount(inv, weights=E * devA); b = np.bincount(inv, weights=E * devB)
    rng = np.random.default_rng(seed); out = np.empty(B)
    for i in range(B):
        w = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc))
        out[i] = (w @ a - w @ b) / (w @ b)
    return out, float((a.sum() - b.sum()) / b.sum())
