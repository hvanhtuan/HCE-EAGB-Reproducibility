"""KB1 — đối sánh cơ sở trên Tầng 2 (freMTPL2freq, OpenML ID 41214, phiên bản 1).
Mục đích: kiểm chứng tính đúng của quy trình (không kết luận về bảo hiểm tài sản)."""
import numpy as np, pandas as pd, json, time, lightgbm as lgb
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor, Pool
from hce import Tree, node_stats, hce_solve, hce_sequential
from metrics import poisson_dev, gini_conc, calib

SEED = 20260918
f = pd.read_parquet("../data/raw/fremtpl2freq.pq")
prof = dict(n_raw=len(f), claims_raw=int(f.ClaimNb.sum()), expo_raw=float(f.Exposure.sum()))
# làm sạch theo quy ước phổ biến trong tài liệu: ClaimNb<=4, Exposure<=1
f["ClaimNb"] = np.minimum(f.ClaimNb.astype(int), 4); f["Exposure"] = np.minimum(f.Exposure, 1.0)
for c in ["Area", "VehBrand", "Region", "VehGas"]:
    f[c] = f[c].astype(str)
prof.update(n=len(f), claims=int(f.ClaimNb.sum()), expo=float(f.Exposure.sum()), freq=float(f.ClaimNb.sum() / f.Exposure.sum()),
            zero_share=float((f.ClaimNb == 0).mean()), n_region=f.Region.nunique(), n_area=f.Area.nunique(), n_brand=f.VehBrand.nunique())
rng = np.random.default_rng(SEED)
test = rng.random(len(f)) < 0.1
tr, te = f[~test].reset_index(drop=True), f[test].reset_index(drop=True)
prof.update(n_train=len(tr), n_test=len(te))

# cây phạm trù dựng (khai báo): gốc -> Region -> Region×Area
paths_tr = list(zip(tr.Region, tr.Area)); paths_te = list(zip(te.Region, te.Area))
T = Tree(sorted(set(paths_tr) | set(paths_te)))
leaf_tr = np.array([T.index[p] for p in paths_tr]); leaf_te = np.array([T.index[p] for p in paths_te])


def hce_encode(k1, k2, K=5, seq=False, flat=False):
    """Mã hóa ngoài mẫu theo nếp (không có trục thời gian trong freMTPL2 nên giao thức mốc nhãn không áp dụng)."""
    fold = np.random.default_rng(SEED + 1).integers(0, K, len(tr)); z = np.empty(len(tr)); mu0 = tr.ClaimNb.sum() / tr.Exposure.sum()
    k = [0, k1, k2] if not flat else [0, 1e12, k2]  # flat: gộp Region ~ gốc (khóa cấp 1), chỉ co ngót lá về gốc
    solve = hce_sequential if seq else hce_solve
    for kk in range(K):
        m = fold != kk
        Ev, Sv = node_stats(T, leaf_tr[m], tr.Exposure.values[m], tr.ClaimNb.values[m].astype(float))
        th = solve(T, Ev, Sv, k, 1.0, mu0); z[~m] = th[leaf_tr[~m]]
    Ev, Sv = node_stats(T, leaf_tr, tr.Exposure.values, tr.ClaimNb.values.astype(float))
    th = solve(T, Ev, Sv, k, 1.0, mu0)
    return z, th[leaf_te]


num = ["VehPower", "VehAge", "DrivAge", "BonusMalus"]
def Xflat(d):
    X = d[num].astype(float).copy(); X["logDens"] = np.log(d.Density.astype(float)); X["Diesel"] = (d.VehGas == "Diesel").astype(float)
    X["Brand"] = d.VehBrand.astype("category"); return X

res = {}; preds = {}
yN_tr, E_tr = tr.ClaimNb.values.astype(float), tr.Exposure.values
yN_te, E_te = te.ClaimNb.values.astype(float), te.Exposure.values

# (1) GLM Poisson (biến liên tục được chia khoảng, biến phân loại một-nóng) — đối chứng chính
def glm_design(d):
    D = pd.DataFrame({"VehPowerG": np.minimum(d.VehPower, 9).astype(str), "VehAgeG": pd.cut(d.VehAge, [-1, 0, 10, 200]).astype(str),
                      "DrivAgeG": pd.cut(d.DrivAge, [17, 20, 25, 30, 40, 50, 70, 200]).astype(str), "Brand": d.VehBrand, "Gas": d.VehGas,
                      "Area": d.Area, "Region": d.Region})
    return D, np.c_[np.log(np.minimum(d.BonusMalus, 150)), np.log(d.Density)]
Dtr, Ntr = glm_design(tr); Dte, Nte = glm_design(te)
oh = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False).fit(Dtr)
Xg_tr = np.c_[oh.transform(Dtr), Ntr]; Xg_te = np.c_[oh.transform(Dte), Nte]
t0 = time.time(); glm = PoissonRegressor(alpha=1e-6, max_iter=1000).fit(Xg_tr, yN_tr / E_tr, sample_weight=E_tr)
preds["GLM Poisson"] = glm.predict(Xg_te); res["GLM Poisson"] = {"train_sec": time.time() - t0}

# chọn siêu tham số boosting bằng tập theo dõi tách từ train (ngân sách như nhau cho các biến thể boosting)
val = np.random.default_rng(SEED + 2).random(len(tr)) < 0.15
params = dict(objective="poisson", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.9,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=2)

def fit_lgb(Xtr, Xte, name, cat="auto"):
    t0 = time.time()
    dtr = lgb.Dataset(Xtr[~val], yN_tr[~val] / E_tr[~val], weight=E_tr[~val], categorical_feature=cat)
    dva = lgb.Dataset(Xtr[val], yN_tr[val] / E_tr[val], weight=E_tr[val], categorical_feature=cat)
    m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    preds[name] = m.predict(Xte, num_iteration=m.best_iteration); res[name] = {"train_sec": time.time() - t0, "rounds": m.best_iteration}

# (2) LightGBM với Region/Area dạng phân loại nội tại
Xtr, Xte = Xflat(tr), Xflat(te); Xtr["Region"] = tr.Region.astype("category"); Xte["Region"] = pd.Categorical(te.Region, categories=Xtr.Region.cat.categories)
Xtr["Area"] = tr.Area.astype("category"); Xte["Area"] = pd.Categorical(te.Area, categories=Xtr.Area.cat.categories)
Xte["Brand"] = pd.Categorical(te.VehBrand, categories=Xtr.Brand.cat.categories)
fit_lgb(Xtr, Xte, "GBDT phân loại nội tại")

# chọn k cho HCE bằng deviance ngoài nếp trên train (lưới nhỏ, khai báo)
best = None
for kmul in [10, 100, 1000, 10000]:
    z, _ = hce_encode(kmul, kmul)
    d = np.sum(E_tr * poisson_dev(yN_tr / E_tr, np.maximum(z, 1e-6))) / E_tr.sum()
    if best is None or d < best[1]: best = (kmul, d)
res["HCE_k_selected"] = best[0]
variants = {"EAGB (HCE đồng thời)": dict(seq=False), "Đối chứng: co ngót tuần tự": dict(seq=True), "Đối chứng: mã hóa phẳng (lá→gốc)": dict(flat=True)}
for name, kw in variants.items():
    z, zte = hce_encode(best[0], best[0], **kw)
    A, B = Xflat(tr), Xflat(te); B["Brand"] = pd.Categorical(te.VehBrand, categories=A.Brand.cat.categories)
    A["z"], B["z"] = z, zte; fit_lgb(A, B, name)

# (3) CatBoost với thống kê mục tiêu có thứ tự
t0 = time.time()
cols = num + ["Density", "VehBrand", "VehGas", "Region", "Area"]; cats = ["VehBrand", "VehGas", "Region", "Area"]
cb = CatBoostRegressor(loss_function="Poisson", iterations=3000, learning_rate=0.05, depth=6, random_seed=SEED, verbose=0, thread_count=2, od_type="Iter", od_wait=100)
ptr = Pool(tr.loc[~val, cols], np.log(1) + yN_tr[~val], cat_features=cats, baseline=np.log(E_tr[~val]))
pva = Pool(tr.loc[val, cols], yN_tr[val], cat_features=cats, baseline=np.log(E_tr[val]))
cb.fit(ptr, eval_set=pva)
pte = Pool(te[cols], cat_features=cats)
preds["CatBoost (OTS)"] = np.exp(cb.predict(pte, prediction_type="RawFormulaVal")); res["CatBoost (OTS)"] = {"train_sec": time.time() - t0}

# (4) mô hình rỗng
preds["Mô hình rỗng"] = np.full(len(te), yN_tr.sum() / E_tr.sum())

for name, p in preds.items():
    dev = poisson_dev(yN_te, p * E_te)  # độ lệch Poisson trên thang số vụ, trung bình theo hợp đồng (quy ước Noll et al.)
    r = res.setdefault(name, {})
    r.update(poisson_dev_x100=float(100 * dev.mean()), gini=gini_conc(p, E_te, yN_te), **{k: v for k, v in calib(p, E_te, yN_te).items() if k != "dec_OE"})
# bootstrap cấp hợp đồng (phân tích độ nhạy, giả định độc lập) cho chênh lệch so với GLM
rng = np.random.default_rng(SEED + 3); B = 1000; n = len(te)
dG = poisson_dev(yN_te, preds["GLM Poisson"] * E_te)
for name, p in preds.items():
    d = poisson_dev(yN_te, p * E_te) - dG
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)]) * 100
    res[name]["diff_vs_GLM_x100"] = float(100 * d.mean()); res[name]["diff_CI95"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
out = {"profile": prof, "results": res}
json.dump(out, open("../results/kb1_results.json", "w"), indent=1, ensure_ascii=False)
# lưu thêm mảng dự báo trên tập kiểm định (tệp MỚI, không đổi nội dung/mã băm của kb1_results.json)
# để các đối chứng bổ sung sau này (vd. kb1_credibility_transformer.py) có thể bootstrap đối chiếu
# trên ĐÚNG cùng tập kiểm định, không phải chạy lại toàn bộ script này.
np.savez_compressed("../results/kb1_preds.npz", **{n.encode("ascii", "ignore").decode()[:40] + str(i): p
                                                    for i, (n, p) in enumerate(preds.items())})
json.dump({n: n.encode("ascii", "ignore").decode()[:40] + str(i) for i, n in enumerate(preds)},
          open("../results/kb1_preds_index.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, indent=1, ensure_ascii=False))
