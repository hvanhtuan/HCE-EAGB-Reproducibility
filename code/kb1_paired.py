"""Khoảng tin cậy ghép cặp (bootstrap cấp hợp đồng, B=2000, cùng mẫu bootstrap cho mọi cặp) cho freMTPL2freq — phân tích thăm dò."""
import json, numpy as np, pandas as pd
from metrics import poisson_dev
SEED = 20260918
f = pd.read_parquet("../data/raw/fremtpl2freq.pq")
f["ClaimNb"] = np.minimum(f.ClaimNb.astype(int), 4); f["Exposure"] = np.minimum(f.Exposure, 1.0)
rng = np.random.default_rng(SEED); test = rng.random(len(f)) < 0.1; te = f[test].reset_index(drop=True)
z = np.load("../results/kb1_preds.npz"); idx = json.load(open("../results/kb1_preds_index.json"))
P = {k: z[v] for k, v in idx.items()}
y = te.ClaimNb.values.astype(float); E = te.Exposure.values
print(len(te), {k: len(v) for k, v in P.items()}, flush=True)
# các dự báo là số vụ kỳ vọng hay tần suất? kiểm tra bằng tổng
for k, v in P.items(): print(k, float(v.sum()), float(y.sum()), float((v * E).sum()))
D = {k: poisson_dev(y, v * E) for k, v in P.items()}
print({k: round(100 * v.mean(), 3) for k, v in D.items()})
n = len(te); rng = np.random.default_rng(SEED + 4); B = 2000
IDX = [rng.integers(0, n, n) for _ in range(B)]
pairs = [("EAGB (HCE đồng thời)", "Đối chứng: mã hóa phẳng (lá→gốc)"), ("EAGB (HCE đồng thời)", "Đối chứng: co ngót tuần tự"),
         ("Đối chứng: co ngót tuần tự", "Đối chứng: mã hóa phẳng (lá→gốc)"), ("EAGB (HCE đồng thời)", "GBDT phân loại nội tại"), ("EAGB (HCE đồng thời)", "CatBoost (OTS)")]
res = {}
for a, b in pairs:
    dd = D[a] - D[b]; bs = np.array([dd[i].mean() for i in IDX]) * 100
    res[f"{a} - {b}"] = dict(diff_x100=float(100 * dd.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)))
    print(a[:25], "-", b[:30], {k: round(v, 4) for k, v in res[f"{a} - {b}"].items()}, flush=True)
json.dump(res, open("../results/review_kb1_paired.json", "w"), indent=1, ensure_ascii=False)
