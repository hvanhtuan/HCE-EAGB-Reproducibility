"""Chẩn đoán thăm dò (không tiền đăng ký) trên dự báo kb2_test: (1) nghịch lý Bảng 3, (2) phân tích theo sự kiện lũ."""
import json, numpy as np, pandas as pd, lightgbm as lgb
from nfip_lib import *
from metrics import tweedie_dev
XI = 1.5
d = load(); TE = ((d.year >= 2021) & (d.year <= 2023)).values
z = np.load("../results/kb2_test_preds.npz"); P = [z[k] for k in z.files]
NM = ["EAGB", "Tuần tự", "Phẳng", "Cấp quận", "Nội tại", "Không địa lý", "EAGB-PG", "HCE thuần", "GLM", "GLM spline", "CatBoost", "Rỗng"]
preds = dict(zip(NM, P)); dt = d[TE].reset_index(drop=True); E, S, yr, cl = dt.E.values, dt.S.values, dt.year.values, dt.county.values
dev = {k: tweedie_dev(S / E, np.maximum(v, 1e-6), XI) for k, v in preds.items()}
mean = lambda dv, m=None: float(np.sum((E * dv)[m if m is not None else slice(None)]) / np.sum(E[m if m is not None else slice(None)]))
out = {"n": int(len(dt))}
# D1 theo năm
out["by_year"] = {k: {int(y): mean(dev[k], yr == y) for y in [2021, 2022, 2023]} for k in NM}
# D2 độ lệch sau khi bỏ các ô có tổn thất quan sát lớn nhất (tập chung cho mọi mô hình)
o = np.argsort(-S); out["trim"] = {}
for frac in [0.001, 0.01]:
    keep = np.ones(len(dt), bool); keep[o[:int(np.ceil(len(dt) * frac))]] = False
    out["trim"][str(frac)] = {k: mean(dev[k], keep) for k in NM}
out["top_share_own"] = {k: float(np.sort(E * dev[k])[::-1][:int(np.ceil(len(dt) * 0.001))].sum() / (E * dev[k]).sum()) for k in NM}
# D3 cân chỉnh mức theo từng năm bằng nhãn kiểm định (chỉ để chẩn đoán: tách sai lệch mức khỏi sai lệch xếp hạng)
adj = {}
for k, v in preds.items():
    v2 = v.copy()
    for y in [2021, 2022, 2023]:
        m = yr == y; v2[m] = v[m] * S[m].sum() / (v[m] * E[m]).sum()
    adj[k] = mean(tweedie_dev(S / E, np.maximum(v2, 1e-6), XI))
out["oracle_year_level"] = adj
# D4 mức dùng đặc trưng địa lý (gain)
idx = json.load(open("../results/models/test_index.json")); imp = {}
for nm, f in idx.items():
    m = lgb.Booster(model_file="../results/models/" + f); g = m.feature_importance("gain"); imp[nm] = {n: float(x / g.sum()) for n, x in zip(m.feature_name(), g)}
out["importance_gain"] = imp
# D5 sự kiện lũ
ce = pd.read_parquet("../data/claims_by_cell_event.parquet"); KEYC = ["tract", "zone", "occ", "prim", "elev", "pfirm", "floors", "ded", "cov"]
for c in ["prim", "elev", "pfirm", "floors"]: ce[c] = ce[c].astype(str)
ce = ce[ce.year.between(2021, 2023)]
ce["ev2"] = ce.ev.replace({"Tropical Storm Henri": "Henri/Elsa", "Hurricane Elsa": "Henri/Elsa"})
flags = {}
for ev in ["Hurricane Ida", "Henri/Elsa", "December Nor'easter", "(không có nhãn sự kiện)"]:
    w = ce[(ce.ev2 == ev) & (ce.S > 0)][["year"] + KEYC].drop_duplicates(); w["f"] = 1
    m = dt[["year"] + KEYC].merge(w, on=["year"] + KEYC, how="left"); flags[ev] = m.f.fillna(0).values.astype(bool)
out["event_cells"] = {ev: int(f.sum()) for ev, f in flags.items()}
Sev = {}
for ev in flags:
    w = ce[ce.ev2 == ev].groupby(["year"] + KEYC)["S"].sum().reset_index(); m = dt[["year"] + KEYC].merge(w, on=["year"] + KEYC, how="left"); Sev[ev] = m.S.fillna(0).values
out["event_S_share"] = {ev: float(Sev[ev].sum() / S.sum()) for ev in Sev}; out["event_S_share_total_matched_check"] = float(sum(Sev.values()).sum() / S.sum()) if False else float(sum(v.sum() for v in Sev.values()) / S.sum())
share_dev = {}
for k in ["EAGB", "Tuần tự", "Phẳng", "GLM", "Không địa lý", "Rỗng"]:
    tot = (E * dev[k]).sum(); share_dev[k] = {ev: float((E * dev[k])[f].sum() / tot) for ev, f in flags.items()}
out["event_dev_share"] = share_dev
def boot(a, b, m):
    bs, pt = cluster_boot_rel(dev[a][m], dev[b][m], E[m], cl[m], B=2000)
    return dict(rel=pt, lo95=float(np.quantile(bs, .025)), hi95=float(np.quantile(bs, .975)), hi95_1s=float(np.quantile(bs, .95)), n=int(m.sum()))
comp = {"H1a": ("EAGB", "GLM"), "H1c": ("EAGB", "Phẳng"), "GT2": ("EAGB", "Tuần tự")}
out["event_delta"] = {}
allm = np.ones(len(dt), bool)
subsets = {"Toàn bộ": allm}
for ev, f in flags.items():
    subsets["Loại các ô có bồi thường thuộc: " + ev] = ~f
subsets["Loại các ô có bồi thường thuộc Ida hoặc Henri/Elsa"] = ~(flags["Hurricane Ida"] | flags["Henri/Elsa"])
subsets["Chỉ các ô có bồi thường thuộc Ida"] = flags["Hurricane Ida"]
for sn, m in subsets.items():
    out["event_delta"][sn] = {c: boot(a, b, m) for c, (a, b) in comp.items()}
    print(sn, {c: round(v["rel"], 3) for c, v in out["event_delta"][sn].items()}, flush=True)
json.dump(out, open("../results/review_diag.json", "w"), indent=1, ensure_ascii=False, default=float)
