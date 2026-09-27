"""Thống kê bổ sung cho v1.5 (đều tính lại từ dự báo đã lưu của KB2 ở chế độ test — không huấn luyện lại):
(a) bootstrap theo khối quận x năm và theo năm; sai lệch khi loại năm sự kiện 2021;
(b) khoảng tin cậy cho ba thành phần của H1b (O/E, độ dốc hiệu chuẩn, lệch thập phân vị cực đại);
(c) dao (jackknife) loại một bang cho thống kê chính của H1a — đây là phân tích *phía đánh giá*, không phải kiểm định ngoài không gian;
(d) dải bất định cho hiệu chuẩn theo thập phân vị.
"""
import json, numpy as np, pandas as pd
from nfip_lib import load
from metrics import tweedie_dev, calib, gini_conc

XI = 1.5
B = 2000
SEED = 20260925  # hạt giống lịch sử của phân tích bổ sung/hậu kiểm đã lưu
KEY = {"EAGB": "EAGB (HCE ng thi)0", "GLM": "GLM Tweedie (i chng chnh)8",
       "FLAT": "GBDT + m ha phng co ngt (H1c)2", "SEQ": "GBDT + HCE tun t (GT2)1"}

d = load()
TE = ((d.year >= 2021) & (d.year <= 2023)).values
E, S = d.E.values[TE], d.S.values[TE]
yr = d.year.values[TE]
cnty = (d.state.values[TE].astype(str) + "-" + d.county.values[TE].astype(str))
st = d.state.values[TE].astype(str)
z = np.load("../results/kb2_test_preds.npz")
P = {k: np.maximum(z[v], 1e-6) for k, v in KEY.items()}
dev = {k: tweedie_dev(S / E, p, XI) for k, p in P.items()}
out = {"n_eval": int(TE.sum()), "B": B, "seed": SEED}


def rel(a, b, mask=None):
    m = slice(None) if mask is None else mask
    return float((np.sum(E[m] * dev[a][m]) - np.sum(E[m] * dev[b][m])) / np.sum(E[m] * dev[b][m]))


def boot_rel(a, b, blocks, mask=None, B=B, seed=SEED):
    """Bootstrap theo khối: khối là đơn vị lấy lại. Trả về (điểm, phân vị)."""
    m = np.ones(len(E), bool) if mask is None else mask
    key = blocks[m]
    uc, inv = np.unique(key, return_inverse=True)
    na = np.bincount(inv, weights=(E * dev[a])[m], minlength=len(uc))
    nb = np.bincount(inv, weights=(E * dev[b])[m], minlength=len(uc))
    rng = np.random.default_rng(seed); o = np.empty(B)
    for i in range(B):
        w = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc))
        o[i] = (w @ na - w @ nb) / (w @ nb)
    return dict(n_blocks=int(len(uc)), rel=rel(a, b, m),
                lo95=float(np.quantile(o, .025)), hi95=float(np.quantile(o, .975)),
                hi95_1s=float(np.quantile(o, .95)), lo95_1s=float(np.quantile(o, .05)))


# ---------- (a) các lược đồ khối khác nhau cho H1a và H1c
blocks = {"quan": cnty, "quan_x_nam": np.char.add(cnty.astype(str), np.char.add("|", yr.astype(str))),
          "nam": yr.astype(str), "bang_x_nam": np.char.add(st, np.char.add("|", yr.astype(str)))}
out["block_boot"] = {}
for pair in [("EAGB", "GLM"), ("EAGB", "FLAT"), ("EAGB", "SEQ")]:
    for bn, bv in blocks.items():
        out["block_boot"][f"{pair[0]}/{pair[1]}|{bn}"] = boot_rel(pair[0], pair[1], bv)

# loại năm sự kiện 2021 (phân tích độ nhạy, không tiền đăng ký)
out["drop_event_year"] = {}
for pair in [("EAGB", "GLM"), ("EAGB", "FLAT"), ("EAGB", "SEQ")]:
    out["drop_event_year"][f"{pair[0]}/{pair[1]}"] = boot_rel(pair[0], pair[1], cnty, mask=(yr != 2021))
out["per_year"] = {}
for y in [2021, 2022, 2023]:
    out["per_year"][int(y)] = {f"{a}/{b}": boot_rel(a, b, cnty, mask=(yr == y))
                              for a, b in [("EAGB", "GLM"), ("EAGB", "FLAT")]}

# ---------- (b) khoảng tin cậy cho ba thành phần của H1b
def comp(mask, k):
    c = calib(P[k][mask], E[mask], S[mask])
    return np.array([c["OE"], c["slope"], c["max_dec_dev"]])


uc, inv = np.unique(cnty, return_inverse=True)
members = [np.where(inv == c)[0] for c in range(len(uc))]
rng = np.random.default_rng(SEED)
Bc = 600  # ít lần hơn vì mỗi lần phải tính lại hiệu chuẩn theo thập phân vị
samp = {k: np.empty((Bc, 3)) for k in ("EAGB", "GLM")}
for i in range(Bc):
    idx = np.concatenate([members[c] for c in rng.integers(0, len(uc), len(uc))])
    for k in samp:
        samp[k][i] = comp(idx, k)
out["h1b_ci"] = {}
for k in samp:
    pt = comp(np.arange(len(E)), k)
    for j, nm in enumerate(["OE", "slope", "max_dec_dev"]):
        s = samp[k][:, j]; s = s[np.isfinite(s)]
        out["h1b_ci"][f"{k}|{nm}"] = dict(point=float(pt[j]), lo95=float(np.quantile(s, .025)),
                                          hi95=float(np.quantile(s, .975)), B=int(len(s)))
# hiệu số EAGB - GLM cho từng thành phần
for j, nm in enumerate(["OE", "slope", "max_dec_dev"]):
    s = samp["EAGB"][:, j] - samp["GLM"][:, j]; s = s[np.isfinite(s)]
    pt = comp(np.arange(len(E)), "EAGB")[j] - comp(np.arange(len(E)), "GLM")[j]
    out["h1b_ci"][f"DIFF|{nm}"] = dict(point=float(pt), lo95=float(np.quantile(s, .025)),
                                       hi95=float(np.quantile(s, .975)), B=int(len(s)))

# ---------- (c) jackknife loại một bang
ST = {"09": "CT", "10": "DE", "23": "ME", "24": "MD", "25": "MA", "33": "NH", "34": "NJ", "44": "RI"}
out["loo_state"] = {}
for code, ab in ST.items():
    m = st != code
    out["loo_state"][ab] = dict(
        share_E=float(E[st == code].sum() / E.sum()),
        share_S=float(S[st == code].sum() / S.sum()),
        **{f"{a}/{b}": rel(a, b, m) for a, b in [("EAGB", "GLM"), ("EAGB", "FLAT")]})
out["loo_state_full"] = {f"{a}/{b}": rel(a, b) for a, b in [("EAGB", "GLM"), ("EAGB", "FLAT")]}
# và chỉ giữ một bang
out["single_state"] = {}
for code, ab in ST.items():
    m = st == code
    if m.sum() < 1000:
        continue
    out["single_state"][ab] = {f"{a}/{b}": rel(a, b, m) for a, b in [("EAGB", "GLM"), ("EAGB", "FLAT")]}

# ---------- (d) dải bất định cho hiệu chuẩn theo thập phân vị
def dec_oe(mask, k, nb=10):
    p = P[k][mask]; Em, Sm = E[mask], S[mask]
    o = np.argsort(p, kind="stable"); cE = np.cumsum(Em[o]) / Em.sum()
    b = np.minimum((cE * nb).astype(int), nb - 1)
    sh = p[o] * Em[o]
    return np.array([Sm[o][b == j].sum() / max(sh[b == j].sum(), 1e-12) for j in range(nb)])


rng = np.random.default_rng(SEED + 1)
Bd = 600
band = {k: np.empty((Bd, 10)) for k in ("EAGB", "GLM", "FLAT")}
for i in range(Bd):
    idx = np.concatenate([members[c] for c in rng.integers(0, len(uc), len(uc))])
    for k in band:
        band[k][i] = dec_oe(idx, k)
out["calib_band"] = {k: dict(point=dec_oe(np.arange(len(E)), k).tolist(),
                            lo95=np.quantile(band[k], .025, axis=0).tolist(),
                            hi95=np.quantile(band[k], .975, axis=0).tolist()) for k in band}

json.dump(out, open("../results/extra_stats.json", "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: out[k] for k in ("block_boot", "drop_event_year", "loo_state_full")}, indent=1, ensure_ascii=False)[:4000])
print("ok")
