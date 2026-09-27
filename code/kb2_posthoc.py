"""Phân tích khám phá (KHÔNG tiền đăng ký) sau khi mở tập kiểm định KB2: độ lệch theo năm, loại năm có sự kiện, hình thập phân vị và Lorenz."""
import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from nfip_lib import *
from metrics import tweedie_dev
plt.rcParams["font.family"] = "DejaVu Sans"
d = load(); TE = ((d.year >= 2021) & (d.year <= 2023)).values
P = np.load("../results/kb2_test_preds.npz"); keys = list(P.keys())
names = json.load(open("../results/kb2_test.json"))["eval"].keys(); names = list(names)
pred = {n: P[k] for n, k in zip(names, keys)}
E, S, yr, cl = d.E.values[TE], d.S.values[TE], d.year.values[TE], d.county.values[TE]
out = {}
for n, p in pred.items():
    dv = tweedie_dev(S / E, np.maximum(p, 1e-6), 1.5)
    out[n] = {str(y): float(np.sum(E[yr == y] * dv[yr == y]) / E[yr == y].sum()) for y in (2021, 2022, 2023)}
    m = yr != 2021; out[n]["2022-2023"] = float(np.sum(E[m] * dv[m]) / E[m].sum())
    # đóng góp của 0,1% ô có độ lệch lớn nhất
    c = E * dv; o = np.sort(c)[::-1]; out[n]["share_top0.1pct"] = float(o[: max(1, len(o) // 1000)].sum() / o.sum())
b = {}
dvE = tweedie_dev(S / E, pred["EAGB (HCE đồng thời)"], 1.5); dvG = tweedie_dev(S / E, pred["GLM Tweedie (đối chứng chính)"], 1.5)
for lab, m in [("2022-2023", yr != 2021), ("2021", yr == 2021)]:
    bs, pt = cluster_boot_rel(dvE[m], dvG[m], E[m], cl[m], B=2000)
    b[lab] = dict(rel=pt, lo95=float(np.quantile(bs, .025)), hi95=float(np.quantile(bs, .975)))
out["_boot_EAGB_vs_GLM"] = b
json.dump(out, open("../results/kb2_posthoc.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
# Hình: O/E theo thập phân vị và đường Lorenz tập trung
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for n, lab in [("EAGB (HCE đồng thời)", "EAGB"), ("GLM Tweedie (đối chứng chính)", "GLM Tweedie"), ("GBDT + mã hóa phẳng co ngót (H1c)", "Mã hóa phẳng"), ("EAGB-PG (Poisson–Gamma)", "EAGB-PG")]:
    p = pred[n]; o = np.argsort(p, kind="stable"); cE = np.cumsum(E[o]) / E.sum(); bb = np.minimum((cE * 10).astype(int), 9)
    obs = np.array([S[o][bb == k].sum() / E[o][bb == k].sum() for k in range(10)]); exp_ = np.array([(p * E)[o][bb == k].sum() / E[o][bb == k].sum() for k in range(10)])
    ax[0].plot(range(1, 11), np.log10(np.maximum(obs, 1e-2)) - np.log10(exp_), marker="o", label=lab)
    u, inv = np.unique(-p, return_inverse=True); Eg = np.bincount(inv, weights=E); Sg = np.bincount(inv, weights=S)
    ax[1].plot(np.r_[0, np.cumsum(Eg)] / E.sum(), np.r_[0, np.cumsum(Sg)] / S.sum(), label=lab)
ax[0].axhline(0, color="k", lw=.8); ax[0].set_xlabel("Thập phân vị phơi nhiễm theo phí thuần dự báo"); ax[0].set_ylabel("log10(quan sát/dự báo)"); ax[0].legend(fontsize=8)
ax[1].plot([0, 1], [0, 1], "k--", lw=.8); ax[1].set_xlabel("Tỷ lệ phơi nhiễm tích lũy (giảm dần theo dự báo)"); ax[1].set_ylabel("Tỷ lệ tổn thất tích lũy"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("../doc/figs/fig4_1_calib_lorenz.png", dpi=200); print("fig ok")
