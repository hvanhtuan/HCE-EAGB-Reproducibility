"""Phân tích thăm dò bổ sung theo yêu cầu phản biện (không tiền đăng ký). Chế độ:
  k      : độ nhạy theo k của HCE đồng thời và tuần tự (k vượt biên lưới), giữ nguyên cấu hình khóa còn lại
  rule   : quy tắc ghép bồi thường thay thế (tract | county), huấn luyện lại EAGB, tuần tự, phẳng, GLM
  diag   : chẩn đoán nghịch lý Bảng 3 và phân tích theo sự kiện lũ, trên dự báo kb2_test đã có"""
import sys, json, time, pickle, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import nfip_lib
from nfip_lib import *
from metrics import tweedie_dev
MODE = sys.argv[1]; ARG = sys.argv[2] if len(sys.argv) > 2 else None
XI_EVAL = 1.5
pre = json.load(open("../results/prereg_H1.json"))
if MODE == "rule":
    def load_rule():
        d = pd.read_parquet(f"../data/nfip_cells_rule_{ARG}.parquet")
        d = d[(d.year >= 2010) & (d.year <= 2024) & d.state.isin(["09", "10", "23", "24", "25", "33", "34", "44"])].reset_index(drop=True)
        for c in CAT: d[c] = d[c].astype(str)
        d["rate"] = d.S / d.E
        return d
    d = load_rule()
else:
    d = load()
T, node = build_tree(d); Tflat, nodeflat = build_tree(d, levels=("tract",))
TR = (d.year >= 2012) & (d.year <= 2018); TE = (d.year >= 2021) & (d.year <= 2023)
E, S, rate = d.E.values, d.S.values, d.rate.values
years_all = list(range(2010, 2025)); ksel = pre["k_selected"]; xi = pre["xi_selected"]


def enc(kmul, method, tree):
    TT, nd = {"full": (T, node), "flat": (Tflat, nodeflat)}[tree]; L = {"full": 3, "flat": 1}[tree]
    return temporal_encoder(d, TT, nd, years_all, [0] + [kmul] * L, k0=1.0, method=method)[0]


def fit_pred(Z, name):
    X = features(d, Z); Xtr = X[TR.values]; Xev = align_cats(Xtr, X[TE.values].copy())
    m = lgb.train(dict(LGB_BASE, objective="tweedie", tweedie_variance_power=xi), lgb.Dataset(Xtr, rate[TR], weight=E[TR]), pre["rounds"][name])
    return m.predict(Xev)


def boot(devA, devB, Ee, cl, B=2000):
    bs, pt = cluster_boot_rel(devA, devB, Ee, cl, B=B)
    return dict(rel=pt, lo95=float(np.quantile(bs, .025)), hi95=float(np.quantile(bs, .975)), hi95_1s=float(np.quantile(bs, .95)))


Ee, Se, cl = E[TE.values], S[TE.values], d.county.values[TE.values]
dev = lambda p: tweedie_dev(Se / Ee, np.maximum(p, 1e-6), XI_EVAL)
NAMES = dict(simul="EAGB (HCE đồng thời)", seq="GBDT + HCE tuần tự (GT2)", flat="GBDT + mã hóa phẳng co ngót (H1c)")
out = {"mode": MODE, "arg": ARG}
if MODE == "k":
    Zflat = enc(ksel["flat"], "simul", "flat"); pflat = fit_pred(Zflat, NAMES["flat"]); res = {}
    for km in [int(x) for x in ARG.split(",")]:
        t0 = time.time(); ps = fit_pred(enc(km, "simul", "full"), NAMES["simul"]); pq = fit_pred(enc(km, "seq", "full"), NAMES["seq"])
        r = dict(dev_eagb=float(np.sum(Ee * dev(ps)) / Ee.sum()), dev_seq=float(np.sum(Ee * dev(pq)) / Ee.sum()), dev_flat=float(np.sum(Ee * dev(pflat)) / Ee.sum()))
        r["H1c"] = boot(dev(ps), dev(pflat), Ee, cl); r["GT2"] = boot(dev(ps), dev(pq), Ee, cl); r["sec"] = time.time() - t0
        res[km] = r; print(km, json.dumps(r), flush=True)
    out["res"] = res
elif MODE == "rule":
    P = {}
    P["simul"] = fit_pred(enc(ksel["simul"], "simul", "full"), NAMES["simul"]); P["seq"] = fit_pred(enc(ksel["seq"], "seq", "full"), NAMES["seq"])
    P["flat"] = fit_pred(enc(ksel["flat"], "simul", "flat"), NAMES["flat"])
    cats = CAT + ["state", "county"]
    oh = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False, dtype=np.float64).fit(d.loc[TR, cats]); sc = StandardScaler().fit(d.loc[TR, NUM])
    mk = lambda m: np.hstack([oh.transform(d.loc[m, cats]), sc.transform(d.loc[m, NUM])])
    g = TweedieRegressor(power=xi, link="log", alpha=1e-4, solver="newton-cholesky", max_iter=100, tol=1e-8).fit(mk(TR), rate[TR], sample_weight=E[TR])
    P["glm"] = g.predict(mk(TE))
    D = {k: dev(v) for k, v in P.items()}
    out["n_train_claims_S"] = float(S[TR.values].sum()); out["n_test_S"] = float(Se.sum())
    out["dev"] = {k: float(np.sum(Ee * v) / Ee.sum()) for k, v in D.items()}
    out["H1a"] = boot(D["simul"], D["glm"], Ee, cl); out["H1c"] = boot(D["simul"], D["flat"], Ee, cl); out["GT2"] = boot(D["simul"], D["seq"], Ee, cl)
    print(json.dumps(out, indent=1), flush=True)
json.dump(out, open(f"../results/review_{MODE}{'_' + ARG.replace(',', '_') if ARG else ''}.json", "w"), indent=1, default=float)
