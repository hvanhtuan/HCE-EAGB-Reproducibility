"""KB3b — Kiểm định ngoài không gian đúng nghĩa (giữ-lại-một-bang).

Với mỗi bang s trong 8 bang: huấn luyện trên 2012–2018 của 7 bang còn lại, đánh giá trên 2021–2023 của
riêng bang s. Bộ mã hóa HCE cũng chỉ lấy nguồn nhãn từ 7 bang huấn luyện, nên mọi nút của bang s có
phơi nhiễm nguồn bằng 0 và bị co ngót hoàn toàn về nút cha — đây đúng là chế độ khởi động nguội mà
một doanh nghiệp gặp khi mở rộng sang địa bàn mới.

Mỗi ô của tập kiểm định được đánh giá đúng một lần, bởi một mô hình chưa từng thấy bang của nó; gộp
tám lần đánh giá lại cho một thống kê ngoài không gian trên toàn bộ 399 128 ô, có bootstrap cụm quận.

Cấu hình (số vòng, xi, k, hệ số mức) lấy nguyên từ tệp tiền đăng ký của H1 — không tinh chỉnh lại.
"""
import json, time, gc, os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from nfip_lib import load, build_tree, features, align_cats, LGB_BASE, CAT, NUM, LAG, SEED, evaluate
from hce import node_stats, hce_solve
from metrics import tweedie_dev, gini_conc

XI_EVAL = 1.5
B_BOOT = 2000
SEED_B = 20260925
pre = json.load(open("../results/prereg_H1.json"))
XI = pre["xi_selected"]
KSEL = pre["k_selected"]
ROUNDS = pre["rounds"]
ST = {"09": "CT", "10": "DE", "23": "ME", "24": "MD", "25": "MA", "33": "NH", "34": "NJ", "44": "RI"}

d = load()
T, node = build_tree(d)
Tflat, nodeflat = build_tree(d, levels=("tract",))
yrs = list(range(2010, 2025))
TR = ((d.year >= 2012) & (d.year <= 2018)).values
TE = ((d.year >= 2021) & (d.year <= 2023)).values
E, S, rate = d.E.values, d.S.values, d.rate.values
st = d.state.values.astype(str)


def enc_masked(TT, nd, kmul, nlev, src_ok):
    """Bộ mã hóa HCE theo khối năm, nhưng nguồn nhãn bị giới hạn thêm bởi src_ok (mặt nạ boolean).
    Trả về (z, root_by_year) để đo được tỷ lệ ô bị co ngót hoàn toàn về nút gốc (khởi động nguội)."""
    z = np.full(len(d), np.nan)
    root = {}
    k = [0] + [kmul] * nlev
    for t in yrs:
        src = ((d.year.values + LAG) <= t) & src_ok
        blk = d.year.values == t
        if not src.any():
            continue
        Ev, Sv = node_stats(TT, nd[src], E[src], S[src])
        mu0 = Sv.sum() / Ev.sum()
        th = hce_solve(TT, Ev, Sv, k, 1.0, mu0)
        z[blk] = th[nd[blk]]
        root[t] = float(th[0])
    return z, root


def glm_fit_predict(tr_mask, te_mask):
    """GLM Tweedie cùng thiết kế với KB2. Ma trận thiết kế dựng theo lô và ở float32 để vừa bộ nhớ 7 GB."""
    cats = CAT + ["state", "county"]
    oh = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False, dtype=np.float32).fit(d.loc[tr_mask, cats])
    sc = StandardScaler().fit(d.loc[tr_mask, NUM])

    ncol = len(oh.get_feature_names_out()) + len(NUM)

    def design(mask):
        idx = np.where(mask)[0]
        out_ = np.empty((len(idx), ncol), dtype=np.float32)   # cấp phát một lần, không vstack
        for a in range(0, len(idx), 200_000):
            sl = idx[a:a + 200_000]
            out_[a:a + len(sl), :len(oh.get_feature_names_out())] = oh.transform(d.iloc[sl][cats])
            out_[a:a + len(sl), len(oh.get_feature_names_out()):] = sc.transform(d.iloc[sl][NUM]).astype(np.float32)
        return out_

    A = design(tr_mask)
    g = TweedieRegressor(power=XI, link="log", alpha=1e-4, solver="newton-cholesky", max_iter=100, tol=1e-8)
    g.fit(A, rate[tr_mask], sample_weight=E[tr_mask])
    del A; gc.collect()
    Bm = design(te_mask)
    p = g.predict(Bm)
    del Bm, oh, sc, g; gc.collect()
    return p


out = {"design": "giu-lai-mot-bang", "xi": XI, "B": B_BOOT, "seed": SEED_B, "per_state": {}}
pred = {"EAGB": np.full(len(d), np.nan), "FLAT": np.full(len(d), np.nan), "GLM": np.full(len(d), np.nan)}
CKPT = "../results/ckpt_oos_%s.npz"
t00 = time.time()
for code, ab in ST.items():
    if os.path.exists(CKPT % ab):   # đã chạy xong bang này ở lần chạy trước
        ck = np.load(CKPT % ab, allow_pickle=True)
        te0 = TE & (st == code)
        for key in ("EAGB", "FLAT", "GLM"):
            pred[key][te0] = ck[key]
        out["per_state"][ab] = json.loads(str(ck["rec"]))
        print(ab, "(nạp lại từ điểm kiểm)", flush=True)
        continue
    keep = st != code                       # 7 bang huấn luyện
    tr = TR & keep                          # huấn luyện: 2012-2018 của 7 bang
    te = TE & (st == code)                  # đánh giá: 2021-2023 của bang giữ lại
    rec = {"n_train": int(tr.sum()), "n_eval": int(te.sum())}
    # --- bộ mã hóa chỉ lấy nguồn từ 7 bang
    z_sim, root_sim = enc_masked(T, node, KSEL["simul"], 3, keep)
    z_flat, _ = enc_masked(Tflat, nodeflat, KSEL["flat"], 1, keep)
    for nm, z, key in [("EAGB (HCE đồng thời)", z_sim, "EAGB"), ("GBDT + mã hóa phẳng co ngót (H1c)", z_flat, "FLAT")]:
        X = features(d, z)
        Xtr = X[tr]; Xte = align_cats(Xtr, X[te].copy())
        m = lgb.train(dict(LGB_BASE, objective="tweedie", tweedie_variance_power=XI),
                      lgb.Dataset(Xtr, rate[tr], weight=E[tr]), ROUNDS[nm])
        pred[key][te] = np.maximum(m.predict(Xte), 1e-6)
        del X, Xtr, Xte, m
    pred["GLM"][te] = np.maximum(glm_fit_predict(tr, te), 1e-6)
    # --- tỷ lệ nút của bang giữ lại không có nguồn (chẩn đoán khởi động nguội)
    rv = np.array([root_sim.get(int(y), np.nan) for y in d.year.values[te]])
    rec["ty_le_o_co_ngot_ve_goc"] = float(np.mean(np.isclose(z_sim[te], rv, rtol=1e-9)))
    for key in ("EAGB", "FLAT", "GLM"):
        rec[key] = evaluate(pred[key][te], E[te], S[te], XI_EVAL)
    dv = {k: tweedie_dev(S[te] / E[te], pred[k][te], XI_EVAL) for k in pred}
    rec["rel_EAGB_GLM"] = float((np.sum(E[te] * dv["EAGB"]) - np.sum(E[te] * dv["GLM"])) / np.sum(E[te] * dv["GLM"]))
    rec["rel_EAGB_FLAT"] = float((np.sum(E[te] * dv["EAGB"]) - np.sum(E[te] * dv["FLAT"])) / np.sum(E[te] * dv["FLAT"]))
    out["per_state"][ab] = rec
    np.savez_compressed(CKPT % ab, rec=json.dumps(rec, ensure_ascii=False, default=float),
                        **{k: pred[k][te] for k in ("EAGB", "FLAT", "GLM")})
    del z_sim, z_flat, rv, dv; gc.collect()
    print(ab, "n_eval", rec["n_eval"], "relGLM %.4f" % rec["rel_EAGB_GLM"], "relFLAT %.4f" % rec["rel_EAGB_FLAT"],
          "dev %.1f" % rec["EAGB"]["tweedie_dev"], flush=True)

# ---------- gộp: mỗi ô đánh giá đúng một lần bởi mô hình chưa thấy bang của nó
m = TE & ~np.isnan(pred["EAGB"])
Em, Sm = E[m], S[m]
cl = (st[m] + "-" + d.county.values.astype(str)[m])
dev = {k: tweedie_dev(Sm / Em, pred[k][m], XI_EVAL) for k in pred}
out["pooled"] = {"n": int(m.sum())}
for k in pred:
    out["pooled"][k] = evaluate(pred[k][m], Em, Sm, XI_EVAL)


def boot(a, b):
    uc, inv = np.unique(cl, return_inverse=True)
    na = np.bincount(inv, weights=Em * dev[a], minlength=len(uc))
    nb = np.bincount(inv, weights=Em * dev[b], minlength=len(uc))
    rng = np.random.default_rng(SEED_B); o = np.empty(B_BOOT)
    for i in range(B_BOOT):
        w = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc))
        o[i] = (w @ na - w @ nb) / (w @ nb)
    return dict(n_blocks=int(len(uc)), rel=float((na.sum() - nb.sum()) / nb.sum()),
                lo95=float(np.quantile(o, .025)), hi95=float(np.quantile(o, .975)),
                hi95_1s=float(np.quantile(o, .95)), lo95_1s=float(np.quantile(o, .05)))


out["pooled_boot"] = {"EAGB/GLM": boot("EAGB", "GLM"), "EAGB/FLAT": boot("EAGB", "FLAT")}
# ---------- đối chiếu với kết quả trong không gian (mô hình đã thấy mọi bang)
zin = np.load("../results/kb2_test_preds.npz")
pin = {"EAGB": np.maximum(zin["EAGB (HCE ng thi)0"], 1e-6), "GLM": np.maximum(zin["GLM Tweedie (i chng chnh)8"], 1e-6),
       "FLAT": np.maximum(zin["GBDT + m ha phng co ngt (H1c)2"], 1e-6)}
out["trong_khong_gian"] = {k: evaluate(pin[k], E[TE], S[TE], XI_EVAL) for k in pin}
out["suy_giam"] = {k: {"dev_ngoai": out["pooled"][k]["tweedie_dev"], "dev_trong": out["trong_khong_gian"][k]["tweedie_dev"],
                       "ty_le": out["pooled"][k]["tweedie_dev"] / out["trong_khong_gian"][k]["tweedie_dev"],
                       "gini_ngoai": out["pooled"][k]["gini"], "gini_trong": out["trong_khong_gian"][k]["gini"]} for k in pin}
out["sec_total"] = time.time() - t00
json.dump(out, open("../results/spatial_oos.json", "w"), indent=1, ensure_ascii=False, default=float)
np.savez_compressed("../results/spatial_oos_preds.npz", **{k: v[TE] for k, v in pred.items()})
print(json.dumps({k: out[k] for k in ("pooled_boot", "suy_giam")}, indent=1, ensure_ascii=False, default=float))
print("ok", out["sec_total"])
