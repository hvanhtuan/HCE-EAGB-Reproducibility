"""KB1 — đối chứng học sâu hiện đại bổ sung: Credibility Transformer (Richman, Scognamiglio &
Wüthrich, European Actuarial Journal, 2025; arXiv:2409.16653), TÁI CÀI ĐẶT ĐỘC LẬP theo
`credibility_transformer.py` (xem ghi chú bản quyền ở đầu tệp đó — không sao chép mã nguồn tác
giả, giấy phép CC BY-NC-ND của repo gốc không cho phép).

Đây là một đối chứng THĂM DÒ, chạy SAU KB1 (không tiền đăng ký), trên ĐÚNG cùng tập huấn
luyện/kiểm định freMTPL2 của `kb1_fremtpl2.py` (cùng SEED=20260918, cùng phép chia 90/10), nên
so sánh số với các mô hình khác trong KB1 (GLM, GBDT, CatBoost, EAGB, các biến thể mã hóa) là
so sánh HỢP LỆ về mặt tập kiểm định (cùng hàng dữ liệu), không phải so sánh liên nghiên cứu.

Cách chạy: mỗi lần gọi script huấn luyện MỘT thành viên của tập hợp (ensemble), hạt giống truyền
qua đối số dòng lệnh, và ghi log-mu dự báo trên tập kiểm định vào
`../results/kb1_ct_member_<seed>.npy`. Sau khi đã có đủ số thành viên, gọi lại script với đối số
"aggregate" để gộp (trung bình theo thang mu, không phải log-mu — theo đúng cách tập hợp dự báo
tần suất) và tính các độ đo + khoảng tin cậy bootstrap so với GLM/EAGB/GBDT phân loại nội tại.
"""
import sys, json, time, glob
import numpy as np, pandas as pd
from credibility_transformer import fit_predict_one
from metrics import poisson_dev, gini_conc, calib

SEED = 20260918
CAT_COLS = ["VehBrand", "VehGas", "Region", "Area"]
CONT_COLS = ["VehPower", "VehAge", "DrivAge"]  # + BonusMalus_log, Density_log thêm bên dưới
R = "../results/"


def load_split():
    f = pd.read_parquet("../data/raw/fremtpl2freq.pq")
    f["ClaimNb"] = np.minimum(f.ClaimNb.astype(int), 4); f["Exposure"] = np.minimum(f.Exposure, 1.0)
    for c in ["Area", "VehBrand", "Region", "VehGas"]:
        f[c] = f[c].astype(str)
    rng = np.random.default_rng(SEED)
    test = rng.random(len(f)) < 0.1
    tr, te = f[~test].reset_index(drop=True), f[test].reset_index(drop=True)
    val = np.random.default_rng(SEED + 2).random(len(tr)) < 0.15  # ĐÚNG mặt nạ val của kb1_fremtpl2.py
    return tr, te, val


def build_features(tr, te, val):
    vocab = {c: {v: i for i, v in enumerate(sorted(set(tr[c]) | set(te[c])))} for c in CAT_COLS}

    def catmat(d):
        return np.stack([d[c].map(vocab[c]).values for c in CAT_COLS], axis=1).astype("int64")

    def contraw(d):
        X = d[CONT_COLS].astype(float).values.copy()
        return np.c_[X, np.log(np.minimum(d.BonusMalus, 150)), np.log(d.Density)]

    Xc_fit = contraw(tr[~val])
    mu, sd = Xc_fit.mean(0), Xc_fit.std(0)

    def contmat(d):
        return ((contraw(d) - mu) / sd).astype("float32")

    out = dict(
        Xcat_tr=catmat(tr[~val]), Xcont_tr=contmat(tr[~val]),
        y_tr=tr.ClaimNb.values[~val].astype("float32"), v_tr=tr.Exposure.values[~val].astype("float32"),
        Xcat_va=catmat(tr[val]), Xcont_va=contmat(tr[val]),
        y_va=tr.ClaimNb.values[val].astype("float32"), v_va=tr.Exposure.values[val].astype("float32"),
        Xcat_te=catmat(te), Xcont_te=contmat(te),
        y_te=te.ClaimNb.values.astype("float32"), v_te=te.Exposure.values.astype("float32"),
        cat_cardinalities=[len(vocab[c]) for c in CAT_COLS], n_continuous=len(CONT_COLS) + 2,
    )
    return out


def cmd_train_member(seed, max_epochs, patience):
    tr, te, val = load_split()
    d = build_features(tr, te, val)
    t0 = time.time()
    log_mu, meta = fit_predict_one(
        d["Xcat_tr"], d["Xcont_tr"], d["y_tr"], d["v_tr"],
        d["Xcat_va"], d["Xcont_va"], d["y_va"], d["v_va"],
        d["Xcat_te"], d["Xcont_te"],
        d["cat_cardinalities"], d["n_continuous"], seed=seed,
        max_epochs=max_epochs, patience=patience)
    sec = time.time() - t0
    np.save(f"{R}kb1_ct_member_{seed}.npy", log_mu)
    meta["seed"] = seed; meta["train_sec"] = sec
    json.dump(meta, open(f"{R}kb1_ct_member_{seed}.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


def cmd_aggregate():
    tr, te, val = load_split()
    yN_te, E_te = te.ClaimNb.values.astype(float), te.Exposure.values
    member_files = sorted(glob.glob(f"{R}kb1_ct_member_*.npy"))
    seeds = [int(f.split("_")[-1].split(".")[0]) for f in member_files]
    log_mus = np.stack([np.load(f) for f in member_files])  # (n_members, n_test)
    mus = np.exp(log_mus)  # gộp ở thang mu (tần suất), không phải log-mu
    mu_ens = mus.mean(axis=0)
    metas = [json.load(open(f"{R}kb1_ct_member_{s}.json")) for s in seeds]

    dev = poisson_dev(yN_te, mu_ens * E_te)
    res_ct = dict(poisson_dev_x100=float(100 * dev.mean()), gini=gini_conc(mu_ens, E_te, yN_te),
                 **{k: v for k, v in calib(mu_ens, E_te, yN_te).items() if k != "dec_OE"},
                 n_members=len(seeds), seeds=seeds,
                 train_sec_per_member=[m["train_sec"] for m in metas],
                 epochs_run_per_member=[m["epochs_run"] for m in metas],
                 n_params=metas[0]["n_params"])

    # so sánh với các đối chứng đã có trong KB1 (ĐÚNG cùng tập kiểm định — xem kb1_preds.npz)
    idx = json.load(open(f"{R}kb1_preds_index.json"))
    npz = np.load(f"{R}kb1_preds.npz")
    boot = {}
    rng = np.random.default_rng(SEED + 9); B = 1000; n = len(te)
    d_ct = poisson_dev(yN_te, mu_ens * E_te)
    for base_name in ["GLM Poisson", "EAGB (HCE đồng thời)", "GBDT phân loại nội tại", "CatBoost (OTS)"]:
        p_base = npz[idx[base_name]]
        d_base = poisson_dev(yN_te, p_base * E_te)
        diff = d_ct - d_base
        bs = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(B)]) * 100
        boot[f"Credibility Transformer || {base_name}"] = dict(
            diff_x100=float(100 * diff.mean()), CI95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])

    out = dict(model="Credibility Transformer (tái cài đặt độc lập, Richman-Scognamiglio-Wüthrich 2025)",
              results=res_ct, boot_vs_kb1=boot,
              ghi_chu="Thăm dò, chạy SAU KB1, không tiền đăng ký. Cùng tập kiểm định freMTPL2 với "
                      "kb1_fremtpl2.py (SEED=20260918) nên so sánh số với các mô hình KB1 là hợp lệ.")
    json.dump(out, open(f"{R}kb1_credibility_transformer.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1] == "aggregate":
        cmd_aggregate()
    else:
        seed = int(sys.argv[1]); max_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
        patience = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        cmd_train_member(seed, max_epochs, patience)
