"""Độ đo đánh giá (Mục 3.4 của CĐTS2): độ lệch Tweedie/Poisson/Gamma, Gini có trọng số phơi nhiễm, hiệu chuẩn, bootstrap cụm."""
import numpy as np


def tweedie_dev(y, mu, xi):
    """Độ lệch đơn vị Tweedie d_xi(y, mu) (xi ∉ {0,1,2}); y>=0, mu>0."""
    y = np.asarray(y, float); mu = np.asarray(mu, float)
    return 2 * (np.power(np.maximum(y, 0), 2 - xi) / ((1 - xi) * (2 - xi)) - y * np.power(mu, 1 - xi) / (1 - xi) + np.power(mu, 2 - xi) / (2 - xi))


def poisson_dev(y, mu):
    y = np.asarray(y, float); mu = np.asarray(mu, float)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * (t - (y - mu))


def gamma_dev(y, mu):
    return 2 * (-np.log(y / mu) + (y - mu) / mu)


def w_mean(x, w):
    return float(np.sum(w * x) / np.sum(w))


def gini_conc(pred_rate, E, S):
    """Gini tập trung: sắp xếp giảm dần theo phí thuần dự báo; trục x = tỷ lệ phơi nhiễm tích lũy,
    trục y = tỷ lệ tổn thất tích lũy; Gini = 2·AUC − 1 (0: không phân tách; lớn hơn: phân tách tốt hơn)."""
    # xử lý giá trị trùng: gộp các bản ghi có cùng dự báo thành một bước (nội suy tuyến tính trong nhóm trùng)
    u, inv = np.unique(-np.asarray(pred_rate, float), return_inverse=True)
    Eg = np.bincount(inv, weights=E, minlength=len(u)); Sg = np.bincount(inv, weights=S, minlength=len(u))
    x = np.r_[0, np.cumsum(Eg)] / E.sum(); y = np.r_[0, np.cumsum(Sg)] / S.sum()
    return float(2 * np.trapezoid(y, x) - 1)


def calib(pred_rate, E, S, n_bins=10):
    """Trả về O/E toàn cục (calibration-in-the-large), độ dốc hiệu chuẩn và max |O/E−1| theo thập phân vị phơi nhiễm."""
    s_hat = pred_rate * E
    oe = S.sum() / s_hat.sum()
    o = np.argsort(pred_rate, kind="stable"); cE = np.cumsum(E[o]) / E.sum()
    b = np.minimum((cE * n_bins).astype(int), n_bins - 1)
    dec = np.array([S[o][b == k].sum() / s_hat[o][b == k].sum() for k in range(n_bins)])
    # độ dốc hiệu chuẩn: hồi quy tỷ lệ O/E theo log dự báo trên thang log (bình phương tối thiểu có trọng số theo ŝ ở cấp thập phân vị)
    lp = np.array([np.log(s_hat[o][b == k].sum() / E[o][b == k].sum()) for k in range(n_bins)])
    lo = np.log(np.maximum(np.array([S[o][b == k].sum() / E[o][b == k].sum() for k in range(n_bins)]), 1e-12))
    wk = np.array([s_hat[o][b == k].sum() for k in range(n_bins)])
    X = np.c_[np.ones(n_bins), lp]; W = np.diag(wk)
    slope = np.nan if np.std(lp) < 1e-9 else float(np.linalg.solve(X.T @ W @ X, X.T @ W @ lo)[1])
    return dict(OE=float(oe), slope=slope, max_dec_dev=float(np.max(np.abs(dec - 1))), dec_OE=dec.tolist())


def cluster_boot_diff(stat_fn, clusters, B=1000, seed=0):
    """Bootstrap theo cụm: stat_fn(idx) -> giá trị; trả về mẫu bootstrap."""
    rng = np.random.default_rng(seed)
    uc, inv = np.unique(clusters, return_inverse=True)
    members = [np.where(inv == c)[0] for c in range(len(uc))]
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uc), len(uc))
        idx = np.concatenate([members[c] for c in pick])
        out[b] = stat_fn(idx)
    return out
