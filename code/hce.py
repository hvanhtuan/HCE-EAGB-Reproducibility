"""Mã tham chiếu HCE (Chuyên đề Tiến sĩ 1, Định nghĩa 3.3, Giải thuật 3.1, 3.2, Định nghĩa 3.4).
Cây được biểu diễn bằng mảng: parent[v] (gốc = -1), depth[v]; nút 0 là gốc.
"""
import numpy as np
from fractions import Fraction


class Tree:
    def __init__(self, paths):
        """paths: danh sách tuple nhãn từ cấp 1 đến lá, ví dụ (state, county, tract).
        Trả về cây gốc 'ROOT' với mọi tiền tố là nút."""
        self.index = {(): 0}
        parent = [-1]; depth = [0]; label = [()]
        for p in paths:
            for l in range(1, len(p) + 1):
                key = tuple(p[:l])
                if key not in self.index:
                    self.index[key] = len(parent)
                    parent.append(self.index[key[:-1]]); depth.append(l); label.append(key)
        self.parent = np.array(parent); self.depth = np.array(depth); self.label = label
        self.n = len(parent); self.L = int(self.depth.max())
        # thứ tự tiền duyệt: sắp theo độ sâu là đủ vì cha luôn có độ sâu nhỏ hơn con
        self.pre = np.argsort(self.depth, kind="stable")
        self.post = self.pre[::-1]

    def node_of(self, path):
        """Nút sâu nhất có mặt trong cây (lùi về tổ tiên gần nhất nếu vắng)."""
        p = tuple(path)
        while p not in self.index:
            p = p[:-1]
        return self.index[p]

    def ancestors_at(self, v, level):
        while self.depth[v] > level:
            v = self.parent[v]
        return v


def node_stats(tree, node_ids, E, S):
    Ev = np.bincount(node_ids, weights=E, minlength=tree.n)
    Sv = np.bincount(node_ids, weights=S, minlength=tree.n)
    return Ev, Sv


def hce_solve(tree, Ev, Sv, k, k0, mu0, fixed_root=False, exact=False):
    """Giải thuật 3.1 HCE-SOLVE. k: mảng trọng số theo cấp, k[l] cho cạnh vào nút độ sâu l (l>=1).
    fixed_root=True: biến thể HCE-FIXED-ROOT (Hệ quả 3.1). exact=True: số học hữu tỷ (Fraction)."""
    par, dep = tree.parent, tree.depth
    if exact:
        a = [Fraction(x) for x in Ev]; b = [Fraction(x) for x in Sv]
        kk = [Fraction(x) for x in k]; k0 = Fraction(k0); mu0 = Fraction(mu0)
    else:
        a = np.array(Ev, dtype=float).copy(); b = np.array(Sv, dtype=float).copy(); kk = np.asarray(k, float)
    if not fixed_root:
        if not (k0 > 0 or np.sum(np.asarray(Ev, float)) > 0):
            raise ValueError("A4 vi phạm: không nút nào có phơi nhiễm dương và k0=0")
        a[0] = a[0] + k0; b[0] = b[0] + k0 * mu0
    for v in tree.post:
        if v == 0:
            continue
        u = par[v]; kv = kk[dep[v]]
        den = a[v] + kv
        a[u] = a[u] + kv * a[v] / den
        b[u] = b[u] + kv * b[v] / den
    theta = [None] * tree.n if exact else np.empty(tree.n)
    theta[0] = mu0 if fixed_root else b[0] / a[0]
    for v in tree.pre:
        if v == 0:
            continue
        kv = kk[dep[v]]
        theta[v] = (b[v] + kv * theta[par[v]]) / (a[v] + kv)
    return theta


def hce_objective(tree, Ev, Sv, theta, k, k0, mu0, S2E=None):
    """J(θ) của Định nghĩa 3.3, bỏ hằng số Σ S_i^2/E_i nếu S2E=None."""
    th = np.asarray(theta, dtype=object if isinstance(theta[0], Fraction) else float)
    J = 0
    for v in range(tree.n):
        J += Ev[v] * th[v] ** 2 - 2 * Sv[v] * th[v]
        if v > 0:
            J += k[tree.depth[v]] * (th[v] - th[tree.parent[v]]) ** 2
    J += k0 * (th[0] - mu0) ** 2
    if S2E is not None:
        J += S2E
    return J


def hce_gradient(tree, Ev, Sv, theta, k, k0, mu0):
    th = np.asarray(theta, float); g = 2 * (np.asarray(Ev) * th - np.asarray(Sv))
    for v in range(1, tree.n):
        u = tree.parent[v]; d = 2 * k[tree.depth[v]] * (th[v] - th[u])
        g[v] += d; g[u] -= d
    g[0] += 2 * k0 * (th[0] - mu0)
    return g


def hce_dense(tree, Ev, Sv, k, k0, mu0):
    """Bộ giải đối chứng độc lập: lập ma trận H và giải trực tiếp (dùng cho TK1)."""
    n = tree.n; H = np.diag(np.asarray(Ev, float)); b = np.asarray(Sv, float).copy()
    for v in range(1, n):
        u = tree.parent[v]; kv = k[tree.depth[v]]
        H[v, v] += kv; H[u, u] += kv; H[u, v] -= kv; H[v, u] -= kv
    H[0, 0] += k0; b[0] += k0 * mu0
    return np.linalg.solve(H, b), H


def hce_sequential(tree, Ev, Sv, k, k0, mu0):
    """Định nghĩa 3.4: co ngót tuần tự từ trên xuống với thống kê cộng dồn cây con."""
    Et = np.array(Ev, float).copy(); St = np.array(Sv, float).copy()
    for v in tree.post:
        if v:
            Et[tree.parent[v]] += Et[v]; St[tree.parent[v]] += St[v]
    th = np.empty(tree.n); th[0] = (St[0] + k0 * mu0) / (Et[0] + k0)
    for v in tree.pre:
        if v:
            kv = k[tree.depth[v]]
            th[v] = (St[v] + kv * th[tree.parent[v]]) / (Et[v] + kv)
    return th
