"""Hình 4.3 — Mặt trội do NSGA-II sinh ra cho AMORO danh nghĩa, hạt giống 11:
lợi nhuận kỳ vọng, tái tục kỳ vọng và bất ổn phí F3 (màu).

Chạy:  python3 fig4_3_pareto.py
Vào:   data/fig4_3_pareto.csv
Ra:    output/fig4_3_pareto.png

Mỗi điểm là một nghiệm không bị trội. Muốn vẽ hạt giống khác thì sửa
code/export_figdata.py để xuất từ tệp amoro_front_nominal_<seed>.npz tương ứng.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"

NAM = [2021, 2022, 2023]        # mỗi năm một khung con
BANG_MAU = "viridis"            # bảng màu cho F3
KHO = (11, 3.6)
DPI = 200
CO_DIEM = 26
NHAN_X = "lợi nhuận kỳ vọng (triệu USD)"
NHAN_Y = "tỷ lệ tái tục kỳ vọng"
NHAN_MAU = "F3 (bất ổn phí)"

df = pd.read_csv("data/fig4_3_pareto.csv")

fig, ax = plt.subplots(1, len(NAM), figsize=KHO)
for k, y in enumerate(NAM):
    g = df[df.year == y]
    sc = ax[k].scatter(g.profit_musd, g.retention, c=g.F3, cmap=BANG_MAU, s=CO_DIEM)
    ax[k].set_title(str(y))
    ax[k].set_xlabel(NHAN_X)
    if k == 0:
        ax[k].set_ylabel(NHAN_Y)
    cb = fig.colorbar(sc, ax=ax[k])
    cb.set_label(NHAN_MAU, fontsize=8)

plt.tight_layout()
import os; os.makedirs("output", exist_ok=True)
plt.savefig("output/fig4_3_pareto.png", dpi=DPI)
print("da ghi output/fig4_3_pareto.png")
