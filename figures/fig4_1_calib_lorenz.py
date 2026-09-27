"""Hình 4.1 — Hiệu chuẩn theo thập phân vị (trái) và đường Lorenz tập trung (phải),
tập kiểm định 2021–2023.

Chạy:  python3 fig4_1_calib_lorenz.py
Vào:   data/fig4_1_deciles.csv, data/fig4_1_lorenz.json
Ra:    output/fig4_1_calib_lorenz.png

Dữ liệu đã được tính sẵn từ dự báo thật của KB2 (xem code/export_figdata.py).
Muốn cập nhật số liệu thì chạy lại export_figdata.py; muốn đổi hình thức thì sửa ở đây.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"

# ---- Chọn mô hình hiển thị và thứ tự đường ---------------------------------------
MO_HINH = ["EAGB", "GLM Tweedie", "Mã hóa phẳng", "EAGB-PG"]
MAU = {"EAGB": "#1f77b4", "GLM Tweedie": "#ff7f0e",
       "Mã hóa phẳng": "#2ca02c", "EAGB-PG": "#d62728"}

KHO = (11, 4.2)
DPI = 200
NHAN_TRAI_X = "Thập phân vị phơi nhiễm theo phí thuần dự báo"
NHAN_TRAI_Y = "log10(quan sát/dự báo)"
NHAN_PHAI_X = "Tỷ lệ phơi nhiễm tích lũy (giảm dần theo dự báo)"
NHAN_PHAI_Y = "Tỷ lệ tổn thất tích lũy"

dec = pd.read_csv("data/fig4_1_deciles.csv")
lor = json.load(open("data/fig4_1_lorenz.json", encoding="utf-8"))

fig, ax = plt.subplots(1, 2, figsize=KHO)
for m in MO_HINH:
    g = dec[dec.model == m].sort_values("decile")
    ax[0].plot(g.decile, g.log10_ratio, marker="o", label=m, color=MAU.get(m))
    ax[1].plot(lor[m]["x"], lor[m]["y"], label=m, color=MAU.get(m))

ax[0].axhline(0, color="k", lw=.8)
ax[0].set_xlabel(NHAN_TRAI_X)
ax[0].set_ylabel(NHAN_TRAI_Y)
ax[0].legend(fontsize=8)

ax[1].plot([0, 1], [0, 1], "k--", lw=.8)          # đường chéo: không phân tách
ax[1].set_xlabel(NHAN_PHAI_X)
ax[1].set_ylabel(NHAN_PHAI_Y)
ax[1].legend(fontsize=8)

plt.tight_layout()
import os; os.makedirs("output", exist_ok=True)
plt.savefig("output/fig4_1_calib_lorenz.png", dpi=DPI)
print("da ghi output/fig4_1_calib_lorenz.png")
