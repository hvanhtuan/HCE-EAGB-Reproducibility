"""Hình 4.2 — Tầm quan trọng toàn cục: cấp đặc trưng (trái) và cấp nhóm (phải),
mô hình EAGB, 300 ô năm 2021–2023.

Chạy:  python3 fig4_2_importance.py
Vào:   data/fig4_2_importance.json
Ra:    output/fig4_2_importance.png

Giá trị là trung bình |đóng góp| trên thang log. Cột xanh là giá trị Owen (H-SHAP),
cột cam là giá trị Shapley — hai cột gần trùng nhau chính là kết quả cần thấy.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"

# ---- Tên hiển thị cho 12 đặc trưng và 3 nhóm -------------------------------------
TEN_DAC_TRUNG = {
    "geo": "địa lý (tract)", "cyear": "năm xây dựng", "logcov": "log STBH",
    "crs": "hạng CRS", "ded": "mức khấu trừ", "zone": "vùng lũ",
    "occ": "mục đích SD", "prim": "nơi ở chính", "cov": "nhóm STBH",
    "elev": "nâng tầng", "floors": "số tầng", "pfirm": "sau FIRM",
}
TEN_NHOM = ["Hiểm họa & vị trí", "Công trình", "Hợp đồng & giá trị"]

MAU_OWEN, MAU_SHAP = "#1f77b4", "#ff7f0e"
KHO = (11, 4.2)
DPI = 200
RONG_COT = 0.4
XOAY_NHAN = 35          # độ xoay nhãn trục x
NHAN_Y = "trung bình |đóng góp| (thang log)"

gi = json.load(open("data/fig4_2_importance.json", encoding="utf-8"))
ow_f, sh_f = gi["owen_feature"], gi["shap_feature"]
keys = sorted(ow_f, key=lambda k: -ow_f[k])          # sắp giảm dần theo Owen
labels = [TEN_DAC_TRUNG.get(k, k) for k in keys]

fig, ax = plt.subplots(1, 2, figsize=KHO, gridspec_kw={"width_ratios": [2.1, 1]})

x = np.arange(len(keys))
ax[0].bar(x - RONG_COT / 2, [ow_f[k] for k in keys], RONG_COT, label="H-SHAP (Owen) ψ", color=MAU_OWEN)
ax[0].bar(x + RONG_COT / 2, [sh_f[k] for k in keys], RONG_COT, label="SHAP φ", color=MAU_SHAP)
ax[0].set_xticks(x)
ax[0].set_xticklabels(labels, rotation=XOAY_NHAN, ha="right")
ax[0].set_ylabel(NHAN_Y)
ax[0].legend(fontsize=8)

xg = np.arange(len(TEN_NHOM))
ax[1].bar(xg - RONG_COT / 2, gi["owen_group"], RONG_COT, label="Φ_g (Owen)", color=MAU_OWEN)
ax[1].bar(xg + RONG_COT / 2, gi["shap_group_sum"], RONG_COT, label="Σ φ_j trong nhóm", color=MAU_SHAP)
ax[1].set_xticks(xg)
ax[1].set_xticklabels(TEN_NHOM, rotation=XOAY_NHAN, ha="right")
ax[1].legend(fontsize=8)

plt.tight_layout()
import os; os.makedirs("output", exist_ok=True)
plt.savefig("output/fig4_2_importance.png", dpi=DPI)
print("da ghi output/fig4_2_importance.png")
