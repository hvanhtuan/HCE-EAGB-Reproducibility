"""Hình 2.2 — Giao thức thời gian: vai trò của từng năm và nguồn của bộ mã hóa.

Chạy:  python3 fig2_2_temporal.py
Ra:    output/fig2_2_temporal.png

Sơ đồ khái niệm. Đổi các mốc chia dữ liệu ở VAI_TRO là đủ; màu và chú giải
tự suy ra từ đó.
"""
from matplotlib.patches import Patch
from _common import canvas, box, arrow, save, C

NAM_DAU, NAM_CUOI = 2010, 2024
LAG = 2                     # độ trễ nhãn (năm); khối năm t chỉ dùng nguồn có năm ≤ t − LAG
NAM_MINH_HOA = 2017         # năm được vẽ mũi tên "nguồn cho t = ..."

# (nhãn chú giải, màu, năm đầu, năm cuối) — thứ tự quyết định thứ tự chú giải
VAI_TRO = [
    ("chỉ làm nguồn (khối không hợp lệ)", "#dddddd", 2010, 2011),
    ("huấn luyện 2012–2018",               C["eagb"], 2012, 2018),
    ("thí điểm/hiệu chỉnh 2019–2020",      C["eval"], 2019, 2020),
    ("kiểm định khóa 2021–2023",           C["amoro"], 2021, 2023),
    ("dịch chuyển 2024",                   "#f5f5f5", 2024, 2024),
]

TIEU_DE = (f"Nguồn mã hóa cho khối năm t = mọi ô có năm ≤ t − {LAG} (nhãn khả dụng); "
           "khối năm t không bao giờ nằm trong nguồn của chính nó")

# ---- Bố cục ----------------------------------------------------------------------
X0, BUOC_X = 0.03, 0.063
RONG_O, CAO_O, Y_O = 0.055, 0.2, 0.55

def mau_cua(nam):
    for _, fc, a, b in VAI_TRO:
        if a <= nam <= b:
            return fc
    return "#ffffff"

fig, ax = canvas(11, 3.4)
nams = list(range(NAM_DAU, NAM_CUOI + 1))
for i, y in enumerate(nams):
    box(ax, X0 + i * BUOC_X, Y_O, RONG_O, CAO_O, str(y), mau_cua(y), 8)

ax.text(X0, 0.82, TIEU_DE, fontsize=8.5)

# mũi tên minh họa: từ năm (NAM_MINH_HOA − LAG) tới NAM_MINH_HOA
i_src = nams.index(NAM_MINH_HOA - LAG)
i_dst = nams.index(NAM_MINH_HOA)
x_src = X0 + i_src * BUOC_X + RONG_O / 2
x_dst = X0 + i_dst * BUOC_X + RONG_O / 2
arrow(ax, x_src, 0.46, x_dst, 0.46)
ax.text((x_src + x_dst) / 2, 0.40, f"nguồn cho t={NAM_MINH_HOA}",
        ha="center", va="top", fontsize=7.5, color="#444")

# chú giải: dùng legend của matplotlib để tránh chữ chồng lên nhau
ax.legend(handles=[Patch(facecolor=fc, edgecolor=C["edge"], label=lab) for lab, fc, _, _ in VAI_TRO],
          loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=8,
          handlelength=1.6, handleheight=1.0, columnspacing=1.6)
save(fig, "fig2_2_temporal.png")
