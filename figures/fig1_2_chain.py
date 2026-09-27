"""Hình 1.2 — Chuỗi liên kết từ mục tiêu tới kết luận.

Chạy:  python3 fig1_2_chain.py
Ra:    output/fig1_2_chain.png

Sơ đồ khái niệm. Thêm/bớt mắt xích bằng cách sửa danh sách MAT_XICH;
bố cục tự giãn theo số phần tử.
"""
from _common import canvas, box, arrow, save, C

MAT_XICH = [
    ("Mục tiêu\nMT1–MT6", C["data"]),
    ("Dữ liệu\nTầng 1c, 2, 3c", C["data"]),
    ("Thuật toán\nHCE/EAGB, H-SHAP,\nAMORO", C["eagb"]),
    ("Kịch bản\nKB1–KB9", C["hshap"]),
    ("Kiểm định\nH1, H2, H3, RQ0\n(tiêu chí đã khóa)", C["eval"]),
    ("Kết luận\ntheo mức\nbằng chứng", C["amoro"]),
]

KHO = (11, 2.6)
CO_CHU = 8.5
Y_HOP, CAO_HOP = 0.2, 0.6

n = len(MAT_XICH)
buoc = 1.0 / n                       # bề rộng mỗi ô kể cả khoảng hở
rong_hop = buoc * 0.87               # phần còn lại dành cho mũi tên

fig, ax = canvas(*KHO)
for i, (text, fc) in enumerate(MAT_XICH):
    x = 0.005 + i * buoc
    box(ax, x, Y_HOP, rong_hop, CAO_HOP, text, fc, CO_CHU)
    if i < n - 1:
        y_mid = Y_HOP + CAO_HOP / 2
        arrow(ax, x + rong_hop, y_mid, x + buoc + 0.005, y_mid)
save(fig, "fig1_2_chain.png")
