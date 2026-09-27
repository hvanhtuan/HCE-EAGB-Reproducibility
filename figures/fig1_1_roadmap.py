"""Hình 1.1 — Lộ trình bốn giai đoạn của luận án.

Chạy:  python3 fig1_1_roadmap.py      (từ trong thư mục Figures/)
Ra:    output/fig1_1_roadmap.png

Sơ đồ khái niệm, không dùng dữ liệu thực nghiệm. Sửa nội dung ở danh sách GIAI_DOAN.
"""
from _common import canvas, box, arrow, save, C

# ---- Nội dung bốn hộp. Sửa chữ ở đây; "\n" xuống dòng trong hộp. ------------------
GIAI_DOAN = [
    ("Giai đoạn 1\n(tháng 1–12)\nTổng quan PRISMA,\nđề cương, G1–G4", C["data"]),
    ("Giai đoạn 2\n(tháng 13–24)\nCĐTS1: đặc tả HCE,\nEAGB, H-SHAP, AMORO;\nđịnh lý; TK1–TK7 (thiết kế)", C["eagb"]),
    ("Giai đoạn 3\n(tháng 25–30)\nCĐTS2: dữ liệu, mã tham chiếu,\nTK1–TK7 (thực hiện), KB1–KB9,\nH1–H3, RQ0 sơ bộ", C["amoro"]),
    ("Giai đoạn 4\n(tháng 31–36)\nKiểm định RQ0 chính thức,\ntoàn văn luận án", C["data"]),
]
GIAI_DOAN_NOI_BAT = 2          # chỉ số hộp được in đậm (0-based); đặt None để bỏ in đậm

CHU_THICH_DUOI = ("Sản phẩm chuyển giai đoạn: đặc tả + bảng tuyên bố (GĐ2) → "
                  "mã tham chiếu + dữ liệu chuẩn hóa + kết quả kiểm định (GĐ3) → luận án (GĐ4)")

# ---- Bố cục ----------------------------------------------------------------------
KHO = (10, 3.2)        # (rộng, cao) inch
BUOC_X = 0.25          # khoảng cách giữa hai hộp liên tiếp
RONG_HOP, CAO_HOP = 0.22, 0.65
Y_HOP = 0.2
CO_CHU = 8.5

fig, ax = canvas(*KHO)
for i, (text, fc) in enumerate(GIAI_DOAN):
    x = 0.01 + i * BUOC_X
    box(ax, x, Y_HOP, RONG_HOP, CAO_HOP, text, fc, CO_CHU, bold=(i == GIAI_DOAN_NOI_BAT))
    if i < len(GIAI_DOAN) - 1:
        y_mid = Y_HOP + CAO_HOP / 2
        arrow(ax, x + RONG_HOP + 0.00, y_mid, x + BUOC_X + 0.01, y_mid)

ax.text(0.5, 0.05, CHU_THICH_DUOI, ha="center", fontsize=8)
save(fig, "fig1_1_roadmap.png")
