"""Hình 2.1 — Kiến trúc tổng thể của khung EAGB → H-SHAP → AMORO.

Chạy:  python3 fig2_1_architecture.py
Ra:    output/fig2_1_architecture.png

Sơ đồ khái niệm. Mỗi hộp là một mục trong HOP: (x, y, rộng, cao, chữ, màu, cỡ chữ).
Toạ độ theo hệ 0–1, gốc ở góc dưới trái. Mỗi mũi tên là một mục trong MUI_TEN.
"""
from _common import canvas, box, arrow, save, C

# ---- Các hộp ---------------------------------------------------------------------
HOP = [
    (0.025, 0.62, 0.185, 0.30,
     "Dữ liệu ô rủi ro × năm\n(x, E, S, N, năm, tract)\n+ họ cây phạm trù\nbang→quận→tract\n"
     "+ mốc nhãn khả dụng a = năm + 2", C["data"], 8),
    (0.270, 0.62, 0.220, 0.30,
     "EAGB\n① HCE-FIT theo khối năm\n(nguồn: a_j ≤ τ_κ)\n② GBDT Tweedie hoặc\nPoisson–Gamma trên (x, z)\n"
     "③ dự báo π̂, ŝ = E·π̂", C["eagb"], 8),
    (0.550, 0.62, 0.200, 0.30,
     "Hiệu chuẩn hợp quy\nMondrian theo vùng lũ\nŝᵁ = ŝ·max(1, q̂_g)", C["eval"], 8),
    (0.800, 0.62, 0.185, 0.30,
     "AMORO\nNSGA-II (loại bỏ vi phạm)\nF1 lợi nhuận, F2 tái tục,\nF3 bất ổn phí; ràng buộc\n"
     "hộp–ổn định–tỷ lệ tổn thất\n→ tập P̂ + nhãn chứng nhận", C["amoro"], 7.5),
    (0.270, 0.12, 0.300, 0.33,
     "H-SHAP\ntrò chơi ν_x trên thang log;\ntoán tử che: giá trị tham chiếu,\nđịa lý → nút gốc;\n"
     "Owen 2 tầng (3 nhóm)\n→ ψ_j, Φ_g, phân rã tần suất–mức độ,\nchỉ số ổn định, danh sách ứng viên",
     C["hshap"], 7.5),
    (0.660, 0.12, 0.320, 0.33,
     "Hồ sơ kiểm toán & đánh giá\nđộ lệch, Gini, hiệu chuẩn; độ phủ;\nhiệu suất I3; ổn định; HV;\n"
     "nhật ký bất biến I1–I4\n(đánh giá chuyên gia: thiết kế)", C["eval"], 7.5),
]

# ---- Các mũi tên: (x1, y1, x2, y2, nhãn) -----------------------------------------
MUI_TEN = [
    (0.21, 0.77, 0.27, 0.77, "S, E, cây"),
    (0.49, 0.77, 0.55, 0.77, "ŝ"),
    (0.75, 0.77, 0.80, 0.77, "ŝ, ŝᵁ"),
    (0.38, 0.62, 0.42, 0.45, "F (mô hình)"),
    (0.57, 0.28, 0.66, 0.28, "ψ, Φ"),
    (0.90, 0.62, 0.86, 0.45, "P̂"),
]

CHU_THICH_DUOI = ("Không có đường phản hồi từ H-SHAP/AMORO về EAGB "
                  "(thiết kế một chiều, CĐTS1 Mục 3.2.1)")

fig, ax = canvas(11, 6)
for x, y, w, h, text, fc, fs in HOP:
    box(ax, x, y, w, h, text, fc, fs)
for x1, y1, x2, y2, lab in MUI_TEN:
    arrow(ax, x1, y1, x2, y2, lab)
ax.text(0.5, 0.02, CHU_THICH_DUOI, ha="center", fontsize=8, style="italic")
save(fig, "fig2_1_architecture.png")
