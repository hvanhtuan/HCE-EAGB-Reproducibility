"""Tiện ích dùng chung cho các script vẽ sơ đồ (Hình 1.1, 1.2, 2.1, 2.2).

Các script vẽ biểu đồ số liệu (Hình 4.1–4.3) KHÔNG cần tệp này.
Sửa bảng màu ở đây để đổi màu đồng loạt cho cả bốn sơ đồ.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Phông DejaVu Sans có đủ dấu tiếng Việt. Đổi sang "Times New Roman" nếu máy có phông đó
# và muốn hình khớp phông thân bài.
plt.rcParams["font.family"] = "DejaVu Sans"

OUT = "output/"          # thư mục ghi tệp PNG
DPI = 200                # tăng lên 300 nếu cần in chất lượng cao

# Bảng màu theo hợp phần
C = {
    "eagb":  "#dbe8f6",   # xanh nhạt  — EAGB / huấn luyện
    "hshap": "#d7f0ec",   # lục nhạt   — H-SHAP
    "amoro": "#fde6d2",   # cam nhạt   — AMORO / kiểm định
    "data":  "#eeeeee",   # xám nhạt   — dữ liệu
    "eval":  "#f3e8fa",   # tím nhạt   — đánh giá / thí điểm
    "edge":  "#333333",   # màu viền và mũi tên
}


def canvas(w=10, h=5):
    """Khung vẽ toạ độ 0–1 theo cả hai chiều, đã tắt trục."""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc, fs=9, bold=False):
    """Hộp bo góc có chữ ở giữa. (x, y) là góc dưới trái, theo toạ độ 0–1."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.02",
                                fc=fc, ec=C["edge"], lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, wrap=True, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, text=None, fs=7.5, style="-|>", ls="-"):
    """Mũi tên từ (x1, y1) tới (x2, y2), kèm nhãn tuỳ chọn đặt ở giữa."""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, lw=1, color=C["edge"], linestyle=ls))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.015, text,
                ha="center", va="bottom", fontsize=fs, color="#444")


def save(fig, name):
    import os
    os.makedirs(OUT, exist_ok=True)
    path = OUT + name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("da ghi", path)
