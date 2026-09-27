# Figures — mã nguồn vẽ hình của Chuyên đề Tiến sĩ 2

Mỗi hình trong chuyên đề có đúng một script `.py` tương ứng. Sửa script, chạy lại, hình mới
xuất hiện trong `output/`.

## Danh sách

| Hình trong chuyên đề | Script | Nguồn dữ liệu |
|---|---|---|
| Hình 1.1 — Lộ trình bốn giai đoạn | `fig1_1_roadmap.py` | sơ đồ khái niệm (không dùng dữ liệu) |
| Hình 1.2 — Chuỗi liên kết mục tiêu → kết luận | `fig1_2_chain.py` | sơ đồ khái niệm |
| Hình 2.1 — Kiến trúc tổng thể EAGB → H-SHAP → AMORO | `fig2_1_architecture.py` | sơ đồ khái niệm |
| Hình 2.2 — Giao thức thời gian và nguồn bộ mã hóa | `fig2_2_temporal.py` | sơ đồ khái niệm |
| Hình 4.1 — Hiệu chuẩn thập phân vị và đường Lorenz | `fig4_1_calib_lorenz.py` | `data/fig4_1_deciles.csv`, `data/fig4_1_lorenz.json` |
| Hình 4.2 — Tầm quan trọng toàn cục (Owen vs Shapley) | `fig4_2_importance.py` | `data/fig4_2_importance.json` |
| Hình 4.3 — Mặt trội AMORO theo năm | `fig4_3_pareto.py` | `data/fig4_3_pareto.csv` |

Bốn sơ đồ khái niệm dùng chung `_common.py` (bảng màu, hàm vẽ hộp và mũi tên).
Ba biểu đồ số liệu độc lập hoàn toàn, chỉ cần `matplotlib` và `pandas`.

## Cách chạy

Mở terminal **trong thư mục `Figures`** rồi:

```
python3 fig4_2_importance.py      # vẽ lại một hình
python3 build_all.py              # vẽ lại cả bảy hình vào output/
python3 build_all.py --copy       # vẽ lại rồi chép đè vào ../doc/figs/ của chuyên đề
```

Yêu cầu: Python 3, `matplotlib`, `pandas`. Cài nếu thiếu: `pip install matplotlib pandas`.

## Sửa gì ở đâu

**Bốn sơ đồ khái niệm** — mọi nội dung nằm ở các hằng viết HOA gần đầu tệp:

- `fig1_1_roadmap.py`: danh sách `GIAI_DOAN` (chữ trong hộp, màu), `CHU_THICH_DUOI`.
- `fig1_2_chain.py`: danh sách `MAT_XICH`. Thêm hoặc bớt mắt xích thì bố cục tự giãn.
- `fig2_1_architecture.py`: danh sách `HOP` (toạ độ, kích thước, chữ, màu) và `MUI_TEN`.
  Toạ độ theo hệ 0–1, gốc ở góc dưới bên trái.
- `fig2_2_temporal.py`: `NAM_DAU`, `NAM_CUOI`, `LAG`, và bảng `VAI_TRO`. Đổi mốc chia dữ liệu
  ở `VAI_TRO` là đủ — màu ô và chú giải tự suy ra.

Đổi màu đồng loạt cho cả bốn sơ đồ: sửa từ điển `C` trong `_common.py`.
Đổi phông hoặc độ phân giải: sửa `plt.rcParams["font.family"]` và `DPI` trong `_common.py`.

**Ba biểu đồ số liệu** — tên trục, màu, cỡ hình nằm ở các hằng viết HOA đầu tệp.
`fig4_2_importance.py` có từ điển `TEN_DAC_TRUNG` để đổi tên hiển thị của 12 đặc trưng.

## Cập nhật số liệu

Các tệp trong `data/` được sinh từ kết quả thực nghiệm thật bằng `code/export_figdata.py`
(nằm trong gói mã nguồn). Khi chạy lại thực nghiệm:

```
cd ../code && python3 export_figdata.py
cd ../Figures && python3 build_all.py --copy
```

Không sửa tay các tệp trong `data/` — chúng phải khớp với tệp kết quả JSON để mọi con số
trong chuyên đề vẫn truy nguyên được.
