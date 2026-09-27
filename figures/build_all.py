"""Chạy lại toàn bộ 7 script vẽ hình, và (tuỳ chọn) chép kết quả vào thư mục hình của chuyên đề.

    python3 build_all.py             # chỉ vẽ vào output/
    python3 build_all.py --copy      # vẽ rồi chép đè vào ../doc/figs/
"""
import subprocess, sys, shutil, os, glob

SCRIPTS = [
    "fig1_1_roadmap.py",
    "fig1_2_chain.py",
    "fig2_1_architecture.py",
    "fig2_2_temporal.py",
    "fig4_1_calib_lorenz.py",
    "fig4_2_importance.py",
    "fig4_3_pareto.py",
]
DICH = "../doc/figs/"

os.chdir(os.path.dirname(os.path.abspath(__file__)))
loi = []
for s in SCRIPTS:
    print("=== " + s)
    r = subprocess.run([sys.executable, s])
    if r.returncode != 0:
        loi.append(s)

if loi:
    print("\nCAC SCRIPT LOI:", ", ".join(loi))
    sys.exit(1)

if "--copy" in sys.argv:
    os.makedirs(DICH, exist_ok=True)
    for p in glob.glob("output/*.png"):
        shutil.copy(p, DICH + os.path.basename(p))
        print("chep ->", DICH + os.path.basename(p))

print("\nXong", len(SCRIPTS), "hinh.")
