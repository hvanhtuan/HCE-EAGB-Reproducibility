"""Phân tích các bản ghi bồi thường không khớp được với ô phơi nhiễm (khoảng 11–12% theo báo cáo chất lượng).
Phép nối lại được thực hiện trên đúng khóa của bảng ô cuối cùng, nên tổng số ở đây là số của phân tích này,
không nhất thiết trùng từng đơn vị với báo cáo chất lượng ban đầu (báo cáo đó nối trước khi lọc E > 1e-3).
Mục tiêu: xác định *nguyên nhân* không khớp và kiểm tra xem phần không khớp có lệch hệ thống theo năm,
bang, vùng lũ hay mục đích sử dụng — điều kiện để biết kết luận có bị chệch chọn mẫu hay không.
"""
import glob, json, numpy as np, pandas as pd
# định nghĩa khóa lấy đúng từ prep_nfip.py (sao lại để không chạy lại toàn bộ bước chuẩn hóa)
import re as _re
_src = open("prep_nfip.py", encoding="utf-8").read()
_ns = {"np": np, "pd": pd}
exec(_re.search(r"^OCC = .*?^KEY = \[.*?\]$", _src, _re.M | _re.S).group(0), _ns)
exec(_re.search(r"^YEARS = .*?$", _src, _re.M).group(0), _ns)
keys, KEY, YEARS = _ns["keys"], _ns["KEY"], _ns["YEARS"]

C = pd.concat([pd.read_parquet(f) for f in glob.glob("../data/raw/NfipClaims_*.parquet")
               + glob.glob("../data/raw/part/NfipClaims_*.parquet")], ignore_index=True).drop_duplicates("id")
C["paid"] = (C.amountPaidOnBuildingClaim.fillna(0) + C.amountPaidOnContentsClaim.fillna(0)).clip(lower=0)
C = C[C.yearOfLoss.isin(YEARS) & C.censusGeoid.notna()]
K = keys(C, "numberOfFloorsInTheInsuredBuilding")
K["year"] = C.yearOfLoss.astype(int).values
K["N"] = (C.paid > 0).astype(int).values
K["S"] = C.paid.values
cl = K.groupby(["year"] + KEY, observed=True).agg(N=("N", "sum"), S=("S", "sum")).reset_index()

cells = pd.read_parquet("../data/nfip_cells.parquet", columns=["year"] + KEY)
cells["_ok"] = 1
j = cl.merge(cells, on=["year"] + KEY, how="left")
un = j[j._ok.isna()].copy()
mt = j[j._ok == 1]

out = {"tong_N": int(cl.N.sum()), "tong_S": float(cl.S.sum()),
       "khop_N": int(mt.N.sum()), "khop_S": float(mt.S.sum()),
       "khong_khop_N": int(un.N.sum()), "khong_khop_S": float(un.S.sum())}
out["ty_le_N"] = out["khong_khop_N"] / out["tong_N"]
out["ty_le_S"] = out["khong_khop_S"] / out["tong_S"]

# ---- nguyên nhân: thiếu hoàn toàn tract x năm, hay chỉ lệch tổ hợp thuộc tính?
ty = pd.read_parquet("../data/nfip_cells.parquet", columns=["year", "tract"]).drop_duplicates()
ty["_t"] = 1
un = un.merge(ty, on=["year", "tract"], how="left")
no_ty = un._t.isna()
out["nguyen_nhan"] = {
    "thieu_tract_x_nam_N": int(un.loc[no_ty, "N"].sum()),
    "thieu_tract_x_nam_S": float(un.loc[no_ty, "S"].sum()),
    "lech_to_hop_thuoc_tinh_N": int(un.loc[~no_ty, "N"].sum()),
    "lech_to_hop_thuoc_tinh_S": float(un.loc[~no_ty, "S"].sum()),
}

# ---- trong nhóm lệch tổ hợp: bỏ lần lượt từng thuộc tính, xem thuộc tính nào "gỡ" được nhiều nhất
sub = un.loc[~no_ty].copy()
out["thuoc_tinh_gay_lech"] = {}
for a in KEY:
    rest = ["year"] + [k for k in KEY if k != a]
    cc = cells[rest].drop_duplicates(); cc["_h"] = 1
    h = sub.merge(cc, on=rest, how="left")
    out["thuoc_tinh_gay_lech"][a] = {"N_go_duoc": int(h.loc[h._h == 1, "N"].sum()),
                                     "ty_le": float(h.loc[h._h == 1, "N"].sum() / max(sub.N.sum(), 1))}

# ---- phần không khớp có lệch hệ thống hay không: so cấu trúc khớp / không khớp
def cau_truc(df, col):
    g = df.groupby(col, observed=True).N.sum()
    return (g / g.sum()).round(4).to_dict()


out["so_sanh_cau_truc"] = {}
for col in ["year", "zone", "occ", "prim", "elev", "pfirm", "cov"]:
    out["so_sanh_cau_truc"][col] = {"khop": cau_truc(mt, col), "khong_khop": cau_truc(un, col)}
un["state"] = un.tract.astype(str).str[:2]
mt2 = mt.copy(); mt2["state"] = mt2.tract.astype(str).str[:2]
out["so_sanh_cau_truc"]["state"] = {"khop": cau_truc(mt2, "state"), "khong_khop": cau_truc(un, "state")}
# mức độ tổn thất trung bình
out["mucdo_trung_binh"] = {"khop": float(mt.S.sum() / max(mt.N.sum(), 1)),
                           "khong_khop": float(un.S.sum() / max(un.N.sum(), 1))}
# riêng tập kiểm định 2021–2023
te = un[(un.year >= 2021) & (un.year <= 2023)]; tm = mt[(mt.year >= 2021) & (mt.year <= 2023)]
out["tap_kiem_dinh_2021_2023"] = {"khong_khop_N": int(te.N.sum()), "khop_N": int(tm.N.sum()),
                                  "ty_le_N": float(te.N.sum() / max(te.N.sum() + tm.N.sum(), 1)),
                                  "khong_khop_S": float(te.S.sum()), "khop_S": float(tm.S.sum()),
                                  "ty_le_S": float(te.S.sum() / max(te.S.sum() + tm.S.sum(), 1))}

json.dump(out, open("../results/unmatched_claims.json", "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps(out, indent=1, ensure_ascii=False, default=float)[:6000])
