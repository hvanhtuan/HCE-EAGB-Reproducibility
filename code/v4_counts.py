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


un["state"] = un.tract.astype(str).str[:2]; mt = mt.copy(); mt["state"] = mt.tract.astype(str).str[:2]
res = {}
for nm, col in [("year", "year"), ("state", "state")]:
    a = mt.groupby(col).agg(khop_N=("N", "sum"), khop_S=("S", "sum")); b = un.groupby(col).agg(kk_N=("N", "sum"), kk_S=("S", "sum"))
    t = a.join(b, how="outer").fillna(0); t["ty_le_N"] = t.kk_N / (t.kk_N + t.khop_N); t["ty_le_S"] = t.kk_S / (t.kk_S + t.khop_S)
    res[nm] = json.loads(t.reset_index().to_json(orient="records"))
res["tong"] = out
json.dump(res, open("../results/v4_counts.json", "w"), indent=1)
print(json.dumps(res, indent=0)[:4000])
