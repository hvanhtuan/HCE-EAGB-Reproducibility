"""Chuẩn bị bổ sung cho phản biện: (a) gắn nhãn sự kiện lũ cho bồi thường trong từng ô-năm; (b) các quy tắc ghép bồi thường thay thế.
Không thay đổi ../data/nfip_cells.parquet; ghi ../data/nfip_cells_event.parquet và ../data/nfip_cells_rule_*.parquet."""
import numpy as np, pandas as pd, glob, json
src = open("prep_nfip.py", encoding="utf8").read().split("qa = {}")[0]
exec(src)  # keys, zone, KEY, hằng số
cells = pd.read_parquet("../data/nfip_cells.parquet")
C = pd.concat([pd.read_parquet(f) for f in glob.glob("../data/raw/NfipClaims_*.parquet")], ignore_index=True).drop_duplicates("id")
C["paid"] = (C.amountPaidOnBuildingClaim.fillna(0) + C.amountPaidOnContentsClaim.fillna(0)).clip(lower=0)
C = C[C.yearOfLoss.isin(YEARS) & C.censusGeoid.notna()]
KC = keys(C, "numberOfFloorsInTheInsuredBuilding"); KC["year"] = C.yearOfLoss.astype(int).values
KC["N"] = (C.paid > 0).astype(int).values; KC["S"] = C.paid.values; KC["ev"] = C.floodEvent.fillna("(không có nhãn sự kiện)").values
# (a) bồi thường theo ô-năm-sự kiện
ce = KC.groupby(["year"] + KEY + ["ev"], observed=True).agg(N=("N", "sum"), S=("S", "sum")).reset_index()
ce.to_parquet("../data/claims_by_cell_event.parquet")
print("claims_by_cell_event", len(ce), flush=True)
# (b) quy tắc ghép thay thế: chuyển bồi thường không ghép được vào ô có phơi nhiễm cùng (tract, năm) hoặc cùng (quận, năm)
cl = KC.groupby(["year"] + KEY, observed=True).agg(N=("N", "sum"), S=("S", "sum")).reset_index()
cells = cells.drop(columns=["N", "S", "Nev"])
m = cells.merge(cl, on=["year"] + KEY, how="outer", indicator=True)
un = m[m._merge == "right_only"][["year"] + KEY + ["N", "S"]].copy()
base = m[m._merge == "both"].drop(columns="_merge").copy()
# ô có phơi nhiễm, cho phép
cx = cells[cells.E > 1e-3].copy()
cx["county"] = cx.tract.str[:5]; cx["state"] = cx.tract.str[:2]
un["county"] = un.tract.str[:5]
for rule, gkeys in [("tract", ["year", "tract"]), ("county", ["year", "county"])]:
    tot = cx.groupby(gkeys)["E"].sum().rename("Etot").reset_index()
    u = un.groupby(gkeys)[["N", "S"]].sum().reset_index().merge(tot, on=gkeys, how="inner")  # chỉ chuyển nếu nhóm có phơi nhiễm
    moved = u[["N", "S"]].sum()
    w = cx.merge(u.rename(columns={"N": "uN", "S": "uS"}), on=gkeys, how="left")
    w[["uN", "uS", "Etot"]] = w[["uN", "uS", "Etot"]].fillna(0)
    share = np.where(w.Etot > 0, w.E / w.Etot.replace(0, np.nan), 0)
    w["N2"] = w.uN * share; w["S2"] = w.uS * share
    out = cx.merge(cl, on=["year"] + KEY, how="left"); out[["N", "S"]] = out[["N", "S"]].fillna(0)
    out["N"] = out.N.values + w.N2.values; out["S"] = out.S.values + w.S2.values; out["Nev"] = 0.0
    out.to_parquet(f"../data/nfip_cells_rule_{rule}.parquet")
    print(rule, "chuyển được", float(moved.N), float(moved.S), "trên tổng không ghép", float(un.N.sum()), float(un.S.sum()), flush=True)
