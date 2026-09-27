"""Chuẩn hóa dữ liệu OpenFEMA NFIP Redacted Policies/Claims v3 thành bảng ô rủi ro × năm (Tầng 1c — dữ liệu công khai thay thế).
Đơn vị quan sát: ô = (năm lịch, census tract, nhóm vùng lũ, nhóm mục đích sử dụng, nơi ở chính, nhà nâng tầng, post-FIRM, số tầng,
nhóm mức khấu trừ, nhóm số tiền bảo hiểm tòa nhà). Phơi nhiễm = số năm–hợp đồng (phân bổ theo thời gian hiệu lực trong năm)."""
import numpy as np, pandas as pd, glob, json

STATES = ["CT", "DE", "MA", "MD", "ME", "NH", "NJ", "RI"]
YEARS = list(range(2010, 2026))
OCC = {1: "SF", 11: "SF", 14: "SF", 16: "SF", 2: "MF", 3: "MF", 12: "MF", 13: "MF", 15: "MF", 4: "NR", 6: "NR", 17: "NR", 18: "NR", 19: "NR"}
DED = {"0": 500, "1": 1000, "2": 2000, "3": 3000, "4": 4000, "5": 5000, "9": 750, "A": 10000, "B": 15000, "C": 20000, "D": 25000,
       "E": 50000, "F": 1250, "G": 1500, "H": 200}


def zone(z):
    z = z.fillna("U").astype(str).str.upper()
    return np.select([z.str.startswith("V"), z.str.startswith("A"), z.isin(["X", "B", "C"]) | z.str.startswith("X"), z == "D"], ["V", "A", "X", "D"], "U")


def keys(df, floors_col):
    k = pd.DataFrame(index=df.index)
    k["tract"] = df.censusGeoid.astype(str).str[:11]
    k["zone"] = zone(df.ratedFloodZone)
    k["occ"] = df.occupancyType.map(OCC).fillna("U")
    k["prim"] = df.primaryResidenceIndicator.fillna(False).astype(int)
    k["elev"] = df.elevatedBuildingIndicator.fillna(False).astype(int)
    k["pfirm"] = df.postFIRMConstructionIndicator.fillna(False).astype(int)
    k["floors"] = df[floors_col].fillna(0).astype(int).clip(0, 6)
    ded = df.buildingDeductibleCode.astype(str).map(DED)
    k["ded"] = pd.cut(ded, [0, 1000, 2000, 5000, 1e9], labels=["le1k", "1-2k", "2-5k", "gt5k"]).astype(str)
    cov = df.totalBuildingInsuranceCoverage.fillna(0)
    k["cov"] = pd.cut(cov, [-1, 99999, 199999, 249999, 1e12], labels=["lt100k", "100-200k", "200-250k", "ge250k"]).astype(str)
    return k


KEY = ["tract", "zone", "occ", "prim", "elev", "pfirm", "floors", "ded", "cov"]
qa = {}
# ---- hợp đồng -> phơi nhiễm theo năm (xử lý theo từng bang để tiết kiệm bộ nhớ)
qa["policies_rows"] = 0; qa["policies_by_state"] = {}; qa["policies_missing_geoid"] = 0; qa["policies_zero_term"] = 0
agg_parts = []
seen_ids = set()
for stt in STATES:
    fl = glob.glob(f"../data/raw/NfipPolicies_{stt}.parquet") + glob.glob(f"../data/raw/part/NfipPolicies_{stt}_*.parquet")
    P = pd.concat([pd.read_parquet(f) for f in fl], ignore_index=True).drop_duplicates("id")
    qa["policies_rows"] += len(P); qa["policies_by_state"][stt] = len(P)
    st = pd.to_datetime(P.policyEffectiveDate, utc=True).dt.tz_localize(None)
    en = pd.to_datetime(P.policyTerminationDate, utc=True).dt.tz_localize(None)
    ca = pd.to_datetime(P.cancellationDateOfFloodPolicy, utc=True, errors="coerce").dt.tz_localize(None)
    en = en.where(ca.isna() | (ca >= en) | (ca < st), ca)
    qa["policies_missing_geoid"] += int(P.censusGeoid.isna().sum()); qa["policies_zero_term"] += int(((en - st).dt.days <= 0).sum())
    ok = P.censusGeoid.notna() & ((en - st).dt.days > 0)
    P, st, en = P[ok].copy(), st[ok], en[ok]
    term = (en - st).dt.days.clip(lower=1)
    K = keys(P, "numberOfFloorsInInsuredBuilding")
    logcov = np.log1p(P.totalBuildingInsuranceCoverage.fillna(0) + P.totalContentsInsuranceCoverage.fillna(0))
    cyear = pd.to_datetime(P.originalConstructionDate, utc=True, errors="coerce").dt.year.fillna(1970)
    crs = P.crsClassCode.fillna(10).clip(1, 10); prem = P.totalInsurancePremiumOfThePolicy.fillna(0).clip(lower=0)
    pc = P.policyCount.fillna(1).clip(lower=1)
    for y in YEARS:
        a = pd.Timestamp(f"{y}-01-01"); b = pd.Timestamp(f"{y + 1}-01-01")
        ov = (np.minimum(en, b) - np.maximum(st, a)).dt.days.clip(lower=0)
        m = (ov > 0).values
        if not m.any():
            continue
        Ey = (ov[m] / 365.25 * pc[m]).values
        dd = K[m].copy(); dd["year"] = y; dd["E"] = Ey
        dd["prem"] = (prem[m] * ov[m] / term[m]).values
        dd["logcov_w"] = logcov[m].values * Ey; dd["cyear_w"] = cyear[m].values * Ey; dd["crs_w"] = crs[m].values * Ey
        agg_parts.append(dd.groupby(["year"] + KEY, observed=True)[["E", "prem", "logcov_w", "cyear_w", "crs_w"]].sum().reset_index())
    del P, K
    print("policies", stt, flush=True)
cells = pd.concat(agg_parts, ignore_index=True).groupby(["year"] + KEY, observed=True)[["E", "prem", "logcov_w", "cyear_w", "crs_w"]].sum().reset_index()
for c in ["logcov", "cyear", "crs"]:
    cells[c] = cells[c + "_w"] / cells.E; cells.drop(columns=c + "_w", inplace=True)
# ---- bồi thường
C = pd.concat([pd.read_parquet(f) for f in glob.glob("../data/raw/NfipClaims_*.parquet") + glob.glob("../data/raw/part/NfipClaims_*.parquet")], ignore_index=True).drop_duplicates("id")
qa["claims_rows"] = len(C)
C["paid"] = (C.amountPaidOnBuildingClaim.fillna(0) + C.amountPaidOnContentsClaim.fillna(0))
qa["claims_negative_paid"] = int((C.paid < 0).sum()); C["paid"] = C.paid.clip(lower=0)
C = C[C.yearOfLoss.isin(YEARS) & C.censusGeoid.notna()]
qa["claims_in_years"] = len(C); qa["claims_paid_pos"] = int((C.paid > 0).sum())
KC = keys(C, "numberOfFloorsInTheInsuredBuilding"); KC["year"] = C.yearOfLoss.astype(int).values
KC["N"] = (C.paid > 0).astype(int).values; KC["S"] = C.paid.values; KC["event"] = C.floodEvent.notna().astype(int).values
cl = KC.groupby(["year"] + KEY, observed=True).agg(N=("N", "sum"), S=("S", "sum"), Nev=("event", "sum")).reset_index()
m = cells.merge(cl, on=["year"] + KEY, how="outer", indicator=True)
qa["claim_cells_without_exposure"] = int((m._merge == "right_only").sum())
qa["claims_without_exposure_N"] = int(m.loc[m._merge == "right_only", "N"].sum())
qa["claims_without_exposure_S"] = float(m.loc[m._merge == "right_only", "S"].sum())
qa["claims_matched_N"] = int(m.loc[m._merge == "both", "N"].sum()); qa["claims_matched_S"] = float(m.loc[m._merge == "both", "S"].sum())
m = m[m._merge != "right_only"].drop(columns="_merge")
m[["N", "S", "Nev"]] = m[["N", "S", "Nev"]].fillna(0)
m = m[m.E > 1e-3]
m["county"] = m.tract.str[:5]; m["state"] = m.tract.str[:2]
m.to_parquet("../data/nfip_cells.parquet")
qa["cells"] = len(m); qa["cells_by_year"] = m.groupby("year").agg(E=("E", "sum"), N=("N", "sum"), S=("S", "sum"), cells=("E", "size")).reset_index().to_dict("records")
qa["n_tract"] = int(m.tract.nunique()); qa["n_county"] = int(m.county.nunique()); qa["n_state"] = int(m.state.nunique())
json.dump(qa, open("../results/nfip_qa.json", "w"), indent=1, default=float)
print(json.dumps(qa, indent=1, default=float)[:3000])
