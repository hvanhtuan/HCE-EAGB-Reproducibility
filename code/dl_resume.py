"""Tải NfipPolicies theo keyset, tiếp tục được giữa các lần chạy (mỗi lần tối đa BUDGET giây).
Dùng: python3 dl_resume.py <STATE> <BUDGET_S>. Ghi mảnh vào data/raw/chunks/ và gộp thành data/raw/NfipPolicies_<STATE>.parquet khi xong."""
import sys, json, time, urllib.request, urllib.parse, os, glob, pandas as pd
st=sys.argv[1]; budget=float(sys.argv[2]); t0=time.time()
SEL="policyEffectiveDate,policyTerminationDate,policyCount,propertyState,censusGeoid,ratedFloodZone,occupancyType,primaryResidenceIndicator,elevatedBuildingIndicator,postFIRMConstructionIndicator,numberOfFloorsInInsuredBuilding,totalBuildingInsuranceCoverage,totalContentsInsuranceCoverage,totalInsurancePremiumOfThePolicy,policyCost,buildingDeductibleCode,originalConstructionDate,crsClassCode,cancellationDateOfFloodPolicy,id"
out=f"data/raw/NfipPolicies_{st}.parquet"
if os.path.exists(out): print("exists",st); sys.exit()
os.makedirs("data/raw/chunks",exist_ok=True)
fs=sorted(glob.glob(f"data/raw/chunks/{st}_*.parquet"))
last=-1
if fs: last=int(fs[-1].split("_")[-1].split(".")[0])
rows=[]; n=0; finished=False
while time.time()-t0<budget:
    q=urllib.parse.urlencode({"$select":SEL,"$filter":f"propertyState eq '{st}' and id gt {last}","$top":10000,"$orderby":"id"})
    for a in range(5):
        try:
            with urllib.request.urlopen("https://www.fema.gov/api/open/v3/NfipPolicies?"+q, timeout=100) as r: d=json.load(r)["NfipPolicies"]; break
        except Exception as e:
            time.sleep(3)
    else: break
    if not d: finished=True; break
    last=int(d[-1]["id"]); rows+=d; n+=1
    if len(d)<10000: finished=True; break
    if n%10==0:
        pd.DataFrame(rows).to_parquet(f"data/raw/chunks/{st}_{last:012d}.parquet"); rows=[]
if rows: pd.DataFrame(rows).to_parquet(f"data/raw/chunks/{st}_{last:012d}.parquet")
if finished:
    D=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"data/raw/chunks/{st}_*.parquet"))],ignore_index=True)
    D=D[D.policyEffectiveDate.fillna("0")>="2009"]
    D.to_parquet(out); print("done",st,len(D))
else: print("partial",st,last)
