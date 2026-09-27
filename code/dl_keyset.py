import sys, json, time, urllib.request, urllib.parse, os, pandas as pd
st=sys.argv[1]
SEL="policyEffectiveDate,policyTerminationDate,policyCount,propertyState,censusGeoid,ratedFloodZone,occupancyType,primaryResidenceIndicator,elevatedBuildingIndicator,postFIRMConstructionIndicator,numberOfFloorsInInsuredBuilding,totalBuildingInsuranceCoverage,totalContentsInsuranceCoverage,totalInsurancePremiumOfThePolicy,policyCost,buildingDeductibleCode,originalConstructionDate,crsClassCode,cancellationDateOfFloodPolicy,id"
out=f"data/raw/NfipPolicies_{st}.parquet"
if os.path.exists(out): sys.exit()
rows=[]; last=-1; chunk=0
while True:
    q=urllib.parse.urlencode({"$select":SEL,"$filter":f"propertyState eq '{st}' and id gt {last}","$top":10000,"$orderby":"id"})
    for a in range(20):
        try:
            with urllib.request.urlopen("https://www.fema.gov/api/open/v3/NfipPolicies?"+q, timeout=120) as r: d=json.load(r); break
        except Exception as e:
            print("retry",st,last,e,flush=True); time.sleep(5+3*a)
    else: raise SystemExit("fail")
    rec=[x for x in d["NfipPolicies"] if (x.get("policyEffectiveDate") or "0")>="2009"]
    rows+=rec
    if not d["NfipPolicies"]: break
    last=int(d["NfipPolicies"][-1]["id"])
    chunk+=1
    if chunk%20==0: print(st,chunk,len(rows),flush=True)
    if len(d["NfipPolicies"])<10000: break
pd.DataFrame(rows).to_parquet(out); print("done",st,len(rows),flush=True)
