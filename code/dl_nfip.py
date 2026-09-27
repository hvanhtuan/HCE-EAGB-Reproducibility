import sys, json, time, urllib.request, urllib.parse, os, pandas as pd
ent, st = sys.argv[1], sys.argv[2]
base = "https://www.fema.gov/api/open/v3/"+ent
if ent=="NfipPolicies":
    sel="policyEffectiveDate,policyTerminationDate,policyCount,propertyState,censusGeoid,ratedFloodZone,occupancyType,primaryResidenceIndicator,elevatedBuildingIndicator,postFIRMConstructionIndicator,numberOfFloorsInInsuredBuilding,totalBuildingInsuranceCoverage,totalContentsInsuranceCoverage,totalInsurancePremiumOfThePolicy,policyCost,buildingDeductibleCode,originalConstructionDate,crsClassCode,cancellationDateOfFloodPolicy,id"
    flt=f"propertyState eq '{st}' and policyEffectiveDate ge '2009-01-01T00:00:00.000Z'"
else:
    sel="dateOfLoss,yearOfLoss,policyCount,state,countyCode,censusGeoid,ratedFloodZone,occupancyType,primaryResidenceIndicator,elevatedBuildingIndicator,postFIRMConstructionIndicator,numberOfFloorsInTheInsuredBuilding,amountPaidOnBuildingClaim,amountPaidOnContentsClaim,totalBuildingInsuranceCoverage,buildingDeductibleCode,originalConstructionDate,crsClassCode,floodEvent,causeOfDamage,id"
    flt=f"state eq '{st}' and yearOfLoss ge 2009"
out=f"data/raw/{ent}_{st}.parquet"
if os.path.exists(out): sys.exit(0)
rows=[]; skip=0; top=10000
while True:
    q=urllib.parse.urlencode({"$select":sel,"$filter":flt,"$top":top,"$skip":skip,"$orderby":"id","$format":"json","$metadata":"false"})
    for a in range(6):
        try:
            with urllib.request.urlopen(base+"?"+q, timeout=300) as r: d=json.load(r); break
        except Exception as e:
            print("retry",st,skip,e,flush=True); time.sleep(10*(a+1))
    else: raise SystemExit("fail")
    recs=d[ent]; rows+=recs; skip+=len(recs)
    print(ent,st,skip,flush=True)
    if len(recs)<top: break
pd.DataFrame(rows).to_parquet(out)
print("done",ent,st,len(rows))
