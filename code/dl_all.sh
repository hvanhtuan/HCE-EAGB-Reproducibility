for s in DE RI ME NH CT MA MD; do python3 code/dl_nfip.py NfipClaims $s; done
for s in NH ME RI DE CT MA MD; do python3 code/dl_nfip.py NfipPolicies $s; done
echo ALLDONE
