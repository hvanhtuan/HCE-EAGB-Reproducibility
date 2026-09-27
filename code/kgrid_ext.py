"""Kiểm tra hậu thí điểm: mở rộng lưới k cho bộ mã hóa HCE đồng thời (cùng tiêu chí thí điểm)."""
import json, numpy as np
from nfip_lib import *
from metrics import tweedie_dev
XI_EVAL = 1.5
d = load(); T, node = build_tree(d)
TR = (d.year >= 2012) & (d.year <= 2018)
years_all = list(range(2010, 2025))
E = d.E.values; rate = d.rate.values
res = {}
for km in [100000, 300000, 1000000, 3000000]:
    z, _ = temporal_encoder(d, T, node, years_all, [0] + [km]*3, k0=1.0, method="simul")
    res[km] = float(np.sum(E[TR]*tweedie_dev(rate[TR], np.maximum(z[TR],1e-3), XI_EVAL))/E[TR].sum())
    print(km, res[km], flush=True)
json.dump(res, open("../results/kgrid_ext.json","w"), indent=1)
