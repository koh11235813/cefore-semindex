# after.md: how firm the reading of T37r is (see after.py; added after the results were seen)

gain and diff in %, diff = gain(T37r) - gain(T37), intervals from 2000 seed resamples (default_rng(20261117)). '<= 0' = resamples with diff at or below zero; the 2.5 % point sits
between the 50th and 51st smallest, so 50 or more means the lower bound is not above zero under every
percentile convention.

## cut t0+72 s

| workload | cell | cens T37 / T37r / fix10.043 | T37 gain | T37r gain | T37r gain 95 % | diff | diff 95 % | <= 0 | onoff_gain.csv diff |
|---|---|---|---|---|---|---|---|---|---|
| W0 | s6 p0.2 | 0.0 / 0.0 / 0.1 % | +5.1 | +6.6 | +2.2 .. +11.0 | +1.5 | -6.1163 .. +9.3031 | 702/2000 | +1.5 |
| W0 | s6 p0.4 | 1.9 / 0.0 / 4.0 % | -8.9 | +8.4 | +1.6 .. +14.8 | +17.4 | +7.2794 .. +26.4816 | 1/2000 | +17.4 |
| WO | s6 p0.2 | 0.0 / 0.0 / 0.2 % | -5.2 | +2.0 | -6.1 .. +10.4 | +7.1 | +0.0003 .. +14.2255 | 50/2000 | +7.1 |
| WO | s6 p0.4 | 3.3 / 0.0 / 3.3 % | -13.8 | +1.7 | -5.5 .. +8.3 | +15.5 | +6.7695 .. +23.9810 | 1/2000 | +15.5 |
| WV | s6 p0.2 | 0.0 / 0.0 / 0.0 % | -4.9 | +0.7 | -7.6 .. +8.5 | +5.5 | -3.6211 .. +14.5333 | 243/2000 | +5.5 |
| WV | s6 p0.4 | 0.2 / 0.0 / 5.0 % | -8.6 | +4.9 | -2.5 .. +11.2 | +13.5 | +1.4386 .. +25.3345 | 24/2000 | +13.5 |

Cases (of 6) with the lower bound above zero under 25 other bootstrap seeds (default_rng(1000..1024)): 3 cases x15, 4 cases x10.

## cut t0+60 s

| workload | cell | cens T37 / T37r / fix10.043 | T37 gain | T37r gain | T37r gain 95 % | diff | diff 95 % | <= 0 | onoff_gain.csv diff |
|---|---|---|---|---|---|---|---|---|---|
| W0 | s6 p0.2 | 0.0 / 0.0 / 0.0 % | +3.9 | +4.7 | -0.9 .. +10.2 | +0.7 | -8.4607 .. +10.3503 | 871/2000 | - |
| W0 | s6 p0.4 | 1.0 / 0.0 / 1.6 % | -12.9 | +6.7 | -1.0 .. +13.9 | +19.7 | +8.0619 .. +30.0577 | 1/2000 | - |
| WO | s6 p0.2 | 0.0 / 0.0 / 0.0 % | -4.1 | +1.0 | -9.3 .. +11.6 | +5.1 | -5.4949 .. +15.1745 | 332/2000 | - |
| WO | s6 p0.4 | 1.3 / 0.0 / 1.0 % | -18.6 | +0.4 | -8.3 .. +8.2 | +19.0 | +8.3232 .. +29.7751 | 1/2000 | - |
| WV | s6 p0.2 | 0.0 / 0.0 / 0.0 % | -4.0 | -0.7 | -11.3 .. +8.9 | +3.3 | -7.6498 .. +13.3025 | 586/2000 | - |
| WV | s6 p0.4 | 0.0 / 0.0 / 2.7 % | -13.1 | +3.9 | -4.8 .. +11.9 | +17.0 | +3.7951 .. +30.7554 | 11/2000 | - |

Cases (of 6) with the lower bound above zero under 25 other bootstrap seeds (default_rng(1000..1024)): 3 cases x25.

