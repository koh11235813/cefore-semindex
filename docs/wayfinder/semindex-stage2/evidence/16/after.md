# evidence/16: added after the results were read

Outside spec.md, no pass line. Same bootstrap as ana.py (2000 resamples of seeds 1 to 20, `default_rng(20261117)`, all stages from the same resampled seeds). Values in percent.

| quantity | point | 95 % interval |
|---|---|---|
| gain of s7t37 against the line through the two fixed arms | -6.4 | -14.4 to +1.8 |
| gain of s7t37r against the line through the two fixed arms | +3.4 | -5.7 to +12.5 |
| delay of s7t37r relative to s7t37 | -23.0 | -30.9 to -12.1 |
| polls per run of s7t37r relative to s7t37 | +26.8 | +23.1 to +30.2 |

Share of resamples with gain difference s7t37r minus s7t37 above zero: 95.0 %.

Per-seed delay in seconds (`lat4 / n4`), seeds 1 to 20:

- s6b3p695: 19.4 22.5 16.9 23.7 30.6 16.9 23.6 16.7 22.4 19.4 23.1 21.9 21.4 19.9 17.6 22.9 25.2 20.5 23.7 20.9
- s6b10p043: 36.3 29.1 31.6 46.0 48.2 46.2 38.5 31.1 54.0 44.8 52.5 37.0 47.5 45.0 35.5 51.9 41.3 47.6 45.2 41.0
- s7t37: 29.1 24.8 31.7 37.5 37.7 35.8 43.3 32.6 45.5 36.6 30.6 31.6 28.1 21.2 37.2 38.9 33.5 48.9 39.3 26.5
- s7t37r: 21.6 23.6 24.4 25.0 27.4 25.3 34.4 27.9 30.0 29.7 17.1 25.2 31.1 38.1 19.9 31.0 22.4 23.8 23.2 30.1
