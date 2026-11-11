# Replay on the 8-12 s outage shape (s6): calibration and wait retune

Replay half of evidence/10. Everything here is the evidence/09 replay model (`../09/replay.py`, imported unchanged), not a measurement, except the live columns of the calibration. Scripts: `replay_common.py` (shared), `replay_calib.py`, `replay_retune.py`; commands and runtimes in `RUN_replay.txt`.

## Rule mapping and caveats

| live stage | replay rule | jitter |
|---|---|---|
| s6b0p5 / s6b1p359 / s6b3p695 / s6b10p043 | `R.Fixed(x)` | every interval +/-10 % |
| s6short (`[10.043, 3.695]`, jitter `[0.1, 0.1]`) | `R.Table((10.043, 3.695))`, ideal mode | answer interval and wait jittered |
| s6wait (`[10.043, 5.0]`, jitter `[0.1, 0.0]`) | `R.SlotBlind(10.043, wait=5.0)` | answer interval jittered, wait exact |

- Ideal mode: the live engine re-decides on answer or timeout, which evidence/09's README matches to the replay's `ideal` form.
- `R.Table` jitters the wait, `SlotBlind` does not. Each retune family is compared only with the unchanged rule of the same jitter form (short-form and two-step with s6short, wait-form with s6wait). A short-form and a wait-form rule at the same w are different rules.
- `REP_B` (reply bytes per answered poll) has no value above p 0.4. `replay_common` sets `REP_B[0.5] = REP_B[0.6] = NaN` in-process so `simulate` does not raise; bytes and [4] are therefore not reported at p 0.5/0.6.
- The replay's loop overhead `0.05 + 0.035 p` s is extrapolated to p 0.5/0.6, beyond the p range it was fitted on; evidence/09 section 5.3 already doubted the p slope, and the poll counts below confirm it (see Calibration).
- Schedules: the six s6 stages share the link and record schedule of a (p, seed); `replay_calib.py` asserts this for all 480 stored runs. Training schedules (seeds 21-120) are regenerated with `semindex.evalgen.build_experiment` from the worktree's `src` (6b036ff; the runs used 5365e07); `replay_common.check_loader` asserts the regenerated schedule equals the stored one for seeds 1-3 at p 0.2 and 0.4.
- Seeding: `random.Random(hash(tag + (p, seed, rep)) & 0xffffffff)`, numeric tags, 10 reps per seed (replay_runs.py's paired convention). The unchanged rules' and arms' per-seed values on seeds 1-20 are identical in the two scripts (240 rows checked by `replay_retune.py`).

## Calibration: paired replay vs live, seeds 1-20

Delay = per-seed mean of uncensored 4-hop latency, then the mean over seeds. diff = replay - live, mean over seeds; SE = SD of 2000 seed-bootstrap means of the per-seed differences. Live cens = censored 4-hop pairs / all 4-hop pairs; replay cens = unfinished 4-hop record walks / generation times (different definitions, not compared as a bias). Full numbers in `replay_calib.csv`, replay per seed in `replay_calib_perseed.csv`.

| stage | p | replay delay ms | live delay ms | replay - live (SE) | rel | replay B/rh | replay waste | live waste | diff (SE) | replay polls | live polls | diff (SE) | replay cens | live cens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s6b0p5 | 0.2 | 5451 | 5531 | -79 (48) | -1.4% | 275.9 | 0.212 | 0.211 | +0.001 (0.000) | 1723.6 | 1745.5 | -21.9 (0.8) | 0.00% | 0.00% |
| s6b0p5 | 0.4 | 9558 | 9671 | -113 (97) | -1.2% | 220.8 | 0.413 | 0.413 | +0.000 (0.000) | 1701.9 | 1745.3 | -43.4 (0.6) | 0.00% | 0.00% |
| s6b0p5 | 0.5 | 12347 | 12361 | -14 (99) | -0.1% | - | 0.514 | 0.513 | +0.001 (0.000) | 1691.5 | 1744.3 | -52.9 (0.7) | 0.00% | 0.00% |
| s6b0p5 | 0.6 | 14905 | 14504 | +400 (148) | +2.8% | - | 0.604 | 0.602 | +0.001 (0.001) | 1681.4 | 1744.2 | -62.8 (0.9) | 0.00% | 0.00% |
| s6b1p359 | 0.2 | 7541 | 7484 | +57 (99) | +0.8% | 108.7 | 0.212 | 0.212 | +0.000 (0.001) | 678.9 | 682.4 | -3.5 (0.4) | 0.00% | 0.00% |
| s6b1p359 | 0.4 | 12384 | 12227 | +157 (193) | +1.3% | 87.7 | 0.413 | 0.413 | +0.000 (0.001) | 676.0 | 682.5 | -6.5 (0.4) | 0.00% | 0.52% |
| s6b1p359 | 0.5 | 16247 | 16713 | -466 (287) | -2.8% | - | 0.514 | 0.514 | +0.000 (0.001) | 674.2 | 682.5 | -8.3 (0.4) | 0.00% | 0.00% |
| s6b1p359 | 0.6 | 19410 | 19627 | -218 (329) | -1.1% | - | 0.604 | 0.603 | +0.001 (0.001) | 672.9 | 682.5 | -9.6 (0.3) | 0.02% | 0.00% |
| s6b3p695 | 0.2 | 13634 | 13829 | -195 (306) | -1.4% | 41.1 | 0.211 | 0.212 | -0.001 (0.001) | 256.7 | 256.1 | +0.6 (0.3) | 0.00% | 0.00% |
| s6b3p695 | 0.4 | 21325 | 21463 | -138 (504) | -0.6% | 33.2 | 0.414 | 0.416 | -0.002 (0.001) | 256.1 | 255.9 | +0.2 (0.2) | 0.05% | 0.00% |
| s6b3p695 | 0.5 | 26498 | 26078 | +420 (579) | +1.6% | - | 0.513 | 0.514 | -0.001 (0.003) | 256.0 | 255.8 | +0.2 (0.2) | 0.79% | 0.92% |
| s6b3p695 | 0.6 | 32636 | 31987 | +649 (823) | +2.0% | - | 0.602 | 0.599 | +0.002 (0.002) | 255.7 | 256.1 | -0.4 (0.3) | 0.78% | 1.33% |
| s6b10p043 | 0.2 | 30584 | 31561 | -977 (676) | -3.1% | 15.3 | 0.212 | 0.211 | +0.001 (0.003) | 95.3 | 96.0 | -0.7 (0.3) | 0.19% | 0.08% |
| s6b10p043 | 0.4 | 40987 | 42519 | -1532 (912) | -3.6% | 12.7 | 0.406 | 0.412 | -0.006 (0.005) | 95.0 | 96.0 | -1.0 (0.3) | 7.02% | 3.00% |
| s6b10p043 | 0.5 | 50591 | 46900 | +3690 (1980) | +7.9% | - | 0.510 | 0.512 | -0.001 (0.005) | 95.3 | 96.1 | -0.8 (0.3) | 13.98% | 22.33% |
| s6b10p043 | 0.6 | 57482 | 56725 | +757 (2329) | +1.3% | - | 0.604 | 0.609 | -0.006 (0.004) | 95.2 | 96.0 | -0.8 (0.2) | 29.30% | 31.67% |
| s6short | 0.2 | 26518 | 26681 | -163 (575) | -0.6% | 17.4 | 0.307 | 0.313 | -0.007 (0.006) | 118.7 | 120.8 | -2.2 (0.6) | 0.03% | 0.17% |
| s6short | 0.4 | 31656 | 32159 | -503 (901) | -1.6% | 16.5 | 0.502 | 0.493 | +0.008 (0.004) | 139.5 | 140.1 | -0.6 (0.6) | 0.00% | 0.00% |
| s6short | 0.5 | 36246 | 35402 | +844 (1085) | +2.4% | - | 0.578 | 0.574 | +0.004 (0.005) | 150.6 | 151.7 | -1.1 (0.9) | 1.15% | 0.33% |
| s6short | 0.6 | 40733 | 40920 | -187 (1033) | -0.5% | - | 0.641 | 0.635 | +0.006 (0.005) | 159.4 | 160.5 | -1.1 (1.0) | 2.21% | 1.00% |
| s6wait | 0.2 | 27035 | 27346 | -311 (824) | -1.1% | 16.8 | 0.265 | 0.263 | +0.003 (0.005) | 110.3 | 110.8 | -0.5 (0.4) | 0.05% | 0.00% |
| s6wait | 0.4 | 34337 | 35353 | -1016 (1405) | -2.9% | 15.3 | 0.469 | 0.460 | +0.009 (0.005) | 124.6 | 126.2 | -1.5 (0.5) | 0.79% | 0.50% |
| s6wait | 0.5 | 39742 | 40541 | -799 (1136) | -2.0% | - | 0.547 | 0.545 | +0.002 (0.006) | 131.5 | 132.9 | -1.5 (0.6) | 3.69% | 7.58% |
| s6wait | 0.6 | 46051 | 47826 | -1776 (1300) | -3.7% | - | 0.625 | 0.623 | +0.003 (0.006) | 138.1 | 140.1 | -1.9 (0.7) | 6.40% | 5.42% |

Bias of the two unchanged rules next to evidence/09's s5 bias (`../09/live_vs_replay.csv`, sign flipped to replay - live; its SE is SD(diff)/sqrt(20), not a bootstrap).

| rule | p | metric | s6 replay - live (SE) | s5 replay - live (SE) |
|---|---|---|---|---|
| short | 0.2 | delay ms | -163 (575) | -1048 (616) |
| short | 0.2 | polls/run | -2.2 (0.6) | -1.1 (0.6) |
| short | 0.2 | waste | -0.007 (0.006) | +0.001 (0.005) |
| short | 0.4 | delay ms | -503 (901) | +1051 (551) |
| short | 0.4 | polls/run | -0.6 (0.6) | -2.2 (0.6) |
| short | 0.4 | waste | +0.008 (0.004) | -0.004 (0.004) |
| wait | 0.2 | delay ms | -311 (824) | -574 (520) |
| wait | 0.2 | polls/run | -0.5 (0.4) | -1.3 (0.4) |
| wait | 0.2 | waste | +0.003 (0.005) | -0.003 (0.004) |
| wait | 0.4 | delay ms | -1016 (1405) | -323 (815) |
| wait | 0.4 | polls/run | -1.5 (0.5) | -1.2 (0.5) |
| wait | 0.4 | waste | +0.009 (0.005) | +0.003 (0.005) |

p 0.5/0.6 have no s5 stage; their s6 delay bias is +844 (1085) / -187 (1033) for short and -799 (1136) / -1776 (1300) for wait.

Reading:
- Delay: every s6 bias is within 2 SE of 0 except s6b0p5 p 0.6 (+400 ms, 2.7 SE, 2.8 %). The rules' delay bias is at most 3.7 % (s6wait p 0.6) and no larger than on s5. The replay is as trustworthy on the 8-12 s shape as it was on the 5 s slots, for delay and waste.
- Polls: the replay undercounts polls on the 0.5 s arm by 1.3 % (p 0.2) to 3.6 % (p 0.6), growing with p. This is the `0.035 p` loop-overhead term: live polls/run are flat in p (1745.5 to 1744.2), the replay's fall. Same effect as evidence/09 section 5.3, now visible over a wider p range. On the rules (about 120-160 polls/run) it is 0.4-1.8 %.
- The delay SE on the rules is 0.6-1.4 s, so a bias of up to about 2-3 s is not excluded at p 0.4 even though the point estimates are small.
- s6b10p043 cens: replay 7.0 % vs live 3.0 % at p 0.4, 14.0 % vs 22.3 % at p 0.5. The two shares count different things (record walks vs pairs) and are not a bias estimate.

### Replay [4] of the two unchanged rules (readings 1 and 2)

`replay_calib_front4.csv`. Seeds 1-20, CI95 from 400 seed-bootstrap reps shared by both rules and the four arms. The training-seed (21-120) values come from `replay_retune_train.csv` (no CI).

| p | s6short [4] (CI95) | s6wait [4] (CI95) | wait - short [4] (CI95) | wait - short delay ms (CI95) | training: short / wait [4], wait - short [4], wait - short delay |
|---|---|---|---|---|---|
| 0.2 | -2644 (-3378, -1903) | -2518 (-3300, -1781) | +127 (-566, 771) | +517 (-159, 1128) | -3843 / -3234, +609, +1016 |
| 0.4 | -5731 (-7143, -4138) | -4177 (-5547, -2718) | +1554 (785, 2494) | +2681 (1902, 3669) | -6967 / -5750, +1217, +2379 |

- Both unchanged rules stay below the fixed arms' line at both p (all CIs below 0).
- At p 0.4 the exact 5 s wait is worse than the jittered 3.695 s wait by 1.6 s in [4] (CI excludes 0); at p 0.2 they are not distinguishable on seeds 1-20 (training seeds: +0.6 s). On s5 the two were within 0.6 s of each other in paired replay [4], with wait the better one at p 0.4 (-8378 vs -7780; evidence/09 section 4). The live [4] for this comparison needs the live bytes (other half of evidence/10).

## Retune of the post-failure wait

Procedure as fixed before computing (module docstring of `replay_retune.py`). Training on regenerated schedules, seeds 21-120, p 0.2/0.4; test on the live schedules, seeds 1-20. [4] = rule's seed-mean delay - lower hull of the four fixed arms' seed-mean (bytes, delay) at the rule's bytes, ana.py's convention; arms evaluated in replay on the same seeds. "best" = smallest training [4]; "cap" = smallest training delay among candidates whose training bytes are <= the unchanged rule's. Deltas are pick - unchanged rule on test; CI95 from 400 seed-bootstrap reps with weights shared by pick, unchanged rule and arms. Training and test numbers are in `replay_retune.csv`; every candidate's training curve in `replay_retune_train.csv`.

| family | p | pick | w | train [4] (unchanged) | train bytes (unch.) | test [4] pick / unch. | test Δ[4] (CI95) | Δdelay ms (CI95) | Δbytes B/rh (CI95) | within 300 ms of best training [4] | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| short | 0.2 | best | 3.0 | -3948 (-3843) | 17.92 (17.48) | -3558 / -2644 | -914 (-1519, -356) | -1172 (-1796, -621) | +0.39 (0.33, 0.46) | w 1.75-4.0 (9 points; 3.75 is 332 ms off) | fail |
| short | 0.2 | cap | 3.75 | -3616 (-3843) | 17.45 (17.48) | -3140 / -2644 | -496 (-1045, 77) | -441 (-1006, 163) | -0.08 (-0.14, -0.03) | - | fail |
| two | 0.2 | best | (4.5, 2.0) | -4311 (-3843) | 17.84 (17.48) | -4079 / -2644 | -1435 (-2207, -745) | -1614 (-2409, -956) | +0.27 (0.19, 0.35) | 18 of 576 (w1 3.0-6.5, w2 1.0-3.0) | fail |
| two | 0.2 | cap | (5.5, 2.0) | -4091 (-3843) | 17.44 (17.48) | -3665 / -2644 | -1020 (-1886, -305) | -958 (-1820, -227) | -0.10 (-0.17, -0.02) | - | fail |
| wait | 0.2 | best | 3.25 | -3985 (-3234) | 17.75 (16.86) | -3501 / -2518 | -983 (-1530, -445) | -1547 (-2128, -1058) | +0.86 (0.77, 0.95) | w 1.75-3.75 (9 points, contiguous) | fail |
| wait | 0.2 | cap | 5.0 | -3234 (-3234) | 16.86 (16.86) | -2518 / -2518 | 0 | 0 | 0 | - | fail |
| short | 0.4 | best | 2.75 | -7489 (-6967) | 17.70 (16.49) | -5766 / -5731 | -35 (-595, 682) | -1224 (-1785, -491) | +1.24 (1.17, 1.31) | w 2.5-3.0 (3 points, contiguous) | fail |
| short | 0.4 | cap | 3.75 | -6742 (-6967) | 16.43 (16.49) | -5834 / -5731 | -103 (-474, 249) | -56 (-416, 298) | -0.05 (-0.10, 0.00) | - | fail |
| two | 0.4 | best | (4.5, 2.0) | -7807 (-6967) | 17.27 (16.49) | -6645 / -5731 | -914 (-1933, 73) | -1667 (-2712, -653) | +0.78 (0.66, 0.93) | 13 of 576 (w1 3.0-5.5, w2 1.5-3.0) | fail |
| two | 0.4 | cap | (4.5, 3.0) | -7545 (-6967) | 16.48 (16.49) | -5994 / -5731 | -263 (-967, 507) | -269 (-980, 498) | +0.01 (-0.05, 0.07) | - | fail |
| wait | 0.4 | best | 2.75 | -7437 (-5750) | 17.73 (15.34) | -6430 / -4177 | **-2254 (-3220, -1360)** | -4538 (-5624, -3417) | +2.38 (2.25, 2.53) | w 2.25-3.0 (4 points, contiguous) | fail (bias) |
| wait | 0.4 | cap | 5.0 | -5750 (-5750) | 15.34 (15.34) | -4177 / -4177 | 0 | 0 | 0 | - | fail |

R.Oracle(10.043) on the test seeds (bound no real rule can reach):

| p | oracle [4] | oracle delay ms | oracle B/rh | Δdelay vs s6short (CI95) | Δdelay vs s6wait (CI95) |
|---|---|---|---|---|---|
| 0.2 | -5965 | 23974 | 16.25 | -2544 (-3411, -1708) | -3061 (-3947, -2233) |
| 0.4 | -12241 | 27349 | 14.15 | -4307 (-5307, -3412) | -6988 (-8118, -5853) |

Gate per (family, p): delta[4] < -2000 ms AND its CI95 excludes 0 AND the unchanged rule's replay - live delay bias is well under 2 s. "Well under 2 s" was not given a number before computing; it is operationalised here as |bias| + 1.96 SE < 2000 ms (s6short: 1290 ms at p 0.2, 2269 ms at p 0.4; s6wait: 1926 ms at p 0.2, 3770 ms at p 0.4). For the bytes-capped pick the same test is applied to delta delay instead of delta[4].

- Only wait-form at p 0.4 meets the first two conditions (delta[4] -2254 ms, CI95 -3220 to -1360). It fails the bias condition under this operationalisation (s6wait p 0.4 bias -1016 ms, SE 1405 ms); with the point estimate alone (1.0 s) it would be borderline, not "well under".
- Its bytes-capped comparison does not pass: no wait-form candidate with bytes <= s6wait's is faster than s6wait itself (the capped pick is w = 5.0, the unchanged rule; on the training seeds delay rises and bytes fall monotonically with w, apart from noise). So [4] and the bytes-capped comparison disagree for wait-form p 0.4; under the procedure no live run happens without asking.
- The wait-form p 0.4 gain is mostly s6wait losing on this shape, not a gain over the better existing rule: its pick (w 2.75, test [4] -6430) is 700 ms better than unchanged s6short (-5731) on the same seeds, and the short-form pick at the same w gains only -35 ms over s6short.
- Every other family and p fails delta[4] < -2000 ms. The p 0.2 threshold in the map's reading 3 is not met by any family: best is two-step at -1435 ms (CI -2207 to -745), and its training-set gain is only -468 ms.
- Training vs test: on the training seeds the retuned gain in [4] is -105 (short), -468 (two), -751 (wait) ms at p 0.2 and -522 / -839 / -1687 ms at p 0.4. The test seeds 1-20 give larger gains at p 0.2 (-914 / -1435 / -983) because unchanged s6short scores worse there (test [4] -2644 vs training -3843). The 100 training seeds are the better estimate of the size of the gain.
- Optimum: both one-wait families settle at w about 2.75-3.25 s, whatever the jitter; the training curve is flat within 300 ms over w 1.75-4.0 at p 0.2 and 2.25-3.0 at p 0.4, with point-to-point noise of about 100-200 ms. The two-step optimum is (4.5, 2.0) at both p (wait longer after the first failure, shorter after more).

## Deviations and choices

- No deviation from the fixed procedure. Choices it left open: training uses 10 jitter reps per seed (replay_runs.py's paired count); "well under 2 s" operationalised as above; the bytes-capped pick's pass test uses delta delay; short-form and wait-form candidates at the same w share the same numeric seed tag `(10.043, w)` (as s5short/s5wait did with `(10.043, 3.695)` / `(10.043, 5.0)`), so s6wait's tag is the wait-form w = 5.0 tag and its test values are identical to the unchanged rule's.
- In `replay_retune.csv`, the columns `gate_d4_lt_m2000` and `gate_ci_excl0` test delta[4] on `pick == best` rows but delta delay on `pick == cap` rows.
- Candidates whose training bytes fall outside the arms' bytes range ([4] NaN, skipped): w >= 10.25 (10.0 at wait-form p 0.4) for the one-wait families, 61 (p 0.2) and 44 (p 0.4) of 576 two-step candidates.
