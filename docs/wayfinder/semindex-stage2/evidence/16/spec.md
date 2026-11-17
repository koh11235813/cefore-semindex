# evidence/16 spec: one testbed check of the timeout-reset rule

Written 2026-11-18, before any run of this step. User decisions Q65 to Q67 (ticket
`../../tickets/05-timeout-reset-rule.md`). The pass lines below are copied from the ticket's Acceptance and
are not moved after the runs.

## Question

The replays of evidence/13 to 15 left one difference they could state firmly: in the cell with many outages
(s6 shape, p 0.4), a rule that returns the state-poll interval to its shortest value on a timeout (T37r) is
better than the same rule that leaves the interval where it was (T37). No measurement exists for either
rule. This step measures both on the testbed and compares them with the prediction.

Not answered here: whether a learned policy beats a formula; other p; other workloads; the variant that
lengthens on a timeout.

## Rules

`--poll-rule`, both with `i_min_s` 3.695, `factor` 2, `i_max_s` 10.043, `jitter_frac` 0.1:

- T37: `on_timeout` "keep"
- T37r: `on_timeout` "reset"

## Runs (`main6.sh`, from a detached clean checkout of the merged commit)

Common settings, as the s6 stages of evidence/10: `--k 4 --store-synchronous NORMAL
--disconnect-duration-s 8 12`. The outage schedule depends on (p, seed) only, so the new stages see the
schedules of the s6 stages.

1. Reruns of three existing settings with the new code, p 0.4, seed 1, under new stage names
   (`s7short`, `s7b3p695`, `s7b10p043`). Run first. 3 runs.
2. `s7t37` and `s7t37r`, p 0.4, seeds 1 to 20. 40 runs. Started only if pass line 4 holds.

Before each part: no emulator and no Cefore daemon running; the checkout is not edited while runs load it.

## Prediction (`predict.py`, output `predict.txt`, both placed here before the runs)

The evidence/15 engine on the logged workload and outages of `s6b10p043` p 0.4, seeds 1 to 20, 40 replicate
streams; one stream is one experiment of 20 single runs, the testbed's condition. Medians over the 40
experiments:

| | T37 | T37r |
|---|---|---|
| delay (4 hops, uncensored mean) | 34.627 s | 25.160 s |
| state polls per run | 142.50 | 180.15 |
| censored share | 0.016 | 0.000 |

Gain difference T37r minus T37 against the line through the two fixed arms 3.695 s and 10.043 s: mean
+15.9 points over the 40 experiments, range +9.3 to +22.0; the 95 % seed-bootstrap interval lies entirely
above zero in 38 of 40.

## Pass lines

1. All 40 runs valid (no `evalvalid` flag).
2. Is the difference seen? Read by the gain difference T37r minus T37, computed as follows.
   - delay = over all records created inside the window, the time until stored 4 hops away, mean of the
     uncensored ones (the `delay` of evidence/10: `lat4 / n4` summed over the runs of a stage). The censored
     share (`c4 / (n4 + c4)`) is reported beside it.
   - line = the measured `s6b3p695` and `s6b10p043` at p 0.4 (per-seed values of `../10/perrun.csv`), state
     polls per run on the horizontal axis and delay on the vertical, joined by a straight line in log-log.
   - gain = 1 - delay(rule) / delay on the line at the rule's polls per run. If a rule's polls per run fall
     outside the two arms, no gain is computed and that is reported.
   - interval = 2000 resamples of seeds 1 to 20 with `default_rng(20261117)`; in each resample the four
     stages (two arms, two rules) are recomputed from the same resampled seeds.
   - Entirely above zero: "seen". Spanning zero: "not seen". Entirely below zero: "reversed".
3. Did the replay predict it? For each of the two stages, measured delay and measured polls per run inside
   plus or minus 6 % of the prediction: T37 delay 32.549 to 36.704 s and polls 133.95 to 151.05; T37r delay
   23.650 to 26.669 s and polls 169.34 to 190.96. The 6 % comes from the largest deviation the replay showed
   on the existing rules (5.9 %, evidence/13). The replay has never been compared with a measurement on a
   rule that reacts to timeouts, so if a value falls outside, the decision sequence around timeouts is
   compared between the testbed log and the replay and the point of disagreement is reported before any
   conclusion on line 2.
4. Existing policies unchanged.
   - The existing policy tests pass without being rewritten, with the added test of an undecodable reply.
     (Checked at merge.)
   - In the three reruns, every `interval_s` of the decision log is a value the setting allows: for
     `s7short` 10.043 with zero consecutive timeouts and 3.695 otherwise, in `state_poll` and
     `state_poll_reschedule` alike; for `s7b3p695` 3.695; for `s7b10p043` 10.043. Exact.
   - In the same three runs, polls sent within 5 % and delay within 20 % of the original run (`s6short`,
     `s6b3p695`, `s6b10p043`, p 0.4, seed 1, from `../10/perrun.csv`). Answered against timed out is a race
     on the testbed, so a decision-by-decision match is not required; this band is a coarse guide and the
     two checks above are the exact ones.
   - If any of these fails, stop before the 40 runs and report.

Also reported, without a pass line: the share of answered polls whose reply was empty (`empty` true), the
number of undecodable replies under a rule (`empty` null with trigger answered), `state_poll_late_reply`
counts and clamped re-decisions. The replay treats an unreadable reply as empty and has no late reply, so
these say whether the two can have diverged for that reason.

## Deliverables (this directory)

`main6.sh`, `predict.py`, `predict.txt`, this file, before the runs. After the runs: the analysis script,
its csv and md output, `README.md`, `RUN.txt`.
