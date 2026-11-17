# ev13 spec2: replay of a "lengthen the interval while replies are empty" rule

Written 2026-11-17 after the validation of spec.md and before any number of the new rule was seen.
User decision Q62: trust only cell aggregates (Mode B); event-level agreement (Mode A) is not required.

## Part 1. Fixes to replay.py before it is used (from the two reviews)

1. Link boundary back to spec.md line 15: `t0 + actual_at*1000` for link_down AND link_up. validate.md then
   reports A1 39/42, A2 24/42, A3 41/42 as the result, and the split boundary (down = actual_at, up = ISO ts;
   40/25/42) as a variant tried after seeing results, marked "tuned".
2. Provisional send-time timer: engine.py:477 sets `_next_state_poll[neighbor] = now + applied_interval` on
   EVERY send, policy stages included; the resolution only overrides it. Implement that, including the
   consequence that with an interval below the 2 s lifetime the next poll is sent before the previous one
   resolved, and `_resolve_timed_out_poll` / the reply bookkeeping then behave as the source says (read
   engine.py around 440-540 and 600-660; cite lines in comments). The existing 42 cells must not move by this
   change (every existing policy row exceeds 2 s): show validate_cells.csv before/after is identical apart
   from the boundary change of item 1 (do item 2 first, compare, then item 1).
3. validate.md corrections: mark B1 timed_out as passing only after scorer fix 4; mark the serve-instant tie
   rule as chosen by score on 12 runs ("tuned"), and state that for record_created the source decides the tie
   the other way (agent.py:1072-1073, 861-872); fetch_timeout count is 12 % low (132 logged vs 116 simulated
   over 126 runs), not "well under 1 %"; the timer evidence is facts-timers §8, not Mode A (Mode A's lag is
   derived with the same equation); B2 cens4 must use a true median; the comment of delivers() must describe
   the code; facts-metrics.md "5 of 42 cells" is 7.

## Part 2. The rule

Per (node, neighbor) the rule keeps one number, the current interval I, and works inside the existing policy
mechanism (decide at send, re-decide at resolution with the same jitter draw):

- start: I = I_min.
- at send: applied = I * (1 + jf*(2u-1)), jf = 0.1, u from the node's jitter stream (one draw per poll, as
  now); provisional next poll = sent + applied.
- at resolution, answered: the reply is EMPTY when every seq 1..hi[o] of every origin o in the reply is
  already held by the polling node at that instant (the evidence/11 definition, rx_no_new; a reply with
  hi == {} is EMPTY). EMPTY: I = min(I * m, I_max). Not empty: I = I_min.
  Then next poll = max(sent + I_new * (1 + jf*(2u-1)), now), the same u.
- at resolution, timed out: I unchanged (the rule says nothing about outages; the existing short/wait
  policies cover that question and mixing them would confound the comparison).

Variants (name: I_min, m, I_max, seconds):
- T05:   0.5,   2,   10.043
- T136:  1.359, 2,   10.043
- T37:   3.695, 2,   10.043
- T136s: 1.359, 1.5, 10.043   (slower growth)
- T136c: 1.359, 2,   20.086   (higher cap)

Identity test (must hold exactly, before any variant is read): the rule with m = 1 and I_min = I_max = X,
run through the new-rule code path, gives the same simulated events as the fixed-interval arm X run through
the existing code path, for X in (0.5, 1.359, 3.695, 10.043), on 3 runs each. If the RNG stream usage of the
two paths differs (the b* arms draw a choice u and a jitter u), make the new-rule path consume the streams
the same way and say so.

## Part 3. What is simulated and how it is compared

Exogenous inputs (outages, record creation, agent_start) from a donor arm: s6b10p043 for p 0.2/0.4/0.5/0.6
(outage 8-12 s) and s5b10p043 for p 0/0.2/0.4 (the other outage shape); 20 seeds; Mode B; R = 5 replicates.
On the same inputs simulate 11 rules: the six existing ones (0.5, 1.359, 3.695, 10.043 fixed; short; wait,
with their meta.json settings taken from the stage of the same family) and the five variants.
All comparisons are simulated against simulated, paired by seed and replicate. The measured cell values of
the six existing rules are printed beside them for reference only.

Per cell and rule: delay (4-hop, uncensored mean, pooled), cens4, polls_sent, polls answered, polls timed
out, state-poll bytes (where REP_B exists; p 0.5/0.6 have none), and the share of answered polls that were
EMPTY. Bytes use the same REP_B for every rule (stated as an assumption: the mean reply size is taken not to
depend on the rule).

Headline, fixed before results: "at the same polls_sent, is the variant's delay lower than a fixed interval's?"
- Frontier of the fixed arms per cell: log(delay) as a piecewise-linear function of log(polls_sent) through
  the four simulated fixed arms. polls_sent is used instead of bytes so that p 0.5/0.6 are included; bytes are
  reported too.
- gain = 1 - delay(variant) / delay_frontier(polls_sent(variant)). Positive = the variant is faster than a
  fixed interval that costs the same number of polls. Also the reverse reading: polls saved at equal delay,
  save = 1 - polls(variant) / polls_frontier(delay(variant)).
- 95 % interval: bootstrap over the 20 seeds (2000 resamples, default_rng(20261117)), recomputing the frontier
  and the variant in each resample from the same resampled seeds. A variant outside the range of the four
  fixed arms gets no gain (reported as "outside").
- The same gain is computed for short and wait, so that the variants are read beside the two existing
  policies.
- cens4 is reported beside every gain; a gain bought with more censoring is flagged (cens4 higher than the
  frontier neighbours by more than 1 percentage point).

Not claimed: anything about real bytes on the wire, about a rule's effect on record-fetch traffic, or about
workloads other than this one (120 records per origin every 2 s from t0+13 s).

## Deliverables (in this directory)

- replay.py (fixed, with a `rule` argument to the simulator), validate_cells.csv, validate.md (regenerated).
- trickle.py: runs Part 3; trickle_cells.csv (one row per cell x rule), trickle_gain.csv (one row per
  cell x variant-or-policy: gain, save, intervals, cens flag), trickle.md (tables + the identity test result).
- No tuning: the variants above are the only ones run. If something forces a deviation, write it in
  trickle.md.
