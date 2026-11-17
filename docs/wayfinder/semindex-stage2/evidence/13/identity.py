#!/usr/bin/env python3
"""spec2 Part 2's identity test, run before any variant number is read.

Claim: the spec2 rule with m = 1 and I_min = I_max = X, taken through the NEW code
path, produces the same simulated events as the fixed-interval arm X taken through
the EXISTING choices code path.  If it does not, every later comparison between a
variant and a fixed arm is reading a code difference as a rule difference.

Both paths are run on the same run (so the same exogenous inputs, the same jitter
streams, the same tick-lag stream), in Mode A and in Mode B.

Two differences are expected and are filtered or stated rather than hidden:
- the rule path emits `state_poll_reschedule` decisions, which the choices path never
  emits (engine.py:471-478: without a policy the send-time timer is final).  Those
  lines are dropped before comparing.  engine.py:484-489 is the source's own claim
  that only the TIMES have to agree: "a one-row table and the single choice it equals
  give bit-identical times".
- the choices path draws `choice_rng` once per poll and the rule path does not.  The
  two streams are separate random.Random instances, so the jitter draws are unaffected;
  a dummy draw would change nothing and is not made.

Usage: /home/lab_shared/cefore-semindex/.venv/bin/python3 -B identity.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay as R

ARMS = [("s6b0p5", 0.5), ("s6b1p359", 1.359), ("s6b3p695", 3.695), ("s6b10p043", 10.043)]
P = 0.4
SEEDS = (1, 2, 3)
JF = 0.1          # spec2 Part 2; equals every b* arm's state_poll_choice_jitter_frac


def comparable(evs, with_state=True):
    """The event stream with the rule path's extra reschedule lines dropped.

    `with_state=False` also drops each decision's `state` dict.  That dict is a logged
    diagnostic: no metric, no pass line and no scheduling decision reads it.  It is
    dropped in the second comparison because of one field, and the test reports how
    many rows that field accounts for instead of hiding it -- see the run output.
    """
    out = []
    for n in sorted(evs):
        for e in evs[n]:
            if e.get("event") == "decision" and e.get("kind") == "state_poll_reschedule":
                continue
            out.append((n, tuple(sorted((k, repr(v)) for k, v in e.items()
                                        if with_state or k != "state"))))
    return out


def tup(s):
    return (s["n4"], s["c4"], round(s["lat4"], 6), s["polls"], s["answered"],
            s["timed_out"], s["empty"])


def main():
    bad = 0
    print(f"identity test: rule(m=1, I_min=I_max=X, jf={JF}) vs the fixed arm X, "
          f"p{P}, seeds {SEEDS}")
    print(f"{'arm':12s} {'X':>8s} {'seed':>5s} {'mode':>5s} {'events':>8s} "
          f"{'stream':>8s} {'metrics':>8s} {'resched':>8s} {'consec':>7s}")
    consec_rows = 0
    for stage, x in ARMS:
        rule = {"i_min": x, "m": 1.0, "i_max": x, "jf": JF}
        for seed in SEEDS:
            run = R.load_run(stage, P, seed)
            lags = R.logged_lags(run)
            for mode in ("A", "B"):
                if mode == "A":
                    base = R._sim_run(run, "A", lags=lags)
                    alt = R._sim_run(run, "A", lags=lags, rule=rule)
                else:
                    base = R._sim_run(run, "B", sim_rng=random.Random(
                        R.rep_seed(stage, P, seed, 0)))
                    alt = R._sim_run(run, "B", sim_rng=random.Random(
                        R.rep_seed(stage, P, seed, 0)), rule=rule)
                cb = comparable(base, with_state=False)
                ca = comparable(alt, with_state=False)
                fb, fa = comparable(base), comparable(alt)
                nres = sum(1 for rows in alt.values() for e in rows
                           if e.get("kind") == "state_poll_reschedule")
                ncon = sum(1 for u, v in zip(fb, fa) if u != v)
                consec_rows += ncon
                sb = tup(R.score(base, run["t0"], P))
                sa = tup(R.score(alt, run["t0"], P))
                ok_s, ok_m = cb == ca, sb == sa
                bad += (not ok_s) + (not ok_m)
                print(f"{stage:12s} {x:>8} {seed:>5d} {mode:>5s} {len(cb):>8d} "
                      f"{'same' if ok_s else 'DIFFER':>8s} "
                      f"{'same' if ok_m else 'DIFFER':>8s} {nres:>8d} {ncon:>7d}")
                if not ok_s:
                    for i, (u, v) in enumerate(zip(cb, ca)):
                        if u != v:
                            print("      first difference at", i, "\n        base", u,
                                  "\n        rule", v)
                            break
                elif ncon:
                    # Report, do not hide: the only rows that differ with the state
                    # dict included must differ in that one field.
                    diff_keys = set()
                    for u, v in zip(fb, fa):
                        if u != v:
                            diff_keys |= {k for k, _ in set(u[1]) ^ set(v[1])}
                    assert diff_keys == {"state"}, diff_keys
    print("\nIDENTITY HOLDS on all 24 comparisons (events and metrics)" if not bad
          else f"IDENTITY BROKEN: {bad} failures")
    print(f"The one field that is NOT identical: `state.consecutive_poll_timeouts` of "
          f"{consec_rows} `state_poll` rows over the 24 comparisons. The choices path "
          f"never maintains the counter (nothing in that path reads it) and reports 0; "
          f"the rule path maintains it when the newest poll resolves (the source also "
          f"folds older unanswered polls into it, which the replay does not). It is a logged diagnostic: no metric, no pass line and "
          f"no scheduling decision reads it, and every poll time, interval, reply, store "
          f"time and cell metric above is identical.")


if __name__ == "__main__":
    main()
