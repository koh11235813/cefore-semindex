#!/usr/bin/env python3
"""Pin the formula F to one member the present testbed code can run.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src \
        /home/lab_shared/cefore-semindex/.venv/bin/python3 -B fpin.py > fpin.txt
    (the same with a first argument 2.2 and `> fpin_floor22.txt` re-measures the floor)

WHY THIS EXISTS
  evidence/18 reports F as a cross-fitted procedure, not as one rule: fold A (seeds
  1-10) and fold B (seeds 11-20) select different members in the main cell
  s6b10p043 p 0.2 W0 (evidence/18/bound.md "F, the formula point").  A testbed stage
  has to launch one `--poll-rule` JSON string, so the member has to be fixed before
  any run.  This script re-runs evidence/18's own coordinate ascent on seeds 1 to 20
  pooled as one set, over the sub-family `src/semindex` can express, and prints the
  winner.

WHAT IT REUSES
  Everything simulated here comes from `evidence/18/bound.py` by import: its engine
  subclass (`sim`), its member rule dict (`member_rule`), its per-member job
  (`job_member`), its seven-arm frontier (`Frontier`) and its cell row (`cell_row`).
  No simulation code is copied.  The only new code is the narrowed coordinate grid,
  the pooled-seed driver and the printing.

WHAT IT CANNOT SHOW
  The winner's gain is in-sample on seeds 1 to 20: the same 20 seeds choose it and
  score it.  It is not an estimate of its gain on new seeds, and it must not be read
  as one.  No number here has a testbed measurement behind it.
"""

import json
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EV18 = os.path.normpath(os.path.join(HERE, "..", "18"))
sys.path.insert(0, EV18)
import bound as B  # noqa: E402  evidence/18's engine, frontier, members and jobs

assert B.__file__.startswith(EV18), B.__file__

DONOR, P, WL = "s6b10p043", 0.2, "W0"
SEEDS = B.SEEDS                       # 1..20, pooled as one selection set
WORKERS = min(B.WORKERS, os.cpu_count() or 1)

# The action range of this plan (../../map.md, the 2026-11-22 section): the next state-poll interval
# lives in [3.695, 10.043] s, so no coordinate of F may leave that band.  Within the
# band the grid is evidence/18's own (ARM_INTERVALS and M_VALUES).
# 2026-11-22: an optional first argument moves the floor, so that what the tighter floor
# costs can be re-measured (`fpin.py 2.2 > fpin_floor22.txt`, evidence/18's own floor).
# Without it the script is the one that pinned F, and fpin.txt is its output.
I_MIN_FLOOR = float(sys.argv[1]) if len(sys.argv) > 1 else 3.695
I_MAX_CEIL = 10.043
# `src/semindex` has no k_cap at all, and `engine.StatePollRule.on_timeout` offers
# exactly "keep" and "reset", where reset goes to `i_min_s`.  So k_cap is pinned to
# None and the timeout coordinate holds two values, one of which tracks i_min.
K_CAP = None
# The five coordinates the ascent sweeps, in the order `coord_values` indexes them.
# Named so that each sweep's record can say which coordinate moved, which is what
# ticket 07's Acceptance 1 asks the script's output to carry.
COORD_NAMES = ("i_min_s", "factor", "i_max_s", "on_timeout", "k_cap")

# evidence/18/bound.md, the F table, row `s6b10p043 p0.2 W0 F_restricted`.
FOLD_MEM = {"A": (6.0, 1.2, 10.043, 3.695, None),
            "B": (3.695, 1.5, 10.043, 3.695, None)}
PUB_F_GAIN = 10.7                               # points, the pooled out-of-sample gain
# The rules whose published row this script re-derives: the seven fixed arms, which are
# the frontier every gain is read against, and T37r, which is the ascent's start and the
# second baseline of the stage.  Values are read from bound_cells.csv, never hard coded.
CHECKED = tuple(B.FIXED) + ("T37r",)


TIMEOUT_NOTE = """
Where `on_timeout: reset` on the testbed and `i_timeout = i_min` in the replay differ,
read by comparing src/semindex/engine.py with evidence/13/replay.py and
evidence/18/bound.py line by line:
  Same: a timeout resolves the newest poll only (engine.py _resolve_timed_out_poll,
  replay.py resolve's first guard); the interval then goes to i_min; the next poll is
  due at the poll's own send time plus that interval with the poll's own send-time
  jitter draw, clamped forward to now (engine.py _redecide_next_poll, replay.py
  resolve's `due_at = sent_at + applied * 1000.0` and `max(due_at, t)`); a reply that
  brought nothing new multiplies by factor up to i_max, one that brought something new
  drops to i_min (engine.py _advance_rule_interval, replay.py resolve).
  One difference found: a reply that arrives and fails to decode hands the testbed
  `empty is None`, which says a reply came back and could not be read. No branch of
  engine.py _advance_rule_interval fires on it, so the testbed leaves the interval
  where it is and a backed-off interval stays backed off. The replay's `if empty` is
  falsy for None, so the replay drops the interval to i_min and lumps an unreadable
  reply together with one that brought something new. The replay never produces that
  case: evidence/13/replay.py's `no_new` returns a bool for every simulated reply, so
  no number printed above moves. On the testbed a decode failure would leave the
  interval backed off, so the rule would poll later than the replay says.
  One difference in reach, not in behaviour: evidence/18's `i_timeout` may be any arm
  interval, and `--poll-rule` can only reset to i_min_s. That is why evidence/18's
  fold A winner cannot be launched, and why this family forces the reset target onto
  i_min (see `norm`)."""


def in_family(mem):
    """Whether one evidence/18 member is a rule the present testbed code can launch."""
    i_min, m, i_max, i_timeout, k_cap = mem
    return (k_cap is None and m in B.M_VALUES
            and I_MIN_FLOOR <= i_min <= I_MAX_CEIL
            and i_min <= i_max <= I_MAX_CEIL
            and (i_timeout == "keep" or i_timeout == i_min))


def norm(mem):
    """A member with its reset target put back on i_min.

    Added restriction over evidence/18, forced by `engine._advance_rule_interval`:
    a reset goes to `i_min_s` and nowhere else.  Moving the i_min coordinate therefore
    has to move the reset target with it, or the ascent would walk out of the family.
    """
    return mem if mem[3] == "keep" else mem[:3] + (mem[0],) + mem[4:]


def coord_values(mem, c):
    """The values coordinate `c` takes, in evidence/18's own order, narrowed."""
    if c == 0:
        return [x for x in B.ARM_INTERVALS if I_MIN_FLOOR <= x <= I_MAX_CEIL]
    if c == 1:
        return list(B.M_VALUES)
    if c == 2:
        return [x for x in B.ARM_INTERVALS if mem[0] <= x <= I_MAX_CEIL]
    if c == 3:
        return ["keep", mem[0]]
    return [K_CAP]


def family():
    """Every member of the narrowed family, so the ascent can be told whether it
    reached the family optimum or stopped below it."""
    out = []
    for i_min in coord_values(None, 0):
        for m in B.M_VALUES:
            for i_max in [x for x in B.ARM_INTERVALS if i_min <= x <= I_MAX_CEIL]:
                for tmo in ("keep", i_min):
                    out.append((i_min, m, i_max, tmo, K_CAP))
    return out


def rule_json(mem):
    """The `--poll-rule` string that launches this member on the testbed."""
    i_min, m, i_max, i_timeout, _k = mem
    return json.dumps({"i_min_s": i_min, "factor": m, "i_max_s": i_max,
                       "jitter_frac": B.T.JF,
                       "on_timeout": "keep" if i_timeout == "keep" else "reset"})


def line(tag, gain, delay, polls, boot=None):
    g = "no gain (poll count outside the arms)" if gain is None else f"{gain:+.1f}"
    ci = ""
    if boot is not None:
        lo, hi, n = B.ci_pts(boot)
        if lo is not None:
            ci = f" ({lo:+.1f} .. {hi:+.1f}, {n} resamples)"
    print(f"  {tag:<44} gain {g}{ci}  delay {delay:.1f} ms  polls {polls:.2f} per run")


def main():
    with mp.Pool(WORKERS) as pool:
        # ---- step 1: reproduce evidence/18's published numbers for this cell ----
        res = pool.map(B.job_base, [(DONOR, P, s, (WL,)) for s in SEEDS], chunksize=1)
        per_seed = {(k[1], s): v for _d, _p, s, out in res for k, v in out.items()
                    if k[0] == WL}
        records = [out[("records", WL)] for _d, _p, _s, out in res]
        pub, _cols = B.read_cells(os.path.join(EV18, "bound_cells.csv"))
        bad = []
        print(f"Cell {DONOR} p {P} {WL}, seeds {SEEDS[0]} to {SEEDS[-1]}, "
              f"{B.REPS} Mode B replicates, through evidence/18's engine.")
        print("\nStep 1, reproduction of evidence/18/bound_cells.csv "
              "(early-record delay and polls per run):")
        for name, kind, _spec in B.RULE_SPECS:
            if name not in CHECKED:
                continue
            row = B.cell_row(WL, DONOR, P, name, kind,
                             [per_seed[(name, s)] for s in SEEDS], records)
            want = pub[(WL, DONOR, P, name)]
            ok = (f"{row['early_delay']:.6f}" == want["early_delay"]
                  and f"{row['early_polls_sent']:.6f}" == want["early_polls_sent"])
            bad += [] if ok else [name]
            print(f"  {name:<12} delay {row['early_delay']:.6f} ms "
                  f"(csv {want['early_delay']})  polls {row['early_polls_sent']:.6f} "
                  f"(csv {want['early_polls_sent']})  {'same' if ok else 'DIFFERENT'}")

        arm_mats = {n: B.seed_mat([per_seed[(n, s)] for s in SEEDS], 1, 4)
                    for n in B.FIXED}
        fr = B.Frontier(arm_mats, P)

        # F_restricted's pooled out-of-sample value: each fold's seeds are scored with
        # the OTHER fold's member, exactly as evidence/18/bound.py f_stage pools them.
        # The two members are read from bound.md; the selection itself is not re-run.
        oos = {}
        for fold, seeds in B.FOLDS.items():
            other = "B" if fold == "A" else "A"
            mem = FOLD_MEM[other]
            for key, reps, _w in pool.map(
                    B.job_member, [(DONOR, P, s, WL, mem) for s in seeds], chunksize=1):
                oos[key[3]] = reps
        fmat = B.seed_mat([oos[s] for s in SEEDS], 1, 4)
        fg, _fboot, fv = fr.gain(fmat)
        fg = B.pct(fg)
        fwant = pub[(WL, DONOR, P, "F_restricted")]
        f_ok = (fg is not None and f"{fg:.1f}" == f"{PUB_F_GAIN:.1f}"
                and f"{fv['delay']:.6f}" == fwant["early_delay"]
                and f"{fv['polls_sent']:.6f}" == fwant["early_polls_sent"])
        bad += [] if f_ok else ["F_restricted pooled out of sample"]
        print(f"  {'F_restricted':<12} gain {fg:+.1f} (bound.md {PUB_F_GAIN:+.1f})  "
              f"delay {fv['delay']:.6f} ms (csv {fwant['early_delay']})  "
              f"polls {fv['polls_sent']:.6f} (csv {fwant['early_polls_sent']})  "
              f"{'same' if f_ok else 'DIFFERENT'}")
        if bad:
            print("\nSTOP: the reproduction failed for " + ", ".join(bad) + ".")
            print("Nothing below would rest on evidence/18's numbers, so no search "
                  "was run. Reproduce this cell first.")
            return 1
        print("\nReproduced to the printed digits of bound_cells.csv: the seven fixed "
              "arms' and T37r's early-record delay and polls per run, and "
              "F_restricted's pooled out-of-sample delay and polls per run, plus its "
              "pooled out-of-sample gain to the one digit bound.md prints.")
        print("The two fold members were read from bound.md; the selection that chose "
              "them was not re-run, so the cross-fitted search itself is unchecked.")

        # ---- step 2: the coordinate ascent on seeds 1 to 20 pooled ----
        cache = {}

        def ev(mem):
            if mem not in cache:
                res = pool.map(B.job_member,
                               [(DONOR, P, s, WL, mem) for s in SEEDS], chunksize=1)
                by = {r[0][3]: r[1] for r in res}
                mat = B.seed_mat([by[s] for s in SEEDS], 1, 4)
                g, b, v = fr.gain(mat)
                cache[mem] = dict(gain=B.pct(g), boot=b, delay=v["delay"],
                                  polls=v["polls_sent"])
            return cache[mem]

        def rank(mem):
            g = ev(mem)["gain"]
            return -float("inf") if g is None else g

        cur = B.START_MEMBER            # T37r, as evidence/18's search_f starts
        assert in_family(cur), cur
        rank(cur)
        order = [cur]
        # The whole family rests on `i_timeout == i_min` being the same thing as the
        # testbed's `on_timeout: reset`.  In the replay the two are different code
        # paths: evidence/15/onoff.py's `reset` writes `rule["i_min"]`, evidence/18's
        # `i_timeout` writes the literal value.  T37r sits on both, so its two
        # encodings have to produce the same row, or the family is not what it says.
        t37r = B.cell_row(WL, DONOR, P, "T37r", "rule",
                          [per_seed[("T37r", s)] for s in SEEDS], records)
        enc_ok = (f"{ev(cur)['delay']:.6f}" == f"{t37r['early_delay']:.6f}"
                  and f"{ev(cur)['polls']:.6f}" == f"{t37r['early_polls_sent']:.6f}")
        print(f"\nT37r as an evidence/15 `on_timeout: reset` rule and as the member "
              f"(3.695, 2.0, 10.043, 3.695, no cap): "
              f"{'the same row' if enc_ok else 'DIFFERENT rows'}.")
        if not enc_ok:
            print("STOP: `i_timeout = i_min` is not the same as `on_timeout: reset` "
                  "in the replay, so this family does not map onto `--poll-rule`.")
            return 1
        # One entry per sweep, so that the member each sweep ended on can be printed
        # after the header below. Printing inside the loop would put the trajectory
        # above the header that explains what the ascent is.
        trail = []
        for sweep in range(3):
            moved = []
            for c in range(5):
                cands = []
                for val in coord_values(cur, c):
                    mem = norm(cur[:c] + (val,) + cur[c + 1:])
                    if mem[2] < mem[0] or mem in cands:
                        continue        # i_max is a value at or above i_min
                    cands.append(mem)
                best, bestr = None, None
                for mem in cands:
                    r = rank(mem)       # ties go to the earliest value: strict >
                    if mem not in order:
                        order.append(mem)
                    if best is None or r > bestr:
                        best, bestr = mem, r
                if best != cur:
                    moved.append(COORD_NAMES[c])
                cur = best
            trail.append((sweep + 1, cur, ev(cur)["gain"], tuple(moved)))
        for mem in family():            # the rest of the family, for the ranking only
            ev(mem)

        # ---- step 3: the winner and what stands beside it ----
        print(f"\nStep 2, coordinate ascent from T37r, 3 sweeps, ties to the earliest "
              f"value, objective the early-record gain against the seven arms of the "
              f"same 20 seeds. Family {len(family())} members, "
              f"{len(order)} visited by the ascent.")
        print("Family: i_min and i_max on evidence/18's arm grid inside "
              f"[{I_MIN_FLOOR}, {I_MAX_CEIL}] s with i_max >= i_min, factor in "
              f"{B.M_VALUES}, jitter_frac {B.T.JF}, on a timeout keep or reset to "
              "i_min, no k_cap.")
        print("\nWhat each of the three sweeps ended on, in order:")
        for n, mem, g, moved in trail:
            gt = "no gain (poll count outside the arms)" if g is None else f"{g:+.1f}"
            print(f"  sweep {n}: {rule_json(mem)}")
            print(f"           gain {gt}, coordinates that moved in this sweep: "
                  f"{', '.join(moved) if moved else 'none'}")
        print("\nWinner (the pinned F):")
        print(f"  --poll-rule '{rule_json(cur)}'")
        print(f"  {B.member_text(cur)}")
        d = ev(cur)
        line("F pinned", d["gain"], d["delay"], d["polls"], d["boot"])
        print("  The gain above is IN SAMPLE on seeds 1 to 20: the same 20 seeds "
              "chose this member and scored it.")
        print("  It is not an estimate of this member's gain on new seeds, and the "
              "bootstrap interval does not make it one.")

        print("\nBeside it, on the same 20 seeds and the same frontier:")
        d = ev(B.START_MEMBER)
        line("T37r (i_min 3.695, m 2, i_max 10.043, reset)",
             d["gain"], d["delay"], d["polls"], d["boot"])
        for fold in ("A", "B"):
            mem = FOLD_MEM[fold]
            if in_family(mem):
                d = ev(mem)
                line(f"evidence/18 fold {fold} winner", d["gain"], d["delay"],
                     d["polls"], d["boot"])
            else:
                print(f"  evidence/18 fold {fold} winner {B.member_text(mem)}")
                print("    outside the family: its reset target is not i_min, and "
                      "engine.StatePollRule resets to i_min_s only.")

        print("\nTop five of the members the ascent visited, in sample on seeds "
              "1 to 20:")
        vis = [m for m in order if cache[m]["gain"] is not None]
        for i, mem in enumerate(sorted(vis, key=lambda m: -cache[m]["gain"])[:5]):
            d = cache[mem]
            print(f"  {i + 1}. {rule_json(mem)}"
                  f"{'  <- the pinned F' if mem == cur else ''}")
            line("", d["gain"], d["delay"], d["polls"])
        rankable = [m for m in cache if cache[m]["gain"] is not None]
        best_all = max(rankable, key=lambda m: cache[m]["gain"])
        print(f"\nEvery member of the family was scored as well, beyond the ascent's "
              f"own path. The ascent's winner is "
              f"{'also' if best_all == cur else 'NOT'} the best of the "
              f"{len(rankable)} scorable members on these 20 seeds"
              + ("." if best_all == cur else
                 f"; the best is {rule_json(best_all)} at "
                 f"{cache[best_all]['gain']:+.1f}. The pin stays the ascent's winner, "
                 "because the search is the thing evidence/18 fixed."))
        top = sorted(rankable, key=lambda m: -cache[m]["gain"])[:3]
        print("\nThe family-wide top three, so that the winner's margin can be read "
              "here and not off the ascent's own path:")
        for i, mem in enumerate(top):
            d = cache[mem]
            print(f"  {i + 1}. {rule_json(mem)}"
                  f"{'  <- the pinned F' if mem == cur else ''}"
                  f"{'' if mem in order else '  (never visited by the ascent)'}")
            line("", d["gain"], d["delay"], d["polls"], d["boot"])
        if len(top) > 1:
            gap = cache[top[0]]["gain"] - cache[top[1]]["gain"]
            print(f"  The winner leads the runner-up by {gap:.2f} points. Both "
                  "bootstrap intervals are printed above, and they are about 10 "
                  "points wide, so these 20 seeds rank the two members without "
                  "separating them.")
        print(TIMEOUT_NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
