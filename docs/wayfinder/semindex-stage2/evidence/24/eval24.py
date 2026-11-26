#!/usr/bin/env python3
"""Evaluate decision maker M and the four no-model comparators on the replay port.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B eval24.py [--control | --dry]

Modes:
  --control   Run only the 4 no-model deciders (const3.695, const6.000, const10.043, F)
              on seeds 81 to 100 x 40 streams (3200 rows), write eval24_control.csv, and
              assert byte-for-byte equality against ../23/eval.csv in every column except
              script_sha.  Requires no coefficient files.
  --dry       Run all 6 deciders on tuning seeds 1 and 2, streams 0 and 1 (24 rows),
              write eval24_dry.csv, and gate the 16 no-model rows against ../23/eval_dry.csv.
  (default)   Run all 6 deciders on selection seeds 81 to 100 x 40 streams (4800 rows),
              write eval24.csv, and gate the 3200 control rows against BOTH ../23/eval.csv
              (all columns except script_sha) and eval24_control.csv (every column).
"""
import collections
import hashlib
import os
import sys

for _name, _value in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
                      ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_name, _value)

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
EV23 = os.path.normpath(os.path.join(HERE, "..", "23"))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if EV22 not in sys.path:
    sys.path.insert(0, EV22)

import label as LBL     # noqa: E402
import m24              # noqa: E402
import port as PORT     # noqa: E402
import score as SCORE   # noqa: E402

R = PORT.R
ARMS = PORT.ARMS
MAIN_P = 0.2
WEIGHT = LBL.WEIGHT
NREP = LBL.PR.NREP
LAG_PREFIX = "pairlag:%s" % LBL.PR.STREAM_LABEL

DRY_SEEDS = (1, 2)
DRY_REPS = (0, 1)
REAL_SEEDS = tuple(range(81, 101))
REAL_REPS = tuple(range(NREP))

ARM_NAMES = tuple("const%.3f" % arm for arm in ARMS)
CONTROL_ORDER = ARM_NAMES + ("F",)
FULL_ORDER = CONTROL_ORDER + ("a", "b")

COLUMNS = ("decider", "seed", "stream", "n4", "c4", "lat_ms", "censored_ms", "polls",
           "delay_ms", "score_ms", "score_dropped_ms", "weight_ms_per_poll",
           "reads", "sends", "ties", "n_short", "n_mid", "n_long",
           "ord_poll", "ord_resched", "k", "script_sha", "model_sha")


class Fail(Exception):
    """A gate, control comparison, or invariant that did not hold."""


def say(text=""):
    """Print one line of output."""
    print(text, flush=True)


def head(title):
    """Print a section header."""
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


def file_sha256(path):
    """Hex sha256 digest of a file."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def script_digests():
    """Return `(entries, combined_sha)` covering eval24.py and its imported modules."""
    paths = [
        ("eval24.py", os.path.abspath(__file__)),
        ("m24.py", os.path.join(HERE, "m24.py")),
        ("features.py", os.path.join(HERE, "features.py")),
        ("charges.py", os.path.join(HERE, "charges.py")),
        ("../22/port.py", os.path.join(EV22, "port.py")),
        ("../22/label.py", os.path.join(EV22, "label.py")),
        ("../22/score.py", os.path.join(EV22, "score.py")),
        ("../13/replay.py", os.path.realpath(R.__file__)),
    ]
    entries = [(name, file_sha256(p)) for name, p in paths]
    h = hashlib.sha256()
    for name, sh in entries:
        h.update(("%s:%s\n" % (name, sh)).encode("utf-8"))
    return entries, h.hexdigest()


class FirstCallEmpty:
    """Wrapper asserting per-pair memory is empty on the simulation's first call."""

    def __init__(self, inner, where, keeps):
        """Store wrapped `inner` decider, `where` label, and `keeps` flag."""
        self.inner = inner
        self.where = where
        self.keeps = keeps
        self.checked = False

    def memory(self):
        """Return `mem` (M) or `ival` (F), refusing if stateful and missing both."""
        for attr in ("mem", "ival"):
            if hasattr(self.inner, attr):
                return getattr(self.inner, attr)
        if self.keeps:
            raise Fail("%s: stateful decider lacks `mem` and `ival`" % self.where)
        return None

    def __call__(self, obs):
        """Assert empty memory on the very first call, then delegate."""
        if not self.checked:
            self.checked = True
            mem = self.memory()
            if mem is not None and len(mem) != 0:
                raise Fail("%s: non-empty memory at first call: %d entry" % (self.where, len(mem)))
        return self.inner(obs)


def keeps_memory(decider):
    """Return True if `decider` keeps per-pair state across decisions."""
    return isinstance(decider, (m24.M, LBL.F))


def check_choices(run, where):
    """Assert the run's choice set equals `port.union_choices()`."""
    want = list(LBL.CHOICES)
    if list(run["choices"]) != want:
        raise Fail("%s: choices %r != union_choices %r" % (where, run["choices"], want))
    return len(want)


def check_counters(reads, sends, ord_resched, ord_poll, where):
    """Assert M's call counters match the engine's own `decider_ord` counters."""
    if reads != ord_resched or sends != ord_poll:
        raise Fail("%s: reads/sends (%d, %d) != engine (%d, %d)"
                   % (where, reads, sends, ord_resched, ord_poll))


def count_mind_changes(events):
    """Count decisions whose resolution-time returned interval differs from that pair's
    previous send-time (previously applied) interval, plus total resolution decisions.
    """
    last_send_ival = {}
    changes = 0
    res_total = 0
    for node, evs in events.items():
        for e in evs:
            if e.get("event") != "decision":
                continue
            pair = (node, e["action"]["neighbor"])
            kind = e.get("kind")
            if kind == "state_poll":
                last_send_ival[pair] = float(e["action"]["interval_s"])
            elif kind == "state_poll_reschedule":
                res_total += 1
                prev = last_send_ival.get(pair)
                if prev is not None and float(e["action"]["interval_s"]) != prev:
                    changes += 1
    return changes, res_total


def sim_with_lag_counts(run, base):
    """Run `PORT.sim(run, None, base, base)` while passively counting `tick_lag` calls
    per ordered pair without altering any RNG call or return value.
    """
    lag_counts = collections.Counter()
    orig = PORT.Engine.tick_lag

    def counted_tick_lag(self, node, nb, kind):
        lag_counts[(node, nb)] += 1
        return orig(self, node, nb, kind)

    PORT.Engine.tick_lag = counted_tick_lag
    try:
        events, engine = PORT.sim(run, None, base, base)
    finally:
        PORT.Engine.tick_lag = orig
    return events, engine, lag_counts


def one_run(name, seed, rep, make_decider, extra_stats):
    """Execute one `(seed, stream)` simulation and return the csv row dict."""
    run = LBL.run_dict(MAIN_P, seed)
    where = "%s seed %d stream %d" % (name, seed, rep)
    check_choices(run, where)
    inner = make_decider()
    fresh_checked = keeps_memory(inner)
    wrapper = FirstCallEmpty(inner, where, fresh_checked)
    run = dict(run, decider=wrapper)
    base = R.rep_seed(LAG_PREFIX, MAIN_P, seed, rep)
    events, engine, lag_counts = sim_with_lag_counts(run, base)
    if not wrapper.checked:
        raise Fail("%s: wrapper was never called" % where)

    t = SCORE.terms(events, run["t0"])
    n4, c4, lat, censored, polls = t
    ordp = engine.decider_ord["state_poll"]
    ordr = engine.decider_ord["state_poll_reschedule"]
    mc, rt = count_mind_changes(events)
    extra_stats[name]["fresh_checked"] += 1 if fresh_checked else 0
    extra_stats[name]["fresh_unchecked"] += 0 if fresh_checked else 1
    extra_stats[name]["mind_changes"] += mc
    extra_stats[name]["res_decisions"] += rt
    for pair, cnt in lag_counts.items():
        extra_stats[name]["lag_draws"][pair] += cnt

    row = {
        "decider": name, "seed": seed, "stream": rep,
        "n4": n4, "c4": c4, "lat_ms": lat, "censored_ms": censored, "polls": polls,
        "delay_ms": SCORE.delay(t), "score_ms": SCORE.delay(t) + WEIGHT * polls,
        "score_dropped_ms": SCORE.delay_dropped(t) + WEIGHT * polls,
        "weight_ms_per_poll": WEIGHT, "ord_poll": ordp, "ord_resched": ordr, "k": 1,
        "reads": "", "sends": "", "ties": "", "n_short": "", "n_mid": "", "n_long": "",
    }
    if isinstance(inner, m24.M):
        check_counters(inner.reads, inner.sends, ordr, ordp, where)
        if sum(inner.arms.values()) != inner.reads:
            raise Fail("%s: arm sum %d != reads %d" % (where, sum(inner.arms.values()), inner.reads))
        row.update(reads=inner.reads, sends=inner.sends, ties=inner.ties,
                   n_short=inner.arms[ARMS[0]], n_mid=inner.arms[ARMS[1]],
                   n_long=inner.arms[ARMS[2]])
    return row


def csv_line(row, sha, msha):
    """Format one row as csv with `repr` floats."""
    out = dict(row, script_sha=sha, model_sha=msha)
    return ",".join(repr(out[c]) if isinstance(out[c], float) else str(out[c])
                    for c in COLUMNS)


def write_csv(path, rows_with_shas):
    """Write rows in `(decider_order, seed, stream)` order to `path`."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(COLUMNS) + "\n")
        for row, sha, msha in rows_with_shas:
            fh.write(csv_line(row, sha, msha) + "\n")


def load_csv_raw(path):
    """Return `(header_cols, {(decider, seed, stream): dict_of_str_values})`."""
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    cols = lines[0].split(",")
    by_key = {}
    for ln in lines[1:]:
        vals = ln.split(",")
        rec = dict(zip(cols, vals))
        key = (rec["decider"], int(rec["seed"]), int(rec["stream"]))
        by_key[key] = rec
    return cols, by_key


def compare_control(my_rows, ref_path, deciders, check_script_sha, where):
    """Assert `my_rows` match `ref_path` byte for byte on every column (except `script_sha`
    when `check_script_sha` is False).
    """
    ref_cols, ref_map = load_csv_raw(ref_path)
    if tuple(ref_cols) != COLUMNS:
        raise Fail("%s: column mismatch against %s" % (where, os.path.basename(ref_path)))
    checked = 0
    for row, sha, msha in my_rows:
        if row["decider"] not in deciders:
            continue
        key = (row["decider"], int(row["seed"]), int(row["stream"]))
        if key not in ref_map:
            raise Fail("%s: missing key %r in %s" % (where, key, os.path.basename(ref_path)))
        mine_str = dict(zip(COLUMNS, csv_line(row, sha, msha).split(",")))
        theirs_str = ref_map[key]
        for col in COLUMNS:
            if col == "script_sha" and not check_script_sha:
                continue
            if mine_str[col] != theirs_str[col]:
                raise Fail("%s: mismatch at %r col %s: mine %r != ref %r"
                           % (where, key, col, mine_str[col], theirs_str[col]))
        checked += 1
    if checked == 0:
        raise Fail("%s: 0 rows compared" % where)
    return checked


def main():
    """Run `--control`, `--dry`, or real evaluation, plus checks and mutations."""
    argv = sys.argv[1:]
    is_control = "--control" in argv
    is_dry = "--dry" in argv
    if is_control and is_dry:
        raise SystemExit("cannot combine --control and --dry")

    mode = "control" if is_control else ("dry" if is_dry else "real")
    seeds = list(DRY_SEEDS if is_dry else REAL_SEEDS)
    reps = list(DRY_REPS if is_dry else REAL_REPS)
    deciders = list(CONTROL_ORDER if is_control else FULL_ORDER)

    head("0  MODE, CODE DIGESTS AND SEED ROLE SEPARATION")
    say("mode            %s" % mode)
    say("deciders        %s" % deciders)
    say("seeds           %d to %d (%d distinct) x %d stream(s) = %d runs per decider"
        % (seeds[0], seeds[-1], len(seeds), len(reps), len(seeds) * len(reps)))
    entries, combined_sha = script_digests()
    say("script digests (sha256):")
    for nm, sh in entries:
        say("  %-16s %s" % (nm, sh))
    say("  %-16s %s" % ("combined", combined_sha))

    PORT.check_seeds(seeds, "eval24.py (%s)" % mode)
    if is_dry and any(s > 80 for s in seeds):
        raise Fail("dry mode touched seed > 80")
    if set(seeds) & set(range(121, 161)):
        raise Fail("measuring seeds 121-160 touched")
    say("seed guards: port.check_seeds (>100 refused) passed; overlap with 121-160: 0")

    coef_info = {}
    if not is_control:
        for cname in ("a", "b"):
            cpath = os.path.join(HERE, "coef_%s.npy" % cname)
            m_sha = file_sha256(cpath)
            w_blocks, mu, sigma = m24.load_coef(cpath)
            coef_info[cname] = (w_blocks, mu, sigma, m_sha)
            say("model %-2s        %s  sha256 %s" % (cname, os.path.basename(cpath), m_sha))
        sample_obs = m24.first_logged_resched_obs("0.2", 1, max_seed=20)
        for cname in ("a", "b"):
            w_b, mu, sg, _sh = coef_info[cname]
            vals, arm = m24.repeat_check(m24.M(w_b, mu, sg), sample_obs, "eval24 %s" % cname)
            say("  repeat check %-2s (p0.2 seed 1 first resched): %s -> %.3f s"
                % (cname, [round(v, 6) for v in vals], arm))

    head("1  SIMULATIONS AND CONTROL GATES")
    extra_stats = {d: {"fresh_checked": 0, "fresh_unchecked": 0,
                       "mind_changes": 0, "res_decisions": 0,
                       "lag_draws": collections.Counter()} for d in deciders}
    all_rows = []
    for d in deciders:
        if d.startswith("const"):
            arm = float(d.replace("const", ""))
            maker = lambda a=arm: PORT.const_decider(a)  # noqa: E731
            msha = "none"
        elif d == "F":
            maker = lambda: LBL.F()  # noqa: E731
            msha = "none"
        else:
            w_b, mu, sg, msha = coef_info[d]
            maker = lambda wb=w_b, m=mu, s=sg: m24.M(wb, m, s)  # noqa: E731

        d_rows = []
        for seed in seeds:
            for rep in reps:
                row = one_run(d, seed, rep, maker, extra_stats)
                d_rows.append((row, combined_sha, msha))
        all_rows.extend(d_rows)
        st = extra_stats[d]
        mean_sc = sum(r["score_ms"] for r, _, _ in d_rows) / len(d_rows)
        say("  %-12s runs=%4d  mean_score=%9.3f  fresh_asserted=%d  not_stateful=%d"
            % (d, len(d_rows), mean_sc, st["fresh_checked"], st["fresh_unchecked"]))

    if is_control:
        out_csv = os.path.join(HERE, "eval24_control.csv")
        write_csv(out_csv, all_rows)
        n_chk = compare_control(all_rows, os.path.join(EV23, "eval.csv"), CONTROL_ORDER,
                                check_script_sha=False, where="control vs ../23/eval.csv")
        say("CONTROL GATE PASSED: %d rows match ../23/eval.csv byte for byte (except script_sha)" % n_chk)
    elif is_dry:
        out_csv = os.path.join(HERE, "eval24_dry.csv")
        write_csv(out_csv, all_rows)
        n_chk = compare_control(all_rows, os.path.join(EV23, "eval_dry.csv"), CONTROL_ORDER,
                                check_script_sha=False, where="dry control vs ../23/eval_dry.csv")
        say("DRY CONTROL GATE PASSED: %d rows match ../23/eval_dry.csv byte for byte (except script_sha)" % n_chk)
    else:
        out_csv = os.path.join(HERE, "eval24.csv")
        write_csv(out_csv, all_rows)
        n_chk1 = compare_control(all_rows, os.path.join(EV23, "eval.csv"), CONTROL_ORDER,
                                 check_script_sha=False, where="real control vs ../23/eval.csv")
        n_chk2 = compare_control(all_rows, os.path.join(HERE, "eval24_control.csv"), CONTROL_ORDER,
                                 check_script_sha=True, where="real control vs eval24_control.csv")
        say("CONTROL GATE PASSED: %d rows match ../23/eval.csv and %d match eval24_control.csv (every column)"
            % (n_chk1, n_chk2))

    head("2  THREAT 7: MIND-CHANGE COUNTS AND PER-PAIR LAG-DRAW COUNTS")
    for d in deciders:
        st = extra_stats[d]
        rt = st["res_decisions"]
        mc = st["mind_changes"]
        share = mc / float(rt) if rt else 0.0
        if d.startswith("const") and mc != 0:
            raise Fail("constant decider %s reported %d mind changes" % (d, mc))
        top_pairs = ", ".join("%s->%s:%d" % (p[0], p[1], st["lag_draws"][p])
                              for p in sorted(st["lag_draws"]))
        say("  %-12s mind_changes=%6d / %6d (%.4f)  lag_draws_total=%d"
            % (d, mc, rt, share, sum(st["lag_draws"].values())))
        say("    per_pair: %s" % top_pairs)

    if not is_control:
        head("3  DECIDER ARM COUNTS AND TIES (MODEL DECIDERS)")
        for d in ("a", "b"):
            sub = [r for r, _, _ in all_rows if r["decider"] == d]
            reads = sum(r["reads"] for r in sub)
            ties = sum(r["ties"] for r in sub)
            n_s = sum(r["n_short"] for r in sub)
            n_m = sum(r["n_mid"] for r in sub)
            n_l = sum(r["n_long"] for r in sub)
            say("  %-4s reads=%6d  ties=%5d (%.4f)  short=%6d (%.4f)  mid=%6d (%.4f)  long=%6d (%.4f)"
                % (d, reads, ties, ties / float(reads),
                   n_s, n_s / float(reads), n_m, n_m / float(reads), n_l, n_l / float(reads)))

    head("4  EVERY CHECK SHOWN FAILING ON A MUTATED INPUT")
    fired = []

    def show_mut(label, fn):
        try:
            fn()
        except (Fail, m24.Fail, SystemExit) as exc:
            say("  %-34s FIRES: %s" % (label, exc))
            return 1
        say("  %-34s DID NOT FIRE" % label)
        return 0

    fired.append(show_mut("port.check_seeds > 100",
                          lambda: PORT.check_seeds([101], "mut/101")))
    fired.append(show_mut("label.run_dict > 100",
                          lambda: LBL.run_dict(MAIN_P, 121)))
    f_dirty = LBL.F({"h0->h1": 6.0})
    fired.append(show_mut("FirstCallEmpty: dirty F memory",
                          lambda: FirstCallEmpty(f_dirty, "mut/dirty", True)({"where": {"kind": "state_poll"}})))
    bad_ctrl = [(dict(all_rows[0][0], polls=all_rows[0][0]["polls"] + 1), all_rows[0][1], all_rows[0][2])]
    ref_ctrl = os.path.join(EV23, "eval_dry.csv" if is_dry else "eval.csv")
    fired.append(show_mut("control gate: 1 poll moved",
                          lambda: compare_control(bad_ctrl, ref_ctrl, CONTROL_ORDER, False, "mut/ctrl")))
    fired.append(show_mut("choices subset refused",
                          lambda: check_choices({"choices": list(ARMS)}, "mut/choices")))
    say()
    say("%d of %d mutations produced a failure" % (sum(fired), len(fired)))
    say()
    say("end of eval24.py")
    return 0 if sum(fired) == len(fired) else 1


if __name__ == "__main__":
    sys.exit(main())
