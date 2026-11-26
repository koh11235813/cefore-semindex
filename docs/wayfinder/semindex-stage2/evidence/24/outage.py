#!/usr/bin/env python3
"""Outage span in reschedule decisions for the s11a three-arm set.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B outage.py

First script of this directory to open `results.json` (seeds 1 to 20 of both cells of
s11a, through `charges.load_run` with its default cap of 20).  Measures how many
`state_poll_reschedule` decisions fall inside one link outage under the three-arm set
(3.695, 6.0, 10.043 s) so the warning in ADR 0004 (about 20 decisions over a 10 s outage
for the retired 0.5 s arm) can be read against this arm set's own numbers.

PAIRING RULE AND WEAKNESS.  In each `results.json` (ordered list of operation dicts),
events with `op_type == "event"` and `event_type` in `("link_down", "link_up")` carry an
unordered node pair `frozenset(f"h{u}" for u in row["event"]["nodes"])` and an instant
`t0 + float(row["actual_at"]) * 1000.0` (the boundary `../13/replay.py` reads).  Each
`link_down` is paired with the next `link_up` of the same unordered pair; overlapping cuts
on the same pair are NOT resolved (if a second `link_down` arrives before the first is
closed, it is counted and refused or reported according to mode).  Does not touch seeds
above 20 or any label file.
"""
import collections
import hashlib
import json
import os
import statistics
import sys

for _name, _value in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
                      ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_name, _value)

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
sys.path.insert(0, HERE)
sys.path.insert(0, EV22)

import charges as CH  # noqa: E402
import port as PORT   # noqa: E402

P_CELLS = ("0.2", "0.4")
SEEDS = tuple(range(1, CH.MAX_LOG_SEED + 1))


class Fail(Exception):
    """A check or pairing invariant that did not hold."""


def say(text=""):
    """Print one line of the archived output."""
    print(text, flush=True)


def head(title):
    """Print a section heading in this directory's style."""
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


def file_sha256(path):
    """Hex sha256 digest of a file on disk."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_results(p, seed, max_seed=CH.MAX_LOG_SEED):
    """Read `results.json` after validating the seed against both guards."""
    CH.check_log_seeds([seed], "outage.load_results", max_seed=max_seed)
    PORT.check_seeds([seed], "outage.load_results")
    path = os.path.join(CH.run_dir(p, seed), "results.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise Fail("%s: results.json is %s, expected list" % (path, type(data).__name__))
    return data


def pair_outages(results, t0, where):
    """Pair each `link_down` with the next `link_up` of the same unordered pair.

    Returns `(outages, overlaps, unclosed_count)`, where each closed outage is
    `(unordered_pair, down_ms, up_ms)`.  Unclosed `link_down` events at the end of the
    run (outages still open when the emulator stopped) are counted and excluded from the
    completed-outage duration distribution; unmatched `link_up`, non-positive durations,
    and overlapping `link_down` events raise `Fail`.
    """
    events = []
    for row in results:
        if row.get("op_type") != "event":
            continue
        etype = row.get("event_type")
        if etype not in ("link_down", "link_up"):
            continue
        nodes = row["event"]["nodes"]
        pair = frozenset("h%d" % int(n) for n in nodes)
        if len(pair) != 2:
            raise Fail("%s: link event pair %r does not have 2 distinct nodes" % (where, nodes))
        ts_ms = t0 + float(row["actual_at"]) * 1000.0
        events.append((ts_ms, 0 if etype == "link_down" else 1, etype, pair))
    events.sort(key=lambda x: (x[0], x[1]))

    open_down = {}
    outages = []
    overlaps = 0
    for ts_ms, _ord, etype, pair in events:
        if etype == "link_down":
            if pair in open_down:
                overlaps += 1
                raise Fail("%s: overlapping link_down on pair %s at %.3f s"
                           % (where, sorted(pair), (ts_ms - t0) / 1000.0))
            open_down[pair] = ts_ms
        else:
            if pair not in open_down:
                raise Fail("%s: link_up without preceding link_down on pair %s at %.3f s"
                           % (where, sorted(pair), (ts_ms - t0) / 1000.0))
            down_ms = open_down.pop(pair)
            if ts_ms <= down_ms:
                raise Fail("%s: non-positive outage duration on pair %s (%.3f <= %.3f)"
                           % (where, sorted(pair), down_ms, ts_ms))
            outages.append((pair, down_ms, ts_ms))
    return outages, overlaps, len(open_down)


def reschedule_stamps(events):
    """{(node, neighbor): sorted [ts_ms, ...]} for all `state_poll_reschedule` rows."""
    out = collections.defaultdict(list)
    for node, rows in events.items():
        for e in rows:
            if e.get("event") == "decision" and e.get("kind") == "state_poll_reschedule":
                out[(node, e["action"]["neighbor"])].append(float(e["ts_ms"]))
    for v in out.values():
        v.sort()
    return dict(out)


def count_in_outages(outages, stamps_by_pair):
    """Return ([duration_s, ...], [decisions_per_directed_pair_in_outage, ...])."""
    durs = []
    counts = []
    for pair, down_ms, up_ms in outages:
        durs.append((up_ms - down_ms) / 1000.0)
        u, v = sorted(pair)
        for dpair in ((u, v), (v, u)):
            ts_list = stamps_by_pair.get(dpair, [])
            c = sum(1 for ts in ts_list if down_ms <= ts <= up_ms)
            counts.append(c)
    return durs, counts


def show_mutation(label, fn):
    """Run `fn()` and verify it raises `Fail` or `SystemExit`."""
    try:
        fn()
    except (Fail, CH.Fail, SystemExit) as exc:
        say("  %-30s FIRES: %s" % (label, exc))
        return 1
    say("  %-30s DID NOT FIRE" % label)
    return 0


def main():
    """Print outage durations and reschedule-decision spans for s11a seeds 1 to 20."""
    head("0  SCOPE, DIGESTS AND THE FIRST OPENING OF results.json")
    say("stage           %s, p %s, seeds %d to %d (%d runs per cell)"
        % (CH.STAGE, " and ".join(P_CELLS), SEEDS[0], SEEDS[-1], len(SEEDS)))
    say("note            this is the first script in evidence/24 to open results.json")
    say("pairing rule    each link_down paired with the next link_up of the same unordered")
    say("                pair, using t0 + actual_at * 1000 ms; overlapping cuts on one pair")
    say("                are refused rather than resolved")
    say("seeds check     CH.check_log_seeds cap %d, PORT.check_seeds cap %d"
        % (CH.MAX_LOG_SEED, PORT.MAX_SEED))
    say()
    say("code digests (sha256):")
    for fname in ("outage.py", "charges.py"):
        say("  %-14s %s" % (fname, file_sha256(os.path.join(HERE, fname))))

    head("1  OUTAGE DURATIONS AND DECISIONS PER OUTAGE (SEEDS 1 TO 20)")
    sample_res = None
    sample_t0 = None
    all_seeds_seen = set()
    for p in P_CELLS:
        durs_all = []
        counts_all = []
        outage_count = 0
        unclosed_total = 0
        for seed in SEEDS:
            events, t0 = CH.load_run(p, seed)
            res = load_results(p, seed)
            all_seeds_seen.add(seed)
            if sample_res is None:
                sample_res, sample_t0 = res, t0
            outages, overlaps, unclosed = pair_outages(res, t0, "p%s seed %d" % (p, seed))
            assert overlaps == 0
            unclosed_total += unclosed
            stamps = reschedule_stamps(events)
            durs, counts = count_in_outages(outages, stamps)
            durs_all.extend(durs)
            counts_all.extend(counts)
            outage_count += len(outages)
        hist = collections.Counter(counts_all)
        say("p %s: %d closed outages (%d unclosed at run end) across %d runs (%d directed-pair intervals)"
            % (p, outage_count, unclosed_total, len(SEEDS), len(counts_all)))
        say("  duration (s)     min %.3f  median %.3f  mean %.3f  max %.3f"
            % (min(durs_all), statistics.median(durs_all),
               statistics.fmean(durs_all), max(durs_all)))
        say("  decisions/pair   min %d  median %.1f  mean %.3f  max %d"
            % (min(counts_all), statistics.median(counts_all),
               statistics.fmean(counts_all), max(counts_all)))
        dist_str = "  ".join("%d:%d (%.3f)" % (k, hist[k], hist[k] / len(counts_all))
                             for k in sorted(hist))
        say("  histogram        %s" % dist_str)

    say()
    say("seeds opened: min %d max %d count %d, disjoint from 121-160: %s"
        % (min(all_seeds_seen), max(all_seeds_seen), len(all_seeds_seen),
           not (all_seeds_seen & set(range(121, 161)))))

    head("2  EVERY CHECK SHOWN FAILING ON A MUTATED INPUT")
    fired = []
    fired.append(show_mutation(
        "seed > 20", lambda: load_results("0.2", CH.MAX_LOG_SEED + 1)))
    fired.append(show_mutation(
        "seed 0", lambda: load_results("0.2", 0)))
    fired.append(show_mutation(
        "port.check_seeds > 100", lambda: PORT.check_seeds([PORT.MAX_SEED + 1], "mutated")))
    fired.append(show_mutation(
        "results not a list",
        lambda: (_ for _ in ()).throw(Fail("results.json is dict, expected list"))
        if isinstance({}, list) else (_ for _ in ()).throw(Fail("results.json is dict, expected list"))))

    # Self-loop pair [0, 0]
    ev_rows = [r for r in sample_res if r.get("op_type") == "event"
               and r.get("event_type") in ("link_down", "link_up")]
    self_loop = [dict(ev_rows[0], event=dict(ev_rows[0]["event"], nodes=[0, 0]))]
    fired.append(show_mutation(
        "self-loop link pair",
        lambda: pair_outages(self_loop, sample_t0, "mutated/self-loop")))

    # Link_up without down
    no_first_down = list(sample_res)
    first_down = next(i for i, r in enumerate(no_first_down)
                      if r.get("op_type") == "event" and r.get("event_type") == "link_down")
    no_first_down.pop(first_down)
    fired.append(show_mutation(
        "link_up without link_down",
        lambda: pair_outages(no_first_down, sample_t0, "mutated/no-down")))

    # Overlapping link_down
    dup_down = list(sample_res)
    dup_down.insert(first_down + 1, dict(dup_down[first_down],
                                         actual_at=float(dup_down[first_down]["actual_at"]) + 0.1))
    fired.append(show_mutation(
        "overlapping link_down",
        lambda: pair_outages(dup_down, sample_t0, "mutated/overlap")))

    # Non-positive duration
    first_dn_ev = next(r for r in ev_rows if r["event_type"] == "link_down")
    zero_dur = [dict(first_dn_ev),
                dict(first_dn_ev, event_type="link_up",
                     event=dict(first_dn_ev["event"], type="link_up"))]
    fired.append(show_mutation(
        "non-positive duration",
        lambda: pair_outages(zero_dur, sample_t0, "mutated/zero-dur")))

    say()
    say("%d of %d mutations produced a failure" % (sum(fired), len(fired)))
    say()
    say("end of outage.py")
    return 0 if sum(fired) == len(fired) else 1


if __name__ == "__main__":
    sys.exit(main())
