#!/usr/bin/env python3
"""The one delay and the one score that both the weight and a label are measured in.

Not a script.  It is its own file rather than a function in `port.py` because `port.py` is
imported by checks that need no scorer, and its own docstring keeps it that light.

2026-11-23, a correction of a definition and not a choice made after seeing labels.  There
were TWO delay definitions in this directory and the weight sat on the wrong one:

  * `weight.py` levelled the two outer arms on `../14/after.py`'s `early_score`, whose
    delay is `sum(lat4) / sum(n4)` over the DELIVERED records alone;
  * `label.py`'s score entered a CENSORED record at its lower bound, the time from the
    record's creation to the end of the window.

So the weight was the indifference point of a quantity no label was ever measured in, and
a label was a comparison under a weight solved for a different number.  Dropping the
censored records also pays an arm for the records it never delivered: an arm that polls
rarely delivers fewer records and then looks faster on the ones it did deliver.  The
definition below is the label's, and `weight.py` now re-measures the weight under it.

`early_score` stays the authority for WHICH records, WHICH window and WHICH polls.  It
cannot be called for the delay itself, because it returns the censored COUNT and not the
censored TIME, so its traversal is written again here -- and the rewrite is held to it:
`labelcheck.py` check 6 requires the four numbers the two have in common (delivered,
censored, delay sum, polls) to agree exactly, and `weight.py` compares them on every one
of its own simulations rather than on a sample.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import port as PORT  # noqa: E402

R = PORT.R
# evidence/14's early-record scorer, taken through evidence/18's own import of it, which
# `port.py` has already resolved.  Through that module and not imported again: a second
# import off another sys.path entry would give a second module object, and the check that
# compares this scorer against `early_score` would then compare two scorers.
A = PORT.B.A


def terms(events, t0):
    """(delivered, censored, delay sum ms, censored lower-bound sum ms, windowed polls).

    The record set, the window and the poll count are `../14/after.py:39-65`'s: records
    created at or before `t0 + CUT_MS`, delivery measured at the far node (`h4` for origin
    `h0`, `h0` otherwise) at or before `t0 + WE_MS`, `state_poll` rows counted over
    `[t0 + WINDOW_START_MS, t0 + WE_MS]`.

    The two sums are kept apart rather than added, so that one traversal serves both delay
    definitions: `delay` adds them and `delay_dropped` uses the delivered one alone, and
    that is what lets a label row carry `label_drop_censored` beside `label` at no extra
    simulation.  The row carries the censored COUNT per arm and per pass, not the censored
    time, so the split is visible in the row only as that count.
    """
    created, stored, polls = {}, {}, 0
    cut, we, ws = t0 + A.CUT_MS, t0 + A.WE_MS, t0 + R.WINDOW_START_MS
    for node, rows in events.items():
        for e in rows:
            kind = e.get("event")
            if kind == "record_created":
                created[(e["origin"], e["seq"])] = e["ts_ms"]
            elif kind == "record_stored":
                key = (e["origin"], e["seq"], node)
                if key not in stored or e["ts_ms"] < stored[key]:
                    stored[key] = e["ts_ms"]
            elif kind == "decision" and e.get("kind") == "state_poll":
                polls += ws <= e["ts_ms"] <= we
    n4 = c4 = 0
    lat = censored = 0.0
    for (origin, seq), made in created.items():
        if made > cut:
            continue
        ts = stored.get((origin, seq, "h4" if origin == "h0" else "h0"))
        if ts is None or ts > we:
            c4 += 1
            # The lower bound: the record had not arrived by the end of the window, so its
            # delay is at least this.  The alternative, dropping it, is the definition this
            # directory used for the weight until 2026-11-23 and is kept only as the
            # sensitivity number printed beside the labels.
            censored += we - made
        else:
            n4 += 1
            lat += ts - made
    return n4, c4, lat, censored, polls


def early_four(t):
    """`terms` cut down to exactly what `../14/after.py`'s `early_score` returns."""
    n4, c4, lat, _censored, polls = t
    return n4, c4, lat, polls


def delay(t):
    """The run's delay in ms, with every censored record at its lower bound.

    This is the `D` the weight is solved on and the delay term of a label's score.  It is
    defined on every run of these cells: `n4 + c4` is the number of records created at or
    before the cut, which the workload fixes at 240 and no arm can make zero (`weight.txt`
    prints the set of values it saw, over both cells, so a cell where it was not constant
    would say so rather than pass).  A workload that created NO record before the cut
    would divide by zero here; that workload does not exist in this effort and no guard is
    written for it.
    """
    n4, c4, lat, censored, _polls = t
    return (lat + censored) / (n4 + c4)


def delay_dropped(t):
    """The same delay with the censored records DROPPED, which is not the label's score.

    Kept because `labels.txt` reports how many labels that other choice moves.  It is
    infinite when an arm delivered nothing at all, which is a run no reader should see
    silently ranked: there is no delay to average.
    """
    n4, _c4, lat, _censored, _polls = t
    return lat / n4 if n4 else float("inf")


def score_run(events, t0, weight):
    """(score, score with the censored dropped, n4, c4, lat ms, censored ms, polls).

    `score = delay + weight x polls`, in ms of record delay per run, with the poll count
    priced at `weight` ms each.  The second element is the same score computed with
    `delay_dropped`, so one traversal serves both and the two can only disagree about the
    censored records.
    """
    t = terms(events, t0)
    polls = t[4]
    return (delay(t) + weight * polls, delay_dropped(t) + weight * polls, *t)
