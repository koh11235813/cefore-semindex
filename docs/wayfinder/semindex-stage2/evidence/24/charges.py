#!/usr/bin/env python3
"""The cost of one decision, as the DECISION PERIOD it owns on one emulator run.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B charges.py

Importable: everything that reads a run is a function and every table is behind `main`,
because the later steps of this ticket import `load_run`, `periods` and `charge_run`
rather than re-deriving them.

THE UNIT, in one paragraph, so another agent can reimplement every number below.
`../22/score.py` reports a run as `score = (lat + censored) / N + W * polls`, where N is
the scored record count (`n4 + c4`, 240 on every run read here) and W is the weight in ms
per poll.  Multiplying by N gives the run's TOTAL COST in ms, which is the unit this
module partitions: a poll costs `W * N` ms and one millisecond of record waiting costs one
ms.  Dividing a total-cost number by N puts it back in score ms, which is the unit the
earlier scratch precheck's rate tables were printed in; every table below says which unit
it uses.

THE PERIOD, and why it is not the earlier precheck's gap.  A decision is one
`state_poll_reschedule` row.  Its PERIOD runs from its own `ts_ms` to the next decision of
the same ordered pair (node, `action.neighbor`), half-open on the right.  That is
decision-to-decision.  The earlier scratch precheck charged send-to-send -- from the
resolved poll's `action.poll_epoch_ms` to the pair's next send -- which hands a decision
the time that had already passed before it was taken, about 2 s of it after a timeout,
where the resolution follows the send by the poll timeout.  Period cost accrues forward
only: every in-window `state_poll` send inside the period costs `W * N`, and every
millisecond a scored record spends waiting to cross that pair's hop inside the period
costs one ms.

WHAT NO DECISION OWNS.  A pair's periods tile the run, so the time before its first
decision and after its last belong to no decision; they are built here as explicit HEAD
and TAIL periods rather than dropped, because dropping them would hide a cost from the
identity and make the rest look exact.  `main` prints their share.

HOPS ARE READ, NOT RECONSTRUCTED.  `record_stored` carries `holder`, so a record's chain
of nodes is in the log.  The one thing the log cannot name is the hop a censored record
never crossed: nothing downstream ever stored that record.  That one hop is taken from the
pair structure the log itself shows -- the deepest node's only neighbour outside the
chain -- and cross-checked against the successor map the run's DELIVERED records spell
out.  A record whose chain cannot be walked, or whose next hop two readings disagree on,
becomes a NAMED RESIDUAL and is never forced into a period.
"""
import bisect
import collections
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
sys.path.insert(0, EV22)

import port as PORT    # noqa: E402  the >100 seed guard, and the replay's constants
import score as SCORE  # noqa: E402  `terms`: the one scorer, imported and not re-written

R = PORT.R             # ../13/replay.py, which owns WINDOW_START_MS
A = SCORE.A            # ../14/after.py, which owns CUT_MS and WE_MS

# 2026-11-26: allow pointing STORE at the remote host's filtered copy while keeping
# the default path unchanged for all local runs.
STORE = os.environ.get("STORE", "/home/lab_shared/semindex-eval")
STAGE = "s11a"
NODE_GLOB = os.path.join("debug", "node_dirs", "h*", "semindex", "events.jsonl")

# This workflow reads seeds 1 to 20 of the emulator logs and no others.  `port.check_seeds`
# refuses the measuring seeds (above 100) and lets 81-100 through, so it cannot stand in
# for this bound: 81-100 are the selection half and a number computed from one of them
# would make the later choice between the candidates circular.  Both guards run inside the
# one function here that opens a file.
MAX_LOG_SEED = 20

# The same key `../22/label.py` reads, out of the same file, so that the poll price here
# and the poll price a label was measured under cannot drift apart.
with open(os.path.join(EV22, "weight.json"), encoding="utf-8") as _fh:
    WEIGHT = json.load(_fh)["weight_ms_per_poll"]

ARMS = PORT.ARMS
# The five cells of the scalar state ADR 0004 line 5 permits a policy to read.  Named here
# beside the charges because the later steps read the cost of a period against the cell its
# decision was taken in, and a second spelling of the bucketing would be a second
# definition of what those steps compare the feature vector against.
CELLS = ("ans_empty", "ans_content", "to1", "to2", "to3")


class Fail(Exception):
    """A check that did not hold.

    Its own class and not `AssertionError`, so that the mutation section can catch a check
    firing without also catching a typo in the mutation and printing that as if the check
    had worked.
    """


# ---------------------------------------------------------------------------
# reading one run
# ---------------------------------------------------------------------------
def run_dir(p, seed):
    """The run's directory. `p` is the cell as the store spells it, the string "0.2"."""
    return os.path.join(STORE, STAGE, "p%s" % p, str(int(seed)))


def check_log_seeds(seeds, where, max_seed=MAX_LOG_SEED):
    """Every emulator seed this workflow may open is a whole number 1 to `max_seed`.

    2026-11-25: the first version tested `int(s) > MAX_LOG_SEED` only, which bounds from
    above and nothing else.  `load_run("0.2", 20.9)` then passed the guard and went on to
    read seed 20's directory, because `run_dir` truncates with `int`, and a seed of 0 or
    -1 was accepted outright.  A guard that lets a seed become a DIFFERENT seed is worse
    than no guard, so the bound is now two-sided and the value must be whole.
    """
    bad = sorted({float(s) for s in seeds
                  if float(s) != int(float(s))
                  or not 1 <= int(float(s)) <= max_seed})
    if bad:
        raise Fail("%s: an emulator seed read by this workflow is a whole number 1 to %d;"
                   " refused %s" % (where, max_seed, bad[:8]))
    return sorted({int(s) for s in seeds})


def load_run(p, seed, max_seed=MAX_LOG_SEED):
    """({node: [row, ...] in file order}, t0) for one logged run.

    Opens only the five `debug/node_dirs/h*/semindex/events.jsonl` files.  A run's
    `meta.json` and `results.json` are permitted but nothing here needs them, and
    `../13/replay.py`'s own `load_run` also opens the run's `*.experiment.json`, which this
    workflow may not read -- so the lines below are written rather than that function
    reused.
    """
    check_log_seeds([seed], "load_run", max_seed=max_seed)
    PORT.check_seeds([seed], "charges.load_run")
    events = {}
    for path in sorted(glob.glob(os.path.join(run_dir(p, seed), NODE_GLOB))):
        node = path.split(os.sep)[-3]
        with open(path, encoding="utf-8") as fh:
            events[node] = [json.loads(line) for line in fh if line.strip()]
    return events, check_one_t0(events, "p%s seed %s" % (p, seed))


def check_one_t0(events, where):
    """The run's one t0, refusing a log whose nodes disagree about when it started.

    Every instant below is an offset from this one, so two values would mean two clocks and
    every period boundary would be cut against whichever one `set.pop` happened to give.
    It takes the loaded EVENTS and not a path, so the mutation that shows it firing is one
    row changed in memory and no second copy of a run is written anywhere.
    """
    t0s = {e["t0_ms"] for rows in events.values() for e in rows
           if e.get("event") == "agent_start"}
    if len(t0s) != 1:
        raise Fail("%s: %d distinct t0_ms over the agent_start rows" % (where, len(t0s)))
    return float(t0s.pop())


def cell(row):
    """The reschedule row's scalar state, one of CELLS.

    The resolution's own outcome, whether an answered reply advertised nothing, and the
    real timeout count bucketed 1 / 2 / 3+, which is the information set ADR 0004 line 5
    permits.
    """
    action, state = row["action"], row["state"]
    if action["trigger"] == "answered":
        return "ans_empty" if action["empty"] else "ans_content"
    return "to%d" % min(int(state["consecutive_poll_timeouts"]), 3)


# ---------------------------------------------------------------------------
# the pair timelines
# ---------------------------------------------------------------------------
def timelines(events):
    """({pair: sorted send instants}, {pair: reschedule rows}, (first, last) instant).

    The reschedule rows are ordered by the send they resolved, which is the pair's own
    poll order.
    """
    sends = collections.defaultdict(list)
    resched = collections.defaultdict(list)
    lo = hi = None
    for node, rows in events.items():
        for e in rows:
            ts = e.get("ts_ms")
            if ts is not None:
                lo = ts if lo is None else min(lo, ts)
                hi = ts if hi is None else max(hi, ts)
            if e.get("event") != "decision":
                continue
            pair = (node, e["action"]["neighbor"])
            if e.get("kind") == "state_poll":
                sends[pair].append(float(e["ts_ms"]))
            elif e.get("kind") == "state_poll_reschedule":
                resched[pair].append(e)
    for v in sends.values():
        v.sort()
    for v in resched.values():
        v.sort(key=lambda e: float(e["action"]["poll_epoch_ms"]))
    return dict(sends), dict(resched), (float(lo), float(hi))


def span_of(events, t0):
    """The interval every period of every pair tiles, half-open on the right.

    From the run's first logged instant to its last, widened to cover the scoring window's
    own end and then one millisecond further.  The window's end is where a censored
    record's waiting stops and where a send may still be priced, so a span ending before
    it -- or ending exactly on it, with half-open periods -- would leave that waiting and
    that send outside every period and the identity would not close.

    2026-11-25: the widening to the window's end is not decoration.  The last logged
    instant of a run is whatever event happened to be last, and it is NOT guaranteed to be
    at or after `t0 + WE_MS`; on a run where it is earlier, a censored record's last
    segment, which runs to the window end, would have fallen outside every period.
    """
    _sends, _resched, (lo, hi) = timelines(events)
    return lo, max(hi, t0 + A.WE_MS) + 1.0


def check_chain(sends, resched, where):
    """Three things about every pair's decision chain, all of which the log satisfies.

    1. every decision resolved a poll this pair really sent (`poll_epoch_ms` is one of the
       pair's send instants), and no two decisions resolved the same send;
    2. the decisions come in the same order by the send they resolved and by their own
       `ts_ms`, STRICTLY, which is what makes the periods below run forwards and keeps
       one (pair, `ts_ms`) key naming one decision;
    3. every send before the pair's LAST decision was resolved by a decision.  An
       unresolved send can only be one the run ended on, so a dropped decision row shows up
       here: the send it resolved is suddenly unresolved with decisions after it.
    """
    for pair in sorted(resched):
        rows = resched[pair]
        ss = sends.get(pair, [])
        pos = set(ss)
        epochs = [float(e["action"]["poll_epoch_ms"]) for e in rows]
        # Instants are reported as offsets from the pair's first send and never as the
        # absolute epoch millisecond they are logged as: an absolute stamp in an output
        # says nothing a reader can use and is the one thing `../22/observe.py` takes out
        # of a decision's record.
        base = ss[0] if ss else epochs[0]
        for a in epochs:
            if a not in pos:
                raise Fail("%s: %s->%s resolved a poll sent %.3f s after its first, which"
                           " is not a send instant of that pair"
                           % (where, pair[0], pair[1], (a - base) / 1000.0))
        if len(set(epochs)) != len(epochs):
            raise Fail("%s: %s->%s has two decisions resolving one send"
                       % (where, pair[0], pair[1]))
        stamps = [float(e["ts_ms"]) for e in rows]
        # STRICTLY increasing, not merely sorted.  2026-11-25: the first version tested
        # `stamps != sorted(stamps)`, which equal stamps satisfy.  Two decisions of one
        # pair at one ts_ms would give a zero-length owned period and ONE (pair, ts_ms)
        # key downstream covering both rows, so one of the two decisions would vanish
        # without any check saying so.  This log has no such pair -- its smallest gap
        # between a pair's consecutive decisions is over a second -- and the test is
        # strict so that a later log that repeats a stamp fails here instead.
        late = [(a, b) for a, b in zip(stamps, stamps[1:]) if b <= a]
        if late:
            raise Fail("%s: %s->%s has a decision at or before its predecessor's own"
                       " ts_ms (%.3f s then %.3f s after the pair's first send), so its"
                       " periods would not run forwards"
                       % (where, pair[0], pair[1], (late[0][0] - base) / 1000.0,
                          (late[0][1] - base) / 1000.0))
        unresolved = [t for t in ss if t not in set(epochs) and t < stamps[-1]]
        if unresolved:
            raise Fail("%s: %s->%s sent %d poll(s) before its last decision that no"
                       " decision resolved, e.g. the one %.3f s after its first send"
                       % (where, pair[0], pair[1], len(unresolved),
                          (unresolved[0] - base) / 1000.0))
    return sum(len(v) for v in resched.values())


# ---------------------------------------------------------------------------
# the periods
# ---------------------------------------------------------------------------
def periods(events, t0):
    """{pair: [period, ...]} covering the whole run, head and tail included.

    A period is a dict: `kind` ("head", "owned" or "tail"), `index` (its position in the
    pair's own list), `next_index` (the following period, None for the tail), `start`,
    `end`, `length_s`, `interval_s` and `applied_interval_s` (the owning decision's chosen
    and applied interval, None on a head or tail period, which no decision owns),
    `trigger`, `cell`, and three fields `charge_run` fills: `sends` (the pair's IN-WINDOW
    sends inside the period, which are what the score prices), `sends_any` (its sends
    whatever the window, which is the structural count) and `wait_ms`.

    Why the owning decision's interval governs its period: the send inside the period is
    the one this decision scheduled, one applied interval after the poll it resolved.  So
    the length is that interval less the time the resolution itself had already taken, plus
    the lag of the next resolution; `main` prints the two shapes that makes.

    The span the periods tile is `span_of`'s, which is the log's own extent widened to the
    scoring window's end: that is what `t0` is for here.
    """
    sends, resched, _raw = timelines(events)
    lo, hi = span_of(events, t0)
    out = {}
    for pair in sorted(set(resched) | set(sends)):
        rows = resched.get(pair, [])
        edges = [float(e["ts_ms"]) for e in rows]
        # The head carries the pair's first send and whatever waiting preceded the pair's
        # first decision.  A pair with no decision at all gets one head period over the
        # whole run: its cost exists and is owned by nobody, which is a thing to print
        # rather than a KeyError to raise.
        chain = [dict(pair=pair, kind="head", start=lo, end=edges[0] if edges else hi,
                      interval_s=None, applied_interval_s=None, trigger=None, cell=None)]
        for k, e in enumerate(rows):
            action = e["action"]
            chain.append(dict(
                pair=pair, kind="owned", start=edges[k],
                end=edges[k + 1] if k + 1 < len(edges) else hi,
                interval_s=float(action["interval_s"]),
                applied_interval_s=float(action["applied_interval_s"]),
                trigger=action["trigger"], cell=cell(e)))
        if edges:
            # The pair's last decision owns no period: there is no next decision of that
            # pair to close one, and calling it owned would charge one decision everything
            # that happened for the rest of the run.
            chain[-1] = dict(chain[-1], kind="tail", interval_s=None,
                             applied_interval_s=None, trigger=None, cell=None)
        for i, per in enumerate(chain):
            per["index"] = i
            per["next_index"] = i + 1 if i + 1 < len(chain) else None
            per["length_s"] = (per["end"] - per["start"]) / 1000.0
            per["sends"] = 0
            per["sends_any"] = 0
            per["wait_ms"] = 0.0
        out[pair] = chain
    return out


def gap_periods(events, t0):
    """The earlier scratch precheck's SEND-TO-SEND gaps, in the same period shape.

    Here so that the two conventions can be read off ONE code path and the difference
    between them is a computed number rather than a figure copied out of an older output.
    A decision's gap runs from the send it resolved (`action.poll_epoch_ms`) to the pair's
    next send, or to the window end when no send follows, which is the earlier precheck's
    `seg` allocation.  The gap therefore starts BEFORE the decision was taken and books the
    time the resolution itself had already spent, about 2 s of it after a timeout.

    A send no decision resolved owns no gap, so its gap is marked "unowned" rather than
    given to a neighbour; with the head and the tail these periods still tile the span, so
    the identity holds under this rule too and `main` asserts it under both.

    `caused_in_window` is the earlier precheck's own row filter: the gap's own closing send
    is a real send inside the scoring window.  It is carried here so that filter can be
    reproduced and its effect measured, and it is set only on this rule's periods: it
    conditions on where the chosen interval LANDED the send, which is an outcome the arm
    moves, and the shorter the arm the more often it lands a send before the window opens.
    """
    sends, resched, _raw = timelines(events)
    lo, hi = span_of(events, t0)
    ws, we = t0 + R.WINDOW_START_MS, t0 + A.WE_MS
    out = {}
    for pair in sorted(set(resched) | set(sends)):
        rows = {float(e["action"]["poll_epoch_ms"]): e for e in resched.get(pair, [])}
        ss = sends.get(pair, [])
        chain = [dict(pair=pair, kind="head", start=lo, end=ss[0] if ss else hi,
                      interval_s=None, applied_interval_s=None, trigger=None, cell=None)]
        for i, a in enumerate(ss):
            end = ss[i + 1] if i + 1 < len(ss) else max(a, min(we, hi))
            row = rows.get(a)
            action = row["action"] if row is not None else None
            chain.append(dict(
                pair=pair, kind="owned" if row is not None else "unowned",
                start=a, end=end,
                interval_s=None if action is None else float(action["interval_s"]),
                applied_interval_s=(None if action is None
                                    else float(action["applied_interval_s"])),
                trigger=None if action is None else action["trigger"],
                cell=None if row is None else cell(row),
                caused_in_window=bool(i + 1 < len(ss) and ws <= ss[i + 1] <= we)))
        chain.append(dict(pair=pair, kind="tail", start=chain[-1]["end"], end=hi,
                          interval_s=None, applied_interval_s=None, trigger=None,
                          cell=None))
        for i, per in enumerate(chain):
            per["index"] = i
            per["next_index"] = i + 1 if i + 1 < len(chain) else None
            per["length_s"] = (per["end"] - per["start"]) / 1000.0
            per["sends"] = 0
            per["sends_any"] = 0
            per["wait_ms"] = 0.0
        out[pair] = chain
    return out


RULES = {"period": periods, "gap": gap_periods}


def check_tiling(chains, span, where):
    """Every pair's periods run forwards, touch end to start, and cover the whole span.

    The forwards test is the one that catches two periods overlapping: the periods are cut
    at the decisions' own stamps, so an overlap can only appear as a period whose end is
    before its start.
    """
    for pair in sorted(chains):
        chain = chains[pair]
        if chain[0]["start"] != span[0] or chain[-1]["end"] != span[1]:
            raise Fail("%s: %s->%s starts %.3f s into the run and ends %.3f s before its"
                       " end, instead of covering the whole span"
                       % (where, pair[0], pair[1],
                          (chain[0]["start"] - span[0]) / 1000.0,
                          (span[1] - chain[-1]["end"]) / 1000.0))
        for per in chain:
            if per["end"] < per["start"]:
                raise Fail("%s: %s->%s period %d ends %.3f ms before it starts, so it"
                           " overlaps its neighbour"
                           % (where, pair[0], pair[1], per["index"],
                              per["start"] - per["end"]))
        for x, y in zip(chain, chain[1:]):
            if x["end"] != y["start"]:
                raise Fail("%s: %s->%s leaves a hole of %.3f ms between period %d and %d"
                           % (where, pair[0], pair[1], y["start"] - x["end"],
                              x["index"], y["index"]))
    return sum(len(v) for v in chains.values())


def locate(chain, instant):
    """The period of that pair holding `instant`, or None when it is outside the span."""
    i = bisect.bisect_right([per["start"] for per in chain], instant) - 1
    if i < 0 or instant >= chain[i]["end"]:
        return None
    return chain[i]


# ---------------------------------------------------------------------------
# the waiting time of the scored records, at the hop it is spent on
# ---------------------------------------------------------------------------
def record_tracks(events, t0):
    """(segments, residuals): where every scored record's waiting ms was spent.

    A segment is (pair, start, end, key) and says the record `key` was waiting to cross
    `pair`'s hop over [start, end).  The record set and the window are `score.terms`'s own:
    created at or before `t0 + CUT_MS`, delivered at the far node (`h4` for an `h0` record,
    `h0` otherwise) at or before the window end.

    A DELIVERED record contributes its hops, which telescope to the end-to-end delay
    `score.terms` adds into `lat`.  A CENSORED record contributes the hops it did complete
    inside the window plus one last segment, from its arrival at the deepest node it
    reached to the window end, at the hop it never crossed -- so the two together are the
    `we - made` lower bound `score.terms` adds into `censored`, and no part of a censored
    record's cost is left unplaced.

    `residuals` is a list of (key, reason, ms): the records whose waiting could not be
    placed.  That is the thing never to force -- a chain that branches, a deepest node with
    no single next hop, a next hop two readings disagree on, or a sum that does not
    reconcile with the scorer's own term.
    """
    we, cut = t0 + A.WE_MS, t0 + A.CUT_MS
    created, stored, holder = {}, {}, {}
    neighbors = collections.defaultdict(set)
    for node, rows in events.items():
        for e in rows:
            ev = e.get("event")
            if ev == "record_created":
                created[(e["origin"], e["seq"])] = float(e["ts_ms"])
            elif ev == "record_stored":
                key = (e["origin"], e["seq"], node)
                if key not in stored or e["ts_ms"] < stored[key]:
                    stored[key] = float(e["ts_ms"])
                    holder[key] = e["holder"]
            elif ev == "decision":
                neighbors[node].add(e["action"]["neighbor"])

    reached = collections.defaultdict(dict)       # (origin, seq) -> {holder: node}
    for (origin, seq, node), up in holder.items():
        reached[(origin, seq)].setdefault(up, []).append(node)

    def chain_of(key):
        """The nodes this record reached, origin first, read off the `holder` field."""
        forward = reached[key]
        path, seen = [key[0]], {key[0]}
        while path[-1] in forward:
            nxt = sorted(forward[path[-1]])
            if len(nxt) != 1 or nxt[0] in seen:
                return path, "the holder chain branches at %s" % path[-1]
            path.append(nxt[0])
            seen.add(nxt[0])
        return path, None

    def far_of(origin):
        """The node a record's delay is measured at, by `score.terms`'s own rule."""
        return "h4" if origin == "h0" else "h0"

    scored = [k for k in sorted(created) if created[k] <= cut]
    # The successor map the DELIVERED records spell out: toward[(far, node)] is the node
    # that fetched from `node` on a record bound for `far`.  It is what names the hop a
    # censored record never crossed, and it is read off the log and not off a line
    # topology, which this module never writes down.
    toward = {}
    for key in scored:
        origin, _seq = key
        far = far_of(origin)
        ts = stored.get((origin, _seq, far))
        if ts is None or ts > we:
            continue
        path, why = chain_of(key)
        if why is not None:
            continue
        for upstream, downstream in zip(path, path[1:]):
            toward[(far, upstream)] = downstream

    segments, residuals = [], []
    for key in scored:
        origin, seq = key
        made = created[key]
        far = far_of(origin)
        ts = stored.get((origin, seq, far))
        delivered = ts is not None and ts <= we
        owed = (ts - made) if delivered else (we - made)
        path, why = chain_of(key)
        if why is not None:
            residuals.append((key, why, owed))
            continue
        mine, placed = [], 0.0
        for upstream, downstream in zip(path, path[1:]):
            arrival = made if upstream == origin else stored[(origin, seq, upstream)]
            done = stored[(origin, seq, downstream)]
            start, end = min(arrival, we), min(done, we)
            if end > start:
                mine.append(((downstream, upstream), start, end, key))
                placed += end - start
        deep = path[-1]
        if deep != far:
            arrival = made if deep == origin else stored[(origin, seq, deep)]
            outside = sorted(neighbors[deep] - set(path))
            named = toward.get((far, deep))
            if named is not None and named in set(path):
                residuals.append((key, "the delivered records fetch from %s to %s, which"
                                  " this record had already passed" % (deep, named),
                                  owed))
                continue
            if named is None and len(outside) != 1:
                residuals.append((key, "%s has %d neighbour(s) outside the chain and no"
                                  " delivered record names its next hop"
                                  % (deep, len(outside)), owed))
                continue
            if named is not None and outside and named not in outside:
                residuals.append((key, "the delivered records fetch from %s to %s, which"
                                  " is not a neighbour this log shows" % (deep, named),
                                  owed))
                continue
            nxt = named if named is not None else outside[0]
            if we > min(arrival, we):
                mine.append(((nxt, deep), min(arrival, we), we, key))
                placed += we - min(arrival, we)
        if abs(placed - owed) > 1e-6:
            residuals.append((key, "the segments hold %.6f ms of the %.6f ms the scorer"
                              " charges" % (placed, owed), owed))
            continue
        segments.extend(mine)
    return segments, residuals


# ---------------------------------------------------------------------------
# the charge, and the identity
# ---------------------------------------------------------------------------
def charge_run(events, t0, terms=None, rule="period", tracks=None):
    """One run: its periods filled with sends and waiting ms, and every term of the cost.

    `terms` is `score.terms`'s tuple and defaults to it.  It is an argument because a
    mutated run has to be charged against the UNMUTATED scorer: recomputing both sides from
    one mutated log would move the identity's two halves together, and the check would then
    pass on a log that had lost a record.  `tracks` is `record_tracks`'s pair and is an
    argument for the same reason, and so that charging one run under both rules walks its
    records once.  `rule` names the period rule: "period" is this module's own
    decision-to-decision one, "gap" the earlier precheck's send-to-send one.
    """
    if terms is None:
        terms = SCORE.terms(events, t0)
    n4, c4, lat, censored, polls = terms
    n = n4 + c4
    ws, we = t0 + R.WINDOW_START_MS, t0 + A.WE_MS
    sends, resched, _raw = timelines(events)
    chains = RULES[rule](events, t0)
    check_tiling(chains, span_of(events, t0), "charge_run")
    check_chain(sends, resched, "charge_run")

    # A decision pays for the poll it CAUSED.  Under the period rule that poll lies inside
    # the period: the decision schedules it and the next decision of the pair resolves it,
    # which is where the period ends.  Under the gap rule the gap ENDS at that poll, so the
    # send instant itself belongs to the next gap and the charge is shifted one period
    # back; that is the earlier precheck's own accounting, and the shift is what makes the
    # two rules' tables comparable row for row instead of off by one decision.
    # 2026-11-25: written after charging the gap rule the send at its own left edge
    # instead.  The run totals were the same, because each send is still charged once, and
    # the per-cell rates were wrong by one decision's worth of poll -- which looked like a
    # small level difference and not like a bug.
    shift = rule == "gap"
    off_span = 0
    for pair in sorted(sends):
        chain = chains[pair]
        for instant in sends[pair]:
            per = locate(chain, instant)
            if per is None:
                off_span += ws <= instant <= we
                continue
            per["sends_any"] += 1
            if shift:
                per = chain[max(per["index"] - 1, 0)]
            per["sends"] += ws <= instant <= we
    segments, residuals = record_tracks(events, t0) if tracks is None else tracks
    unplaced = 0.0
    for pair, start, end, _key in segments:
        chain = chains.get(pair)
        if chain is None:
            unplaced += end - start
            continue
        i = max(bisect.bisect_right([per["start"] for per in chain], start) - 1, 0)
        got = 0.0
        while i < len(chain) and chain[i]["start"] < end:
            overlap = min(end, chain[i]["end"]) - max(start, chain[i]["start"])
            if overlap > 0:
                chain[i]["wait_ms"] += overlap
                got += overlap
            i += 1
        unplaced += (end - start) - got

    flat = [per for pair in sorted(chains) for per in chains[pair]]
    for per in flat:
        per["cost_ms"] = per["wait_ms"] + WEIGHT * n * per["sends"]
    owned = [per for per in flat if per["kind"] == "owned"]
    edge = [per for per in flat if per["kind"] != "owned"]
    return dict(
        rule=rule, t0=t0, n=n, n4=n4, c4=c4, lat=lat, censored=censored, polls=polls,
        score=SCORE.delay(terms) + WEIGHT * polls,
        total_cost=lat + censored + WEIGHT * n * polls,
        wait_owned=sum(per["wait_ms"] for per in owned),
        wait_edge=sum(per["wait_ms"] for per in edge),
        sends_owned=sum(per["sends"] for per in owned),
        sends_edge=sum(per["sends"] for per in edge),
        cost_owned=sum(per["cost_ms"] for per in owned),
        cost_edge=sum(per["cost_ms"] for per in edge),
        n_periods=len(flat), n_owned=len(owned),
        residual_ms=sum(ms for _k, _w, ms in residuals), residuals=residuals,
        unplaced_ms=unplaced, sends_off_span=off_span, periods=chains, flat=flat)


def check_identity(run, where, tol=1e-6):
    """The scorer's own cost equals the sum over every period, to `tol` ms.

    Left: `(lat + censored) + W * N * polls`, which is `score.py`'s score times N.  Right:
    over the owned, head and tail periods alike, the waiting ms plus `W * N` per in-window
    send.  The content of the equality is not the arithmetic -- `lat` IS a sum of hop
    waits -- but that every waiting millisecond landed in a period of the RIGHT pair, that
    every in-window send has a period, and that a censored record's lower bound is placed
    at a hop instead of being left over.
    """
    got = run["wait_owned"] + run["wait_edge"] + WEIGHT * run["n"] * (
        run["sends_owned"] + run["sends_edge"])
    if abs(got - run["total_cost"]) > tol:
        raise Fail("%s: the periods hold %.6f ms, the scorer charges %.6f ms (difference"
                   " %.6f ms)" % (where, got, run["total_cost"], got - run["total_cost"]))
    return got


def check_placed(run, where):
    """No scored record's waiting was left unplaced, and no record became a residual.

    Printed as a number in `main` as well, even when it is zero: a reader cannot tell a
    zero that was computed from a zero that was assumed.
    """
    if run["unplaced_ms"] > 1e-6 or run["residuals"]:
        names = "; ".join("%s seq %s: %s" % (k[0], k[1], why)
                          for k, why, _ms in run["residuals"][:3])
        raise Fail("%s: %.6f ms of waiting fell outside every period and %d record(s)"
                   " could not be placed%s"
                   % (where, run["unplaced_ms"], len(run["residuals"]),
                      " -- " + names if names else ""))
    if run["sends_off_span"]:
        raise Fail("%s: %d in-window send(s) fell outside every period"
                   % (where, run["sends_off_span"]))
    return run["unplaced_ms"]


# ---------------------------------------------------------------------------
# the output
# ---------------------------------------------------------------------------
def say(line=""):
    """One line of the archived output."""
    print(line, flush=True)


def head(title):
    """One part's heading, so the archived output can be read part by part."""
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


def quant(values, qs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Order statistics by position, with no interpolation between two samples."""
    ordered = sorted(values)
    return [ordered[min(int(q * (len(ordered) - 1) + 0.5), len(ordered) - 1)]
            for q in qs]


def owned_of(flat, arm=None, trigger=None, since=None):
    """The owned periods of one run, optionally by interval, trigger and start instant."""
    out = []
    for per in flat:
        if per["kind"] != "owned":
            continue
        if arm is not None and per["interval_s"] != arm:
            continue
        if trigger is not None and per["trigger"] != trigger:
            continue
        if since is not None and per["start"] < since:
            continue
        out.append(per)
    return out


def main():
    """Every table, and every check shown failing on a mutated run."""
    cells = ("0.2", "0.4")
    seeds = list(range(1, MAX_LOG_SEED + 1))
    head("0  WHAT IS READ")
    say("stage            %s, cells p %s" % (STAGE, " and p ".join(cells)))
    say("seeds            %d to %d (%d runs in all)"
        % (seeds[0], seeds[-1], len(seeds) * len(cells)))
    say("seed guards      this module refuses an emulator seed that is not a whole")
    say("                 number 1 to %d -- above it, below it, or fractional -- and"
        % MAX_LOG_SEED)
    say("                 port.check_seeds refuses one above %d; both run inside"
        % PORT.MAX_SEED)
    say("                 load_run, the only function here that opens a file")
    say("files per run    debug/node_dirs/h*/semindex/events.jsonl, and nothing else")
    say("weight           %r ms per poll, read from ../22/weight.json" % WEIGHT)
    say("window           [t0 + %.0f ms, t0 + %.0f ms]; records created by t0 + %.0f ms"
        % (R.WINDOW_START_MS, A.WE_MS, A.CUT_MS))
    say("unit             TOTAL COST in ms = score x N: a poll costs W x N, a waiting")
    say("                 millisecond costs 1 ms.  Score ms = total cost / N.")

    runs, flats, gaps = {}, {}, {}
    for p in cells:
        for seed in seeds:
            events, t0 = load_run(p, seed)
            terms = SCORE.terms(events, t0)
            tracks = record_tracks(events, t0)
            for rule, into in (("period", runs), ("gap", gaps)):
                run = charge_run(events, t0, terms=terms, rule=rule, tracks=tracks)
                check_identity(run, "p%s seed %d, rule %s" % (p, seed, rule))
                check_placed(run, "p%s seed %d, rule %s" % (p, seed, rule))
                into[(p, seed)] = run
            flats[(p, seed)] = runs[(p, seed)]["flat"]
    say("runs read        %d, every one charged under both period rules, with the"
        % len(runs))
    say("                 identity and the placement asserted on each one before any")
    say("                 table below")
    say("N per run        %s" % sorted({run["n"] for run in runs.values()}))

    head("1  THE IDENTITY, PER RUN AND SUMMED")
    say("left   = (lat + censored) + W x N x polls, which is score.terms' own cost x N")
    say("right  = over every period, owned and head and tail, waiting ms + W x N x sends")
    say("The difference is asserted to 1e-6 ms per run, so a non-zero is a bug.")
    say()
    say("The score column is `score.py`'s own number for the run, so that a reader can")
    say("hold these runs against any other reading of them without rerunning anything.")
    say()
    say("  p    seed       score      lat ms     cens ms  polls   wait owned  wait edge"
        "   snd  edge     diff")
    for p in cells:
        for seed in seeds:
            run = runs[(p, seed)]
            got = check_identity(run, "p%s seed %d" % (p, seed))
            say("  %-4s %4d %11.3f %11.1f %11.1f %6d %12.1f %10.1f %5d %5d %8.1e"
                % (p, seed, run["score"], run["lat"], run["censored"], run["polls"],
                   run["wait_owned"], run["wait_edge"], run["sends_owned"],
                   run["sends_edge"], got - run["total_cost"]))
    say()
    for p in cells:
        pick = [runs[(p, s)] for s in seeds]
        total = sum(r["total_cost"] for r in pick)
        say("p %s, %d runs" % (p, len(pick)))
        for name, got in (("lat", sum(r["lat"] for r in pick)),
                          ("censored lower bounds", sum(r["censored"] for r in pick)),
                          ("W x N x polls",
                           sum(WEIGHT * r["n"] * r["polls"] for r in pick)),
                          ("owned periods", sum(r["cost_owned"] for r in pick)),
                          ("head and tail periods", sum(r["cost_edge"] for r in pick))):
            say("  %-24s %18.1f ms  %7.3f %%" % (name, got, 100.0 * got / total))
        say("  %-24s %18.1f ms  %7.3f %%" % ("total cost", total, 100.0))
        shares = [(r["cost_edge"] / r["total_cost"], s) for s, r in zip(seeds, pick)]
        say("  largest head+tail share in one run %.3f %% (seed %d), smallest %.3f %%"
            " (seed %d)" % (100.0 * max(shares)[0], max(shares)[1],
                            100.0 * min(shares)[0], min(shares)[1]))
        say("  residual, the waiting no hop could carry: %.6f ms over %d record(s)"
            % (sum(r["residual_ms"] for r in pick),
               sum(len(r["residuals"]) for r in pick)))
        say("  censored records, every one placed at the hop it never crossed: %d"
            % sum(r["c4"] for r in pick))

    head("2  WHAT THE HEAD AND TAIL PERIODS HOLD")
    say("The head runs from the run's first logged instant to the pair's first decision,")
    say("the tail from the pair's last decision to the end of the run.  The last decision")
    say("of a pair owns no period: there is no next decision of that pair to close one.")
    say()
    for p in cells:
        pick = [runs[(p, s)] for s in seeds]
        say("p %s  head+tail waiting %.1f ms of %.1f, in-window sends %d of %d"
            % (p, sum(r["wait_edge"] for r in pick),
               sum(r["lat"] + r["censored"] for r in pick),
               sum(r["sends_edge"] for r in pick), sum(r["polls"] for r in pick)))

    head("3  PERIODS PER RUN")
    for p in cells:
        counts = [runs[(p, s)]["n_periods"] for s in seeds]
        owned = [runs[(p, s)]["n_owned"] for s in seeds]
        say("p %s  periods per run min %d median %d max %d; owned min %d median %d max %d"
            % (p, min(counts), quant(counts)[2], max(counts), min(owned),
               quant(owned)[2], max(owned)))
    pairs = sorted({pair for key in runs for pair in runs[key]["periods"]})
    say("directed pairs    %d: %s" % (len(pairs), ", ".join("%s->%s" % q for q in pairs)))
    nodec = [(p, s, pair) for p in cells for s in seeds
             for pair, chain in sorted(runs[(p, s)]["periods"].items())
             if not any(per["kind"] == "owned" for per in chain)]
    say("pairs whose whole cost is one head period, because they never decided: %d"
        % len(nodec))

    head("4  HOW MANY SENDS A PERIOD HOLDS")
    # 2026-11-25: this part exists because the first draft of the module asserted one send
    # per owned period.  The assertion fired: 15 owned periods hold none and 12 hold two.
    say("One decision schedules one send, so an owned period holds one.  Two things break")
    say("that and both are printed: a send that went out BEFORE its predecessor's")
    say("resolution was taken, which is what the send-time provisional timer does and")
    say("which leaves one period with none and the one before it with two; and a send")
    say("outside the scoring window, which the score does not price at all, so the period")
    say("holding it is charged no poll.  The first row set counts every send of the pair,")
    say("the second only the in-window sends, which are the ones the cost uses.")
    say()
    say("  counting         p     kind    sends in the period   how many periods")
    for field, label in (("sends_any", "every send"), ("sends", "in-window sends")):
        for p in cells:
            counts = collections.Counter()
            for seed in seeds:
                for per in flats[(p, seed)]:
                    counts[(per["kind"], per[field])] += 1
            for kind, got in sorted(counts):
                say("  %-16s %-5s %-7s %19d %18d"
                    % (label, p, kind, got, counts[(kind, got)]))

    head("5  THE LENGTH OF AN OWNED PERIOD, BY THE INTERVAL ITS DECISION CHOSE")
    say("A period runs from one resolution to the next of the same pair.  The send it")
    say("holds is due one applied interval after the PREVIOUS send, so the length is that")
    say("interval less the time the resolution itself had already taken, plus the lag of")
    say("the next resolution.  After an ANSWERED poll the resolution follows its send by")
    say("about a tick, so the length sits just above the interval.  After a TIMED OUT poll")
    say("the resolution comes about 2 s after the send, so the period starts 2 s late and")
    say("the length sits about 2 s BELOW the interval -- unless the next poll times out")
    say("too, which pays the same 2 s back at the far end.  Seconds.")
    say()
    say("  p    trigger     interval     n      min      q25   median      q75      max"
        "     mean")
    for p in cells:
        for trigger in ("answered", "timed_out"):
            for arm in ARMS:
                got = [per["length_s"] for seed in seeds
                       for per in owned_of(flats[(p, seed)], arm=arm, trigger=trigger)]
                if not got:
                    continue
                q = quant(got)
                say("  %-4s %-11s %8.3f %5d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
                    % (p, trigger, arm, len(got), q[0], q[1], q[2], q[3], q[4],
                       statistics.fmean(got)))
    say()
    say("Split by the NEXT decision's trigger as well, which closes that arithmetic:")
    say("  p    own trigger  next trigger  interval     n   median length   interval -"
        " median")
    for p in cells:
        by_pair = {}
        for seed in seeds:
            for pair, chain in runs[(p, seed)]["periods"].items():
                by_pair[(seed, pair)] = chain
        for own in ("answered", "timed_out"):
            for nxt in ("answered", "timed_out"):
                for arm in ARMS:
                    got = []
                    for (seed, pair), chain in sorted(by_pair.items()):
                        for per in chain:
                            if (per["kind"] != "owned" or per["trigger"] != own
                                    or per["interval_s"] != arm):
                                continue
                            if chain[per["next_index"]]["trigger"] == nxt:
                                got.append(per["length_s"])
                    if not got:
                        continue
                    med = quant(got)[2]
                    say("  %-4s %-12s %-13s %8.3f %5d %15.3f %18.3f"
                        % (p, own, nxt, arm, len(got), med, arm - med))

    head("6  THE COST OF AN OWNED PERIOD, PER PERIOD AND PER SECOND")
    say("Cost of a period = its waiting ms + W x N per in-window send, divided by N, so")
    say("the numbers are SCORE ms -- the unit the earlier scratch precheck's rate table")
    say("used.  The rate is the pooled sum of cost over the pooled sum of seconds and not")
    say("a mean of per-period rates, so one short period cannot dominate it.  Two row")
    say("sets: every owned period, and the ones whose period STARTS inside the scoring")
    say("window.  The second filter reads the decision's own instant only, so it cannot")
    say("keep one arm more often than another for having landed its send in the window,")
    say("which is what a caused-send filter does and the reason none is used here.")
    say("The cost is split into its poll half and its waiting half, because the two move")
    say("in opposite directions with the interval and the split is the whole mechanism.")

    flat_by = {"period": flats,
               "gap": {key: run["flat"] for key, run in gaps.items()}}

    def gather(p, arm, since=None, one_cell=None, seed_list=None, rule="period",
               caused=False):
        """[(poll score ms, waiting score ms, seconds)] over the owned periods asked for."""
        got = []
        for seed in (seeds if seed_list is None else seed_list):
            run = runs[(p, seed)]
            start = None if since is None else run["t0"] + R.WINDOW_START_MS
            for per in owned_of(flat_by[rule][(p, seed)], arm=arm, since=start):
                if one_cell is not None and per["cell"] != one_cell:
                    continue
                if caused and not per.get("caused_in_window"):
                    continue
                got.append((WEIGHT * per["sends"], per["wait_ms"] / run["n"],
                            per["length_s"]))
        return got

    def rate_of(got):
        """(rate per second, poll per second, waiting per second, mean cost, mean s)."""
        secs = sum(x for _q, _w, x in got)
        poll = sum(q for q, _w, _x in got)
        wait = sum(w for _q, w, _x in got)
        return ((poll + wait) / secs, poll / secs, wait / secs,
                (poll + wait) / len(got), secs / len(got))

    pooled = {}
    for subset in (None, "in_window"):
        say()
        say("  %s" % ("every owned period" if subset is None
                      else "owned periods starting inside the window"))
        say("    p    interval      n  mean cost ms  mean length s  per second"
            "  poll  waiting")
        for p in cells:
            rate = {}
            for arm in ARMS:
                got = gather(p, arm, since=subset)
                if not got:
                    continue
                per_s, poll_s, wait_s, mean_cost, mean_len = rate_of(got)
                rate[arm] = per_s
                pooled[(p, subset, arm)] = (per_s, poll_s, wait_s)
                say("    %-4s %8.3f %6d %13.3f %14.3f %11.3f %5.1f %8.1f"
                    % (p, arm, len(got), mean_cost, mean_len, per_s, poll_s, wait_s))
            dear = max(rate, key=lambda a: rate[a])
            cheap = min(rate, key=lambda a: rate[a])
            pooled[(p, subset, "spread")] = 100.0 * (rate[dear] / rate[cheap] - 1.0)
            pooled[(p, subset, "dearest")] = dear
            say("    p %s: dearest %.3f s at %.3f ms/s, cheapest %.3f s at %.3f ms/s,"
                " spread %.1f %%"
                % (p, dear, rate[dear], cheap, rate[cheap],
                   pooled[(p, subset, "spread")]))
    say()
    say("Per seed, so the pooled rate is not one run talking: of the %d seeds, how many"
        % len(seeds))
    say("have each arm as their dearest per second, over every owned period")
    for p in cells:
        won = collections.Counter()
        for seed in seeds:
            rate = {}
            for arm in ARMS:
                got = gather(p, arm, seed_list=[seed])
                if got:
                    rate[arm] = rate_of(got)[0]
            won[max(rate, key=lambda a: rate[a])] += 1
        say("  p %s  %s" % (p, "   ".join("%.3f s in %d seed(s)" % (a, won[a])
                                          for a in ARMS)))
    head("7  THE TWO UNITS SIDE BY SIDE, AND WHAT THE EARLIER PRECHECK READ")
    say("The earlier scratch precheck charged SEND-TO-SEND: a decision's gap ran from the")
    say("send it resolved to the pair's next send.  That gap starts before the decision")
    say("was taken and books the time the resolution itself had already spent, about 2 s")
    say("of it after a timeout.  This module charges DECISION-TO-DECISION.  Both rules are")
    say("computed here, off one code path and one set of record tracks, so the difference")
    say("below is a measured number and not a figure copied out of an older output.")
    say("Both rules tile the run and both satisfy the identity on all %d runs, asserted"
        % len(runs))
    say("above; a decision's cost differs between them only in which period owns which")
    say("millisecond.  Score ms per second of period, by cell.  Three row sets:")
    say("  gap/caused   the earlier precheck's own rows: its gap, and its own filter --")
    say("               the gap's closing send is a real send inside the window, with no")
    say("               condition on where the gap STARTS")
    say("  gap/start    its gap, filtered instead on the decision's resolved send being")
    say("               inside the window, which is a condition no arm can move")
    say("  period       this module's rule, filtered on the decision's own instant")
    say()
    say("  p    cell          rows        %s  dear  cheap     n"
        % " ".join("%9.3f" % arm for arm in ARMS))
    verdicts = {}
    rows = (("gap", True, None, "gap/caused"), ("gap", False, "in_window", "gap/start"),
            ("period", False, "in_window", "period"))
    for p in cells:
        for one in CELLS:
            for rule, caused, since, label in rows:
                rate, counted = {}, 0
                for arm in ARMS:
                    got = gather(p, arm, since=since, one_cell=one, rule=rule,
                                 caused=caused)
                    if got:
                        rate[arm] = rate_of(got)[0]
                        counted += len(got)
                if len(rate) < len(ARMS):
                    say("  %-4s %-13s %-12s one arm has no period here, not reported"
                        % (p, one, label))
                    continue
                dear = max(rate, key=lambda a: rate[a])
                cheap = min(rate, key=lambda a: rate[a])
                verdicts[(p, one, label)] = (dear, cheap)
                say("  %-4s %-13s %-12s %9.3f %9.3f %9.3f %5.3f %6.3f %5d"
                    % (p, one, label, rate[ARMS[0]], rate[ARMS[1]], rate[ARMS[2]],
                       dear, cheap, counted))
    say()
    for left, right in (("gap/caused", "gap/start"), ("gap/start", "period")):
        differ = [key for key in sorted(verdicts) if key[2] == left
                  and verdicts[key] != verdicts[(key[0], key[1], right)]]
        say("%s against %s, over the %d (p, cell) combinations:"
            % (left, right, len(CELLS) * len(cells)))
        say("  the dearest and the cheapest arm both agree in %d, and differ in %d%s"
            % (len(CELLS) * len(cells) - len(differ), len(differ),
               ":" if differ else ""))
        for p, one, _label in differ:
            say("  p %s %-12s %-10s dear %.3f cheap %.3f | %-6s dear %.3f cheap %.3f"
                % (p, one, left, *verdicts[(p, one, left)], right,
                   *verdicts[(p, one, right)]))
    say()
    say("So the earlier precheck's own rows are the first set, and the two moves away")
    say("from them are separable.")
    say("- Its FILTER moves the long arm's rate the most, because that filter keeps a")
    say("  pre-window decision -- one poll charge over a gap holding almost no record")
    say("  time -- far more often for the long arm, which is the defect the earlier")
    say("  verification named.  The move is visible in the ans_empty rows, where the early")
    say("  decisions of a run sit.")
    say("- The UNIT raises every timeout cell's level, because a period then holds the")
    say("  next poll's own 2 s of waiting instead of handing it to the next decision.")
    say("- At p %s nothing is censored (part 1), so the gap/caused rows there are the"
        % cells[0])
    say("  earlier precheck's quantity exactly.  At p %s the earlier precheck could not"
        % cells[1])
    say("  place %.3f %% of the cost (its censored residual, part 1), so its rates there"
        % (100.0 * sum(runs[(cells[1], s)]["censored"] for s in seeds)
           / sum(runs[(cells[1], s)]["total_cost"] for s in seeds)))
    say("  are lower than these by construction and the two are not the same quantity.")
    say()
    say("So, reading by reading:")
    say("- The residual.  The earlier precheck left a censored record's lower bound")
    say("  unplaced, which was its whole residual.  Here that record waits at the hop it")
    say("  never crossed and part 1's residual is 0 ms in both cells under both rules.")
    say("  What is non-zero instead is the head and tail share of part 1, which the")
    say("  earlier gap partition did not have: its gaps began at a send, so the time")
    say("  before a pair's first decision sat inside its first gap.")
    say("- The decision-counted reading.  The earlier precheck found 3.695 s cheapest per")
    say("  DECISION in all five cells.  Part 6's mean-cost-per-period column reproduces")
    say("  that ordering, and it is the same artefact: a period pays one poll whatever")
    say("  interval it chose, so counting by decisions prices 3.695 s over 3.7 s the same")
    say("  as 10.043 s over 10 s.")
    spreads = [v for k, v in pooled.items() if k[2] == "spread"]
    say("- The time-fair reading per cell.  The earlier precheck's headline was that what")
    say("  separates is always that 3.695 s is the DEAREST per second.  Of the %d"
        % (len(CELLS) * len(cells)))
    say("  (p, cell) combinations, the number in which 3.695 s is the dearest:")
    for _r, _c, _s, label in rows:
        say("    %-12s %d" % (label, sum(1 for key in verdicts if key[2] == label
                                         and verdicts[key][0] == ARMS[0])))
    kept = [key for key in sorted(verdicts) if key[2] == "gap/start"
            and verdicts[key] == verdicts[(key[0], key[1], "period")]]
    say("  So the reading is not a property of the data alone: %d of the %d combinations"
        % (len(kept), len(CELLS) * len(cells)))
    say("  keep their dearest and cheapest arm across the two units and %d do not, and"
        % (len(CELLS) * len(cells) - len(kept)))
    say("  the list above says which.")
    say("- The time-fair reading does NOT survive pooling the cells.  Pooled (part 6) the")
    say("  three arms sit within %.1f to %.1f %% of each other; the dearest per second is"
        % (min(spreads), max(spreads)))
    say("  %s; and the per-seed count splits"
        % ", ".join("%.3f s at p %s" % (pooled[(p, None, "dearest")], p)
                    for p in cells))
    say("  the dearest arm across all three arms.  The pooled rate is a quantity the")
    say("  earlier precheck never computed, and")
    say("  it is a worse statistic than the per-cell one for exactly the reason the cells")
    say("  exist: an arm spends a different share of its seconds in the expensive timeout")
    say("  cells, so the pooled rate mixes the cost of a cell with the cost of an arm.")
    say("- What both units agree on is the SPLIT in part 6.  At p %s, over the in-window"
        % cells[0])
    say("  periods, the poll cost per second falls with the interval (%s"
        % " / ".join("%.1f" % pooled[(cells[0], "in_window", arm)][1] for arm in ARMS))
    say("  score ms/s) while the waiting cost per second rises with it (%s),"
        % " / ".join("%.1f" % pooled[(cells[0], "in_window", arm)][2] for arm in ARMS))
    say("  and the two nearly cancel.  That near-cancellation is the finding to carry")
    say("  forward, not either arm's name.")

    head("8  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT")
    say("The mutation is of the INPUT, inside this process.  Where a mutation moves the")
    say("scorer's own terms as well as the periods', the UNMUTATED terms are handed to")
    say("charge_run: recomputing both halves of the identity from one mutated log would")
    say("move them together and the check would pass on a log that had lost a record.")
    say()
    fired = []

    def shows(name, call):
        """Run a check on a mutated input and require it to refuse."""
        try:
            call()
        except (Fail, SystemExit) as exc:
            say("  %-26s FIRES: %s" % (name, exc))
            return True
        say("  %-26s DID NOT FIRE -- this check cannot fail on this input" % name)
        return False

    base_events, base_t0 = load_run("0.2", 1)
    base_terms = SCORE.terms(base_events, base_t0)
    base_we = base_t0 + A.WE_MS

    def copy_events(events):
        """A copy whose row lists can be rebuilt; the rows themselves are shared."""
        return {node: list(rows) for node, rows in events.items()}

    def a_delivered_record():
        """(origin, seq) of one scored record that was delivered, and its hop nodes."""
        created, stored = {}, {}
        for node, rows in base_events.items():
            for e in rows:
                if e.get("event") == "record_created":
                    created[(e["origin"], e["seq"])] = float(e["ts_ms"])
                elif e.get("event") == "record_stored":
                    stored.setdefault((e["origin"], e["seq"], node), float(e["ts_ms"]))
        for key in sorted(created):
            if created[key] > base_t0 + A.CUT_MS:
                continue
            far = "h4" if key[0] == "h0" else "h0"
            ts = stored.get((key[0], key[1], far))
            if ts is not None and ts <= base_we:
                return key
        raise Fail("the mutation found no delivered scored record")

    def drop_a_record_stored():
        """One INTERMEDIATE hop of one delivered record, unlogged.

        An intermediate node's row and not the far node's: dropping the far node's row
        would censor the record, which the scorer and the periods would both follow, and
        the identity would then hold on a log that had lost a hop.
        """
        origin, seq = a_delivered_record()
        events = copy_events(base_events)
        for node in sorted(events):
            if node in (origin, "h4" if origin == "h0" else "h0"):
                continue
            rows = events[node]
            for i, e in enumerate(rows):
                if (e.get("event") == "record_stored" and e["origin"] == origin
                        and e["seq"] == seq):
                    events[node] = rows[:i] + rows[i + 1:]
                    run = charge_run(events, base_t0, terms=base_terms)
                    check_placed(run, "one record_stored removed")
                    check_identity(run, "one record_stored removed")
                    return
        raise Fail("the mutation found no intermediate record_stored row to remove")

    def send_out_of_window():
        """One RESOLVED send moved to before the window opens."""
        events = copy_events(base_events)
        _s, resched, _span = timelines(events)
        epochs = {float(e["action"]["poll_epoch_ms"])
                  for rows in resched.values() for e in rows}
        for node in sorted(events):
            rows = events[node]
            for i, e in enumerate(rows):
                if (e.get("event") == "decision" and e.get("kind") == "state_poll"
                        and float(e["ts_ms"]) in epochs):
                    events[node] = rows[:i] + [dict(e, ts_ms=base_t0 - 1000.0)] + \
                        rows[i + 1:]
                    sends, after, _span2 = timelines(events)
                    check_chain(sends, after, "one send moved out of the window")
                    return
        raise Fail("the mutation found no resolved state_poll row to move")

    def drop_a_decision():
        """One reschedule row dropped, and not the pair's last.

        Not the last, because a pair's final send is allowed to be unresolved -- the run
        ends on it -- so dropping the last row would leave a log this check accepts.
        """
        events = copy_events(base_events)
        _s, resched, _span = timelines(events)
        pair, rows = sorted(resched.items())[0]
        victim = rows[len(rows) // 2]
        events[pair[0]] = [e for e in events[pair[0]] if e is not victim]
        sends, after, _span2 = timelines(events)
        check_chain(sends, after, "one decision row dropped")

    def overlap_two_periods():
        """One decision's ts_ms moved past the next decision of the same pair.

        `check_chain`'s ordering test catches this mutation as well; the call below is to
        `check_tiling`, because it is the check that says the periods do not overlap.
        """
        events = copy_events(base_events)
        _s, resched, _span = timelines(events)
        pair, rows = sorted(resched.items())[0]
        i = len(rows) // 2
        moved = dict(rows[i], ts_ms=float(rows[i + 1]["ts_ms"]) + 1.0)
        events[pair[0]] = [moved if e is rows[i] else e for e in events[pair[0]]]
        check_tiling(periods(events, base_t0), span_of(events, base_t0),
                     "two periods overlapping")

    def two_t0s():
        """One extra agent_start row, one millisecond off the run's own t0.

        2026-11-25: this arm, and the five below it, were reachable and in no mutation row
        of the first version of this part, while the heading claimed every check.  Each is
        mutated in memory -- a hand-built chain dict, a copy of one run's row list, a run
        dict with one field moved -- so no second copy of a log is written and no machine
        path can reach this output.
        """
        events = copy_events(base_events)
        node = sorted(events)[0]
        start = next(e for e in events[node] if e.get("event") == "agent_start")
        events[node] = events[node] + [dict(start, t0_ms=float(start["t0_ms"]) + 1.0)]
        check_one_t0(events, "one node's t0 moved by 1 ms")

    def hand_chain(epochs, stamps):
        """({pair: sends}, {pair: rows}) for one pair, built by hand and not from a log.

        The two chain arms below cannot be reached from any log of these runs: the log's
        own decisions resolve distinct sends and their stamps rise by over a second.  A
        hand-built pair is the input that would break them.
        """
        pair = ("h0", "h1")
        rows = [{"ts_ms": ts, "action": {"poll_epoch_ms": ep, "neighbor": pair[1]}}
                for ep, ts in zip(epochs, stamps)]
        return {pair: sorted(set(epochs))}, {pair: rows}

    def one_send_two_decisions():
        """Two decisions of one pair resolving the same send."""
        sends, resched = hand_chain([1000.0, 1000.0], [1100.0, 1200.0])
        check_chain(sends, resched, "two decisions on one send")

    def equal_stamps():
        """Two decisions of one pair at the same ts_ms, which the strict test refuses."""
        sends, resched = hand_chain([1000.0, 2000.0], [1100.0, 1100.0])
        check_chain(sends, resched, "two decisions at one instant")

    def chain_not_covering():
        """The real chains against a span that starts one millisecond earlier."""
        chains = periods(base_events, base_t0)
        lo, hi = span_of(base_events, base_t0)
        check_tiling(chains, (lo - 1.0, hi), "a span the periods do not cover")

    def chain_with_a_hole():
        """One pair's second period starting 10 ms after the first one ends."""
        chains = periods(base_events, base_t0)
        pair = sorted(chains)[0]
        chain = [dict(per) for per in chains[pair]]
        chain[1]["start"] += 10.0
        holed = dict(chains)
        holed[pair] = chain
        check_tiling(holed, span_of(base_events, base_t0),
                     "a hole between two periods")

    fired.append(shows("load_run/two t0 values", two_t0s))
    fired.append(shows("chain/one send twice", one_send_two_decisions))
    fired.append(shows("chain/one instant twice", equal_stamps))
    fired.append(shows("tiling/span not covered", chain_not_covering))
    fired.append(shows("tiling/a hole", chain_with_a_hole))
    fired.append(shows("placed/a send off the span",
                       lambda: check_placed(dict(runs[("0.2", 1)], sends_off_span=1),
                                            "one in-window send outside every period")))
    fired.append(shows("placed/record_stored", drop_a_record_stored))
    fired.append(shows("chain/send out of window", send_out_of_window))
    fired.append(shows("chain/decision dropped", drop_a_decision))
    fired.append(shows("tiling/overlap", overlap_two_periods))
    fired.append(shows("identity/a period short",
                       lambda: check_identity(
                           dict(runs[("0.2", 1)],
                                wait_owned=runs[("0.2", 1)]["wait_owned"] - 1.0),
                           "one period's waiting time lost")))
    fired.append(shows("log seeds > %d" % MAX_LOG_SEED,
                       lambda: load_run("0.2", MAX_LOG_SEED + 1)))
    fired.append(shows("a log seed of 0",
                       lambda: load_run("0.2", 0)))
    fired.append(shows("a fractional log seed",
                       lambda: load_run("0.2", MAX_LOG_SEED + 0.9)))
    fired.append(shows("port.check_seeds > %d" % PORT.MAX_SEED,
                       lambda: PORT.check_seeds([PORT.MAX_SEED + 1], "mutated")))
    say()
    say("%d of %d mutations produced a failure" % (sum(fired), len(fired)))

    head("9  WHAT THIS MODULE DOES NOT SHOW")
    say("- Nothing about which interval is better.  Part 6 is the cost that FOLLOWED a")
    say("  decision over one period; the same decision also moves where a record arrives")
    say("  inside the DOWNSTREAM pair's periods, and that cost is charged to the")
    say("  downstream pair.  The partition is additive across decisions and is not")
    say("  separable across pairs.")
    say("- Nothing about the replay, where a score is taken.  Every number here is read")
    say("  off the emulator log; the per-decision agreement between the two worlds is")
    say("  recorded as failed in evidence/13 and no simulation is run here.")
    say("- Nothing about any seed above %d, and nothing about the label files."
        % MAX_LOG_SEED)
    say("- No standard error anywhere.  The per-seed count in part 6 counts seeds; it is")
    say("  not an interval, and the pooled rate carries no spread.")
    say("- The head and tail share is a cost no decision owns UNDER THIS PERIOD RULE.  It")
    say("  is not a measurement error and it does not shrink by being charged to the")
    say("  nearest decision: it would move onto that decision, which is the convention")
    say("  the earlier precheck used and the reason its gaps began at a send.")
    say("- Nothing about a discount.  Every number is undiscounted, as the score is.")
    say()
    say("no check fired on real input; the failures in part 8 are the mutations")
    say("end of charges.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
