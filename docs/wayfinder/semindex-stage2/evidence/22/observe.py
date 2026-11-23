#!/usr/bin/env python3
"""Acceptance 7: the one road from the observable state to a record, and to text.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B observe.py [label dir]

Two functions, and both learners call the first one:

    observation(state, resolved, history, now, neighbor) -> dict   the ordered record
    render(record, legend=True)                          -> str    what a model reads

and one adapter, `from_logged_rows`, that builds the same five inputs from the decision
rows of one node's testbed `events.jsonl`, so that the two functions run unchanged on the
logs of stage `s11a`.

WHAT IS FIXED HERE, AND WHY (the ticket's own open point)
  * **No absolute time.**  Every instant becomes an age in seconds relative to `now`, to
    one decimal.  An absolute stamp is an epoch millisecond: it differs between two runs
    of the same experiment, so a policy keyed on one could not transfer at all, and a
    language model reading one would spend tokens on 13 digits that carry no information.
  * **No node names.**  Origins and advertisers get POSITIONS inside one record, never
    names.  A policy keyed on `h2` is a policy about one topology.  The order, its
    tie-breaks included, is stated in the text itself, because an order a reader cannot
    see is a name by another route.
  * **No node name and no ordinal.**  `where` carries five things and they are not one
    kind: the node NAME and the `ordinal` (which counts decider calls over all five nodes)
    are simulator-only, while `neighbor`, `kind` and `ts_ms` are what a live node has at
    that instant.  Nothing name-derived and nothing ordinal-derived enters the record.
  * **Which peer is being polled IS in the record** (2026-11-23).  `state` lists every
    advertiser of every origin and does not say which of them this decision is about, so
    the function takes the polled neighbour as a fifth input and writes its POSITION as
    `polled_peer`; the name itself never reaches the record, and the position is not
    independent of it (see `peer_order`).  Before this the text said "the record does not
    say which peer this decision polls", which withheld real-node context: the source
    decides the interval per neighbour (`engine.py`'s `_choose_poll_interval` and
    `_advance_rule_interval` both take one) and logs the neighbour on the reschedule row
    it writes, and on a node with two neighbours a learner could not otherwise tell its
    own peer's told value from the other one's.
    `polled_peer` is null when that neighbour advertises none of the shown origins, and
    the text says which of the two it is.  2026-11-23, corrected: the reason given here
    used to be "a pair that has never answered a poll is in no `advertised` map", which is
    the MINORITY case.  Measured over the label files, most null records name no origin at
    all, so there is no advertiser map for the pair to be absent from; the rest are the
    case that was stated.  `observe.txt` prints both counts.
  * **Field order.**  The resolution first, because it is the one fact that is new since
    this pair's previous decision and the one every hand-written rule branches on; then
    the pair's own scalars; then the per-origin rows, which are the largest block and the
    one a reader can stop reading; then the exchanges.
  * **Exchanges newest first.**  The most recent exchange is the one a rule reacts to, and
    a fixed-length vector built from this record then has the live exchange at index 0
    whatever the history's length.  The ages make the direction unambiguous in the text.
  * **Rounding.**  Ages to one decimal (a tick is 100 ms, so a tenth of a second is the
    resolution of the thing being measured).  Intervals and waited times to three
    decimals, because the six intervals the labelling can produce are 3.695, 5.5425, 6.0,
    8.31375, 9.0 and 10.043 and two decimals would collapse none of them but three keeps
    each one recognisable as itself.
  * **Length bounded by construction.**  At most 8 origins (the source's own cap), at most
    4 advertisers shown per origin, at most 8 exchanges (the port's own cap).  The number
    of FIELDS is therefore a constant; the number of DIGITS is not, and a count in this
    workload is at most three digits.

WHAT IT CANNOT SHOW
  That the record carries what a policy needs.  It carries what the state carries, less
  the node names and less the ordinal, plus the polled peer's position.  Whether that is
  enough is the next ticket's question.
"""
import collections
import gzip
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import port as PORT  # noqa: E402

R = PORT.R

AGE_DECIMALS = 1
INTERVAL_DECIMALS = 3
MAX_ORIGINS = R.DECISION_STATE_MAX_ORIGINS   # the source's own cap, 8
MAX_PEERS = 4                                # advertisers shown per origin
MAX_EXCHANGES = R.DECIDER_HISTORY            # the port's own cap, 8

LEARN_DIR = "/home/lab_shared/semindex-learn/ev22"
OUT_TXT = os.path.join(HERE, "observe.txt")

# The one reference run: the stage whose logged rows carry the full state.
STAGE, P, SEED = "s11a", 0.2, PORT.check_seeds([1], "observe.py")[0]

# The fields of one logged `state_poll_reschedule` action that are the poll's OUTCOME, in
# the order `resolved` carries them.
RESOLVED_FIELDS = ("poll_epoch_ms", "trigger", "empty")
# The fields of one exchange, which are a reschedule action's less the neighbour plus the
# row's own stamp -- the same set the port's `history` element carries.
EXCHANGE_FIELDS = ("poll_epoch_ms", "trigger", "empty", "interval_s",
                   "applied_interval_s", "clamped")


def age(now, stamp):
    """One instant as an age in seconds before `now`, rounded.

    Negative is possible in principle and is not clamped: a clamped age would hide a
    stamp from the future, which would be a bug worth seeing rather than smoothing.
    """
    return round((now - stamp) / 1000.0, AGE_DECIMALS)


def peer_order(state):
    """{advertiser name: its position in this record}, highest advertisement first.

    The ranking has to be one ranking for the whole record, not one per origin, or the
    same position would mean a different advertiser on two lines of the same text.  The
    key is (highest `notified_max` over the origins, how many origins it advertises), both
    descending, and the tie-break is FIRST-APPEARANCE order over the state -- origin order
    first, then the advertiser-name order inside an origin -- which is `seen.index(n)`
    below and is NOT pure advertiser-name order: an origin advertised by the last
    advertiser alone, followed by one advertised by the first and the last, puts the last
    one first.  2026-11-23: the two orders coincide in every record measured here (0 of
    the 148 logged states and 0 of the 32 219 labelled ones differ, and the resulting
    ranking differs in none), because on this workload every neighbour advertises every
    origin; the general statement is the first-appearance one, and that is what the
    rendered legend says.  Either way the ordering leaks a name-derived ORDER while no
    name appears; that is stated rather than denied, and it is the only name-derived ORDER
    left in the record.  (`polled_peer` is derived from a name too.  It names no node --
    it points at one of the positions this ranking already produced -- but it is NOT
    independent of the polled pair's name: because the tie-break order and plain name
    order coincide on this workload, the marked position agrees with the pair's rank in
    name order far more often than a coin would, which `observe.txt` measures and prints.
    2026-11-23: this used to say the position "carries nothing about that name", which is
    stronger than anything measured.)
    """
    best = collections.defaultdict(lambda: (0, 0))
    seen = []
    for row in state["origins"].values():
        for name, adv in row["advertised"].items():
            if name not in best:
                seen.append(name)
            hi, count = best[name]
            best[name] = (max(hi, adv["notified_max"]), count + 1)
    ranked = sorted(seen, key=lambda n: (-best[n][0], -best[n][1], seen.index(n)))
    return {name: i + 1 for i, name in enumerate(ranked)}


def observation(state, resolved, history, now, neighbor):
    """The canonical record of one decision: a dict whose key order is the field order.

    `state` and `resolved` are the port's own (`obs["state"]`, `obs["resolved"]`), or the
    same two built from a logged row by `from_logged_rows`.  `history` is the pair's
    earlier exchanges OLDEST first, as the port hands them over; the record reverses them.
    `now` is the decision's own stamp, in the same millisecond clock as every stamp inside
    `resolved` and `history` -- the three are only ever subtracted, so which epoch the
    clock counts from never enters the record.

    `neighbor` is the NAME of the peer this decision polls, and it is the one input whose
    value never reaches the record: it is looked up in the record's own peer ranking and
    only the POSITION is written, as `polled_peer`.  2026-11-23: it was not an input at
    all and the rendered text said "the record does not say which peer this decision
    polls", which denied a learner something a real node has -- the live loop decides
    the interval per neighbour and writes that neighbour into the reschedule action it
    logs, so the polled peer is local context and not simulator metadata.  Without it
    every per-peer line of the record was unattributable: a node with two neighbours
    could not tell its own peer's `told` value from the other one's.

    `polled_peer` is None when that neighbour advertises no origin the record shows, which
    is a real state and not a missing input.  Two states produce it, and `observe.txt`
    counts both: most often the state names no origin at all (early in a run, before any
    reply has arrived), and less often a known origin's advertisers simply do not include
    this pair.  The text says which of the two it is in as many words.
    """
    peers = peer_order(state)
    origins = []
    for row in sorted(state["origins"].values(),
                      key=lambda r: (-r["missing"], -r["hi"])):
        advertised = sorted(
            ({"peer": peers[name], "hi": adv["hi"],
              "notified_max": adv["notified_max"]}
             for name, adv in row["advertised"].items()),
            key=lambda a: a["peer"])
        origins.append({"hi": row["hi"], "missing": row["missing"],
                        "advertised": advertised[:MAX_PEERS],
                        "advertisers_omitted": max(0, len(advertised) - MAX_PEERS)})
    # Truncated before `origins_shown` is taken, so the count is the length of the list
    # that follows it and not of the list before the cap.  The state is already capped at
    # the same number by the source, so this cap has never fired; it is here because the
    # record's length has to be bounded by this function and not by its caller's input.
    origins = origins[:MAX_ORIGINS]
    exchanges = []
    for element in list(reversed(history))[:MAX_EXCHANGES]:
        exchanges.append({
            "age_s": age(now, element["ts_ms"]),
            "trigger": element["trigger"],
            "empty": element["empty"],
            "interval_s": round(element["interval_s"], INTERVAL_DECIMALS),
            "waited_s": round(element["applied_interval_s"], INTERVAL_DECIMALS),
            "clamped": element["clamped"]})
    return {
        "resolved": None if resolved is None else {
            "poll_age_s": age(now, resolved["poll_epoch_ms"]),
            "trigger": resolved["trigger"],
            "empty": resolved["empty"]},
        "in_flight": state["in_flight"],
        "consecutive_poll_timeouts": state["consecutive_poll_timeouts"],
        "last_poll_status": state["last_poll_status"],
        "fetch_timeout_age_s": (None if state["last_timeout_s"] is None else
                                age(now, state["last_timeout_s"] * 1000.0)),
        "origins_shown": len(origins),
        "origins_omitted": bool(state["truncated"]),
        # How many distinct advertisers the whole state names, which can exceed the
        # MAX_PEERS shown under any one origin; each origin states its own omitted count.
        "peers_known": len(peers),
        # Before the origin rows, because every `p` number below is read against it.
        "polled_peer": peers.get(neighbor),
        "origins": origins,
        "exchanges": exchanges,
    }


# The legend states the WHOLE ordering, tie-breaks included: an order the reader cannot
# see is a name by another route, and the peer order's third key is the only name-derived
# thing left in the record.  2026-11-23: the first version of this text stopped after the
# two printed peer keys, so a reader of the text could not reconstruct the order of two
# peers that tie on both of them; the clause below is that third key.
LEGEND = (
    "o1.. and p1.. are positions in this record, not names. origins are ordered by "
    "missing then held, both highest first, ties in the state's own order; peers by "
    "their highest told value then by how many origins they cover, ties in the order the "
    "state first names them. the same p number is "
    "the same peer on every line. the peer this decision polls is named on the origins "
    "line."
)


def flag(value, yes, no):
    return yes if value else no


def render(record, legend=True):
    """The record as text, deterministic and in the record's own order.

    `legend` is on by default, because a reader that has not been told the ordering reads
    the positions as names.  A caller that renders many records in one prompt can hoist it
    and pass False; the printed summary reports both lengths, since the per-item length is
    what the next ticket budgets against.
    """
    lines = []
    got = record["resolved"]
    lines.append("state poll reschedule" if got is not None else "state poll")
    if got is not None:
        lines.append("just resolved: the poll sent %.1f s ago was %s, reply %s"
                     % (got["poll_age_s"], got["trigger"],
                        flag(got["empty"], "empty", "with content")
                        if got["empty"] is not None else "unknown"))
    lines.append("in flight %d, consecutive poll timeouts %d, previous poll %s, "
                 "last fetch timeout %s"
                 % (record["in_flight"], record["consecutive_poll_timeouts"],
                    record["last_poll_status"] or "none yet",
                    "none" if record["fetch_timeout_age_s"] is None
                    else "%.1f s ago" % record["fetch_timeout_age_s"]))
    lines.append("origins %d%s, peers %d, this decision polls %s"
                 % (record["origins_shown"],
                    " (more left out)" if record["origins_omitted"] else "",
                    record["peers_known"],
                    "p%d" % record["polled_peer"]
                    if record["polled_peer"] is not None
                    else "a peer none of these origins names"))
    for i, row in enumerate(record["origins"], 1):
        told = "; ".join("p%d held %d told %d"
                         % (a["peer"], a["hi"], a["notified_max"])
                         for a in row["advertised"])
        more = (" (+%d more)" % row["advertisers_omitted"]
                if row["advertisers_omitted"] else "")
        lines.append("  o%d held %d missing %d | %s%s"
                     % (i, row["hi"], row["missing"], told or "no peer told it", more))
    lines.append("last %d exchanges, newest first: age, outcome, reply, decided, waited"
                 % len(record["exchanges"]))
    for ex in record["exchanges"]:
        lines.append("  %.1f s ago %s %s %.3f s waited %.3f s%s"
                     % (ex["age_s"], ex["trigger"],
                        "-" if ex["empty"] is None
                        else flag(ex["empty"], "empty", "content"),
                        ex["interval_s"], ex["waited_s"],
                        " (clamped)" if ex["clamped"] else ""))
    if legend:
        lines.append(LEGEND)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The adapter: the same five inputs out of one node's logged decision rows
# ---------------------------------------------------------------------------

def from_logged_rows(rows):
    """Yield (neighbour, state, resolved, history, now) per logged reschedule row.

    `rows` is one node's `events.jsonl`, parsed, in file order -- what `replay.load_run`
    hands back per node.  The neighbour is the action's own `neighbor` field, which a real
    node writes on the row, and 2026-11-23 it is `observation`'s fifth input as well as
    something a caller can group by: the record marks which peer POSITION this decision
    polls, and the name is what that position is looked up from.

    The history is rebuilt from the SEQUENCE of that pair's own earlier reschedule rows,
    which is what the port's `history` is: the source keeps no bounded exchange history of
    its own, so there is nothing else to mirror.  Appended after the row is yielded and
    truncated to the port's cap, which is the port's own order of operations.
    """
    history = {}
    for e in rows:
        if e.get("event") != "decision" or e.get("kind") != "state_poll_reschedule":
            continue
        action = e["action"]
        nb = action["neighbor"]
        earlier = history.setdefault(nb, [])
        yield (nb, e["state"], {k: action[k] for k in RESOLVED_FIELDS},
               list(earlier), e["ts_ms"])
        earlier.append({"ts_ms": e["ts_ms"],
                        **{k: action[k] for k in EXCHANGE_FIELDS}})
        del earlier[:-MAX_EXCHANGES]


# ---------------------------------------------------------------------------
# The check and the character counts
# ---------------------------------------------------------------------------

def shape(obj, path=()):
    """{path: type name} over a nested dict AND list, list positions collapsed to '[]'.

    `statecheck.py`'s `walk` is not reused: it recurses into dicts only, because the state
    it compares has no list in it, and this record's two largest blocks are lists.  Six
    lines here rather than a list branch there, so that the state checks keep comparing
    exactly what they were written to compare, and so that this module -- which both
    learners import -- does not pull in a check's imports.
    """
    out = {path: type(obj).__name__} if path else {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(shape(value, path + (key,)))
    elif isinstance(obj, list):
        for value in obj:
            out.update(shape(value, path + ("[]",)))
    return out


def merged(records):
    """{path: set of type names} over many records."""
    out = {}
    for record in records:
        for path, name in shape(record).items():
            out.setdefault(path, set()).add(name)
    return out


def dotted(path):
    return ".".join(path)


def say(ok, text, lines):
    lines.append(f"{'PASS' if ok else 'FAIL'} {text}")
    print(lines[-1], flush=True)
    return 0 if ok else 1


def from_replay(run):
    """Records built from a replay roll-in run of the same logged run.

    A roll-in, not a constant arm: the record's `resolved` and its exchanges differ by
    outcome, and a constant arm reaches fewer of them.
    """
    # Imported here and not at the top: both learners import this module, and
    # `label.py` pulls in the generator and numpy, which a learner does not need.
    import label as L
    tag = L.stream_label_of(P, SEED)
    decider = L.Labeller(tag, None, None, None)
    engine = PORT.Engine(dict(run, policy=None, rule=None, redecide=True,
                              choices=list(L.CHOICES), choice_jitter=L.JF,
                              decider=decider), "B", None, None, None, tag, tag)
    decider.engine = engine
    engine.go()
    return [observation(c["state"], c["resolved"], c["history"], c["ts_ms"],
                        c["neighbor"])
            for c in decider.capture]


def polled_peer_facts(records, names):
    """(null, of those with no origin at all, >=2 peers and marked, marked == name rank).

    Two facts the text used to assert without measuring them.  The first splits the NULL
    `polled_peer` records by their cause, because the two are not the same gap: a record
    with `peers_known` 0 has no advertiser map for the pair to be missing from, while one
    with origins known has a pair that those origins' advertisers do not include.  The
    second counts how often the marked position is the polled pair's rank in plain
    advertiser-NAME order, because "the position carries nothing about the name" is a
    claim about independence and the two are not independent on this workload.

    `records` and `names` are parallel: one built record and its polled neighbour's name.
    """
    null = no_origin = rank_n = rank_same = 0
    for record, (state, neighbor) in zip(records, names):
        if record["polled_peer"] is None:
            null += 1
            no_origin += record["peers_known"] == 0
            continue
        order = peer_order(state)
        if len(order) >= 2:
            rank_n += 1
            rank_same += record["polled_peer"] == sorted(order).index(neighbor) + 1
    return null, no_origin, rank_n, rank_same


def label_file_stats(path, lines):
    """Characters per item, and the polled-peer facts, over one cell's label file."""
    with gzip.open(path, "rb") as fh:
        rows = [json.loads(line) for line in fh]
    with_legend, without, records, names = [], [], [], []
    for row in rows:
        record = observation(row["state"], row["resolved"], row["history"],
                             row["ts_ms"], row["neighbor"])
        with_legend.append(len(render(record)))
        without.append(len(render(record, legend=False)))
        records.append(record)
        names.append((row["state"], row["neighbor"]))
    lines.append(f"  {os.path.basename(path)}: {len(rows)} labelled decisions, "
                 f"characters per item median {statistics.median(with_legend):.0f} "
                 f"max {max(with_legend)} (without the legend: median "
                 f"{statistics.median(without):.0f} max {max(without)})")
    print(lines[-1], flush=True)
    return with_legend, without, polled_peer_facts(records, names), len(rows)


def main(argv=None):
    learn = (argv or sys.argv[1:] or [LEARN_DIR])[0]
    lines = []

    def out(text=""):
        print(text, flush=True)
        lines.append(text)

    run = R.load_run(STAGE, P, SEED)
    logged, counts, no_legend, per_node = [], [], [], {}
    logged_names = []
    for node, rows in run["events"].items():
        for nb, state, resolved, history, now in from_logged_rows(rows):
            record = observation(state, resolved, history, now, nb)
            text = render(record)
            logged.append(record)
            logged_names.append((state, nb))
            counts.append(len(text))
            no_legend.append(len(render(record, legend=False)))
            per_node[node] = per_node.get(node, 0) + 1
            if not text or "\n" not in text:
                raise AssertionError(f"{node}->{nb} at {now} rendered nothing usable")
    rows_total = sum(
        1 for evs in run["events"].values() for e in evs
        if e.get("event") == "decision" and e.get("kind") == "state_poll_reschedule")
    out(f"The observable state as one record and one text, on {STAGE} p {P} seed {SEED}")
    out("=" * 78)
    out(f"Logged run: {rows_total} state_poll_reschedule rows over "
        f"{len(run['events'])} nodes {sorted(per_node)}, "
        f"{len(logged)} rendered.")
    bad = say(len(logged) == rows_total,
              f"every logged state_poll_reschedule row of all {len(run['events'])} nodes "
              f"renders: {len(logged)} of {rows_total}", lines)
    simulated = from_replay(run)
    a, b = merged(logged), merged(simulated)
    extra = sorted(dotted(p) for p in set(b) - set(a))
    missing = sorted(dotted(p) for p in set(a) - set(b))
    bad += say(not extra and not missing,
               f"the record's key structure is the same built from the log and built from "
               f"the replay: {len(a)} paths from {len(logged)} logged decisions against "
               f"{len(b)} from {len(simulated)} simulated ones", lines)
    if extra:
        out(f"  only the replay's records have: {extra}")
    if missing:
        out(f"  only the logged records have: {missing}")
    out("  the record's paths and their value types, from the logged run:")
    for path, types in sorted(a.items()):
        out(f"    {dotted(path)}={'|'.join(sorted(types))}")
    out()
    out("Characters per item")
    out(f"  the logged run, {len(counts)} decisions: median "
        f"{statistics.median(counts):.0f}, max {max(counts)}; without the legend line "
        f"median {statistics.median(no_legend):.0f}, max {max(no_legend)}")
    files = sorted(f for f in (os.listdir(learn) if os.path.isdir(learn) else [])
                   if f.endswith(".jsonl.gz"))
    if not files:
        out(f"  over the labelled decisions: NOT MEASURED -- no label file under {learn} "
            "yet, so this script has to be run again after label.py")
    else:
        every = []
        facts = [0, 0, 0, 0]
        n_labelled = 0
        for name in files:
            with_legend, _n, cell, nrows = label_file_stats(
                os.path.join(learn, name), lines)
            every += with_legend
            facts = [a + b for a, b in zip(facts, cell)]
            n_labelled += nrows
        out(f"  every cell together, {len(every)} labelled decisions: median "
            f"{statistics.median(every):.0f}, max {max(every)}")
    out()
    out("One rendered item, in full (the median-length one of the logged run):")
    median_len = statistics.median_low(counts)
    for record in logged:
        if len(render(record)) == median_len:
            for line in render(record).splitlines():
                out("  | " + line)
            break
    out()
    lnull, lno_origin, lrank_n, lrank_same = polled_peer_facts(logged, logged_names)
    polled = len(logged) - lnull
    out(f"The polled peer's position is in the record ({polled} of {len(logged)} logged "
        "records name one). It is derived from the polled neighbour's NAME, which is the "
        "fifth input and never reaches the record. What the record still does not carry "
        "is any node name and the decider ordinal, neither of which a live loop can hand "
        "a policy in a form that transfers.")
    out()
    # 2026-11-23: the two sentences below are the ones this paragraph used to get wrong.
    # It gave the minority cause for a null as if it were the cause, and it reported the
    # null branch only as "live", which reads as a good result for the very gap the
    # polled peer was added to close.  Both are now stated with their counts and with a
    # predicate that reads as what it is.
    out(f"IN {lnull} OF THE {len(logged)} LOGGED RECORDS THE MARKER IS NULL, so for those "
        "decisions the record still cannot tell a learner which peer the decision is "
        f"about. Two different states produce that: in {lno_origin} the state names no "
        "origin at all, so there is no advertiser map for the pair to be missing from, "
        f"and in {lnull - lno_origin} a known origin's advertisers do not include the "
        "polled pair.")
    if files:
        null, no_origin, rank_n, rank_same = facts
        out(f"Over the {n_labelled} labelled decisions of the label files the same marker "
            f"is null {null} times, so THE ATTRIBUTION THIS RECORD ADDED IS ABSENT FROM "
            f"{null / n_labelled:.3f} OF THE RECORDS A LEARNER WILL SEE. "
            f"{no_origin} of those name no origin at all and "
            f"{null - no_origin} have a known origin whose advertisers do not include the "
            "pair.")
        out("The position names no node, but it is NOT independent of the neighbour's "
            f"name: over the {rank_n} labelled records that mark a position and name at "
            f"least two peers, the position is the polled pair's rank in plain "
            f"advertiser-NAME order in {rank_same} of them "
            f"({rank_same / rank_n:.3f}), against 0.5 for a coin on a two-peer record. "
            "That is the ranking's own tie-break and name order coinciding on this "
            "workload, not a name being recoverable from the record.")
        if lrank_n:
            out(f"  on the logged run: {lrank_same} of {lrank_n}.")
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_TXT}")
    print("every check passed" if not bad else f"{bad} CHECK(S) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
