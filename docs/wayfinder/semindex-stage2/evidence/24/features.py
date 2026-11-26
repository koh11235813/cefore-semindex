#!/usr/bin/env python3
"""The numeric form of one decision's input: `../22/observe.py`'s record as a vector.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B features.py

Importable: the vector and its inverse are functions and every table is behind `main`,
because the later steps of this ticket import `vector`, `NAMES` and the three record
roads rather than re-deriving them.

WHAT THIS CARRIES, AND WHY EXACTLY THAT.  The one shared writing of a decision's input is
`observe.observation`, whose record `observe.render` turns into the text the text-reading
decider reads.  A numeric decider must see the SAME information, so the rule here is not
"every useful number" but "every quantity the rendered text carries, and nothing else":

  * the resolution -- its trigger, whether the reply was empty, the age of the poll it
    resolved -- or, at a send, the fact that there is no resolution;
  * the pair's scalars: polls in flight, consecutive timeouts, the previous poll's status,
    the age of the last fetch timeout;
  * the counts the text's third line carries: origins shown, whether origins were left
    out, how many peers are known, and which peer POSITION this decision polls;
  * per origin shown, in the record's own order: what this node holds, how many records
    are missing, what each advertiser announced (its position, what it holds, what it told
    us), and how many advertisers were left out;
  * per exchange with this neighbour, newest first: its age, outcome, whether the reply
    was empty, the interval decided, the interval waited, and whether it was clamped.

`main` proves the "nothing else" half by inverting the vector and rendering the result: the
text comes back character for character, so the vector holds no less, and two records that
render differently never share a vector, so it holds no more that matters.

WHAT IS DELIBERATELY NOT READ.  `vector` takes the RECORD and nothing else, so the node
name, the decider ordinal and the emulator-only `fetched` field of a `fetch_plan` row
cannot reach it: they are not in the record.  `main` shows this as three equalities -- the
vector of a logged row is unchanged when every node name in the row is replaced, when the
port's `where` carries another node and ordinal, and when a label row's own `node` and `k`
are overwritten.

VARIABLE-LENGTH PARTS.  Origins (at most 8), advertisers per origin (at most 4) and
exchanges (at most 8) get a fixed number of slots in the record's own order, each with a
presence flag, plus the count the text prints.  An absent slot's cells are FILL, and the
presence flag is what tells an absent slot from a present one whose value is FILL: a
record with no exchange and a record whose newest exchange is 0.0 s old differ in the flag
and not in the value.  An input with MORE than a cap's worth is refused rather than
truncated, because a silent truncation is exactly the shape of error that would make two
different texts share one vector.

CLOSED VOCABULARIES.  A trigger, a poll status and a three-valued `empty` are one-hot over
vocabularies taken from the source, and a value outside one is REFUSED.  Mapping an
unknown string to a default would put two different texts on one vector, which is the one
property this module exists to keep.
"""
import collections
import gzip
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
sys.path.insert(0, HERE)
sys.path.insert(0, EV22)

import charges as CH    # noqa: E402  the run loader and the five-cell state, one copy
import observe as OBS   # noqa: E402  `observation`, `render`, `from_logged_rows`
import port as PORT     # noqa: E402  the >100 seed guard

MAX_ORIGINS = OBS.MAX_ORIGINS        # 8, the source's own cap, through observe
MAX_PEERS = OBS.MAX_PEERS            # 4, the advertisers one origin line shows
MAX_EXCHANGES = OBS.MAX_EXCHANGES    # 8, the port's own history cap

# An absent slot's cells.  0.0 and not NaN: the later steps solve a linear system in these
# vectors and one NaN would spread to every coefficient, while 0.0 beside a presence flag
# of 0 is distinguishable from a present 0.0 by the flag.
FILL = 0.0

# The resolution's trigger and an exchange's.  `../13/replay.py`'s `resolve` is called with
# "answered" or "timed_out" and the source's `_redecide_next_poll` with the same two.
TRIGGERS = ("answered", "timed_out")
# `last_poll_status`.  The four values the source itself enumerates (`src/semindex/
# evalagg.py`'s `_POLL_STATUS_VALUES`), null first, which is a pair that has not polled yet.
STATUSES = (None, "answered", "timed_out", "pending")
# `empty` is three-valued and the three render as three different words ("empty", "with
# content", "unknown"), so it is three columns and not one.
EMPTIES = (None, False, True)
WORDS = {None: "none", False: "false", True: "true"}

# The label file this module reads its replay-side records from, and the seed bound on it.
# The tuning half is seeds 1 to 80; `port.check_seeds` lets 81-100 through because they are
# the selection half, so it cannot stand in for this bound.
SHA_FILE = os.path.join(EV22, "labels.sha256")
LABEL_CELL = "labels_p0.2.jsonl.gz"
MAX_LABEL_SEED = 80


class Refuse(Exception):
    """An input this module will not encode, or a vector it will not decode.

    Its own class and not `AssertionError`, so that the mutation section can catch a check
    firing without also catching a typo in the mutation and printing that as if the check
    had worked.
    """


# ---------------------------------------------------------------------------
# the layout: one generator, so the names and the values cannot drift apart
# ---------------------------------------------------------------------------
def _hot(value, vocab, prefix, present, where):
    """One-hot columns for a closed vocabulary, refusing anything outside it."""
    if present and value not in vocab:
        raise Refuse("%s: %r is not one of %s" % (where, value, list(vocab)))
    for item in vocab:
        yield ("%s_%s" % (prefix, WORDS.get(item, item)),
               1.0 if present and value == item else 0.0)


def cells(record):
    """(name, value) for every cell of the vector, in the record's own field order.

    THE one place the layout is written.  `NAMES` is this generator's names and `vector` is
    its values, so a cell cannot be added to one and forgotten in the other.  The order is
    `observation`'s own: the resolution, the pair's scalars, the counts, the origin rows,
    the exchanges -- which is also the order `render` prints them in.
    """
    if list(record) != list(OBS.RECORD_TYPES):
        want, got = list(OBS.RECORD_TYPES), list(record)
        raise Refuse("the record's fields are not observation's: %d extra (%s), %d"
                     " missing (%s), order %s"
                     % (len(set(got) - set(want)), sorted(set(got) - set(want)),
                        len(set(want) - set(got)), sorted(set(want) - set(got)),
                        "kept" if [k for k in got if k in want] == [k for k in want
                                                                    if k in got]
                        else "changed"))
    got = record["resolved"]
    yield ("res_present", 1.0 if got is not None else 0.0)
    yield ("res_poll_age_s", FILL if got is None else float(got["poll_age_s"]))
    yield from _hot(None if got is None else got["trigger"], TRIGGERS, "res_trigger",
                    got is not None, "the resolution's trigger")
    yield from _hot(None if got is None else got["empty"], EMPTIES, "res_empty",
                    got is not None, "the resolution's empty flag")
    yield ("in_flight", float(record["in_flight"]))
    yield ("consecutive_poll_timeouts", float(record["consecutive_poll_timeouts"]))
    yield from _hot(record["last_poll_status"], STATUSES, "last_poll_status", True,
                    "last_poll_status")
    age = record["fetch_timeout_age_s"]
    yield ("fetch_timeout_present", 0.0 if age is None else 1.0)
    yield ("fetch_timeout_age_s", FILL if age is None else float(age))
    yield ("origins_shown", float(record["origins_shown"]))
    yield ("origins_omitted", 1.0 if record["origins_omitted"] else 0.0)
    yield ("peers_known", float(record["peers_known"]))
    polled = record["polled_peer"]
    yield ("polled_peer_present", 0.0 if polled is None else 1.0)
    yield ("polled_peer", FILL if polled is None else float(polled))
    origins = record["origins"]
    if len(origins) > MAX_ORIGINS:
        raise Refuse("the record shows %d origins, more than the %d slots here"
                     % (len(origins), MAX_ORIGINS))
    for i in range(MAX_ORIGINS):
        row = origins[i] if i < len(origins) else None
        tag = "o%d" % (i + 1)
        yield ("%s_present" % tag, 0.0 if row is None else 1.0)
        yield ("%s_hi" % tag, FILL if row is None else float(row["hi"]))
        yield ("%s_missing" % tag, FILL if row is None else float(row["missing"]))
        yield ("%s_advertisers_omitted" % tag,
               FILL if row is None else float(row["advertisers_omitted"]))
        told = [] if row is None else row["advertised"]
        if len(told) > MAX_PEERS:
            raise Refuse("origin %d names %d advertisers, more than the %d slots here"
                         % (i + 1, len(told), MAX_PEERS))
        for j in range(MAX_PEERS):
            adv = told[j] if j < len(told) else None
            leg = "%s_p%d" % (tag, j + 1)
            yield ("%s_present" % leg, 0.0 if adv is None else 1.0)
            yield ("%s_peer" % leg, FILL if adv is None else float(adv["peer"]))
            yield ("%s_hi" % leg, FILL if adv is None else float(adv["hi"]))
            yield ("%s_notified_max" % leg,
                   FILL if adv is None else float(adv["notified_max"]))
    history = record["exchanges"]
    if len(history) > MAX_EXCHANGES:
        raise Refuse("the record shows %d exchanges, more than the %d slots here"
                     % (len(history), MAX_EXCHANGES))
    yield ("exchanges_count", float(len(history)))
    for i in range(MAX_EXCHANGES):
        ex = history[i] if i < len(history) else None
        tag = "x%d" % (i + 1)
        yield ("%s_present" % tag, 0.0 if ex is None else 1.0)
        yield ("%s_age_s" % tag, FILL if ex is None else float(ex["age_s"]))
        yield from _hot(None if ex is None else ex["trigger"], TRIGGERS,
                        "%s_trigger" % tag, ex is not None,
                        "exchange %d's trigger" % (i + 1))
        yield from _hot(None if ex is None else ex["empty"], EMPTIES, "%s_empty" % tag,
                        ex is not None, "exchange %d's empty flag" % (i + 1))
        yield ("%s_interval_s" % tag, FILL if ex is None else float(ex["interval_s"]))
        yield ("%s_waited_s" % tag, FILL if ex is None else float(ex["waited_s"]))
        yield ("%s_clamped" % tag, FILL if ex is None else (1.0 if ex["clamped"] else 0.0))


def empty_record():
    """The record of a decision that shows nothing: no resolution, no origin, no exchange.

    `NAMES` is read off this one, so that the name list is a property of the layout and not
    of whichever record happened to be first.  It is a record a send-time decision of a
    pair that has never polled really produces, and it is held to `observe`'s own schema in
    `main` rather than trusted.
    """
    return {"resolved": None, "in_flight": 0, "consecutive_poll_timeouts": 0,
            "last_poll_status": None, "fetch_timeout_age_s": None, "origins_shown": 0,
            "origins_omitted": False, "peers_known": 0, "polled_peer": None,
            "origins": [], "exchanges": []}


NAMES = tuple(name for name, _value in cells(empty_record()))
LENGTH = len(NAMES)
INDEX = {name: i for i, name in enumerate(NAMES)}
assert len(INDEX) == LENGTH, "two cells share a name"
NAMES_DIGEST = hashlib.sha256("\n".join(NAMES).encode()).hexdigest()


def _slots():
    """{presence flag: the cells that hold FILL while that flag is 0}.

    Written out rather than guessed from the names, because an origin slot's cells include
    its advertiser slots' and a prefix rule would have to know that; `check_fill` is only
    worth having if what it covers is exact.

    2026-11-25: the first version of `check_fill` did derive the slots from the name
    prefixes and had to special-case the nested advertiser legs, which is how a check that
    is meant to say "nothing hides in the padding" becomes a check whose own coverage
    nobody can state.

    2026-11-25: and the first version written out BY HAND left out the two one-cell
    optional slots, `polled_peer` and `fetch_timeout_age_s`.  A value written into either
    of those cells behind a presence flag of 0 passed `check_fill`, rendered the identical
    text and gave a different vector -- one text with two vectors, the single thing this
    module exists to exclude.  The assertion under `SLOTS` is what stops the next optional
    field from being added to `cells` and forgotten here: every presence flag in the
    layout has to name the cells it covers.
    """
    out = {"res_present": ["res_poll_age_s"]
           + ["res_trigger_%s" % t for t in TRIGGERS]
           + ["res_empty_%s" % WORDS[e] for e in EMPTIES],
           "polled_peer_present": ["polled_peer"],
           "fetch_timeout_present": ["fetch_timeout_age_s"]}
    for i in range(MAX_ORIGINS):
        tag = "o%d" % (i + 1)
        own = ["%s_hi" % tag, "%s_missing" % tag, "%s_advertisers_omitted" % tag]
        for j in range(MAX_PEERS):
            leg = "%s_p%d" % (tag, j + 1)
            inner = ["%s_peer" % leg, "%s_hi" % leg, "%s_notified_max" % leg]
            out["%s_present" % leg] = inner
            own += ["%s_present" % leg] + inner
        out["%s_present" % tag] = own
    for i in range(MAX_EXCHANGES):
        tag = "x%d" % (i + 1)
        out["%s_present" % tag] = (
            ["%s_age_s" % tag]
            + ["%s_trigger_%s" % (tag, t) for t in TRIGGERS]
            + ["%s_empty_%s" % (tag, WORDS[e]) for e in EMPTIES]
            + ["%s_interval_s" % tag, "%s_waited_s" % tag, "%s_clamped" % tag])
    return out


SLOTS = _slots()
assert all(name in INDEX for flag, names in SLOTS.items() for name in [flag] + names)
# Every presence flag in the layout covers its slot's cells.  A one-hot block is named
# `*_trigger_*` and a flag `*_present`, so the flags are exactly the names that end in
# `_present`, and this is what makes `check_fill`'s coverage EXACT rather than nearly so.
assert {name for name in NAMES if name.endswith("_present")} == set(SLOTS), (
    "a presence flag names no cells in SLOTS: %s"
    % sorted({name for name in NAMES if name.endswith("_present")} ^ set(SLOTS)))


def vector(record):
    """One decision's record as a fixed-length numpy float vector, parallel to `NAMES`.

    The record is `observe.observation`'s, from a logged row or from a port observation; no
    other input is read, which is what keeps a node name, the decider ordinal and the
    emulator's own action fields out of the vector.
    """
    out = np.fromiter((value for _name, value in cells(record)), dtype=float,
                      count=LENGTH)
    if out.shape != (LENGTH,):
        raise Refuse("the record produced %s cells, not %d" % (out.shape, LENGTH))
    return out


def unvector(vec):
    """The record back from a vector: the inverse, so that the vector can be checked.

    Not used by a decider.  It exists so that `main` can render the decoded record and
    compare it with the text the original record renders to, which is the only check that
    says the vector carries every quantity the text carries.  Ints come back as ints,
    because `render` prints them with `%d` and a float would print the same but is not what
    the record held.
    """
    if len(vec) != LENGTH:
        raise Refuse("a vector of %d cells cannot be a record of %d" % (len(vec), LENGTH))

    def at(name):
        """One named cell as a float."""
        return float(vec[INDEX[name]])

    def count(name):
        """One named cell as the int the record held and `render` prints with %d."""
        return int(round(at(name)))

    def pick(prefix, vocab, where, needed):
        """The value a one-hot block names, refusing a block with two or no columns set."""
        hot = [item for item in vocab if at("%s_%s" % (prefix, WORDS.get(item, item)))]
        if len(hot) > 1 or (needed and not hot):
            raise Refuse("%s: %d of the %d columns are set" % (where, len(hot),
                                                               len(vocab)))
        return hot[0] if hot else None

    def flagged(flag, value, as_int=False):
        """An optional scalar: None when its presence flag is 0, the cell otherwise."""
        if not at(flag):
            return None
        return count(value) if as_int else at(value)

    resolved = None
    if at("res_present"):
        resolved = {"poll_age_s": at("res_poll_age_s"),
                    "trigger": pick("res_trigger", TRIGGERS, "the resolution's trigger",
                                    True),
                    "empty": pick("res_empty", EMPTIES, "the resolution's empty flag",
                                  True)}
    origins = []
    for i in range(MAX_ORIGINS):
        tag = "o%d" % (i + 1)
        if not at("%s_present" % tag):
            continue
        told = []
        for j in range(MAX_PEERS):
            leg = "%s_p%d" % (tag, j + 1)
            if not at("%s_present" % leg):
                continue
            told.append({"peer": count("%s_peer" % leg), "hi": count("%s_hi" % leg),
                         "notified_max": count("%s_notified_max" % leg)})
        origins.append({"hi": count("%s_hi" % tag), "missing": count("%s_missing" % tag),
                        "advertised": told,
                        "advertisers_omitted": count("%s_advertisers_omitted" % tag)})
    history = []
    for i in range(MAX_EXCHANGES):
        tag = "x%d" % (i + 1)
        if not at("%s_present" % tag):
            continue
        history.append({
            "age_s": at("%s_age_s" % tag),
            "trigger": pick("%s_trigger" % tag, TRIGGERS,
                            "exchange %d's trigger" % (i + 1), True),
            "empty": pick("%s_empty" % tag, EMPTIES,
                          "exchange %d's empty flag" % (i + 1), True),
            "interval_s": at("%s_interval_s" % tag),
            "waited_s": at("%s_waited_s" % tag),
            "clamped": bool(at("%s_clamped" % tag))})
    return {"resolved": resolved, "in_flight": count("in_flight"),
            "consecutive_poll_timeouts": count("consecutive_poll_timeouts"),
            "last_poll_status": pick("last_poll_status", STATUSES, "last_poll_status",
                                     False),
            "fetch_timeout_age_s": flagged("fetch_timeout_present",
                                           "fetch_timeout_age_s"),
            "origins_shown": count("origins_shown"),
            "origins_omitted": bool(at("origins_omitted")),
            "peers_known": count("peers_known"),
            "polled_peer": flagged("polled_peer_present", "polled_peer", as_int=True),
            "origins": origins, "exchanges": history}


# ---------------------------------------------------------------------------
# the three roads to a record, all of them through observe.observation
# ---------------------------------------------------------------------------
def records_from_logged_rows(rows):
    """(neighbour, record) per logged `state_poll_reschedule` row of one node.

    `observe.from_logged_rows` is what rebuilds the five inputs from the log, including
    this pair's own exchange history, and it is imported rather than imitated: a second
    derivation of the history would be a second definition of what a decider sees.  The
    neighbour NAME is yielded beside the record for grouping and never enters it.
    """
    for neighbor, state, resolved, history, now in OBS.from_logged_rows(rows):
        yield neighbor, OBS.observation(state, resolved, history, now, neighbor)


def record_from_port_obs(obs):
    """The record of one replay decision, from the port's own observation dict.

    The port hands a decider `state`, `history`, `resolved` and `where`, and of `where`
    only `ts_ms` and `neighbor` are things a live node has: the node NAME and the `ordinal`
    are simulator-only and are not read here.  This is the same call `../22/label.py`'s
    `text_of` and `../23/decider.py`'s L make before rendering.
    """
    where = obs["where"]
    return OBS.observation(obs["state"], obs["resolved"], obs["history"],
                           where["ts_ms"], where["neighbor"])


def port_obs_of_label_row(row):
    """A label row in the port's observation shape, which is how its consumers read it.

    A label row carries the four things the port hands over, with `ts_ms` and `neighbor` at
    the top level instead of inside `where`; `../22/label.py`'s own self-check rebuilds the
    dict exactly this way.  Rebuilding it here, rather than calling `observation` directly
    on the row's fields, is what makes the logged road and the replay road ONE road: both
    end in `record_from_port_obs`.
    """
    return {"state": row["state"], "resolved": row["resolved"],
            "history": row["history"],
            "where": {"neighbor": row["neighbor"], "ts_ms": row["ts_ms"],
                      "kind": "state_poll_reschedule"}}


def record_from_label_row(row):
    """The record of one label row, through the port shape and the one shared road."""
    return record_from_port_obs(port_obs_of_label_row(row))


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
def check_layout(record, where):
    """The names this record yields are NAMES, in that order.

    Cheap and worth running on every source: it is what says the name list printed with its
    digest describes the vector a reader will actually get, for a record with origins and
    exchanges in it and not only for the empty one NAMES was read off.
    """
    got = tuple(name for name, _value in cells(record))
    if got != NAMES:
        bad = [i for i, (a, b) in enumerate(zip(got, NAMES)) if a != b]
        raise Refuse("%s: the layout differs from NAMES at %d cell(s), first at %s"
                     % (where, len(bad), bad[0] if bad else len(NAMES)))
    return len(got)


def check_fill(vec, where):
    """Every cell of an absent slot holds FILL, so nothing can hide in the padding.

    Without this a vector could carry a value in a slot whose presence flag is 0, which the
    inverse would never read and which would therefore be information the text does not
    carry -- the "nothing else" half of this module's promise.
    """
    bad = [name for flag, names in sorted(SLOTS.items()) if not vec[INDEX[flag]]
           for name in names if vec[INDEX[name]] != FILL]
    if bad:
        raise Refuse("%s: %d cell(s) of an absent slot are not the fill value %r, e.g. %s"
                     % (where, len(bad), FILL, bad[0]))
    return len(bad)


def check_renders_as(vec, want, where):
    """The vector's inverse renders `want`, character for character.

    The headline check, through `check_roundtrip` below, and the whole of "the vector
    carries what the text carries": the inverse is rendered by `observe.render` itself, so
    a quantity the vector dropped, or rounded differently, shows up as a different
    character.  It takes the vector rather than the record so that a mutated VECTOR can be
    handed to the same check.
    """
    got = OBS.render(unvector(vec))
    if got != want:
        at = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b),
                  min(len(got), len(want)))
        raise Refuse("%s: the text differs at character %d: %r against %r"
                     % (where, at, got[at:at + 40], want[at:at + 40]))
    return want


def check_roundtrip(record, where):
    """`render(unvector(vector(record))) == render(record)` for one record."""
    return check_renders_as(vector(record), OBS.render(record), where)


def check_one_vector_per_text(pairs, where):
    """No two records that render alike have different vectors.

    `pairs` is [(text, vector bytes)].  This is the direction that fails when the vector
    carries something the text does not: the text would be the same and the vector would
    not, so a model could act on a difference its text-reading counterpart cannot see.
    """
    by_text = collections.defaultdict(set)
    for text, key in pairs:
        by_text[text].add(key)
        if len(by_text[text]) > 1:
            raise Refuse("%s: one text has %d vectors; its first line is %r"
                         % (where, len(by_text[text]), text.splitlines()[0]))
    return len(by_text)


def check_one_text_per_vector(pairs, where):
    """No two records with the same vector render differently.

    This check cannot be made to fail by any input to this module, and that is stated
    rather than hidden: `unvector` reconstructs every field `render` reads, so one vector
    determines one text by construction.  What makes it non-vacuous is elsewhere -- the
    cap refusals, which stop a long input from being truncated into a short one's vector,
    and `check_fill`, which stops a value from hiding in the padding.  Both are shown
    firing in `main`.
    """
    by_vector = collections.defaultdict(set)
    for text, key in pairs:
        by_vector[key].add(text)
        if len(by_vector[key]) > 1:
            raise Refuse("%s: one vector has %d texts" % (where, len(by_vector[key])))
    return len(by_vector)


def check_no_name(record, where):
    """No value of the record is a node name, which is what makes the vector name-free.

    `observe.py` strips the names and `observe.txt` checks its own records; this is the same
    rule stated where the vector is built, because a record handed to `vector` need not have
    come from this process.
    """
    hits = sorted({hit for text in _strings(record)
                   for hit in OBS.NAME_RE.findall(text)})
    if hits:
        raise Refuse("%s: the record carries the node name(s) %s" % (where, hits))
    return 0


def _strings(obj):
    """Every string inside a nested record, so the name rule can be read over all of it."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _strings(value)


def check_label_seeds(seeds, where):
    """Every label seed read here is a whole number in the tuning half, 1 to 80.

    2026-11-25: the first version tested `int(s) > MAX_LABEL_SEED` only, so a seed of 0 or
    a fractional one passed and would then have been truncated into some other seed by the
    code that uses it.  The bound is two-sided for the same reason `charges.check_log_seeds`
    is: a guard that lets a seed become a DIFFERENT seed is worse than no guard.
    """
    bad = sorted({float(s) for s in seeds
                  if float(s) != int(float(s))
                  or not 1 <= int(float(s)) <= MAX_LABEL_SEED})
    if bad:
        raise Refuse("%s: a label seed read here is a whole number 1 to %d, the tuning"
                     " half; refused %s" % (where, MAX_LABEL_SEED, bad[:8]))
    return sorted({int(s) for s in seeds})


# ---------------------------------------------------------------------------
# reading the label file
# ---------------------------------------------------------------------------
def label_file(lines=None):
    """(path, committed sha256) of the main cell's label file.

    Both come out of `../22/labels.sha256`, so a file whose location moved takes its hash
    line with it.  `../23/rows.py` reads the same file, but its own analysis runs at
    import, so it cannot be imported for the reader and these lines are written here.

    `lines` is that file's lines, as an argument: it defaults to reading the file, and the
    mutation that shows the refusal firing hands in a line list with no label cell in it,
    so the arm is covered without a second copy of the hash file being written anywhere.
    """
    if lines is None:
        with open(SHA_FILE, encoding="utf-8") as fh:
            lines = fh.readlines()
    for line in lines:
        if line.strip() and os.path.basename(line.split()[1]) == LABEL_CELL:
            return line.split()[1], line.split()[0]
    # The BASENAME, not the path: this message is printed by part 6's mutation into an
    # output that has to reproduce byte for byte and carry no machine path.
    raise Refuse("%s names no %s" % (os.path.basename(SHA_FILE), LABEL_CELL))


def label_rows(path, want):
    """(rows of seeds 1 to MAX_LABEL_SEED, the file's sha256).

    The seed cut happens as the lines are parsed, before anything is computed from a row,
    because the selection half's rows may not be counted here at all.  The digest is taken
    over the gzip bytes, which is what the committed one covers.

    2026-11-25, said exactly rather than loosely: a selection-half line IS parsed into a
    dict and then dropped, because the seed is a field of the row.  What the hard rule
    forbids is computing from such a row, and nothing downstream of this function ever
    sees one.  The whole file is hashed and decompressed either way -- the committed digest
    covers the whole file -- so no reading of the rule can be satisfied by reading less.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise Refuse("%s has sha256 %s, and labels.sha256 says %s"
                     % (os.path.basename(path), got, want))
    rows = []
    for line in gzip.decompress(raw).splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["seed"]) <= MAX_LABEL_SEED:
            rows.append(row)
    check_label_seeds([r["seed"] for r in rows], "the rows this module reads")
    PORT.check_seeds([r["seed"] for r in rows], "features.label_rows")
    return rows, got


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


def rename_row(row, mapping):
    """One logged decision row with every node name replaced, values and keys alike.

    The mapping is onto names whose alphabetical order is the reverse of the originals', so
    that anything keyed on name order would move.  Used to show that the vector does not.
    """
    text = json.dumps(row)
    for old in sorted(mapping, key=len, reverse=True):
        text = text.replace('"%s"' % old, '"%s"' % mapping[old])
    return json.loads(text)


def main():
    """Every table, and every check shown failing on a mutated input."""
    p_cells = ("0.2", "0.4")
    seeds = list(range(1, CH.MAX_LOG_SEED + 1))
    head("0  WHAT IS READ, AND WHAT THE VECTOR IS")
    say("emulator logs    stage %s, p %s, seeds %d to %d: the"
        % (CH.STAGE, " and p ".join(p_cells), seeds[0], seeds[-1]))
    say("                 state_poll_reschedule rows of every node, through")
    say("                 observe.from_logged_rows")
    say("label rows       %s, seeds 1 to %d only" % (LABEL_CELL, MAX_LABEL_SEED))
    say("                 2026-11-25, stated exactly: label_rows hashes the gzip bytes,")
    say("                 decompresses them and parses each line, and drops the row if")
    say("                 its seed is above %d -- so a selection-half row is parsed and"
        % MAX_LABEL_SEED)
    say("                 thrown away, and nothing is counted, rendered, fitted or")
    say("                 summarised from one.  The cut is before anything is computed")
    say("                 from a row, not before the line is parsed; the digest the file")
    say("                 is held to covers the whole file and needs every byte of it.")
    say("seed guards      an emulator seed outside 1 to %d and a label seed outside 1 to"
        % CH.MAX_LOG_SEED)
    say("                 %d are refused here, fractional ones included, and"
        % MAX_LABEL_SEED)
    say("                 port.check_seeds refuses anything above %d; part 6 shows all"
        % PORT.MAX_SEED)
    say("                 three firing")
    say("vector length    %d cells" % LENGTH)
    say("name list        sha256 %s" % NAMES_DIGEST)
    say("                 of the %d names joined by newlines" % LENGTH)
    say("fill value       %r in every cell of an absent slot, beside a presence flag of 0"
        % FILL)
    say("caps             %d origins, %d advertisers per origin, %d exchanges; more than"
        % (MAX_ORIGINS, MAX_PEERS, MAX_EXCHANGES))
    say("                 a cap is refused, never truncated")
    say("vocabularies     trigger %s" % list(TRIGGERS))
    say("                 last_poll_status %s" % list(STATUSES))
    say("                 empty %s" % list(EMPTIES))

    logged = []       # (p, seed, node, neighbour, record)
    for p in p_cells:
        for seed in seeds:
            events, _t0 = CH.load_run(p, seed)
            for node in sorted(events):
                for neighbor, record in records_from_logged_rows(events[node]):
                    logged.append((p, seed, node, neighbor, record))
    path, want = label_file()
    rows, digest = label_rows(path, want)
    labelled = [record_from_label_row(row) for row in rows]
    say("logged records   %d" % len(logged))
    for p in p_cells:
        say("                 p %s: %d" % (p, sum(1 for r in logged if r[0] == p)))
    say("label records    %d, out of a file of sha256" % len(labelled))
    say("                 %s == ../22/labels.sha256" % digest)
    say("label seeds      %d distinct, %d to %d"
        % (len({int(r["seed"]) for r in rows}), min(int(r["seed"]) for r in rows),
           max(int(r["seed"]) for r in rows)))

    head("1  THE THREE ROADS TO A RECORD, AND WHAT THEY CANNOT READ")
    say("A logged row and a replay decision reach the vector through ONE function,")
    say("observe.observation, and the replay side goes through the port's own observation")
    say("dict -- the shape label.py's self-check and decider.py's L both build.  A send-")
    say("time decision, where there is no resolution, goes through the same road.")
    say()
    events, _t0 = CH.load_run(p_cells[0], 1)
    sample_rows = events["h2"]
    first = next(e for e in sample_rows
                 if e.get("event") == "decision"
                 and e.get("kind") == "state_poll_reschedule")
    names = sorted({"h0", "h1", "h2", "h3", "h4"})
    mapping = {name: "q%d" % (9 - i) for i, name in enumerate(names)}
    renamed_rows = [rename_row(e, mapping) for e in sample_rows]
    # The renamed rows with the OLD polled name put back on the action alone, so that the
    # decision polls a peer the renamed state no longer names anywhere.
    spoiled_rows = [dict(e, action=dict(e["action"], neighbor=first["action"]["neighbor"]))
                    if e.get("event") == "decision" else e for e in renamed_rows]
    plain = [record for _nb, record in records_from_logged_rows(sample_rows)]
    under = [record for _nb, record in records_from_logged_rows(renamed_rows)]
    spoilt = [record for _nb, record in records_from_logged_rows(spoiled_rows)]
    # A record whose polled peer IS one of the shown positions: on a record where it is
    # null the last comparison below would be an equality for a second reason and would
    # say nothing.  2026-11-25: the first draft used the pair's FIRST decision, whose
    # state names no origin at all, so the "a peer the state no longer names" mutation
    # left the vector identical and the line printed that it must not be.
    k = next(i for i, record in enumerate(plain) if record["polled_peer"] is not None)
    base = plain[k]
    say("one logged row, rendered from its vector and back: %d characters"
        % len(check_roundtrip(base, "the sample logged record")))
    say("the same row with every node name replaced by one of reversed alphabetical")
    say("order (%s): the vector is %s"
        % (", ".join("%s->%s" % (name, mapping[name]) for name in names),
           "unchanged" if np.array_equal(vector(base), vector(under[k]))
           else "DIFFERENT"))
    say("the same rename with the polled neighbour's OLD name put back on the action")
    say("alone, so the decision polls a peer the state no longer names -- a different")
    say("decision and not a leak: the vector is %s"
        % ("unchanged, which it must NOT be"
           if np.array_equal(vector(base), vector(spoilt[k]))
           else "different, which is what says the equality above is not vacuous"))
    obs = port_obs_of_label_row(rows[0])
    other = dict(obs, where=dict(obs["where"], node="q9", ordinal=12345))
    say("one label row through the port shape, then with another node name and ordinal in")
    say("`where`: the vector is %s"
        % ("unchanged" if np.array_equal(vector(record_from_port_obs(obs)),
                                         vector(record_from_port_obs(other)))
           else "DIFFERENT"))
    spoiled = dict(rows[0], node="q9", k=-1, rollin_arm=0.0, label=None)
    say("the same label row with its own `node`, `k`, `rollin_arm` and `label` overwritten:")
    say("the vector is %s"
        % ("unchanged" if np.array_equal(vector(record_from_label_row(rows[0])),
                                         vector(record_from_label_row(spoiled)))
           else "DIFFERENT"))
    send = {"state": rows[0]["state"], "resolved": None, "history": rows[0]["history"],
            "where": {"neighbor": rows[0]["neighbor"], "ts_ms": rows[0]["ts_ms"],
                      "kind": "state_poll"}}
    send_record = record_from_port_obs(send)
    say("the same row as a SEND decision (no resolution): it renders as %r and its"
        % OBS.render(send_record).splitlines()[0])
    say("res_present cell is %.0f, against %.0f at the resolution"
        % (vector(send_record)[INDEX["res_present"]],
           vector(record_from_label_row(rows[0]))[INDEX["res_present"]]))
    say()
    say("What the vector cannot read, because `vector` takes the record and nothing else:")
    say("the node name, the decider ordinal, and the emulator's own action fields.  Of")
    say("those, `fetched` is worth naming: it is on no state_poll_reschedule row of these")
    say("runs at all (it is written on `fetch_plan` rows, which this road never reads),")
    say("and observe.from_logged_rows takes only the %d resolution fields and the %d"
        % (len(OBS.RESOLVED_FIELDS), len(OBS.EXCHANGE_FIELDS)))
    say("exchange fields it names.")

    head("2  THE ROUND TRIP, OVER EVERY RECORD OF EVERY SOURCE")
    say("For each record: render it, build its vector, invert the vector, render the")
    say("inverse, and compare the two texts character for character.  Every quantity the")
    say("text carries is therefore recoverable from the vector, at the rounding render")
    say("itself uses.  Also checked per record: the layout equals NAMES, every absent")
    say("slot is filled with %r, the record carries no node name, and the decoded record"
        % FILL)
    say("passes observe.py's own schema.")
    say()
    say("  The record NAMES itself is read off -- a send-time decision of a pair that has")
    say("  never polled, with no resolution, no origin and no exchange -- fails observe's")
    say("  schema in %d ways, and its round trip %s"
        % (len(OBS.record_faults(empty_record())),
           "matches" if check_roundtrip(empty_record(), "the empty record") else "fails"))
    say()
    sources = (("logged", [r[4] for r in logged]), ("label", labelled))
    pairs = []
    for name, records in sources:
        trips = schema = 0
        for i, record in enumerate(records):
            where = "%s record %d" % (name, i)
            check_layout(record, where)
            check_no_name(record, where)
            vec = vector(record)
            check_fill(vec, where)
            text = OBS.render(record)
            check_renders_as(vec, text, where)
            trips += 1
            schema += 0 if OBS.record_faults(unvector(vec)) else 1
            pairs.append((text, vec.tobytes()))
        say("  %-7s %6d records: %d round trips match, %d decoded records pass"
            % (name, len(records), trips, schema))
        say("          observe.py's own schema")

    head("3  COLLISIONS, COUNTED IN BOTH DIRECTIONS")
    say("Over all %d records of both sources together: how many distinct texts share one"
        % len(pairs))
    say("vector, and how many distinct vectors share one text.  Both are counted, because")
    say("the two failures are different: the first would mean the vector carries something")
    say("the text does not, the second that it lost something the text carries.")
    say()
    by_text = collections.defaultdict(set)
    by_vector = collections.defaultdict(set)
    for text, key in pairs:
        by_text[text].add(key)
        by_vector[key].add(text)
    many_vectors = sorted(len(keys) for keys in by_text.values() if len(keys) > 1)
    many_texts = sorted(len(got) for got in by_vector.values() if len(got) > 1)
    say("  distinct texts                          %6d" % len(by_text))
    say("  distinct vectors                        %6d" % len(by_vector))
    say("  texts rendered by more than one vector  %6d" % len(many_vectors))
    say("  vectors rendering more than one text    %6d" % len(many_texts))
    say("  the two counts of distinct are equal:   %s" % (len(by_text) == len(by_vector)))
    check_one_vector_per_text(pairs, "both sources")
    check_one_text_per_vector(pairs, "both sources")
    say("  both checks run on all %d pairs above and neither refused" % len(pairs))

    head("4  EVERY FEATURE OVER THE EMULATOR RESCHEDULE ROWS")
    say("Rows: the %d logged reschedule records of seeds %d to %d, both cells pooled."
        % (len(logged), seeds[0], seeds[-1]))
    say("A CONSTANT feature is a dead input on this data: it is kept in the vector, so")
    say("that the length and the name list are the same everywhere, and named here rather")
    say("than dropped, so a later fit that leans on one is readable as leaning on nothing.")
    say()
    matrix = np.stack([vector(r[4]) for r in logged])
    say("  matrix %d x %d" % matrix.shape)
    say()
    say("  %-28s %12s %12s %8s %s" % ("feature", "min", "max", "distinct", "constant"))
    constant = []
    for i, name in enumerate(NAMES):
        column = matrix[:, i]
        distinct = len(np.unique(column))
        if distinct == 1:
            constant.append(name)
        say("  %-28s %12.3f %12.3f %8d %s"
            % (name, column.min(), column.max(), distinct,
               "DEAD" if distinct == 1 else ""))
    say()
    say("  constant, i.e. dead on this data: %d of %d" % (len(constant), LENGTH))
    for name in constant:
        say("    %s" % name)
    say()
    say("  NEARLY constant: a cell whose commonest value covers at least %.0f %% of the"
        % (100.0 * 0.99))
    say("  rows, and how many rows depart from it.  These are the cells a fit can lean on")
    say("  while resting on a handful of rows, and the three state fields earlier evidence")
    say("  recorded as never moving are here rather than in the dead list above.")
    say("  %-28s %12s %10s %10s" % ("feature", "commonest", "its rows", "the rest"))
    for i, name in enumerate(NAMES):
        column = matrix[:, i]
        if len(np.unique(column)) == 1:
            continue
        values, counts = np.unique(column, return_counts=True)
        top = int(counts.max())
        if top < 0.99 * len(column):
            continue
        say("  %-28s %12.3f %10d %10d"
            % (name, float(values[counts.argmax()]), top, len(column) - top))

    head("5  WHAT THE VECTOR SEPARATES THAT THE FIVE-CELL STATE DOES NOT")
    say("The five-cell scalar state is the information ADR 0004 line 5 permits a policy")
    say("to read.  Counted over the same %d rows." % len(logged))
    say()
    cell_of = collections.Counter()
    for p, seed, node, _nb, record in logged:
        got = record["resolved"]
        if got["trigger"] == "answered":
            name = "ans_empty" if got["empty"] else "ans_content"
        else:
            name = "to%d" % min(record["consecutive_poll_timeouts"], 3)
        cell_of[name] += 1
    say("  distinct five-cell states   %d" % len(cell_of))
    for name in CH.CELLS:
        say("    %-12s %6d" % (name, cell_of[name]))
    say("  distinct feature vectors    %d" % len({row.tobytes() for row in matrix}))
    say("  distinct rendered texts     %d" % len({OBS.render(r[4]) for r in logged}))
    say("  So a decider reading the vector can tell %d of these %d decisions apart, and"
        % (len({row.tobytes() for row in matrix}), len(logged)))
    say("  one reading the five cells can tell %d apart.  That is a statement about how"
        % len(cell_of))
    say("  much the vector DISTINGUISHES and not about how much of that distinction is")
    say("  signal: nothing here says any of it predicts a cost.")

    head("6  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT")
    say("The mutation is of the INPUT, inside this process: a record or a vector made")
    say("wrong on purpose.  A check that cannot be made to fail by any input is named as")
    say("such below rather than left out.")
    say()
    fired = []

    def shows(name, call):
        """Run a check on a mutated input and require it to refuse."""
        try:
            call()
        except (Refuse, SystemExit, CH.Fail) as exc:
            say("  %-24s FIRES: %s" % (name, exc))
            return True
        say("  %-24s DID NOT FIRE -- this check cannot fail on this input" % name)
        return False

    def with_field(record, **fields):
        """The record with some fields replaced, which is a record of the same shape."""
        return dict(record, **fields)

    deep = next(r for r in labelled
                if r["exchanges"] and r["origins"] and r["origins"][0]["advertised"])
    fired.append(shows("layout/extra field",
                       lambda: vector(dict(base, node="q9"))))
    fired.append(shows("layout/missing field",
                       lambda: vector({k: v for k, v in base.items()
                                       if k != "peers_known"})))
    fired.append(shows("vocabulary/trigger",
                       lambda: vector(with_field(
                           base, resolved=dict(base["resolved"], trigger="pending")))))
    fired.append(shows("vocabulary/status",
                       lambda: vector(with_field(base, last_poll_status="unknown"))))
    fired.append(shows("vocabulary/empty",
                       lambda: vector(with_field(
                           base, resolved=dict(base["resolved"], empty="yes")))))
    fired.append(shows("cap/origins",
                       lambda: vector(with_field(
                           base, origins=base["origins"] * (MAX_ORIGINS + 1)))))
    fired.append(shows("cap/advertisers",
                       lambda: vector(with_field(
                           deep, origins=[dict(row, advertised=row["advertised"]
                                               * (MAX_PEERS + 1))
                                          for row in deep["origins"]]))))
    fired.append(shows("cap/exchanges",
                       lambda: vector(with_field(
                           deep, exchanges=(deep["exchanges"]
                                            * (MAX_EXCHANGES + 1))))))

    def fill_written():
        """A value hidden in the padding of an absent exchange slot."""
        vec = vector(base)
        vec[INDEX["x%d_age_s" % MAX_EXCHANGES]] = 1.0
        check_fill(vec, "a value in an absent slot")
    fired.append(shows("fill/padding", fill_written))

    def fill_in_a_flagged_cell(flag, cell):
        """A value hidden in a ONE-CELL optional slot, behind its presence flag of 0.

        2026-11-25: these two rows exist because `SLOTS` left both of these slots out and
        `check_fill` passed on exactly this vector, while the record it decodes to renders
        the identical text -- one text, two vectors.  They are the two mutations that would
        have caught that, so they are run on every run now.
        """
        def call():
            """Write 7.0 into the slot's cell on a record whose flag is 0."""
            vec = vector(empty_record())
            if vec[INDEX[flag]]:
                raise Refuse("%s is present on the empty record, so this mutation hides"
                             " nothing" % flag)
            vec[INDEX[cell]] = 7.0
            check_fill(vec, "a value behind %s = 0" % flag)
        return call
    for _flag, _cell in (("polled_peer_present", "polled_peer"),
                         ("fetch_timeout_present", "fetch_timeout_age_s")):
        fired.append(shows("fill/%s" % _cell, fill_in_a_flagged_cell(_flag, _cell)))

    def moved_cell():
        """One cell of a present slot changed, against the original record's own text."""
        vec = vector(deep)
        vec[INDEX["o1_missing"]] += 7.0
        check_renders_as(vec, OBS.render(deep), "a vector cell moved by 7")
    fired.append(shows("renders_as/moved cell", moved_cell))

    def unrounded_age():
        """Two records whose ages render alike and whose vectors differ.

        `render` prints an age with one decimal and `observation` has already rounded it
        to one, so a record built BY HAND with an unrounded age renders the same text as
        the rounded one and gets another vector.  That is the shape of "the vector carries
        something the text does not", it is a record this road cannot produce, and it is
        the reason the check is the round trip and not text equality on its own.
        """
        bent = with_field(deep, resolved=dict(deep["resolved"],
                                              poll_age_s=deep["resolved"]["poll_age_s"]
                                              + 0.04))
        check_one_vector_per_text(
            [(OBS.render(r), vector(r).tobytes()) for r in (deep, bent)],
            "a record with an unrounded age beside its rounded self")
    fired.append(shows("one vector per text", unrounded_age))
    fired.append(shows("no name/in the record",
                       lambda: check_no_name(with_field(base, last_poll_status="h3"),
                                             "a name in the record")))
    fired.append(shows("inverse/two columns hot",
                       lambda: unvector(np.ones(LENGTH))))
    fired.append(shows("inverse/wrong length",
                       lambda: unvector(np.zeros(LENGTH + 1))))
    fired.append(shows("label seeds > %d" % MAX_LABEL_SEED,
                       lambda: check_label_seeds([MAX_LABEL_SEED + 1], "mutated")))
    fired.append(shows("a label seed of 0",
                       lambda: check_label_seeds([0], "mutated")))
    fired.append(shows("a fractional label seed",
                       lambda: check_label_seeds([1.5], "mutated")))
    fired.append(shows("log seeds > %d" % CH.MAX_LOG_SEED,
                       lambda: CH.load_run("0.2", CH.MAX_LOG_SEED + 1)))
    fired.append(shows("port.check_seeds > %d" % PORT.MAX_SEED,
                       lambda: PORT.check_seeds([PORT.MAX_SEED + 1], "mutated")))
    fired.append(shows("label file digest",
                       lambda: label_rows(path, "deadbeef")))
    fired.append(shows("the hash file names no cell",
                       lambda: label_file(["%s  labels_p0.4.jsonl.gz\n"
                                           % ("0" * 64)])))
    say()
    say("%d of %d mutations produced a failure" % (sum(fired), len(fired)))
    say()
    say("Two checks no input mutation can fail, named rather than left out:")
    say("- one text per vector (part 3's second direction).  `unvector` reconstructs every")
    say("  field `render` reads, so one vector determines one text by construction.  The")
    say("  way it COULD fail is a long input truncated into a short one's vector, which")
    say("  the three cap refusals above are what prevent, and a value hidden in the")
    say("  padding, which `fill/padding` is what prevents.")
    say("- that the vector reads no node name.  `vector`'s only argument is the record and")
    say("  the record has no name in it, so no input can make it read one.  What can be")
    say("  shown is part 1's equality under a full rename, part 1's difference when the")
    say("  polled name alone is left behind, and the `no name/in the record` refusal.")

    head("7  WHAT THIS MODULE DOES NOT SHOW")
    say("- Nothing about whether these features predict anything.  The round trip says")
    say("  the vector carries what the text carries; whether that is enough to choose an")
    say("  interval is the next step's question and not this one's.")
    say("- Nothing about the rendering being the right input.  It is the input the")
    say("  text-reading decider gets, which is why the vector is held to it.")
    say("- Nothing about a record this road cannot produce.  The vector is held to the")
    say("  records of these two sources; a hand-built record with an unrounded age is the")
    say("  one case shown in part 6 where a text has two vectors, and it is not a record")
    say("  observe.observation ever returns.")
    say("- The three constant-looking state fields are constant ON THIS DATA ONLY.  When")
    say("  a poll is in flight, a fetch has timed out, or an origin is left out, nothing")
    say("  measured here applies, and the cells for them are in the vector either way.")
    say("- Nothing about seeds above %d of the logs or above %d of the labels, and"
        % (CH.MAX_LOG_SEED, MAX_LABEL_SEED))
    say("  nothing about the p 0.4 label cell, which is not read here.")
    say("- No timing.  The cost of one `vector` call is Acceptance 6's question and is")
    say("  measured where the decider is, not here.")
    say()
    say("no check fired on real input; the failures in part 6 are the mutations")
    say("end of features.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
