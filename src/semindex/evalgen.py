"""semindex.evalgen: build one Semantic Index stage-1 evaluation run.

map.md ("トポロジーと負荷" / "切断の起こし方") fixes the evaluation's shape:
a 5-host line h0-h1-h2-h3-h4 with an agent on every host, h0 and h4 each
creating 30 Index records (one every 2s), h2 probing every 10s, and each
link independently going down for whole 5s slots with probability `p`
(consecutive down slots merged into one link_down/link_up interval). The
only thing that varies run to run is `(p, seed)`: everything else in this
module is a fixed constant taken straight from map.md. Stage 2 adds a
second disconnect shape behind `disconnect_duration_s` (stage-2 map "ほかの
切断の形"): down periods uniform on [LO, HI] s, exponential up periods, p
the down-time fraction; without it the slot model above applies unchanged.

This module's only job is turning `(p, seed, stage, outdir)` into the one
`semindex.gen` experiment description that shape implies, then handing it
to `semindex.gen` itself -- `write_experiment()` already owns every byte
of the generated `<name>.yaml` and `<name>/<node>.json` (derives the
content names, fills in `k`/`prefix_routes`/`host_apps`/etc, writes the
files); duplicating any of that here would let the two copies drift.
`build_experiment()` only needs to produce a dict `gen.parse_experiment`
accepts.

Run as:

    python -m semindex.evalgen --p 0.2 --seed 7 --stage s1 --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys

from semindex import gen, schema
from semindex.store import DEFAULT_STORE_SYNCHRONOUS, STORE_SYNCHRONOUS_MODES

# Fixed topology: a 5-host line, agents on every host, h2 in the middle.
HOSTS = 5
LINKS: tuple[tuple[int, int], ...] = tuple((h, h + 1) for h in range(HOSTS - 1))
AGENTS: tuple[int, ...] = tuple(range(HOSTS))
RECORD_ORIGIN_HOSTS: tuple[int, ...] = (0, HOSTS - 1)  # h0 and h4 (map.md: "Index origin は両端の h0 と h4")
PROBE_HOST = HOSTS // 2  # h2

# Lead-in and measurement window (2026-10-23, measured: in ReCefore's
# autotest mode -- no_cli + results_json -- the seed `put` plus the
# warmup_gets block takes about 2.6s after t0 before any scheduled event
# can fire on time, and the scheduler's own `duration` sleep only starts
# once that block finishes). LEAD_IN_S gives that startup slack generous
# margin: every link stays up and no agent does anything until it ends.
# Every record, probe and disconnect slot below is then placed inside
# [WINDOW_START_S, WINDOW_END_S] -- never in the lead-in, never past the
# window's close.
LEAD_IN_S = 10.0
WINDOW_START_S = LEAD_IN_S

# Record generation (map.md "それぞれ 2 秒に 1 件、合計 30 件"), starting
# gen's own startup margin after the window opens rather than after t0 --
# that margin only guards a host_apps readiness race, which is already
# long past by WINDOW_START_S.
RECORD_COUNT = 30
RECORD_INTERVAL_S = 2.0
RECORD_START_S = WINDOW_START_S + gen.MIN_FIRST_RECORD_MARGIN_S

# Probe generation (map.md "probe は真ん中の h2 から 10 秒ごとに出す").
PROBE_INTERVAL_S = 10.0
PROBE_START_S = WINDOW_START_S + 5.0

# Disconnect schedule (map.md "切断の起こし方"): one Bernoulli(p) draw per
# link per 5s slot, independent per link and slot, over the whole
# measurement window -- map.md's own "落ちている時間の割合の期待値が p に
# なる" and this generator's acceptance criterion both describe the window,
# not some sub-interval excluded from it.
DISCONNECT_SLOT_S = 5.0
DISCONNECT_WINDOW_S = 120.0
assert DISCONNECT_WINDOW_S % DISCONNECT_SLOT_S == 0, "DISCONNECT_WINDOW_S must be a whole number of slots"
NUM_SLOTS = int(DISCONNECT_WINDOW_S / DISCONNECT_SLOT_S)
WINDOW_END_S = WINDOW_START_S + DISCONNECT_WINDOW_S

# 2026-11-11 (stage-2 map "ほかの切断の形"): the second disconnect shape,
# chosen by giving `disconnect_duration_s` (LO, HI). Each link alternates
# down periods D ~ Uniform[LO, HI] and up periods U ~ Exponential with mean
# E[D]*(1-p)/p, so p is the long-run down-time fraction there (the slot
# model's p is a per-slot probability, whose down-time fraction is also p).
# Event times are rounded to OUTAGE_AT_DECIMALS decimal places (ms): no 5 s
# grid, and a millisecond is far below the 0.5 s on-time tolerance the
# validator applies to every link event.
OUTAGE_AT_DECIMALS = 3

# 2026-10-23 (review r2, major) fix: an earlier version force-closed every
# link still down at the window's end with a synthetic link_up at
# `at == WINDOW_END_S`, then ran ReCefore TEARDOWN_MARGIN_S seconds past
# that boundary -- fully connected the whole time, but still inside what
# every agent's own final `counters` line (written at ReCefore teardown,
# not at the window boundary) would cover. The measured disconnect
# fraction this produced was below `p` (a 1000-seed check at p=0.4 landed
# at 0.4036 with a 120s denominator, 0.3875 with the actual 125s run
# length) -- the window the fraction was drawn over and the window the run
# actually measured were two different numbers. The fix is two parts:
# `_link_events` below no longer forces a closing link_up (see its own
# comment), and AGENT_STOP_AT_S/AGENT_COUNTERS_SNAPSHOT_AT_S exist so an
# agent's own measurement can stop exactly at the window boundary, not
# whenever ReCefore happens to tear the scenario down.
#
# TEARDOWN_MARGIN_S itself exists for a separate, still-current reason:
# disaster.py (autotest mode -- `no_cli` + `results_json`) runs a
# `duration > 0` scenario as `time.sleep(duration)` then tears down
# immediately, with no `EventScheduler.wait_all()` call to drain the event
# heap first (that call only happens in the `duration == 0` branch) -- an
# event scheduled at exactly `at == duration` races that teardown. Keeping
# every event strictly below `duration` (WINDOW_END_S is, and "duration"
# itself is WINDOW_END_S + this margin) avoids that race regardless of
# which event turns out to be the last one scheduled.
TEARDOWN_MARGIN_S = 5  # int: gen.parse_experiment requires "duration" to be an int, not a float
RECEFORE_DURATION_S = int(WINDOW_END_S) + TEARDOWN_MARGIN_S

# Agent-side measurement boundary, carried into every generated node's
# JSON config via semindex.gen's `stop_at_s`/`counters_snapshot_at_s`
# experiment keys. `stop_at_s` marks when an agent is meant to stop
# synchronizing and write its final `counters` line; `counters_snapshot_at_s`
# marks when it is meant to also write a `counters_snapshot` event (same
# fields as `counters`, plus `ts_ms`) -- a distinct event name, not a
# second `counters` line, because `semindex.analyze` already assumes
# exactly one `counters` line per agent life. Both values only shape what
# this generator writes into each node's config; nothing in this module
# depends on how an agent process reads them.
AGENT_STOP_AT_S = WINDOW_END_S
AGENT_COUNTERS_SNAPSHOT_AT_S: tuple[float, ...] = (WINDOW_START_S,)

# The publisher body (map.md "本体を put する publisher は h0 の 1 台だけ
# （生成器の制約）"): h0 is both the sole put publisher and an Index origin.
# Values match the sibling experiments/t1.json and t2a.json fixtures this
# repo already ships -- no reason for this generator to pick different ones.
PUBLISHER_HOST = 0
PUBLISHER_URI = "ccnx:/content/h0/body"
PUBLISHER_FILE = "./sample-putfile"
PUBLISHER_EXPIRY = 3000
PUBLISHER_CACHE_TIME = 3000

# ReCefore's own "num" option only affects how it names its run directory
# under output_dir (ex{num}_seed{seed}/, core/artifacts.py experiment_dir_name)
# and must be >= 1 (validator.py OptionSpec minimum=1); every run this module
# generates gets its own `output_dir` (the caller's `outdir`), so there is
# nothing for a varying `num` to disambiguate here.
RECEFORE_NUM = 1


def _derive_seed(seed: int, purpose: str) -> int:
    """Derive a random.Random seed deterministically from (seed, purpose).

    Every independently-drawn piece of a run (each link's disconnect
    schedule, each origin's record attrs, the probe's zone draws) gets its
    own `purpose` string and therefore its own, otherwise-uncorrelated
    random.Random instance (map.md "no global random"). Hashing
    (seed, purpose) with SHA-256 and taking the first 8 bytes as a big
    -endian integer means: the same two inputs always derive the same
    stream (determinism/"byte-identical output" requirement), two
    different `purpose` values derive unrelated streams even for the same
    seed (so adding or removing one draw site, e.g. a 6th host, can never
    perturb another site's draws), and the derivation does not depend on
    Python's per-process string hash randomization the way `hash()` or
    iterating a plain `set`/`frozenset` would.

    2026-10-23 (review r1, finding 3) fix: `p` used to be part of this
    hash's input, for every purpose including "records:hN" and "probe:h2"
    -- so the same seed with a different p drew a completely unrelated
    record/probe stream (a reviewer's reproduction: seed=7's h0 first
    record was ".../zone1/landslide/video" at p=0.2 but
    ".../zone5/fire/sensor" at p=0.3), instead of the spec's "乱数は seed
    だけから決まり". `p` is no longer
    an input here at all; `_link_events` (the only purpose where p is
    supposed to matter) still takes p as a parameter, but only to compare
    against this seed-only stream's draws, never to pick the stream.
    """
    payload = json.dumps({"seed": seed, "purpose": purpose}, sort_keys=True).encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def _rng_for(seed: int, purpose: str) -> random.Random:
    """A random.Random seeded deterministically from (seed, purpose). See `_derive_seed`."""
    return random.Random(_derive_seed(seed, purpose))


def _validate_p(p: float) -> None:
    """Reject a disconnect probability that is not a finite number in [0, 1].

    2026-10-23 (review r1, finding 4) fix: `rng.random() < p` is simply
    False for every draw when p is negative or NaN, and simply True for
    every draw when p is >1 or +inf -- so an out-of-range or non-finite p
    used to produce a normal-looking, fully valid experiment ("never down"
    or "always down") instead of an error, silently mislabeling the run's
    condition. This runs once, before any random draw, so no path through
    this module can read a p value that was never range-checked.
    """
    if not isinstance(p, (int, float)) or isinstance(p, bool) or not (0.0 <= p <= 1.0):
        raise gen.ExperimentError(f"p must be a finite number in [0, 1], got {p!r}")


def _format_p(p: float) -> str:
    """Render `p` as a short decimal string safe to use as a path/name component.

    Rounds to 4 decimal places (ample for the evaluation grid's p in
    {0, 0.1, 0.2, 0.3, 0.4}) and strips trailing zeros so p=0 renders as
    "0" rather than "0.0000".
    """
    text = f"{p:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _format_p_fixed(p: float) -> str:
    """Render `p` as a fixed-width, three-digit percentage for the run name and group
    (0 -> "000", 0.1 -> "010", 1 -> "100"), so every p gives the same name length.

    2026-10-26 (stage-2 map "生成器と集計"): stage 1 named runs with `_format_p`, so
    p = 0 gave `s1_p0_seedN`, two characters shorter than every other p; the
    group is part of every Interest name, and that alone made p = 0's control
    traffic about 47 B/record lower (evidence/s1-breakdown/). Three digits is
    also exactly as long as stage 1's "0.1".."0.4", so stage-2 names cost what
    stage 1's p > 0 names did. Store directory names keep `_format_p` (p0/,
    p0.1/, ...) so stage-1 paths do not move.

    A p that is not a whole percent is rejected rather than rounded: two
    different p values would otherwise share a label, the same silent
    mislabeling `_validate_p` exists to prevent.
    """
    percent = round(p * 100)
    if abs(p * 100 - percent) > 1e-9:
        raise gen.ExperimentError(f"p must be a whole percent (a multiple of 0.01), got {p!r}")
    return f"{percent:03d}"


# 2026-10-26 (stage-2 map "選択肢の間隔のゆらぎ"): the choices-mode jitter j a
# run gets when it gives choices but no j -- the same value as SyncConfig's own
# default, written into every agent config explicitly so a stored run says
# which j it ran with.
DEFAULT_CHOICE_JITTER_FRAC = 0.1


def resolve_choice_jitter_frac(
    state_poll_interval_choices_s: list[float] | None, state_poll_choice_jitter_frac: float | None
) -> float | None:
    """The j a run actually uses: the given one, else DEFAULT_CHOICE_JITTER_FRAC when
    choices are given; None without choices (stage 1's policy has no j)."""
    if state_poll_interval_choices_s is None:
        return None
    return DEFAULT_CHOICE_JITTER_FRAC if state_poll_choice_jitter_frac is None else state_poll_choice_jitter_frac


def resolve_store_synchronous(store_synchronous: str | None) -> str:
    """The index.db `PRAGMA synchronous` mode a run's agents actually use: the given
    one, else the agents' own default (DEFAULT_STORE_SYNCHRONOUS, NORMAL), which is
    what an agent config without the key runs with."""
    return DEFAULT_STORE_SYNCHRONOUS if store_synchronous is None else store_synchronous


def parse_poll_policy_arg(text: str | None) -> dict | None:
    """Parse the `--poll-policy` CLI string as JSON, or None when the flag was not given.

    Only the JSON itself is checked here; its shape (an object with intervals_s
    and jitter_frac, the agent's rules) is gen.parse_state_poll_policy's to
    check, reached through validate_settings. Public so every CLI that takes the
    flag parses it the same way. A fixed message rather than the decoder's:
    the decoder's wording and offsets are not this module's to promise.
    """
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise gen.ExperimentError("--poll-policy must be valid JSON") from None


def parse_disconnect_duration_arg(values: list[float] | None) -> tuple[float, float] | None:
    """The `--disconnect-duration-s LO HI` pair as the (lo, hi) tuple the kwarg takes,
    or None when the flag was not given. Range checks are `validate_disconnect_duration`'s."""
    return None if values is None else (values[0], values[1])


def validate_settings(
    k: int,
    state_poll_interval_choices_s: list[float] | None,
    state_poll_choice_jitter_frac: float | None = None,
    max_in_flight_per_neighbor: int | None = None,
    store_synchronous: str | None = None,
    state_poll_policy: dict | None = None,
) -> None:
    """Reject a records-per-generation count that is not an integer >= 1, or interval
    choices / a choice jitter / a poll policy table / a per-neighbor in-flight cap /
    a store synchronous mode semindex.gen would refuse (see
    gen.parse_state_poll_interval_choices_s, gen.parse_state_poll_choice_jitter_frac,
    gen.parse_state_poll_policy, gen.parse_max_in_flight_per_neighbor and
    gen.parse_store_synchronous), a poll policy given together with choices or j
    included.

    Public so a sweep can refuse bad settings once, up front, instead of failing
    every (p, seed) run of it the same way.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise gen.ExperimentError(f"k must be an integer >= 1, got {k!r}")
    # 2026-11-03: before the choices keys, in gen.parse_experiment's order, so a
    # policy with j but no choices reports the combination, not "requires choices".
    gen.parse_state_poll_policy(
        state_poll_policy,
        have_choices=state_poll_interval_choices_s is not None,
        have_choice_jitter=state_poll_choice_jitter_frac is not None,
    )
    gen.parse_state_poll_interval_choices_s(state_poll_interval_choices_s)
    gen.parse_state_poll_choice_jitter_frac(
        state_poll_choice_jitter_frac, have_choices=state_poll_interval_choices_s is not None
    )
    gen.parse_max_in_flight_per_neighbor(max_in_flight_per_neighbor)
    gen.parse_store_synchronous(store_synchronous)


def validate_disconnect_duration(p: float, disconnect_duration_s: tuple[float, float] | None) -> None:
    """Reject a down-period range (LO, HI) that is not two finite numbers with
    0 < LO <= HI, or, with one given, a p of 1 or more; None (the slot model) passes.

    p must stay below 1 with a range given because p is then the long-run
    down-time fraction, and the mean up period E[D]*(1-p)/p is 0 at p = 1:
    the link would never come back up, which is no outage *shape* at all.
    (The slot model accepts p = 1 as "down for every slot"; that is its own
    per-slot meaning of p, and left as it was.) p itself is range-checked by
    `_validate_p` first wherever a run is built, so this only adds the < 1 rule.

    Public, taking one p, so a sweep can check every p of its grid once, up
    front (same reasoning as `validate_settings`).
    """
    if disconnect_duration_s is None:
        return
    value = disconnect_duration_s
    shape_ok = (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value)
        and 0 < value[0] <= value[1]
    )
    if not shape_ok:
        raise gen.ExperimentError(
            f"disconnect_duration_s must be two finite numbers lo, hi with 0 < lo <= hi, got {value!r}"
        )
    if p >= 1:
        raise gen.ExperimentError(f"p must be < 1 with disconnect_duration_s (the down-time fraction), got {p!r}")


def _record_schedule(host: int, seed: int, k: int = 1) -> list[dict]:
    """Build one origin host's schedule: RECORD_COUNT generation times, k records at each,
    attrs drawn uniformly from the seed record by record.

    k records share one `at_s` (stage-2 map "負荷"): the agent turns every due
    entry into a record in one step (agent.py's _drain_schedule), so a batch
    needs no agent change. Each record draws its own attrs from the same
    per-host stream, in schedule order, so k = 1 draws exactly stage 1's
    sequence and gives stage 1's schedule.

    `zone`/`disaster`/`media` are drawn from `sorted(schema.<VOCAB>)`, not
    the frozenset itself or a plain `list()` of it: frozenset iteration
    order depends on Python's per-process string hash randomization, which
    would make the same seed draw different attrs across process runs. No
    `p` parameter here on purpose (finding 3, see `_derive_seed`) -- these
    attrs must be identical for a given seed no matter what p is.
    """
    rng = _rng_for(seed, f"records:h{host}")
    zones = sorted(schema.ZONES)
    disasters = sorted(schema.DISASTERS)
    media = sorted(schema.MEDIA)
    entries = []
    for i in range(RECORD_COUNT):
        for _ in range(k):
            entries.append(
                {
                    "at_s": RECORD_START_S + i * RECORD_INTERVAL_S,
                    "zone": rng.choice(zones),
                    "disaster": rng.choice(disasters),
                    "media": rng.choice(media),
                }
            )
    return entries


def _probe_schedule(seed: int) -> list[dict]:
    """Build h2's probe schedule: every PROBE_INTERVAL_S, zone drawn uniformly from the seed.

    No `p` parameter here on purpose (finding 3, see `_derive_seed`) -- the
    probe's zone draws must be identical for a given seed no matter what p is.
    """
    rng = _rng_for(seed, f"probe:h{PROBE_HOST}")
    zones = sorted(schema.ZONES)
    entries = []
    at_s = PROBE_START_S
    while at_s < WINDOW_END_S:
        entries.append({"at_s": at_s, "zone": rng.choice(zones)})
        at_s += PROBE_INTERVAL_S
    return entries


def _link_events(seed: int, p: float, disconnect_duration_s: tuple[float, float] | None = None) -> list[dict]:
    """Build every link's link_down/link_up events for this (seed, p).

    With `disconnect_duration_s` given (2026-11-11), the outage shape of
    `_outage_link_events` instead: each link from its own rng stream under
    the purpose "outage:{a}:{b}", deliberately not the slot model's
    "link:{a}:{b}", so the two shapes never read as drawing one coupled
    pattern. The rest of this docstring is the slot model, the default; its
    output is unchanged by the option's existence.

    One Bernoulli(p) draw per link per DISCONNECT_SLOT_S slot (independent
    per link -- its own random.Random stream -- and per slot), with
    consecutive down slots on the same link merged into a single
    link_down/link_up pair. A link still down in the last drawn slot is
    left open -- no closing link_up is appended for it (2026-10-23, review
    r2: an earlier version forced one at `at == WINDOW_END_S`, which kept
    every link connected for the rest of the run and biased the measured
    disconnect fraction low; see WINDOW_END_S's own comment). Every event
    this function emits falls inside [WINDOW_START_S, WINDOW_END_S]; the
    separate teardown cushion that keeps a run started this way from
    racing disaster.py's teardown lives entirely in RECEFORE_DURATION_S
    (see that constant's comment for why).

    2026-10-23 (review r1, finding 3): each link's rng stream is seeded
    from (seed, purpose) only, never p (see `_derive_seed`) -- so the same
    seed draws the exact same sequence of `rng.random()` values at every
    p, and only the `< p` comparison below changes with p. This is what
    lets a seed=7 comparison across p values isolate the disconnect rate
    alone, instead of also redrawing an unrelated down/up pattern. The
    outage shape keeps a weaker form of the same promise (2026-11-11): for
    every p > 0 the stream is the same and is read in the same fixed order,
    but p also scales every up period, so the pattern changes with p beyond
    one comparison (see `_outage_link_events`); p = 0 draws nothing.
    """
    events: list[dict] = []
    if disconnect_duration_s is not None:
        lo, hi = disconnect_duration_s
        for a, b in LINKS:
            if p == 0:
                continue  # no outages; also keeps p = 0 from dividing by p below
            rng = _rng_for(seed, f"outage:{a}:{b}")
            events.extend(_outage_link_events(rng, p, lo, hi, [a, b]))
        events.sort(key=lambda e: (e["at"], e["nodes"][0], e["nodes"][1], e["type"]))
        # 2026-11-11 trap: evalvalid._check_link_events matches results rows to
        # events by exact `scheduled_at` and relies on one link never having two
        # events at one `at`; and at a shared `at` the sort above (and
        # ReCefore's scheduler, which keeps this order) would run link_down
        # before link_up and leave the link up. `_outage_link_events` rules
        # this out by construction; this assert keeps it ruled out.
        keys = [(tuple(e["nodes"]), e["at"]) for e in events]
        assert len(set(keys)) == len(keys), "two events of one link share an `at`"
        return events
    for a, b in LINKS:
        rng = _rng_for(seed, f"link:{a}:{b}")
        down_slots = [rng.random() < p for _ in range(NUM_SLOTS)]
        run_start = None
        for i in range(NUM_SLOTS):
            is_down = down_slots[i]
            if is_down and run_start is None:
                run_start = i
            elif not is_down and run_start is not None:
                events.append({"at": WINDOW_START_S + run_start * DISCONNECT_SLOT_S, "type": "link_down", "nodes": [a, b]})
                events.append({"at": WINDOW_START_S + i * DISCONNECT_SLOT_S, "type": "link_up", "nodes": [a, b]})
                run_start = None
        if run_start is not None:
            events.append({"at": WINDOW_START_S + run_start * DISCONNECT_SLOT_S, "type": "link_down", "nodes": [a, b]})
    # Sorted so the generated YAML reads chronologically (map.md "引いた予定は
    # 生成した YAML にそのまま残るので、あとから見直せる") -- ReCefore's own
    # scheduler (runtime/scheduler.py) heap-sorts events at run time
    # regardless, so this ordering is for a human reader, not correctness.
    events.sort(key=lambda e: (e["at"], e["nodes"][0], e["nodes"][1], e["type"]))
    return events


def _outage_link_events(rng, p: float, lo: float, hi: float, nodes: list[int]) -> list[dict]:
    """One link's link_down/link_up events under the outage shape, 0 < p < 1.

    The link alternates down periods D ~ Uniform[lo, hi] and up periods
    U ~ Exponential with mean E[D]*(1-p)/p, E[D] = (lo+hi)/2, so the
    long-run share of time spent down is p. No 5 s grid, and no minimum up
    time (by the user's choice: a floor would tune the environment to the
    state-poll policies' intervals).

    Stationary start (2026-11-11 trap): a process that always starts up
    runs below p over its first mean up period or so (40 s at p 0.2), while
    the stage-2 map's criterion is that the expected down fraction over the
    window is p. Measured over 20 000 (seed, link) windows, an always-up
    start gives 0.192 at p 0.2 and 0.390 at p 0.4 (about p(1-p)*5.067/120
    short; small because the up periods are memoryless, but systematic).
    So at WINDOW_START_S the link is down with probability p. If down,
    link_down is emitted at
    WINDOW_START_S (as the slot model does for a down slot 0) and the
    residual down time is v * D_lb, with D_lb a length-biased draw of D,
    sqrt(lo^2 + u*(hi^2 - lo^2)) (an observer more likely lands in a long
    outage), and u, v fresh uniforms; for [8, 12] its mean is 5.067 s. If
    up, the residual up time is a fresh exponential (memoryless). The
    exponential up draw also serves as the up period that follows a
    down start, so both branches read the same stream positions.

    Draw order is fixed whatever the state: four start uniforms (state,
    D_lb's u, v, the up period after the start), then per cycle one for the
    down period and one for the up period. So for every p > 0 the same seed
    reads the same uniform sequence at the same positions; p changes the
    state comparison and scales the up periods. `rng` is anything with
    `random()`, so a test can script the draws.

    Times stay continuous while drawing; only the emitted `at` is rounded to
    OUTAGE_AT_DECIMALS (ms). Two rules then apply to the rounded times:

    - When an outage's rounded start is at or before the previous outage's
      rounded end, the two are merged into one. That merge exists solely so
      one link never gets two events at the same `at` (evalvalid matches
      link events by exact `scheduled_at`, and at a shared `at` the
      link_down would sort first and leave the link up). It is not a floor
      on up time: every gap of 1 ms or more is kept, and a gap under 1 ms is
      kept or merged depending on where its two ends fall against the ms
      grid (2026-11-11 review nit: this used to say "a gap that rounds to
      0 ms", which is not what the comparison below tests).
    - An outage whose own rounded length is 0 ms (the start's residual when
      v is tiny, or any outage if lo < 0.5 ms) is no outage and is dropped,
      for the same same-`at` reason. It covers no time, so nothing is lost.

    As in the slot model (2026-10-23 review r2), a link still down at
    WINDOW_END_S is left open with no closing link_up, and only events inside
    [WINDOW_START_S, WINDOW_END_S) are emitted: a link_down that would round
    to WINDOW_END_S or later covers no window time and is not emitted.
    """
    mean_up_s = (lo + hi) / 2 * (1 - p) / p
    u_state, u_length, u_fraction, u_up = rng.random(), rng.random(), rng.random(), rng.random()
    # Inverse CDFs on rng.random()'s [0, 1): log(1 - u) is always finite.
    downs: list[tuple[float, float]] = []
    t = WINDOW_START_S
    if u_state < p:
        residual_s = u_fraction * math.sqrt(lo * lo + u_length * (hi * hi - lo * lo))
        downs.append((t, t + residual_s))
        t += residual_s
    t += -mean_up_s * math.log(1 - u_up)
    while t < WINDOW_END_S:
        down_s = lo + rng.random() * (hi - lo)
        up_s = -mean_up_s * math.log(1 - rng.random())
        downs.append((t, t + down_s))
        t += down_s + up_s

    merged: list[list[float]] = []
    for start, end in downs:
        start_r, end_r = round(start, OUTAGE_AT_DECIMALS), round(end, OUTAGE_AT_DECIMALS)
        if end_r <= start_r:
            continue
        if merged and start_r <= merged[-1][1]:
            merged[-1][1] = end_r
            continue
        merged.append([start_r, end_r])

    events: list[dict] = []
    for start_r, end_r in merged:
        if start_r >= WINDOW_END_S:
            break
        events.append({"at": start_r, "type": "link_down", "nodes": list(nodes)})
        if end_r < WINDOW_END_S:
            events.append({"at": end_r, "type": "link_up", "nodes": list(nodes)})
    return events


def build_experiment(
    p: float,
    seed: int,
    stage: str,
    outdir: str,
    *,
    k: int = 1,
    state_poll_interval_choices_s: list[float] | None = None,
    state_poll_choice_jitter_frac: float | None = None,
    max_in_flight_per_neighbor: int | None = None,
    store_synchronous: str | None = None,
    state_poll_policy: dict | None = None,
    disconnect_duration_s: tuple[float, float] | None = None,
) -> dict:
    """Build the one `semindex.gen` experiment description this (p, seed, stage) run needs.

    `stage` and `outdir` only ever affect bookkeeping fields (`name`,
    `group`, `output_dir`) -- never a random draw. So the random SCHEDULE
    this returns (record attrs, probe zones, disconnect events) depends
    only on (p, seed): it is identical for two calls that share (p, seed)
    but differ in `stage`/`outdir`. The written FILES are not, though --
    2026-10-23 (review r1, finding 5) corrects an earlier version of this
    docstring that claimed byte-identical output regardless of stage/
    outdir: `name`/`group` are derived from `stage`, and `output_dir`
    (baked into the generated YAML) from `outdir`, so those bytes do
    change. Byte-identical generated files require all four of (p, seed,
    stage, outdir) to match, not just (p, seed).

    `k` (stage-2 map "負荷") is how many records each origin creates at each
    of its RECORD_COUNT generation times; `state_poll_interval_choices_s`, when
    given, is passed through to every agent config (with the run seed, which
    the agents' choice rng needs -- see semindex.gen). `max_in_flight_per_neighbor`,
    when given, is passed through to every agent config as its cap on outstanding
    fetch Interests per neighbor; None writes nothing, so the agents keep their
    own default (8). `store_synchronous`, likewise only when given, is passed
    through as the agents' index.db `PRAGMA synchronous` mode; None writes
    nothing, so the agents use their own default. `state_poll_policy` (2026-11-03),
    the failure-reactive poll table {"intervals_s": [...], "jitter_frac": [...]},
    is passed through as given, with the run seed, to every agent config; it
    cannot be combined with choices or j, and None writes nothing.
    `disconnect_duration_s` (2026-11-11), a down-period range (LO, HI) in
    seconds, switches the disconnect schedule from the 5 s slot model to the
    outage shape (see `_outage_link_events`), where p is the down-time
    fraction and must be < 1; it changes only `events` and writes no key of
    its own (the events are explicit, and evalrun's meta.json records the
    range). The defaults give stage 1's schedule and agent configs, byte for byte.

    2026-10-29: the agents' own synchronous default changed from SQLite's FULL
    (no PRAGMA issued) to NORMAL. Files generated without `store_synchronous`
    are unchanged, but the runs they drive now open index.db with NORMAL; to
    reproduce a run recorded before that change whose meta.json has no mode
    (null or no key), pass "FULL".
    """
    _validate_p(p)
    validate_disconnect_duration(p, disconnect_duration_s)
    validate_settings(
        k,
        state_poll_interval_choices_s,
        state_poll_choice_jitter_frac,
        max_in_flight_per_neighbor,
        store_synchronous=store_synchronous,
        state_poll_policy=state_poll_policy,
    )
    name = f"{stage}_p{_format_p_fixed(p)}_seed{seed}"
    schedule = {str(host): _record_schedule(host, seed, k) for host in RECORD_ORIGIN_HOSTS}
    raw = {
        "name": name,
        "num": RECEFORE_NUM,
        "seed": seed,
        "group": name,
        "hosts": HOSTS,
        "links": [list(pair) for pair in LINKS],
        # RECEFORE_DURATION_S, not WINDOW_END_S: ReCefore must run
        # TEARDOWN_MARGIN_S past the window's close so no event (an open
        # link_down at WINDOW_END_S included) ever sits at `at == duration`
        # (see RECEFORE_DURATION_S's comment).
        "duration": RECEFORE_DURATION_S,
        "output_dir": os.path.abspath(outdir),
        "agents": list(AGENTS),
        "publisher": {
            "host": PUBLISHER_HOST,
            "uri": PUBLISHER_URI,
            "file": PUBLISHER_FILE,
            "expiry": PUBLISHER_EXPIRY,
            "cache_time": PUBLISHER_CACHE_TIME,
        },
        "schedule": schedule,
        "probes": {str(PROBE_HOST): _probe_schedule(seed)},
        "events": _link_events(seed, p, disconnect_duration_s),
        "stop_at_s": AGENT_STOP_AT_S,
        "counters_snapshot_at_s": list(AGENT_COUNTERS_SNAPSHOT_AT_S),
    }
    if state_poll_interval_choices_s is not None:
        raw["state_poll_interval_choices_s"] = list(state_poll_interval_choices_s)
        raw["state_poll_choice_jitter_frac"] = resolve_choice_jitter_frac(
            state_poll_interval_choices_s, state_poll_choice_jitter_frac
        )
    # No default is made explicit here, unlike j: leaving the key out keeps
    # every generated file byte-identical to the ones made before the cap
    # could be set, and the agent's default is then the one that applies.
    if max_in_flight_per_neighbor is not None:
        raw["max_in_flight_per_neighbor"] = max_in_flight_per_neighbor
    # Same reasoning: absent, the files stay what they were before the key
    # (the agents' own default, NORMAL since 2026-10-29, then applies).
    if store_synchronous is not None:
        raw["store_synchronous"] = store_synchronous
    # Same reasoning (2026-11-03): absent, the files stay what they were before
    # the key, so stages A/B/C re-collected on this code generate main's bytes.
    if state_poll_policy is not None:
        raw["state_poll_policy"] = state_poll_policy
    return raw


def generate_run(
    p: float,
    seed: int,
    stage: str,
    outdir: str,
    *,
    k: int = 1,
    state_poll_interval_choices_s: list[float] | None = None,
    state_poll_choice_jitter_frac: float | None = None,
    max_in_flight_per_neighbor: int | None = None,
    store_synchronous: str | None = None,
    state_poll_policy: dict | None = None,
    disconnect_duration_s: tuple[float, float] | None = None,
) -> list[str]:
    """Build and write one run's experiment description, ReCefore YAML and agent configs.

    The experiment description itself is written first, as
    `<outdir>/<name>.experiment.json` -- not a temp file discarded after
    use -- so the drawn disconnect schedule and record attrs stay
    inspectable next to the generated YAML they produced (map.md, same
    rationale as leaving the schedule's `events` in the YAML itself). It
    is then handed to `semindex.gen.write_experiment`, the single place
    that already turns an experiment description into `<name>.yaml` and
    `<name>/<node>.json`, so this module never re-implements that step.

    2026-10-23 (review r1, finding 1) fix: `gen.parse_experiment(raw)` runs
    here, before any directory is created or any file opened. It used to
    run only inside `gen.write_experiment`, several lines after the
    `<outdir>/<name>.experiment.json` write above -- so a `stage` that made
    `name` escape `outdir` (e.g. "../escaped", or an absolute path, for
    which `os.path.join` discards `outdir_abspath` entirely) still got
    rejected, but only after that escaping write had already landed on
    disk, or outside it.

    `k`, `state_poll_interval_choices_s`, `state_poll_choice_jitter_frac`,
    `max_in_flight_per_neighbor`, `store_synchronous`, `state_poll_policy` and
    `disconnect_duration_s` are passed straight to `build_experiment`.

    Returns every path written, experiment description first.
    """
    raw = build_experiment(
        p,
        seed,
        stage,
        outdir,
        k=k,
        state_poll_interval_choices_s=state_poll_interval_choices_s,
        state_poll_choice_jitter_frac=state_poll_choice_jitter_frac,
        max_in_flight_per_neighbor=max_in_flight_per_neighbor,
        store_synchronous=store_synchronous,
        state_poll_policy=state_poll_policy,
        disconnect_duration_s=disconnect_duration_s,
    )
    gen.parse_experiment(raw)  # raises gen.ExperimentError before any filesystem write

    outdir_abspath = os.path.abspath(outdir)
    os.makedirs(outdir_abspath, exist_ok=True)

    experiment_path = os.path.join(outdir_abspath, f"{raw['name']}.experiment.json")
    with open(experiment_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, sort_keys=True)
        f.write("\n")

    written = gen.write_experiment(experiment_path, outdir_abspath)
    return [experiment_path] + written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the evaluation-run generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p",
        type=float,
        required=True,
        help="per-link disconnect probability: per 5 s slot for the slot model, "
        "the long-run down-time fraction (< 1) with --disconnect-duration-s",
    )
    parser.add_argument("--seed", type=int, required=True, help="seed for every random draw in this run")
    parser.add_argument("--stage", type=str, required=True, help="stage label, used only in the run's name/group")
    parser.add_argument("--out", type=str, required=True, dest="outdir", help="directory to write the run into")
    parser.add_argument("--k", type=int, default=1, help="records each origin creates per generation time (default 1)")
    parser.add_argument(
        "--interval-choices",
        type=float,
        nargs="+",
        default=None,
        dest="interval_choices",
        help="state-poll interval choices (s) for every agent; omit for the default 2 s +/- 20%% policy",
    )
    parser.add_argument(
        "--choice-jitter",
        type=float,
        default=None,
        dest="choice_jitter",
        help=f"jitter fraction j applied to each chosen interval (default {DEFAULT_CHOICE_JITTER_FRAC} with choices)",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        dest="max_in_flight",
        help="cap on outstanding fetch Interests per neighbor for every agent; omit for the agent default (8)",
    )
    parser.add_argument(
        "--store-synchronous",
        choices=STORE_SYNCHRONOUS_MODES,
        default=None,
        dest="store_synchronous",
        help=f"PRAGMA synchronous for every agent's index.db; omit for the agent default ({DEFAULT_STORE_SYNCHRONOUS})",
    )
    parser.add_argument(
        "--poll-policy",
        default=None,
        dest="poll_policy",
        help='state-poll table as JSON, e.g. \'{"intervals_s": [10.043, 3.695], "jitter_frac": [0.1, 0.1]}\'; '
        "not with --interval-choices/--choice-jitter",
    )
    parser.add_argument(
        "--disconnect-duration-s",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=None,
        dest="disconnect_duration_s",
        help="down periods uniform on [LO, HI] s with exponential up periods (p = down-time fraction); "
        "omit for the 5 s slot model",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: generate one (p, seed, stage) run into --out."""
    args = parse_args(argv)
    try:
        written = generate_run(
            args.p,
            args.seed,
            args.stage,
            args.outdir,
            k=args.k,
            state_poll_interval_choices_s=args.interval_choices,
            state_poll_choice_jitter_frac=args.choice_jitter,
            max_in_flight_per_neighbor=args.max_in_flight,
            store_synchronous=args.store_synchronous,
            state_poll_policy=parse_poll_policy_arg(args.poll_policy),
            disconnect_duration_s=parse_disconnect_duration_arg(args.disconnect_duration_s),
        )
    except gen.ExperimentError as exc:
        print(f"semindex.evalgen: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
