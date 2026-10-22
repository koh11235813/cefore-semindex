"""semindex.evalgen: build one Semantic Index stage-1 evaluation run.

map.md ("トポロジーと負荷" / "切断の起こし方") fixes the evaluation's shape:
a 5-host line h0-h1-h2-h3-h4 with an agent on every host, h0 and h4 each
creating 30 Index records (one every 2s), h2 probing every 10s, and each
link independently going down for whole 5s slots with probability `p`
(consecutive down slots merged into one link_down/link_up interval). The
only thing that varies run to run is `(p, seed)`: everything else in this
module is a fixed constant taken straight from map.md.

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
import os
import random
import sys

from semindex import gen, schema

# Fixed topology: a 5-host line, agents on every host, h2 in the middle.
HOSTS = 5
LINKS: tuple[tuple[int, int], ...] = tuple((h, h + 1) for h in range(HOSTS - 1))  # (0,1),(1,2),(2,3),(3,4)
AGENTS: tuple[int, ...] = tuple(range(HOSTS))
RECORD_ORIGIN_HOSTS: tuple[int, ...] = (0, HOSTS - 1)  # h0 and h4 (map.md: "Index origin は両端の h0 と h4")
PROBE_HOST = HOSTS // 2  # h2

# Record generation (map.md "それぞれ 2 秒に 1 件、合計 30 件"). The first
# record is placed at gen's own startup margin, not at 0s -- gen.ExperimentError
# already rejects an earlier schedule entry (no host_apps readiness hook), so
# reusing its constant keeps this module from re-deriving (and potentially
# drifting from) that floor.
RECORD_COUNT = 30
RECORD_INTERVAL_S = 2.0
RECORD_START_S = gen.MIN_FIRST_RECORD_MARGIN_S

# Probe generation (map.md "probe は真ん中の h2 から 10 秒ごとに出す").
PROBE_INTERVAL_S = 10.0
PROBE_START_S = 5.0

# Run length (map.md "run の長さは、生成が終わってから 60 秒の余裕を取って、
# およそ 120 秒"): last record at RECORD_START_S + (RECORD_COUNT-1)*RECORD_INTERVAL_S
# = 3 + 58 = 61s, so 120s leaves ~59s of margin after generation ends.
DURATION_S = 120

# Disconnect schedule (map.md "切断の起こし方"): one Bernoulli(p) draw per
# link per 5s slot, independent per link and slot.
#
# 2026-10-23: the disconnectable window stops LINK_EVENT_SAFETY_MARGIN_S
# before DURATION_S, not at it. Read via disaster.py (our config is
# autotest mode: `no_cli` + `results_json`): when `duration > 0` it does
# `time.sleep(duration)` then tears the scenario down immediately, with NO
# call to EventScheduler.wait_all() to drain the heap first (that call
# only happens in the `duration == 0` branch). The eval EventScheduler is
# started with `start_time=origin` -- the same `origin` the generation
# phase (one put, one warmup get) already ran against before that sleep
# was even reached -- so an event at `at == duration` fires at
# `origin + duration`, while teardown begins at
# `origin + <seed/warmup wall time> + duration`. The margin before this
# fix was that unmeasured seed/warmup wall time: real, but accidental and
# un-guaranteed (a slow warmup get could shrink or erase it). With p=0.4,
# P(at least one of the 4 links still down in the final slot) =
# 1-(1-p)**4 ~= 0.87, so most p=0.4 runs would lean entirely on that
# accident to avoid a restoring link_up racing teardown. This constant
# replaces the accident with an explicit, guaranteed 5s: leaving the last
# slot's width always-up means that by the time sleep(duration) returns,
# the event thread has had at least this long to finish handling
# anything scheduled before it.
DISCONNECT_SLOT_S = 5.0
LINK_EVENT_SAFETY_MARGIN_S = 5.0
DISCONNECT_WINDOW_S = DURATION_S - LINK_EVENT_SAFETY_MARGIN_S
assert DISCONNECT_WINDOW_S % DISCONNECT_SLOT_S == 0, "DISCONNECT_WINDOW_S must be a whole number of slots"
NUM_SLOTS = int(DISCONNECT_WINDOW_S / DISCONNECT_SLOT_S)

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
    record/probe stream (ticket 03's ".../zone1/landslide/video" at p=0.2
    vs ".../zone5/fire/sensor" at p=0.3 for seed=7's h0 first record),
    instead of the ticket's "乱数は seed だけから決まり". `p` is no longer
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
    condition. This is the one common entry point `build_experiment` calls
    before any draw, so every caller (the CLI and every direct test call)
    gets the same check.
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


def _record_schedule(host: int, seed: int) -> list[dict]:
    """Build one origin host's 30-record schedule, attrs drawn uniformly from the seed.

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
    while at_s < DURATION_S:
        entries.append({"at_s": at_s, "zone": rng.choice(zones)})
        at_s += PROBE_INTERVAL_S
    return entries


def _link_events(seed: int, p: float) -> list[dict]:
    """Build every link's link_down/link_up events for this (seed, p).

    One Bernoulli(p) draw per link per DISCONNECT_SLOT_S slot (independent
    per link -- its own random.Random stream -- and per slot), with
    consecutive down slots on the same link merged into a single
    link_down/link_up pair. A sentinel "up" slot appended past the last
    real slot closes any run still open at NUM_SLOTS, so every down
    interval is restored by `at <= DISCONNECT_WINDOW_S` -- never left
    dangling, and always with LINK_EVENT_SAFETY_MARGIN_S to spare before
    DURATION_S (see that constant's comment for why the margin exists).

    2026-10-23 (review r1, finding 3): each link's rng stream is seeded
    from (seed, purpose) only, never p (see `_derive_seed`) -- so the same
    seed draws the exact same sequence of `rng.random()` values at every
    p, and only the `< p` comparison below changes with p. This is what
    lets a seed=7 comparison across p values isolate the disconnect rate
    alone, instead of also redrawing an unrelated down/up pattern.
    """
    events: list[dict] = []
    for a, b in LINKS:
        rng = _rng_for(seed, f"link:{a}:{b}")
        down_slots = [rng.random() < p for _ in range(NUM_SLOTS)]
        run_start = None
        for i in range(NUM_SLOTS + 1):
            is_down = down_slots[i] if i < NUM_SLOTS else False
            if is_down and run_start is None:
                run_start = i
            elif not is_down and run_start is not None:
                events.append({"at": run_start * DISCONNECT_SLOT_S, "type": "link_down", "nodes": [a, b]})
                events.append({"at": i * DISCONNECT_SLOT_S, "type": "link_up", "nodes": [a, b]})
                run_start = None
    # Sorted so the generated YAML reads chronologically (map.md "引いた予定は
    # 生成した YAML にそのまま残るので、あとから見直せる") -- ReCefore's own
    # scheduler (runtime/scheduler.py) heap-sorts events at run time
    # regardless, so this ordering is for a human reader, not correctness.
    events.sort(key=lambda e: (e["at"], e["nodes"][0], e["nodes"][1], e["type"]))
    return events


def build_experiment(p: float, seed: int, stage: str, outdir: str) -> dict:
    """Build the one `semindex.gen` experiment description this (p, seed, stage) run needs.

    `stage` and `outdir` only ever affect bookkeeping fields (`name`,
    `group`, `output_dir`) -- never a random draw -- so that "same (p,
    seed) => byte-identical output" holds regardless of which stage label
    or output directory a caller passes.
    """
    _validate_p(p)
    name = f"{stage}_p{_format_p(p)}_seed{seed}"
    schedule = {str(host): _record_schedule(host, seed) for host in RECORD_ORIGIN_HOSTS}
    return {
        "name": name,
        "num": RECEFORE_NUM,
        "seed": seed,
        "group": name,
        "hosts": HOSTS,
        "links": [list(pair) for pair in LINKS],
        "duration": DURATION_S,
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
        "events": _link_events(seed, p),
    }


def generate_run(p: float, seed: int, stage: str, outdir: str) -> list[str]:
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

    Returns every path written, experiment description first.
    """
    raw = build_experiment(p, seed, stage, outdir)
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
    parser.add_argument("--p", type=float, required=True, help="per-link, per-slot disconnect probability")
    parser.add_argument("--seed", type=int, required=True, help="seed for every random draw in this run")
    parser.add_argument("--stage", type=str, required=True, help="stage label, used only in the run's name/group")
    parser.add_argument("--out", type=str, required=True, dest="outdir", help="directory to write the run into")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: generate one (p, seed, stage) run into --out."""
    args = parse_args(argv)
    try:
        written = generate_run(args.p, args.seed, args.stage, args.outdir)
    except gen.ExperimentError as exc:
        print(f"semindex.evalgen: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
