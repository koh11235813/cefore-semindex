"""semindex.agent: the single-threaded Sync agent process.

Wires a Store, SyncEngine and CefpycoTransport together from a JSON
config, runs the receive/dispatch/tick loop, fires scheduled record
creations and probes relative to a scenario's t0, and writes
hN/semindex/events.jsonl. Run as:

    python -m semindex.agent --config <path.json> \
        [--node ...] [--t0-ms ...] [--cefore-conf-dir ...] [--state-dir ...]

The four CLI overrides exist because ReCefore's host_apps only expands
{node}/{node_dir}/{cefore_conf_dir}/{t0_ms} inside a host app's own argv,
never inside a file that argv happens to point at -- so a config file
written ahead of a scenario cannot itself contain the one value (t0_ms)
that is only known once the scenario starts. The static part of a node's
config (group, neighbors, schedule, probes, fault, timing overrides)
lives in the JSON file; these four runtime values are merged into it
before the single validate_config() pass that checks the merged result,
so a config given entirely on the command line and one given entirely in
the file are validated identically.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from typing import Callable

from semindex import schema
from semindex.cefpyco_transport import DEFAULT_RECEIVE_TIMEOUT_MS, CefpycoHandle, CefpycoTransport
from semindex.engine import SyncConfig, SyncEngine
from semindex.store import DEFAULT_STORE_SYNCHRONOUS, STORE_SYNCHRONOUS_MODES, Store

# cefpyco_smoke's own default Interest lifetime for a chunk fetch
# (/home/lab_shared/cefpyco/semindex_smoke/chunk_fetch.py); reused here as
# this agent's default for a probe's content-body retrieval.
DEFAULT_CHUNK_LIFETIME_MS = 2000


class ConfigError(ValueError):
    """A config file (or a CLI override merged into it) failed validation."""


@dataclass(frozen=True)
class ScheduledRecord:
    """One scheduled own-record creation: relative seconds from t0, plus its content and attrs."""

    at_s: float
    content_name: str
    zone: str
    disaster: str
    media: str


@dataclass(frozen=True)
class ScheduledProbe:
    """One scheduled probe: relative seconds from t0, plus an attrs filter (None = any value)."""

    at_s: float
    zone: str | None
    disaster: str | None
    media: str | None


@dataclass(frozen=True)
class AgentConfig:
    """Everything one agent process needs, parsed and validated from its JSON config."""

    node: str
    group: str
    neighbors: tuple[str, ...]
    cefore_conf_dir: str
    state_dir: str
    t0_ms: int
    schedule: tuple[ScheduledRecord, ...]
    probes: tuple[ScheduledProbe, ...]
    fault_drop_rec: dict[tuple[str, int], int]
    sync_overrides: dict[str, float | int | tuple[float, ...]]
    receive_timeout_ms: int
    chunk_lifetime_ms: int
    # t0-relative seconds; None means never stop early (run until SIGTERM/SIGINT).
    stop_at_s: float | None
    # t0-relative seconds, strictly increasing; empty means no mid-run snapshots.
    counters_snapshot_at_s: tuple[float, ...]
    # The run's seed, or None if the config gave none. Only the
    # state_poll_interval_choices_s policy draws from it (see _poll_choice_rng).
    seed: int | None
    # The index.db `PRAGMA synchronous` mode; NORMAL unless the config names
    # another (see Store.__init__ and DEFAULT_STORE_SYNCHRONOUS).
    store_synchronous: str = DEFAULT_STORE_SYNCHRONOUS


# The SyncConfig fields an agent config may override; map.md's "時間・サ
# イズの初期値" table is SyncConfig's own defaults, so a key left out of
# the config simply keeps that default. `bool` is a subtype of `int` in
# Python, so _check_numeric rejects it explicitly rather than silently
# accepting True/False as 1/0.
_SYNC_OVERRIDE_KEYS: dict[str, type] = {
    "state_poll_interval_s": float,
    "state_poll_jitter_frac": float,
    "interest_lifetime_ms": int,
    "fetch_backoff_initial_s": float,
    "fetch_backoff_multiplier": float,
    "fetch_backoff_max_s": float,
    "max_in_flight_per_neighbor": int,
    "max_missing_per_origin": int,
    "state_data_expiry_s": float,
    "state_data_cache_time_s": float,
    "record_data_expiry_s": float,
    "record_data_cache_time_s": float,
}

# Agent/transport-level numeric overrides: not SyncEngine fields (kept out
# of _SYNC_OVERRIDE_KEYS for that reason), validated by the same rule.
_AGENT_NUMERIC_KEYS: dict[str, type] = {
    "receive_timeout_ms": int,
    "chunk_lifetime_ms": int,
}

_REQUIRED_KEYS = (
    "node",
    "group",
    "neighbors",
    "cefore_conf_dir",
    "state_dir",
    "t0_ms",
    "schedule",
    "probes",
)
_ALLOWED_KEYS = (
    frozenset(_REQUIRED_KEYS)
    | {
        "fault",
        "stop_at_s",
        "counters_snapshot_at_s",
        "seed",
        "state_poll_interval_choices_s",
        "state_poll_choice_jitter_frac",
        # A Store setting, not a SyncConfig field, so it is not in
        # _SYNC_OVERRIDE_KEYS; nor is it numeric, so not in _AGENT_NUMERIC_KEYS.
        "store_synchronous",
    }
    | set(_SYNC_OVERRIDE_KEYS)
    | set(_AGENT_NUMERIC_KEYS)
)

_SCHEDULE_ENTRY_REQUIRED = ("at_s", "content_name", "zone", "disaster", "media")
_SCHEDULE_ENTRY_KEYS = frozenset(_SCHEDULE_ENTRY_REQUIRED)

_PROBE_ENTRY_REQUIRED = ("at_s",)
_PROBE_ENTRY_OPTIONAL = ("zone", "disaster", "media")
_PROBE_ENTRY_KEYS = frozenset(_PROBE_ENTRY_REQUIRED) | frozenset(_PROBE_ENTRY_OPTIONAL)

_FAULT_DROP_REC_ENTRY_KEYS = frozenset({"origin", "seq", "times"})


# 2026-10-22 fix: receive_timeout_ms/chunk_lifetime_ms/interest_lifetime_ms
# used to only be checked >= 0, so 0 passed validation -- receive_timeout_ms:
# 0 spins the agent's poll() loop at full CPU (every receive() returns
# instantly), and chunk_lifetime_ms/interest_lifetime_ms: 0 makes every
# fetch/sync Interest time out the instant it is sent, before any reply
# could ever arrive. All three must be strictly positive; every other
# numeric override (e.g. state_poll_interval_s) may still legitimately be 0.
_POSITIVE_NUMERIC_KEYS = frozenset({"receive_timeout_ms", "chunk_lifetime_ms", "interest_lifetime_ms"})


def _check_numeric(key: str, value, kind: type, *, positive: bool = False) -> None:
    """Validate one numeric config value: the right numeric kind, never a bool, and
    either >= 0 (the general case) or > 0 when `positive` is set -- see
    _POSITIVE_NUMERIC_KEYS for which keys need the stricter bound and why.
    """
    if kind is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
        label = "an integer"
    else:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        label = "a number"
    if not ok:
        raise ConfigError(f"{key} must be {label}")
    if positive:
        if value <= 0:
            raise ConfigError(f"{key} must be > 0")
    elif value < 0:
        raise ConfigError(f"{key} must be >= 0")


def _check_name_component(value, label: str) -> str:
    """Validate `value` as a non-empty string with no '/' -- names.validate_component's own
    rule, reused here so a bad node/group/neighbor name is rejected with a config-level
    diagnostic before any wire name is ever built from it.
    """
    if not isinstance(value, str) or not value or "/" in value:
        raise ConfigError(f"{label} must be a non-empty string without '/'")
    return value


def _check_non_empty_string(value, label: str) -> str:
    """Validate `value` as a non-empty string."""
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty string")
    return value


def _check_attr_value(value, label: str, vocabulary: frozenset) -> str:
    """Validate `value` as a non-empty string drawn from `vocabulary` (schema.py's own fixed attrs vocabulary)."""
    if not isinstance(value, str) or value not in vocabulary:
        raise ConfigError(f"{label} must be one of {sorted(vocabulary)}")
    return value


def _validate_schedule_entry(entry, idx: int) -> ScheduledRecord:
    """Validate one schedule[] entry and build the ScheduledRecord it describes."""
    prefix = f"schedule[{idx}]"
    if not isinstance(entry, dict):
        raise ConfigError(f"{prefix} must be an object")
    unknown = sorted(set(entry) - _SCHEDULE_ENTRY_KEYS)
    if unknown:
        raise ConfigError(f"{prefix} has unknown key: {unknown[0]!r}")
    missing = [k for k in _SCHEDULE_ENTRY_REQUIRED if k not in entry]
    if missing:
        raise ConfigError(f"{prefix} missing required key: {missing[0]!r}")
    _check_numeric(f"{prefix}.at_s", entry["at_s"], float)
    content_name = _check_non_empty_string(entry["content_name"], f"{prefix}.content_name")
    zone = _check_attr_value(entry["zone"], f"{prefix}.zone", schema.ZONES)
    disaster = _check_attr_value(entry["disaster"], f"{prefix}.disaster", schema.DISASTERS)
    media = _check_attr_value(entry["media"], f"{prefix}.media", schema.MEDIA)
    return ScheduledRecord(
        at_s=float(entry["at_s"]), content_name=content_name, zone=zone, disaster=disaster, media=media
    )


def _validate_probe_entry(entry, idx: int) -> ScheduledProbe:
    """Validate one probes[] entry and build the ScheduledProbe it describes."""
    prefix = f"probes[{idx}]"
    if not isinstance(entry, dict):
        raise ConfigError(f"{prefix} must be an object")
    unknown = sorted(set(entry) - _PROBE_ENTRY_KEYS)
    if unknown:
        raise ConfigError(f"{prefix} has unknown key: {unknown[0]!r}")
    missing = [k for k in _PROBE_ENTRY_REQUIRED if k not in entry]
    if missing:
        raise ConfigError(f"{prefix} missing required key: {missing[0]!r}")
    _check_numeric(f"{prefix}.at_s", entry["at_s"], float)
    vocab_by_field = {"zone": schema.ZONES, "disaster": schema.DISASTERS, "media": schema.MEDIA}
    values: dict[str, str | None] = {}
    for field_name, vocabulary in vocab_by_field.items():
        if field_name in entry:
            values[field_name] = _check_attr_value(entry[field_name], f"{prefix}.{field_name}", vocabulary)
        else:
            values[field_name] = None
    return ScheduledProbe(at_s=float(entry["at_s"]), **values)


def _validate_fault(raw: dict) -> dict[tuple[str, int], int]:
    """Validate the optional `fault` block and build the (origin, seq) -> times mapping
    SyncConfig.fault_drop_rec expects. Absent entirely, this is just {} -- fault is
    test-only and must never appear in a production config (map.md).
    """
    if "fault" not in raw:
        return {}
    fault = raw["fault"]
    if not isinstance(fault, dict):
        raise ConfigError("fault must be an object")
    unknown = sorted(set(fault) - {"drop_rec"})
    if unknown:
        raise ConfigError(f"fault has unknown key: {unknown[0]!r}")
    drop_rec = fault.get("drop_rec", [])
    if not isinstance(drop_rec, list):
        raise ConfigError("fault.drop_rec must be a list")
    result: dict[tuple[str, int], int] = {}
    for idx, entry in enumerate(drop_rec):
        prefix = f"fault.drop_rec[{idx}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{prefix} must be an object")
        unknown_entry = sorted(set(entry) - _FAULT_DROP_REC_ENTRY_KEYS)
        if unknown_entry:
            raise ConfigError(f"{prefix} has unknown key: {unknown_entry[0]!r}")
        missing_entry = [k for k in ("origin", "seq", "times") if k not in entry]
        if missing_entry:
            raise ConfigError(f"{prefix} missing required key: {missing_entry[0]!r}")
        origin = _check_name_component(entry["origin"], f"{prefix}.origin")
        seq = entry["seq"]
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
            raise ConfigError(f"{prefix}.seq must be an integer >= 1")
        times = entry["times"]
        if not isinstance(times, int) or isinstance(times, bool) or times < 1:
            raise ConfigError(f"{prefix}.times must be an integer >= 1")
        result[(origin, seq)] = times
    return result


def _validate_counters_snapshot_at_s(raw: dict) -> tuple[float, ...]:
    """Validate the optional top-level `counters_snapshot_at_s` key: a strictly
    increasing list of t0-relative seconds (each >= 0), one counters_snapshot
    event per entry (Agent._drain_counters_snapshots). Absent entirely means
    none are scheduled -- the same "a key left out just keeps doing nothing new"
    convention as every other optional config key here.
    """
    if "counters_snapshot_at_s" not in raw:
        return ()
    raw_list = raw["counters_snapshot_at_s"]
    if not isinstance(raw_list, list):
        raise ConfigError("counters_snapshot_at_s must be a list")
    result: list[float] = []
    for idx, value in enumerate(raw_list):
        _check_numeric(f"counters_snapshot_at_s[{idx}]", value, float)
        if result and value <= result[-1]:
            raise ConfigError("counters_snapshot_at_s must be strictly increasing")
        result.append(float(value))
    return tuple(result)


def _validate_poll_interval_choices(raw) -> tuple[float, ...] | None:
    """Validate the optional `state_poll_interval_choices_s` list (stage-2 map "行動と確率").

    Duplicates are rejected, not collapsed: the engine records action_prob as
    1/len(choices), which is only the true probability of the chosen value when
    every value appears once. Combining it with state_poll_interval_s or
    state_poll_jitter_frac is rejected because the choices policy ignores both,
    and a config that sets them would silently not mean what it says.
    """
    if "state_poll_interval_choices_s" not in raw:
        return None
    raw_list = raw["state_poll_interval_choices_s"]
    if not isinstance(raw_list, list) or not raw_list:
        raise ConfigError("state_poll_interval_choices_s must be a non-empty list")
    for idx, value in enumerate(raw_list):
        label = f"state_poll_interval_choices_s[{idx}]"
        _check_numeric(label, value, float, positive=True)
        # 2026-10-26 fix: NaN and +Infinity (both accepted by Python's json)
        # passed the > 0 check, and a non-finite interval makes the next poll's
        # due time unreachable, so that neighbor was never polled again.
        if not math.isfinite(value):
            raise ConfigError(f"{label} must be finite")
    if len(set(raw_list)) != len(raw_list):
        raise ConfigError("state_poll_interval_choices_s must not contain duplicates")
    for key in ("state_poll_interval_s", "state_poll_jitter_frac"):
        if key in raw:
            raise ConfigError(f"state_poll_interval_choices_s cannot be combined with {key}")
    return tuple(float(value) for value in raw_list)


def _validate_choice_jitter_frac(raw, *, have_choices: bool) -> float | None:
    """Validate the optional `state_poll_choice_jitter_frac` (stage-2 map "選択肢の間隔のゆらぎ").

    Only meaningful with choices, so it is refused without them rather than
    silently ignored. Must be finite and in [0, 1): 1 or more would let the
    applied interval reach 0 or go negative. Absent means SyncConfig's default.
    """
    if "state_poll_choice_jitter_frac" not in raw:
        return None
    if not have_choices:
        raise ConfigError("state_poll_choice_jitter_frac requires state_poll_interval_choices_s")
    value = raw["state_poll_choice_jitter_frac"]
    _check_numeric("state_poll_choice_jitter_frac", value, float)
    if not math.isfinite(value):
        raise ConfigError("state_poll_choice_jitter_frac must be finite")
    if value >= 1:
        raise ConfigError("state_poll_choice_jitter_frac must be < 1")
    return float(value)


def _validate_seed(raw, *, required: bool) -> int | None:
    """Validate the optional run `seed`; `required` when a policy actually draws from it."""
    if "seed" not in raw:
        if required:
            raise ConfigError("state_poll_interval_choices_s requires seed")
        return None
    value = raw["seed"]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError("seed must be an integer")
    return value


def _validate_store_synchronous(raw) -> str:
    """Validate the optional `store_synchronous` (index.db's `PRAGMA synchronous` mode), or NORMAL if absent.

    Exact upper-case mode names only: SQLite itself would take "normal" or 1
    as well, but then one setting could be written several ways in configs
    and the run records made from them. isinstance rejects a bool before the
    membership test could ever see it. An explicit null is refused like any
    other non-name: absent is the one way to ask for the default.

    2026-10-29: absent used to mean "issue no PRAGMA" (SQLite's FULL); it now
    means NORMAL (see DEFAULT_STORE_SYNCHRONOUS for why), so configs written
    without the key -- every stage-1 and earlier stage-2 one -- now run NORMAL.
    """
    if "store_synchronous" not in raw:
        return DEFAULT_STORE_SYNCHRONOUS
    value = raw["store_synchronous"]
    if not isinstance(value, str) or value not in STORE_SYNCHRONOUS_MODES:
        raise ConfigError(f"store_synchronous must be one of {list(STORE_SYNCHRONOUS_MODES)}")
    return value


def validate_config(raw) -> AgentConfig:
    """Validate a merged config dict (JSON file plus any CLI overrides) and build the AgentConfig it describes.

    Checks run in a fixed order -- object shape, unknown keys, missing
    required keys, then each field in a fixed order -- so the same
    malformed input always raises the exact same ConfigError message.
    """
    if not isinstance(raw, dict):
        raise ConfigError("config must be a JSON object")
    unknown = sorted(set(raw) - _ALLOWED_KEYS)
    if unknown:
        raise ConfigError(f"unknown config key: {unknown[0]!r}")
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise ConfigError(f"missing required config key: {missing[0]!r}")

    node = _check_name_component(raw["node"], "node")
    group = _check_name_component(raw["group"], "group")

    neighbors_raw = raw["neighbors"]
    if not isinstance(neighbors_raw, list):
        raise ConfigError("neighbors must be a list of strings")
    neighbors = tuple(
        _check_name_component(n, f"neighbors[{idx}]") for idx, n in enumerate(neighbors_raw)
    )
    # 2026-10-22 fix: neither check used to run, so a node could list
    # itself as a neighbor (SyncEngine would then sync state/records with
    # itself) or list the same neighbor twice (double-counting its
    # in-flight fetch slots against max_in_flight_per_neighbor).
    if node in neighbors:
        raise ConfigError("neighbors must not contain the node itself")
    if len(neighbors) != len(set(neighbors)):
        raise ConfigError("neighbors must not contain duplicates")

    cefore_conf_dir = _check_non_empty_string(raw["cefore_conf_dir"], "cefore_conf_dir")
    state_dir = _check_non_empty_string(raw["state_dir"], "state_dir")

    _check_numeric("t0_ms", raw["t0_ms"], int)
    t0_ms = raw["t0_ms"]

    schedule_raw = raw["schedule"]
    if not isinstance(schedule_raw, list):
        raise ConfigError("schedule must be a list")
    schedule = tuple(_validate_schedule_entry(entry, idx) for idx, entry in enumerate(schedule_raw))

    probes_raw = raw["probes"]
    if not isinstance(probes_raw, list):
        raise ConfigError("probes must be a list")
    probes = tuple(_validate_probe_entry(entry, idx) for idx, entry in enumerate(probes_raw))

    fault_drop_rec = _validate_fault(raw)

    # stop_at_s has no sensible non-None default (unlike every _AGENT_NUMERIC_KEYS
    # entry, which already defaults to a real timeout): absent must mean "never
    # stop early", not 0 seconds, so it is read and checked directly rather than
    # through _AGENT_NUMERIC_KEYS' raw.get()-with-a-default pattern.
    stop_at_s_raw = raw.get("stop_at_s")
    if stop_at_s_raw is not None:
        _check_numeric("stop_at_s", stop_at_s_raw, float)
    stop_at_s = float(stop_at_s_raw) if stop_at_s_raw is not None else None

    counters_snapshot_at_s = _validate_counters_snapshot_at_s(raw)

    sync_overrides: dict[str, float | int] = {}
    for key, kind in _SYNC_OVERRIDE_KEYS.items():
        if key in raw:
            _check_numeric(key, raw[key], kind, positive=key in _POSITIVE_NUMERIC_KEYS)
            sync_overrides[key] = raw[key]

    for key, kind in _AGENT_NUMERIC_KEYS.items():
        if key in raw:
            _check_numeric(key, raw[key], kind, positive=key in _POSITIVE_NUMERIC_KEYS)

    poll_interval_choices = _validate_poll_interval_choices(raw)
    if poll_interval_choices is not None:
        sync_overrides["state_poll_interval_choices_s"] = poll_interval_choices
    choice_jitter_frac = _validate_choice_jitter_frac(raw, have_choices=poll_interval_choices is not None)
    if choice_jitter_frac is not None:
        sync_overrides["state_poll_choice_jitter_frac"] = choice_jitter_frac
    seed = _validate_seed(raw, required=poll_interval_choices is not None)
    store_synchronous = _validate_store_synchronous(raw)

    return AgentConfig(
        node=node,
        group=group,
        neighbors=neighbors,
        cefore_conf_dir=cefore_conf_dir,
        state_dir=state_dir,
        t0_ms=t0_ms,
        schedule=schedule,
        probes=probes,
        fault_drop_rec=fault_drop_rec,
        sync_overrides=sync_overrides,
        receive_timeout_ms=raw.get("receive_timeout_ms", DEFAULT_RECEIVE_TIMEOUT_MS),
        chunk_lifetime_ms=raw.get("chunk_lifetime_ms", DEFAULT_CHUNK_LIFETIME_MS),
        stop_at_s=stop_at_s,
        counters_snapshot_at_s=counters_snapshot_at_s,
        seed=seed,
        store_synchronous=store_synchronous,
    )


def load_config(
    path: str,
    *,
    node: str | None = None,
    t0_ms: int | None = None,
    cefore_conf_dir: str | None = None,
    state_dir: str | None = None,
) -> AgentConfig:
    """Read the JSON config at `path`, merge in any given overrides, and validate the result.

    See the module docstring for why node/t0_ms/cefore_conf_dir/state_dir
    are accepted as overrides rather than required to already be in the
    file.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ConfigError("config must be a JSON object")
    merged = dict(raw)
    if node is not None:
        merged["node"] = node
    if t0_ms is not None:
        merged["t0_ms"] = t0_ms
    if cefore_conf_dir is not None:
        merged["cefore_conf_dir"] = cefore_conf_dir
    if state_dir is not None:
        merged["state_dir"] = state_dir
    return validate_config(merged)


def _poll_jitter_rng(node: str) -> Callable[[], float]:
    """Build the [0, 1) jitter source for one node's state-poll timing (map.md "2 秒 ±20%").

    Seeded from the node's own name (random.Random(str) hashes it with
    SHA-512 internally, so this is stable across processes and
    independent of PYTHONHASHSEED): two agents in the same run draw
    different jitter sequences and so poll on different offsets from the
    2s grid, while the same node name always reproduces the exact same
    sequence from one run to the next.
    """
    return random.Random(node).random


def _poll_choice_rng(seed: int, node: str) -> Callable[[], float]:
    """Build the [0, 1) source one node draws its state-poll interval choices from
    (stage-2 map "行動と確率").

    Seeded from the run seed *and* the node name, unlike _poll_jitter_rng's
    node-only seed: stage 1's jitter repeated the exact same sequence in every
    run, so seeds never varied the poll timing. Here a different seed draws a
    different action sequence (more coverage of each choice across runs), while
    the same (seed, node) still reproduces it exactly. The "state_poll_interval"
    prefix keeps this stream apart from any other draw ever seeded from the same
    pair; random.Random(str) hashes it with SHA-512, so it is stable across
    processes and independent of PYTHONHASHSEED.
    """
    return random.Random(f"state_poll_interval:{seed}:{node}").random


def _poll_choice_jitter_rng(seed: int, node: str) -> Callable[[], float]:
    """Build the [0, 1) source for one node's choices-mode interval jitter.

    Seeded from (seed, node) like _poll_choice_rng but under its own prefix, so
    it is a separate stream: changing j, or the number of choices, never shifts
    which value the choice rng picks.
    """
    return random.Random(f"state_poll_interval_jitter:{seed}:{node}").random


def build_sync_config(config: AgentConfig) -> SyncConfig:
    """Build the SyncEngine's own SyncConfig from an AgentConfig, applying its timing overrides."""
    return SyncConfig(
        group=config.group,
        node=config.node,
        neighbors=config.neighbors,
        fault_drop_rec=dict(config.fault_drop_rec),
        **config.sync_overrides,
    )


class EventWriter:
    """Appends one JSON object per line to events.jsonl, flushing after every write.

    ReCefore's host_apps contract gives a host app SIGTERM, then SIGKILL
    2s later -- every event this process has already decided to record
    must be durable well before that, so this flushes on every write
    rather than relying on buffered I/O to catch up eventually.
    """

    def __init__(self, path: str, *, clock: Callable[[], float] = time.time) -> None:
        """Open `path` for appending; `clock` stamps each event with the current wall-clock ms."""
        self._file = open(path, "a", encoding="utf-8")
        self._clock = clock

    def write(self, event: dict) -> None:
        """Append `event` as one JSON line, stamping it with the current wall-clock epoch ms
        only when it does not already carry its own `ts_ms`.

        `{"ts_ms": ..., **event}` puts the write-time stamp first so any
        `ts_ms` already in `event` overrides it in the merged dict -- never
        the other way around. This is how counters_snapshot's own
        captured-ahead-of-write ts_ms (agent.py's _capture_due_snapshots) has
        always survived to the written line, and is also what now lets
        SyncEngine's own queued events (record_created/record_stored/
        decision/state_rx/fetch_timeout/conflict -- every engine._emit call)
        carry the processing-time ts_ms the engine stamped them with at
        creation, rather than the time this write() call happens to run at,
        which can be seconds later for an event queued during a probe's
        synchronous content fetch (see SyncEngine._emit_decision's own
        comment). An agent-level event with no `now` of its own to draw on
        (agent_start, probe_discovered, probe_retrieved, the final
        `counters` line) still gets today's stamp-at-write-time default.
        """
        stamped = {"ts_ms": round(self._clock() * 1000), **event}
        self._file.write(json.dumps(stamped, sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file."""
        self._file.close()


class Agent:
    """The single-threaded Sync agent loop.

    Each step() receives/dispatches at most one cefpyco packet, ticks the
    engine's own due state-polls and fetch plans, fires any schedule/probe
    entries whose time has come, and flushes every event produced. run()
    repeats step() until request_stop() has been called (from a signal
    handler, or directly in a test), then shuts down.
    """

    def __init__(
        self,
        config: AgentConfig,
        engine: SyncEngine,
        store: Store,
        transport: CefpycoTransport,
        events: EventWriter,
    ) -> None:
        """Wire the four already-constructed pieces together; schedule/probe entries are consumed in time order.

        2026-10-22 fix: a restart (SIGTERM, then relaunched with the same
        config and state_dir -- ReCefore's host_apps gives no other way to
        stop/start a node) used to re-fire every schedule/probe entry whose
        at_s had already come due in a prior life, duplicating that node's
        own records and re-emitting past probe_discovered/probe_retrieved
        events. Both fixes below skip exactly the entries a prior life
        already ran, using progress durably persisted in `store` (which
        outlives this process) rather than anything held only in memory.
        """
        self._config = config
        self._engine = engine
        self._store = store
        self._transport = transport
        self._events = events

        # 2026-10-23: records this life's own t0_ms directly, as the very
        # first thing this life ever writes -- a later measurement tool
        # (semindex.evalagg) used to have to INFER t0 from indirect
        # evidence (a scheduler event's timestamp, or a boundary like
        # stop_at_s/counters_snapshot_at_s firing a tick or so late)
        # because the value this process actually ran with never reached
        # events.jsonl literally. Written before anything else in this
        # life, restart included (a second life re-runs Agent.__init__
        # from scratch, so it gets its own agent_start line too), so a
        # reader never has to guess which life an agent_start belongs to.
        # Uses ts_ms's normal stamp-at-write-time default rather than a
        # pre-set one (contrast counters_snapshot's own queued write) --
        # this event has nothing to capture ahead of time, it is written
        # synchronously right here.
        self._events.write({"event": "agent_start", "node": config.node, "t0_ms": config.t0_ms})

        self._schedule = sorted(config.schedule, key=lambda e: e.at_s)
        self._probes = sorted(config.probes, key=lambda e: e.at_s)

        # Own records are created 1:1 with schedule entries, in at_s order
        # (_drain_schedule pops exactly one entry per create_record call,
        # and received records never add to this count: a neighbor can
        # never advertise a higher seq for this node's own origin than
        # this node itself already holds, since that neighbor only knows
        # of a seq because this node created and propagated it first --
        # so the tracker never plans a fetch for this node's own origin),
        # so store.get_progress(node).hi -- the count of this node's own
        # records already durably written -- is exactly how many of the
        # earliest (lowest at_s) schedule entries a prior life already
        # fired. A schedule shorter than that is not a timing question at
        # all, it is the config no longer matching the state_dir it was
        # paired with (e.g. the wrong config file after an edit), so this
        # fails at startup rather than silently running a truncated
        # schedule or re-deriving the wrong entries to skip.
        already_created = self._store.get_progress(self._config.node).hi
        if already_created > len(self._schedule):
            raise ConfigError(
                f"{self._config.node}: state_dir already holds {already_created} own "
                f"record(s) but schedule has only {len(self._schedule)} entries"
            )
        self._schedule = self._schedule[already_created:]

        # Probes have no per-origin progress counter to reuse (they are not
        # records), so the store keeps a dedicated cursor instead -- see
        # Store.get_probe_count/record_probe_executed. A persisted count
        # ahead of this life's probe list simply leaves no probes to run
        # (list slicing past the end yields []); that is a config/state
        # mismatch of the same shape as the schedule one above, but no
        # restart scenario in this baseline exercises it yet, so it is
        # left unenforced here rather than guessed at ahead of need.
        already_probed = self._store.get_probe_count()
        self._probes = self._probes[already_probed:]

        self._stop_at_s = config.stop_at_s

        # counters_snapshot_at_s has no store-backed cursor like schedule/probes
        # above: CefpycoTransport's byte counters are process-local and start at
        # 0 every life, so re-emitting a "past" entry on restart would report
        # this life's own fresh-from-zero counters under a timestamp that
        # belongs to a different life entirely. An entry already past --
        # strictly before this life's own start, so one landing exactly at this
        # instant still gets its chance on the very first step -- is dropped
        # right here, against the transport's own clock, rather than carried
        # into the per-step drain loop where elapsed time has already
        # overtaken it.
        init_elapsed = self._elapsed_s(self._transport.now())
        self._counters_snapshot_at_s = [t for t in config.counters_snapshot_at_s if t >= init_elapsed]

        self._stop_requested = False

        # Captured counters_snapshot events awaiting their actual write --
        # see _capture_due_snapshots/_emit_pending_snapshots for why the two
        # are split.
        self._pending_snapshots: list[dict] = []

    def request_stop(self) -> None:
        """Ask the loop to exit after its current step; safe to call from a signal handler."""
        self._stop_requested = True

    def _on_receive_boundary(self) -> bool:
        """Run once every time a blocking receive() call returns control to this
        agent -- step()'s own pre-poll and post-poll calls, and (wired as
        the transport's should_stop) CefpycoTransport.poll() and a probe's
        fetch_content wait loop alike -- so none of these call sites can
        act on whatever that receive() returned without first re-checking
        both measurement boundaries against the clock reading *at this
        exact point*, not the reading taken before that receive() started
        blocking.

        Captures (does not yet write -- see _emit_pending_snapshots) any
        counters_snapshot_at_s entry now due, using the counters as they
        stand right here, then reports whether stop_at_s has also been
        reached, setting the stop flag if so.

        2026-10-23 fix: stop_at_s and counters_snapshot_at_s used to each be
        checked/captured only once per step(), against the clock reading
        taken *before* that step's own poll() call. A blocking receive (up
        to receive_timeout_ms), or a probe's fetch_content (up to
        chunk_lifetime_ms per chunk), could each cross either boundary
        while blocked, and the step/probe would still dispatch/tick/send,
        or a snapshot would not be captured until a later step/never at all
        if that step turned out to be the run's last one, by which point
        the counters it would report had already moved on. Calling this
        same check at every receive boundary bounds the remaining error by
        the one packet/chunk a single receive() call can return, not by
        receive_timeout_ms or chunk_lifetime_ms -- see step()'s own
        docstring for the exact remaining bound.

        Capture is split from the actual event write on purpose: a probe's
        fetch_content runs synchronously inside _run_probe, strictly
        between that probe's own probe_discovered and probe_retrieved
        writes, with nothing else from this node logged in between by
        design -- check.py's own retrieval-pairing check (see its own
        comment on retrieved_after) depends on that adjacency. Writing a
        counters_snapshot line from inside that wait would separate them,
        so this only ever queues it in self._pending_snapshots; step() (and,
        as a last-resort safety net, shutdown()) is what actually calls
        _emit_pending_snapshots, always from outside any probe's own call
        stack.
        """
        now = self._transport.now()
        self._capture_due_snapshots(now)
        if self._stop_at_s is not None and self._elapsed_s(now) >= self._stop_at_s:
            self.request_stop()
        return self._stop_requested

    def _elapsed_s(self, now: float) -> float:
        """Seconds since this agent's t0, given `now` in the same wall-clock seconds as t0_ms."""
        return now - self._config.t0_ms / 1000.0

    def _drain_schedule(self, now: float) -> None:
        """Create every scheduled record whose at_s has come due."""
        elapsed = self._elapsed_s(now)
        while self._schedule and self._schedule[0].at_s <= elapsed:
            entry = self._schedule.pop(0)
            self._engine.create_record(
                content_name=entry.content_name,
                zone=entry.zone,
                disaster=entry.disaster,
                media=entry.media,
                observed_at=round(now * 1000),
            )

    def _drain_probes(self, now: float) -> None:
        """Run every scheduled probe whose at_s has come due.

        2026-10-23 fix: stops firing further due probes, leaving them queued,
        the instant request_stop() has been set -- a probe's own
        fetch_content can block long enough (up to chunk_lifetime_ms per
        chunk) for _on_receive_boundary to notice stop_at_s crossed mid-call,
        and a second probe already due at this same `now` must not then
        start a fresh fetch into a life that has already decided to stop.
        """
        elapsed = self._elapsed_s(now)
        while self._probes and self._probes[0].at_s <= elapsed and not self._stop_requested:
            entry = self._probes.pop(0)
            self._run_probe(entry)

    def _run_probe(self, probe: ScheduledProbe) -> None:
        """Query the local Semantic Index by `probe`'s attrs, then fetch the first match's content body.

        Discovery (probe_discovered) and retrieval (probe_retrieved) are
        recorded as separate events even when both happen, and a failed
        retrieval affects only this one probe: the loop continues either way.
        """
        matches = self._store.query_by_attrs(zone=probe.zone, disaster=probe.disaster, media=probe.media)
        self._events.write(
            {
                "event": "probe_discovered",
                "matches": [{"origin": m.origin, "seq": m.seq, "content_name": m.content_name} for m in matches],
            }
        )
        # A probe counts as executed once its probe_discovered line is
        # written (map.md's restart contract), regardless of whether a
        # match or a retrieval followed -- so this is recorded right here,
        # not after the retrieval below, and unconditionally on the early
        # `return` for no matches.
        self._store.record_probe_executed()
        if not matches:
            return
        target = matches[0]
        success, total_bytes = self._transport.fetch_content(
            target.content_name, chunk_lifetime_ms=self._config.chunk_lifetime_ms
        )
        self._events.write(
            {
                "event": "probe_retrieved",
                "origin": target.origin,
                "seq": target.seq,
                "content_name": target.content_name,
                "success": success,
                "bytes": total_bytes,
            }
        )

    def _capture_due_snapshots(self, now: float) -> None:
        """Capture (queue in self._pending_snapshots, do not yet write -- see
        _emit_pending_snapshots) one counters_snapshot entry for every due
        entry of config key counters_snapshot_at_s: this life's current byte
        counters, under the same four fields shutdown()'s own `counters`
        line uses, but a different event name on purpose -- analyze.py and
        check.py both key their own counters handling off the exact string
        "counters", so a written counters_snapshot line needs no
        special-casing in either to stay invisible to them.

        Called from _on_receive_boundary only -- step()'s own pre-poll and
        post-poll calls, and (via transport.should_stop) from inside
        poll()/fetch_content's own receive-wait loops -- so an entry due at
        exactly this instant always captures the counters as they stood at
        that exact receive boundary, never counters this same step's own
        post-receive tick/send (or an in-progress probe's own fetch) is
        about to add. `now` is captured into the queued dict's own ts_ms
        (EventWriter.write lets a pre-set key in the event win over its own
        stamp-at-write-time default), since the write itself may happen one
        or more receive boundaries later than this capture -- so a
        counters_snapshot line's ts_ms can be *earlier* than the line
        immediately before it in events.jsonl (e.g. a probe's own
        probe_retrieved, written after a snapshot captured mid-fetch but
        emitted only once that fetch returns): this is not a clock bug, it
        is the write catching up to a capture it was deliberately held back
        from making sooner. Neither check.py nor analyze.py sorts raw
        events.jsonl lines by ts_ms before pairing anything, so this never
        confuses either reader.
        """
        elapsed = self._elapsed_s(now)
        while self._counters_snapshot_at_s and self._counters_snapshot_at_s[0] <= elapsed:
            self._counters_snapshot_at_s.pop(0)
            self._pending_snapshots.append(
                {
                    "event": "counters_snapshot",
                    "ts_ms": round(now * 1000),
                    "bytes_sent": self._transport.bytes_sent,
                    "bytes_received": self._transport.bytes_received,
                    "control_bytes_sent": self._transport.control_bytes_sent,
                    "control_bytes_received": self._transport.control_bytes_received,
                }
            )

    def _emit_pending_snapshots(self) -> None:
        """Write every counters_snapshot captured (by _capture_due_snapshots)
        since the last call, in capture order, then clear the queue; a no-op
        when nothing is pending. Called by step() itself, from outside any
        probe's own call stack, so a written line never lands between that
        probe's probe_discovered and probe_retrieved (see
        _on_receive_boundary's own docstring); shutdown() also calls this,
        once, as a last-resort safety net in case a life ends between two of
        step()'s own calls to it.
        """
        for snapshot in self._pending_snapshots:
            self._events.write(snapshot)
        self._pending_snapshots = []

    def _drain_engine_events(self) -> None:
        """Flush every event the engine has queued since the last drain."""
        for event in self._engine.drain_events():
            self._events.write(event)

    def step(self) -> None:
        """One loop iteration: capture/emit any due counters_snapshot, stop if
        stop_at_s has been reached, else poll the transport, tick the
        engine, fire due schedule/probe entries, and flush events.

        stop_at_s is a measurement-window boundary, not a real SIGTERM: the
        step that first notices it has been reached must not send one more
        Interest or fire one more schedule/probe entry only to have the
        result thrown away. Unlike a real SIGTERM, which only asks the step
        already in progress to be the last one, nothing here is allowed to
        finish "in progress" past this boundary.

        2026-10-23 fix: stop_at_s and counters_snapshot_at_s used to each be
        checked/captured only once per step, against `now` read *before*
        poll() -- but poll() itself blocks for up to receive_timeout_ms,
        long enough to cross either on its own, and this step would still
        tick/send against the post-poll `now` (or a snapshot would not be
        captured until the *next* step, by which point this step's own
        post-receive send had already mutated the counters it reports, or
        never at all if this step turned out to be the run's last one) once
        control returned. _on_receive_boundary is called again right here,
        with the post-poll `now`, for exactly that reason. CefpycoTransport
        .poll() runs the same check itself (via transport.should_stop =
        Agent._on_receive_boundary) right after its own receive() returns,
        before dispatching anything, so a packet is never dispatched past
        the boundary either -- and fetch_content's wait loop does the same
        around each of its own receive() calls, so a probe can never block
        past stop_at_s for longer than one receive_timeout_ms either (see
        _on_receive_boundary's own docstring for why what it captures there
        isn't written until _emit_pending_snapshots runs, below, outside
        that probe's own call stack).

        The remaining bound: should_stop() is checked against the clock
        reading *after* receive() returns, not against when a packet
        physically arrived, so two single-step errors are possible, never
        more. (a) A packet that in truth arrived a moment before the
        deadline can still be dropped unprocessed, if receive() itself
        happened to return at or after it -- an in-window packet
        under-counted, not an out-of-window one leaking in. (b) When poll()
        is instead allowed to continue (its post-receive `now` still short
        of the deadline), on_tick below runs against that same `now` and may
        send -- modeled as landing exactly then, but a real send has no
        wall-clock cost here, so if `now` was a hair before the deadline the
        physical send could land a hair after it. Both are bounded by one
        receive()/dispatch/tick, never by the full receive_timeout_ms or
        chunk_lifetime_ms.

        This also gives restart its "already past" behavior for free, with no
        separate check: a relaunched life whose very first step already finds
        itself past stop_at_s takes this same early return before ever calling
        poll() or on_tick(), so it stops immediately -- see run()/shutdown()
        for where the final `counters` line actually gets written.

        An entry due at exactly stop_at_s is still captured on this, the
        last step, before that same step's early return skips everything
        else, and is emitted right there -- _on_receive_boundary captures
        before testing the deadline, not after.
        """
        if self._on_receive_boundary():
            self._emit_pending_snapshots()
            return
        self._transport.poll()
        if self._on_receive_boundary():
            self._emit_pending_snapshots()
            return
        now = self._transport.now()
        self._engine.on_tick(now)
        self._drain_schedule(now)
        self._drain_probes(now)
        self._emit_pending_snapshots()
        self._drain_engine_events()

    def run(self) -> None:
        """Loop step() until request_stop() has been called, then shut down."""
        while not self._stop_requested:
            self.step()
        self.shutdown()

    def shutdown(self) -> None:
        """Flush any remaining events, write the final byte counters, and close the store.

        Counters are written once, here, rather than periodically: this
        baseline's scenarios are short enough that a single end-of-run
        total is sufficient, and it keeps every other event line free of
        a running total that would otherwise have to be repeated on each one.

        2026-10-23 fix: _emit_pending_snapshots() is called here too, before
        the `counters` line, as a last-resort safety net -- step() itself
        already calls it on every one of its own exit paths, so this should
        normally find nothing pending, but a stop triggered from somewhere
        this file hasn't been made to call it from must still never lose an
        already-captured entry rather than silently dropping it.
        """
        self._emit_pending_snapshots()
        self._drain_engine_events()
        self._events.write(
            {
                "event": "counters",
                "bytes_sent": self._transport.bytes_sent,
                "bytes_received": self._transport.bytes_received,
                # 2026-10-22 fix: bytes_sent/received above also
                # include a probe's own content-body fetch, so an analysis
                # reading this line for *sync protocol* overhead needs a
                # total that excludes it -- see cefpyco_transport's own
                # "Byte counters" docstring entry.
                "control_bytes_sent": self._transport.control_bytes_sent,
                "control_bytes_received": self._transport.control_bytes_received,
            }
        )
        self._events.close()
        self._store.close()


def build_agent(config: AgentConfig, handle: CefpycoHandle, *, clock: Callable[[], float] = time.time) -> Agent:
    """Wire one agent's Store, SyncEngine, CefpycoTransport and EventWriter together from `config`.

    `handle` only needs to match cefpyco_transport.CefpycoHandle's shape:
    a real cefpyco handle in main(), a fake one in tests. See
    CefpycoTransport.__init__ for why the engine is bound to the
    transport only after the engine itself has finished constructing.
    """
    os.makedirs(config.state_dir, exist_ok=True)
    store = Store(os.path.join(config.state_dir, "index.db"), synchronous=config.store_synchronous)
    transport = CefpycoTransport(handle, clock=clock, receive_timeout_ms=config.receive_timeout_ms)
    # validate_config guarantees a seed whenever choices are set; without
    # choices the engine never draws either choices-mode rng, so their defaults stay.
    choice_rng = (
        {}
        if config.seed is None
        else {
            "poll_choice_rng": _poll_choice_rng(config.seed, config.node),
            "poll_choice_jitter_rng": _poll_choice_jitter_rng(config.seed, config.node),
        }
    )
    engine = SyncEngine(
        build_sync_config(config), store, transport, poll_jitter_rng=_poll_jitter_rng(config.node), **choice_rng
    )
    transport.engine = engine
    events = EventWriter(os.path.join(config.state_dir, "events.jsonl"), clock=clock)
    agent = Agent(config, engine, store, transport, events)
    # Same construction-order reason as transport.engine above: the Agent
    # that owns the real stop flag (and the counters_snapshot/stop_at_s
    # boundary check that now lives alongside it, _on_receive_boundary) can
    # only be built from this transport, not the other way around.
    transport.should_stop = agent._on_receive_boundary
    return agent


def install_signal_handlers(agent: Agent) -> None:
    """Make SIGTERM and SIGINT request a clean stop rather than killing the process outright.

    Only sets a flag: the current receive()/send_data() call is allowed
    to finish first, so a signal can never land mid-call across the
    Python/C boundary and leave the cefpyco handle in an inconsistent
    state (the same reasoning cefpyco's own semindex_smoke/producer.py
    uses for its SIGTERM/SIGINT handler).
    """

    def _request_stop(_signum, _frame) -> None:
        agent.request_stop()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the agent process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to this node's JSON config")
    parser.add_argument("--node", default=None, help="Override the config's node name")
    parser.add_argument("--t0-ms", type=int, default=None, dest="t0_ms", help="Override the config's t0_ms")
    parser.add_argument("--cefore-conf-dir", default=None, help="Override the config's cefore_conf_dir")
    parser.add_argument("--state-dir", default=None, help="Override the config's state_dir")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: load the config, build the agent against a real cefpyco handle, and run until signaled.

    cefpyco is imported here, not at module scope, so this module (and
    everything testable in it) stays importable without cefpyco's native
    extension ever having to load -- this repo's own CLAUDE.md: "the sync
    core is transport-independent; cefpyco is reached only through the
    transport adapter."
    """
    import cefpyco

    args = parse_args(argv)
    config = load_config(
        args.config,
        node=args.node,
        t0_ms=args.t0_ms,
        cefore_conf_dir=args.cefore_conf_dir,
        state_dir=args.state_dir,
    )
    with cefpyco.create_handle(ceforedir=config.cefore_conf_dir, portnum=0) as handle:
        agent = build_agent(config, handle)
        install_signal_handlers(agent)
        agent.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
