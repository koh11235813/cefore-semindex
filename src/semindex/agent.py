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
from semindex.store import Store

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
    sync_overrides: dict[str, float | int]
    receive_timeout_ms: int
    chunk_lifetime_ms: int


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
_ALLOWED_KEYS = frozenset(_REQUIRED_KEYS) | {"fault"} | set(_SYNC_OVERRIDE_KEYS) | set(_AGENT_NUMERIC_KEYS)

_SCHEDULE_ENTRY_REQUIRED = ("at_s", "content_name", "zone", "disaster", "media")
_SCHEDULE_ENTRY_KEYS = frozenset(_SCHEDULE_ENTRY_REQUIRED)

_PROBE_ENTRY_REQUIRED = ("at_s",)
_PROBE_ENTRY_OPTIONAL = ("zone", "disaster", "media")
_PROBE_ENTRY_KEYS = frozenset(_PROBE_ENTRY_REQUIRED) | frozenset(_PROBE_ENTRY_OPTIONAL)

_FAULT_DROP_REC_ENTRY_KEYS = frozenset({"origin", "seq", "times"})


def _check_numeric(key: str, value, kind: type) -> None:
    """Validate one numeric config value: the right numeric kind, never a bool, never negative."""
    if kind is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
        label = "an integer"
    else:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        label = "a number"
    if not ok:
        raise ConfigError(f"{key} must be {label}")
    if value < 0:
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

    sync_overrides: dict[str, float | int] = {}
    for key, kind in _SYNC_OVERRIDE_KEYS.items():
        if key in raw:
            _check_numeric(key, raw[key], kind)
            sync_overrides[key] = raw[key]

    for key, kind in _AGENT_NUMERIC_KEYS.items():
        if key in raw:
            _check_numeric(key, raw[key], kind)

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
        """Stamp `event` with the current wall-clock epoch ms and append it as one JSON line."""
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
        """Wire the four already-constructed pieces together; schedule/probe entries are consumed in time order."""
        self._config = config
        self._engine = engine
        self._store = store
        self._transport = transport
        self._events = events
        self._schedule = sorted(config.schedule, key=lambda e: e.at_s)
        self._probes = sorted(config.probes, key=lambda e: e.at_s)
        self._stop_requested = False

    def request_stop(self) -> None:
        """Ask the loop to exit after its current step; safe to call from a signal handler."""
        self._stop_requested = True

    def is_stop_requested(self) -> bool:
        """Whether request_stop() has already been called.

        build_agent wires this in as the transport's should_stop, so a
        probe's fetch_content call -- which runs synchronously inside one
        step(), unlike every other transport call -- notices a stop
        request within one receive_timeout_ms instead of blocking for up
        to its own chunk_lifetime_ms.
        """
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
        """Run every scheduled probe whose at_s has come due."""
        elapsed = self._elapsed_s(now)
        while self._probes and self._probes[0].at_s <= elapsed:
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

    def _drain_engine_events(self) -> None:
        """Flush every event the engine has queued since the last drain."""
        for event in self._engine.drain_events():
            self._events.write(event)

    def step(self) -> None:
        """One loop iteration: poll the transport, tick the engine, fire due schedule/probe entries, flush events."""
        self._transport.poll()
        now = self._transport.now()
        self._engine.on_tick(now)
        self._drain_schedule(now)
        self._drain_probes(now)
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
        """
        self._drain_engine_events()
        self._events.write(
            {
                "event": "counters",
                "bytes_sent": self._transport.bytes_sent,
                "bytes_received": self._transport.bytes_received,
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
    store = Store(os.path.join(config.state_dir, "index.db"))
    transport = CefpycoTransport(handle, clock=clock, receive_timeout_ms=config.receive_timeout_ms)
    engine = SyncEngine(
        build_sync_config(config), store, transport, poll_jitter_rng=_poll_jitter_rng(config.node)
    )
    transport.engine = engine
    events = EventWriter(os.path.join(config.state_dir, "events.jsonl"), clock=clock)
    agent = Agent(config, engine, store, transport, events)
    # Same construction-order reason as transport.engine above: the Agent
    # that owns the real stop flag can only be built from this transport,
    # not the other way around.
    transport.should_stop = agent.is_stop_requested
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
