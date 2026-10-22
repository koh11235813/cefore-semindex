"""semindex.gen: one experiment description -> a ReCefore scenario plus per-node agent configs.

A hand-written ReCefore YAML plus N hand-written agent JSON files (the
shape examples/two_host_sync.yaml/h0.json/h1.json were built as) risks the
two files disagreeing about a content name: the put event's `uri` and an
Index record's `content_name` are independent strings an author could
type differently without anything noticing. This module removes that risk
structurally by deriving both from one experiment description, in one
place (`_content_name_for`), so they cannot drift apart.

It also always emits the handful of ReCefore config shapes map.md and the
agent's own schema settled on for this baseline (ADR-0001/0002): `k: 1`
(see the module docstring link below for why), `hosts >= 3`, explicit
`links`, `prefix_routes` for every Sync-agent host, one `host_apps` entry
covering every Sync-agent host, `warmup_gets` restricted to the publisher
fetching its own body, `debug.artifacts: ["node_dirs"]` so events.jsonl
survives teardown, and the publisher's `put` always at `at: 0`.

Run as:

    python -m semindex.gen <experiment.json> <outdir>

stdlib only: the emitted `<name>.yaml` is written as indented JSON, which
is valid YAML (JSON is a structural subset of YAML 1.1/1.2, and
ReCefore's own loader reads a `.yaml` file with `yaml.safe_load`) -- this
avoids hand-rolling a YAML block-style serializer for a format this
module fully controls the shape of.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

from semindex import schema

# This generator and the agent it configures live in, and are only ever
# run from, this one repo's own venv -- ReCefore's host_apps argv is a
# literal argv list (no shell, no $PATH lookup), so the interpreter must
# be named by an absolute path matching examples/two_host_sync.yaml's own
# host_apps entry.
SEMINDEX_VENV_PYTHON = "/home/lab_shared/cefore-semindex/.venv/bin/python3"

# 2026-10-23 fix: that venv's editable install resolves `semindex` to the
# main checkout's src/, so an agent launched from it ran whatever branch
# that checkout had out, not the code that generated its config (a
# generator on a feature worktree emitted agent keys the main checkout's
# agent rejected at startup). Pinning PYTHONPATH to the src/ directory of
# the package doing the generating keeps generator and agent the same
# code; PYTHONPATH precedes the editable install's path entry.
SEMINDEX_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# map.md has no readiness hook between a host_apps process starting and
# cefnetd being reachable from it; the two-host kit budgets a few seconds
# for that gap before any node tries to register a prefix or send
# anything. A schedule entry due before this margin has elapsed could fire
# before the agent's cefpyco handle is even usable.
MIN_FIRST_RECORD_MARGIN_S = 3.0

_ATTR_FIELDS = ("zone", "disaster", "media")


class ExperimentError(ValueError):
    """An experiment description failed validation."""


@dataclass(frozen=True)
class ScheduleSpec:
    """One record-creation entry for one agent host, before `content_name` is filled in."""

    at_s: float
    zone: str
    disaster: str
    media: str


@dataclass(frozen=True)
class ProbeSpec:
    """One probe entry for one agent host; `None` attrs mean "any value", as agent.py accepts."""

    at_s: float
    zone: str | None
    disaster: str | None
    media: str | None


@dataclass(frozen=True)
class FaultDropRecSpec:
    """One fault.drop_rec entry: withhold replies for (origin, seq) `times` times."""

    origin: str
    seq: int
    times: int


@dataclass(frozen=True)
class EventSpec:
    """One ReCefore scheduler event passed through into the generated YAML's `events` list.

    `fields` holds every key besides `at`/`type` (e.g. `nodes` for
    link_down/link_up, `host`/`downtime_s` for host_app_restart), kept as
    a plain dict since this module does not need to interpret them --
    only pass them through unchanged to the one place that does
    (ReCefore's own scheduler/validator).
    """

    at: float
    type: str
    fields: dict


@dataclass(frozen=True)
class PublisherSpec:
    """The one host that publishes a real content body via ReCefore's `put`."""

    host: int
    uri: str
    file: str
    expiry: float
    cache_time: float


@dataclass(frozen=True)
class ExperimentSpec:
    """A fully validated experiment description: everything `generate()` needs, nothing it has to re-derive."""

    name: str
    num: int
    seed: int
    group: str
    hosts: int
    links: tuple[tuple[int, int], ...]
    duration: int
    output_dir: str
    agents: tuple[int, ...]
    publisher: PublisherSpec
    schedule: dict[int, tuple[ScheduleSpec, ...]]
    probes: dict[int, tuple[ProbeSpec, ...]]
    fault: dict[int, tuple[FaultDropRecSpec, ...]]
    events: tuple[EventSpec, ...]
    stop_at_s: float | None
    counters_snapshot_at_s: tuple[float, ...]


_TOP_LEVEL_REQUIRED = (
    "name",
    "num",
    "seed",
    "group",
    "hosts",
    "links",
    "duration",
    "output_dir",
    "agents",
    "publisher",
)
_TOP_LEVEL_OPTIONAL = ("schedule", "probes", "fault", "events", "stop_at_s", "counters_snapshot_at_s")
_TOP_LEVEL_ALLOWED = frozenset(_TOP_LEVEL_REQUIRED) | frozenset(_TOP_LEVEL_OPTIONAL)

_PUBLISHER_REQUIRED = ("host", "uri", "file", "expiry", "cache_time")
_SCHEDULE_ENTRY_ALLOWED = frozenset({"at_s", "zone", "disaster", "media"})
_PROBE_ENTRY_ALLOWED = frozenset({"at_s", "zone", "disaster", "media"})
_FAULT_DROP_REC_ALLOWED = frozenset({"origin", "seq", "times"})
_EVENT_ALLOWED_TYPES = frozenset({"link_down", "link_up", "host_app_restart"})
_LINK_EVENT_REQUIRED = frozenset({"nodes"})
_RESTART_EVENT_REQUIRED = frozenset({"host", "downtime_s"})


def _node_name(host: int) -> str:
    """The one place a host index becomes a node name; every other module reads this convention, none re-derives it."""
    return f"h{host}"


def _content_name_for(host: int, spec_publisher: PublisherSpec) -> str:
    """The content name this host's own Index records point at.

    The publisher's own records must name exactly the body it actually
    published via `put` -- any other string here would make the put event
    and the record schedule disagree about what was published. Every
    other agent host has no real `put` behind it in these experiments, so
    its records get a per-node placeholder name instead: no generated
    experiment in this baseline has a second real publisher, so nothing
    else ever has to match it.
    """
    if host == spec_publisher.host:
        return spec_publisher.uri
    return f"ccnx:/content/{_node_name(host)}/body"


def _require_int(value, label: str) -> int:
    """Validate `value` as an int (not a bool, which is a Python int subtype) and return it."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ExperimentError(f"{label} must be an integer")
    return value


def _require_number(value, label: str) -> float:
    """Validate `value` as an int or float (not a bool) and return it as a float."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ExperimentError(f"{label} must be a number")
    return float(value)


def _require_non_empty_string(value, label: str) -> str:
    """Validate `value` as a non-empty string and return it."""
    if not isinstance(value, str) or not value:
        raise ExperimentError(f"{label} must be a non-empty string")
    return value


def _require_host_index(value, label: str, hosts: int) -> int:
    """Validate `value` as an integer host index in range [0, hosts) and return it."""
    host = _require_int(value, label)
    if host < 0 or host >= hosts:
        raise ExperimentError(f"{label} must be in range [0, {hosts})")
    return host


def _parse_host_key(host_key: str, label: str) -> int:
    """Parse a schedule/probes/fault dict key (a JSON object key, always a string) as a host index."""
    try:
        return int(host_key)
    except (TypeError, ValueError):
        raise ExperimentError(f"{label} must be an integer host index") from None


def _parse_publisher(raw, hosts: int) -> PublisherSpec:
    """Validate the required `publisher` block and build the `PublisherSpec` it describes."""
    if not isinstance(raw, dict):
        raise ExperimentError("publisher must be an object")
    unknown = sorted(set(raw) - set(_PUBLISHER_REQUIRED))
    if unknown:
        raise ExperimentError(f"publisher has unknown key: {unknown[0]!r}")
    missing = [k for k in _PUBLISHER_REQUIRED if k not in raw]
    if missing:
        raise ExperimentError(f"publisher missing required key: {missing[0]!r}")
    host = _require_host_index(raw["host"], "publisher.host", hosts)
    uri = _require_non_empty_string(raw["uri"], "publisher.uri")
    file_ = _require_non_empty_string(raw["file"], "publisher.file")
    expiry = _require_number(raw["expiry"], "publisher.expiry")
    cache_time = _require_number(raw["cache_time"], "publisher.cache_time")
    return PublisherSpec(host=host, uri=uri, file=file_, expiry=expiry, cache_time=cache_time)


def _parse_links(raw, hosts: int) -> tuple[tuple[int, int], ...]:
    """Validate the required `links` list of [host_a, host_b] pairs and build the tuple of pairs it describes."""
    if not isinstance(raw, list) or not raw:
        raise ExperimentError("links must be a non-empty list of [host_a, host_b] pairs")
    links = []
    for idx, entry in enumerate(raw):
        prefix = f"links[{idx}]"
        if not isinstance(entry, list) or len(entry) != 2:
            raise ExperimentError(f"{prefix} must be a 2-element list")
        a = _require_host_index(entry[0], f"{prefix}[0]", hosts)
        b = _require_host_index(entry[1], f"{prefix}[1]", hosts)
        if a == b:
            raise ExperimentError(f"{prefix} must not link a host to itself")
        links.append((a, b))
    return tuple(links)


def _parse_agents(raw, hosts: int) -> tuple[int, ...]:
    """Validate the required `agents` list of host indices and return it sorted, as a tuple."""
    if not isinstance(raw, list) or not raw:
        raise ExperimentError("agents must be a non-empty list of host indices")
    agents = [_require_host_index(h, f"agents[{idx}]", hosts) for idx, h in enumerate(raw)]
    if len(agents) != len(set(agents)):
        raise ExperimentError("agents must not contain duplicates")
    return tuple(sorted(agents))


def _neighbors_of(host: int, links: tuple[tuple[int, int], ...], agents: tuple[int, ...]) -> tuple[str, ...]:
    """Every other agent host directly linked to `host`, as node names, in ascending host order.

    Scoped to `agents`, not every linked host: a non-agent host (e.g. an
    idle router kept only to satisfy ReCefore's `hosts >= 3`) runs no
    Sync agent and so can never be a sync neighbor, even if it sits
    directly on the link.
    """
    neighbor_hosts = set()
    for a, b in links:
        if a == host and b in agents:
            neighbor_hosts.add(b)
        elif b == host and a in agents:
            neighbor_hosts.add(a)
    neighbor_hosts.discard(host)
    return tuple(_node_name(h) for h in sorted(neighbor_hosts))


def _parse_schedule(raw, hosts: int, agents: tuple[int, ...]) -> dict[int, tuple[ScheduleSpec, ...]]:
    """Validate the optional `schedule` block (host index -> record-creation entries) and build it."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ExperimentError("schedule must be an object keyed by host index")
    result: dict[int, tuple[ScheduleSpec, ...]] = {}
    for host_key, entries in raw.items():
        host = _require_host_index(_parse_host_key(host_key, "schedule key"), f"schedule[{host_key!r}]", hosts)
        if host not in agents:
            raise ExperimentError(f"schedule[{host_key!r}] names a host that is not in agents")
        if not isinstance(entries, list):
            raise ExperimentError(f"schedule[{host_key!r}] must be a list")
        parsed = []
        for idx, entry in enumerate(entries):
            prefix = f"schedule[{host_key!r}][{idx}]"
            if not isinstance(entry, dict):
                raise ExperimentError(f"{prefix} must be an object")
            unknown = sorted(set(entry) - _SCHEDULE_ENTRY_ALLOWED)
            if unknown:
                raise ExperimentError(f"{prefix} has unknown key: {unknown[0]!r}")
            missing = [k for k in ("at_s",) + _ATTR_FIELDS if k not in entry]
            if missing:
                raise ExperimentError(f"{prefix} missing required key: {missing[0]!r}")
            at_s = _require_number(entry["at_s"], f"{prefix}.at_s")
            zone = entry["zone"]
            if zone not in schema.ZONES:
                raise ExperimentError(f"{prefix}.zone must be one of {sorted(schema.ZONES)}")
            disaster = entry["disaster"]
            if disaster not in schema.DISASTERS:
                raise ExperimentError(f"{prefix}.disaster must be one of {sorted(schema.DISASTERS)}")
            media = entry["media"]
            if media not in schema.MEDIA:
                raise ExperimentError(f"{prefix}.media must be one of {sorted(schema.MEDIA)}")
            parsed.append(ScheduleSpec(at_s=at_s, zone=zone, disaster=disaster, media=media))
        result[host] = tuple(parsed)
    return result


def _parse_probes(raw, hosts: int, agents: tuple[int, ...]) -> dict[int, tuple[ProbeSpec, ...]]:
    """Validate the optional `probes` block (host index -> probe entries) and build it."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ExperimentError("probes must be an object keyed by host index")
    result: dict[int, tuple[ProbeSpec, ...]] = {}
    for host_key, entries in raw.items():
        host = _require_host_index(_parse_host_key(host_key, "probes key"), f"probes[{host_key!r}]", hosts)
        if host not in agents:
            raise ExperimentError(f"probes[{host_key!r}] names a host that is not in agents")
        if not isinstance(entries, list):
            raise ExperimentError(f"probes[{host_key!r}] must be a list")
        parsed = []
        for idx, entry in enumerate(entries):
            prefix = f"probes[{host_key!r}][{idx}]"
            if not isinstance(entry, dict):
                raise ExperimentError(f"{prefix} must be an object")
            unknown = sorted(set(entry) - _PROBE_ENTRY_ALLOWED)
            if unknown:
                raise ExperimentError(f"{prefix} has unknown key: {unknown[0]!r}")
            if "at_s" not in entry:
                raise ExperimentError(f"{prefix} missing required key: 'at_s'")
            at_s = _require_number(entry["at_s"], f"{prefix}.at_s")
            values: dict[str, str | None] = {}
            vocab_by_field = {"zone": schema.ZONES, "disaster": schema.DISASTERS, "media": schema.MEDIA}
            for field_name, vocabulary in vocab_by_field.items():
                if field_name in entry:
                    if entry[field_name] not in vocabulary:
                        raise ExperimentError(f"{prefix}.{field_name} must be one of {sorted(vocabulary)}")
                    values[field_name] = entry[field_name]
                else:
                    values[field_name] = None
            parsed.append(ProbeSpec(at_s=at_s, **values))
        result[host] = tuple(parsed)
    return result


def _parse_fault(raw, hosts: int, agents: tuple[int, ...]) -> dict[int, tuple[FaultDropRecSpec, ...]]:
    """Validate the optional `fault` block (host index -> drop_rec entries) and build it."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ExperimentError("fault must be an object keyed by host index")
    result: dict[int, tuple[FaultDropRecSpec, ...]] = {}
    for host_key, block in raw.items():
        host = _require_host_index(_parse_host_key(host_key, "fault key"), f"fault[{host_key!r}]", hosts)
        if host not in agents:
            raise ExperimentError(f"fault[{host_key!r}] names a host that is not in agents")
        if not isinstance(block, dict):
            raise ExperimentError(f"fault[{host_key!r}] must be an object")
        unknown = sorted(set(block) - {"drop_rec"})
        if unknown:
            raise ExperimentError(f"fault[{host_key!r}] has unknown key: {unknown[0]!r}")
        drop_rec = block.get("drop_rec", [])
        if not isinstance(drop_rec, list):
            raise ExperimentError(f"fault[{host_key!r}].drop_rec must be a list")
        parsed = []
        for idx, entry in enumerate(drop_rec):
            prefix = f"fault[{host_key!r}].drop_rec[{idx}]"
            if not isinstance(entry, dict):
                raise ExperimentError(f"{prefix} must be an object")
            unknown_entry = sorted(set(entry) - _FAULT_DROP_REC_ALLOWED)
            if unknown_entry:
                raise ExperimentError(f"{prefix} has unknown key: {unknown_entry[0]!r}")
            missing_entry = [k for k in _FAULT_DROP_REC_ALLOWED if k not in entry]
            if missing_entry:
                raise ExperimentError(f"{prefix} missing required key: {sorted(missing_entry)[0]!r}")
            origin = _require_non_empty_string(entry["origin"], f"{prefix}.origin")
            seq = _require_int(entry["seq"], f"{prefix}.seq")
            if seq < 1:
                raise ExperimentError(f"{prefix}.seq must be >= 1")
            times = _require_int(entry["times"], f"{prefix}.times")
            if times < 1:
                raise ExperimentError(f"{prefix}.times must be >= 1")
            parsed.append(FaultDropRecSpec(origin=origin, seq=seq, times=times))
        result[host] = tuple(parsed)
    return result


def _parse_events(raw, hosts: int) -> tuple[EventSpec, ...]:
    """Validate the optional `events` list (link_down/link_up/host_app_restart) and build it."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ExperimentError("events must be a list")
    parsed = []
    for idx, entry in enumerate(raw):
        prefix = f"events[{idx}]"
        if not isinstance(entry, dict):
            raise ExperimentError(f"{prefix} must be an object")
        if "at" not in entry:
            raise ExperimentError(f"{prefix} missing required key: 'at'")
        if "type" not in entry:
            raise ExperimentError(f"{prefix} missing required key: 'type'")
        at = _require_number(entry["at"], f"{prefix}.at")
        if at < 0:
            raise ExperimentError(f"{prefix}.at must be >= 0")
        event_type = entry["type"]
        if event_type not in _EVENT_ALLOWED_TYPES:
            raise ExperimentError(f"{prefix}.type must be one of {sorted(_EVENT_ALLOWED_TYPES)}")
        fields = {k: v for k, v in entry.items() if k not in ("at", "type")}
        if event_type in ("link_down", "link_up"):
            missing = sorted(_LINK_EVENT_REQUIRED - set(fields))
            if missing:
                raise ExperimentError(f"{prefix} missing required key: {missing[0]!r}")
            nodes = fields["nodes"]
            if not isinstance(nodes, list) or len(nodes) != 2:
                raise ExperimentError(f"{prefix}.nodes must be a 2-element list")
            for pos, h in enumerate(nodes):
                _require_host_index(h, f"{prefix}.nodes[{pos}]", hosts)
        elif event_type == "host_app_restart":
            missing = sorted(_RESTART_EVENT_REQUIRED - set(fields))
            if missing:
                raise ExperimentError(f"{prefix} missing required key: {missing[0]!r}")
            _require_host_index(fields["host"], f"{prefix}.host", hosts)
            downtime_s = _require_number(fields["downtime_s"], f"{prefix}.downtime_s")
            if downtime_s <= 0:
                raise ExperimentError(f"{prefix}.downtime_s must be > 0")
        parsed.append(EventSpec(at=at, type=event_type, fields=fields))
    return tuple(parsed)


def _parse_stop_at_s(raw) -> float | None:
    """Validate the optional `stop_at_s` top-level key and return it, or None if absent.

    Forwarded verbatim into every generated agent config; this module
    checks only that the value is a number, never what an agent process
    does with it.
    """
    if raw is None:
        return None
    return _require_number(raw, "stop_at_s")


def _parse_counters_snapshot_at_s(raw) -> tuple[float, ...]:
    """Validate the optional `counters_snapshot_at_s` top-level key and build the tuple it describes.

    Forwarded verbatim into every generated agent config; this module
    checks only that the value is a list of numbers. An absent key builds
    an empty tuple, which is indistinguishable from an explicit `[]`.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ExperimentError("counters_snapshot_at_s must be a list of numbers")
    return tuple(_require_number(v, f"counters_snapshot_at_s[{idx}]") for idx, v in enumerate(raw))


def parse_experiment(raw: dict) -> ExperimentSpec:
    """Validate an experiment description dict and build the `ExperimentSpec` it describes.

    Checks run in a fixed order (object shape, unknown/missing top-level
    keys, then each field) so the same malformed input always raises the
    same `ExperimentError` message, the same discipline agent.py's
    validate_config uses for its own config.
    """
    if not isinstance(raw, dict):
        raise ExperimentError("experiment description must be a JSON object")
    unknown = sorted(set(raw) - _TOP_LEVEL_ALLOWED)
    if unknown:
        raise ExperimentError(f"unknown experiment key: {unknown[0]!r}")
    missing = [k for k in _TOP_LEVEL_REQUIRED if k not in raw]
    if missing:
        raise ExperimentError(f"missing required experiment key: {missing[0]!r}")

    name = _require_non_empty_string(raw["name"], "name")
    # 2026-10-22 fix: `name` becomes a path component verbatim when the
    # generated config and per-node files are written out
    # (`<outdir>/<name>.yaml`, `<outdir>/<name>/<node>.json`), with no
    # further sanitizing at that point -- an absolute path, an embedded
    # separator (e.g. "../escaped"), or "." / ".." would let a generated
    # experiment write outside the `outdir` it was meant to stay under.
    # Checked once here, at the one validation boundary every path into an
    # ExperimentSpec already goes through.
    if os.path.isabs(name) or "/" in name or os.sep in name or name in (".", ".."):
        raise ExperimentError(f"name must not be a path: {name!r}")
    num = _require_int(raw["num"], "num")
    seed = _require_int(raw["seed"], "seed")
    group = _require_non_empty_string(raw["group"], "group")

    hosts = _require_int(raw["hosts"], "hosts")
    if hosts < 3:
        # ReCefore's own disaster scenario rejects hosts < 3 outright
        # (two_host_sync.yaml's own comment); every generated experiment
        # inherits that floor rather than discovering it at run time.
        raise ExperimentError("hosts must be >= 3")

    links = _parse_links(raw["links"], hosts)
    # 2026-10-22: cross-checked a generated t1.yaml against the companion
    # ReCefore checkout's own validate_config() and it rejected duration as
    # a float -- that option is declared type "int" there (OptionSpec in
    # src/core/config/validator.py), and a whole-numbered float like 100.0
    # still fails it. Stricter than _require_number on purpose.
    duration = _require_int(raw["duration"], "duration")
    if duration <= 0:
        raise ExperimentError("duration must be > 0")
    output_dir = _require_non_empty_string(raw["output_dir"], "output_dir")
    agents = _parse_agents(raw["agents"], hosts)
    publisher = _parse_publisher(raw["publisher"], hosts)

    schedule = _parse_schedule(raw.get("schedule"), hosts, agents)
    probes = _parse_probes(raw.get("probes"), hosts, agents)
    fault = _parse_fault(raw.get("fault"), hosts, agents)
    events = _parse_events(raw.get("events"), hosts)
    stop_at_s = _parse_stop_at_s(raw.get("stop_at_s"))
    counters_snapshot_at_s = _parse_counters_snapshot_at_s(raw.get("counters_snapshot_at_s"))

    all_at_s = [entry.at_s for entries in schedule.values() for entry in entries]
    if all_at_s and min(all_at_s) < MIN_FIRST_RECORD_MARGIN_S:
        raise ExperimentError(
            f"earliest schedule at_s ({min(all_at_s)}) is below the "
            f"{MIN_FIRST_RECORD_MARGIN_S}s startup margin (no host_apps readiness hook)"
        )

    return ExperimentSpec(
        name=name,
        num=num,
        seed=seed,
        group=group,
        hosts=hosts,
        links=links,
        duration=duration,
        output_dir=output_dir,
        agents=agents,
        publisher=publisher,
        schedule=schedule,
        probes=probes,
        fault=fault,
        events=events,
        stop_at_s=stop_at_s,
        counters_snapshot_at_s=counters_snapshot_at_s,
    )


@dataclass(frozen=True)
class GeneratedFiles:
    """What `generate()` produces: the ReCefore config dict plus one agent config dict per agent host."""

    recefore_config: dict
    agent_configs: dict[str, dict]  # node name -> agent JSON config


def _build_recefore_config(spec: ExperimentSpec, config_dir_abspath: str) -> dict:
    """Build the ReCefore disaster config dict (written out as `<name>.yaml`)."""
    put_event = {
        "at": 0,
        "type": "put",
        "host": spec.publisher.host,
        "uri": spec.publisher.uri,
        "file": spec.publisher.file,
        "expiry": spec.publisher.expiry,
        "cache_time": spec.publisher.cache_time,
    }
    passthrough_events = [{"at": e.at, "type": e.type, **e.fields} for e in spec.events]

    prefix_routes = {f"ccnx:/semidx/{spec.group}/{_node_name(h)}": h for h in spec.agents}

    config_path_template = f"{config_dir_abspath}/" + "{node}.json"
    host_apps = [
        {
            "hosts": list(spec.agents),
            "argv": [
                SEMINDEX_VENV_PYTHON,
                "-m",
                "semindex.agent",
                "--config",
                config_path_template,
                "--node",
                "{node}",
                "--t0-ms",
                "{t0_ms}",
                "--cefore-conf-dir",
                "{cefore_conf_dir}",
                "--state-dir",
                "{node_dir}/semindex",
            ],
            "env": {"PYTHONPATH": SEMINDEX_SRC_DIR},
            "log": "{node_dir}/semindex_agent.log",
        }
    ]

    return {
        "hosts": spec.hosts,
        "seed": spec.seed,
        # ADR-0002's naming scheme registers exactly one prefix per node
        # (map.md "重なるプレフィックスを複数登録しない"); FIB fan-out
        # beyond the direct next hop plays no role a Sync agent relies on
        # here, so every generated experiment pins the floor value.
        "k": 1,
        "num": spec.num,
        "output_dir": spec.output_dir,
        "no_cli": True,
        "duration": spec.duration,
        "results_json": "results.json",
        "publisher_host": spec.publisher.host,
        "links": [list(pair) for pair in spec.links],
        "prefix_routes": prefix_routes,
        "debug": {"artifacts": ["node_dirs"]},
        "warmup_gets": [{"host": spec.publisher.host, "uri": spec.publisher.uri}],
        "events": [put_event] + passthrough_events,
        "host_apps": host_apps,
    }


def _build_agent_config(spec: ExperimentSpec, host: int) -> dict:
    """Build one agent host's JSON config (written out as `<name>/<node>.json`)."""
    node = _node_name(host)
    content_name = _content_name_for(host, spec.publisher)

    schedule = [
        {"at_s": e.at_s, "content_name": content_name, "zone": e.zone, "disaster": e.disaster, "media": e.media}
        for e in spec.schedule.get(host, ())
    ]
    probes = []
    for p in spec.probes.get(host, ()):
        entry: dict = {"at_s": p.at_s}
        for field_name in _ATTR_FIELDS:
            value = getattr(p, field_name)
            if value is not None:
                entry[field_name] = value
        probes.append(entry)

    config = {
        "node": node,
        "group": spec.group,
        "neighbors": list(_neighbors_of(host, spec.links, spec.agents)),
        # Overridden on every real invocation by host_apps' own
        # --cefore-conf-dir/--state-dir/--t0-ms CLI flags (agent.py's
        # module docstring); kept as an obviously-a-placeholder string so
        # a run that forgot to pass the override fails loudly instead of
        # writing into some directory that happens to exist.
        "cefore_conf_dir": "/placeholder-overridden-by---cefore-conf-dir",
        "state_dir": "/placeholder-overridden-by---state-dir",
        "t0_ms": 0,
        "schedule": schedule,
        "probes": probes,
    }

    # Both optional and, when present, identical for every agent host --
    # this experiment description has no per-host notion of "stop" or
    # "snapshot" times, only one shared measurement boundary.
    if spec.stop_at_s is not None:
        config["stop_at_s"] = spec.stop_at_s
    if spec.counters_snapshot_at_s:
        config["counters_snapshot_at_s"] = list(spec.counters_snapshot_at_s)

    drop_rec = spec.fault.get(host, ())
    if drop_rec:
        config["fault"] = {
            "drop_rec": [{"origin": d.origin, "seq": d.seq, "times": d.times} for d in drop_rec]
        }
    return config


def generate(spec: ExperimentSpec, config_dir_abspath: str) -> GeneratedFiles:
    """Build the ReCefore config and every agent config `spec` describes.

    `config_dir_abspath` is the absolute path the generated `<name>/<node>.json`
    files will be written under -- it must be absolute because it is
    baked into the ReCefore config's own `host_apps[].argv` (a literal
    argv list, resolved with no shell and no cwd assumption).
    """
    return GeneratedFiles(
        recefore_config=_build_recefore_config(spec, config_dir_abspath),
        agent_configs={_node_name(h): _build_agent_config(spec, h) for h in spec.agents},
    )


def write_experiment(experiment_path: str, outdir: str) -> list[str]:
    """Read, validate and generate `experiment_path` into `outdir`; return the paths written.

    Writes `<outdir>/<name>.yaml` and `<outdir>/<name>/<node>.json` for
    every agent host. `outdir` is created if missing; `<outdir>/<name>`
    (the per-node config directory) likewise.
    """
    with open(experiment_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    spec = parse_experiment(raw)

    outdir_abspath = os.path.abspath(outdir)
    config_dir_abspath = os.path.join(outdir_abspath, spec.name)
    generated = generate(spec, config_dir_abspath)

    os.makedirs(outdir_abspath, exist_ok=True)
    os.makedirs(config_dir_abspath, exist_ok=True)

    written = []

    yaml_path = os.path.join(outdir_abspath, f"{spec.name}.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"# Generated by semindex.gen from {os.path.abspath(experiment_path)}; do not hand-edit.\n")
        json.dump(generated.recefore_config, f, indent=2, sort_keys=True)
        f.write("\n")
    written.append(yaml_path)

    for node, config in generated.agent_configs.items():
        node_path = os.path.join(config_dir_abspath, f"{node}.json")
        with open(node_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
            f.write("\n")
        written.append(node_path)

    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="Path to the experiment description JSON file")
    parser.add_argument("outdir", help="Directory to write <name>.yaml and <name>/<node>.json into")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: validate and generate one experiment description into `outdir`."""
    args = parse_args(argv)
    try:
        written = write_experiment(args.experiment, args.outdir)
    except ExperimentError as exc:
        print(f"semindex.gen: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
