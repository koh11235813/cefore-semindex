"""Tests for the semindex.agent module: config validation and the agent loop."""

from __future__ import annotations

import json
import random
import signal
import sqlite3
from pathlib import Path

import pytest

from semindex import evalagg, evalgen, gen, holder_state, names, schema
from semindex.agent import (
    ConfigError,
    ScheduledProbe,
    _poll_choice_rng,
    _poll_jitter_rng,
    build_agent,
    build_sync_config,
    install_signal_handlers,
    load_config,
    validate_config,
)
from semindex.store import OriginProgress, Store

from test_cefpyco_transport import _FakeClock, _FakeHandle, _FakePacket


def _minimal_raw() -> dict:
    """A minimal valid raw config dict, for validate_config tests that only need a baseline to mutate."""
    return {
        "node": "h0",
        "group": "g",
        "neighbors": [],
        "cefore_conf_dir": "/does/not/need/to/exist",
        "state_dir": "/does/not/need/to/exist/either",
        "t0_ms": 0,
        "schedule": [],
        "probes": [],
    }


def _config(tmp_path: Path, **overrides):
    """Build a validated AgentConfig for agent-loop tests, with real cefore_conf_dir/state_dir under tmp_path."""
    raw = {
        "node": "h0",
        "group": "g",
        "neighbors": [],
        "cefore_conf_dir": str(tmp_path / "cefnetd"),
        "state_dir": str(tmp_path / "state"),
        "t0_ms": 0,
        "schedule": [],
        "probes": [],
    }
    raw.update(overrides)
    return validate_config(raw)


# -- config validation: exact diagnostics ------------------------------------------


def test_validate_config_rejects_a_non_object():
    """A config that is not a JSON object at all is rejected before any key is even looked at."""
    with pytest.raises(ConfigError) as exc_info:
        validate_config([])
    assert str(exc_info.value) == "config must be a JSON object"


def test_validate_config_rejects_an_unknown_key():
    """An unrecognized top-level key is rejected, not silently ignored."""
    raw = _minimal_raw()
    raw["bogus"] = 1
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "unknown config key: 'bogus'"


def test_validate_config_rejects_a_missing_required_key():
    """A missing required key is reported by name."""
    raw = _minimal_raw()
    del raw["group"]
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "missing required config key: 'group'"


def test_validate_config_rejects_a_node_containing_a_slash():
    """node must be a valid wire name component before any name is ever built from it."""
    raw = _minimal_raw()
    raw["node"] = "h0/evil"
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "node must be a non-empty string without '/'"


def test_validate_config_rejects_a_negative_t0_ms():
    """t0_ms is an epoch millisecond and can never be negative."""
    raw = _minimal_raw()
    raw["t0_ms"] = -1
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "t0_ms must be >= 0"


def test_validate_config_rejects_a_schedule_entry_missing_a_key():
    """Each schedule entry must have every one of its five required keys."""
    raw = _minimal_raw()
    raw["schedule"] = [{"at_s": 0, "content_name": "x", "zone": "zone1", "disaster": "flood"}]
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "schedule[0] missing required key: 'media'"


def test_validate_config_rejects_a_schedule_entry_with_an_out_of_vocabulary_zone():
    """zone must be one of schema.ZONES, the same fixed vocabulary the record itself enforces."""
    raw = _minimal_raw()
    raw["schedule"] = [{"at_s": 0, "content_name": "x", "zone": "zone9", "disaster": "flood", "media": "image"}]
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == f"schedule[0].zone must be one of {sorted(schema.ZONES)}"


def test_validate_config_rejects_a_probe_entry_with_an_unknown_key():
    """A probe entry rejects any key beyond at_s/zone/disaster/media."""
    raw = _minimal_raw()
    raw["probes"] = [{"at_s": 0, "bogus": 1}]
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "probes[0] has unknown key: 'bogus'"


def test_validate_config_accepts_a_probe_entry_with_only_some_attrs_filters():
    """zone/disaster/media are each independently optional on a probe entry; an absent one becomes None (any value)."""
    raw = _minimal_raw()
    raw["probes"] = [{"at_s": 1.5, "zone": "zone1"}]
    config = validate_config(raw)
    assert config.probes == (ScheduledProbe(at_s=1.5, zone="zone1", disaster=None, media=None),)


def test_validate_config_rejects_a_fault_drop_rec_entry_missing_times():
    """Each fault.drop_rec entry must have all of origin, seq and times."""
    raw = _minimal_raw()
    raw["fault"] = {"drop_rec": [{"origin": "h0", "seq": 1}]}
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "fault.drop_rec[0] missing required key: 'times'"


def test_validate_config_rejects_a_non_integer_interest_lifetime_ms():
    """A timing override's numeric kind is checked: interest_lifetime_ms must be an integer, not a float."""
    raw = _minimal_raw()
    raw["interest_lifetime_ms"] = 2.5
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "interest_lifetime_ms must be an integer"


def test_validate_config_rejects_a_zero_interest_lifetime_ms():
    """interest_lifetime_ms: 0 would make every sync Interest time out the instant
    it is sent, before any reply could ever arrive -- it must be > 0, not just >= 0.
    """
    raw = _minimal_raw()
    raw["interest_lifetime_ms"] = 0
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "interest_lifetime_ms must be > 0"


def test_validate_config_rejects_a_zero_receive_timeout_ms():
    """receive_timeout_ms: 0 would spin the agent's poll() loop at full CPU (every
    receive() returns instantly) -- it must be > 0, not just >= 0.
    """
    raw = _minimal_raw()
    raw["receive_timeout_ms"] = 0
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "receive_timeout_ms must be > 0"


def test_validate_config_rejects_a_zero_chunk_lifetime_ms():
    """chunk_lifetime_ms: 0 would make every probe's content fetch time out
    instantly -- it must be > 0, not just >= 0.
    """
    raw = _minimal_raw()
    raw["chunk_lifetime_ms"] = 0
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "chunk_lifetime_ms must be > 0"


def test_validate_config_rejects_a_neighbor_that_is_the_node_itself():
    """A node listing itself as a neighbor would have SyncEngine try to sync state
    and records with itself; this must be rejected at config validation, not left
    to misbehave at runtime.
    """
    raw = _minimal_raw()
    raw["node"] = "h0"
    raw["neighbors"] = ["h1", "h0"]
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "neighbors must not contain the node itself"


def test_validate_config_rejects_duplicate_neighbors():
    """The same neighbor listed twice would double-count its in-flight fetch slots
    against max_in_flight_per_neighbor; this must be rejected, not silently folded.
    """
    raw = _minimal_raw()
    raw["neighbors"] = ["h1", "h1"]
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "neighbors must not contain duplicates"


def test_validate_config_applies_sync_overrides():
    """A given timing override reaches the SyncConfig built from the parsed AgentConfig."""
    raw = _minimal_raw()
    raw["state_poll_interval_s"] = 5.0
    config = validate_config(raw)
    sync_config = build_sync_config(config)
    assert sync_config.state_poll_interval_s == 5.0


def test_validate_config_defaults_stop_at_s_to_none_when_absent():
    """stop_at_s is not required: a config that omits it never stops early (runs
    until SIGTERM/SIGINT, same as before this key existed)."""
    config = validate_config(_minimal_raw())
    assert config.stop_at_s is None


def test_validate_config_accepts_a_given_stop_at_s():
    """A given stop_at_s reaches the parsed AgentConfig unchanged."""
    raw = _minimal_raw()
    raw["stop_at_s"] = 130.0
    config = validate_config(raw)
    assert config.stop_at_s == 130.0


def test_validate_config_rejects_a_non_number_stop_at_s():
    """stop_at_s must be a number, same rule _check_numeric already applies to every
    other timing config key."""
    raw = _minimal_raw()
    raw["stop_at_s"] = "130"
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "stop_at_s must be a number"


def test_validate_config_rejects_a_negative_stop_at_s():
    """stop_at_s is t0-relative and must be >= 0: negative has no meaning (t0 is this
    agent's own start, nothing happens before it)."""
    raw = _minimal_raw()
    raw["stop_at_s"] = -1.0
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "stop_at_s must be >= 0"


def test_validate_config_defaults_counters_snapshot_at_s_to_empty_when_absent():
    """counters_snapshot_at_s is not required: a config that omits it schedules no
    mid-run snapshots."""
    config = validate_config(_minimal_raw())
    assert config.counters_snapshot_at_s == ()


def test_validate_config_accepts_a_given_counters_snapshot_at_s():
    """A given, strictly increasing counters_snapshot_at_s reaches the parsed
    AgentConfig unchanged, as a tuple."""
    raw = _minimal_raw()
    raw["counters_snapshot_at_s"] = [10.0, 20.5]
    config = validate_config(raw)
    assert config.counters_snapshot_at_s == (10.0, 20.5)


def test_validate_config_rejects_a_non_list_counters_snapshot_at_s():
    """counters_snapshot_at_s must be a list, not a bare number."""
    raw = _minimal_raw()
    raw["counters_snapshot_at_s"] = 10.0
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "counters_snapshot_at_s must be a list"


def test_validate_config_rejects_a_non_number_counters_snapshot_at_s_entry():
    """Each counters_snapshot_at_s entry is checked the same way as any other timing value."""
    raw = _minimal_raw()
    raw["counters_snapshot_at_s"] = [10.0, "20"]
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "counters_snapshot_at_s[1] must be a number"


def test_validate_config_rejects_a_negative_counters_snapshot_at_s_entry():
    """Each counters_snapshot_at_s entry is t0-relative and must be >= 0."""
    raw = _minimal_raw()
    raw["counters_snapshot_at_s"] = [-1.0]
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "counters_snapshot_at_s[0] must be >= 0"


def test_validate_config_rejects_a_non_increasing_counters_snapshot_at_s():
    """Two equal (or out-of-order) entries would make the second never due on its own
    -- the first drain already consumed it -- so this is rejected rather than
    silently accepted and then silently short one snapshot.
    """
    raw = _minimal_raw()
    raw["counters_snapshot_at_s"] = [10.0, 10.0]
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == "counters_snapshot_at_s must be strictly increasing"


def test_load_config_applies_cli_overrides_onto_the_file(tmp_path: Path):
    """--node/--t0-ms/--cefore-conf-dir/--state-dir override whatever the file holds for
    those four keys -- ReCefore host_apps only knows them at scenario-start time, so a
    config file written ahead of time cannot carry the real t0_ms itself.
    """
    raw = _minimal_raw()
    raw["node"] = "placeholder"
    config_path = tmp_path / "h0.json"
    config_path.write_text(json.dumps(raw))
    config = load_config(
        str(config_path),
        node="h1",
        t0_ms=1_700_000_000_000,
        cefore_conf_dir="/real/h1",
        state_dir=str(tmp_path / "h1state"),
    )
    assert config.node == "h1"
    assert config.t0_ms == 1_700_000_000_000
    assert config.cefore_conf_dir == "/real/h1"
    assert config.state_dir == str(tmp_path / "h1state")


# -- the agent loop, against a fake cefpyco handle ----------------------------------


def test_build_agent_registers_its_own_prefix_and_creates_the_state_dir(tmp_path: Path):
    """Construction registers exactly this node's own prefix and creates state_dir if it does not exist yet."""
    handle = _FakeHandle()
    config = _config(tmp_path)
    build_agent(config, handle, clock=_FakeClock(0.0))
    assert handle.registered == ["ccnx:/semidx/g/h0"]
    assert Path(config.state_dir).is_dir()


def test_a_scheduled_record_fires_only_once_its_at_s_has_come_due(tmp_path: Path):
    """A schedule entry does not fire before its at_s, and fires exactly once once elapsed time reaches it."""
    clock = _FakeClock(1000.0)
    handle = _FakeHandle()
    config = _config(
        tmp_path,
        t0_ms=1_000_000,  # t0 at clock=1000.0, so elapsed = clock.value - 1000.0
        schedule=[
            {"at_s": 5, "content_name": "ccnx:/content/h0/a", "zone": "zone1", "disaster": "flood", "media": "image"}
        ],
    )
    agent = build_agent(config, handle, clock=clock)
    events_path = Path(config.state_dir) / "events.jsonl"

    agent.step()  # elapsed = 0: not due yet (agent_start was already written at construction)
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [e["event"] for e in lines] == ["agent_start"]

    clock.value = 1005.0  # elapsed = 5
    agent.step()
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    created = [e for e in lines if e["event"] == "record_created"]
    assert len(created) == 1
    assert (created[0]["origin"], created[0]["seq"]) == ("h0", 1)
    # map.md "時刻は壁時計": every event line gets a wall-clock ts_ms, stamped by
    # EventWriter.write from the agent's own injected clock (1005.0s here) --
    # not merely present, but the exact value that clock reads at this step.
    assert created[0]["ts_ms"] == 1_005_000


def test_a_k_4_generated_schedule_becomes_4_records_at_each_generation_time(tmp_path: Path):
    """stage-2 map "負荷": the generator lists k = 4 entries per at_s, and the agent turns
    each due batch into 4 records in one step -- seqs 1-4 at the first time, 5-8 at the
    next -- with no agent change."""
    raw = evalgen.build_experiment(0.2, 7, "s2", "/out", k=4)
    node_config = gen.generate(gen.parse_experiment(raw), "/cfg").agent_configs["h0"]
    node_config.update(
        cefore_conf_dir=str(tmp_path / "cefnetd"), state_dir=str(tmp_path / "state"), t0_ms=1_000_000
    )
    config = validate_config(node_config)
    clock = _FakeClock(1000.0)
    agent = build_agent(config, _FakeHandle(), clock=clock)
    events_path = Path(config.state_dir) / "events.jsonl"

    def _created_seqs() -> list[int]:
        """Seqs of every own record created so far, in order."""
        lines = [json.loads(line) for line in events_path.read_text().splitlines()]
        return [e["seq"] for e in lines if e["event"] == "record_created"]

    clock.value = 1000.0 + evalgen.RECORD_START_S
    agent.step()
    assert _created_seqs() == [1, 2, 3, 4]
    clock.value += evalgen.RECORD_INTERVAL_S
    agent.step()
    assert _created_seqs() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_a_record_fetch_that_gets_no_reply_in_time_emits_fetch_timeout(tmp_path: Path):
    """A record-fetch Interest sent to a neighbor that never replies is reported as
    fetch_timeout once its own lifetime elapses, via the real engine and store, driven
    only through the Agent's step() loop and a fake handle.
    """
    clock = _FakeClock(0.0)
    handle = _FakeHandle()
    config = _config(tmp_path, neighbors=["h1"], interest_lifetime_ms=1000)
    agent = build_agent(config, handle, clock=clock)
    events_path = Path(config.state_dir) / "events.jsonl"

    agent.step()  # on_tick sends the first state-request Interest to h1
    state_name = handle.sent_interests[-1][0]

    # Simulate h1 answering that it holds its own seq 1.
    state_payload = holder_state.encode({"h1": OriginProgress(hi=1, extra=())})
    handle.packets.append(_FakePacket(is_data=True, name=state_name, payload=state_payload))
    agent.step()  # routes the state reply, which immediately plans and sends a fetch Interest

    fetch_name = names.build_record_name("g", "h1", "h1", 1)
    assert any(sent_name == fetch_name for sent_name, *_ in handle.sent_interests)

    clock.value = 1.5  # past the fetch Interest's own 1000ms lifetime, with no Data ever arriving
    agent.step()

    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    timeouts = [e for e in lines if e["event"] == "fetch_timeout"]
    assert len(timeouts) == 1
    assert (timeouts[0]["origin"], timeouts[0]["seq"], timeouts[0]["neighbor"]) == ("h1", 1, "h1")


def test_fault_drop_rec_withholds_exactly_n_replies_then_answers(tmp_path: Path):
    """fault.drop_rec withholds the configured number of replies for one (origin, seq)
    and answers normally from the next request onward -- the real engine's
    handle_interest, reached end-to-end through Agent.step() and CefpycoTransport.
    """
    name = "ccnx:/semidx/g/h0/rec/h0/1"
    handle = _FakeHandle()
    config = _config(
        tmp_path,
        schedule=[
            {"at_s": 0, "content_name": "ccnx:/content/h0/1", "zone": "zone1", "disaster": "flood", "media": "image"}
        ],
        fault={"drop_rec": [{"origin": "h0", "seq": 1, "times": 2}]},
    )
    agent = build_agent(config, handle, clock=_FakeClock(0.0))

    agent.step()  # creates the record; no Interest queued yet, so this is a no-op otherwise
    assert handle.sent_data == []

    for _ in range(2):
        handle.packets.append(_FakePacket(is_interest=True, name=name))
        agent.step()
        assert handle.sent_data == []  # withheld

    handle.packets.append(_FakePacket(is_interest=True, name=name))
    agent.step()
    assert len(handle.sent_data) == 1
    assert handle.sent_data[0][0] == name


def test_probe_discovers_and_retrieves_the_first_matching_record(tmp_path: Path):
    """A probe emits probe_discovered with every match, then fetches the first match's
    content body in full, emitting probe_retrieved with success=True and the byte count.

    Also pins down that the agent's own receive_timeout_ms/chunk_lifetime_ms config
    overrides actually reach CefpycoTransport, rather than silently falling back to
    its module-level defaults: both are given distinguishing values below (not equal
    to either default), and every receive() call and the fetch's own Interest must
    carry them.
    """
    content_name = "ccnx:/content/h0/body"
    handle = _FakeHandle(
        [
            _FakePacket(),  # absorbed by step()'s own poll() before the probe's own fetch runs
            _FakePacket(is_data=True, name=content_name, payload=b"hello", chunk_num=0, end_chunk_num=0),
        ]
    )
    config = _config(
        tmp_path,
        schedule=[
            {"at_s": 0, "content_name": content_name, "zone": "zone1", "disaster": "flood", "media": "image"}
        ],
        probes=[{"at_s": 0, "zone": "zone1"}],
        receive_timeout_ms=55,
        chunk_lifetime_ms=777,
    )
    agent = build_agent(config, handle, clock=_FakeClock(0.0))
    agent.step()

    events_path = Path(config.state_dir) / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    discovered = next(e for e in lines if e["event"] == "probe_discovered")
    assert discovered["matches"] == [{"origin": "h0", "seq": 1, "content_name": content_name}]
    retrieved = next(e for e in lines if e["event"] == "probe_retrieved")
    assert (retrieved["success"], retrieved["bytes"]) == (True, 5)
    assert handle.sent_interests[-1] == (content_name, 0, 777)
    assert handle.receive_timeout_calls == [55, 55]


def test_probe_retrieval_failure_does_not_block_the_loop(tmp_path: Path):
    """A probe whose match's content never arrives still emits probe_retrieved with
    success=False and bytes=0, and the loop keeps running afterward.
    """
    content_name = "ccnx:/content/h0/body"
    clock = _FakeClock(0.0)
    handle = _FakeHandle([], clock=clock, step_s=0.2)  # auto-ticks so the chunk deadline is actually crossed
    config = _config(
        tmp_path,
        schedule=[
            {"at_s": 0, "content_name": content_name, "zone": "zone1", "disaster": "flood", "media": "image"}
        ],
        probes=[{"at_s": 0, "zone": "zone1"}],
        chunk_lifetime_ms=300,
    )
    agent = build_agent(config, handle, clock=clock)
    agent.step()

    events_path = Path(config.state_dir) / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    retrieved = next(e for e in lines if e["event"] == "probe_retrieved")
    assert (retrieved["success"], retrieved["bytes"]) == (False, 0)

    agent.step()  # the loop is still usable afterward; this must not raise


def test_probe_targets_the_lowest_origin_seq_match_not_the_highest(tmp_path: Path):
    """_run_probe fetches matches[0], and query_by_attrs itself orders its results by
    (origin, seq) ascending -- so with two equally-matching records from different
    origins, the probe must always target the lower (origin, seq) pair
    deterministically, not whichever happened to be inserted or discovered last.
    """
    content_name_h0 = "ccnx:/content/h0/body"
    content_name_h1 = "ccnx:/content/h1/body"  # never backed by any Data below
    clock = _FakeClock(0.0)
    handle = _FakeHandle(
        [
            _FakePacket(),  # absorbed by step()'s own poll()
            _FakePacket(is_data=True, name=content_name_h0, payload=b"from-h0", chunk_num=0, end_chunk_num=0),
        ],
        # Correct code resolves the fetch on the second packet above without ever
        # waiting out a deadline; a clock is still wired here (ticking, not frozen)
        # so that IF a regression picked matches[-1] (h1's unreachable content) the
        # fetch would time out cleanly instead of spinning forever against a clock
        # that can never reach its own deadline.
        clock=clock,
        step_s=0.2,
    )
    config = _config(
        tmp_path,
        schedule=[
            {"at_s": 0, "content_name": content_name_h0, "zone": "zone1", "disaster": "flood", "media": "image"}
        ],
        probes=[{"at_s": 0, "zone": "zone1"}],
    )
    agent = build_agent(config, handle, clock=clock)
    # Seed a second, received record from a *later* origin with the same zone (as if
    # h1 had synced it from a neighbor), so the probe's own query sees two matches.
    record_h1 = schema.IndexRecord(
        origin="h1", seq=1, content_name=content_name_h1, zone="zone1", disaster="flood", media="image",
        observed_at=1,
    )
    payload_h1 = schema.encode(record_h1)
    agent._store.append_received("h1", 1, payload_h1, schema.content_hash(payload_h1))

    agent.step()  # fires the h0 schedule entry (creating ("h0", 1)) and then the probe

    events_path = Path(config.state_dir) / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    discovered = next(e for e in lines if e["event"] == "probe_discovered")
    assert [(m["origin"], m["seq"]) for m in discovered["matches"]] == [("h0", 1), ("h1", 1)]
    retrieved = next(e for e in lines if e["event"] == "probe_retrieved")
    # If the probe targeted matches[-1] (h1's record) instead, this would be
    # ("h1", 1) and the fetch would fail (no Data for content_name_h1 above).
    assert (retrieved["origin"], retrieved["seq"]) == ("h0", 1)
    assert retrieved["success"] is True


def test_a_stop_request_mid_probe_fetch_ends_the_wait_promptly_and_still_shuts_down(tmp_path: Path):
    """request_stop() landing while a probe's fetch_content is mid-wait for an
    unreachable chunk must not block that one step() for the full
    chunk_lifetime_ms -- a real SIGTERM can land at exactly that point (a signal
    handler calls request_stop() between bytecode instructions, same as here),
    and host_apps only gives a process 2s between SIGTERM and SIGKILL. run()
    must then also exit and shut down (flush events, close the store) rather
    than needing a second, separate SIGTERM.
    """
    content_name = "ccnx:/content/h0/body"
    clock = _FakeClock(0.0)
    handle = _FakeHandle([], clock=clock, step_s=0.1)
    config = _config(
        tmp_path,
        schedule=[
            {"at_s": 0, "content_name": content_name, "zone": "zone1", "disaster": "flood", "media": "image"}
        ],
        probes=[{"at_s": 0, "zone": "zone1"}],
        chunk_lifetime_ms=2000,
    )
    agent = build_agent(config, handle, clock=clock)

    # Simulate a SIGTERM landing on the second receive() this step() makes:
    # the first is step()'s own poll(), the second is inside the probe's
    # fetch_content wait loop.
    real_receive = handle.receive
    calls = {"n": 0}

    def receive_then_stop(timeout_ms: int):
        calls["n"] += 1
        if calls["n"] == 2:
            agent.request_stop()
        return real_receive(timeout_ms)

    handle.receive = receive_then_stop

    agent.step()  # must return promptly, not block for chunk_lifetime_ms (2s of fake-clock ticks)
    assert clock.value < 1.0

    events_path = Path(config.state_dir) / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    retrieved = next(e for e in lines if e["event"] == "probe_retrieved")
    assert (retrieved["success"], retrieved["bytes"]) == (False, 0)

    agent.run()  # _stop_requested is already True, so this exits immediately and shuts down
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert lines[-1]["event"] == "counters"


def test_stop_at_s_abandons_an_in_progress_probe_when_the_deadline_is_crossed_mid_wait(tmp_path: Path):
    """A probe's fetch_content waiting on content that never arrives must
    notice stop_at_s mid-wait (via should_stop(), wired to the same boundary
    check as stop_at_s itself -- see CefpycoTransport.fetch_content) and
    abandon the fetch, instead of blocking for the full chunk_lifetime_ms
    (2s of fake-clock ticks here) and only being noticed on run()'s next
    step().

    The abandoned fetch is recorded as probe_retrieved(success=False, bytes=0)
    -- the same representation an ordinary unreachable-content failure already
    uses (test_probe_retrieval_failure_does_not_block_the_loop); stop_at_s
    needs no new event shape of its own.
    """
    content_name = "ccnx:/content/h0/body"
    clock = _FakeClock(0.0)
    handle = _FakeHandle([], clock=clock, step_s=0.1)
    config = _config(
        tmp_path,
        schedule=[
            {"at_s": 0, "content_name": content_name, "zone": "zone1", "disaster": "flood", "media": "image"}
        ],
        probes=[{"at_s": 0, "zone": "zone1"}],
        stop_at_s=0.5,
        chunk_lifetime_ms=2000,
    )
    agent = build_agent(config, handle, clock=clock)

    agent.step()  # schedule entry creates the record this step's own probe then discovers and fetches

    # Abandoned promptly at the deadline (0.5), nowhere near the 2.1s this
    # chunk's own Interest lifetime would otherwise have run to.
    assert clock.value == 0.5
    assert agent._stop_requested

    events_path = Path(config.state_dir) / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    retrieved = next(e for e in lines if e["event"] == "probe_retrieved")
    assert (retrieved["success"], retrieved["bytes"]) == (False, 0)

    agent.run()  # _stop_requested is already True, so this exits immediately and still shuts down cleanly
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert lines[-1]["event"] == "counters"


def test_a_counters_snapshot_due_mid_probe_never_splits_its_discovery_from_its_retrieval(tmp_path: Path):
    """should_stop (CefpycoTransport.fetch_content's own boundary check,
    Agent._on_receive_boundary) does capture a counters_snapshot_at_s entry
    that becomes due while a probe's fetch_content is blocked waiting on
    content that never arrives -- but only ever queues it in memory
    (Agent._pending_snapshots); the write itself waits for a receive
    boundary outside that probe's own call stack. A probe's fetch_content
    runs synchronously inside _run_probe, strictly between that probe's own
    probe_discovered and probe_retrieved writes, and check.py's own
    retrieval-pairing check depends on nothing else from this node being
    logged in between (see check.py's own comment on retrieved_after) -- a
    counters_snapshot line written from inside that wait would break it.
    """
    content_name = "ccnx:/content/h0/body"
    clock = _FakeClock(0.0)
    handle = _FakeHandle([], clock=clock, step_s=0.1)
    config = _config(
        tmp_path,
        schedule=[
            {"at_s": 0, "content_name": content_name, "zone": "zone1", "disaster": "flood", "media": "image"}
        ],
        probes=[{"at_s": 0, "zone": "zone1"}],
        counters_snapshot_at_s=[0.25],  # falls strictly inside the probe's own fetch_content wait below
        chunk_lifetime_ms=500,
    )
    agent = build_agent(config, handle, clock=clock)

    agent.step()  # schedule entry, then the probe: its fetch_content's wait captures the 0.25 entry mid-call

    events_path = Path(config.state_dir) / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    kinds = [e["event"] for e in lines]
    discovered_i = kinds.index("probe_discovered")
    retrieved_i = kinds.index("probe_retrieved")
    assert retrieved_i == discovered_i + 1, f"something else was logged in between: {kinds}"
    # Captured mid-fetch, but not lost -- written out right after
    # drain_probes returns, still within this same step, never before the
    # probe's own probe_retrieved.
    assert kinds.count("counters_snapshot") == 1
    assert kinds.index("counters_snapshot") > retrieved_i


def test_a_counters_snapshot_captured_mid_probe_excludes_that_probes_own_chunk_once_it_arrives(
    tmp_path: Path,
):
    """Pins down *where exactly* inside fetch_content's wait loop the capture
    happens: before the matching chunk is counted, not after. Two blank
    packets (absorbed by step()'s own poll() and fetch_content's first
    receive() call) put the chunk's own Data on the very receive() call
    whose tick crosses elapsed 0.25 (step_s=0.1) -- the capture and the
    chunk's own arrival happen on the *same* should_stop()/receive() call.
    If the post-receive should_stop() check inside fetch_content's wait
    loop were ever moved below the `bytes_received +=` line (or capture
    happened only outside fetch_content, never from inside it), this
    chunk's bytes would already be counted by the time the snapshot
    captures, and the first assert below would fail.
    """
    content_name = "ccnx:/content/h0/body"
    clock = _FakeClock(0.0)
    payload = b"a probe's own chunk 0 payload"
    handle = _FakeHandle(
        [
            _FakePacket(),  # absorbed by step()'s own poll()
            _FakePacket(),  # fetch_content's receive() #1
            # fetch_content's receive() #2 -- this tick (0.2 -> 0.3) crosses
            # elapsed 0.25, on the exact same call that delivers this chunk.
            _FakePacket(is_data=True, name=content_name, payload=payload, chunk_num=0, end_chunk_num=0),
        ],
        clock=clock,
        step_s=0.1,
    )
    config = _config(
        tmp_path,
        schedule=[
            {"at_s": 0, "content_name": content_name, "zone": "zone1", "disaster": "flood", "media": "image"}
        ],
        probes=[{"at_s": 0, "zone": "zone1"}],
        counters_snapshot_at_s=[0.25],
        chunk_lifetime_ms=2000,
    )
    agent = build_agent(config, handle, clock=clock)

    agent.step()

    events_path = Path(config.state_dir) / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    kinds = [e["event"] for e in lines]
    discovered_i = kinds.index("probe_discovered")
    retrieved_i = kinds.index("probe_retrieved")
    assert retrieved_i == discovered_i + 1
    retrieved = lines[retrieved_i]
    assert retrieved["success"] is True

    snapshot = next(e for e in lines if e["event"] == "counters_snapshot")
    assert snapshot["bytes_received"] == 0  # the chunk hadn't arrived yet when this was captured
    assert agent._transport.bytes_received > 0  # but it has, by the time this step ends


def test_a_counters_snapshot_captured_mid_probe_is_not_lost_when_that_probe_is_what_triggers_stop(
    tmp_path: Path,
):
    """The scenario that defeats a naive 'defer the write to the next step'
    fix: when the very probe whose fetch_content captured a due
    counters_snapshot is also what notices stop_at_s has been reached
    (should_stop, Agent._on_receive_boundary, sets the flag mid-wait), there
    is no next step -- run() sees _stop_requested and goes straight to
    shutdown(). The captured entry must still be written before shutdown's
    own `counters` line, not dropped for want of a later step to flush it.
    """
    content_name = "ccnx:/content/h0/body"
    clock = _FakeClock(0.0)
    handle = _FakeHandle([], clock=clock, step_s=0.1)
    config = _config(
        tmp_path,
        schedule=[
            {"at_s": 0, "content_name": content_name, "zone": "zone1", "disaster": "flood", "media": "image"}
        ],
        probes=[{"at_s": 0, "zone": "zone1"}],
        counters_snapshot_at_s=[0.5],
        stop_at_s=0.5,  # same instant: the probe's own should_stop check is what notices both
        chunk_lifetime_ms=2000,
    )
    agent = build_agent(config, handle, clock=clock)

    agent.run()  # the probe's fetch_content both captures the snapshot and triggers the stop, mid-wait

    events_path = Path(config.state_dir) / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    kinds = [e["event"] for e in lines]
    assert kinds.count("counters_snapshot") == 1, f"the entry was lost: {kinds}"
    assert kinds.index("counters_snapshot") < kinds.index("counters")


def test_run_exits_and_shuts_down_cleanly_once_stop_is_requested(tmp_path: Path):
    """run() exits its loop and shuts down (flush remaining events, write the counters
    line, close the store) once request_stop() has been called -- the SIGTERM path,
    minus the signal delivery itself (see test_install_signal_handlers_requests_a_stop).
    """
    handle = _FakeHandle()
    config = _config(tmp_path)
    agent = build_agent(config, handle, clock=_FakeClock(0.0))
    agent.request_stop()
    agent.run()  # must return, not loop forever

    events_path = Path(config.state_dir) / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert lines[-1]["event"] == "counters"
    assert lines[-1]["bytes_sent"] == 0
    assert lines[-1]["bytes_received"] == 0
    assert lines[-1]["control_bytes_sent"] == 0
    assert lines[-1]["control_bytes_received"] == 0

    # The store connection is closed: a further call must raise, proving close() really ran.
    with pytest.raises(sqlite3.ProgrammingError):
        agent._store.get_progress("h0")


def test_install_signal_handlers_requests_a_stop_without_raising(tmp_path: Path):
    """SIGTERM calls request_stop() rather than the default terminate-the-process behavior.

    Restores pytest's own prior SIGTERM/SIGINT handlers afterward (captured via
    signal.getsignal, not assumed to be SIG_DFL): leaving SIGINT on the OS default
    would make a later Ctrl-C raise KeyboardInterrupt instead of pytest's own handling.
    """
    import os

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigint = signal.getsignal(signal.SIGINT)
    handle = _FakeHandle()
    config = _config(tmp_path)
    agent = build_agent(config, handle, clock=_FakeClock(0.0))
    install_signal_handlers(agent)
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        assert agent._stop_requested is True
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)


def test_poll_jitter_rng_is_deterministic_per_node_and_differs_across_nodes():
    """The same node name reproduces the exact same jitter sequence every time; a
    different node name draws a different one -- map.md "2 秒 ±20%" needs real
    per-neighbor jitter, not two hosts polling in exact lockstep, but a run must
    still be reproducible from the node names alone.
    """
    first_h0 = [_poll_jitter_rng("h0")() for _ in range(5)]
    second_h0 = [_poll_jitter_rng("h0")() for _ in range(5)]
    assert first_h0 == second_h0

    h1 = [_poll_jitter_rng("h1")() for _ in range(5)]
    assert first_h0 != h1


def test_build_agent_seeds_the_engine_s_poll_jitter_from_the_node_name(tmp_path: Path):
    """build_agent must not leave the engine's poll_jitter_rng at SyncEngine's own
    zero-jitter default (lambda: 0.5), or every agent would poll on the exact same grid.
    """
    handle = _FakeHandle()
    config = _config(tmp_path)
    agent = build_agent(config, handle, clock=_FakeClock(0.0))
    assert agent._engine._poll_jitter_rng() == random.Random(config.node).random()


# -- state-poll interval choices (stage-2 map "行動と確率") ---------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"state_poll_interval_choices_s": []}, "state_poll_interval_choices_s must be a non-empty list"),
        ({"state_poll_interval_choices_s": 2.0}, "state_poll_interval_choices_s must be a non-empty list"),
        ({"state_poll_interval_choices_s": [1, 0]}, "state_poll_interval_choices_s[1] must be > 0"),
        ({"state_poll_interval_choices_s": [1, True]}, "state_poll_interval_choices_s[1] must be a number"),
        ({"state_poll_interval_choices_s": [1, 2, 1.0]}, "state_poll_interval_choices_s must not contain duplicates"),
        (
            {"state_poll_interval_choices_s": [1, 2], "state_poll_jitter_frac": 0.2},
            "state_poll_interval_choices_s cannot be combined with state_poll_jitter_frac",
        ),
        ({"state_poll_interval_choices_s": [1, 2]}, "state_poll_interval_choices_s requires seed"),
        ({"state_poll_interval_choices_s": [1, 2], "seed": "7"}, "seed must be an integer"),
    ],
)
def test_validate_config_rejects_a_bad_state_poll_interval_choices_s(overrides: dict, message: str):
    """Each malformed choices/seed combination fails with its own exact diagnostic.
    Duplicates matter because the engine records action_prob = 1/len(choices), which
    is only the true probability when every value appears once; a missing seed matters
    because the choice sequence must be reproducible from the run's own seed."""
    raw = _minimal_raw()
    raw.update(overrides)
    with pytest.raises(ConfigError) as exc_info:
        validate_config(raw)
    assert str(exc_info.value) == message


def test_validate_config_carries_choices_as_floats_and_the_seed():
    """Valid choices reach SyncConfig as a float tuple (so interval_s serializes as
    e.g. 2.0, like stage 1's), and the seed reaches the AgentConfig."""
    raw = _minimal_raw()
    raw["state_poll_interval_choices_s"] = [1, 2, 4.5]
    raw["seed"] = 7
    config = validate_config(raw)
    assert config.seed == 7
    assert build_sync_config(config).state_poll_interval_choices_s == (1.0, 2.0, 4.5)


def test_validate_config_leaves_choices_unset_and_seed_none_by_default():
    """A stage-1 config (no choices, no seed) keeps the jittered policy."""
    config = validate_config(_minimal_raw())
    assert config.seed is None
    assert build_sync_config(config).state_poll_interval_choices_s is None


def test_poll_choice_rng_is_deterministic_per_seed_and_node():
    """The same (seed, node) reproduces the same draws; changing either the seed or
    the node draws a different sequence -- unlike _poll_jitter_rng, a seed sweep
    really does vary the poll actions."""
    first = [_poll_choice_rng(7, "h0")() for _ in range(5)]
    assert first == [_poll_choice_rng(7, "h0")() for _ in range(5)]
    assert first != [_poll_choice_rng(8, "h0")() for _ in range(5)]
    assert first != [_poll_choice_rng(7, "h1")() for _ in range(5)]


def _chosen_intervals(tmp_path: Path, seed: int, node: str = "h0") -> list[tuple[float, float]]:
    """Build one agent with choices (1, 2, 4) and `seed`, tick its engine through 30
    polls to h1, and return each state_poll decision's (interval_s, action_prob)."""
    config = _config(
        tmp_path / f"{node}-seed{seed}",
        node=node,
        neighbors=["h1"] if node != "h1" else ["h0"],
        state_poll_interval_choices_s=[1, 2, 4],
        seed=seed,
    )
    agent = build_agent(config, _FakeHandle(), clock=_FakeClock(0.0))
    engine = agent._engine
    rows: list[tuple[float, float]] = []
    now = 0.0
    while len(rows) < 30:
        engine.on_tick(now)
        for event in engine.drain_events():
            if event["event"] == "decision" and event["kind"] == "state_poll":
                rows.append((event["action"]["interval_s"], event["action_prob"]))
                now += event["action"]["interval_s"]
    return rows


def test_build_agent_draws_choices_from_the_run_seed_and_node(tmp_path: Path):
    """End to end through build_agent: every chosen interval is one of the choices
    with action_prob exactly 1/3; the same seed and node reproduce the sequence and a
    different seed changes it."""
    rows = _chosen_intervals(tmp_path, seed=7)
    assert {interval for interval, _ in rows} <= {1.0, 2.0, 4.0}
    assert {prob for _, prob in rows} == {1 / 3}
    assert rows == _chosen_intervals(tmp_path / "again", seed=7)
    assert rows != _chosen_intervals(tmp_path, seed=8)


def test_two_agents_converge_their_stores_over_cross_wired_fake_handles(tmp_path: Path):
    """Two agents, each driven by its own fake handle that this test manually bridges to
    the other's, converge: h0's own record reaches h1 and h1's reaches h0, through the
    real engine, store and CefpycoTransport end to end on both sides -- not just one leg
    of the round trip tested in isolation, as the other tests in this file do.
    """
    clock = _FakeClock(0.0)
    handle0 = _FakeHandle()
    handle1 = _FakeHandle()
    config0 = _config(
        tmp_path / "h0",
        node="h0",
        neighbors=["h1"],
        schedule=[
            {"at_s": 0, "content_name": "ccnx:/content/h0/a", "zone": "zone1", "disaster": "flood", "media": "image"}
        ],
    )
    config1 = _config(
        tmp_path / "h1",
        node="h1",
        neighbors=["h0"],
        schedule=[
            {"at_s": 0, "content_name": "ccnx:/content/h1/a", "zone": "zone1", "disaster": "flood", "media": "image"}
        ],
    )
    agent0 = build_agent(config0, handle0, clock=clock)
    agent1 = build_agent(config1, handle1, clock=clock)

    def _bridge(sender: _FakeHandle, receiver: _FakeHandle) -> None:
        """Deliver everything `sender` has sent since the last bridge call into `receiver`'s queue."""
        while sender.sent_interests:
            name, _chunk_num, _lifetime = sender.sent_interests.pop(0)
            receiver.packets.append(_FakePacket(is_interest=True, name=name))
        while sender.sent_data:
            name, payload, _chunk_num, _expiry, _cache_time = sender.sent_data.pop(0)
            receiver.packets.append(_FakePacket(is_data=True, name=name, payload=payload))

    for _round in range(10):
        # Drain each side's queue fully before bridging again: step() reads at most one
        # packet per call, so a round that only steps once per side would let the queue
        # grow without bound instead of ever catching up to what the other side just sent.
        while handle0.packets:
            agent0.step()
        while handle1.packets:
            agent1.step()
        agent0.step()  # also let on_tick fire once with an empty queue, to plan new fetches
        agent1.step()
        _bridge(handle0, handle1)
        _bridge(handle1, handle0)
        clock.value += 3.0  # past state_poll_interval_s even at its full +20% jitter

    assert agent0._store.get_progress("h1").hi == 1
    assert agent1._store.get_progress("h0").hi == 1
    assert agent0._store.get_record("h1", 1) == agent1._store.get_record("h1", 1)
    assert agent1._store.get_record("h0", 1) == agent0._store.get_record("h0", 1)


# -- restart re-fire: a second life on the same state_dir must not redo the first's work ---


def test_a_second_life_on_the_same_state_dir_creates_no_duplicate_records_and_reprobes_nothing_past(
    tmp_path: Path,
):
    """Two lives of the same node, same config, same state_dir (ReCefore's host_apps
    gives no other way to stop/restart one host app than SIGTERM then relaunch): the
    first life runs two schedule entries and one probe, then stops (shutdown(), the
    same flush+close a real SIGTERM's run() does); a second life, built fresh against
    the same state_dir and a later clock, must not recreate the two already-made
    records and must not re-emit the already-run probe -- but it must still run the
    schedule entry and probe that are genuinely new in this life.
    """
    clock = _FakeClock(0.0)
    config = _config(
        tmp_path,
        schedule=[
            {"at_s": 1, "content_name": "ccnx:/content/h0/a", "zone": "zone1", "disaster": "flood", "media": "image"},
            {"at_s": 2, "content_name": "ccnx:/content/h0/b", "zone": "zone1", "disaster": "flood", "media": "image"},
            {"at_s": 10, "content_name": "ccnx:/content/h0/c", "zone": "zone1", "disaster": "flood", "media": "image"},
        ],
        probes=[{"at_s": 1, "zone": "zone1"}, {"at_s": 10, "zone": "zone1"}],
    )

    # First life: elapsed reaches 3s, so only the at_s=1 and at_s=2 schedule entries
    # and the at_s=1 probe are due; at_s=10 entries are not reached yet. The probe's
    # own fetch_content has nothing to retrieve (no Data queued), so the handle's
    # clock must auto-tick (as in the probe-failure tests above) or that wait loop's
    # deadline, compared against an otherwise-frozen clock, would never be crossed.
    handle1 = _FakeHandle([], clock=clock, step_s=0.2)
    agent1 = build_agent(config, handle1, clock=clock)
    clock.value = 3.0
    agent1.step()
    agent1.shutdown()  # same flush+close a real SIGTERM's run() would have done

    events_path = Path(config.state_dir) / "events.jsonl"
    first_life = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [(e["origin"], e["seq"]) for e in first_life if e["event"] == "record_created"] == [
        ("h0", 1),
        ("h0", 2),
    ]
    assert len([e for e in first_life if e["event"] == "probe_discovered"]) == 1

    # Second life: a fresh Agent/CefpycoTransport/EventWriter (a real relaunch would
    # reopen events.jsonl too) built against the *same* state_dir, with elapsed
    # already at 20s by the time it first steps -- every at_s <= 20 entry would be
    # "due" if nothing were skipped, including the two already fired in the first life.
    handle2 = _FakeHandle([], clock=clock, step_s=0.2)
    agent2 = build_agent(config, handle2, clock=clock)
    clock.value = 20.0
    agent2.step()

    second_life = [json.loads(line) for line in events_path.read_text().splitlines()][len(first_life) :]
    created_second = [e for e in second_life if e["event"] == "record_created"]
    # Only the at_s=10 entry ("c", seq 3) is new in this life; seq 1/2 must not reappear.
    assert [(e["origin"], e["seq"]) for e in created_second] == [("h0", 3)]
    assert agent2._store.get_progress("h0").hi == 3  # 2 from the first life + 1 new, never 4+

    discovered_second = [e for e in second_life if e["event"] == "probe_discovered"]
    # Only the at_s=10 probe is new; the at_s=1 probe must not re-fire.
    assert len(discovered_second) == 1


def test_stop_at_s_ends_the_loop_exactly_as_sigterm_would_after_the_last_due_step(tmp_path: Path):
    """Once elapsed time reaches stop_at_s, run() stops -- same final-counters-then-
    close shutdown a real SIGTERM's run() loop does.

    state_poll_interval_s=0.3 (jitter_frac=0, so exactly 0.3, not +/-20%) and the
    handle's clock advancing 0.3s per poll() put a state-poll due at exactly
    0.3 and another at exactly 0.6 -- the same instant stop_at_s=0.6 is reached.
    Step 1 (pre-poll elapsed 0.0, < 0.6) polls to 0.3 and sends the first poll.
    Step 2 (pre-poll elapsed 0.3, still < 0.6, so poll() is still allowed to
    start) polls to 0.6 -- but unlike a real SIGTERM, which would let a step
    already in progress finish, stop_at_s is a measurement-window boundary:
    poll()'s own should_stop() check (right after its receive() returns, see
    CefpycoTransport.poll()) notices 0.6 has been reached immediately, before
    dispatch/on_tick ever run for this step, so this step sends nothing at
    all. Only step 1's Interest ever goes out.
    """
    clock = _FakeClock(0.0)
    handle = _FakeHandle([], clock=clock, step_s=0.3)
    config = _config(
        tmp_path, neighbors=["h1"], stop_at_s=0.6, state_poll_interval_s=0.3, state_poll_jitter_frac=0.0
    )
    agent = build_agent(config, handle, clock=clock)

    agent.run()

    # Two receive() calls happened (step 1's and step 2's -- the second is
    # what carries the clock to 0.6 and is itself what notices the
    # deadline), but a third never does: that would have ticked the clock to
    # 0.9 and sent a third Interest once the next 0.3s interval elapsed.
    assert len(handle.receive_timeout_calls) == 2
    assert clock.value == 0.6

    # Only step 1's poll went out; step 2's own poll is held back by the
    # deadline it crossed getting there -- this is exactly the leak the
    # 2026-10-23 fix closes (see _on_receive_boundary's docstring).
    assert len(handle.sent_interests) == 1
    sent_bytes = sum(len(name.encode("utf-8")) for name, _, _ in handle.sent_interests)

    events_path = Path(config.state_dir) / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    counters = [e for e in lines if e["event"] == "counters"]
    assert len(counters) == 1
    assert (
        counters[0]["bytes_sent"],
        counters[0]["bytes_received"],
        counters[0]["control_bytes_sent"],
        counters[0]["control_bytes_received"],
    ) == (sent_bytes, 0, sent_bytes, 0)


def test_stop_at_s_sends_no_bytes_after_the_deadline_when_a_receive_wait_crosses_it(tmp_path: Path):
    """A receive wait that straddles stop_at_s (starts before it, returns
    after it) must not let this step go on to tick and send -- even though
    the step itself started while still before the deadline.

    state_poll_interval_s=0.2 (no jitter) and step_s=0.3 on the fake handle
    put a state-poll due at elapsed 0.2 and another at 0.4, with stop_at_s=0.5
    landing strictly between them. Step 1's poll() carries the clock from 0.0
    to 0.3 (not yet past 0.5), ticks, and sends the first Interest. Step 2's
    poll() carries the clock from 0.3 to 0.6 -- past 0.5 -- and under the
    pre-fix code this step still ticked at elapsed 0.6 and sent the second
    poll anyway. Every Interest's own send time is recorded (the clock
    reading at the moment send_interest is actually called) to pin that none
    of them lands at or after the deadline, not just that the final count is
    right.
    """
    clock = _FakeClock(0.0)
    sent_at: list[float] = []

    class _TimestampingHandle(_FakeHandle):
        def send_interest(self, name: str, chunk_num: int, lifetime: int) -> None:
            sent_at.append(clock.value)
            super().send_interest(name, chunk_num, lifetime)

    handle = _TimestampingHandle([], clock=clock, step_s=0.3)
    config = _config(
        tmp_path, neighbors=["h1"], stop_at_s=0.5, state_poll_interval_s=0.2, state_poll_jitter_frac=0.0
    )
    agent = build_agent(config, handle, clock=clock)

    agent.step()  # clock 0.0 -> 0.3, ticks and sends the first poll (due at 0.2)
    assert not agent._stop_requested
    agent.step()  # clock 0.3 -> 0.6, crossing stop_at_s=0.5 inside this step's own receive wait

    assert clock.value == 0.6
    assert agent._stop_requested
    assert sent_at == [0.3]
    assert all(t < 0.5 for t in sent_at), f"an Interest was sent at or after the deadline: {sent_at}"


def test_a_relaunched_life_with_stop_at_s_already_past_stops_immediately(tmp_path: Path):
    """A relaunched life (same config, same state_dir, ReCefore's host_apps contract
    giving no other way to stop/start one host app than SIGTERM then relaunch)
    whose clock is already past t0+stop_at_s on its very first step must not poll
    or tick at all -- just write its own `counters` line (all-zero: nothing was
    ever sent or received in this life) and stop.
    """
    clock = _FakeClock(0.0)
    config = _config(tmp_path, neighbors=["h1"], stop_at_s=100.0)

    handle1 = _FakeHandle()
    agent1 = build_agent(config, handle1, clock=clock)
    clock.value = 1.0  # well before stop_at_s=100
    agent1.step()
    agent1.shutdown()

    events_path = Path(config.state_dir) / "events.jsonl"
    first_life = [json.loads(line) for line in events_path.read_text().splitlines()]

    clock.value = 150.0  # already past t0+stop_at_s=100 before life 2 is even built
    handle2 = _FakeHandle()
    agent2 = build_agent(config, handle2, clock=clock)
    agent2.run()

    assert handle2.sent_interests == []
    second_life = [json.loads(line) for line in events_path.read_text().splitlines()][len(first_life) :]
    # agent_start (written at construction, life 2's own) then counters
    # (written at shutdown) -- nothing else, since this life never polled
    # or ticked at all.
    assert [e["event"] for e in second_life] == ["agent_start", "counters"]
    counters = second_life[1]
    assert (
        counters["bytes_sent"],
        counters["bytes_received"],
        counters["control_bytes_sent"],
        counters["control_bytes_received"],
    ) == (0, 0, 0, 0)


def test_counters_snapshot_fires_once_per_entry_when_its_time_comes(tmp_path: Path):
    """Each counters_snapshot_at_s entry emits exactly one counters_snapshot event,
    not before its own at_s and not more than once."""
    clock = _FakeClock(0.0)
    handle = _FakeHandle()  # no neighbors, no clock wiring: fully manual timing
    config = _config(tmp_path, counters_snapshot_at_s=[0.5, 1.0])
    agent = build_agent(config, handle, clock=clock)
    events_path = Path(config.state_dir) / "events.jsonl"

    clock.value = 0.3
    agent.step()
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [e["event"] for e in lines] == ["agent_start"]  # neither entry is due yet

    clock.value = 0.5
    agent.step()
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [e["event"] for e in lines] == ["agent_start", "counters_snapshot"]

    clock.value = 1.0
    agent.step()
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [e["event"] for e in lines] == ["agent_start", "counters_snapshot", "counters_snapshot"]


def test_counters_snapshot_reports_this_lifes_current_byte_counters(tmp_path: Path):
    """A counters_snapshot event's four byte fields are this life's own transport
    counters as of the end of the previous step (see _drain_counters_snapshots'
    own docstring for why not this step's own tick) -- not placeholders, and not
    the counters at shutdown.
    """
    clock = _FakeClock(0.0)
    handle = _FakeHandle()
    config = _config(tmp_path, neighbors=["h1"], counters_snapshot_at_s=[1.0])
    agent = build_agent(config, handle, clock=clock)
    events_path = Path(config.state_dir) / "events.jsonl"

    clock.value = 0.0
    agent.step()  # sends the first state-request Interest to h1 (due immediately)
    bytes_sent_so_far = agent._transport.bytes_sent
    assert bytes_sent_so_far > 0

    clock.value = 1.0
    agent.step()  # counters_snapshot_at_s[0]=1.0 fires, drained before this step's own tick
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    snapshots = [e for e in lines if e["event"] == "counters_snapshot"]
    assert len(snapshots) == 1
    assert (
        snapshots[0]["bytes_sent"],
        snapshots[0]["bytes_received"],
        snapshots[0]["control_bytes_sent"],
        snapshots[0]["control_bytes_received"],
    ) == (bytes_sent_so_far, 0, bytes_sent_so_far, 0)
    assert snapshots[0]["ts_ms"] == 1000


def test_counters_snapshot_excludes_a_send_whose_receive_wait_crossed_the_snapshot_time(tmp_path: Path):
    """A receive wait that starts before counters_snapshot_at_s but returns
    after it must not let the snapshot include bytes this same step's own
    on_tick is about to send.

    state_poll_interval_s=0.2 (no jitter) and step_s=0.3 on the fake handle
    put a state-poll due at elapsed 0.2 and another at 0.5, with the snapshot
    at 0.5 landing exactly on that second poll's own due time. Step 1's
    poll() carries the clock from 0.0 to 0.3 (not yet at 0.5) and sends the
    first Interest. Step 2's poll() carries the clock from 0.3 to 0.6,
    crossing 0.5 -- the pre-fix code only drained the snapshot against the
    pre-poll `now` (0.3, not yet due) and would not have taken it until the
    *next* step, by which point step 2's own second send had already landed
    in the counters it reports.
    """
    clock = _FakeClock(0.0)
    handle = _FakeHandle([], clock=clock, step_s=0.3)
    config = _config(
        tmp_path, neighbors=["h1"], counters_snapshot_at_s=[0.5], state_poll_interval_s=0.2,
        state_poll_jitter_frac=0.0,
    )
    agent = build_agent(config, handle, clock=clock)

    agent.step()  # clock 0.0 -> 0.3, sends the first poll (due at 0.2)
    bytes_sent_after_step1 = agent._transport.bytes_sent
    assert bytes_sent_after_step1 > 0

    agent.step()  # clock 0.3 -> 0.6, crossing the 0.5 snapshot time inside this step's own receive wait

    events_path = Path(config.state_dir) / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    snapshots = [e for e in lines if e["event"] == "counters_snapshot"]
    assert len(snapshots) == 1
    # The snapshot reports step 1's bytes only -- not step 2's own second send,
    # which happens after the snapshot is taken (see on the next assert).
    assert snapshots[0]["bytes_sent"] == bytes_sent_after_step1
    assert agent._transport.bytes_sent > bytes_sent_after_step1


def test_a_counters_snapshot_entry_exactly_at_stop_at_s_still_fires_before_the_stop(tmp_path: Path):
    """A counters_snapshot_at_s entry due at exactly the same instant as
    stop_at_s still fires on the step that notices the deadline, and its four
    byte fields exactly match the final `counters` line shutdown() writes --
    nothing more is ever sent once _on_receive_boundary has drained the
    snapshot and found the deadline reached in the same call, so the two
    lines describe the identical state.
    """
    clock = _FakeClock(0.0)
    handle = _FakeHandle([], clock=clock, step_s=0.3)
    config = _config(
        tmp_path, neighbors=["h1"], stop_at_s=0.5, counters_snapshot_at_s=[0.5],
        state_poll_interval_s=0.2, state_poll_jitter_frac=0.0,
    )
    agent = build_agent(config, handle, clock=clock)

    agent.run()  # step 1 polls to 0.3 and sends; step 2's poll crosses 0.5, snapshotting then stopping

    events_path = Path(config.state_dir) / "events.jsonl"
    all_lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    # Step 1's on_tick also emits its own "decision" event (it decides to poll
    # h1's state) -- irrelevant to this test, which only cares about the two
    # counters-shaped lines and their relative order.
    lines = [e for e in all_lines if e["event"] in ("counters_snapshot", "counters")]
    assert [e["event"] for e in lines] == ["counters_snapshot", "counters"]
    snapshot, counters = lines
    fields = ("bytes_sent", "bytes_received", "control_bytes_sent", "control_bytes_received")
    assert tuple(snapshot[f] for f in fields) == tuple(counters[f] for f in fields)
    assert snapshot["bytes_sent"] > 0  # step 1's send did happen, so this pins down a non-trivial match
    # Both lines matching is not enough on its own -- the pre-fix code also
    # ended up with snapshot == counters (just one step late, after *both*
    # sends). Only step 1's Interest (due at 0.2) must ever have gone out;
    # step 2's own poll (due at 0.5, the same instant as the deadline) must
    # never have been sent at all.
    assert len(handle.sent_interests) == 1


def test_a_counters_snapshot_entry_exactly_at_a_fresh_lifes_start_still_fires(tmp_path: Path):
    """An entry whose at_s exactly equals elapsed time at construction is not
    'already past' (strict '<', not '<=' -- see Agent.__init__'s own comment): it
    is still due on this life's very first step, same as in an uninterrupted run.
    """
    clock = _FakeClock(2.0)
    config = _config(tmp_path, t0_ms=2000, counters_snapshot_at_s=[0.0])  # elapsed at construction: 0.0
    handle = _FakeHandle()
    agent = build_agent(config, handle, clock=clock)
    agent.step()
    events_path = Path(config.state_dir) / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [e["event"] for e in lines] == ["agent_start", "counters_snapshot"]


def test_a_restarted_life_skips_counters_snapshot_entries_already_past(tmp_path: Path):
    """counters_snapshot_at_s has no store-backed progress cursor (byte counters are
    process-local, reset to 0 each life): an entry whose at_s has already elapsed
    by the time a relaunched life first constructs its Agent must never fire at
    all in that life -- re-emitting it would report this life's own
    fresh-from-zero counters under a timestamp that belongs to a different life
    entirely.
    """
    clock = _FakeClock(0.0)
    config = _config(tmp_path, counters_snapshot_at_s=[1.0, 5.0])

    handle1 = _FakeHandle()
    agent1 = build_agent(config, handle1, clock=clock)
    clock.value = 1.0
    agent1.step()  # the at_s=1.0 entry fires in life 1
    agent1.shutdown()

    events_path = Path(config.state_dir) / "events.jsonl"
    first_life = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [e["event"] for e in first_life] == ["agent_start", "counters_snapshot", "counters"]

    # Second life: the clock is already at 20s (past both entries) *before* this
    # life's Agent is even constructed -- the filter must see this life's own
    # start time, not life 1's.
    clock.value = 20.0
    handle2 = _FakeHandle()
    agent2 = build_agent(config, handle2, clock=clock)
    agent2.step()

    second_life = [json.loads(line) for line in events_path.read_text().splitlines()][len(first_life) :]
    # Life 2's own agent_start (unconditional, at construction) -- nothing
    # else, since both counters_snapshot entries are already past by then.
    assert [e["event"] for e in second_life] == ["agent_start"]


def test_build_agent_fails_at_startup_when_the_schedule_is_shorter_than_the_persisted_own_record_count(
    tmp_path: Path,
):
    """A state_dir already holding more own records than the given config's schedule
    has entries is not a timing question -- it is the config no longer matching the
    state_dir it was paired with (e.g. the wrong config file after an edit). This
    must fail at startup with an exact diagnostic, not silently run a truncated
    schedule or crash later on a negative slice doing something unintended.
    """
    import os

    config = _config(
        tmp_path,
        schedule=[
            {"at_s": 1, "content_name": "ccnx:/content/h0/a", "zone": "zone1", "disaster": "flood", "media": "image"}
        ],
    )
    os.makedirs(config.state_dir, exist_ok=True)
    pre_seeded = Store(Path(config.state_dir) / "index.db")
    pre_seeded.append_own(
        origin="h0", content_name="x", zone="zone1", disaster="flood", media="image", observed_at=1
    )
    pre_seeded.append_own(
        origin="h0", content_name="y", zone="zone1", disaster="flood", media="image", observed_at=2
    )
    pre_seeded.close()

    handle = _FakeHandle()
    with pytest.raises(ConfigError) as exc_info:
        build_agent(config, handle, clock=_FakeClock(0.0))
    assert str(exc_info.value) == "h0: state_dir already holds 2 own record(s) but schedule has only 1 entries"


# -- 2026-10-23 regression (external review, P1 #1): engine event ts_ms is the
# processing time, not the flush time --------------------------------------


def _run_store_during_probe_wait(tmp_path: Path):
    """Build and run one step() of a real h0 agent that, mid-probe-fetch_content wait,
    receives and stores a record fetched from neighbor h1 -- the exact shape the
    external review's repro used (SyncEngine.on_record queues record_stored while
    this node's own probe is blocked in a synchronous content fetch, and that queued
    event is not flushed to events.jsonl until the probe returns).

    The clock starts at 129.7s (chosen so the in-flight record's own processing time
    lands at 129.9s -- inside evalgen's stage-1 [10s, 130s] measurement window -- while
    the probe's own wait only gives up once the clock reaches 130.3s, past the window's
    close) specifically so the two tests built on this scenario (ts_ms correctness, and
    evalagg non-censoring) both exercise the exact boundary the review's repro crossed.

    Returns (config, agent) after exactly one agent.step() call.
    """
    content_name = "ccnx:/content/h0/body"
    clock = _FakeClock(129.7)
    record = schema.IndexRecord(
        origin="h1", seq=1, content_name="ccnx:/content/h1/1", zone="zone1", disaster="flood", media="image",
        observed_at=1,
    )
    payload = schema.encode(record)
    fetch_name = names.build_record_name("g", "h1", "h1", 1)
    handle = _FakeHandle(
        [
            _FakePacket(),  # absorbed by step()'s own poll()
            # Arrives on fetch_content's own first receive() call, strictly
            # inside the probe's wait, at clock 129.9s: this is the record
            # this test is about.
            _FakePacket(is_data=True, name=fetch_name, payload=payload),
            _FakePacket(),  # fetch_content receive() #2 -- nothing
            _FakePacket(),  # fetch_content receive() #3 -- nothing
            _FakePacket(),  # fetch_content receive() #4 -- the chunk_lifetime_ms deadline is reached here
        ],
        clock=clock,
        step_s=0.1,
    )
    config = _config(
        tmp_path,
        neighbors=["h1"],
        schedule=[
            {"at_s": 0, "content_name": content_name, "zone": "zone1", "disaster": "flood", "media": "image"}
        ],
        probes=[{"at_s": 0, "zone": "zone1"}],
        chunk_lifetime_ms=400,
    )
    agent = build_agent(config, handle, clock=clock)
    # Preload the tracker with h1 advertising (h1, 1) as missing -- the same
    # shortcut test_engine.py's own on_tick/decision tests use to put a
    # fetch in flight without first driving a real state_rx round trip.
    agent._engine._tracker.observe_state("h1", "h1", hi=1, extra=())

    agent.step()  # on_tick sends the fetch; the probe's own fetch_content blocks through it
    return config, agent


def test_record_stored_during_a_probes_content_wait_gets_its_own_processing_time(tmp_path: Path):
    """The record fetched from h1 while this node's probe is blocked in fetch_content
    must log ts_ms = 129900 (the clock reading when SyncEngine.on_record actually ran),
    never 130300 (the clock reading once fetch_content gives up and this step's
    _drain_engine_events() finally flushes the queued event) -- the external review's
    exact repro (its own numbers: a store at 125.1s logged as 127s).
    """
    config, _agent = _run_store_during_probe_wait(tmp_path)
    events_path = Path(config.state_dir) / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    stored = next(e for e in lines if e["event"] == "record_stored")
    assert stored["ts_ms"] == 129_900
    retrieved = next(e for e in lines if e["event"] == "probe_retrieved")
    assert retrieved["ts_ms"] == 130_300  # the flush-time reading the old bug used instead


def test_evalagg_does_not_censor_a_store_whose_processing_time_lands_before_window_end(tmp_path: Path):
    """Feeding h0's real events.jsonl from the scenario above into evalagg.analyze_run,
    with window_end_ms = 130000 (evalgen's own stage-1 WINDOW_END_S), the (h1, 1) store
    must be read as delivered (129900 <= 130000), not censored.

    Before this fix, record_stored's ts_ms would have been the flush-time 130200 --
    strictly past window_end_ms -- and this same row would have come back censored
    with no delivery latency at all, exactly the "in-window store misclassified as
    cut off" the review flagged.
    """
    config, _agent = _run_store_during_probe_wait(tmp_path)

    run_dir = tmp_path / "run"
    h0_src = Path(config.state_dir) / "events.jsonl"
    h0_dst = run_dir / "debug" / "node_dirs" / "h0" / "semindex" / "events.jsonl"
    h0_dst.parent.mkdir(parents=True)
    h0_dst.write_text(h0_src.read_text())

    # h1 is not a real agent in this test; its own events.jsonl only needs
    # the one line _created_ts_by_key actually reads: when it created the
    # record h0 goes on to fetch and store. 60s mirrors the review's own
    # repro ("60秒に作られたレコード").
    h1_dst = run_dir / "debug" / "node_dirs" / "h1" / "semindex" / "events.jsonl"
    h1_dst.parent.mkdir(parents=True)
    h1_dst.write_text(json.dumps({"event": "record_created", "origin": "h1", "seq": 1, "ts_ms": 60_000}) + "\n")

    run = evalagg.analyze_run(0.0, 1, str(run_dir))
    assert run.t0_source == "agent_start"

    rows = {(r["origin"], r["seq"], r["node"]): r for r in run.discovery_rows}
    row = rows[("h1", 1, "h0")]
    assert row["censored"] is False, row
    assert row["latency_ms"] == pytest.approx(69_900.0)
