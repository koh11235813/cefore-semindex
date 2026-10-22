"""Tests for the semindex.agent module: config validation and the agent loop."""

from __future__ import annotations

import json
import random
import signal
import sqlite3
from pathlib import Path

import pytest

from semindex import holder_state, names, schema
from semindex.agent import (
    ConfigError,
    ScheduledProbe,
    _poll_jitter_rng,
    build_agent,
    build_sync_config,
    install_signal_handlers,
    load_config,
    validate_config,
)
from semindex.store import OriginProgress

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


def test_validate_config_applies_sync_overrides():
    """A given timing override reaches the SyncConfig built from the parsed AgentConfig."""
    raw = _minimal_raw()
    raw["state_poll_interval_s"] = 5.0
    config = validate_config(raw)
    sync_config = build_sync_config(config)
    assert sync_config.state_poll_interval_s == 5.0


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

    agent.step()  # elapsed = 0: not due yet
    assert events_path.read_text() == ""

    clock.value = 1005.0  # elapsed = 5
    agent.step()
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    created = [e for e in lines if e["event"] == "record_created"]
    assert len(created) == 1
    assert (created[0]["origin"], created[0]["seq"]) == ("h0", 1)


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
    )
    agent = build_agent(config, handle, clock=_FakeClock(0.0))
    agent.step()

    events_path = Path(config.state_dir) / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines()]
    discovered = next(e for e in lines if e["event"] == "probe_discovered")
    assert discovered["matches"] == [{"origin": "h0", "seq": 1, "content_name": content_name}]
    retrieved = next(e for e in lines if e["event"] == "probe_retrieved")
    assert (retrieved["success"], retrieved["bytes"]) == (True, 5)


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
