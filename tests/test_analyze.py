"""Tests for the semindex.analyze script, against synthetic events.jsonl fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semindex.analyze import (
    all_nodes_consistent,
    analyze,
    control_bytes,
    discover_node_events,
    discovery_latency_ms,
    main,
    probe_pairs,
)
from semindex.check import CheckUsageError


def _write_jsonl(path: Path, events: list[dict]) -> None:
    """Write `events` to `path` as one JSON object per line, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def _node_events_path(run_dir: Path, node: str) -> Path:
    """Where semindex.analyze itself expects to find one node's archived events.jsonl under `run_dir`."""
    return run_dir / "debug" / "node_dirs" / node / "semindex" / "events.jsonl"


# -- discover_node_events -----------------------------------------------


def test_discover_node_events_sorts_by_host_index_not_lexicographically(tmp_path: Path):
    """Discover node events sorts by host index not lexicographically."""
    for node in ("h10", "h2", "h0", "h1"):
        _write_jsonl(_node_events_path(tmp_path, node), [])
    nodes = discover_node_events(str(tmp_path))
    assert list(nodes) == ["h0", "h1", "h2", "h10"]


def test_discover_node_events_raises_when_nothing_found(tmp_path: Path):
    """Discover node events raises when nothing found."""
    with pytest.raises(CheckUsageError) as exc_info:
        discover_node_events(str(tmp_path))
    assert str(exc_info.value) == f"no events.jsonl found under {tmp_path}/debug/node_dirs/h*/semindex/"


# -- discovery_latency_ms -------------------------------------------------


def test_discovery_latency_ms_computes_delay_from_creation_to_each_stored_copy():
    """Discovery latency ms computes delay from creation to each stored copy."""
    nodes = {
        "h0": [{"ts_ms": 100, "event": "record_created", "origin": "h0", "seq": 1}],
        "h1": [{"ts_ms": 150, "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h0"}],
        "h2": [{"ts_ms": 220, "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h1"}],
    }
    assert discovery_latency_ms(nodes) == [
        {"origin": "h0", "seq": 1, "node": "h1", "latency_ms": 50},
        {"origin": "h0", "seq": 1, "node": "h2", "latency_ms": 120},
    ]


def test_discovery_latency_ms_skips_a_stored_record_whose_creation_was_not_observed():
    """Discovery latency ms skips a stored record whose creation was not observed."""
    nodes = {"h1": [{"ts_ms": 150, "event": "record_stored", "origin": "ghost", "seq": 1, "holder": "h0"}]}
    assert discovery_latency_ms(nodes) == []


# -- all_nodes_consistent -------------------------------------------------


def test_all_nodes_consistent_reports_the_latest_completion_time_across_nodes():
    """All nodes consistent reports the latest completion time across nodes."""
    nodes = {
        "h0": [
            {"ts_ms": 100, "event": "record_created", "origin": "h0", "seq": 1},
            {"ts_ms": 200, "event": "record_created", "origin": "h0", "seq": 2},
        ],
        "h1": [
            {"ts_ms": 250, "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h0"},
            {"ts_ms": 260, "event": "record_stored", "origin": "h0", "seq": 2, "holder": "h0"},
        ],
        "h2": [
            {"ts_ms": 300, "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h1"},
            {"ts_ms": 310, "event": "record_stored", "origin": "h0", "seq": 2, "holder": "h1"},
        ],
    }
    assert all_nodes_consistent(nodes) == {"per_origin_ts_ms": {"h0": 310}, "overall_ts_ms": 310}


def test_all_nodes_consistent_is_null_when_a_node_never_completes():
    """All nodes consistent is null when a node never completes."""
    nodes = {
        "h0": [
            {"ts_ms": 100, "event": "record_created", "origin": "h0", "seq": 1},
            {"ts_ms": 200, "event": "record_created", "origin": "h0", "seq": 2},
        ],
        "h1": [{"ts_ms": 250, "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h0"}],
    }
    assert all_nodes_consistent(nodes) == {"per_origin_ts_ms": {"h0": None}, "overall_ts_ms": None}


# -- control_bytes ---------------------------------------------------------


def test_control_bytes_sums_every_counters_line_per_node_not_just_the_last():
    """Control bytes sums every counters line per node not just the last."""
    nodes = {
        "h0": [
            {
                "ts_ms": 100,
                "event": "counters",
                "bytes_sent": 10,
                "bytes_received": 5,
                "control_bytes_sent": 10,
                "control_bytes_received": 5,
            },
            {
                "ts_ms": 200,
                "event": "counters",
                "bytes_sent": 7,
                "bytes_received": 3,
                "control_bytes_sent": 7,
                "control_bytes_received": 3,
            },
        ],
        "h1": [
            {
                "ts_ms": 150,
                "event": "counters",
                "bytes_sent": 2,
                "bytes_received": 1,
                "control_bytes_sent": 2,
                "control_bytes_received": 1,
            }
        ],
    }
    assert control_bytes(nodes) == {
        "per_node": {"h0": {"bytes_sent": 17, "bytes_received": 8}, "h1": {"bytes_sent": 2, "bytes_received": 1}},
        "total": {"bytes_sent": 19, "bytes_received": 9},
    }


def test_control_bytes_reads_the_control_counters_not_the_probe_inflated_totals():
    """2026-10-22 fix: control_bytes reads control_bytes_sent/received, not
    bytes_sent/received -- a probe's own content-body fetch inflates bytes_sent/bytes_received (every
    chunk is counted there too), so a large probe body must not move control_bytes()'s total."""
    nodes = {
        "h0": [
            {
                "ts_ms": 100,
                "event": "counters",
                "bytes_sent": 10 + 433390,  # T1's own live retrieval, folded into the total column
                "bytes_received": 5,
                "control_bytes_sent": 10,
                "control_bytes_received": 5,
            }
        ],
    }
    assert control_bytes(nodes) == {
        "per_node": {"h0": {"bytes_sent": 10, "bytes_received": 5}},
        "total": {"bytes_sent": 10, "bytes_received": 5},
    }


# -- probe_pairs -----------------------------------------------------------


def test_probe_pairs_pairs_a_discovery_with_its_adjacent_retrieval():
    """Probe pairs pairs a discovery with its adjacent retrieval."""
    nodes = {
        "h2": [
            {"ts_ms": 100, "event": "probe_discovered", "matches": [{"origin": "h0", "seq": 1}]},
            {
                "ts_ms": 101,
                "event": "probe_retrieved",
                "origin": "h0",
                "seq": 1,
                "success": True,
                "bytes": 500,
            },
        ]
    }
    assert probe_pairs(nodes) == [
        {
            "node": "h2",
            "discovered_ts_ms": 100,
            "match_count": 1,
            "retrieved": {"ts_ms": 101, "origin": "h0", "seq": 1, "success": True, "bytes": 500},
        }
    ]


def test_probe_pairs_leaves_retrieved_null_when_nothing_was_discovered():
    """Probe pairs leaves retrieved null when nothing was discovered."""
    nodes = {"h2": [{"ts_ms": 100, "event": "probe_discovered", "matches": []}]}
    assert probe_pairs(nodes) == [
        {"node": "h2", "discovered_ts_ms": 100, "match_count": 0, "retrieved": None}
    ]


def test_probe_pairs_keeps_two_probes_on_the_same_node_in_order():
    """Probe pairs keeps two probes on the same node in order."""
    nodes = {
        "h2": [
            {"ts_ms": 100, "event": "probe_discovered", "matches": [{"origin": "h0", "seq": 1}]},
            {"ts_ms": 101, "event": "probe_retrieved", "origin": "h0", "seq": 1, "success": False, "bytes": 0},
            {"ts_ms": 500, "event": "probe_discovered", "matches": [{"origin": "h0", "seq": 1}]},
            {"ts_ms": 501, "event": "probe_retrieved", "origin": "h0", "seq": 1, "success": True, "bytes": 500},
        ]
    }
    pairs = probe_pairs(nodes)
    assert [p["discovered_ts_ms"] for p in pairs] == [100, 500]
    assert [p["retrieved"]["success"] for p in pairs] == [False, True]


# -- analyze()/main() integration -----------------------------------------


def test_analyze_integrates_every_section(tmp_path: Path):
    """Analyze integrates every section."""
    _write_jsonl(
        _node_events_path(tmp_path, "h0"),
        [
            {"ts_ms": 100, "event": "record_created", "origin": "h0", "seq": 1},
            {
                "ts_ms": 900,
                "event": "counters",
                "bytes_sent": 10,
                "bytes_received": 0,
                "control_bytes_sent": 10,
                "control_bytes_received": 0,
            },
        ],
    )
    _write_jsonl(
        _node_events_path(tmp_path, "h1"),
        [
            {"ts_ms": 150, "event": "record_stored", "origin": "h0", "seq": 1, "holder": "h0"},
            {"ts_ms": 160, "event": "probe_discovered", "matches": [{"origin": "h0", "seq": 1}]},
            {"ts_ms": 161, "event": "probe_retrieved", "origin": "h0", "seq": 1, "success": True, "bytes": 42},
            # The probe's own 42-byte retrieval above must not reach
            # control_bytes_sent/received: both stay at the sync-only
            # values this counters line reports, unrelated to `bytes`.
            {
                "ts_ms": 900,
                "event": "counters",
                "bytes_sent": 5,
                "bytes_received": 5 + 42,
                "control_bytes_sent": 5,
                "control_bytes_received": 5,
            },
        ],
    )

    result = analyze(str(tmp_path))

    assert result["nodes"] == ["h0", "h1"]
    assert result["discovery_latency_ms"] == [{"origin": "h0", "seq": 1, "node": "h1", "latency_ms": 50}]
    assert result["all_nodes_consistent"] == {"per_origin_ts_ms": {"h0": 150}, "overall_ts_ms": 150}
    assert result["control_bytes"]["total"] == {"bytes_sent": 15, "bytes_received": 5}
    assert len(result["probes"]) == 1
    assert result["probes"][0]["retrieved"]["bytes"] == 42


def test_main_prints_json_and_exits_0(tmp_path: Path, capsys):
    """Main prints json and exits 0."""
    _write_jsonl(
        _node_events_path(tmp_path, "h0"),
        [{"ts_ms": 100, "event": "record_created", "origin": "h0", "seq": 1}],
    )

    rc = main([str(tmp_path)])

    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["nodes"] == ["h0"]


def test_main_exits_1_on_a_usage_error(tmp_path: Path, capsys):
    """Main exits 1 on a usage error."""
    rc = main([str(tmp_path)])

    assert rc == 1
    assert capsys.readouterr().err == f"semindex.analyze: no events.jsonl found under {tmp_path}/debug/node_dirs/h*/semindex/\n"
