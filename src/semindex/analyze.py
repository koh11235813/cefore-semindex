"""semindex.analyze: discovery latency, convergence time, control bytes and probe pairs.

Reads every agent host's events.jsonl under a ReCefore run directory
(the same `debug/node_dirs/hN/semindex/events.jsonl` layout semindex.check
reads) and prints one JSON object to stdout with:

- `discovery_latency_ms`: for every (origin, seq, node) a record_stored
  was logged for, the delay from that origin's own record_created to this
  node storing it.
- `all_nodes_consistent`: per origin, the wall-clock time every
  participating node first held that origin's full created set, plus the
  overall time every origin was consistent everywhere -- `null` if some
  node never got there within this run.
- `control_bytes`: control_bytes_sent/control_bytes_received (sync
  protocol traffic only, excluding a probe's own content-body fetch)
  summed from every `counters` line in each node's file -- *every* line,
  not the last one, since a restarted node's events.jsonl holds one
  `counters` line per life (agent.py's EventWriter is opened in append
  mode across a restart).
- `probes`: every probe_discovered this run logged, paired with the
  probe_retrieved that followed it when it found a match (agent.py's
  _run_probe writes exactly one of these, back to back, for a probe with
  a non-empty discovery, and writes nothing else to events.jsonl in
  between -- see that module's _run_probe for why the two always land as
  adjacent lines).

Nothing here is written back into ReCefore's own results.json (map.md:
"ReCefore の results.json には何も足さない") -- this module only reads
events.jsonl files and prints to stdout.

Run as:

    python -m semindex.analyze <run_dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

from semindex.check import CheckUsageError, read_events

_NODE_DIR_RE = re.compile(r"^h(\d+)$")


def _node_sort_key(node: str) -> int:
    """Sort node names by host index (h2 before h10), falling back to the string itself if unexpected."""
    match = _NODE_DIR_RE.match(node)
    return int(match.group(1)) if match else -1


def discover_node_events(run_dir: str) -> dict[str, list[dict]]:
    """Every agent host's events.jsonl under `run_dir`, keyed by node name, in host-index order."""
    pattern = os.path.join(run_dir, "debug", "node_dirs", "h*", "semindex", "events.jsonl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise CheckUsageError(f"no events.jsonl found under {run_dir}/debug/node_dirs/h*/semindex/")
    nodes: dict[str, list[dict]] = {}
    for path in paths:
        # .../debug/node_dirs/<node>/semindex/events.jsonl
        node = os.path.basename(os.path.dirname(os.path.dirname(path)))
        nodes[node] = read_events(path)
    return dict(sorted(nodes.items(), key=lambda kv: _node_sort_key(kv[0])))


def _created_ts_by_key(nodes: dict[str, list[dict]]) -> dict[tuple[str, int], float]:
    """(origin, seq) -> that origin's own record_created ts_ms, from whichever node's file holds it.

    Only the origin ever logs record_created for its own records, so one
    pass over every node's file is enough: each (origin, seq) key is
    written at most once across the whole run.
    """
    created: dict[tuple[str, int], float] = {}
    for events in nodes.values():
        for event in events:
            if event.get("event") == "record_created":
                created[(event["origin"], event["seq"])] = event["ts_ms"]
    return created


def discovery_latency_ms(nodes: dict[str, list[dict]]) -> list[dict]:
    """One entry per (origin, seq, node) a record_stored was logged for: its propagation delay."""
    created = _created_ts_by_key(nodes)
    rows = []
    for node, events in nodes.items():
        for event in events:
            if event.get("event") != "record_stored":
                continue
            key = (event["origin"], event["seq"])
            created_ts = created.get(key)
            if created_ts is None:
                continue  # the origin's own record_created line was not found in this run's data
            rows.append(
                {
                    "origin": event["origin"],
                    "seq": event["seq"],
                    "node": node,
                    "latency_ms": event["ts_ms"] - created_ts,
                }
            )
    rows.sort(key=lambda r: (r["origin"], r["seq"], r["node"]))
    return rows


def all_nodes_consistent(nodes: dict[str, list[dict]]) -> dict:
    """Per origin, the wall-clock ts every participating node first held that origin's full created set."""
    created = _created_ts_by_key(nodes)
    target_seqs: dict[str, set[int]] = {}
    for origin, seq in created:
        target_seqs.setdefault(origin, set()).add(seq)

    stored_ts: dict[str, dict[str, dict[int, float]]] = {}  # origin -> node -> seq -> ts_ms
    for node, events in nodes.items():
        for event in events:
            if event.get("event") == "record_stored":
                stored_ts.setdefault(event["origin"], {}).setdefault(node, {})[event["seq"]] = event["ts_ms"]

    per_origin: dict[str, float | None] = {}
    for origin, seqs in target_seqs.items():
        node_complete_ts: list[float] = []
        complete = True
        for node, events in nodes.items():
            if any(e.get("event") == "record_created" and e["origin"] == origin for e in events):
                # This node IS the origin: it holds its own full set from
                # the moment it created the last of them.
                node_complete_ts.append(max(created[(origin, s)] for s in seqs))
                continue
            held = stored_ts.get(origin, {}).get(node, {})
            if set(held) >= seqs:
                node_complete_ts.append(max(held[s] for s in seqs))
            else:
                complete = False
        per_origin[origin] = max(node_complete_ts) if complete and node_complete_ts else None

    overall = None
    if per_origin and all(ts is not None for ts in per_origin.values()):
        overall = max(per_origin.values())
    return {"per_origin_ts_ms": per_origin, "overall_ts_ms": overall}


def control_bytes(nodes: dict[str, list[dict]]) -> dict:
    """control_bytes_sent/control_bytes_received summed from every `counters` line per node (and the grand total).

    Summed, not the last line taken alone: a restarted node's
    events.jsonl holds one `counters` line per life (agent.py's
    EventWriter is append-mode, and Agent.shutdown() writes one on every
    clean stop), and each life's totals only cover that life's own traffic.

    2026-10-22 fix: reads control_bytes_sent/control_bytes_received,
    not bytes_sent/bytes_received -- the latter also include a probe's own
    content-body fetch (cefpyco_transport.py's "Byte counters" docstring
    entry), so this section's name ("control") would otherwise be wrong
    whenever a run's probe retrieved a large body.

    2026-10-23 fix: a node that never wrote a `counters` line at all (an
    unclean shutdown -- SIGKILLed before Agent.shutdown() could run) used
    to report 0 here, indistinguishable from a node that cleanly shut
    down having truly sent/received nothing. `None` marks "no data",
    never silently "zero traffic"; the grand total only sums the nodes
    that actually reported a number.
    """
    per_node = {}
    for node, events in nodes.items():
        counters = [e for e in events if e.get("event") == "counters"]
        if not counters:
            per_node[node] = {"control_bytes_sent": None, "control_bytes_received": None}
            continue
        # 2026-10-22 fix: these dict keys were bytes_sent/bytes_received
        # even though the values are the control-only counters -- a reader of this JSON alone
        # (no docstring in hand) could misread them as the probe-inflated totals this whole
        # function exists to exclude.
        per_node[node] = {
            "control_bytes_sent": sum(e["control_bytes_sent"] for e in counters),
            "control_bytes_received": sum(e["control_bytes_received"] for e in counters),
        }
    total = {
        "control_bytes_sent": sum(
            v["control_bytes_sent"] for v in per_node.values() if v["control_bytes_sent"] is not None
        ),
        "control_bytes_received": sum(
            v["control_bytes_received"] for v in per_node.values() if v["control_bytes_received"] is not None
        ),
    }
    return {"per_node": per_node, "total": total}


def node_probe_pairs(node: str, events: list[dict]) -> list[dict]:
    """One node's probe_discovered events, each paired with the probe_retrieved that followed it, in FILE order.

    File order is the agent's own firing order; ts_ms is wall clock and
    can step backwards, so a caller that must line events up with the
    config's probe list (semindex.evalagg) needs this order, not
    `probe_pairs`' ts-sorted one.
    """
    pairs = []
    discovered_idxs = [i for i, e in enumerate(events) if e.get("event") == "probe_discovered"]
    retrieved_iter = iter(i for i, e in enumerate(events) if e.get("event") == "probe_retrieved")
    for idx in discovered_idxs:
        discovered = events[idx]
        retrieved = None
        if discovered.get("matches"):
            retrieved_idx = next(retrieved_iter, None)
            if retrieved_idx is not None:
                retrieved = events[retrieved_idx]
        pairs.append(
            {
                "node": node,
                "discovered_ts_ms": discovered["ts_ms"],
                "match_count": len(discovered.get("matches", [])),
                "retrieved": (
                    None
                    if retrieved is None
                    else {
                        "ts_ms": retrieved["ts_ms"],
                        "origin": retrieved.get("origin"),
                        "seq": retrieved.get("seq"),
                        "success": retrieved.get("success"),
                        "bytes": retrieved.get("bytes"),
                    }
                ),
            }
        )
    return pairs


def probe_pairs(nodes: dict[str, list[dict]]) -> list[dict]:
    """Every probe_discovered, paired with the probe_retrieved that followed it (None if no match), sorted by (node, ts)."""
    pairs = [pair for node, events in nodes.items() for pair in node_probe_pairs(node, events)]
    pairs.sort(key=lambda p: (p["node"], p["discovered_ts_ms"]))
    return pairs


def analyze(run_dir: str) -> dict:
    """Build the full analysis dict for `run_dir`."""
    nodes = discover_node_events(run_dir)
    return {
        "nodes": sorted(nodes),
        "discovery_latency_ms": discovery_latency_ms(nodes),
        "all_nodes_consistent": all_nodes_consistent(nodes),
        "control_bytes": control_bytes(nodes),
        "probes": probe_pairs(nodes),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the analysis script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="The ReCefore experiment directory to analyze")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: python3 -m semindex.analyze <run_dir>."""
    args = parse_args(argv)
    try:
        result = analyze(args.run_dir)
    except CheckUsageError as exc:
        print(f"semindex.analyze: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
