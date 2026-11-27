#!/usr/bin/env python3
"""Replay a real run's resolutions against a model server and time the answers.

A recording directory holds, one file per node, every decision the
engine's decider seam was handed during an actual emulator run. This
script answers one question with that history: when the nodes of that run
ask the one server at the instants they really asked, how long does an
answer take. It drives one `ldecider.LClient` per node, on its own thread,
replaying that node's recorded resolutions at their recorded relative
offsets, and reports the round-trip time distribution.
"""
import sys
# Set before importing ldecider, which sets the same flag on itself too
# late to matter here: a run of this script may not be invoked with -B,
# and the point of the flag is to stop a __pycache__ directory being
# written by a process whose owner cannot always clean it up.
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import ldecider  # noqa: E402

REQUEST_TIMEOUT_S = 5.0
POLL_SLEEP_S = 0.0005
# The two numbers the design's latency budget gets reopened at if a
# round trip exceeds them; this script only counts crossings, it does not
# judge whether that count is acceptable.
THRESHOLDS_MS = (560.0, 1125.5)


def load_node_resolutions(path):
    """Return `[(t_mono, obs), ...]` for every recorded resolution in `path`.

    Only `decide` lines whose `obs` is a resolution (not a send) are kept:
    a send never produces a request on the wire, so replaying it would
    only add noise to a round-trip measurement.
    """
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("rec") != "decide":
                continue
            obs = rec["obs"]
            if obs["where"]["kind"] != "state_poll_reschedule":
                continue
            out.append((rec["t_mono"], obs))
    out.sort(key=lambda pair: pair[0])
    return out


def percentile(sorted_values, pct):
    """Nearest-rank percentile of an already-sorted list; `None` if empty.

    `int(pct * (n - 1))` looks like nearest-rank but truncates low (it
    understates p99 on a small sample); `ceil(pct * n) - 1` is the
    standard nearest-rank formula and matches it at the ends.
    """
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, math.ceil(pct * len(sorted_values)) - 1))
    return sorted_values[idx]


def replay_node(node, client, items, t0, speed, out):
    """Fire `node`'s recorded resolutions at their recorded offsets and time the answers.

    One thread runs this for one node's client. Firing due resolutions and
    polling for answers share the same tight loop so a request is never
    delayed by waiting on a previous one's answer, which would compress a
    real run's overlap into something slower than it actually was.
    """
    pending = {}
    idx = 0
    while idx < len(items) or pending:
        now = time.monotonic()
        while idx < len(items) and now >= t0 + items[idx][0] / speed:
            t_offset_s, obs = items[idx]
            t_start = time.monotonic()
            client.decide(obs)
            where = obs["where"]
            token = (where["neighbor"], obs["resolved"]["poll_epoch_ms"])
            pending[token] = (t_start, t_offset_s / speed)
            idx += 1
            now = time.monotonic()

        for token, interval, _info in client.collect():
            start = pending.pop(token, None)
            if start is None:
                continue
            t_start, t_offset_s = start
            out.append({"node": node, "neighbor": token[0], "poll_epoch_ms": token[1],
                        "t_offset_s": t_offset_s, "rt_ms": (time.monotonic() - t_start) * 1000.0,
                        "interval_s": interval})

        timed_out = [tok for tok, (t_start, _off) in pending.items()
                     if time.monotonic() - t_start > REQUEST_TIMEOUT_S]
        for tok in timed_out:
            t_start, t_offset_s = pending.pop(tok)
            out.append({"node": node, "neighbor": tok[0], "poll_epoch_ms": tok[1],
                        "t_offset_s": t_offset_s, "rt_ms": None, "interval_s": None})

        time.sleep(POLL_SLEEP_S)


def run_warmup(client, items, count):
    """Send the first `count` resolutions of one node once, untimed, and wait for answers.

    Exists so the replay's own timing is not the thing paying for a cold
    connection or a model's first, slower inference; that cost is real but
    it is a one-off, not part of the steady-state latency this script
    reports.
    """
    tokens = set()
    for _t_mono, obs in items[:count]:
        client.decide(obs)
        where = obs["where"]
        tokens.add((where["neighbor"], obs["resolved"]["poll_epoch_ms"]))
    deadline = time.monotonic() + REQUEST_TIMEOUT_S
    while tokens and time.monotonic() < deadline:
        for token, _interval, _info in client.collect():
            tokens.discard(token)
        time.sleep(POLL_SLEEP_S)


def report(label, recs):
    """Print one line of n/timeout/percentile/threshold-crossing stats for `recs`."""
    rts = sorted(r["rt_ms"] for r in recs if r["rt_ms"] is not None)
    timeouts = sum(1 for r in recs if r["rt_ms"] is None)
    over = [sum(1 for v in rts if v > t) for t in THRESHOLDS_MS]
    print("%s: n=%d timeout=%d p50_ms=%s p99_ms=%s max_ms=%s over_560ms=%d over_1125.5ms=%d" % (
        label, len(recs), timeouts,
        "%.2f" % percentile(rts, 0.50) if rts else "n/a",
        "%.2f" % percentile(rts, 0.99) if rts else "n/a",
        "%.2f" % rts[-1] if rts else "n/a",
        over[0], over[1]))


def worker(node, client, items, t0, speed, records, out_lock):
    """Thread body: replay one node and merge its records into the shared list.

    An exception here prints its own traceback (Python's default thread
    behaviour) and this node's records simply stay short; `main` catches
    that afterwards by comparing each node's record count against its
    recorded resolution count, so a dead thread still fails the run.
    """
    local = []
    replay_node(node, client, items, t0, speed, local)
    with out_lock:
        records.extend(local)


def main():
    """Parse arguments, run the replay, print the report, and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--out")
    parser.add_argument("--warmup", type=int, default=0)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.record_dir, "h*.jsonl")))
    nodes = [os.path.splitext(os.path.basename(f))[0] for f in files]
    per_node_items = {node: load_node_resolutions(f) for node, f in zip(nodes, files)}
    earliest = min(items[0][0] for items in per_node_items.values() if items)
    offset_items = {node: [(t_mono - earliest, obs) for t_mono, obs in items]
                    for node, items in per_node_items.items()}

    with tempfile.TemporaryDirectory(prefix="lbench-stats-") as stats_dir:
        clients = {node: ldecider.build(node=node, socket_path=args.socket,
                                         send_start=6.0, stats_dir=stats_dir)
                   for node in nodes}

        if args.warmup and "h0" in clients and per_node_items.get("h0"):
            run_warmup(clients["h0"], per_node_items["h0"], args.warmup)

        records = []
        out_lock = threading.Lock()
        t0 = time.monotonic()
        threads = [threading.Thread(target=worker,
                                     args=(node, clients[node], offset_items[node],
                                           t0, args.speed, records, out_lock))
                   for node in nodes]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for client in clients.values():
            client.close()

        counters = {}
        for node in nodes:
            with open(os.path.join(stats_dir, "%s.lclient.json" % node), encoding="utf-8") as fh:
                counters[node] = json.load(fh)

    by_node = {}
    for rec in records:
        by_node.setdefault(rec["node"], []).append(rec)

    for node in nodes:
        report(node, by_node.get(node, []))
    report("overall", records)

    print("counters:")
    for node in nodes:
        print("  %s: %s" % (node, counters[node]))

    arms = {}
    for rec in records:
        if rec["interval_s"] is not None:
            arms[rec["interval_s"]] = arms.get(rec["interval_s"], 0) + 1
    print("arms answered: %s" % arms)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    # A thread that raised produced fewer records than its node's recorded
    # resolution count; that must fail the run exactly like a timeout does,
    # since both mean "this node's answers are not all accounted for".
    short_node = any(len(by_node.get(node, [])) < len(per_node_items[node]) for node in nodes)
    any_timeout = any(r["rt_ms"] is None for r in records)
    return 1 if (any_timeout or short_node) else 0


if __name__ == "__main__":
    sys.exit(main())
