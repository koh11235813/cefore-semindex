#!/usr/bin/env python3
"""check_events: verify a two_host_sync.yaml run synced records both ways.

Reads the two archived events.jsonl files from a two_host_sync.yaml run
(one per host, under debug/node_dirs/hN/semindex/events.jsonl once
debug.artifacts: ["node_dirs"] has kept them past teardown) and prints
one PASS/FAIL line per assertion against them. Exits 0 only if every
assertion passed, 1 otherwise, 2 on a usage error -- so a caller can read
pass/fail from the exit code alone, the same convention cefpyco's own
semindex_smoke/consumer.py uses.
"""

from __future__ import annotations

import json
import sys


def _read_events(path: str) -> list[dict] | None:
    """Parse one events.jsonl file into a list of event dicts, in file order.

    Returns None (after printing one FAIL line naming the bad line) if any
    line cannot be parsed as JSON. 2026-10-22 fix: a host_apps process that
    is still writing events.jsonl when ReCefore's SIGKILL lands (2s after
    SIGTERM, if shutdown() has not finished flushing by then) can leave
    that last line truncated mid-write; json.loads on it used to raise a
    traceback instead of going through this script's own PASS/FAIL/exit-code
    convention.
    """
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"FAIL: {path} line {lineno} unparsable")
                return None
    return events


def _stored(events: list[dict], origin: str) -> list[dict]:
    """Every record_stored event for `origin` in `events`."""
    return [e for e in events if e.get("event") == "record_stored" and e.get("origin") == origin]


def _check(label: str, condition: bool, detail: str = "") -> bool:
    """Print one PASS/FAIL line for one assertion and return whether it passed.

    Always prints, regardless of earlier results: callers combine this
    function's return value with `&=`, never `and`, specifically so a
    later assertion is never skipped (and its PASS/FAIL line never
    suppressed) just because an earlier one failed.
    """
    status = "PASS" if condition else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"{status}: {label}{suffix}")
    return condition


def run(h0_path: str, h1_path: str) -> int:
    """Check both sides' events.jsonl and return 0 if every assertion passed, else 1."""
    h0_events = _read_events(h0_path)
    h1_events = _read_events(h1_path)
    if h0_events is None or h1_events is None:
        return 1  # _read_events already printed its own FAIL line

    ok = True

    h0_stored_h1 = _stored(h0_events, "h1")
    ok &= _check(
        "h0 stored every record h1 created (origin=h1, seq 1-2)",
        {e["seq"] for e in h0_stored_h1} == {1, 2},
        f"seqs stored: {sorted(e['seq'] for e in h0_stored_h1)}",
    )
    ok &= _check(
        "h0's stored h1 records all came from holder h1 (the direct neighbor; h2 runs no Sync agent, so there is no relay)",
        all(e["holder"] == "h1" for e in h0_stored_h1),
        f"holders seen: {sorted({e['holder'] for e in h0_stored_h1})}",
    )

    h1_stored_h0 = _stored(h1_events, "h0")
    ok &= _check(
        "h1 stored every record h0 created (origin=h0, seq 1-3)",
        {e["seq"] for e in h1_stored_h0} == {1, 2, 3},
        f"seqs stored: {sorted(e['seq'] for e in h1_stored_h0)}",
    )
    ok &= _check(
        "h1's stored h0 records all came from holder h0 (the direct neighbor; h2 runs no Sync agent, so there is no relay)",
        all(e["holder"] == "h0" for e in h1_stored_h0),
        f"holders seen: {sorted({e['holder'] for e in h1_stored_h0})}",
    )

    discoveries = [e for e in h1_events if e.get("event") == "probe_discovered"]
    non_empty = [e for e in discoveries if e.get("matches")]
    ok &= _check(
        "h1's probe discovered at least one match",
        len(non_empty) > 0,
        f"{len(discoveries)} probe_discovered event(s), {len(non_empty)} non-empty",
    )

    # Beyond the three checks above (the orchestrator's own acceptance
    # list): discovery succeeding does not by itself prove the content
    # body round-tripped, so this also checks the chunked retrieval.
    retrievals = [e for e in h1_events if e.get("event") == "probe_retrieved"]
    ok &= _check(
        "h1's probe retrieved the matched content body",
        any(e.get("success") is True for e in retrievals),
        f"{len(retrievals)} probe_retrieved event(s): {[e.get('success') for e in retrievals]}",
    )

    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point: python3 check_events.py <h0_events.jsonl> <h1_events.jsonl>."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: check_events.py <h0_events.jsonl> <h1_events.jsonl>", file=sys.stderr)
        return 2
    return run(argv[0], argv[1])


if __name__ == "__main__":
    sys.exit(main())
