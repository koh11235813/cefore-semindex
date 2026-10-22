"""Tests for examples/two_host_sync/check_events.py's own parse-error handling.

check_events.py lives outside any Python package (it is a standalone
script shipped next to the live-run kit it checks, not part of the
semindex package), so it is loaded here by file path rather than imported
by name.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "two_host_sync" / "check_events.py"


def _load_check_events():
    """Load check_events.py from its file path and return the executed module."""
    spec = importlib.util.spec_from_file_location("check_events", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_events = _load_check_events()


def test_read_events_reports_an_unparsable_line_and_returns_none_not_a_traceback(tmp_path: Path, capsys):
    """2026-10-22 fix: a truncated/unparsable line (e.g. a host_apps process SIGKILLed
    mid-write) must print one FAIL line naming the file and the 1-indexed line number,
    and _read_events must return None -- never let json.JSONDecodeError propagate.
    """
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "counters", "bytes_sent": 1}\n{"event": "broken\n')

    result = check_events._read_events(str(path))

    assert result is None
    captured = capsys.readouterr()
    assert captured.out == f"FAIL: {path} line 2 unparsable\n"


def test_run_exits_1_when_either_input_file_has_an_unparsable_line(tmp_path: Path):
    """run() returns 1 (the checker's own "something failed" exit code), not a
    traceback, when either side's events.jsonl has an unparsable line.
    """
    good = tmp_path / "h0_events.jsonl"
    bad = tmp_path / "h1_events.jsonl"
    good.write_text('{"event": "counters"}\n')
    bad.write_text("not json at all\n")

    assert check_events.run(str(good), str(bad)) == 1
