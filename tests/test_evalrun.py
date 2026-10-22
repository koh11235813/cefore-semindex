"""Tests for semindex.evalrun, against a fake ReCefore that never needs root.

The fake `run_subprocess` below stands in for both `mn -c` and the actual
ReCefore `disaster` subprocess: for the latter, it reads the YAML
`evalgen`/`gen` already generated (the same file the real ReCefore would
have been given as `--config`) and fabricates the output tree a clean run
would have produced -- `results.json` plus `debug/node_dirs/<node>/...`,
Cefore key files included, so tests can confirm `evalrun` never copies
them out.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from semindex import evalgen, evalrun

_T0_EPOCH = 1_700_000_000.0


def _node_config(config_dir: str, host: int) -> dict:
    with open(os.path.join(config_dir, f"h{host}.json"), encoding="utf-8") as f:
        return json.load(f)


def _write_fake_node_dir(run_dir: str, host: int, node_config: dict, *, with_records: bool, skip: bool = False) -> None:
    """Fabricate one node's `debug/node_dirs/hN/` the way a clean ReCefore run would have left it."""
    if skip:
        return
    node = f"h{host}"
    node_dir = os.path.join(run_dir, "debug", "node_dirs", node)
    semindex_dir = os.path.join(node_dir, "semindex")
    os.makedirs(semindex_dir, exist_ok=True)

    # The first line every real agent life writes (semindex.agent.Agent.__init__) -- evalvalid's t0
    # derivation now prefers this over any record-based estimate, and its own stage-1 single-life
    # check flags a node missing it outright, so the fake run this module fabricates must include it
    # too for `run_one`'s stored validity verdict to stay "valid" the way these tests expect.
    agent_start_t0_ms = round(_T0_EPOCH * 1000)
    events = [{"ts_ms": agent_start_t0_ms, "event": "agent_start", "node": node, "t0_ms": agent_start_t0_ms}]
    if with_records:
        for i, entry in enumerate(node_config["schedule"]):
            events.append(
                {"ts_ms": round((_T0_EPOCH + entry["at_s"]) * 1000), "event": "record_created", "origin": node, "seq": i + 1}
            )
    counters_fields = {"bytes_sent": 10, "bytes_received": 10, "control_bytes_sent": 10, "control_bytes_received": 10}
    for snap_at_s in node_config.get("counters_snapshot_at_s", []):
        events.append({"ts_ms": round((_T0_EPOCH + snap_at_s) * 1000), "event": "counters_snapshot", **counters_fields})
    events.append({"ts_ms": round((_T0_EPOCH + node_config["stop_at_s"]) * 1000), "event": "counters", **counters_fields})

    with open(os.path.join(semindex_dir, "events.jsonl"), "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    with open(os.path.join(node_dir, "semindex_agent.log"), "w", encoding="utf-8") as f:
        f.write(f"{node}: agent started\n")

    # Stands in for the real Cefore key material (config/templates/h*/default-private-key,
    # cefnetd.key, ...) that lives directly in node_dirs/<node>/ on a real run -- never
    # globbed by _copy_run_outputs, so a test can assert it never reaches the store.
    with open(os.path.join(node_dir, "default-private-key"), "w", encoding="utf-8") as f:
        f.write("not-a-real-key\n")


def _load_generated_yaml(yaml_path: str) -> dict:
    """gen.write_experiment's `<name>.yaml` is a `# comment` line followed by indented JSON (stdlib only, no PyYAML)."""
    with open(yaml_path, encoding="utf-8") as f:
        text = f.read()
    _, _, json_text = text.partition("\n")
    return json.loads(json_text)


def make_fake_run_subprocess(*, skip_node: int | None = None, calls: list | None = None):
    """A `run_subprocess` fake: no-ops `mn -c`, fabricates a synthetic tree for the ReCefore invocation."""
    calls = calls if calls is not None else []

    def _fake(argv, *, cwd=None):
        calls.append((list(argv), cwd))
        if argv[:2] == ["mn", "-c"]:
            return 0
        assert "disaster" in argv and "--config" in argv, f"unexpected argv: {argv}"
        yaml_path = argv[argv.index("--config") + 1]
        recefore_config = _load_generated_yaml(yaml_path)
        config_dir = os.path.splitext(yaml_path)[0]  # gen.write_experiment's "<outdir>/<name>" config dir
        run_dir = os.path.join(recefore_config["output_dir"], f"ex{recefore_config['num']}_seed{recefore_config['seed']}")
        os.makedirs(run_dir, exist_ok=True)

        put_event = recefore_config["events"][0]
        link_events = recefore_config["events"][1:]
        results = [
            {
                "op_type": "put",
                "uri": put_event["uri"],
                "ts": "2023-01-01T00:00:00+00:00",
                "success": True,
            }
        ]
        for event in link_events:
            at = event["at"]
            results.append(
                {
                    "op_type": "event",
                    "event_type": event["type"],
                    "ts": "2023-01-01T00:00:00+00:00",
                    "success": True,
                    "error": None,
                    "scheduled_at": at,
                    "actual_at": at,
                    "event": {"nodes": list(event["nodes"])},
                }
            )
        with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f)

        for host in recefore_config["host_apps"][0]["hosts"]:
            node_config = _node_config(config_dir, host)
            _write_fake_node_dir(
                run_dir,
                host,
                node_config,
                with_records=bool(node_config["schedule"]),
                skip=(host == skip_node),
            )
        return 0

    return _fake


def _fake_capture_subprocess(argv, *, cwd=None):
    """A `capture_subprocess` fake: no real git/python available in a test sandbox.

    2026-10-23 fix: this used to be `"status --porcelain" in " ".join(argv) and "" or "deadbeef"` --
    the classic `cond and A or B` trap, broken whenever A is falsy: `"" or "deadbeef"` always
    picks `"deadbeef"` regardless of `cond`, so EVERY git call (including `status --porcelain`)
    actually returned a truthy, non-empty string here, which `_git_dirty` reads as dirty. Nothing
    asserted the dirty flag's actual value before `run_all`'s own dirty-sweep guard was added, so
    this never surfaced; that guard now reads every happy-path fixture run as dirty and refuses to
    start, via this exact fixture.
    """
    if argv[:1] == ["git"]:
        return "" if "status" in argv else "deadbeef"
    return "Python 3.12.0 (fake)"


# -- parse_seeds / build_run_order ------------------------------------------


def test_parse_seeds_handles_ranges_and_lists():
    """Parse seeds handles ranges and lists."""
    assert evalrun.parse_seeds("1-3,7,10-12") == [1, 2, 3, 7, 10, 11, 12]


def test_build_run_order_is_seed_major_sorted_and_deduplicated():
    """Build run order is seed-major, sorted and deduplicated."""
    order = evalrun.build_run_order([0.2, 0.1, 0.1], [2, 1])
    assert order == [(0.1, 1), (0.2, 1), (0.1, 2), (0.2, 2)]


# -- _guarded_rmtree ----------------------------------------------------------


def test_guarded_rmtree_deletes_a_path_strictly_inside_root(tmp_path: Path):
    """Guarded rmtree deletes a path strictly inside root."""
    root = tmp_path / "root"
    victim = root / "victim"
    victim.mkdir(parents=True)
    (victim / "f.txt").write_text("x", encoding="utf-8")
    evalrun._guarded_rmtree(str(victim), root=str(root))
    assert not victim.exists()


def test_guarded_rmtree_refuses_a_path_outside_root(tmp_path: Path):
    """Guarded rmtree refuses a path outside root."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError):
        evalrun._guarded_rmtree(str(outside), root=str(root))
    assert outside.exists()


def test_guarded_rmtree_refuses_a_symlink(tmp_path: Path):
    """Guarded rmtree refuses a symlink even when it resolves inside root."""
    root = tmp_path / "root"
    root.mkdir()
    real_target = tmp_path / "real_target"
    real_target.mkdir()
    link = root / "link"
    link.symlink_to(real_target, target_is_directory=True)
    with pytest.raises(ValueError):
        evalrun._guarded_rmtree(str(link), root=str(root))
    assert real_target.exists()


def test_guarded_rmtree_refuses_root_itself(tmp_path: Path):
    """Guarded rmtree refuses deleting root itself (not strictly inside it)."""
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError):
        evalrun._guarded_rmtree(str(root), root=str(root))
    assert root.exists()


# -- _validate_stage ------------------------------------------------------------


def test_validate_stage_rejects_unsafe_values():
    """Validate stage rejects anything but a single, bare path component."""
    for bad in ["/tmp/outside", "..", ".", "", "a/b", "a" + os.sep + "b"]:
        with pytest.raises(ValueError):
            evalrun._validate_stage(bad)


def test_validate_stage_accepts_a_plain_name():
    """Validate stage accepts a plain name."""
    evalrun._validate_stage("s1")  # must not raise


# -- run_one: path-safety regressions (review r1) --------------------------------


def test_run_one_rejects_an_absolute_stage_before_touching_anything(tmp_path: Path, monkeypatch):
    """Run one rejects an absolute --stage before building or deleting any path from it.

    Reviewer counterexample: `--stage /tmp/outside` with `/tmp/outside/p0.2/7` already
    present and no DONE marker -- the deletion guard used to be reachable before `stage`
    was ever validated, since `os.path.join(store, stage, ...)` discards `store` entirely
    once `stage` is absolute.
    """
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    outside_stage = tmp_path / "outside"
    sentinel_dir = outside_stage / "p0.2" / "7"
    sentinel_dir.mkdir(parents=True)
    (sentinel_dir / "sentinel.txt").write_text("do not delete me", encoding="utf-8")

    def _must_not_run(argv, *, cwd=None):
        raise AssertionError("run_subprocess must not be called: stage validation should fail first")

    with pytest.raises(ValueError):
        evalrun.run_one(
            0.2, 7, str(outside_stage), str(store), str(recefore), str(staging_root), run_subprocess=_must_not_run
        )

    assert (sentinel_dir / "sentinel.txt").exists()
    assert not store.exists()


def test_run_one_refuses_to_delete_through_a_symlinked_stage_dir(tmp_path: Path, monkeypatch):
    """Run one refuses the delete when `store/<stage>` was replaced with a symlink pointing
    outside `store`, even though `--stage` itself is a normal, already-valid name.

    Reviewer counterexample: `store/s1` is a symlink to an external directory; both the
    delete target and (before this fix) the delete's own root were derived from that same
    symlinked path, so the containment check always passed. Anchoring the guard at `store`
    itself (not `store/<stage>`) makes the symlinked stage dir resolve outside the root
    the guard actually checks against.
    """
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    store.mkdir()
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    outside = tmp_path / "outside"
    sentinel_dir = outside / "p0.2" / "7"
    sentinel_dir.mkdir(parents=True)
    (sentinel_dir / "sentinel.txt").write_text("do not delete me", encoding="utf-8")
    (store / "s1").symlink_to(outside, target_is_directory=True)

    def _must_not_run(argv, *, cwd=None):
        raise AssertionError("run_subprocess must not be called: the delete guard should fail first")

    with pytest.raises(ValueError):
        evalrun.run_one(0.2, 7, "s1", str(store), str(recefore), str(staging_root), run_subprocess=_must_not_run)

    assert (sentinel_dir / "sentinel.txt").exists()


# -- _verify_safe_run_dir / run_one: path-safety regressions (review r2) --------


def test_verify_safe_run_dir_accepts_a_path_that_does_not_exist_yet(tmp_path: Path):
    """Verify safe run dir accepts a path that does not exist yet, as long as every
    component down to it is a plain (non-symlink) name under store."""
    store = tmp_path / "store"
    store.mkdir()
    run_dir = store / "s1" / "p0.2" / "7"
    evalrun._verify_safe_run_dir(str(run_dir), str(store))  # must not raise
    assert not run_dir.exists()


def test_verify_safe_run_dir_refuses_a_symlinked_component_whose_target_does_not_exist_yet(tmp_path: Path):
    """Verify safe run dir refuses a symlinked intermediate component even when its target has not
    been created at all yet: `os.path.isdir` on the full run dir is False here, which is exactly what
    let an isdir-gated check (the only boundary check this used to have) skip this case entirely."""
    store = tmp_path / "store"
    store.mkdir()
    outside_target = tmp_path / "outside_target"  # deliberately never created
    (store / "s1").symlink_to(outside_target, target_is_directory=True)
    run_dir = store / "s1" / "p0.2" / "7"
    with pytest.raises(ValueError):
        evalrun._verify_safe_run_dir(str(run_dir), str(store))
    assert not outside_target.exists()


def test_verify_safe_run_dir_refuses_a_symlinked_component_whose_target_already_exists(tmp_path: Path):
    """Verify safe run dir refuses a symlinked intermediate component whose target already exists too
    (the case the old check -- gated on the full run dir existing -- did catch)."""
    store = tmp_path / "store"
    store.mkdir()
    outside_target = tmp_path / "outside_target"
    outside_target.mkdir()
    (store / "s1").symlink_to(outside_target, target_is_directory=True)
    run_dir = store / "s1" / "p0.2" / "7"
    with pytest.raises(ValueError):
        evalrun._verify_safe_run_dir(str(run_dir), str(store))


def test_run_one_refuses_a_symlinked_stage_dir_whose_target_run_dir_does_not_exist_yet(tmp_path: Path, monkeypatch):
    """Run one refuses the whole run before touching anything when `store/<stage>` is a symlink to an
    external directory that does not yet exist at all.

    Unlike the existing symlink test above, where the external target's `p0.2/7` leaf already
    existed, here NOTHING under the external target exists yet. `os.path.isdir(store_run_dir)` is
    therefore False, which is exactly what let the old,
    isdir-gated boundary check skip straight past the DONE check and into the full pipeline --
    `os.makedirs(store_run_dir, exist_ok=True)` would then have created the external target from
    scratch and written `meta.json`/DONE into it, entirely outside `store`.
    """
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    store.mkdir()
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    outside_target = tmp_path / "outside_target"  # deliberately never created
    (store / "s1").symlink_to(outside_target, target_is_directory=True)

    def _must_not_run(argv, *, cwd=None):
        raise AssertionError("run_subprocess must not be called: the safety check should fail first")

    with pytest.raises(ValueError):
        evalrun.run_one(0.2, 7, "s1", str(store), str(recefore), str(staging_root), run_subprocess=_must_not_run)

    assert not outside_target.exists()


# -- ReCefore invocation: --kill-after (review r1) -------------------------------


def test_run_one_invokes_recefore_with_a_kill_after_timeout(tmp_path: Path, monkeypatch):
    """Run one's ReCefore invocation uses `timeout --kill-after ...`, not a plain `timeout N`,
    so a child that catches or ignores TERM still gets forced to exit (reviewer finding:
    GNU `timeout` never escalates to KILL on its own without `--kill-after`)."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    calls: list = []
    evalrun.run_one(
        0.2,
        7,
        "s1",
        str(store),
        str(recefore),
        str(staging_root),
        run_subprocess=make_fake_run_subprocess(calls=calls),
        capture_subprocess=_fake_capture_subprocess,
    )
    recefore_argv = calls[1][0]
    assert recefore_argv[:5] == [
        "setsid",
        "timeout",
        "--kill-after",
        str(evalrun.RECEFORE_KILL_AFTER_S),
        str(evalrun.RECEFORE_TIMEOUT_S),
    ]


def test_kill_after_actually_kills_a_term_ignoring_child():
    """A live (no mock, no root, no Mininet) check that the `timeout --kill-after ...` shape
    evalrun uses actually forces a child that ignores SIGTERM to die, rather than hanging
    forever -- the reviewer's concrete claim about plain `timeout N`'s behavior."""
    if shutil.which("timeout") is None:
        pytest.skip("GNU coreutils 'timeout' is not installed")
    ignore_term_and_sleep = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    start = time.monotonic()
    proc = subprocess.run(
        ["timeout", "--kill-after", "1", "1", sys.executable, "-c", ignore_term_and_sleep],
        timeout=10,
    )
    wall_s = time.monotonic() - start
    assert proc.returncode in (137, -9)  # killed by SIGKILL after ignoring TERM
    assert wall_s < 5.0


# -- git lookups under sudo (orchestrator trial: recefore_git_commit came back null) -------------


def test_default_capture_subprocess_raises_with_stderr_on_a_nonzero_exit():
    """Default capture subprocess raises CalledProcessError carrying stderr (not a silently empty
    stdout) when the command exits non-zero -- the real failure reason must survive to the caller."""
    argv = [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(1)"]
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        evalrun.default_capture_subprocess(argv)
    assert exc_info.value.stderr == "boom"


def test_git_commit_scopes_safe_directory_and_no_optional_locks_to_the_repo(tmp_path: Path):
    """_git_commit's own git invocation passes `-c safe.directory=<repo_dir>` (scoped to this one
    call, never `git config --global`) and `--no-optional-locks` -- the orchestrator's live trial
    found `recefore_git_commit` came back null because `git -C <recefore> rev-parse HEAD`, run as
    root, refused a directory root does not own."""
    calls: list = []

    def _fake(argv, *, cwd=None):
        calls.append(argv)
        return "deadbeef"

    real_repo = str(Path(tmp_path).resolve())
    result = evalrun._git_commit(str(tmp_path), _fake)
    assert result == "deadbeef"
    assert calls[0][0] == "git"
    assert "--no-optional-locks" in calls[0]
    assert f"safe.directory={real_repo}" in calls[0]
    assert calls[0][calls[0].index("-C") + 1] == real_repo


def test_git_commit_stores_the_error_string_rather_than_a_bare_none_on_failure(tmp_path: Path):
    """_git_commit never returns a bare null for the caller to store silently -- a failure (git
    refusing the repo under sudo, a non-repo directory, ...) comes back as a readable
    `"error: ..."` string instead, so it is visible in meta.json rather than indistinguishable from
    "produced no output"."""

    def _fails(argv, *, cwd=None):
        raise subprocess.CalledProcessError(128, argv, output="", stderr="fatal: detected dubious ownership\n")

    result = evalrun._git_commit(str(tmp_path), _fails)
    assert result == "error: fatal: detected dubious ownership"


def test_git_commit_stores_the_error_string_on_oserror_too(tmp_path: Path):
    """_git_commit also stores an error string (not None) when git itself cannot even be run."""

    def _missing(argv, *, cwd=None):
        raise FileNotFoundError("git not found")

    result = evalrun._git_commit(str(tmp_path), _missing)
    assert result.startswith("error: ")


def test_git_dirty_scopes_safe_directory_and_reports_true_false_or_an_error_string(tmp_path: Path):
    """_git_dirty uses the same safe.directory/--no-optional-locks scoping as _git_commit, returns
    a plain bool when git could be asked, and never returns None on failure."""
    calls: list = []

    def _clean(argv, *, cwd=None):
        calls.append(argv)
        return ""

    assert evalrun._git_dirty(str(tmp_path), _clean) is False
    real_repo = str(Path(tmp_path).resolve())
    assert f"safe.directory={real_repo}" in calls[0]
    assert "--no-optional-locks" in calls[0]

    def _dirty(argv, *, cwd=None):
        return "M some_file.py\n"

    assert evalrun._git_dirty(str(tmp_path), _dirty) is True

    def _fails(argv, *, cwd=None):
        raise subprocess.CalledProcessError(128, argv, output="", stderr="fatal: boom\n")

    assert evalrun._git_dirty(str(tmp_path), _fails) == "error: fatal: boom"


# -- default_chown_tree --------------------------------------------------------


def test_default_chown_tree_is_a_noop_without_sudo_env(tmp_path: Path, monkeypatch):
    """Default chown tree is a no-op without SUDO_UID/SUDO_GID set."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    path = tmp_path / "f.txt"
    path.write_text("x", encoding="utf-8")
    evalrun.default_chown_tree(str(path))  # must not raise


# -- default_chown_dir ----------------------------------------------------------


def test_default_chown_dir_is_a_noop_without_sudo_env(tmp_path: Path, monkeypatch):
    """Default chown dir is a no-op without SUDO_UID/SUDO_GID set."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    evalrun.default_chown_dir(str(tmp_path))  # must not raise


def test_default_chown_dir_refuses_a_symlink(tmp_path: Path, monkeypatch):
    """Default chown dir refuses to chown through a symlink rather than silently following it."""
    monkeypatch.setenv("SUDO_UID", str(os.getuid()))
    monkeypatch.setenv("SUDO_GID", str(os.getgid()))
    target = tmp_path / "real_target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        evalrun.default_chown_dir(str(link))


# -- run_one / run_all: chown newly-created ancestor dirs (orchestrator trial) --


def test_run_one_chowns_newly_created_stage_and_p_dirs_but_not_preexisting_ones(tmp_path: Path, monkeypatch):
    """Run one chowns `<store>/<stage>` and `<store>/<stage>/p<p>` when its own makedirs call
    creates them fresh, but leaves an already-existing ancestor alone on a later run that reuses
    it -- the orchestrator's live trial found both of these directories left root-owned while
    every run dir underneath them was correctly chowned."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    chowned_dirs: list = []
    evalrun.run_one(
        0.2, 7, "s1", str(store), str(recefore), str(staging_root),
        run_subprocess=make_fake_run_subprocess(),
        capture_subprocess=_fake_capture_subprocess,
        chown_dir=chowned_dirs.append,
    )
    stage_dir = store / "s1"
    p_dir = store / "s1" / "p0.2"
    assert set(chowned_dirs) == {str(stage_dir), str(p_dir)}

    # A second seed under the SAME (stage, p): both ancestors already exist now, so neither
    # should be chowned again.
    chowned_dirs.clear()
    evalrun.run_one(
        0.2, 8, "s1", str(store), str(recefore), str(staging_root),
        run_subprocess=make_fake_run_subprocess(),
        capture_subprocess=_fake_capture_subprocess,
        chown_dir=chowned_dirs.append,
    )
    assert chowned_dirs == []


def test_run_one_default_chown_dir_actually_fixes_ownership_of_the_ancestors(tmp_path: Path, monkeypatch):
    """A live (no fake chown_dir) run_one call, with SUDO_UID/SUDO_GID set to this process's own
    uid/gid, leaves `<store>/<stage>` and `<store>/<stage>/p<p>` owned by that uid/gid -- not just
    recorded as "would have been chowned" by a recording fake."""
    monkeypatch.setenv("SUDO_UID", str(os.getuid()))
    monkeypatch.setenv("SUDO_GID", str(os.getgid()))
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    evalrun.run_one(
        0.2, 7, "s1", str(store), str(recefore), str(staging_root),
        run_subprocess=make_fake_run_subprocess(),
        capture_subprocess=_fake_capture_subprocess,
    )
    stage_dir = store / "s1"
    p_dir = store / "s1" / "p0.2"
    assert stage_dir.stat().st_uid == os.getuid()
    assert p_dir.stat().st_uid == os.getuid()


def test_run_all_chowns_the_store_dir_only_when_it_creates_it(tmp_path: Path, monkeypatch):
    """Run all chowns `store` itself when its own makedirs call creates it fresh, but leaves an
    already-existing store dir alone -- the same "only what this call created" rule as `run_one`'s
    ancestor handling, applied to the one ancestor above everything `run_one` ever sees."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    chowned_dirs: list = []
    evalrun.run_all(
        [0.2], [1], "s1", str(store), str(recefore), str(staging_root),
        run_subprocess=make_fake_run_subprocess(),
        capture_subprocess=_fake_capture_subprocess,
        chown_dir=chowned_dirs.append,
    )
    assert str(store) in chowned_dirs

    # store already exists now: a second sweep must not chown it again.
    chowned_dirs.clear()
    evalrun.run_all(
        [0.2], [2], "s1", str(store), str(recefore), str(staging_root),
        run_subprocess=make_fake_run_subprocess(),
        capture_subprocess=_fake_capture_subprocess,
        chown_dir=chowned_dirs.append,
    )
    assert str(store) not in chowned_dirs


# -- run_one: happy path -------------------------------------------------------


def test_run_one_happy_path_stores_exactly_the_right_files_and_writes_done(tmp_path: Path, monkeypatch):
    """Run one happy path stores exactly the right files and writes DONE."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    calls: list = []
    outcome = evalrun.run_one(
        0.2,
        7,
        "s1",
        str(store),
        str(recefore),
        str(staging_root),
        run_subprocess=make_fake_run_subprocess(calls=calls),
        capture_subprocess=_fake_capture_subprocess,
    )

    assert outcome.status == "done"
    assert outcome.recefore_rc == 0
    assert outcome.valid is True
    assert outcome.flags == []

    store_run_dir = store / "s1" / "p0.2" / "7"
    assert (store_run_dir / "DONE").exists()
    assert (store_run_dir / "results.json").exists()
    assert (store_run_dir / "meta.json").exists()

    experiment_files = list(store_run_dir.glob("*.experiment.json"))
    yaml_files = list(store_run_dir.glob("*.yaml"))
    assert len(experiment_files) == 1
    assert len(yaml_files) == 1
    name = yaml_files[0].stem
    for host in evalgen.AGENTS:
        assert (store_run_dir / name / f"h{host}.json").exists()
        node_dir = store_run_dir / "debug" / "node_dirs" / f"h{host}"
        assert (node_dir / "semindex" / "events.jsonl").exists()
        assert (node_dir / "semindex_agent.log").exists()
        # map.md: node_dirs also holds Cefore's own key material -- never copied.
        assert not (node_dir / "default-private-key").exists()

    # Staging is gone entirely: the per-run staging dir evalrun created was
    # removed by the guard, and nothing else was left behind under the root.
    assert list(staging_root.iterdir()) == []

    # mn -c ran before the ReCefore invocation, in that order.
    assert calls[0][0][:2] == ["mn", "-c"]
    assert "disaster" in calls[1][0]


def test_run_one_meta_json_has_the_documented_fields(tmp_path: Path, monkeypatch):
    """Run one meta.json has the documented fields."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    evalrun.run_one(
        0.1,
        3,
        "s1",
        str(store),
        str(recefore),
        str(staging_root),
        run_subprocess=make_fake_run_subprocess(),
        capture_subprocess=_fake_capture_subprocess,
    )

    meta_path = store / "s1" / "p0.1" / "3" / "meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    assert set(meta) == {
        "p",
        "seed",
        "stage",
        "semindex_git_commit",
        "semindex_git_dirty",
        "recefore_git_commit",
        "recefore_git_dirty",
        "wall_clock_start",
        "wall_clock_end",
        "recefore_exit_code",
        "host_python",
        "validity",
    }
    assert meta["p"] == 0.1
    assert meta["seed"] == 3
    assert meta["stage"] == "s1"
    assert meta["recefore_exit_code"] == 0
    assert meta["validity"]["valid"] is True
    assert set(meta["host_python"]) == {"evalrun", "recefore_venv"}


def test_run_one_flags_an_invalid_run_but_still_completes(tmp_path: Path, monkeypatch):
    """Run one flags an invalid run (one node's output missing) but still writes DONE."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    outcome = evalrun.run_one(
        0.2,
        7,
        "s1",
        str(store),
        str(recefore),
        str(staging_root),
        run_subprocess=make_fake_run_subprocess(skip_node=1),
        capture_subprocess=_fake_capture_subprocess,
    )

    assert outcome.status == "done"
    assert outcome.valid is False
    assert "missing_events:h1" in outcome.flags
    assert (store / "s1" / "p0.2" / "7" / "DONE").exists()


# -- resume semantics ----------------------------------------------------------


def test_run_one_skips_a_run_whose_done_marker_already_exists(tmp_path: Path, monkeypatch):
    """Run one skips a run whose DONE marker already exists, calling nothing."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    recefore = tmp_path / "recefore"
    recefore.mkdir()
    store_run_dir = store / "s1" / "p0.2" / "7"
    store_run_dir.mkdir(parents=True)
    (store_run_dir / "DONE").write_text("", encoding="utf-8")
    (store_run_dir / "results.json").write_text("[]", encoding="utf-8")

    def _must_not_run(argv, *, cwd=None):
        raise AssertionError("run_subprocess must not be called for a run already marked DONE")

    outcome = evalrun.run_one(
        0.2, 7, "s1", str(store), str(recefore), str(staging_root), run_subprocess=_must_not_run
    )

    assert outcome.status == "skipped"
    assert (store_run_dir / "results.json").read_text(encoding="utf-8") == "[]"


def test_run_one_redoes_a_partial_store_dir_from_scratch(tmp_path: Path, monkeypatch):
    """Run one removes a partial (no-DONE) store dir and redoes the run from scratch."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    recefore = tmp_path / "recefore"
    recefore.mkdir()
    store_run_dir = store / "s1" / "p0.2" / "7"
    store_run_dir.mkdir(parents=True)
    (store_run_dir / "stale_leftover.txt").write_text("from an interrupted attempt", encoding="utf-8")

    outcome = evalrun.run_one(
        0.2,
        7,
        "s1",
        str(store),
        str(recefore),
        str(staging_root),
        run_subprocess=make_fake_run_subprocess(),
        capture_subprocess=_fake_capture_subprocess,
    )

    assert outcome.status == "done"
    assert (store_run_dir / "DONE").exists()
    assert not (store_run_dir / "stale_leftover.txt").exists()


# -- run_all: refuses a sweep on a dirty/undetermined semindex tree ------------


def test_run_all_refuses_a_dirty_semindex_tree_without_allow_dirty(tmp_path: Path, monkeypatch):
    """Run all refuses to start the sweep at all when the semindex tree is reported dirty, before
    even creating the store or staging dirs -- orchestrator trial seeds 2-3 were recorded dirty
    because semindex's own source was being edited mid-sweep."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    def _dirty_capture(argv, *, cwd=None):
        if argv[0] == "git" and "status" in argv:
            return "M some_file.py\n"
        return _fake_capture_subprocess(argv, cwd=cwd)

    def _must_not_run(argv, *, cwd=None):
        raise AssertionError("run_subprocess must not be called: the dirty-tree guard should fail first")

    with pytest.raises(evalrun.DirtySemindexError):
        evalrun.run_all(
            [0.2], [1], "s1", str(store), str(recefore), str(staging_root),
            run_subprocess=_must_not_run,
            capture_subprocess=_dirty_capture,
        )
    assert not store.exists()
    assert not staging_root.exists()


def test_run_all_refuses_when_the_semindex_dirty_status_is_undetermined(tmp_path: Path, monkeypatch):
    """Run all also refuses when the dirty check itself fails (an `"error: ..."` string, not True
    or False) -- an undetermined tree must not be silently treated as clean."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    def _broken_capture(argv, *, cwd=None):
        if argv[0] == "git" and "status" in argv:
            raise subprocess.CalledProcessError(128, argv, output="", stderr="fatal: boom\n")
        return _fake_capture_subprocess(argv, cwd=cwd)

    def _must_not_run(argv, *, cwd=None):
        raise AssertionError("run_subprocess must not be called: the dirty-tree guard should fail first")

    with pytest.raises(evalrun.DirtySemindexError):
        evalrun.run_all(
            [0.2], [1], "s1", str(store), str(recefore), str(staging_root),
            run_subprocess=_must_not_run,
            capture_subprocess=_broken_capture,
        )


def test_run_all_proceeds_when_allow_dirty_is_set(tmp_path: Path, monkeypatch):
    """Run all proceeds through a dirty semindex tree when --allow-dirty's own flag is passed."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    def _dirty_capture(argv, *, cwd=None):
        if argv[0] == "git" and "status" in argv:
            return "M some_file.py\n"
        return _fake_capture_subprocess(argv, cwd=cwd)

    outcomes = evalrun.run_all(
        [0.2], [1], "s1", str(store), str(recefore), str(staging_root),
        run_subprocess=make_fake_run_subprocess(),
        capture_subprocess=_dirty_capture,
        allow_dirty=True,
    )
    assert outcomes[0].status == "done"
    assert (store / "s1" / "p0.2" / "1" / "DONE").exists()


def test_run_all_proceeds_when_the_semindex_tree_is_confirmed_clean(tmp_path: Path, monkeypatch):
    """Run all's existing happy-path behavior survives: a confirmed-clean tree (the default fake's
    `git status --porcelain` returns "") never raises DirtySemindexError, with no --allow-dirty."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    outcomes = evalrun.run_all(
        [0.2], [1], "s1", str(store), str(recefore), str(staging_root),
        run_subprocess=make_fake_run_subprocess(),
        capture_subprocess=_fake_capture_subprocess,
    )
    assert outcomes[0].status == "done"


def test_main_exits_2_and_prints_a_message_on_a_dirty_tree(tmp_path: Path, monkeypatch, capsys):
    """main() surfaces DirtySemindexError as a clear stderr message and exit code 2, not a
    traceback, and cleans up the empty staging dir it created before failing."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    recefore = tmp_path / "recefore"
    recefore.mkdir()
    staging_parent = tmp_path / "staging_parent"
    staging_parent.mkdir()

    def _dirty_run_all(*args, **kwargs):
        raise evalrun.DirtySemindexError("semindex tree is dirty")

    monkeypatch.setattr(evalrun, "run_all", _dirty_run_all)
    rc = evalrun.main(
        ["--stage", "s1", "--p", "0.2", "--seeds", "1", "--store", str(store), "--recefore", str(recefore),
         "--staging", str(staging_parent)]
    )
    assert rc == 2
    assert "semindex tree is dirty" in capsys.readouterr().err


# -- run_all: ordering, logging, crash isolation -------------------------------


def test_run_all_logs_one_line_per_run_in_seed_major_order(tmp_path: Path, monkeypatch, capsys):
    """Run all logs one line per run in seed-major order."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    outcomes = evalrun.run_all(
        [0.2, 0.0],
        [2, 1],
        "s1",
        str(store),
        str(recefore),
        str(staging_root),
        run_subprocess=make_fake_run_subprocess(),
        capture_subprocess=_fake_capture_subprocess,
    )

    assert [(o.p, o.seed) for o in outcomes] == [(0.0, 1), (0.2, 1), (0.0, 2), (0.2, 2)]
    for p, seed in [(0.0, 1), (0.2, 1), (0.0, 2), (0.2, 2)]:
        assert (store / "s1" / f"p{evalgen._format_p(p)}" / str(seed) / "DONE").exists()


def test_run_all_continues_past_one_run_raising(tmp_path: Path, monkeypatch, capsys):
    """Run all continues past one run raising, completing every other (p, seed) pair."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    good_fake = make_fake_run_subprocess()

    def _flaky(argv, *, cwd=None):
        if argv[:2] != ["mn", "-c"] and "--config" in argv:
            yaml_path = argv[argv.index("--config") + 1]
            if _load_generated_yaml(yaml_path)["seed"] == 2:
                raise RuntimeError("simulated crash")
        return good_fake(argv, cwd=cwd)

    evalrun.run_all(
        [0.2],
        [1, 2],
        "s1",
        str(store),
        str(recefore),
        str(staging_root),
        run_subprocess=_flaky,
        capture_subprocess=_fake_capture_subprocess,
    )

    out = capsys.readouterr().out
    assert "status=error" in out
    assert (store / "s1" / "p0.2" / "1" / "DONE").exists()
    assert not (store / "s1" / "p0.2" / "2" / "DONE").exists()
