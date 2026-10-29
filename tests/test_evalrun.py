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
import stat
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


def test_default_chown_tree_sets_exact_modes_not_just_group_write_on_top_of_umask(tmp_path: Path, monkeypatch):
    """2026-10-23 regression (external review, P1 #2): a file/dir created under a permissive
    umask (e.g. 0, giving 0777/0666) used to come out of chown_tree still group- AND
    other-writable (the old code only OR'd in g+w onto whatever umask left behind). Every
    directory must end up at exactly 0755 and every file at exactly 0644, regardless of umask.
    """
    monkeypatch.setenv("SUDO_UID", str(os.getuid()))
    monkeypatch.setenv("SUDO_GID", str(os.getgid()))
    root = tmp_path / "tree"
    sub = root / "sub"
    sub.mkdir(parents=True)
    f = sub / "f.txt"
    f.write_text("x", encoding="utf-8")
    # Simulate a permissive umask having already left these too open --
    # chown_tree must still force them down to the canonical modes.
    os.chmod(root, 0o777)
    os.chmod(sub, 0o777)
    os.chmod(f, 0o666)

    evalrun.default_chown_tree(str(root))

    assert stat.S_IMODE(os.stat(root).st_mode) == 0o755
    assert stat.S_IMODE(os.stat(sub).st_mode) == 0o755
    assert stat.S_IMODE(os.stat(f).st_mode) == 0o644


def test_default_chown_dir_sets_exactly_0755_not_just_group_write_on_top_of_umask(tmp_path: Path, monkeypatch):
    """Same regression as the chown_tree test above, for the single-directory `chown_dir` path."""
    monkeypatch.setenv("SUDO_UID", str(os.getuid()))
    monkeypatch.setenv("SUDO_GID", str(os.getgid()))
    path = tmp_path / "ancestor"
    path.mkdir()
    os.chmod(path, 0o777)

    evalrun.default_chown_dir(str(path))

    assert stat.S_IMODE(os.stat(path).st_mode) == 0o755


# -- _check_dir_permissions / _verify_safe_store_permissions (external review, P1 #2) -----------


def _patched_lstat(monkeypatch, path: str, *, uid: int, mode: int, is_symlink: bool = False):
    """Make `os.lstat(path)` (which `_check_dir_permissions` reads through) report a crafted
    (st_uid, st_mode) for exactly this one path, falling through to the real `os.lstat` for every
    other path. Lets a test exercise an ownership, or a symlink, this process cannot actually
    create on disk (a directory "owned" by some other, unrelated uid) without needing real root
    -- the "injected stat" the external review's fix asked for.
    """
    real_lstat = os.lstat
    full_mode = (stat.S_IFLNK if is_symlink else stat.S_IFDIR) | mode

    def fake_lstat(p, *args, **kwargs):
        if os.fspath(p) == path:
            return os.stat_result((full_mode, 0, 0, 1, uid, uid, 0, 0, 0, 0))
        return real_lstat(p, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", fake_lstat)


def test_check_dir_permissions_accepts_owner_match_not_group_or_other_writable(tmp_path: Path):
    """The ordinary, locked-down case: owned by the invoking user, mode 0755."""
    path = tmp_path / "d"
    path.mkdir()
    os.chmod(path, 0o755)
    evalrun._check_dir_permissions(str(path), invoking_uid=os.getuid())  # must not raise


def test_check_dir_permissions_rejects_group_writable_even_when_owner_matches(tmp_path: Path):
    """Ownership alone is not enough -- a group-writable directory still lets any OTHER group
    member replace one of its entries, regardless of who owns the directory itself."""
    path = tmp_path / "d"
    path.mkdir()
    os.chmod(path, 0o775)
    with pytest.raises(ValueError, match="writable by group or others"):
        evalrun._check_dir_permissions(str(path), invoking_uid=os.getuid())


def test_check_dir_permissions_rejects_other_writable(tmp_path: Path):
    """The other-writable bit is checked independently of the group-writable one."""
    path = tmp_path / "d"
    path.mkdir()
    os.chmod(path, 0o757)
    with pytest.raises(ValueError, match="writable by group or others"):
        evalrun._check_dir_permissions(str(path), invoking_uid=os.getuid())


def test_check_dir_permissions_accepts_root_owned_regardless_of_invoking_uid(tmp_path: Path, monkeypatch):
    """A root-owned, non-group/other-writable directory is accepted even when its uid does not
    match the invoking user -- root itself is always a trusted owner."""
    path = tmp_path / "d"
    path.mkdir()
    _patched_lstat(monkeypatch, str(path), uid=0, mode=0o755)
    evalrun._check_dir_permissions(str(path), invoking_uid=12345)  # must not raise


def test_check_dir_permissions_rejects_owner_mismatch_when_not_root(tmp_path: Path, monkeypatch):
    """2026-10-23 regression (external review, P1 #2): a directory owned by neither root nor the
    invoking user -- e.g. another group member -- is refused outright, even at mode 0755: the
    threat is a user who already owns (or can be handed ownership of) a path component on the
    run's path, not only one who merely has write access to it.
    """
    path = tmp_path / "d"
    path.mkdir()
    _patched_lstat(monkeypatch, str(path), uid=99999, mode=0o755)
    with pytest.raises(ValueError, match="not root or the invoking user"):
        evalrun._check_dir_permissions(str(path), invoking_uid=os.getuid())


def test_check_dir_permissions_rejects_a_symlink_even_at_a_trusted_owner_and_mode(
    tmp_path: Path, monkeypatch
):
    """2026-10-23 regression (external review, P1 #2): a symlink is refused on sight, via a
    single `os.lstat`, before its owner/mode (which would describe the LINK itself, not whatever
    it points at) are ever inspected -- `os.stat`-then-`os.path.islink` would mean two syscalls
    with a gap between them for an attacker to swap the target in.
    """
    path = tmp_path / "d"
    path.mkdir()
    _patched_lstat(monkeypatch, str(path), uid=0, mode=0o755, is_symlink=True)
    with pytest.raises(ValueError, match="is a symlink below the store root"):
        evalrun._check_dir_permissions(str(path), invoking_uid=os.getuid())


def test_check_dir_permissions_is_a_noop_when_the_path_does_not_exist(tmp_path: Path):
    """Nothing there yet to check -- it will be created fresh by `run_one` itself."""
    evalrun._check_dir_permissions(str(tmp_path / "does-not-exist"), invoking_uid=os.getuid())


def test_verify_safe_store_permissions_raises_when_the_store_root_itself_is_a_symlink(
    tmp_path: Path, monkeypatch
):
    """2026-10-23 regression (external review, P1 #2): the store root is checked with
    `_check_dir_permissions` directly, not behind an `os.path.isdir(store) and not
    os.path.islink(store)` guard -- that guard used to be False (and so skip the check entirely)
    for a symlinked store, since `os.path.isdir` follows symlinks and calls a symlink-to-a-
    directory a directory too."""
    monkeypatch.setenv("SUDO_UID", str(os.getuid()))
    store = tmp_path / "store"
    store.mkdir()
    _patched_lstat(monkeypatch, str(store), uid=os.getuid(), mode=0o755, is_symlink=True)
    with pytest.raises(ValueError, match="is a symlink below the store root"):
        evalrun._verify_safe_store_permissions(str(store / "s1" / "p0.2" / "7"), str(store))


def test_verify_safe_store_permissions_is_a_noop_without_sudo_uid(tmp_path: Path, monkeypatch):
    """Outside `sudo` there is no invoking user this check could protect -- same no-op
    convention as default_chown_tree/default_chown_dir."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    store = tmp_path / "store"
    store.mkdir()
    os.chmod(store, 0o777)  # would fail the check below if it ran at all
    evalrun._verify_safe_store_permissions(str(store / "s1" / "p0.2" / "7"), str(store))  # must not raise


def test_verify_safe_store_permissions_skips_components_that_do_not_exist_yet(tmp_path: Path, monkeypatch):
    """Nothing on the run's path below an already-locked-down `store` exists yet -- there is
    nothing there for a symlink or a bad permission to have been swapped into, so this passes."""
    monkeypatch.setenv("SUDO_UID", str(os.getuid()))
    store = tmp_path / "store"
    store.mkdir()
    os.chmod(store, 0o755)
    evalrun._verify_safe_store_permissions(str(store / "s1" / "p0.2" / "7"), str(store))  # must not raise


def test_verify_safe_store_permissions_raises_when_store_itself_is_group_writable(tmp_path: Path, monkeypatch):
    """The store root is checked unconditionally, not just the components below it."""
    monkeypatch.setenv("SUDO_UID", str(os.getuid()))
    store = tmp_path / "store"
    store.mkdir()
    os.chmod(store, 0o775)
    with pytest.raises(ValueError, match="writable by group or others"):
        evalrun._verify_safe_store_permissions(str(store / "s1" / "p0.2" / "7"), str(store))


def test_verify_safe_store_permissions_raises_on_an_existing_ancestor_owned_by_another_user(
    tmp_path: Path, monkeypatch
):
    """2026-10-23 regression (external review, P1 #2): the exact repro -- `<store>/<stage>`
    (here `s1`) is the directory another group member could otherwise swap for a symlink. An
    ancestor owned by neither root nor the invoking user is refused before anything is read
    from or written through it, regardless of the seed/p leaf (which does not exist yet)."""
    monkeypatch.setenv("SUDO_UID", str(os.getuid()))
    store = tmp_path / "store"
    store.mkdir()
    os.chmod(store, 0o755)
    stage_dir = store / "s1"
    stage_dir.mkdir()
    os.chmod(stage_dir, 0o755)
    _patched_lstat(monkeypatch, str(stage_dir), uid=99999, mode=0o755)

    with pytest.raises(ValueError, match="not root or the invoking user"):
        evalrun._verify_safe_store_permissions(str(stage_dir / "p0.2" / "7"), str(store))


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


def test_run_one_refuses_to_write_when_an_ancestor_becomes_group_writable_during_the_recefore_wait(
    tmp_path: Path, monkeypatch
):
    """2026-10-23 regression (external review, P1 #2): the actual race the fix closes. `<store>/s1`
    starts out properly locked down (owned by the invoking user, mode 0755, exactly what a prior
    run's own chown_dir would have left it at) -- but the fake ReCefore subprocess itself, standing
    in for "another group member acted while the real ~160s run was in progress", loosens it to
    group-writable partway through. `run_one`'s re-check right after that subprocess returns must
    catch this before `os.makedirs`/`_copy_run_outputs`/`_write_json` ever touch the store -- the
    one check at the top of this function, before the wait, cannot see a change made during it.
    """
    monkeypatch.setenv("SUDO_UID", str(os.getuid()))
    monkeypatch.setenv("SUDO_GID", str(os.getgid()))
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    store.mkdir()
    os.chmod(store, 0o755)
    stage_dir = store / "s1"
    stage_dir.mkdir()
    os.chmod(stage_dir, 0o755)

    base_fake = make_fake_run_subprocess()

    def attacking_run_subprocess(argv, *, cwd=None):
        if "disaster" in argv and "--config" in argv:
            # The attack: loosen the stage dir's permissions exactly while
            # the long-running ReCefore command is "in progress" (this fake
            # stands in for that whole multi-minute wait).
            os.chmod(stage_dir, 0o775)
        return base_fake(argv, cwd=cwd)

    with pytest.raises(ValueError, match="writable by group or others"):
        evalrun.run_one(
            0.2, 7, "s1", str(store), str(recefore), str(staging_root),
            run_subprocess=attacking_run_subprocess,
            capture_subprocess=_fake_capture_subprocess,
        )

    # Caught before anything was written to the store: no seed dir, no meta.json.
    assert not (stage_dir / "p0.2" / "7").exists()


def test_run_one_refuses_to_write_when_an_ancestor_is_swapped_for_a_symlink_during_the_wait(
    tmp_path: Path, monkeypatch
):
    """2026-10-23 regression (external review, P1 #2): the review's own literal repro -- not a
    permission loosening, but `<store>/s1` itself replaced with a symlink to an external
    directory while the fake ReCefore run is "in progress". The re-check right after that
    subprocess returns (`_verify_safe_run_dir`, re-run alongside `_verify_safe_store_permissions`)
    must catch this before anything is written through the swapped-in symlink.
    """
    monkeypatch.setenv("SUDO_UID", str(os.getuid()))
    monkeypatch.setenv("SUDO_GID", str(os.getgid()))
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    store.mkdir()
    os.chmod(store, 0o755)
    stage_dir = store / "s1"
    stage_dir.mkdir()
    os.chmod(stage_dir, 0o755)
    external = tmp_path / "external"
    external.mkdir()

    base_fake = make_fake_run_subprocess()

    def attacking_run_subprocess(argv, *, cwd=None):
        if "disaster" in argv and "--config" in argv:
            # The attack: swap the real stage dir for a symlink pointing
            # outside `store` entirely, while the long-running ReCefore
            # command is "in progress".
            aside = tmp_path / "s1-real"
            os.rename(stage_dir, aside)
            os.symlink(external, stage_dir, target_is_directory=True)
        return base_fake(argv, cwd=cwd)

    with pytest.raises(ValueError, match="is a symlink below the store root"):
        evalrun.run_one(
            0.2, 7, "s1", str(store), str(recefore), str(staging_root),
            run_subprocess=attacking_run_subprocess,
            capture_subprocess=_fake_capture_subprocess,
        )

    # Caught before anything was written through the swapped-in symlink.
    assert not (external / "p0.2" / "7" / "meta.json").exists()


def test_run_one_creates_directories_at_exactly_0755_regardless_of_umask(tmp_path: Path, monkeypatch):
    """2026-10-23 regression (external review, P1 #2): every directory `run_one` creates while
    copying/writing into the store -- not just what `chown_tree` fixes up at the very end -- must
    already be at `_STORE_DIR_MODE` (0755), even under a maximally permissive umask. Checked via
    the injected `capture_subprocess` fake: `_build_meta`'s own git calls run strictly after
    `_copy_run_outputs` and strictly before the final `chown_tree(store_run_dir)`, so asserting
    the mode from inside it catches exactly the gap this fix closes.
    """
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    old_umask = os.umask(0)
    try:
        store = tmp_path / "store"
        staging_root = tmp_path / "staging"
        staging_root.mkdir()
        recefore = tmp_path / "recefore"
        recefore.mkdir()

        seen_modes: list[int] = []

        def checking_capture_subprocess(argv, *, cwd=None):
            store_run_dir = store / "s1" / "p0.2" / "7"
            if store_run_dir.is_dir():
                seen_modes.append(stat.S_IMODE(os.stat(store_run_dir).st_mode))
            return _fake_capture_subprocess(argv, cwd=cwd)

        evalrun.run_one(
            0.2, 7, "s1", str(store), str(recefore), str(staging_root),
            run_subprocess=make_fake_run_subprocess(),
            capture_subprocess=checking_capture_subprocess,
        )
    finally:
        os.umask(old_umask)

    assert seen_modes, "the fake capture_subprocess was never called with store_run_dir present"
    assert all(mode == 0o755 for mode in seen_modes), seen_modes


def test_run_all_refuses_the_whole_sweep_up_front_when_the_store_is_group_writable(
    tmp_path: Path, monkeypatch
):
    """2026-10-23 regression (external review, P1 #2): without this gate, a mis-permissioned
    `store` would not stop the sweep at all -- `run_one`'s own copy of this same check lives
    inside `run_all`'s per-pair try/except (there precisely so one run's bug never aborts an
    entire overnight sweep), so every single (p, seed) pair would fail individually and print
    `status=error`, with `run_all` itself still returning normally. Raising here, before the loop
    even starts, is what actually stops the sweep -- the same "fail loudly before touching
    anything" treatment `DirtySemindexError` gets for a dirty semindex tree. A `run_subprocess`
    that asserts if ever called proves `run_one` was never even reached.
    """
    monkeypatch.setenv("SUDO_UID", str(os.getuid()))
    monkeypatch.setenv("SUDO_GID", str(os.getgid()))
    store = tmp_path / "store"
    store.mkdir()
    os.chmod(store, 0o775)  # group-writable: exactly what the fix refuses
    staging_root = tmp_path / "staging"
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    def never_called(argv, *, cwd=None):
        raise AssertionError(f"run_subprocess must never be called, got {argv!r}")

    with pytest.raises(ValueError, match="writable by group or others"):
        evalrun.run_all(
            [0.2], [1], "s1", str(store), str(recefore), str(staging_root),
            run_subprocess=never_called,
            capture_subprocess=_fake_capture_subprocess,
        )


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
        "k",
        "state_poll_interval_choices_s",
        "state_poll_choice_jitter_frac",
        "max_in_flight_per_neighbor",
        "store_synchronous",
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
    assert meta["k"] == 1
    assert meta["state_poll_interval_choices_s"] is None
    assert meta["state_poll_choice_jitter_frac"] is None
    assert meta["max_in_flight_per_neighbor"] is None
    assert meta["store_synchronous"] is None
    assert meta["recefore_exit_code"] == 0
    assert meta["validity"]["valid"] is True
    assert set(meta["host_python"]) == {"evalrun", "recefore_venv"}


def test_run_one_generates_with_and_records_k_and_interval_choices(tmp_path: Path, monkeypatch):
    """k and the interval choices reach the generated run (stored experiment) and
    meta.json, the run still validates, and the store path stays <stage>/p<p>/<seed>."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.delenv("SUDO_GID", raising=False)
    store = tmp_path / "store"
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    recefore = tmp_path / "recefore"
    recefore.mkdir()

    evalrun.run_one(
        0.4,
        3,
        "s2",
        str(store),
        str(recefore),
        str(staging_root),
        run_subprocess=make_fake_run_subprocess(),
        capture_subprocess=_fake_capture_subprocess,
        k=2,
        state_poll_interval_choices_s=[1.0, 2.0],
        max_in_flight_per_neighbor=32,
        store_synchronous="NORMAL",
    )

    run_dir = store / "s2" / "p0.4" / "3"
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["k"] == 2
    assert meta["state_poll_interval_choices_s"] == [1.0, 2.0]
    assert meta["state_poll_choice_jitter_frac"] == 0.1  # the default, made explicit
    assert meta["max_in_flight_per_neighbor"] == 32
    assert meta["store_synchronous"] == "NORMAL"
    # evalvalid matches record_created to the schedule by seq index, so 2 records
    # per generation time (60 per origin) is still a valid run.
    assert meta["validity"]["valid"] is True
    assert meta["validity"]["flags"] == []
    experiment = json.loads((run_dir / "s2_p040_seed3.experiment.json").read_text(encoding="utf-8"))
    assert len(experiment["schedule"]["0"]) == 2 * evalgen.RECORD_COUNT
    assert experiment["state_poll_interval_choices_s"] == [1.0, 2.0]
    assert experiment["max_in_flight_per_neighbor"] == 32
    h0 = json.loads((run_dir / "s2_p040_seed3" / "h0.json").read_text(encoding="utf-8"))
    assert h0["max_in_flight_per_neighbor"] == 32
    assert experiment["store_synchronous"] == "NORMAL"
    assert h0["store_synchronous"] == "NORMAL"


def test_main_refuses_a_non_finite_interval_choice_before_any_run(tmp_path: Path, monkeypatch, capsys):
    """An infinite choice is refused once, up front (exit 2), and no run is attempted."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    calls: list = []
    monkeypatch.setattr(evalrun, "run_one", lambda *a, **kw: calls.append(a))
    staging_parent = tmp_path / "staging"
    staging_parent.mkdir()
    rc = evalrun.main(
        ["--stage", "s2", "--p", "0", "--seeds", "1", "--store", str(tmp_path / "store"), "--recefore",
         str(tmp_path / "recefore"), "--staging", str(staging_parent), "--interval-choices", "1", "inf"]
    )
    assert rc == 2
    assert calls == []
    assert capsys.readouterr().err == "semindex.evalrun: state_poll_interval_choices_s[1] must be a finite number > 0\n"


def test_main_passes_k_and_interval_choices_to_run_all(tmp_path: Path, monkeypatch):
    """--k and --interval-choices reach run_all as k / state_poll_interval_choices_s."""
    seen: dict = {}

    def _record_run_all(*args, **kwargs):
        """Capture the keyword arguments main() passed."""
        seen.update(kwargs)
        return []

    monkeypatch.setattr(evalrun, "run_all", _record_run_all)
    staging_parent = tmp_path / "staging"
    staging_parent.mkdir()
    rc = evalrun.main(
        ["--stage", "s2", "--p", "0", "--seeds", "1", "--store", str(tmp_path / "store"), "--recefore",
         str(tmp_path / "recefore"), "--staging", str(staging_parent), "--k", "4", "--interval-choices", "1", "2",
         "--choice-jitter", "0", "--max-in-flight", "32", "--store-synchronous", "OFF"]
    )
    assert rc == 0
    assert seen["k"] == 4
    assert seen["state_poll_interval_choices_s"] == [1.0, 2.0]
    assert seen["state_poll_choice_jitter_frac"] == 0.0
    assert seen["max_in_flight_per_neighbor"] == 32
    assert seen["store_synchronous"] == "OFF"


def test_main_leaves_max_in_flight_unset_when_not_given(tmp_path: Path, monkeypatch):
    """Without --max-in-flight, run_all gets None (no cap written; agents keep 8)."""
    seen: dict = {}
    monkeypatch.setattr(evalrun, "run_all", lambda *a, **kw: seen.update(kw) or [])
    staging_parent = tmp_path / "staging"
    staging_parent.mkdir()
    rc = evalrun.main(
        ["--stage", "s2", "--p", "0", "--seeds", "1", "--store", str(tmp_path / "store"), "--recefore",
         str(tmp_path / "recefore"), "--staging", str(staging_parent)]
    )
    assert rc == 0
    assert seen["max_in_flight_per_neighbor"] is None
    assert seen["store_synchronous"] is None


def test_main_refuses_a_zero_max_in_flight_before_any_run(tmp_path: Path, monkeypatch, capsys):
    """--max-in-flight 0 is refused once, up front (exit 2), and no run is attempted."""
    monkeypatch.delenv("SUDO_UID", raising=False)
    calls: list = []
    monkeypatch.setattr(evalrun, "run_one", lambda *a, **kw: calls.append(a))
    staging_parent = tmp_path / "staging"
    staging_parent.mkdir()
    rc = evalrun.main(
        ["--stage", "s2", "--p", "0", "--seeds", "1", "--store", str(tmp_path / "store"), "--recefore",
         str(tmp_path / "recefore"), "--staging", str(staging_parent), "--max-in-flight", "0"]
    )
    assert rc == 2
    assert calls == []
    assert capsys.readouterr().err == "semindex.evalrun: max_in_flight_per_neighbor must be an integer >= 1\n"


def test_main_refuses_a_lower_case_store_synchronous_before_any_run(tmp_path: Path, monkeypatch):
    """--store-synchronous normal is an argparse usage error (exit 2); no run is attempted."""
    calls: list = []
    monkeypatch.setattr(evalrun, "run_one", lambda *a, **kw: calls.append(a))
    staging_parent = tmp_path / "staging"
    staging_parent.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        evalrun.main(
            ["--stage", "s2", "--p", "0", "--seeds", "1", "--store", str(tmp_path / "store"), "--recefore",
             str(tmp_path / "recefore"), "--staging", str(staging_parent), "--store-synchronous", "normal"]
        )
    assert excinfo.value.code == 2
    assert calls == []


def test_run_all_refuses_a_bad_store_synchronous_before_any_run(tmp_path: Path, monkeypatch):
    """Called directly (no argparse in front), run_all refuses a bad mode once, up front."""
    calls: list = []
    monkeypatch.setattr(evalrun, "run_one", lambda *a, **kw: calls.append(a))
    with pytest.raises(evalrun.gen.ExperimentError) as excinfo:
        evalrun.run_all(
            [0.0], [1], "s2", str(tmp_path / "store"), str(tmp_path / "recefore"), str(tmp_path),
            store_synchronous="normal",
        )
    assert str(excinfo.value) == "store_synchronous must be one of ['FULL', 'NORMAL', 'OFF']"
    assert calls == []


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


def test_store_files_are_closed_to_0644_as_soon_as_they_are_written(tmp_path):
    """Copied and written store files are 0644 right away, even under umask 0, not only after the final chown pass."""
    from semindex import evalrun

    src = tmp_path / "src.json"
    src.write_text("{}")
    os.chmod(src, 0o666)
    old = os.umask(0)
    try:
        evalrun._copy_locked_down(str(src), str(tmp_path / "copied.json"))
        evalrun._write_json(str(tmp_path / "meta.json"), {"a": 1})
        evalrun._write_done(str(tmp_path / "DONE"))
    finally:
        os.umask(old)
    modes = {name: os.stat(tmp_path / name).st_mode & 0o777 for name in ("copied.json", "meta.json", "DONE")}
    assert modes == {"copied.json": 0o644, "meta.json": 0o644, "DONE": 0o644}
