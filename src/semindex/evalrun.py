"""semindex.evalrun: drive one stage-1 evaluation sweep over (p, seed).

Ticket 04's "回し手": given a stage label, a grid of disconnect
probabilities and a seed range, generate/run/store one ReCefore run per
(p, seed) -- skipping whatever `<store>/<stage>/p<p>/<seed>/DONE` already
marks finished, so a sweep interrupted partway through (orchestrator
restart, machine reboot, ...) picks up exactly where it left off.

This module never touches Mininet, cefnetd, or `sudo` itself: every
place it would run a real command (`mn -c`, the ReCefore subprocess) or
chown a file goes through an injectable seam (`run_subprocess`,
`chown_tree`), so the whole pipeline -- generation, copying, the resume
guard, the deletion guard -- is unit-testable with a fake ReCefore that
never needs root. Only the orchestrator, invoking this module with
`sudo`, ever exercises the real seams.

Run as:

    python -m semindex.evalrun --stage s1 --p 0 0.1 0.2 0.3 0.4 \\
        --seeds 1-100 --store /home/lab_shared/semindex-eval \\
        --recefore /home/lab_shared/cefore-emu-sslab
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from semindex import evalgen, evalvalid, gen

# 2026-10-23 (orchestrator's live trial, p=0.2 seed=1): one stage-1 run's
# own ReCefore scenario is 135s of simulated time (evalgen's
# RECEFORE_DURATION_S) and took 158s of real wall time end to end. 400s
# gives that a wide margin without letting one wedged run block an
# overnight sweep indefinitely; `timeout` (not a Python-side
# subprocess.run(timeout=...)) enforces it, so a wedged Mininet process
# tree gets SIGTERM'd by the same mechanism the orchestrator's own manual
# trial used, not by this module re-implementing process-group cleanup.
RECEFORE_TIMEOUT_S = 400

# 2026-10-23 review fix: `timeout 400 ...` alone only ever sends TERM at
# the deadline -- GNU coreutils' `timeout` does not escalate to KILL on
# its own (`timeout --help`: "-k, --kill-after=DURATION ... send a KILL
# signal if COMMAND is still running this long after the initial
# signal was sent"). If ReCefore or a Mininet child ignores or catches
# TERM, `subprocess.run()` here simply blocks forever waiting for it to
# exit, wedging the rest of an unattended overnight sweep. `--kill-after`
# gives TERM a finite grace period before `timeout` follows up with KILL,
# which cannot be caught or ignored. 30s: the orchestrator's live trial
# (p=0.2 seed=1) measured 158s wall time against a 135s scenario
# (RECEFORE_DURATION_S above) -- about 23s of ReCefore's own normal
# Mininet setup/teardown outside the scenario itself -- so a TERM that IS
# being honored finishes well inside this margin; it exists only to catch
# the case where TERM is not honored at all.
RECEFORE_KILL_AFTER_S = 30

# ReCefore's own "num" is baked into evalgen's generated experiment
# (evalgen.RECEFORE_NUM); its disaster.py names a run's output directory
# `ex{num}_seed{seed}` under the YAML's own output_dir (core/artifacts.py
# experiment_dir_name) -- reused here, not re-derived, so this module
# only ever looks in the one directory ReCefore itself will have written to.
_RECEFORE_NUM = evalgen.RECEFORE_NUM


def parse_seeds(text: str) -> list[int]:
    """Parse a `--seeds` value: comma-separated ints and/or inclusive `a-b` ranges (e.g. "1-100" or "1,3,5-8")."""
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def build_run_order(ps: list[float], seeds: list[int]) -> list[tuple[float, int]]:
    """Every (p, seed) pair in seed-major order: `for seed in seeds: for p in ps`.

    map.md's resume design ("途中で止まっても続きから再開できる") means a
    sweep that is interrupted after N pairs must already have covered
    every condition about evenly -- seed-major does that (each seed
    finishes all of `ps` before the next seed starts), where p-major
    would leave every seed after the first only partially explored if
    the sweep stops early. Both axes are de-duplicated and sorted so the
    order depends only on *which* (p, seed) values were asked for, never
    the order they were listed in on the command line.
    """
    ps_sorted = sorted(set(ps))
    seeds_sorted = sorted(set(seeds))
    return [(p, seed) for seed in seeds_sorted for p in ps_sorted]


def default_run_subprocess(argv: list[str], *, cwd: str | None = None) -> int:
    """Run `argv` to completion, inheriting this process's stdio, and return its exit code.

    Stdio is inherited (not captured) deliberately: ReCefore's own run
    prints Mininet's `info()` progress lines, which the orchestrator
    running this live is the intended audience for -- capturing them into
    memory here would only throw them away.
    """
    return subprocess.run(argv, cwd=cwd).returncode


def default_capture_subprocess(argv: list[str], *, cwd: str | None = None) -> str:
    """Run `argv` to completion and return its stripped stdout (used only for one-line queries like `git rev-parse`)."""
    proc = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return proc.stdout.strip()


def default_chown_tree(path: str) -> None:
    """Recursively chown `path` to the invoking user (SUDO_UID/SUDO_GID) and make it group-writable.

    A no-op when SUDO_UID/SUDO_GID are not set -- this process is not
    running under `sudo` (every test, and any direct, non-orchestrator
    invocation), so there is no "invoking user" to hand ownership back to.
    """
    uid = os.environ.get("SUDO_UID")
    gid = os.environ.get("SUDO_GID")
    if uid is None or gid is None:
        return
    uid, gid = int(uid), int(gid)
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            full = os.path.join(root, name)
            os.chown(full, uid, gid, follow_symlinks=False)
            if not os.path.islink(full):
                os.chmod(full, os.stat(full).st_mode | 0o020)  # g+w
    os.chown(path, uid, gid)
    os.chmod(path, os.stat(path).st_mode | 0o020)


def _validate_stage(stage: str) -> None:
    """Reject a `stage` that is not safe to use as a single, bare path component.

    2026-10-23 review fix: `run_one` builds `store_run_dir` (and, until
    this fix, its own deletion boundary) as `os.path.join(store, stage,
    ...)` -- and does so, and may delete whatever already exists there,
    before `evalgen.generate_run` ever gets a chance to run
    `gen.parse_experiment`'s own rejection of a path-unsafe `name` (which
    embeds this same `stage`; see gen.py's 2026-10-22 fix). `os.path.join`
    discards every earlier component once it hits an absolute one, so an
    unvalidated `--stage /tmp/outside` turned `os.path.join(store, stage,
    ...)` into `/tmp/outside/...` outright, with no trace of `store` left
    in it at all. Checked here, before any path is built from `stage`.
    """
    if not stage or os.path.isabs(stage) or "/" in stage or os.sep in stage or stage in (".", ".."):
        raise ValueError(f"--stage must be a single, path-safe name, got {stage!r}")


def _guarded_rmtree(path: str, *, root: str) -> None:
    """Delete `path` recursively, but only if it is not itself a symlink and resolves strictly inside `root`.

    Both staging-run cleanup and partial-store-run cleanup funnel through
    this one function: a path built from caller-controlled strings
    (`stage`, a seed) must never let a stray `..`, an absolute override,
    or a symlink swapped in after the check send a recursive delete
    outside the one directory this call is allowed to tear down.
    """
    if os.path.islink(path):
        raise ValueError(f"refusing to delete a symlink: {path!r}")
    real_root = os.path.realpath(root)
    real_path = os.path.realpath(path)
    if real_path == real_root or not real_path.startswith(real_root + os.sep):
        raise ValueError(f"refusing to delete {path!r}: not strictly inside {root!r}")
    shutil.rmtree(real_path)


def _verify_safe_run_dir(store_run_dir: str, store: str) -> None:
    """Verify `store_run_dir` is safe to read, write, or delete under `store`, before anything else touches it.

    2026-10-23 fix: the only boundary check `run_one` had
    (`_guarded_rmtree`'s own realpath comparison) ran exclusively inside
    `if os.path.isdir(store_run_dir): ...` -- so a `store/<stage>`
    replaced with a symlink to an EXTERNAL directory that did not yet
    exist at all (not even its `p<p>/<seed>` leaf) made `os.path.isdir`
    False, skipping that check entirely. The run then proceeded straight
    through the DONE check (also False, nothing there yet) into the full
    pipeline, and `os.makedirs(store_run_dir, exist_ok=True)` further
    down happily created the external target from scratch and wrote
    `meta.json`/`DONE` into it -- entirely outside `store`, with no
    directory ever having existed there to trip the old check.

    Called unconditionally here, before the DONE check or anything else
    touches `store_run_dir`: every path component of `store_run_dir`
    below `store` (`stage`, `p<p>`, `<seed>`) is checked for being a
    symlink -- existing or not; `os.path.islink` on a component that does
    not exist yet is simply False, which is exactly right: there is
    nothing there yet to be a symlink, and the component will be created
    fresh, under `store`, by `os.makedirs` later. Only an EXISTING
    symlink anywhere in that chain is rejected, regardless of where it
    happens to point (even if that target would itself resolve back
    inside `store`: the point is store_run_dir must be reached by walking
    real directories under store, never by following a redirect). The
    final resolved path is then also required to land strictly inside
    `store`'s own resolved path, as a second, independent check.

    `_guarded_rmtree`'s own check (below) stays as it was -- a second,
    narrower layer that runs right before an actual delete, once
    `store_run_dir` is already known to exist -- this function's job is
    only to catch the case that check could never even reach.
    """
    rel = os.path.relpath(store_run_dir, store)
    if rel == os.curdir or rel.startswith(os.pardir + os.sep) or rel == os.pardir:
        raise ValueError(f"refusing to use {store_run_dir!r}: not under {store!r}")
    cur = store
    for part in rel.split(os.sep):
        cur = os.path.join(cur, part)
        if os.path.islink(cur):
            raise ValueError(f"refusing to use {store_run_dir!r}: {cur!r} is a symlink below the store root")
    real_store = os.path.realpath(store)
    real_run_dir = os.path.realpath(store_run_dir)
    if real_run_dir == real_store or not real_run_dir.startswith(real_store + os.sep):
        raise ValueError(f"refusing to use {store_run_dir!r}: resolves outside {store!r}")


def _write_done(path: str) -> None:
    """Create `path` atomically (write to a tmp sibling, then rename) -- the resume check's one source of truth."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("")
    os.replace(tmp, path)


def _store_run_dir(store: str, stage: str, p: float, seed: int) -> str:
    """Where one (stage, p, seed) run's artifacts live under `store`."""
    return os.path.join(store, stage, f"p{evalgen._format_p(p)}", str(seed))


def _copy_run_outputs(staging_run_dir: str, recefore_run_dir: str, store_run_dir: str, generated_paths: list[str]) -> None:
    """Copy exactly what map.md/ticket 04 ask to keep from one run, nothing else.

    `generated_paths` (evalgen.generate_run's own return value: the
    `.experiment.json`, the `.yaml`, and every `<name>/<node>.json`) are
    copied verbatim, preserving their path relative to `staging_run_dir`
    -- so the agent-config directory structure `semindex.gen` already
    decided stays intact in the store too. `results.json` and each
    node's `events.jsonl`/`semindex_agent.log` are copied preserving
    ReCefore's own `debug/node_dirs/<node>/...` layout (not flattened),
    so `semindex.evalvalid` (and `semindex.check`/`semindex.analyze`, if
    ever pointed at a stored run) read the one layout unchanged.
    `recefore_run_dir`'s `node_dirs/<node>/` also holds Cefore's own key
    files (default-private-key, cefnetd.key, ...) -- never globbed or
    copied here, only the two named files below, which is what "never
    copy anything else from node_dirs" means in practice.

    Every source path is checked with `os.path.exists` first and simply
    skipped if absent: a crashed or timed-out ReCefore run may not have
    written `results.json`, or even `recefore_run_dir`, at all, and this
    function's job is to store whatever *does* exist for
    `semindex.evalvalid` to flag, not to require a clean run.
    """
    results_src = os.path.join(recefore_run_dir, "results.json")
    if os.path.exists(results_src):
        shutil.copy2(results_src, os.path.join(store_run_dir, "results.json"))

    for src in generated_paths:
        rel = os.path.relpath(src, staging_run_dir)
        dest = os.path.join(store_run_dir, rel)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)

    for node_dir in sorted(glob.glob(os.path.join(recefore_run_dir, "debug", "node_dirs", "h*"))):
        node = os.path.basename(node_dir)
        dest_node_dir = os.path.join(store_run_dir, "debug", "node_dirs", node)

        events_src = os.path.join(node_dir, "semindex", "events.jsonl")
        if os.path.exists(events_src):
            os.makedirs(os.path.join(dest_node_dir, "semindex"), exist_ok=True)
            shutil.copy2(events_src, os.path.join(dest_node_dir, "semindex", "events.jsonl"))

        log_src = os.path.join(node_dir, "semindex_agent.log")
        if os.path.exists(log_src):
            os.makedirs(dest_node_dir, exist_ok=True)
            shutil.copy2(log_src, os.path.join(dest_node_dir, "semindex_agent.log"))


def _git_commit(repo_dir: str, capture_subprocess: Callable[..., str]) -> str | None:
    """`repo_dir`'s current HEAD commit, or None if it cannot be determined (not a git repo, git missing, ...)."""
    try:
        out = capture_subprocess(["git", "-C", repo_dir, "rev-parse", "HEAD"])
    except OSError:
        return None
    return out or None


def _git_dirty(repo_dir: str, capture_subprocess: Callable[..., str]) -> bool | None:
    """Whether `repo_dir` has uncommitted changes, or None if that cannot be determined."""
    try:
        out = capture_subprocess(["git", "-C", repo_dir, "status", "--porcelain"])
    except OSError:
        return None
    return bool(out)


def _iso(ts: float) -> str:
    """Render an epoch-seconds float as an ISO 8601 UTC timestamp (results.json's own `timestamp_utc` convention)."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# semindex's own repo root: two levels up from this file (src/semindex/evalrun.py),
# the same `__file__`-relative derivation gen.py's SEMINDEX_SRC_DIR already uses --
# reused here rather than re-derived, so both stay in lockstep if the layout ever moves.
_SEMINDEX_REPO_ROOT = os.path.dirname(gen.SEMINDEX_SRC_DIR)


def _build_meta(
    *,
    p: float,
    seed: int,
    stage: str,
    recefore: str,
    recefore_rc: int,
    start_s: float,
    end_s: float,
    verdict: dict,
    capture_subprocess: Callable[..., str],
) -> dict:
    """Build one run's meta.json dict."""
    recefore_python = os.path.join(recefore, ".venv", "bin", "python3")
    return {
        "p": p,
        "seed": seed,
        "stage": stage,
        "semindex_git_commit": _git_commit(_SEMINDEX_REPO_ROOT, capture_subprocess),
        "semindex_git_dirty": _git_dirty(_SEMINDEX_REPO_ROOT, capture_subprocess),
        "recefore_git_commit": _git_commit(recefore, capture_subprocess),
        "wall_clock_start": _iso(start_s),
        "wall_clock_end": _iso(end_s),
        "recefore_exit_code": recefore_rc,
        "host_python": {
            "evalrun": sys.version,
            "recefore_venv": capture_subprocess([recefore_python, "--version"]),
        },
        "validity": verdict,
    }


@dataclass
class RunOutcome:
    """What one (p, seed) run did, for the one-line stdout log and for tests."""

    p: float
    seed: int
    status: str  # "skipped" | "done"
    recefore_rc: int | None = None
    wall_s: float | None = None
    valid: bool | None = None
    flags: list[str] = field(default_factory=list)


def run_one(
    p: float,
    seed: int,
    stage: str,
    store: str,
    recefore: str,
    staging_root: str,
    *,
    run_subprocess: Callable[..., int] = default_run_subprocess,
    capture_subprocess: Callable[..., str] = default_capture_subprocess,
    chown_tree: Callable[[str], None] = default_chown_tree,
    clock: Callable[[], float] = time.time,
) -> RunOutcome:
    """Generate, run, store and validate one (p, seed) run; or skip it if already DONE.

    A store run dir that exists but lacks DONE is a run some earlier,
    interrupted sweep never finished -- removed (guarded) and redone from
    scratch, never resumed partway, since a staging run dir ReCefore only
    half-wrote is not something any later step here can safely pick back
    up from the middle of.
    """
    _validate_stage(stage)
    store_run_dir = _store_run_dir(store, stage, p, seed)
    _verify_safe_run_dir(store_run_dir, store)
    done_marker = os.path.join(store_run_dir, "DONE")
    if os.path.exists(done_marker):
        return RunOutcome(p=p, seed=seed, status="skipped")

    # 2026-10-23 review fix: anchored to `store` itself, not
    # `os.path.join(store, stage)` -- besides no longer being derived from
    # the same (now-validated, but still caller-controlled) `stage` the
    # delete target is, this also refuses a `store/<stage>` that was
    # replaced with a symlink to outside `store`: `_guarded_rmtree`
    # resolves both sides, and a symlinked `store/<stage>` makes
    # `store_run_dir`'s real path land outside `store`'s real path, where
    # it did not before (it landed outside the equally-symlinked
    # `stage_root` this used to check against instead).
    if os.path.isdir(store_run_dir):
        _guarded_rmtree(store_run_dir, root=store)

    staging_run_dir = tempfile.mkdtemp(
        prefix=f"{stage}_p{evalgen._format_p(p)}_seed{seed}_", dir=staging_root
    )
    try:
        generated_paths = evalgen.generate_run(p, seed, stage, staging_run_dir)
        yaml_path = next(path for path in generated_paths if path.endswith(".yaml"))

        run_subprocess(["mn", "-c"])

        recefore_python = os.path.join(recefore, ".venv", "bin", "python3")
        argv = [
            "setsid",
            "timeout",
            "--kill-after",
            str(RECEFORE_KILL_AFTER_S),
            str(RECEFORE_TIMEOUT_S),
            recefore_python,
            "-m",
            "src",
            "disaster",
            "--config",
            yaml_path,
            "--no-cli",
        ]
        start_s = clock()
        recefore_rc = run_subprocess(argv, cwd=recefore)
        end_s = clock()

        recefore_run_dir = os.path.join(staging_run_dir, f"ex{_RECEFORE_NUM}_seed{seed}")
        os.makedirs(store_run_dir, exist_ok=True)
        _copy_run_outputs(staging_run_dir, recefore_run_dir, store_run_dir, generated_paths)

        try:
            verdict = evalvalid.validate(store_run_dir)
        except evalvalid.ValidityUsageError as exc:
            # Our own copy step failed to produce something evalvalid
            # could even read (e.g. the experiment.json copy itself is
            # missing) -- this is a bug in this pipeline, not a judgment
            # about the run, so it is recorded as a flag rather than
            # raised and left to abort the whole sweep.
            verdict = {"valid": False, "flags": [f"evalvalid_usage_error:{exc}"]}

        meta = _build_meta(
            p=p,
            seed=seed,
            stage=stage,
            recefore=recefore,
            recefore_rc=recefore_rc,
            start_s=start_s,
            end_s=end_s,
            verdict=verdict,
            capture_subprocess=capture_subprocess,
        )
        _write_json(os.path.join(store_run_dir, "meta.json"), meta)

        chown_tree(store_run_dir)
    finally:
        _guarded_rmtree(staging_run_dir, root=staging_root)

    _write_done(done_marker)
    chown_tree(done_marker)

    return RunOutcome(
        p=p,
        seed=seed,
        status="done",
        recefore_rc=recefore_rc,
        wall_s=end_s - start_s,
        valid=verdict["valid"],
        flags=verdict["flags"],
    )


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _format_outcome(outcome: RunOutcome) -> str:
    """One line per run, as ticket 04 asks: (p, seed, rc, wall s, validity verdict)."""
    if outcome.status == "skipped":
        return f"p={outcome.p} seed={outcome.seed} status=skipped"
    return (
        f"p={outcome.p} seed={outcome.seed} rc={outcome.recefore_rc} "
        f"wall_s={outcome.wall_s:.1f} valid={outcome.valid} flags={len(outcome.flags)}"
    )


def run_all(
    ps: list[float],
    seeds: list[int],
    stage: str,
    store: str,
    recefore: str,
    staging_root: str,
    *,
    run_subprocess: Callable[..., int] = default_run_subprocess,
    capture_subprocess: Callable[..., str] = default_capture_subprocess,
    chown_tree: Callable[[str], None] = default_chown_tree,
    clock: Callable[[], float] = time.time,
    out=None,
) -> list[RunOutcome]:
    """Run every (p, seed) pair in `build_run_order`'s order, logging one line per run to `out`.

    One run's own exception (a bug in this pipeline, an unexpected
    filesystem state, ...) is logged and swallowed rather than aborting
    the whole sweep: that run's store dir is left without DONE (so a
    later sweep redoes it from scratch, same as any other interrupted
    run), and every remaining (p, seed) pair still gets attempted -- an
    overnight sweep of ~500 runs should not be taken down entirely by one
    bad run. map.md's own per-condition >5%-flagged threshold is the
    mechanism meant to catch that, at the orchestrator level, not here.

    `out` defaults to None, not `sys.stdout`, resolved to `sys.stdout`
    only once inside the call: a default bound at *def* time would
    capture whatever `sys.stdout` object existed at import time, which a
    test fixture (e.g. pytest's capsys) that swaps `sys.stdout` later can
    then never redirect.
    """
    if out is None:
        out = sys.stdout
    # Validated once, up front: `run_one` validates `stage` too (it is the
    # function that actually builds and deletes paths from it, so it must
    # never trust a caller who skipped this), but failing here as well
    # means one bad `--stage` is one clear error instead of this loop's
    # own per-pair try/except (below) logging the identical failure once
    # per (p, seed) pair and still reporting the sweep as "ran".
    _validate_stage(stage)
    os.makedirs(store, exist_ok=True)
    os.makedirs(staging_root, exist_ok=True)
    outcomes = []
    for p, seed in build_run_order(ps, seeds):
        try:
            outcome = run_one(
                p,
                seed,
                stage,
                store,
                recefore,
                staging_root,
                run_subprocess=run_subprocess,
                capture_subprocess=capture_subprocess,
                chown_tree=chown_tree,
                clock=clock,
            )
        except Exception as exc:  # noqa: BLE001 -- see docstring: one run's crash must not stop the sweep
            print(f"p={p} seed={seed} status=error error={exc!r}", file=out)
            continue
        print(_format_outcome(outcome), file=out)
        outcomes.append(outcome)
    return outcomes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the evaluation runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, help="stage label (e.g. s1); used in the store path and run names")
    parser.add_argument("--p", type=float, nargs="+", required=True, help="one or more disconnect probabilities")
    parser.add_argument("--seeds", required=True, help='seed range/list, e.g. "1-100" or "1,3,5-8"')
    parser.add_argument("--store", required=True, help="directory runs are stored under")
    parser.add_argument("--recefore", required=True, help="path to the ReCefore (cefore-emu-sslab) checkout")
    parser.add_argument("--staging", default=None, help="directory to create per-run staging dirs under (default: system temp)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: python -m semindex.evalrun ..."""
    args = parse_args(argv)
    seeds = parse_seeds(args.seeds)
    staging_root = tempfile.mkdtemp(prefix="semindex-evalrun-", dir=args.staging)
    try:
        run_all(args.p, seeds, args.stage, args.store, args.recefore, staging_root)
    finally:
        if os.path.isdir(staging_root) and not os.listdir(staging_root):
            os.rmdir(staging_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
