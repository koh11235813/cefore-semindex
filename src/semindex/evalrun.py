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
        --recefore /home/lab_shared/cefore-emu-sslab \\
        [--k 4] [--interval-choices 1 2 4]

`--k` and `--interval-choices` (stage-2 map "生成器と集計") go straight to
semindex.evalgen and are recorded in each run's meta.json. They are not
part of the store path, so a sweep with different values needs its own
`--stage` label.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import stat
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
    """Run `argv` to completion and return its stripped stdout (used only for one-line queries like `git rev-parse`).

    2026-10-23 fix: `stderr` used to be discarded (`DEVNULL`) and a non-zero exit was never
    distinguished from a clean one that merely printed nothing -- `_git_commit`/`_git_dirty` need
    the real failure reason (e.g. git's own "detected dubious ownership" message) to store, not an
    empty string that looks identical to success. `stderr` is now captured, and a non-zero exit
    raises `subprocess.CalledProcessError` carrying it.
    """
    proc = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, argv, output=proc.stdout, stderr=proc.stderr)
    return proc.stdout.strip()


# 2026-10-23 (external review, P1 #2): every directory/file this module
# creates or copies into the store ends up at exactly one of these two
# modes, never "whatever umask the sudo environment happened to have, with
# group-write OR'd on top" (the previous behavior here). `os.chmod` sets
# these bits absolutely, so the result is umask-independent regardless of
# what mode `os.makedirs`/`shutil.copy2` actually created the entry with.
# This is the other half of the fix described in `_check_dir_permissions`'
# own docstring: a directory that is not writable by group or others can
# never have one of its entries replaced (by another group member) with a
# symlink in the first place, which is what actually closes the race --
# re-checking permissions right before a write (see
# `_verify_safe_store_permissions`) only catches a window that will always
# exist; removing the write permission that a swap would need removes the
# window itself.
_STORE_DIR_MODE = 0o755
_STORE_FILE_MODE = 0o644


def _makedirs_locked_down(path: str) -> None:
    """`os.makedirs(path, exist_ok=True)`, with every directory this call actually creates
    immediately chmod'd to `_STORE_DIR_MODE`.

    2026-10-23 fix (external review, P1 #2): `os.makedirs`'s own `mode` argument is masked by
    this process's umask at the syscall level (POSIX `mkdir()`), so passing `mode=_STORE_DIR_MODE`
    to it directly would NOT be umask-independent -- a permissive umask (e.g. 0, common in a
    throwaway root shell) would still leave a freshly created directory group/other-writable. The
    deferred `chown_tree`/`chown_dir` calls at the end of `run_one` fix this eventually, but
    `_copy_run_outputs`, `evalvalid.validate` and `_build_meta` (which shells out to `git`) all run
    in between -- a real, if short, gap during which a directory this call just created sits at
    whatever mode umask gave it. `os.chmod` is not subject to umask (it sets the bits given,
    absolutely), so chmod'ing right here, immediately after creation, closes that gap outright
    rather than narrowing it. Walks up from `path` first to find which ancestors do not exist yet
    -- the same "only touch what this call itself created" rule `run_one`'s own `new_ancestors`
    snapshot already uses for chown -- so a directory some earlier run already owns and locked
    down is never redundantly touched.
    """
    to_create = []
    cur = path
    while not os.path.isdir(cur):
        to_create.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    os.makedirs(path, exist_ok=True)
    for created in to_create:
        os.chmod(created, _STORE_DIR_MODE)


def default_chown_tree(path: str) -> None:
    """Recursively chown `path` to the invoking user (SUDO_UID/SUDO_GID) and set every directory
    to `_STORE_DIR_MODE` and every file to `_STORE_FILE_MODE`.

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
        for name in dirs:
            full = os.path.join(root, name)
            os.chown(full, uid, gid, follow_symlinks=False)
            if not os.path.islink(full):
                os.chmod(full, _STORE_DIR_MODE)
        for name in files:
            full = os.path.join(root, name)
            os.chown(full, uid, gid, follow_symlinks=False)
            if not os.path.islink(full):
                os.chmod(full, _STORE_FILE_MODE)
    os.chown(path, uid, gid)
    os.chmod(path, _STORE_DIR_MODE if os.path.isdir(path) else _STORE_FILE_MODE)


def default_chown_dir(path: str) -> None:
    """chown `path` ITSELF (not recursively) to the invoking user (SUDO_UID/SUDO_GID) and set it to
    `_STORE_DIR_MODE`; a no-op without them, same rule as `default_chown_tree`.

    2026-10-23 orchestrator trial fix: `run_one`'s single `os.makedirs(store_run_dir,
    exist_ok=True)` call can create several missing ancestor components at once (`<stage>`,
    `<stage>/p<p>`, the seed dir) in one shot -- but `chown_tree(store_run_dir)` only ever recurses
    DOWN INTO `store_run_dir` itself, never back up into the ancestors above it that the same
    makedirs call also just created. The live trial found exactly this: `<store>/<stage>` and
    `<store>/<stage>/p<p>` were left root-owned while every run dir underneath them was correctly
    chowned. This function covers those ancestors; `run_one` calls it only for the ones it
    confirmed (before makedirs) did not already exist, so an ancestor some earlier run already
    owns is never touched again.

    Deliberately non-recursive: each ancestor's own CONTENTS (sibling seeds, other p values) are
    each chowned by their own run's `chown_tree(store_run_dir)` call already -- walking back into
    them here on every single run would be pure duplicate work repeated across an entire sweep.
    Refuses outright (like `_guarded_rmtree`/`_verify_safe_run_dir`) if `path` is itself a symlink,
    rather than silently chowning through it.
    """
    uid = os.environ.get("SUDO_UID")
    gid = os.environ.get("SUDO_GID")
    if uid is None or gid is None:
        return
    if os.path.islink(path):
        raise ValueError(f"refusing to chown a symlink: {path!r}")
    os.chown(path, int(uid), int(gid))
    os.chmod(path, _STORE_DIR_MODE)


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


def _check_dir_permissions(path: str, *, invoking_uid: int) -> None:
    """Refuse `path` unless it is either absent (nothing there yet to check -- it will be created
    fresh, at `_STORE_DIR_MODE`, by `run_one` itself), or an existing, non-symlink directory owned
    by root or `invoking_uid` and not writable by group or others.

    Uses `os.lstat`, not `os.stat`: a symlinked `path` is refused outright, in the same call that
    reads its mode/owner, rather than first asking `os.path.islink` (itself an `lstat`) and only
    then `os.stat`-ing it -- two syscalls with a gap between them is exactly the kind of
    check-then-use race this whole fix exists to close. A store or ancestor swapped for a symlink
    (2026-10-23 review finding: the store root itself, not just a component below it) is caught
    here the same way `_verify_safe_run_dir`'s own symlink check is, just without following it
    first to find out.

    2026-10-23 fix (external review, P1 #2). Threat model: `run_one` is invoked under `sudo`
    against a store directory shared with other members of the same Unix group (the live store is
    `drwxrwsr-x`, group-owned) -- `_verify_safe_run_dir` above only checks for an EXISTING symlink
    on `store_run_dir`'s path, and it runs once, well before the actual store writes (`os.makedirs`/
    `_copy_run_outputs`/`_write_json`), which only happen after ReCefore's own ~160s run finishes.
    Nothing stops another group member from swapping a directory component -- say `<store>/<stage>`
    -- for a symlink to an external directory DURING that wait: by the time this process writes
    `meta.json` there, root privilege means it is written wherever that symlink now points, and a
    further symlinked file at the target (e.g. an external `meta.json` symlinked to some other
    file this process has no business touching) lets `_write_json` overwrite it outright.
    Re-checking for a symlink right before the write (see `_verify_safe_store_permissions`) narrows
    the window but can never close it -- there is always a gap between a check and the write that
    follows it.

    What actually closes it is access control, not timing: if every directory on the path is owned
    by root or the one user this sweep is running for, and is writable only by its owner, then NO
    other group member ever has the filesystem permission needed to unlink-and-replace one of its
    entries with a symlink in the first place -- the race has nothing left to exploit, regardless
    of how long `run_one` waits between checking and writing. `default_chown_tree`/
    `default_chown_dir` (above) are the other half of this: every directory `run_one` itself
    creates ends up at exactly this mode (`_STORE_DIR_MODE`, 0755) and owner, so each run leaves the
    invariant intact for the next one to check.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(st.st_mode):
        raise ValueError(f"refusing to use {path!r}: is a symlink below the store root")
    if st.st_uid != 0 and st.st_uid != invoking_uid:
        raise ValueError(
            f"refusing to use {path!r}: owned by uid {st.st_uid}, not root or the invoking user (uid {invoking_uid})"
        )
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(
            f"refusing to use {path!r}: writable by group or others (mode {oct(stat.S_IMODE(st.st_mode))})"
        )


def _verify_safe_store_permissions(store_run_dir: str, store: str) -> None:
    """Apply `_check_dir_permissions` to `store` and every directory on the path down to
    `store_run_dir` -- see that function's own docstring for the race this closes, and for why
    `store` itself (2026-10-23 review finding) is checked exactly the same way as every component
    below it, not skipped just because `os.path.isdir` would also call it a directory were it a
    symlink to one.

    A no-op when SUDO_UID is not set, the same convention `default_chown_tree`/`default_chown_dir`
    already use: outside `sudo`, there is no "invoking user" this check could even be protecting
    (every test, and any direct, non-orchestrator invocation, falls here).

    Called twice by `run_one`: once up front (alongside `_verify_safe_run_dir`, before the DONE
    check), and again immediately before the first store write, right after ReCefore's own run
    returns -- a permission change made during that multi-minute wait must still be caught before
    anything is written, not just assumed to still hold from the earlier check. Also called once,
    unconditionally, by `run_all` before its own per-(p, seed) loop starts (2026-10-23 review
    finding): `run_one`'s own call sites are each wrapped in `run_all`'s per-pair try/except, which
    exists so one run's bug never aborts an entire overnight sweep -- but that same try/except
    would otherwise turn a mis-permissioned `store` into hundreds of individually swallowed
    `status=error` lines instead of the sweep refusing to start at all, the same "fail loudly before
    touching anything" treatment `DirtySemindexError` already gets for a dirty semindex tree.
    """
    sudo_uid = os.environ.get("SUDO_UID")
    if sudo_uid is None:
        return
    invoking_uid = int(sudo_uid)
    _check_dir_permissions(store, invoking_uid=invoking_uid)
    rel = os.path.relpath(store_run_dir, store)
    cur = store
    for part in rel.split(os.sep):
        cur = os.path.join(cur, part)
        _check_dir_permissions(cur, invoking_uid=invoking_uid)


def _write_done(path: str) -> None:
    """Create `path` atomically (write to a tmp sibling, then rename) -- the resume check's one source of truth."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("")
    os.chmod(tmp, _STORE_FILE_MODE)
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
        _copy_locked_down(results_src, os.path.join(store_run_dir, "results.json"))

    for src in generated_paths:
        rel = os.path.relpath(src, staging_run_dir)
        dest = os.path.join(store_run_dir, rel)
        if os.path.exists(src):
            _makedirs_locked_down(os.path.dirname(dest))
            _copy_locked_down(src, dest)

    for node_dir in sorted(glob.glob(os.path.join(recefore_run_dir, "debug", "node_dirs", "h*"))):
        node = os.path.basename(node_dir)
        dest_node_dir = os.path.join(store_run_dir, "debug", "node_dirs", node)

        events_src = os.path.join(node_dir, "semindex", "events.jsonl")
        if os.path.exists(events_src):
            _makedirs_locked_down(os.path.join(dest_node_dir, "semindex"))
            _copy_locked_down(events_src, os.path.join(dest_node_dir, "semindex", "events.jsonl"))

        log_src = os.path.join(node_dir, "semindex_agent.log")
        if os.path.exists(log_src):
            _makedirs_locked_down(dest_node_dir)
            _copy_locked_down(log_src, os.path.join(dest_node_dir, "semindex_agent.log"))


def _git_run(repo_dir: str, args: list[str], capture_subprocess: Callable[..., str]) -> str:
    """Run one git query against `repo_dir`, scoped so it is safe to call under `sudo`.

    2026-10-23 orchestrator trial fix: the live trial's `recefore_git_commit` came back null
    because `git -C <recefore> rev-parse HEAD`, run as root, refused a directory root does not own
    ("detected dubious ownership") -- root's own gitconfig (in `/root`, not this invoking user's
    home) has no `safe.directory` entry for it. `-c safe.directory=<repo_dir>` marks it safe for
    just THIS one invocation (never `git config --global`, which would touch this process's own
    git config persistently). `--no-optional-locks` matters specifically for the `status
    --porcelain` dirty check below: without it, `git status` acquires `.git/index.lock` and
    rewrites the index file, which -- run as root -- would flip that index's ownership out from
    under the invoking user for their own, unrelated git use of the same tree afterward.
    `os.path.realpath` is applied to BOTH `-C` and `safe.directory=` so a symlinked `--recefore`
    checkout (or a path with a trailing slash) cannot slip past the exact-string match git does
    between the two.
    """
    real_repo_dir = os.path.realpath(repo_dir)
    argv = ["git", "--no-optional-locks", "-c", f"safe.directory={real_repo_dir}", "-C", real_repo_dir, *args]
    return capture_subprocess(argv)


def _git_commit(repo_dir: str, capture_subprocess: Callable[..., str]) -> str:
    """`repo_dir`'s current HEAD commit, or an `"error: ..."` string describing why it could not be
    determined (not a git repo, git missing, sudo's dubious-ownership refusal, ...).

    2026-10-23 fix: this used to return bare `None` on any failure, which `_build_meta` then stored
    as an uninformative JSON `null` -- including the one failure mode the live trial actually hit
    (see `_git_run`'s own docstring). The actual git/OSError message is stored instead, so a real
    failure is visible in meta.json rather than indistinguishable from "produced no output".
    """
    try:
        out = _git_run(repo_dir, ["rev-parse", "HEAD"], capture_subprocess)
    except OSError as exc:
        return f"error: {exc}"
    except subprocess.CalledProcessError as exc:
        return f"error: {(exc.stderr or '').strip() or exc}"
    return out or "error: git rev-parse HEAD produced no output"


def _git_dirty(repo_dir: str, capture_subprocess: Callable[..., str]) -> bool | str:
    """Whether `repo_dir` has uncommitted changes, or an `"error: ..."` string if that could not be
    determined -- same never-silently-null rule as `_git_commit` above."""
    try:
        out = _git_run(repo_dir, ["status", "--porcelain"], capture_subprocess)
    except OSError as exc:
        return f"error: {exc}"
    except subprocess.CalledProcessError as exc:
        return f"error: {(exc.stderr or '').strip() or exc}"
    return bool(out)


def _iso(ts: float) -> str:
    """Render an epoch-seconds float as an ISO 8601 UTC timestamp (results.json's own `timestamp_utc` convention)."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# semindex's own repo root: two levels up from this file (src/semindex/evalrun.py),
# the same `__file__`-relative derivation gen.py's SEMINDEX_SRC_DIR already uses --
# reused here rather than re-derived, so both stay in lockstep if the layout ever moves.
_SEMINDEX_REPO_ROOT = os.path.dirname(gen.SEMINDEX_SRC_DIR)


class DirtySemindexError(RuntimeError):
    """`run_all` raises this when the semindex tree is not confirmed clean and `allow_dirty` was
    not given.

    2026-10-23 orchestrator trial: stage-1 trial seeds 2-3 were recorded with
    `semindex_git_dirty: true` because semindex's own source files were being edited mid-sweep.
    Each run's own meta.json already records its commit/dirty flag individually, but nothing
    stopped the sweep from running at all on a tree whose results a later run could not reproduce.
    Refusing up front -- before `run_all` touches the store or staging dirs at all -- forces that
    choice (fix the tree, or explicitly accept the risk via `--allow-dirty`) before any run starts,
    not only after the fact when reading meta.json back.

    Refused on anything other than a confirmed-clean `False` (not just a confirmed-dirty `True`):
    `_git_dirty` can itself come back as an `"error: ..."` string (git missing, the dubious-
    ownership refusal `_git_run` scopes around but could still fail for some other reason, ...),
    and an UNDETERMINED tree is exactly the case this guard exists to stop from proceeding
    silently -- `--allow-dirty` is the one escape hatch for "git is broken but I know what I'm
    doing", not a reason to treat silence as clean.
    """


def _build_meta(
    *,
    p: float,
    seed: int,
    stage: str,
    k: int,
    state_poll_interval_choices_s: list[float] | None,
    recefore: str,
    recefore_rc: int,
    start_s: float,
    end_s: float,
    verdict: dict,
    capture_subprocess: Callable[..., str],
) -> dict:
    """Build one run's meta.json dict.

    `k` and `state_poll_interval_choices_s` (None = the default jittered
    policy) are the generator settings this run was made with, so the
    aggregate can tell runs apart by them (runs.csv) without re-reading the
    generated schedule.
    """
    recefore_python = os.path.join(recefore, ".venv", "bin", "python3")
    return {
        "p": p,
        "seed": seed,
        "stage": stage,
        "k": k,
        "state_poll_interval_choices_s": state_poll_interval_choices_s,
        "semindex_git_commit": _git_commit(_SEMINDEX_REPO_ROOT, capture_subprocess),
        "semindex_git_dirty": _git_dirty(_SEMINDEX_REPO_ROOT, capture_subprocess),
        "recefore_git_commit": _git_commit(recefore, capture_subprocess),
        "recefore_git_dirty": _git_dirty(recefore, capture_subprocess),
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
    chown_dir: Callable[[str], None] = default_chown_dir,
    clock: Callable[[], float] = time.time,
    k: int = 1,
    state_poll_interval_choices_s: list[float] | None = None,
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
    _verify_safe_store_permissions(store_run_dir, store)
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
        generated_paths = evalgen.generate_run(
            p, seed, stage, staging_run_dir, k=k, state_poll_interval_choices_s=state_poll_interval_choices_s
        )
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

        # 2026-10-23 fix (external review, P1 #2): re-verify right here, the
        # instant ReCefore's own ~160s run returns and before this process
        # writes anything to the store. The check at the top of this
        # function ran before that whole wait; a symlink swap or a
        # permission change made DURING it must still be caught now, not
        # assumed to still hold just because it held when this call started.
        # See `_check_dir_permissions`' own docstring for the full threat
        # model this re-check (and `_verify_safe_store_permissions` below)
        # exists for -- re-checking only narrows the window, the permission
        # invariant is what actually closes it.
        _verify_safe_run_dir(store_run_dir, store)
        _verify_safe_store_permissions(store_run_dir, store)

        recefore_run_dir = os.path.join(staging_run_dir, f"ex{_RECEFORE_NUM}_seed{seed}")

        # 2026-10-23 orchestrator trial fix: `os.makedirs(store_run_dir, exist_ok=True)` can
        # create several missing ancestor components (`<stage>`, `<stage>/p<p>`) in the same
        # call as the seed dir itself -- the live trial found exactly those two left root-owned,
        # since `chown_tree(store_run_dir)` below only ever recurses DOWN from the seed dir, never
        # back up to ancestors the same makedirs call also just created. Snapshotted BEFORE
        # makedirs runs: only an ancestor that did not already exist gets chowned afterwards, so a
        # directory some earlier run already owns is never touched again.
        stage_dir = os.path.join(store, stage)
        p_dir = os.path.dirname(store_run_dir)
        new_ancestors = [d for d in (stage_dir, p_dir) if not os.path.isdir(d)]

        _makedirs_locked_down(store_run_dir)
        _copy_run_outputs(staging_run_dir, recefore_run_dir, store_run_dir, generated_paths)

        try:
            verdict = evalvalid.validate(store_run_dir)
        except evalvalid.ValidityUsageError as exc:
            # Our own copy step failed to produce something evalvalid
            # could even read (e.g. the experiment.json copy itself is
            # missing) -- this is a bug in this pipeline, not a judgment
            # about the run, so it is recorded as a flag rather than
            # raised and left to abort the whole sweep.
            # "metrics": {} keeps this fallback's shape identical to evalvalid.validate's own
            # return value (which always includes "metrics") -- a reader of meta.json's
            # "validity" field should never need to branch on whether validate() itself ran.
            verdict = {"valid": False, "flags": [f"evalvalid_usage_error:{exc}"], "metrics": {}}

        meta = _build_meta(
            p=p,
            seed=seed,
            stage=stage,
            k=k,
            state_poll_interval_choices_s=state_poll_interval_choices_s,
            recefore=recefore,
            recefore_rc=recefore_rc,
            start_s=start_s,
            end_s=end_s,
            verdict=verdict,
            capture_subprocess=capture_subprocess,
        )
        _write_json(os.path.join(store_run_dir, "meta.json"), meta)

        chown_tree(store_run_dir)
        for new_ancestor in new_ancestors:
            chown_dir(new_ancestor)
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
    """Write `data` as sorted, indented JSON, closed to `_STORE_FILE_MODE` as soon as it exists."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.chmod(path, _STORE_FILE_MODE)


def _copy_locked_down(src: str, dest: str) -> None:
    """Copy `src` to `dest` and close it to `_STORE_FILE_MODE` immediately.

    2026-10-23: shutil.copy2 keeps the source's mode, and the final
    chown_tree pass only runs after evalvalid and the git lookups have
    already read these files; under a umask-0 root shell a copied file
    would stay group/other-writable for that whole stretch, so it could be
    edited in place before being validated and recorded.
    """
    shutil.copy2(src, dest)
    os.chmod(dest, _STORE_FILE_MODE)


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
    chown_dir: Callable[[str], None] = default_chown_dir,
    clock: Callable[[], float] = time.time,
    allow_dirty: bool = False,
    out=None,
    k: int = 1,
    state_poll_interval_choices_s: list[float] | None = None,
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

    Raises `DirtySemindexError` (see its own docstring) before anything else in this call touches
    the store or staging dirs, unless `allow_dirty` is set -- a whole-sweep guard, not a per-run
    one, since the point is to never start a sweep whose provenance is already in question.
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
    # Same reasoning as the stage check above: bad k / interval choices would make
    # every pair fail identically inside the loop's own try/except.
    evalgen.validate_settings(k, state_poll_interval_choices_s)
    if not allow_dirty:
        dirty = _git_dirty(_SEMINDEX_REPO_ROOT, capture_subprocess)
        if dirty is not False:
            raise DirtySemindexError(
                f"semindex tree ({_SEMINDEX_REPO_ROOT}) is not confirmed clean "
                f"(git status: {dirty!r}); commit or stash the changes first, "
                "or pass --allow-dirty to start the sweep anyway"
            )
    # Same "only chown what this call itself created" rule as `run_one`'s own ancestor handling
    # (see `default_chown_dir`'s docstring): `store` is the one ancestor `run_one` never sees
    # created fresh, since by the time it runs `store` already exists (this call just made it).
    store_preexisted = os.path.isdir(store)
    os.makedirs(store, exist_ok=True)
    if not store_preexisted:
        chown_dir(store)
    # 2026-10-23 fix (external review, P1 #2): gated here, once, before the
    # per-(p, seed) loop below -- not left to `run_one`'s own call to the
    # same check. That loop wraps every `run_one` call in a try/except
    # specifically so one run's own bug never aborts an entire overnight
    # sweep (see this function's own docstring); without this gate, a
    # mis-permissioned `store` would not refuse the sweep at all, it would
    # just make every single (p, seed) pair fail individually and print
    # `status=error`, with the sweep itself still exiting 0 as if it had
    # run. See `_check_dir_permissions`'s own docstring for what this is
    # actually checking and why.
    sudo_uid = os.environ.get("SUDO_UID")
    if sudo_uid is not None:
        _check_dir_permissions(store, invoking_uid=int(sudo_uid))
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
                chown_dir=chown_dir,
                clock=clock,
                k=k,
                state_poll_interval_choices_s=state_poll_interval_choices_s,
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
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="start the sweep even if the semindex tree is not confirmed clean (see DirtySemindexError)",
    )
    parser.add_argument("--k", type=int, default=1, help="records each origin creates per generation time (default 1)")
    parser.add_argument(
        "--interval-choices",
        type=float,
        nargs="+",
        default=None,
        dest="interval_choices",
        help="state-poll interval choices (s); omit for the default 2 s +/- 20%% policy",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: python -m semindex.evalrun ..."""
    args = parse_args(argv)
    seeds = parse_seeds(args.seeds)
    staging_root = tempfile.mkdtemp(prefix="semindex-evalrun-", dir=args.staging)
    try:
        run_all(
            args.p,
            seeds,
            args.stage,
            args.store,
            args.recefore,
            staging_root,
            allow_dirty=args.allow_dirty,
            k=args.k,
            state_poll_interval_choices_s=args.interval_choices,
        )
    except (DirtySemindexError, gen.ExperimentError) as exc:
        print(f"semindex.evalrun: {exc}", file=sys.stderr)
        return 2
    finally:
        if os.path.isdir(staging_root) and not os.listdir(staging_root):
            os.rmdir(staging_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
