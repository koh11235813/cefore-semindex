"""Loads a poll decider from an on-disk Python file named by a run's config.

A decider is not vendored into this package: it is arbitrary code a run
points at by absolute path, and it may depend on libraries (a model
framework, for instance) that the agent's own dependency set
(`pyproject.toml`) deliberately excludes. This module exists so that one
run's choice of decider is pinned to one exact file: the config carries a
digest of the file's bytes, and a decider whose file has drifted since the
digest was recorded is refused rather than silently run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys

# Fixed name the decider file is imported under. It is not a package name
# of this repository, so a decider script cannot shadow `semindex` or
# anything `semindex` itself imports.
DECIDER_MODULE_NAME = "semindex_poll_decider"


def load_poll_decider(spec: dict, *, node: str) -> object:
    """Load, verify and instantiate the poll decider named by `spec`.

    `spec` is a run's `state_poll_decider` value: `{"path", "sha256",
    "factory", "args"}`. The file at `path` is read exactly once; its bytes
    are hashed and checked against `sha256` before any of them run, and the
    module that gets imported is built from those same bytes rather than a
    second read of the path, so a file that changes on disk after the check
    cannot be what executes. `factory` is then called as
    `factory(node=node, **args)` and the object it returns is handed back as
    it is: what a decider must offer is defined once, by the engine's own
    calls, and an object lacking one of them fails at that call by name.
    """
    path = spec["path"]
    try:
        with open(path, "rb") as fh:
            source = fh.read()
    except FileNotFoundError as exc:
        raise ValueError(f"poll decider file not found: {path}") from exc

    actual_sha256 = hashlib.sha256(source).hexdigest()
    expected_sha256 = spec["sha256"]
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"poll decider {path} sha256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    # A plain module object, built from the bytes already hashed above. No
    # loader is attached, so nothing can re-read `path` to populate it.
    module_spec = importlib.util.spec_from_loader(DECIDER_MODULE_NAME, loader=None)
    module = importlib.util.module_from_spec(module_spec)
    # 2026-11-27 fix: a decider file may itself be a dataclass module (it is
    # ordinary Python, not sandboxed), and `from __future__ import annotations`
    # makes `dataclasses` resolve its own module from `sys.modules[__module__]`
    # while that module's own top level is still running -- so the module has to
    # be registered under sys.modules, and carry its own path as __file__,
    # *before* exec runs, not after.
    module.__file__ = path
    sys.modules[DECIDER_MODULE_NAME] = module
    try:
        exec(compile(source, path, "exec"), module.__dict__)
    except BaseException:
        # Nothing of a module whose top level raised should be left importable
        # under the fixed name for the rest of the process.
        sys.modules.pop(DECIDER_MODULE_NAME, None)
        raise

    factory_name = spec["factory"]
    factory = getattr(module, factory_name, None)
    if factory is None:
        raise ValueError(f"poll decider {path} has no factory named {factory_name!r}")

    return factory(node=node, **spec["args"])
