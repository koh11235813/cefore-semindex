"""Tests for the poll decider loader."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

from semindex.decider import DECIDER_MODULE_NAME, load_poll_decider

_VALID_DECIDER = """
def build(*, node, **kwargs):
    return _RecordingDecider(node, kwargs)


class _RecordingDecider:
    def __init__(self, node, kwargs):
        self.node = node
        self.kwargs = kwargs

    def decide(self, obs):
        return 1.0

    def collect(self):
        return []

    def close(self):
        pass
"""


def _write_decider(tmp_path: Path, source: str) -> tuple[Path, str]:
    """Write `source` to a decider file in `tmp_path` and return its path and sha256."""
    path = tmp_path / "decider.py"
    path.write_text(source)
    digest = hashlib.sha256(source.encode()).hexdigest()
    return path, digest


@pytest.fixture(autouse=True)
def _clean_decider_module():
    """Keep the fixed import slot empty between tests, since `sys.modules` is a process-wide dict."""
    sys.modules.pop(DECIDER_MODULE_NAME, None)
    yield
    sys.modules.pop(DECIDER_MODULE_NAME, None)


def test_a_correct_file_loads_and_its_factory_receives_the_node_and_the_arguments(tmp_path: Path):
    """A well-formed decider file loads and its factory is called with the node name and the configured args."""
    path, digest = _write_decider(tmp_path, _VALID_DECIDER)
    spec = {"path": str(path), "sha256": digest, "factory": "build", "args": {"coef": 3}}
    decider = load_poll_decider(spec, node="h1")
    assert decider.node == "h1"
    assert decider.kwargs == {"coef": 3}


def test_a_wrong_sha256_refuses_before_any_code_of_the_file_runs(tmp_path: Path):
    """A sha256 mismatch names both digests and the path, and the file's top level never executes."""
    source = "raise RuntimeError('the decider file ran')\n"
    path = tmp_path / "decider.py"
    path.write_text(source)
    wrong_digest = "0" * 64
    spec = {"path": str(path), "sha256": wrong_digest, "factory": "build", "args": {}}
    with pytest.raises(ValueError) as excinfo:
        load_poll_decider(spec, node="h1")
    actual_digest = hashlib.sha256(source.encode()).hexdigest()
    assert str(excinfo.value) == (
        f"poll decider {path} sha256 mismatch: expected {wrong_digest}, got {actual_digest}"
    )


def test_a_missing_file_refuses_naming_the_path(tmp_path: Path):
    """A decider path that does not exist on disk refuses with a message naming that path."""
    path = tmp_path / "nowhere.py"
    spec = {"path": str(path), "sha256": "0" * 64, "factory": "build", "args": {}}
    with pytest.raises(ValueError) as excinfo:
        load_poll_decider(spec, node="h1")
    assert str(excinfo.value) == f"poll decider file not found: {path}"


def test_a_missing_factory_refuses_naming_it(tmp_path: Path):
    """A factory name absent from the decider file refuses with a message naming that factory."""
    path, digest = _write_decider(tmp_path, _VALID_DECIDER)
    spec = {"path": str(path), "sha256": digest, "factory": "no_such_factory", "args": {}}
    with pytest.raises(ValueError) as excinfo:
        load_poll_decider(spec, node="h1")
    assert str(excinfo.value) == f"poll decider {path} has no factory named 'no_such_factory'"


def test_the_loaded_module_s_own_file_attribute_is_the_decider_s_path(tmp_path: Path):
    """A decider file that reads its own `__file__` (to find a sibling file next to
    it, say) must see the path it was loaded from, not an unset attribute: the
    factory is called from inside the module's own top level having already run, so
    this is the same guarantee every ordinarily-imported module gives its code."""
    source = """
_MODULE_FILE = __file__


def build(*, node, **kwargs):
    return _Reporting()


class _Reporting:
    def decide(self, obs):
        return _MODULE_FILE

    def collect(self):
        return []

    def close(self):
        pass
"""
    path, digest = _write_decider(tmp_path, source)
    spec = {"path": str(path), "sha256": digest, "factory": "build", "args": {}}
    decider = load_poll_decider(spec, node="h1")
    assert decider.decide(None) == str(path)


def test_a_decider_file_that_reads_its_own_entry_out_of_sys_modules_at_its_top_level_succeeds(tmp_path: Path):
    """A decider file whose top level reads `sys.modules[__name__]` while it is
    still running -- a dataclass module using `from __future__ import annotations`
    does exactly this of its own accord, to resolve its own annotations -- must see
    itself already registered there: `load_poll_decider` has to put the module in
    `sys.modules` *before* `exec` runs, not after, or this raises KeyError instead
    of returning the loaded decider."""
    source = """
import sys

_SELF = sys.modules[__name__]


def build(*, node, **kwargs):
    return _Reporting()


class _Reporting:
    def decide(self, obs):
        return _SELF.__name__

    def collect(self):
        return []

    def close(self):
        pass
"""
    path, digest = _write_decider(tmp_path, source)
    spec = {"path": str(path), "sha256": digest, "factory": "build", "args": {}}
    decider = load_poll_decider(spec, node="h1")
    assert decider.decide(None) == DECIDER_MODULE_NAME


def test_a_decider_file_whose_top_level_raises_leaves_the_import_slot_empty(tmp_path: Path):
    """A decider file that fails before it ever defines its factory must not leave a
    half-initialised module registered under the fixed name for the rest of the
    process -- the next run's load_poll_decider call (or test) would otherwise see a
    stale, broken module instead of either a clean slate or its own refusal."""
    source = "raise RuntimeError('this decider is broken')\n"
    path, digest = _write_decider(tmp_path, source)
    spec = {"path": str(path), "sha256": digest, "factory": "build", "args": {}}
    with pytest.raises(RuntimeError):
        load_poll_decider(spec, node="h1")
    assert DECIDER_MODULE_NAME not in sys.modules


def test_the_module_imports_only_the_standard_library():
    """`semindex.decider` must gain the agent no dependency, so every name it imports is a stdlib module."""
    import ast
    import inspect

    import semindex.decider as decider_module

    tree = ast.parse(inspect.getsource(decider_module))
    roots = set()
    # ast.walk and not the top-level statements only: an import inside a function
    # would add the dependency just the same.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    assert roots <= sys.stdlib_module_names


def test_the_bytes_that_were_hashed_are_the_bytes_that_run(tmp_path: Path, monkeypatch):
    """A file that changes on disk the instant after it is read cannot be what the loader executes."""
    path, digest = _write_decider(tmp_path, _VALID_DECIDER)

    real_open = open
    opened = []

    def spying_open(file, *args, **kwargs):
        handle = real_open(file, *args, **kwargs)
        if str(file) == str(path) and not opened:
            opened.append(file)
            real_read = handle.read

            def read_then_swap(*a, **kw):
                # Capture the real read's result first, then overwrite the
                # file on disk with content that would raise if it were
                # ever executed. A second read inside the loader would see
                # this instead of the bytes already hashed.
                data = real_read(*a, **kw)
                fd = os.open(str(path), os.O_WRONLY | os.O_TRUNC)
                os.write(fd, b"raise RuntimeError('a second read ran this file')\n")
                os.close(fd)
                return data

            handle.read = read_then_swap
        return handle

    monkeypatch.setattr("builtins.open", spying_open)
    spec = {"path": str(path), "sha256": digest, "factory": "build", "args": {}}
    decider = load_poll_decider(spec, node="h1")
    assert decider.decide(None) == 1.0
