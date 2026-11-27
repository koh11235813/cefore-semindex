"""Loads M (evidence/24's linear decider) as a poll decider for the live engine.

`build` is the factory the loader calls. It checks that the exact bytes of
every file the import below loaded, and of the coefficient file passed in,
are the ones this comparison was scored with, then constructs M and hands
back an object with the three methods the engine calls. M answers every
resolution synchronously, so no delaying wrapper is needed for it; one is
still available, through `fakes.wrap`, for a run that wants to exercise the
deferred path with M's own answers rather than a constant.
"""
import sys
import os
# 2026-11-27: node processes run as root and are started with only PYTHONPATH in
# their environment (no PYTHONDONTWRITEBYTECODE), so without this flag root would
# leave __pycache__ directories in a checkout its own owner cannot delete. It
# matters most here, because importing m24 below pulls in a dozen modules of
# the checkout, every one of which would otherwise be compiled to a .pyc.
sys.dont_write_bytecode = True
# 2026-11-27: `dont_write_bytecode` only stops a NEW .pyc from being WRITTEN; Python
# still READS an existing __pycache__/*.pyc whose header (mtime, size) matches the
# source on disk, and `_check_pins` below hashes the source bytes on disk, not
# whatever bytecode import actually ran. Cache LOOKUPS are disabled outright here,
# before `import m24` below (or anything it pulls in) loads a single byte, for the
# same reason `lserver.py` sets this beside its own `dont_write_bytecode`.
sys.pycache_prefix = os.devnull

import hashlib  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EVID = os.path.dirname(HERE)

_EV24 = os.path.join(EVID, "24")
if _EV24 not in sys.path:
    sys.path.insert(0, _EV24)

_IMPORT_T0 = time.perf_counter()
import m24  # noqa: E402
IMPORT_S = time.perf_counter() - _IMPORT_T0

# Measured by scanning sys.modules for a __file__ under EVID right after
# `import m24` above, in this checkout, on this date. If the import chain
# ever changes, this table and the measurement disagree and _check_pins
# below is what notices: a pinned entry that is no longer loaded is refused
# as stale, and a newly loaded file with no entry is refused as unpinned.
PINS = {
    # A digest was recorded for each of these when the comparison this
    # decider adapts was scored (the "script digests" / "code digests"
    # blocks of eval24.txt and latency24.txt, plus candidate a's coefficient
    # line): a mismatch here means this checkout no longer has the bytes
    # that comparison read. coef_a.npy's hash is taken from the `coef`
    # argument itself, once _check_pins has confirmed that argument's path
    # is this file.
    "24/m24.py": "f93641cb4022a21437365f2f019d8c705e2cdffc99ba1654b879a4cdcdb79719",
    "24/features.py": "437e840b04d10d6e58ff28513f994cb85c63c135bc0baee2b230592b30e7b80e",
    "24/charges.py": "39c9b8669488b5e848e49e97781988f285245ecbe728b1526f70a5e41e8c0c40",
    "22/port.py": "5f74617a04b43d71f2a29153e26251e357cd44436925ecb9fa6b596fc53cca07",
    "22/score.py": "24743e3547ed877cec0a35c8925f12f20c2ed6941d4d805268701fbd0e0f3e49",
    "13/replay.py": "f689d3a82fb3740f2aedf4a6b523329c971de05b4947f103f2c0bb882b58f49d",
    "24/coef_a.npy": "bbf5a5c7abf1c6fa476c728b3855ecdcab601c580452a2101cf4203e560b8ebf",
    # Recorded by the other learned decider's scoring (its latency read-out
    # lists this file), not by the scoring of the model this file adapts.
    "22/observe.py": "9a5808bb40ddfd0f827fa04ff3aa7ad3c2c9c69243104b9651bb7e4da90056c2",
    # No scoring run recorded a digest for these: they are loaded by `import
    # m24` all the same, so what this part of the table enforces is only
    # that every run of this comparison loads the same bytes as every
    # other run, not that those bytes match any particular earlier result.
    "13/trickle.py": "df86f490433b61f7a6529da20b408e47b947a3a5e02cf0079d6d1c9915b647eb",
    "14/after.py": "27325358afb2ed0c1fe171174953f7d17df306efc7ff7c2a628084253f28236f",
    "14/workload.py": "71f6ad01f9a8cafa6ae7a90702fb837b4461e1e2856c92c14f10dd7d9c5c466e",
    "15/onoff.py": "8daa6322fd68d514800eb348e3e03ff5410c6c1b0e28bbb6dc8a654aa60ddc6d",
    "18/bound.py": "564b2cd6cc7fe4fab5dc9e82cf6881d841bf5b784ef261c9cde7371532d739bf",
}


def _sha256_file(path):
    """Return the sha256 hex digest of the file at `path`, read fresh off disk."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _check_pins(coef):
    """Refuse to run on anything but the exact bytes this comparison was scored with.

    Four separate failures, each a different way the checkout could have
    drifted from what the table above describes: `coef` is not the pinned
    coefficient file, a file was loaded that the table does not know about,
    a pinned file's bytes changed, or a pinned `.py` entry names a file
    that `import m24` no longer loads at all.
    """
    want_coef = os.path.join(EVID, "24", "coef_a.npy")
    if os.path.realpath(coef) != os.path.realpath(want_coef):
        raise RuntimeError("mdecider: coef %s is not %s" % (coef, want_coef))

    loaded = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f is None:
            continue
        rf = os.path.realpath(f)
        if rf.startswith(os.path.realpath(EVID) + os.sep) and \
                not rf.startswith(os.path.realpath(HERE) + os.sep):
            loaded[os.path.relpath(rf, os.path.realpath(EVID))] = rf

    for rel in loaded:
        if rel not in PINS:
            raise RuntimeError("mdecider: unpinned file loaded: %s" % rel)

    for rel, expected in PINS.items():
        path = coef if rel == "24/coef_a.npy" else os.path.join(EVID, rel)
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeError("mdecider: %s expected sha256 %s, got %s" % (rel, expected, actual))

    for rel in PINS:
        if rel.endswith(".py") and rel not in loaded:
            raise RuntimeError("mdecider: pinned file %s was not loaded; the table is stale" % rel)


class MDecider:
    """Adapts `m24.M` (a callable) to the engine's `decide`/`collect`/`close` protocol.

    M answers synchronously, so collect() never has anything to return and
    close() has no state to release.
    """

    def __init__(self, m):
        """Wrap the constructed `m24.M` instance `m`."""
        self._m = m

    def decide(self, obs):
        """Answer `obs` by calling M directly; M itself is synchronous."""
        return self._m(obs)

    def collect(self):
        """M never defers, so there is never anything to release."""
        return []

    def close(self):
        """Nothing to flush or close."""
        pass


def build(node, coef, record_dir=None, delay_s=None):
    """Factory for the loader's `factory` key: build M, pinned, and wrap it if asked.

    `record_dir` and `delay_s` are both optional, and importing `fakes` is
    deferred to this call and skipped entirely when neither is given, so a
    plain run never pays for or depends on that module.
    """
    build_t0 = time.perf_counter()
    _check_pins(coef)
    inner = MDecider(m24.M.from_file(coef))
    build_s = time.perf_counter() - build_t0
    if record_dir is None and delay_s is None:
        return inner
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import fakes
    return fakes.wrap(inner, node=node, record_dir=record_dir, delay_s=delay_s,
                       meta={"inner": "mdecider", "coef": coef, "import_s": IMPORT_S,
                             "build_s": build_s, "pins": len(PINS)})
