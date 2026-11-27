#!/usr/bin/env python3
"""The model SERVER: one process, one model, one `L`, answering five nodes.

A decision is `L.__call__(obs)` built straight from the wire request; this file never
renders a prompt, tokenises, runs a forward pass or applies the tie rule itself -- that
is the whole point of a server, that the object scored in the replay is the object that
answers on the emulator, not a second copy of it built to look the same.

Start-up refuses to run on anything but the exact bytes this step was scored on: code
digests, then (unless `--stub`) the adapter's two files, the installed library versions,
and a live reproduction of the repeat check `tune.py` and `latency.py` both ran on seed
71's k 72 row.  A mismatch anywhere exits non-zero naming the file, the expected digest
and the actual one, before a single socket is opened.

What this establishes, precisely: the source files named in `CODE_DIGESTS` and
`LOADED_NO_DIGEST` have, ON DISK, AT START-UP, the exact bytes recorded, with bytecode
caches disabled so nothing stale is ever read in their place.  What it does NOT
establish: it does not hash the bytes the import system actually read.  `CODE_DIGESTS`
is checked before `import decider` runs a line of the checkout's own code, but
`LOADED_NO_DIGEST` is only checked by `check_loaded_pins()`, which in real mode runs
after the adapter digests, the version check, the model load and the repeat check have
all already executed -- so a `LOADED_NO_DIGEST` file that changes on disk any time
before that call is imported, and run, as whatever it changed to, and nothing here
notices until the hash check that follows.
"""
import sys
import os
# The checkout this runs from is shared with the scoring scripts whose digests this file
# itself checks; a stray __pycache__ left here by a root-owned process would be noise in
# every later inventory of what produced a result, so this is set before any import of a
# checkout module (`os` itself is a builtin, not something this flag affects).
sys.dont_write_bytecode = True
# 2026-11-27: `dont_write_bytecode` only stops a NEW .pyc from being WRITTEN; Python still
# READS an existing __pycache__/*.pyc whose header (mtime, size) matches the source on disk,
# and the checkout this runs from on the GPU machine carries such directories from earlier
# scoring runs. `check_code_digests()` below hashes the source bytes on disk, not whatever
# bytecode import actually ran, so cache LOOKUPS are disabled outright here, before `import
# decider` below (or anything it pulls in) loads a single byte.
sys.pycache_prefix = os.devnull

import hashlib       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EVID = os.path.dirname(HERE)
_EV23 = os.path.join(EVID, "23")
if _EV23 not in sys.path:
    sys.path.insert(0, _EV23)

# ---------------------------------------------------------------------------
# The code digests, read off `latency.txt`'s own "the code that produced every figure
# below" block, and confirmed against this checkout with `sha256sum` before this file
# was written.  These are the files the
# scoring this server answers for was taken against; a checkout whose bytes differ here
# answered a question `latency.txt` never asked.
# ---------------------------------------------------------------------------
CODE_DIGESTS = (
    ("23/decider.py",
     "a3a913e9e1c0422d9bae5afb1c9f2974c01c3ddbfbfdc1e616d02f312d7080b0"),
    ("23/prefill.py",
     "bd47719cd131638e741f7c309c12d64b20a91d1ecc73d2999837fb70b064a372"),
    ("22/observe.py",
     "9a5808bb40ddfd0f827fa04ff3aa7ad3c2c9c69243104b9651bb7e4da90056c2"),
    ("22/port.py",
     "5f74617a04b43d71f2a29153e26251e357cd44436925ecb9fa6b596fc53cca07"),
    ("22/label.py",
     "3a7b12bbdd95b1ca2eb83bed835f39b08a53e6ae302f6a776b6db321563c54d0"),
    ("22/score.py",
     "24743e3547ed877cec0a35c8925f12f20c2ed6941d4d805268701fbd0e0f3e49"),
    ("13/replay.py",
     "f689d3a82fb3740f2aedf4a6b523329c971de05b4947f103f2c0bb882b58f49d"),
)

# `import decider` above pulls in `port.py`'s own imports (`bound.py` of evidence/18 and
# `replay.py` of evidence/13, which in turn loads `trickle.py`, `after.py`, `workload.py`
# and `onoff.py`).  No scoring run of this effort recorded a digest for these five: they
# are loaded all the same, so what this half of the table enforces is only that every run
# of this server loads the same bytes as every other run, not that those bytes match any
# particular earlier result.  Pinned to this checkout's bytes, exactly as `mdecider.py`'s
# own `PINS` table pins the same five files for the same reason.
LOADED_NO_DIGEST = (
    ("18/bound.py",
     "564b2cd6cc7fe4fab5dc9e82cf6881d841bf5b784ef261c9cde7371532d739bf"),
    ("13/trickle.py",
     "df86f490433b61f7a6529da20b408e47b947a3a5e02cf0079d6d1c9915b647eb"),
    ("14/after.py",
     "27325358afb2ed0c1fe171174953f7d17df306efc7ff7c2a628084253f28236f"),
    ("14/workload.py",
     "71f6ad01f9a8cafa6ae7a90702fb837b4461e1e2856c92c14f10dd7d9c5c466e"),
    ("15/onoff.py",
     "8daa6322fd68d514800eb348e3e03ff5410c6c1b0e28bbb6dc8a654aa60ddc6d"),
)

# The two adapter files' digests, read off `eval.txt` and `tune.txt`: the digests the
# tuning and the scoring recorded for candidate b's kept epoch.  A directory that answers to these two is the file both of those runs read.
ADAPTER_CONFIG_SHA = "4d76d6c63d6a9a20ba4018f64e11b76743645617211a8aa7f019126f305cd760"
ADAPTER_MODEL_SHA = "268a3dd4db8b6e8fc5452ba3aa1ce174a7990979026f82fe0e0ce896ecc32514"

# Named directly, rather than read off `--adapter`'s own directory name: the two digests
# above already pin candidate b's kept epoch, and deriving the name from a directory that
# could be renamed would let a renamed `--adapter` silently look up the wrong candidate's
# archived triple below instead of failing loudly.
EXPECTED_CANDIDATE = "b"

VERSIONS = {"torch": "2.15.0+cu132", "transformers": "5.24.0", "peft": "0.22.0"}


def die(msg):
    """Exit non-zero with one line naming what differed; nothing after this runs."""
    sys.stderr.write("lserver: %s\n" % msg)
    raise SystemExit(1)


def sha256_file(path):
    """The sha256 hex digest of one file's bytes, read fresh off disk."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def check_code_digests():
    """Every file `CODE_DIGESTS` names has exactly the bytes `latency.txt` recorded."""
    for rel, want in CODE_DIGESTS:
        path = os.path.join(EVID, rel)
        got = sha256_file(path)
        if got != want:
            die("%s: expected sha256 %s, got %s" % (rel, want, got))
    return len(CODE_DIGESTS)


# 2026-11-27: hashed here, at module load, BEFORE `import decider` below runs a single
# line of the checkout's own code (with bytecode caches disabled, above) -- so this check
# is of the bytes on disk that are about to be imported, not of bytes some earlier import
# already executed. `CODE_DIGEST_COUNT` is kept so `main()` can still report how many
# digests were checked, without hashing everything a second time.
CODE_DIGEST_COUNT = check_code_digests()

import decider as D  # noqa: E402  the read-out this server answers with; see decider.py

import argparse      # noqa: E402
import errno         # noqa: E402
import json          # noqa: E402
import select        # noqa: E402
import signal        # noqa: E402
import socket        # noqa: E402
import stat          # noqa: E402
import time          # noqa: E402
from collections import deque  # noqa: E402


def check_loaded_pins():
    """Every module loaded from under EVID, by `import decider` or by what ran since, is
    one of the two tables.

    An unpinned file loaded, a pinned file not loaded (the table is stale), or a pinned
    file whose bytes moved are the three ways this checkout could have drifted; this is
    the one check for all three.  Called AFTER the model is loaded in real mode, not
    right after `import decider`: the renderer module `load_real()`'s repeat check
    touches is imported lazily, on the first rendering, and is in `sys.modules` only
    once that has happened.
    """
    known = {rel: sha for rel, sha in CODE_DIGESTS if rel.endswith(".py")}
    known.update(dict(LOADED_NO_DIGEST))
    loaded = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f is None:
            continue
        rf = os.path.realpath(f)
        evid_real = os.path.realpath(EVID)
        if rf.startswith(evid_real + os.sep) and \
                not rf.startswith(os.path.realpath(HERE) + os.sep):
            loaded[os.path.relpath(rf, evid_real)] = rf
    for rel in loaded:
        if rel not in known:
            die("unpinned file loaded: %s" % rel)
    for rel, want in LOADED_NO_DIGEST:
        if rel not in loaded:
            die("pinned file %s was not loaded; the table is stale" % rel)
        got = sha256_file(os.path.join(EVID, rel))
        if got != want:
            die("%s: expected sha256 %s, got %s" % (rel, want, got))


def check_adapter_digests(adapter_dir):
    """The two adapter files are the bytes the tuning and the scoring both read."""
    for name, want in (("adapter_config.json", ADAPTER_CONFIG_SHA),
                       ("adapter_model.safetensors", ADAPTER_MODEL_SHA)):
        path = os.path.join(adapter_dir, name)
        got = sha256_file(path)
        if got != want:
            die("%s: expected sha256 %s, got %s" % (path, want, got))


def check_versions():
    """The installed torch, transformers and peft are the ones `latency.txt` ran on."""
    import torch
    import transformers
    import peft
    got = {"torch": torch.__version__, "transformers": transformers.__version__,
           "peft": peft.__version__}
    for name, want in VERSIONS.items():
        if got[name] != want:
            die("%s: expected version %s, got %s" % (name, want, got[name]))


def repeat_row():
    """Seed 71's k 72 row, read off `decider.P.LABELS` the way `latency.py` reads it.

    `LABELS` is `prefill.py`'s own module constant for the label file's path, so this
    reads the SAME file `latency.py` read it from rather than a second, hand-typed path.
    """
    import gzip
    with gzip.open(D.P.LABELS, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            if int(row["seed"]) == 71 and int(row["k"]) == 72:
                return row
    die("no seed 71 k 72 row in %s" % D.P.LABELS)


def load_real(adapter_dir):
    """Tokenizer, base and adapter through `decider.py`'s own calls (as `eval.py`'s
    `load_for` makes them), then the live repeat check.  Returns `(read_three, tok,
    cold_ms, warm_ms, repeat_logits)`; nothing here has decided an interval yet.
    """
    import torch
    tok = D.load_tokenizer()
    base = D.load_base()
    model = D.attach_adapter(base, adapter_dir, "lserver start-up")
    row = repeat_row()
    text = D.text_of(row["state"], row["resolved"], row["history"], row["ts_ms"],
                     row["neighbor"], "lserver repeat row")
    prompt = D.P.chat_text(tok, text)
    cand_ids = torch.tensor(D.candidate_ids(tok, prompt), device=D.DEV)

    t0 = time.perf_counter()
    first = D.readout(model, tok, cand_ids, prompt)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    second = D.readout(model, tok, cand_ids, prompt)
    arm_a, _ = D.pick(first)
    arm_b, _ = D.pick(second)
    D.check_repeat(first, second, arm_a, arm_b, "lserver repeat check")
    # 2026-11-27: the expected triple is read off `tune.txt` itself, through
    # `decider.py`'s own `tune_kept()`/`check_triple()` -- the same table and the same
    # comparator `latency.py`'s own repeat check reads its `TUNED_TRIPLES` from and checks
    # against -- rather than a hand-typed literal that a re-tune would leave silently
    # stale here while `tune.txt` itself moved on.
    want = D.tune_kept()[EXPECTED_CANDIDATE]["triple"]
    # 2026-11-27: `check_triple` raises `decider.Fail` (a traceback) rather than exiting
    # with the single line every other start-up refusal in this file uses; caught here and
    # handed to `die()` so a repeat-check mismatch reads the same way as the rest of them.
    try:
        D.check_triple(first, want, EXPECTED_CANDIDATE, "lserver repeat check, first pass")
        D.check_triple(second, want, EXPECTED_CANDIDATE, "lserver repeat check, second pass")
    except D.Fail as exc:
        die(str(exc))

    D.readout(model, tok, cand_ids, prompt)   # one more pass, untimed, as warm-up
    t1 = time.perf_counter()
    D.readout(model, tok, cand_ids, prompt)
    warm_ms = (time.perf_counter() - t1) * 1000.0

    def read_three(rendered_prompt):
        return D.readout(model, tok, cand_ids, rendered_prompt)

    return read_three, tok, cold_ms, warm_ms, list(first)


def stub_read_three(prompt):
    """Model-free stand-in: three floats from the prompt's own hash, so different
    observations answer differently with no card and no checkpoint.  `L` is still built
    with `tok=None`, the card-free path `decider.py`'s own docstring names, so the
    send-time memory, the tie rule and the no-node-name assertion are all real; only the
    forward pass is fake.
    """
    h = hashlib.sha256(prompt.encode("utf-8")).digest()
    return (h[0] / 255.0 * 20.0, h[1] / 255.0 * 20.0, h[2] / 255.0 * 20.0)


class Conn:
    """One accepted connection's buffers; nothing here is specific to this protocol."""

    def __init__(self, sock):
        self.sock = sock
        self.inbuf = b""
        self.outbuf = b""
        self.closed = False


def build_obs(req):
    """The four-field `obs` `L.__call__` expects, straight off the wire request."""
    return {"state": req["state"], "resolved": req["resolved"], "history": req["history"],
            "where": {"node": req["node"], "neighbor": req["neighbor"],
                      "kind": req["kind"], "ts_ms": req["ts_ms"]}}


REQUIRED = ("token", "node", "neighbor", "kind", "ts_ms", "state", "resolved", "history")


def main():
    """Check everything that says which model this is, then serve until told to stop.

    The socket is bound last, after every check and the model load, so that a client
    which can connect is a client that will be answered by the scored model.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args()

    combined = CODE_DIGEST_COUNT

    # `last_triple` is how the three logits of the LAST resolution reach the log: `L`
    # itself only returns the chosen interval, so `read_three` is wrapped to remember
    # what it last returned.  Re-reading the logits any other way would be a second
    # forward pass, and a second pass is not the pass the answer came from.
    last_triple = [None]
    repeat_logits = None
    if args.stub:
        def read_three(prompt):
            """Stub read-out, remembering its own last triple for the log."""
            three = stub_read_three(prompt)
            last_triple[0] = three
            return three
        Lobj = D.L(read_three)
        cold_ms = warm_ms = 0.0
        adapter_report = "stub: adapter not read"
        # 2026-11-27: the real branch below can only call this AFTER the model is
        # loaded, because the renderer module `load_real()` touches is imported
        # lazily on the first rendering and is in `sys.modules` only then. The stub
        # branch has no such ordering constraint -- `D.L` here is built with
        # `stub_read_three`, so nothing it does before the first decision imports
        # that module -- and the renderer is already covered by `CODE_DIGESTS`
        # regardless of when it loads, so one call here, right after `L` is built,
        # is the whole sweep this branch needs.
        check_loaded_pins()
    else:
        check_adapter_digests(args.adapter)
        check_versions()
        inner, tok, cold_ms, warm_ms, repeat_logits = load_real(args.adapter)
        # Only after load_real() has returned: the repeat check inside it is what
        # renders a prompt for the first time, and only a first rendering imports
        # the renderer module this sweep must also find pinned.
        check_loaded_pins()

        def read_three(prompt):
            """Real read-out, remembering its own last triple for the log."""
            three = inner(prompt)
            last_triple[0] = three
            return three
        Lobj = D.L(read_three, tok)
        adapter_report = "%s %s" % (ADAPTER_CONFIG_SHA, ADAPTER_MODEL_SHA)

    log = open(args.log, "a", buffering=1, encoding="utf-8")

    def write_log(ev):
        """Append one JSON object as a line; the file is opened line-buffered."""
        log.write(json.dumps(ev) + "\n")

    # A path that nothing answers on is stale and safe to remove; a path that something
    # DOES answer on means a server is already running there, and starting a second one
    # against the same socket would silently steal half of its traffic.
    try:
        socket_mode = os.lstat(args.socket).st_mode
    except FileNotFoundError:
        socket_mode = None
    if socket_mode is not None:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(args.socket)
            probe.close()
            die("socket %s already answers a connect; refusing to start a second server"
                % args.socket)
        except OSError:
            probe.close()
            # 2026-11-27: only ever unlink a SOCKET left behind by a dead server; anything
            # else at this path (a regular file, a directory) is a mistake worth stopping
            # for, not bytes to delete and silently replace with this run's own listener.
            if not stat.S_ISSOCK(socket_mode):
                die("socket path %s exists and is not a socket; refusing to remove it"
                    % args.socket)
            os.unlink(args.socket)

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(args.socket)
    listener.listen(16)
    listener.setblocking(False)

    served = [0]
    errors = [0]
    stop = [False]

    def on_signal(signum, _frame):
        """Ask the main loop to stop at its next iteration; no work done here."""
        stop[0] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    ready = {"ev": "ready", "adapter": adapter_report, "code_digests": combined,
             "versions": VERSIONS if not args.stub else "stub", "cold_ms": cold_ms,
             "warm_ms": warm_ms, "pid": os.getpid(), "stub": args.stub,
             "repeat_logits": repeat_logits}
    write_log(ready)
    # Flushed by hand: started under nohup with stdout sent to a file, the line would
    # otherwise sit in the buffer until exit, and it is the line a launcher waits for.
    print("READY %s" % json.dumps(ready), flush=True)

    conns = {}             # fd -> Conn
    # ONE FIFO over every connection, batch 1, one request processed at a time.  Three
    # reasons, none of them a style preference:
    #  - batch 1 is the read-out the score, the agreement and the latency of this effort
    #    were all measured at; batching in bf16 changes the argmax of some label rows.
    #  - what the single queue actually guarantees is narrower than "never reorders this
    #    connection's traffic": it guarantees that two INTERVAL answers on one connection
    #    are never reordered relative to each other, which is the engine's own ordering
    #    assumption (one pair's answers arrive in the order they were asked) and an
    #    out-of-order answer for one pair makes the engine raise.  A probe reply or an
    #    error reply is written to `outbuf` at once, in `handle_line`, outside this queue
    #    entirely, and MAY overtake an earlier request's still-queued interval answer.
    #  - per-node fairness would make one node's answer time depend on another node's
    #    load in a way no node can see or compensate for.
    queue = deque()

    def close_conn(fd):
        """Drop a connection and whatever it has queued; log that it closed."""
        c = conns.pop(fd, None)
        if c is None:
            return
        c.closed = True
        try:
            c.sock.close()
        except OSError:
            pass
        write_log({"ev": "conn", "what": "close", "n_open": len(conns)})

    def handle_line(c, line, arrival):
        """Parse one complete wire line and either answer it now (probe, bad envelope)
        or enqueue it for `process_one` to answer through `L`.

        `line` is the raw bytes between two newlines, not yet decoded: decoding happens
        here, in the same `try` as `json.loads`, so that invalid UTF-8 is one more way
        this one line can fail to parse rather than an exception out of the caller's recv
        loop that would end the whole process for the other four nodes it serves.
        """
        if not line:
            return
        try:
            req = json.loads(line.decode("utf-8"))
        except Exception as exc:
            # 2026-11-27 crash fix: `UnicodeDecodeError` (invalid UTF-8) and
            # `RecursionError` (JSON nested deep enough) both used to escape the narrower
            # `except (ValueError, TypeError)` this used to have, and killed the server
            # for the other four nodes along with the one that sent the bad line.
            errors[0] += 1
            msg = "invalid line: %s" % exc
            write_log({"ev": "error", "token": None, "error": msg})
            c.outbuf += (json.dumps({"token": None, "error": msg})
                        + "\n").encode("utf-8")
            return
        if isinstance(req, dict) and req.get("probe") == 1:
            c.outbuf += (json.dumps({"probe": 1, "ready": True, "served": served[0]})
                        + "\n").encode("utf-8")
            return
        if not isinstance(req, dict) or any(k not in req for k in REQUIRED):
            tok = req.get("token") if isinstance(req, dict) else None
            msg = "not a valid request: missing one of %s" % (REQUIRED,)
            errors[0] += 1
            write_log({"ev": "error", "token": tok, "error": msg})
            c.outbuf += (json.dumps({"token": tok, "error": msg}) + "\n").encode("utf-8")
            return
        if req["kind"] != "state_poll_reschedule":
            # Any other `kind` is answered from `L`'s memory (`last_triple`) without a
            # forward pass -- there is no decision for it to make -- and that answer
            # would be logged carrying the PREVIOUS request's logits, which is wrong in
            # a way nothing downstream would notice on its own.
            msg = "unsupported kind: %r" % req["kind"]
            errors[0] += 1
            write_log({"ev": "error", "token": req.get("token"), "error": msg})
            c.outbuf += (json.dumps({"token": req.get("token"), "error": msg})
                        + "\n").encode("utf-8")
            return
        queue.append({"conn": c, "req": req, "arrival": arrival, "t_wall": time.time(),
                     "queued_behind": len(queue)})

    def process_one():
        """Answer the head of the FIFO, skipping entries whose connection has closed.

        Returns after at most one `L` call, so the select loop is re-entered and every
        socket re-polled between one decision and the next.
        """
        while queue:
            item = queue.popleft()
            if item["conn"].closed:
                continue
            t_start = time.monotonic()
            queue_ms = (t_start - item["arrival"]) * 1000.0
            req = item["req"]
            try:
                t0 = time.perf_counter()
                interval_s = Lobj(build_obs(req))
                compute_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:
                errors[0] += 1
                write_log({"ev": "error", "token": req.get("token"), "error": str(exc)})
                answer = {"token": req.get("token"), "error": str(exc)}
            else:
                served[0] += 1
                logits = list(last_triple[0]) if last_triple[0] is not None else None
                write_log({"ev": "req", "t_wall": item["t_wall"], "node": req["node"],
                          "neighbor": req["neighbor"], "token": req["token"],
                          "queue_ms": queue_ms, "compute_ms": compute_ms,
                          "interval_s": interval_s, "logits": logits,
                          "queued_behind": item["queued_behind"]})
                answer = {"token": req["token"], "interval_s": interval_s,
                         "logits": logits}
            item["conn"].outbuf += (json.dumps(answer) + "\n").encode("utf-8")
            return   # one request per loop pass, so sockets are re-polled promptly

    try:
        while not stop[0]:
            rlist = [listener] + [c.sock for c in conns.values() if not c.closed]
            wlist = [c.sock for c in conns.values() if not c.closed and c.outbuf]
            timeout = 0.0 if queue else 0.2
            try:
                readable, writable, _ = select.select(rlist, wlist, [], timeout)
            except (OSError, InterruptedError) as exc:
                if exc.args and exc.args[0] == errno.EINTR:
                    continue
                raise

            if listener in readable:
                try:
                    sock, _addr = listener.accept()
                    sock.setblocking(False)
                    conns[sock.fileno()] = Conn(sock)
                    write_log({"ev": "conn", "what": "open", "n_open": len(conns)})
                except OSError:
                    pass

            for c in list(conns.values()):
                if c.sock not in readable or c.closed:
                    continue
                try:
                    data = c.sock.recv(65536)
                except BlockingIOError:
                    continue
                except OSError:
                    # A reset connection stays readable for ever; closing it here is
                    # what keeps the loop from spinning on it between two decisions.
                    data = b""
                if not data:
                    close_conn(c.sock.fileno())
                    continue
                c.inbuf += data
                arrival = time.monotonic()
                while b"\n" in c.inbuf:
                    line, c.inbuf = c.inbuf.split(b"\n", 1)
                    handle_line(c, line, arrival)

            for c in list(conns.values()):
                if c.sock not in writable or c.closed or not c.outbuf:
                    continue
                try:
                    sent = c.sock.send(c.outbuf)
                    c.outbuf = c.outbuf[sent:]
                except OSError as exc:
                    if exc.errno != errno.EAGAIN:
                        close_conn(c.sock.fileno())

            process_one()
    finally:
        write_log({"ev": "stop", "served": served[0], "errors": errors[0],
                  "queued": len(queue), "reads": Lobj.reads, "ties": Lobj.ties})
        log.close()
        try:
            listener.close()
        except OSError:
            pass
        for c in list(conns.values()):
            try:
                c.sock.close()
            except OSError:
                pass
        try:
            os.unlink(args.socket)
        except OSError:
            pass

    raise SystemExit(0)


if __name__ == "__main__":
    main()
