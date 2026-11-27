"""The node-side client of a model server, loaded as a live poll decider.

This is the half of the design that runs inside a node process: one
`AF_UNIX` socket, one JSON line per request and per answer, nothing that
can block the agent's single tick loop. It asks a question at every
resolution and never waits for the answer there; `collect()` is the only
place an answer is ever picked up, and it is itself bounded to whatever
bytes are already sitting in the kernel's socket buffer. Every failure
mode the socket can produce -- refused, reset, full, slow -- is turned
into a counter instead of an exception, because an exception here does
not just fail one decision, it kills the node's networking for the rest
of the run.
"""
import sys
# 2026-11-27: node processes run as root and are started with only PYTHONPATH in
# their environment (no PYTHONDONTWRITEBYTECODE), so without this flag root would
# leave __pycache__ directories in a checkout its own owner cannot delete.
sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402

# The outbound buffer is bounded so that a server which stops reading turns
# into dropped requests, not unbounded memory growth inside a node process
# that is otherwise expected to run for the length of an experiment.
MAX_OUTBUF_BYTES = 1024 * 1024


class LClient:
    """Speaks the poll-decider wire protocol to one model server over one socket.

    One instance serves one node process and every neighbour it polls: the
    socket, the buffers and the counters are shared across pairs, while
    `mem` (the pair's last received answer) is kept per neighbour so a send
    that goes out before an answer arrives still has something sane to
    report.
    """

    def __init__(self, node, socket_path, send_start, stats_dir):
        """Build the client and attempt the one connection it will use.

        A unix connect either succeeds or fails immediately, so a short
        timeout here is just a guard against a listener that accepted the
        connection but never completes its side of the handshake; once
        connected the socket is switched to non-blocking for every call
        that follows, because nothing after this point is allowed to wait.
        """
        self.node = node
        self.socket_path = socket_path
        self.send_start = send_start
        self.stats_dir = stats_dir

        self.mem = {}
        self.outbuf = b""
        self.inbuf = b""
        self.outstanding = set()
        self.sock = None
        self.connected = False
        self.counters = dict.fromkeys(
            ["sends", "asked", "answered", "errors", "malformed", "dropped",
             "unreachable", "lost", "reconnects", "outstanding_at_close"], 0)

        connected = self._connect()
        # A client that never managed the first connection has nothing to
        # retry towards: the path it was given is wrong or the server is
        # not there at all, and every resolution for the rest of the run
        # is reported `unreachable` rather than spent on reconnect attempts
        # that would just repeat the same failure. A client that *did*
        # connect once and later lost the socket is a different case
        # (transient, worth one reconnect attempt per collect()); that
        # distinction is what this flag records.
        self._never_connected = not connected

    def _connect(self):
        """Try once to open the socket; return whether it is now usable.

        Never raises: every way this can fail (no such path, nothing
        listening, a stale socket file) is an `OSError` subclass, and all
        of them mean the same thing to the caller -- there is no
        connection -- so they are handled identically.
        """
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            sock.connect(self.socket_path)
            sock.setblocking(False)
        except OSError:
            self.sock = None
            self.connected = False
            return False
        self.sock = sock
        self.connected = True
        return True

    def decide(self, obs):
        """Answer a send from `mem`; turn a resolution into a wire request.

        Never blocks and never raises: a send only reads a dict, and a
        resolution only appends to an in-memory buffer and makes one
        non-blocking send() attempt.
        """
        where = obs["where"]
        neighbor = where["neighbor"]

        if where["kind"] != "state_poll_reschedule":
            # `send_start` is a required argument rather than a literal here
            # because the value this client must start a pair at comes from
            # the simulator's own arm table, and importing that module into
            # a node process would pull a whole simulator in with it. The
            # engine itself refuses any value outside its configured
            # choices, so a wrong `send_start` surfaces as a raise at the
            # pair's very first send rather than a silently-wrong interval.
            self.counters["sends"] += 1
            return self.mem.get(neighbor, self.send_start)

        resolved = obs["resolved"]
        poll_epoch_ms = resolved["poll_epoch_ms"]
        token = (neighbor, poll_epoch_ms)

        if not self.connected:
            self.counters["unreachable"] += 1
            return None

        line = json.dumps({
            "token": [neighbor, poll_epoch_ms],
            "node": self.node,
            "neighbor": neighbor,
            "kind": where["kind"],
            "ts_ms": where["ts_ms"],
            "state": obs["state"],
            "resolved": resolved,
            "history": obs["history"],
        }).encode("utf-8") + b"\n"

        if len(self.outbuf) + len(line) > MAX_OUTBUF_BYTES:
            self.counters["dropped"] += 1
            return None

        # The token is recorded as outstanding before the flush attempt
        # below, not after: a request can sit in `outbuf` for several
        # collect() calls before the kernel accepts it, and until it is
        # either answered or the connection is lost it is exactly as
        # "asked" as one already written to the socket.
        self.outbuf += line
        self.outstanding.add(token)
        self.counters["asked"] += 1
        self._flush()
        return None

    def _flush(self):
        """Push as much of `outbuf` as the socket will take right now.

        `BlockingIOError` (kernel buffer full) and any other send failure
        (broken pipe, reset) are treated the same here: the bytes stay
        queued and nothing about the socket's state is touched. Tearing
        the connection down is left entirely to collect()'s read loop, so
        `self.sock` is only ever set to `None` from one place; doing it
        here too would leave `collect()` a tick later calling a method on
        a socket that is already gone.
        """
        if not self.outbuf:
            return
        try:
            n = self.sock.send(self.outbuf)
        except OSError:
            return
        self.outbuf = self.outbuf[n:]

    def collect(self):
        """Drain whatever the socket has, answer by answer, without blocking.

        Reconnecting happens here, not in `decide()`: this is the one call
        the engine makes every tick regardless of whether any pair is
        resolving, so it is the natural place to notice a dead socket and
        retry it at most once before returning.
        """
        results = []

        if not self.connected and not self._never_connected:
            self.counters["reconnects"] += 1
            self._connect()

        if not self.connected:
            return results

        self._flush()

        # Read everything already sitting in the kernel buffer before
        # deciding anything is lost: a server can answer several requests
        # and then close the connection in the same burst, and the answers
        # already in that burst must be parsed, not written off along with
        # the ones that never came.
        closed = False
        while True:
            try:
                chunk = self.sock.recv(65536)
            except BlockingIOError:
                break
            except OSError:
                closed = True
                break
            if chunk == b"":
                closed = True
                break
            self.inbuf += chunk

        while b"\n" in self.inbuf:
            line, self.inbuf = self.inbuf.split(b"\n", 1)
            self._parse_line(line, results)

        if closed:
            self._handle_closed()

        return results

    def _parse_line(self, line, results):
        """Parse one answer or error line and append any answer to `results`.

        Every failure to make sense of the line (bad JSON, a missing key,
        a token of the wrong shape) is counted as `malformed` and the line
        is dropped; nothing here is allowed to raise, because one bad line
        from the server must not stop the rest of the batch from being
        read.
        """
        try:
            obj = json.loads(line)
        except ValueError:
            self.counters["malformed"] += 1
            return
        if not isinstance(obj, dict):
            self.counters["malformed"] += 1
            return

        try:
            neighbor, epoch = obj["token"]
            # The token must be rebuilt as a tuple (str, int): json.loads
            # turns a two-element array into a list, and the engine uses
            # the token as a dict key. Handing it a list raises
            # "TypeError: unhashable type", which kills the node.
            token = (str(neighbor), int(epoch))
        except (KeyError, TypeError, ValueError):
            self.counters["malformed"] += 1
            return

        if "error" in obj:
            self.counters["errors"] += 1
            self.outstanding.discard(token)
            return

        try:
            interval = obj["interval_s"]
            logits = obj["logits"]
        except KeyError:
            self.counters["malformed"] += 1
            return

        # `interval` is kept exactly as json.loads produced it: the engine
        # checks it against its configured choices by `in`, and rounding
        # or otherwise mapping it here would make an exact arm fail that
        # check for no reason visible anywhere in this file.
        #
        # `mem` is updated from every answer received, whether or not the
        # engine goes on to apply it. The alternative -- updating only on
        # an applied answer -- would make this client's send-time value
        # disagree with the row the log carries for that resolution.
        # Keyed by `token[0]`, not the raw `neighbor` above: the token
        # construction already turned it into the plain `str` `mem` is
        # keyed by, and re-using that avoids holding two differently-typed
        # ideas of "the same neighbour".
        self.mem[token[0]] = interval
        self.counters["answered"] += 1
        # A duplicate answer for a token already discarded is a harmless
        # second discard, not an error; it is also yielded a second time,
        # on purpose -- the engine is what recognises and counts a
        # duplicate, not this client.
        self.outstanding.discard(token)
        results.append((token, interval, {"logits": logits}))

    def _handle_closed(self):
        """Tear down a dead connection and account for everything it was holding.

        Both the requests already written to the socket and the ones
        still waiting in `outbuf` are counted `lost` here, because
        `outstanding` is populated when a request is enqueued, not when it
        is actually flushed: by the time a connection is found dead,
        either kind of pending request is equally unanswerable.
        """
        try:
            self.sock.close()
        except OSError:
            pass
        self.sock = None
        self.connected = False
        self.counters["lost"] += len(self.outstanding)
        self.outstanding.clear()
        self.outbuf = b""
        # A partial line left over from the dead connection belongs to a
        # byte stream that no longer exists; keeping it would splice it
        # onto whatever the next connection sends first.
        self.inbuf = b""

    def close(self):
        """Close the socket and write this run's counters to the stats file.

        Written at close only: a node that dies of an uncaught exception
        never reaches this call, so the one run whose counts would explain
        the failure is exactly the run that leaves no stats file behind.
        That is a known gap, not an oversight -- the engine's own log of
        decisions and resolutions is what has to explain that run instead.
        """
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
            self.connected = False
        self.counters["outstanding_at_close"] = len(self.outstanding)
        os.makedirs(self.stats_dir, exist_ok=True)
        path = os.path.join(self.stats_dir, "%s.lclient.json" % self.node)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.counters, fh)


def build(node, socket_path, send_start, stats_dir, record_dir=None):
    """Factory for the loader's `factory` key: build an `LClient`, optionally recorded.

    Importing `fakes` is deferred to this call and skipped entirely when
    `record_dir` is not given, so a real run never depends on a module
    that exists only to exercise this one offline.
    """
    client = LClient(node=node, socket_path=socket_path, send_start=send_start,
                      stats_dir=stats_dir)
    if record_dir is None:
        return client
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import fakes
    return fakes.wrap(client, node=node, record_dir=record_dir,
                       meta={"inner": "ldecider", "socket_path": socket_path,
                             "send_start": send_start})
