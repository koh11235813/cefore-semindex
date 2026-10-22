"""The one real Transport: drives a SyncEngine over a cefpyco handle.

Everything the core's `transport.Transport` protocol asks for (register,
send_interest) is implemented here exactly, plus everything a real network
needs that an in-memory one does not:

- An Interest-deadline table. tests/fakes.FakeNetwork can resolve a timeout
  the instant it knows no reply is coming (a partition, a holder that
  returned None); a real network cannot know that, so this module tracks
  each outstanding Interest's own lifetime and calls the engine's timeout
  path itself once that deadline passes with no Data.
- A guard around handle_interest. holder_state.encode can raise
  HolderStateError (too many origins to fit even with every extra
  truncated away, map.md "ページングは無し"), and a per-message failure
  like that must not crash the single-threaded loop that also serves every
  other neighbor -- it is treated the same as handle_interest returning
  None: the Interest is left to expire.
- A should_stop check both right here in poll(), immediately after its own
  receive() returns, and inside fetch_content's own wait loop. The agent
  loop's run() only checks its stop flag between step()s, but a single
  receive() can itself block up to receive_timeout_ms, and fetch_content
  can block inside a single step() for up to chunk_lifetime_ms per chunk
  waiting on content that never arrives (a partition, or T1's
  partition-phase probe by design) -- that chunk_lifetime_ms wait is
  already the entire SIGTERM-to-SIGKILL budget a ReCefore host_apps
  process gets. should_stop, when given, is polled right after every
  receive() (poll()'s own, and both before and after each one inside
  fetch_content's wait loop) and before every chunk's Interest, so a stop
  request -- a real SIGTERM, or the agent's own measurement-window
  stop_at_s deadline (semindex.agent.Agent._on_receive_boundary) -- lands
  within one receive_timeout_ms instead of up to chunk_lifetime_ms, and
  never leaves a packet/chunk that arrived past that point dispatched,
  counted or acted on.
- A payload-size check before every send_data. cefpyco's send_data does
  not check payload length itself (map.md); the engine already keeps every
  Response within schema.MAX_RECORD_BYTES / holder_state.MAX_ENCODED_BYTES
  by construction, so this is a last line of defense, not the only one.
- Byte counters. bytes_sent counts the UTF-8 length of every Interest name
  this node sends (sync Interests here, content-chunk Interests from
  fetch_content) plus name+payload for every Data this node sends as
  holder. bytes_received counts the name length of every incoming Interest
  this node serves plus name+payload for every Data this node receives as
  a reply (sync Data here, content chunks from fetch_content). An Interest
  Return is not specially counted or dispatched; it is left for the
  Interest's own deadline to resolve as an ordinary timeout.
- control_bytes_sent/control_bytes_received mirror the two counters above
  but exclude fetch_content entirely (send_interest and _serve only).
  2026-10-22 fix: analyze.py's control_bytes() used to read
  bytes_sent/bytes_received directly, so a probe's own content-body fetch
  (T1's own live run moved 433390 bytes this way) was reported as sync
  protocol overhead. These two give a reader of events.jsonl a total that
  is actually bounded by the sync protocol's own traffic, independent of
  how large a probe's retrieved body happened to be.
- A payload=None coercion on every incoming Data. cefpyco's own
  CcnPacketInfo.payload is None for a Data TLV with no payload at all
  (its own payload_s property guards the same way), and len(None) would
  crash this module's single-threaded loop exactly the way an unguarded
  handle_interest would -- every read of a received Data's payload treats
  a falsy payload as b"".

cefpyco's receive() holds the GIL and reads into one process-wide static
buffer (map.md), so exactly one call site may ever call it: poll() and
fetch_content() below are the only two, and fetch_content() routes every
packet that is not the chunk it is waiting for through the same dispatch
path poll() uses, so a probe's one-shot body fetch never drops a sync
Interest or Data reply that arrived while it was waiting.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol

from semindex import holder_state, names, schema
from semindex.engine import SyncEngine

# map.md "cefpyco の既知欠陥は使い方で避ける": the agent's single receive()
# call must poll with a short timeout rather than block, so a SIGTERM can
# always be noticed within about this many milliseconds.
DEFAULT_RECEIVE_TIMEOUT_MS = 100


class CefpycoPacket(Protocol):
    """The slice of cefpyco.CcnPacketInfo this module reads."""

    is_interest: bool
    is_data: bool
    name: str | None
    payload: bytes | None
    chunk_num: int | None
    end_chunk_num: int | None


class CefpycoHandle(Protocol):
    """The slice of cefpyco.CefpycoHandle this module calls.

    A test double only needs this shape (the same shape as cefpyco's own
    semindex_smoke/_FakeHandle test doubles), not a real cefnetd.
    """

    def register(self, name: str) -> None: ...

    def send_interest(self, name: str, chunk_num: int, lifetime: int) -> None: ...

    def send_data(self, name: str, payload: bytes, chunk_num: int, expiry: int, cache_time: int) -> None: ...

    def receive(self, timeout_ms: int) -> CefpycoPacket: ...


class CefpycoTransport:
    """Transport implementation SyncEngine runs against when backed by real Cefore."""

    def __init__(
        self,
        handle: CefpycoHandle,
        engine: SyncEngine | None = None,
        *,
        clock: Callable[[], float] = time.time,
        receive_timeout_ms: int = DEFAULT_RECEIVE_TIMEOUT_MS,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Wrap `handle`; `engine` is the SyncEngine this transport delivers replies, incoming
        Interests and timeouts into. `clock` is the single time source shared with the
        agent loop's own engine.on_tick(now) calls, so Interest-lifetime deadlines, the
        engine's poll/backoff scheduling and events.jsonl timestamps all agree on "now".

        `engine` defaults to None because SyncEngine.__init__ itself calls
        this transport's register() while constructing -- the engine
        cannot be handed to this constructor before it exists. register()
        never touches self._engine, only poll/fetch_content do, and those
        are only ever called once construction has finished, so the
        agent's wiring code builds the transport, builds the engine from
        it, then sets transport.engine = engine before the loop starts.

        `should_stop` has the same construction-order problem as `engine`
        (the Agent that owns the real stop flag is built from this
        transport, not the other way around) and the same fix: it
        defaults to a callable that always says "keep going", and the
        agent's wiring code replaces it with the real agent's boundary
        check (Agent._on_receive_boundary) once the agent itself exists.
        That check reports (and, if newly true, sets) whether a real stop
        request or the agent's own stop_at_s deadline has been reached; it
        also captures any counters_snapshot_at_s entry now due, but only
        ever queues it in memory for the agent's own step() to write out
        later -- never writes it from in here (see
        Agent._on_receive_boundary's own docstring for why a write from
        inside fetch_content's wait loop specifically would be wrong).
        poll() and fetch_content's chunk-wait loop both call it right after
        their own receive(), so neither can ever block a stop request --
        real or stop_at_s -- for longer than one receive_timeout_ms.
        """
        self._handle = handle
        self._engine = engine
        self._clock = clock
        self._receive_timeout_ms = receive_timeout_ms
        self._should_stop = should_stop if should_stop is not None else (lambda: False)
        self._deadlines: dict[str, float] = {}
        self.bytes_sent = 0
        self.bytes_received = 0
        self.control_bytes_sent = 0
        self.control_bytes_received = 0

    @property
    def engine(self) -> SyncEngine | None:
        """The SyncEngine this transport delivers replies, incoming Interests and timeouts into."""
        return self._engine

    @engine.setter
    def engine(self, value: SyncEngine) -> None:
        self._engine = value

    @property
    def should_stop(self) -> Callable[[], bool]:
        """The callable poll() and fetch_content both call right after their own
        receive() returns, to notice a stop (real, or the agent's own
        stop_at_s boundary) before acting on whatever that receive() just
        returned; see __init__."""
        return self._should_stop

    @should_stop.setter
    def should_stop(self, value: Callable[[], bool]) -> None:
        self._should_stop = value

    def now(self) -> float:
        """The transport's own clock reading, for the agent loop to pass to engine.on_tick."""
        return self._clock()

    # -- Transport protocol: the engine calls these directly ------------------

    def register(self, prefix: str) -> None:
        """Register `prefix` with cefnetd so Interests addressed to it reach this process."""
        self._handle.register(prefix)

    def send_interest(self, name: str, lifetime_ms: int) -> None:
        """Fire a sync-protocol Interest for `name` and start tracking its deadline.

        chunk_num=-1 sends no chunk TLV (map.md "同期用の Interest/Data に
        は chunk TLV を付けない") -- cefpyco's own default of chunk_num=0
        would add one, which is why this is passed explicitly rather than
        left to the default.
        """
        self._handle.send_interest(name, chunk_num=-1, lifetime=lifetime_ms)
        self.bytes_sent += len(name.encode("utf-8"))
        self.control_bytes_sent += len(name.encode("utf-8"))
        self._deadlines[name] = self._clock() + lifetime_ms / 1000.0

    # -- driving the real network from the agent loop --------------------------

    def poll(self) -> None:
        """Receive at most one packet (blocking up to receive_timeout_ms), then
        dispatch it and check Interest deadlines -- unless should_stop()
        reports true once that receive() call returns, in which case the
        packet is left unprocessed and deadlines unchecked for this call.

        This is the agent loop's one call per iteration into cefpyco's
        receive(); see the module docstring for why there can only be one
        call site besides fetch_content's.

        2026-10-23 fix: should_stop() used to be polled only inside
        fetch_content's own wait loop, never here -- a stop request (real
        SIGTERM, or the agent's own stop_at_s deadline) landing while this
        receive() was blocked for up to receive_timeout_ms went unnoticed
        until the *next* step()'s own check, by which point this call's
        _dispatch/_check_timeouts had already run against a `now` possibly
        past that deadline. Checking should_stop() right here, before
        either, means stop_at_s never leaks a send/tick from the step this
        poll() call belongs to (see semindex.agent.Agent.step()). One
        tradeoff: a sync Interest whose own deadline falls inside this same
        receive_timeout_ms window reports no fetch_timeout for this life --
        a SIGTERM-driven stop already accepted that (the loop exits right
        after), and stop_at_s now does too.
        """
        info = self._handle.receive(timeout_ms=self._receive_timeout_ms)
        if self._should_stop():
            return
        now = self._clock()
        self._dispatch(info, now)
        self._check_timeouts(now)

    def fetch_content(self, name: str, *, chunk_lifetime_ms: int, max_chunks: int = 100_000) -> tuple[bool, int]:
        """Fetch every chunk of a cefputfile-published object under `name`, chunk 0..end.

        Mirrors cefpyco's own semindex_smoke/chunk_fetch.py: one Interest
        per chunk_num, polling receive() for the matching Data, stopping at
        the chunk whose end_chunk_num equals its own chunk_num. Any packet
        received while waiting that is *not* that chunk's Data is handed to
        _dispatch the same way poll() would -- an incoming neighbor
        Interest or a reply to one of this node's own sync Interests must
        not be silently discarded just because a probe is mid-fetch.

        Returns (True, total_bytes) once every chunk through end_chunk_num
        has been collected, or (False, bytes_so_far) if any one chunk's
        Interest lifetime elapses with no matching Data, or if max_chunks
        is exhausted without ever seeing an end_chunk_num (a guard against
        spinning forever on an object published without one at all), or if
        should_stop() reports true at any point -- checked here (before
        sending the next chunk's Interest), before every receive() in the
        wait loop below, and again immediately after each receive()
        returns. That second, post-receive check matters as much as the
        first: a chunk's Data that only arrives once should_stop() has gone
        true (e.g. the agent's own stop_at_s boundary crossed while this
        receive() was blocked) is abandoned unprocessed and uncounted, not
        treated as a retrieved chunk -- the returned total_bytes is exactly
        what had already been collected strictly before that boundary. This
        keeps a stop noticed within one receive_timeout_ms rather than after
        up to chunk_lifetime_ms, same as poll()'s own should_stop() check.
        """
        total = 0
        chunk_num = 0
        while chunk_num < max_chunks:
            if self._should_stop():
                return False, total
            self._handle.send_interest(name, chunk_num=chunk_num, lifetime=chunk_lifetime_ms)
            self.bytes_sent += len(name.encode("utf-8"))
            deadline = self._clock() + chunk_lifetime_ms / 1000.0
            got: CefpycoPacket | None = None
            while self._clock() < deadline:
                if self._should_stop():
                    return False, total
                info = self._handle.receive(timeout_ms=self._receive_timeout_ms)
                # 2026-10-23 fix: re-check right here too, not only before this
                # receive() -- a matching chunk that only arrived because this
                # one receive() call blocked past the stop boundary must be
                # abandoned the same as a chunk that never arrived at all, not
                # counted into total just because it happened to be the packet
                # this receive() returned.
                if self._should_stop():
                    return False, total
                now = self._clock()
                if info.is_data and info.name == name and info.chunk_num == chunk_num:
                    got = info
                    break
                self._dispatch(info, now)
                self._check_timeouts(now)
            if got is None:
                return False, total
            payload = got.payload or b""
            self.bytes_received += len(name.encode("utf-8")) + len(payload)
            total += len(payload)
            if got.end_chunk_num is not None and got.end_chunk_num == chunk_num:
                return True, total
            chunk_num += 1
        return False, total

    # -- packet dispatch ---------------------------------------------------------

    def _dispatch(self, info: CefpycoPacket, now: float) -> None:
        """Route one received packet: serve an incoming Interest, or deliver a Data reply.

        Anything else -- a bare receive() timeout (nothing arrived) or an
        Interest Return -- is left alone; the deadline table is what
        eventually reports a missing reply, in _check_timeouts.
        """
        if info.is_interest:
            self._serve(info.name, now)
        elif info.is_data:
            # Pop first, then route unconditionally: Data that arrives
            # just after its own deadline fired is still a valid reply --
            # the tracker already released that fetch's in-flight slot on
            # timeout, so on_record/on_state handle a late arrival the
            # same as an on-time one.
            self._deadlines.pop(info.name, None)
            self._route_data(info.name, info.payload or b"", now)

    def _serve(self, name: str, now: float) -> None:
        """Answer an incoming Interest addressed to this node, as holder."""
        self.bytes_received += len(name.encode("utf-8"))
        # Every Interest _serve answers is sync protocol traffic (state or
        # record fetch from a neighbor) -- fetch_content's own content-chunk
        # Interests never reach this method, since they are a requester-side
        # call that this node's own dispatch loop never routes back into
        # _serve. See the module docstring's "Byte counters" entry.
        self.control_bytes_received += len(name.encode("utf-8"))
        try:
            response = self._engine.handle_interest(name)
        except Exception:
            # holder_state.HolderStateError (too many origins to fit even
            # truncated) is the known case, but any other per-message
            # failure here gets the same treatment: this one Interest is
            # left to expire rather than taking down the loop that also
            # serves every other neighbor.
            return
        if response is None:
            return
        # handle_interest already parsed `name` successfully to produce a
        # Response, so this parse cannot fail; it only picks which limit
        # applies.
        parsed = names.parse(name)
        limit = holder_state.MAX_ENCODED_BYTES if isinstance(parsed, names.StateName) else schema.MAX_RECORD_BYTES
        if len(response.payload) > limit:
            # Belt-and-suspenders: the engine already keeps every Response
            # within these limits by construction (map.md calls out that
            # cefpyco's send_data itself does not check payload length).
            # If that guarantee were ever broken upstream, dropping the
            # reply here is the same safe outcome as handle_interest
            # returning None.
            return
        self._handle.send_data(
            name,
            response.payload,
            chunk_num=-1,
            expiry=round(response.expiry_s * 1000),
            cache_time=round(response.cache_time_s * 1000),
        )
        self.bytes_sent += len(name.encode("utf-8")) + len(response.payload)
        self.control_bytes_sent += len(name.encode("utf-8")) + len(response.payload)

    def _route_data(self, name: str, payload: bytes, now: float) -> None:
        """Deliver a Data reply to the engine as a state or record reply, by the name's own kind."""
        self.bytes_received += len(name.encode("utf-8")) + len(payload)
        try:
            parsed = names.parse(name)
        except names.MalformedNameError:
            return  # this node never sent an Interest for a name it cannot parse
        # Only a sync name (state or record) ever reaches this point: a
        # stray content-chunk Data that fetch_content's own wait loop did
        # not consume (e.g. it arrived after that fetch already gave up)
        # fails the parse above and returns before this line, so it is
        # never misclassified as control traffic either.
        self.control_bytes_received += len(name.encode("utf-8")) + len(payload)
        if isinstance(parsed, names.StateName):
            self._engine.on_state(parsed.holder, payload, now)
        else:
            self._engine.on_record(name, payload, now)

    def _check_timeouts(self, now: float) -> None:
        """Report every Interest whose deadline has passed with no Data, then stop tracking it."""
        due = [name for name, deadline in self._deadlines.items() if deadline <= now]
        for name in due:
            del self._deadlines[name]
            self._engine.on_fetch_timeout(name, now)
