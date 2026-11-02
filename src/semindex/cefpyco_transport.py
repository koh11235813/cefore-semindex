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
- A should_stop check in poll(), immediately after its own receive()
  returns. The agent loop's run() only checks its stop flag between
  step()s, but a single receive() can itself block up to
  receive_timeout_ms. should_stop, when given, is polled right after
  every receive(), so a stop request -- a real SIGTERM, or the agent's
  own measurement-window stop_at_s deadline
  (semindex.agent.Agent._on_receive_boundary) -- never leaves a
  packet/chunk that arrived past that point dispatched, counted or acted
  on.
- A non-blocking content fetch (start_fetch / take_fetch_result /
  abort_fetch). 2026-11-02 fix: a probe's body used to be fetched by a
  blocking fetch_content() that waited inside one agent step for every
  chunk (up to chunk_lifetime_ms each), so the agent's engine.on_tick did
  not run meanwhile and h2's state polls went out ~1 s late for 4-9% of
  them (docs/wayfinder/semindex-stage2/evidence/07/README.md section 5).
  The fetch is now a chunk-by-chunk state machine that poll() advances:
  the chunk it waits for is consumed inside poll()'s normal receive path,
  the next chunk's Interest is sent from there, and its per-chunk deadline
  is judged there, so the loop keeps stepping while a body is in flight.
  At most one fetch exists at a time: every eval probe body has the same
  name, so two concurrent fetches' chunks could not be told apart. A chunk
  is never retransmitted (Cefore v0.12's PIT absorbs a same-name retry
  within lifetime + 1 s), and its deadline is kept out of the sync
  Interest deadline table, which would report it to the engine as a
  record fetch timeout.
- A payload-size check before every send_data. cefpyco's send_data does
  not check payload length itself (map.md); the engine already keeps every
  Response within schema.MAX_RECORD_BYTES / holder_state.MAX_ENCODED_BYTES
  by construction, so this is a last line of defense, not the only one.
- Byte counters. bytes_sent counts the UTF-8 length of every Interest name
  this node sends (sync Interests and content-chunk Interests) plus
  name+payload for every Data this node sends as holder. bytes_received
  counts the name length of every incoming Interest this node serves plus
  name+payload for every Data this node receives as a reply (sync Data and
  content chunks, each counted once). An Interest
  Return is not specially counted or dispatched; it is left for the
  Interest's own deadline to resolve as an ordinary timeout.
- control_bytes_sent/control_bytes_received mirror the two counters above
  but exclude the content fetch entirely (send_interest, _serve and
  _route_data's sync names only).
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
buffer (map.md; /home/lab_shared/cefpyco/src/cefpyco_c/cefpyco.c and
cefpyco_parse.c keep the handle, receive buffer and parse position as
statics), so exactly one call site may ever call it, from one thread:
poll() below. That is why the content fetch is driven by poll() rather
than by a thread or a second handle of its own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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


@dataclass
class _ContentFetch:
    """The one in-flight content fetch: which chunk it waits for, until when, and what it has so far.

    `result` stays None while the fetch is still waiting; once set, the
    fetch is finished and only waits for the agent to take the result
    (CefpycoTransport.take_fetch_result), so a finished fetch is never
    advanced or judged again.
    """

    name: str
    chunk_lifetime_ms: int
    max_chunks: int
    chunk_num: int = 0
    deadline: float = 0.0
    total: int = 0
    result: tuple[bool, int] | None = None


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
        never touches self._engine, only poll does, and poll is only
        ever called once construction has finished, so the
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
        Agent._on_receive_boundary's own docstring). poll() calls it right
        after its own receive(), so a stop request -- real or stop_at_s --
        is never acted past for longer than one receive_timeout_ms.
        """
        self._handle = handle
        self._engine = engine
        self._clock = clock
        self._receive_timeout_ms = receive_timeout_ms
        self._should_stop = should_stop if should_stop is not None else (lambda: False)
        self._deadlines: dict[str, float] = {}
        self._fetch: _ContentFetch | None = None
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
        """The callable poll() calls right after its own receive() returns, to
        notice a stop (real, or the agent's own stop_at_s boundary) before
        acting on whatever that receive() just returned; see __init__."""
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
        dispatch it, check Interest deadlines and advance the in-flight
        content fetch -- unless should_stop() reports true once that
        receive() call returns, in which case the packet is left
        unprocessed and nothing is checked or advanced for this call.

        This is the agent loop's one call per iteration into cefpyco's
        receive(); see the module docstring for why there can only be one
        call site.

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
        after), and stop_at_s now does too. The same holds for a content
        chunk: one that only arrives on a receive() past the stop boundary
        is never counted, and the fetch it belongs to is settled as a
        failure by abort_fetch() at shutdown.

        The fetch's chunk is consumed before its deadline is judged, so a
        chunk returned by the very receive() whose wait reaches the deadline
        still counts -- the rule the old blocking fetch_content had too
        (it judged the deadline only before each receive()).
        """
        info = self._handle.receive(timeout_ms=self._receive_timeout_ms)
        if self._should_stop():
            return
        now = self._clock()
        if not self._consume_fetch_chunk(info):
            self._dispatch(info, now)
        self._check_timeouts(now)
        self._check_fetch_deadline(now)

    # -- the one content fetch, advanced by poll() ------------------------------

    def start_fetch(self, name: str, *, chunk_lifetime_ms: int, max_chunks: int = 100_000) -> None:
        """Start fetching every chunk of a cefputfile-published object under `name`, chunk 0..end.

        Sends only chunk 0's Interest and returns; poll() then consumes
        each chunk's Data, sends the next chunk's Interest and judges each
        chunk's own deadline (sent-at + chunk_lifetime_ms). The fetch
        finishes, mirroring cefpyco's own semindex_smoke/chunk_fetch.py and
        the old blocking fetch_content exactly:

        - (True, total_bytes) on the chunk whose end_chunk_num equals its
          own chunk_num;
        - (False, bytes_so_far) once a chunk's deadline passes with no
          matching Data (no retransmit, see the module docstring), once
          max_chunks chunks have been consumed without an end_chunk_num (a
          guard against spinning forever on an object published without
          one; max_chunks must be >= 1), or once abort_fetch() ends it.

        Raises RuntimeError if a previous fetch has not been settled yet
        (still in flight, or finished but its result not taken): there is
        at most one fetch at a time (module docstring).
        """
        if self._fetch is not None:
            raise RuntimeError("a content fetch is still unsettled")
        self._fetch = _ContentFetch(name=name, chunk_lifetime_ms=chunk_lifetime_ms, max_chunks=max_chunks)
        self._send_chunk_interest(self._fetch)

    def fetch_in_flight(self) -> bool:
        """Whether a fetch has been started and is still waiting for a chunk."""
        return self._fetch is not None and self._fetch.result is None

    def take_fetch_result(self) -> tuple[bool, int] | None:
        """The finished fetch's (success, total_bytes), handed out once; None while
        idle or still waiting. Taking it frees the slot for the next start_fetch."""
        if self._fetch is None or self._fetch.result is None:
            return None
        result = self._fetch.result
        self._fetch = None
        return result

    def abort_fetch(self) -> None:
        """End a still-waiting fetch as (False, bytes collected so far), for the
        agent's stop path; a fetch that already finished keeps its own result,
        and with no fetch this does nothing."""
        if self.fetch_in_flight():
            self._fetch.result = (False, self._fetch.total)

    def _send_chunk_interest(self, fetch: _ContentFetch) -> None:
        """Send the Interest for `fetch`'s current chunk and start that chunk's deadline.

        The deadline lives on the fetch itself, never in self._deadlines:
        _check_timeouts would hand it to engine.on_fetch_timeout, which
        reads it as a sync record fetch that timed out. Only bytes_sent
        moves, never control_bytes_sent (module docstring "Byte counters").
        """
        self._handle.send_interest(fetch.name, chunk_num=fetch.chunk_num, lifetime=fetch.chunk_lifetime_ms)
        self.bytes_sent += len(fetch.name.encode("utf-8"))
        fetch.deadline = self._clock() + fetch.chunk_lifetime_ms / 1000.0

    def _consume_fetch_chunk(self, info: CefpycoPacket) -> bool:
        """Take `info` as the in-flight fetch's awaited chunk if it is one; True if it was.

        Intercepted here, ahead of _dispatch, so the chunk's bytes are
        counted exactly once and only into bytes_received: _route_data
        would count them a second time. Any other content Data (a chunk
        number this fetch is not waiting for, or one arriving after the
        fetch already gave up) still goes through _dispatch, which counts
        it in bytes_received and, failing names.parse, never in the control
        counters -- unchanged from before.
        """
        fetch = self._fetch
        if fetch is None or fetch.result is not None:
            return False
        if not (info.is_data and info.name == fetch.name and info.chunk_num == fetch.chunk_num):
            return False
        payload = info.payload or b""
        self.bytes_received += len(fetch.name.encode("utf-8")) + len(payload)
        fetch.total += len(payload)
        if info.end_chunk_num is not None and info.end_chunk_num == fetch.chunk_num:
            fetch.result = (True, fetch.total)
            return True
        fetch.chunk_num += 1
        if fetch.chunk_num >= fetch.max_chunks:
            fetch.result = (False, fetch.total)
        else:
            self._send_chunk_interest(fetch)
        return True

    def _check_fetch_deadline(self, now: float) -> None:
        """Fail a still-waiting fetch whose current chunk's deadline has been reached.

        `>=`, matching the old fetch_content's `while clock < deadline` wait.
        """
        fetch = self._fetch
        if fetch is not None and fetch.result is None and now >= fetch.deadline:
            fetch.result = (False, fetch.total)

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
        # record fetch from a neighbor) -- the content fetch's own chunk
        # Interests never reach this method, since they are a requester-side
        # send that this node's own dispatch loop never routes back into
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
        # stray content-chunk Data that _consume_fetch_chunk did not take
        # (e.g. it arrived after that fetch already gave up)
        # fails the parse above and returns before this line, so it is
        # never misclassified as control traffic either.
        self.control_bytes_received += len(name.encode("utf-8")) + len(payload)
        if isinstance(parsed, names.StateName):
            self._engine.on_state(parsed.holder, payload, now, poll_epoch_ms=parsed.epoch_ms)
        else:
            self._engine.on_record(name, payload, now)

    def _check_timeouts(self, now: float) -> None:
        """Report every Interest whose deadline has passed with no Data, then stop tracking it."""
        due = [name for name, deadline in self._deadlines.items() if deadline <= now]
        for name in due:
            del self._deadlines[name]
            self._engine.on_fetch_timeout(name, now)
