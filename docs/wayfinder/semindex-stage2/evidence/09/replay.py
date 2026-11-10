"""Deterministic replay of poll policies over the logged link schedules (a model, not a measurement).

A poll sent at t is answered iff the link is up at t; in the first 0.5 s after a link_up it fails with
UP_RESIDUAL (measured 2.4-3.9 %, physics.py). A record at m is learned by the first answered poll at or
after its arrival, then stored FETCH_S later; the next hop's arrival is that stored time.

Two information modes for outcome-dependent rules:
  ideal   the outcome of the poll just sent is known when choosing the next interval (needs an engine change)
  engine  semindex engine as it stands: the interval is drawn at send time, from the status of earlier polls,
          and an unanswered poll only counts as timed out 2 s after it was sent
"""
import bisect
import glob
import json
import math
import random

STORE = "/home/lab_shared/semindex-eval"
ARMS = (0.5, 1.359, 3.695, 10.043)
JIT = 0.1
OVERHEAD_S = 0.05   # measured gap - x at p=0: 0.051, 0.052, 0.049, 0.037-0.057 s
RTT_S = 0.0012      # post-no-fail in seq_model.md
FETCH_S = 0.005
UP_RESIDUAL = 0.027
WS, WE = 10.0, 130.0
INT_B = 50.9        # s4a state-request name bytes (B stages differ only by the stage name's length)
REP_B = {0.0: 132.5, 0.1: 131.7, 0.2: 130.4, 0.3: 128.2, 0.4: 125.5}  # s4a reply bytes per answered poll
FLOWS = [("h1", "h0", 1, 0), ("h2", "h1", 2, 1), ("h3", "h2", 3, 2), ("h4", "h3", 4, 3),
         ("h3", "h4", 1, 3), ("h2", "h3", 2, 2), ("h1", "h2", 3, 1), ("h0", "h1", 4, 0)]  # n, m, hop, link


def load_schedule(p, seed, stage="s4a"):
    """(down intervals per link 0..3, record times per origin) of one logged run."""
    f = glob.glob(f"{STORE}/{stage}/p{p:g}/{seed}/*.experiment.json")[0]
    d = json.load(open(f))
    links = [[] for _ in range(4)]
    cur = {}
    for e in sorted((e for e in d["events"] if e["type"] in ("link_down", "link_up")), key=lambda e: e["at"]):
        k = min(e["nodes"])
        if e["type"] == "link_down":
            cur[k] = e["at"]
        else:
            links[k].append((cur.pop(k), e["at"]))
    for k, a in cur.items():
        links[k].append((a, 1e9))
    rec = {}
    for host in ("0", "4"):
        rec[host] = sorted({x["at_s"] for x in d["schedule"][host]})
    return links, rec


class Link:
    """Fast is-up query over sorted down intervals."""

    def __init__(self, ivs):
        self.a = [x[0] for x in ivs]
        self.b = [x[1] for x in ivs]

    def state(self, t):
        """(up, seconds since the last up transition, or 1e9)."""
        i = bisect.bisect_right(self.a, t) - 1
        if i >= 0 and t < self.b[i]:
            return False, 0.0
        return True, (t - self.b[i]) if i >= 0 else 1e9

    def next_up(self, t):
        """Earliest time >= t at which the link is up."""
        i = bisect.bisect_right(self.a, t) - 1
        return self.b[i] if i >= 0 and t < self.b[i] else t


def run_flow(link, policy, rng, p, stop=WE):
    """Poll sequence of one directed pair; returns (answered times, n polls in window, n answered in window)."""
    t = rng.uniform(0, 0.5)
    ans, npoll, nans = [], 0, 0
    st = policy.start(rng)
    while t <= stop:
        up, su = link.state(t)
        ok = up and link.state(t + RTT_S)[0] and not (su < 0.5 and rng.random() < UP_RESIDUAL)
        if ok:
            ans.append(t)
        if t >= WS:
            npoll += 1
            nans += ok
        t = policy.next_send(t, ok, st, rng, link)
        t += OVERHEAD_S + 0.035 * p
    return ans, npoll, nans


def jitter(x, rng):
    """Applied interval for chosen x (agent's +/-10 % choice jitter)."""
    return x * (1 + JIT * (2 * rng.random() - 1))


class Rule:
    """Policy base: state is the outcome history kept per flow."""
    ideal = True

    def start(self, rng):
        """Fresh per-flow state."""
        return dict(cto=0, sent=[], tfail=None, tlast=None)

    def view(self, t, ok, st):
        """(consecutive failures visible when choosing, age since last visible success) per information mode."""
        if self.ideal:
            st["cto"] = 0 if ok else st["cto"] + 1
            if ok:
                st["tlast"] = t
            return st["cto"], (t - st["tlast"]) if st["tlast"] is not None else 0.0
        # engine: outcomes older than 2 s (timeouts) or of answered polls (RTT ~ 1 ms) are visible; this poll's is not.
        sent = st["sent"]
        c = 0
        last_ans = None
        for ts, o in reversed(sent):
            if o:
                last_ans = ts
                break
            if ts + 2.0 <= t:
                c += 1
        sent.append((t, ok))
        if len(sent) > 64:
            del sent[:32]
        return c, (t - last_ans) if last_ans is not None else 0.0

    def next_send(self, t, ok, st, rng, link):
        """Send time of the next poll (before tick overhead)."""
        return t + jitter(self.interval(*self.view(t, ok, st)), rng)


class Fixed(Rule):
    """Always x."""

    def __init__(self, x):
        self.x = x

    def interval(self, c, age):
        return self.x


class Table(Rule):
    """interval = tab[min(cto, len-1)]; tab[0] is the interval after a success."""

    def __init__(self, tab, ideal=True):
        self.tab, self.ideal = tuple(tab), ideal

    def interval(self, c, age):
        return self.tab[min(c, len(self.tab) - 1)]


class Backoff(Rule):
    """b * 2^cto capped."""

    def __init__(self, b, cap=10.043, ideal=True):
        self.b, self.cap, self.ideal = b, cap, ideal

    def interval(self, c, age):
        return min(self.b * 2 ** c, self.cap)


class SlotClock(Rule):
    """After a failure, retry just past the next slot boundary (10 + 5k): knows slot length and phase."""

    def __init__(self, b, eps=0.05):
        self.b, self.eps = b, eps

    def next_send(self, t, ok, st, rng, link):
        if ok:
            return t + jitter(self.b, rng)
        return 10 + 5 * (math.floor((t - 10) / 5) + 1) + self.eps - OVERHEAD_S


class SlotBlind(Rule):
    """After a failure, wait one slot length; knows 5 s but not the phase."""

    def __init__(self, b, wait=5.0):
        self.b, self.wait = b, wait

    def next_send(self, t, ok, st, rng, link):
        return t + (jitter(self.b, rng) if ok else self.wait)


class Oracle(Rule):
    """Bound no real rule can reach: while down, the next poll lands exactly on the link_up, never wasted."""

    def __init__(self, b):
        self.b = b

    def next_send(self, t, ok, st, rng, link):
        nx = t + jitter(self.b, rng)
        up_at = link.next_up(nx)
        if up_at > nx:
            return up_at + 0.001 - OVERHEAD_S
        # next_up looks at the candidate only; also skip a poll that would fall in a later outage
        return nx


class DP(Rule):
    """Tabulated action(failure state, age bin) from dp.py; `elapsed` tables index the state by age bin."""

    def __init__(self, act, states, dage, ideal=True, elapsed=False):
        self.act, self.states, self.dage, self.ideal, self.elapsed = act, states, dage, ideal, elapsed

    def interval(self, c, age):
        if self.elapsed:
            s = 0 if c == 0 else 1 + sum(age >= b for b in (2.0, 5.0, 10.0, 20.0))
        else:
            s = min(c, self.states - 1)
        row = self.act[s]
        return row[min(int(age / self.dage), len(row) - 1)]


def simulate(policy, links, rec, p, rng):
    """One run. Returns (4-hop mean delay s, bytes per record-hop, wasted-poll share, hops-mean per-hop waits)."""
    answered = {}
    npoll = nans = 0
    links = [Link(iv) if not isinstance(iv, Link) else iv for iv in links]
    for n, m, hop, lk in FLOWS:
        a, np_, na = run_flow(links[lk], policy, rng, p)
        answered[(n, m)] = a
        npoll += np_
        nans += na
    lat, cnt = 0.0, 0
    nrh = 0
    for origin, path in (("0", ("h0", "h1", "h2", "h3", "h4")), ("4", ("h4", "h3", "h2", "h1", "h0"))):
        for tc in rec[origin]:
            arr = tc + 0.09
            ok = True
            for h in range(1, 5):
                A = answered[(path[h], path[h - 1])]
                k = bisect.bisect_left(A, arr)
                if k >= len(A) or A[k] + RTT_S + FETCH_S > WE:
                    ok = False
                    break
                arr = A[k] + RTT_S + FETCH_S
                nrh += 1
            if ok:
                lat += arr - (tc + 0.09)
                cnt += 1
    B = npoll * INT_B + nans * REP_B[p]
    # k = 4 records per generation time; the replay walks each time once, so a hop stands for 4 record-hops
    return (lat / cnt if cnt else float("nan")), B / max(4 * nrh, 1), 1 - nans / max(npoll, 1), cnt, npoll
