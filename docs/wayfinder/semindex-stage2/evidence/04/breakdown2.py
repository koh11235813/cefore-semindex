"""Stage-2 per-hop delay / control-byte breakdown, adapted from stage-1's
evidence/s1-breakdown/breakdown.py (read-only over any stage's store).

Changes from the stage-1 copy, and why:
- No sys.path.insert of /home/lab_shared/cefore-semindex/src: that put the
  main checkout ahead of PYTHONPATH, so a branch's semindex would silently lose.
- p, seed, k, interval choices come from the run's meta.json and the agent
  configs from the one `<run_dir>/<name>/h0.json` subdirectory, instead of
  reconstructing `s1_p{p}_seed{seed}` -- stage-2 names use p000/p040.
- Window bounds come from semindex.evalgen so they cannot drift from evalagg.
- `analyze` returns the run's parsed events/configs so the caller can run its
  own per-run passes without re-reading events.jsonl (k=8 runs are large).
The H1 decomposition and H2 byte reconstruction bodies are unchanged.
"""
import bisect, csv, datetime, glob, json, os, statistics, sys
from collections import defaultdict

from semindex import evalagg, evalgen, holder_state, names, schema  # noqa: E402
from semindex.store import OriginProgress  # noqa: E402

WS, WE = evalgen.WINDOW_START_S * 1000.0, evalgen.WINDOW_END_S * 1000.0


def run_name_dir(run_dir):
    """The one `<run_dir>/<name>/` holding the agent configs (hN.json); raises if not exactly one."""
    hits = [os.path.dirname(p) for p in glob.glob(f"{run_dir}/*/h0.json")]
    if len(hits) != 1:
        raise RuntimeError(f"{run_dir}: expected one <name>/h0.json, found {hits}")
    return hits[0]


def iso_ms(s):
    """Epoch milliseconds of an ISO-8601 results.json timestamp."""
    return datetime.datetime.fromisoformat(s).timestamp() * 1000.0


def load_events(run_dir):
    """{node: [event dict, ...]} from every node's events.jsonl of one run."""
    nodes = {}
    for d in sorted(glob.glob(f"{run_dir}/debug/node_dirs/h*")):
        n = os.path.basename(d)
        with open(f"{d}/semindex/events.jsonl") as f:
            nodes[n] = [json.loads(l) for l in f if l.strip()]
    return nodes


def link_intervals(results):
    """{frozenset(hA,hB): [(down_ms, up_ms), ...]} from results.json ts (stamped after the handler)."""
    ev = defaultdict(list)
    for r in results:
        if r.get("op_type") == "event" and r.get("event_type") in ("link_down", "link_up"):
            a, b = r["event"]["nodes"]
            ev[frozenset((f"h{a}", f"h{b}"))].append((iso_ms(r["ts"]), r["event_type"]))
    out = {}
    for k, lst in ev.items():
        lst.sort()
        ivs, down_at = [], None
        for ts, t in lst:
            if t == "link_down" and down_at is None:
                down_at = ts
            elif t == "link_up" and down_at is not None:
                ivs.append((down_at, ts))
                down_at = None
        if down_at is not None:
            ivs.append((down_at, float("inf")))
        out[k] = ivs
    return out


def overlap(ivs, a, b):
    """Milliseconds of [a, b] covered by the down intervals `ivs`."""
    if b <= a:
        return 0.0
    return sum(max(0.0, min(b, e) - max(a, s)) for s, e in ivs)


def last_up_in(ivs, a, b):
    """Latest link-up instant in (a, b], or None."""
    ups = [e for s, e in ivs if a < e <= b]
    return max(ups) if ups else None


def is_down(ivs, t):
    """Whether instant t falls inside one of the down intervals `ivs`."""
    return any(s <= t < e for s, e in ivs)


def analyze(run_dir, del_rows, byte_rows, anomalies, stage=""):
    """Append one run's per-delivery delay rows and per-node byte rows; count oddities in `anomalies`.

    Returns None for a flagged run (counted in anomalies), else a dict with the
    parsed `nodes` events, `cfgs`, `meta`, `t0`, `ws`, `we`, `polls` so callers
    can reuse them.
    """
    meta = json.load(open(f"{run_dir}/meta.json"))
    p, seed = meta["p"], meta["seed"]
    if not meta["validity"]["valid"]:
        anomalies["invalid_runs"] += 1
        return None
    nodes = load_events(run_dir)
    results = json.load(open(f"{run_dir}/results.json"))
    starts = [e for evs in nodes.values() for e in evs if e["event"] == "agent_start"]
    if len(starts) != len(nodes):
        anomalies["multi_life_runs"] += 1
    t0 = starts[0]["t0_ms"]
    ws, we = t0 + WS, t0 + WE
    links = link_intervals(results)
    for r in results:
        if r.get("op_type") == "event" and r.get("actual_at") is not None:
            anomalies["link_ts_lag_ms"].append(iso_ms(r["ts"]) - (t0 + r["actual_at"] * 1000))
    name_dir = run_name_dir(run_dir)
    cfgs = {n: json.load(open(f"{name_dir}/{n}.json")) for n in nodes}
    group = cfgs["h0"]["group"]

    created, stored = {}, {}
    polls = defaultdict(list)       # (node, nbr) -> sorted ts
    srx = defaultdict(list)         # (node, nbr) -> [(ts, hi)]
    first_plan = {}                 # (node, origin, seq) -> (ts, nbr) first fetch_plan containing key
    timeouts = defaultdict(int)
    for n, evs in nodes.items():
        for e in evs:
            k = e["event"]
            if k == "record_created":
                created[(e["origin"], e["seq"])] = e["ts_ms"]
            elif k == "record_stored":
                key = (e["origin"], e["seq"], n)
                if key not in stored or e["ts_ms"] < stored[key][0]:
                    stored[key] = (e["ts_ms"], e["holder"])
            elif k == "decision" and e["kind"] == "state_poll":
                polls[(n, e["action"]["neighbor"])].append(e["ts_ms"])
            elif k == "decision" and e["kind"] == "fetch_plan":
                for o, s in e["action"]["fetched"]:
                    first_plan.setdefault((n, o, s), (e["ts_ms"], e["action"]["neighbor"]))
            elif k == "state_rx":
                srx[(n, e["neighbor"])].append((e["ts_ms"], e["hi"]))
            elif k == "fetch_timeout":
                timeouts[(e["origin"], e["seq"], n)] += 1
    for v in polls.values():
        v.sort()

    # ---- H1: per-hop decomposition -------------------------------------------------
    for (o, s, n), (t_down, holder) in stored.items():
        c = created.get((o, s))
        if c is None or not (ws <= c <= we) or t_down > we:
            continue
        t_up = c if holder == o else stored.get((o, s, holder), (None,))[0]
        if t_up is None:
            anomalies["no_upstream_store"] += 1
            continue
        known = next((ts for ts, hi in srx[(n, holder)] if hi.get(o, 0) >= s), None)
        fp = first_plan.get((n, o, s))
        if known is None or (fp and fp[0] < known):
            # seq advertised only via `extra` (not logged) -- fall back to the reply
            # immediately preceding the first fetch_plan for the key.
            anomalies["known_via_plan_fallback"] += 1
            if fp is None:
                anomalies["no_plan"] += 1
                continue
            cand = [ts for ts, _ in srx[(n, holder)] if ts <= fp[0]]
            known = cand[-1] if cand else fp[0]
        pl = polls[(n, holder)]
        i = bisect.bisect_right(pl, known) - 1
        t_poll = pl[i] if i >= 0 else known
        if t_poll < t_up:
            anomalies["poll_before_upstream_store"] += 1
        ivs = links.get(frozenset((n, holder)), [])
        a = t_up
        b = max(t_up, t_poll)
        down_pre = overlap(ivs, a, b)
        lu = last_up_in(ivs, a, b)
        # 2026-10-26 fix: results.json stamps link_up a few ms after the handler,
        # so a link can still read as down at t_poll with its up-transition just
        # past b. Taking b - lu or b - a whole then counted that down time twice
        # (once in down_pre) and left pre_other negative (43 deliveries, down to
        # -14.8 s). Every wait component now counts only up-time.
        if lu is not None:
            poll_wait_after_up = (b - lu) - overlap(ivs, lu, b)
            poll_wait_noflap = 0.0
        elif down_pre == 0.0:
            poll_wait_after_up = 0.0
            poll_wait_noflap = b - a  # link up throughout [t_up, t_poll]: plain poll-phase wait
        else:
            # down at t_poll with no up-transition inside: the up-time before
            # that outage is neither a post-flap nor a plain wait.
            poll_wait_after_up = 0.0
            poll_wait_noflap = 0.0
        pre_other = (b - a) - down_pre - poll_wait_after_up - poll_wait_noflap
        if min(down_pre, poll_wait_after_up, poll_wait_noflap, pre_other) < -1e-6:
            anomalies["negative_component"] += 1
        poll_rtt = max(0.0, known - b)
        post = t_down - max(known, b)
        down_post = overlap(ivs, max(known, b), t_down)
        total = t_down - t_up
        del_rows.append(dict(
            stage=stage, p=p, seed=seed, origin=o, seq=s, node=n, holder=holder,
            hop=abs(int(n[1:]) - int(o[1:])), t_up=t_up, t_poll=t_poll, t_known=known, t_down=t_down,
            down_at_up=int(is_down(ivs, t_up)), total=total,
            down_pre=down_pre, poll_wait_after_up=poll_wait_after_up, poll_wait_noflap=poll_wait_noflap,
            pre_other_up=pre_other, poll_rtt=poll_rtt, fetch_down=down_post, fetch_up=post - down_post,
            n_fetch_timeout=timeouts[(o, s, n)],
        ))

    # ---- H2: per-type control bytes, reconstructed --------------------------------
    rec_bytes = {}
    for n, cfg in cfgs.items():
        for idx, ent in enumerate(cfg["schedule"]):
            key = (n, idx + 1)
            if key in created:
                rec = schema.IndexRecord(origin=n, seq=idx + 1, content_name=ent["content_name"], zone=ent["zone"],
                                         disaster=ent["disaster"], media=ent["media"], observed_at=created[key])
                rec_bytes[key] = len(schema.canonical_bytes(rec))
    held_tl = defaultdict(list)  # holder -> [(ts, origin, seq)]
    for (o, s), ts in created.items():
        held_tl[o].append((ts, o, s))
    for (o, s, n), (ts, _) in stored.items():
        held_tl[n].append((ts, o, s))
    for v in held_tl.values():
        v.sort()

    def holder_payload_len(h, t):
        """Encoded state-reply payload length holder h would have served at time t."""
        held = defaultdict(set)
        for ts, o, s in held_tl[h]:
            if ts > t:
                break
            held[o].add(s)
        prog = {}
        for o, ss in held.items():
            hi = 0
            while hi + 1 in ss:
                hi += 1
            prog[o] = OriginProgress(hi, tuple(sorted(x for x in ss if x > hi)))
        return len(holder_state.encode(prog)), {o: pr.hi for o, pr in prog.items()}, any(pr.extra for pr in prog.values())

    win = lambda t: ws <= t <= we  # noqa: E731
    per = {n: defaultdict(float) for n in nodes}
    for n, evs in nodes.items():
        for e in evs:
            k, t = e["event"], e["ts_ms"]
            if not win(t):
                continue
            if k == "decision" and e["kind"] == "state_poll":
                nb = e["action"]["neighbor"]
                per[n]["state_req_B"] += len(names.build_state_name(group, nb, n, t))
                per[n]["state_req_n"] += 1
            elif k == "decision" and e["kind"] == "fetch_plan":
                nb = e["action"]["neighbor"]
                for o, s in e["action"]["fetched"]:
                    per[n]["fetch_int_B"] += len(names.build_record_name(group, nb, o, s))
                    per[n]["fetch_int_n"] += 1
            elif k == "state_rx":
                nb = e["neighbor"]
                # the matching request name: the latest poll toward nb at or before this reply
                pl = polls[(n, nb)]
                i = bisect.bisect_right(pl, t) - 1
                nm = len(names.build_state_name(group, nb, n, pl[i] if i >= 0 else t))
                plen, hi_rec, had_extra = holder_payload_len(nb, t)
                if {o: h for o, h in hi_rec.items()} != e["hi"]:
                    anomalies["state_hi_mismatch"] += 1
                if had_extra:
                    anomalies["state_reply_with_extra"] += 1
                anomalies["state_rx_total"] += 1
                # bytes the holder sent == bytes this requester received for this Data
                per[nb]["state_rep_B"] += nm + plen
                per[nb]["state_rep_n"] += 1
                per[n]["state_rep_rx_B"] += nm + plen
            elif k == "record_stored":
                nm = len(names.build_record_name(group, e["holder"], e["origin"], e["seq"]))
                b = nm + rec_bytes[(e["origin"], e["seq"])]
                per[e["holder"]]["fetch_data_B"] += b
                per[e["holder"]]["fetch_data_n"] += 1
                per[n]["fetch_data_rx_B"] += b
    sent_w, recv_w = evalagg._window_control_bytes(nodes, ws, we)
    n_created = sum(1 for t in created.values() if win(t))
    for n in nodes:
        d = per[n]
        recon_sent = d["state_req_B"] + d["fetch_int_B"] + d["state_rep_B"] + d["fetch_data_B"]
        byte_rows.append(dict(
            stage=stage, p=p, seed=seed, node=n, group_len=len(group), n_records_created=n_created,
            **{k: d[k] for k in ("state_req_n", "state_req_B", "fetch_int_n", "fetch_int_B", "state_rep_n",
                                  "state_rep_B", "fetch_data_n", "fetch_data_B", "state_rep_rx_B", "fetch_data_rx_B")},
            logged_sent=sent_w[n], logged_recv=recv_w[n], recon_sent=recon_sent,
            resid_sent=(sent_w[n] - recon_sent) if sent_w[n] is not None else None,
        ))

    return dict(nodes=nodes, cfgs=cfgs, meta=meta, t0=t0, ws=ws, we=we, polls=polls)


def main():
    """Standalone use: breakdown2.py <run_dir>... -> deliveries.csv, bytes.csv in the cwd."""
    del_rows, byte_rows = [], []
    anomalies = defaultdict(int)
    anomalies["link_ts_lag_ms"] = []
    for rd in sys.argv[1:]:
        analyze(rd, del_rows, byte_rows, anomalies)
    for name, rows in (("deliveries.csv", del_rows), ("bytes.csv", byte_rows)):
        if rows:
            with open(name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    anomalies.pop("link_ts_lag_ms")
    print("anomalies:", dict(anomalies))


if __name__ == "__main__":
    main()
