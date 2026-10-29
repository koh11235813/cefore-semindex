"""Per-run attribution of record-hop waits to state-poll decisions (read-only over one run dir).

For each in-window record (o, s) and each hop (m -> n) along the line h0-h1-h2-h3-h4,
the hop is credited to n's `state_poll` decision toward m whose gap (t_i, t_{i+1}]
contains the record's arrival at m. Wait = learned - arrival, fetch = stored_n - learned.
The per-run result is pre-aggregated into small sum dicts keyed by the strata the
downstream tables need, so the parent never holds ~1M row objects.

Reused (not rewritten): semindex.analyze.discover_node_events, semindex.check.read_results,
semindex.evalagg.derive_t0_ms (t0 + window rule), semindex.evalgen window constants.
Copy-adapted from docs/wayfinder/semindex-stage2/evidence/04/breakdown2.py (main @ 4d94da8):
  - holder state-reply payload reconstruction (`holder_payload_len` closure there),
  - record canonical-byte reconstruction from the agent configs' schedules,
  - state-request / fetch name byte accounting (names.build_*_name lengths),
  - reply -> poll matching rule "latest poll toward m at or before the reply".
"""
from __future__ import annotations

import bisect
import glob
import json
import os
from collections import defaultdict

from semindex import evalagg, evalgen, holder_state, names, schema
from semindex.analyze import discover_node_events
from semindex.check import read_results
from semindex.store import OriginProgress

POS = {f"h{i}": i for i in range(5)}
NODE = {i: f"h{i}" for i in range(5)}


def ctob(c):
    """consecutive_poll_timeouts bucket: 0/1/2/3+ (3+ pooled because counts thin out fast)."""
    return "3+" if c >= 3 else str(c)


def run_name_dir(run_dir):
    """The one `<run_dir>/<name>/` holding hN.json (copied from breakdown2.run_name_dir)."""
    hits = [os.path.dirname(p) for p in glob.glob(f"{run_dir}/*/h0.json")]
    if len(hits) != 1:
        raise RuntimeError(f"{run_dir}: expected one <name>/h0.json, found {hits}")
    return hits[0]


def _add(acc, key, **vals):
    """Accumulate named sums into stratum `key` of `acc`; fields absent so far start at 0."""
    d = acc[key]
    for k, v in vals.items():
        d[k] = d.get(k, 0) + v


def attribute_run(stage, p, seed, run_dir, disc=None):
    """Attribute one valid run; returns {'hop': {...}, 'dec': {...}, 'chk': {...}, 'pair': {...}}.

    `disc` is this run's {(origin, seq, node): latency_ms or None} from main-agg discovery.csv,
    used only for the exactness check (None -> check skipped for this run, counted).
    """
    meta = json.load(open(f"{run_dir}/meta.json"))
    chk = defaultdict(float)
    if not meta["validity"]["valid"]:
        chk["runs_invalid"] += 1
        return {"hop": {}, "dec": {}, "chk": dict(chk), "pair": {}, "gapdiff": {}}
    chk["runs_valid"] += 1
    nodes = discover_node_events(run_dir)
    results = read_results(run_dir)
    t0 = evalagg.derive_t0_ms(nodes, results)
    ws = t0.t0_ms + evalgen.WINDOW_START_S * 1000.0
    we = t0.t0_ms + evalgen.WINDOW_END_S * 1000.0
    for evs in nodes.values():
        if sum(1 for e in evs if e.get("event") == "agent_start") > 1:
            chk["multi_life_nodes"] += 1
    name_dir = run_name_dir(run_dir)
    cfgs = {n: json.load(open(f"{name_dir}/{n}.json")) for n in nodes}
    group = cfgs["h0"]["group"]

    created, stored = {}, {}
    polls = defaultdict(list)       # (n, m) -> [(ts, arm, applied_s, status, cto, in_flight)]
    srx = defaultdict(list)         # (n, m) -> [(ts, hi)]
    first_fp = {}                   # (n, o, s) -> (ts, nbr)
    n_fetch_int = defaultdict(int)  # (n, o, s) -> fetch Interests sent (incl. retries)
    dec_ts = defaultdict(list)      # n -> [(ts, state)] every decision, for the notified_max cross-check
    for n, evs in nodes.items():
        for e in evs:
            k = e.get("event")
            if k == "record_created":
                created[(e["origin"], e["seq"])] = e["ts_ms"]
            elif k == "record_stored":
                key = (e["origin"], e["seq"], n)
                if key in stored:
                    chk["dup_record_stored"] += 1
                if key not in stored or e["ts_ms"] < stored[key][0]:
                    stored[key] = (e["ts_ms"], e["holder"])
            elif k == "decision":
                dec_ts[n].append((e["ts_ms"], e["state"]))
                a = e["action"]
                if e["kind"] == "state_poll":
                    st = e["state"]
                    polls[(n, a["neighbor"])].append(
                        (e["ts_ms"], a["interval_s"], a.get("applied_interval_s", a["interval_s"]),
                         st.get("last_poll_status"), st.get("consecutive_poll_timeouts", 0), st.get("in_flight", 0))
                    )
                elif e["kind"] == "fetch_plan":
                    for o, s in a["fetched"]:
                        first_fp.setdefault((n, o, s), (e["ts_ms"], a["neighbor"]))
                        n_fetch_int[(n, o, s)] += 1
            elif k == "state_rx":
                srx[(n, e["neighbor"])].append((e["ts_ms"], e["hi"]))
    for v in list(polls.values()) + list(srx.values()):
        v.sort(key=lambda x: x[0])
    for v in dec_ts.values():
        v.sort(key=lambda x: x[0])
    s2c_like = stage == "s2c"
    if s2c_like:
        # stage-1 policy: interval_s is the jittered value itself, not one of 4 arms; label the arm "2.0".
        for key, lst in polls.items():
            polls[key] = [(t, 2.0, ap, st, c, f) for (t, _a, ap, st, c, f) in lst]

    # ---- record bytes (copy-adapted from breakdown2 H2) --------------------------------
    rec_bytes = {}
    for n, cfg in cfgs.items():
        for idx, ent in enumerate(cfg["schedule"]):
            key = (n, idx + 1)
            if key in created:
                rec = schema.IndexRecord(origin=n, seq=idx + 1, content_name=ent["content_name"], zone=ent["zone"],
                                         disaster=ent["disaster"], media=ent["media"], observed_at=created[key])
                rec_bytes[key] = len(schema.canonical_bytes(rec))

    # ---- holder timeline + reply reconstruction (copy-adapted from breakdown2 holder_payload_len) ----
    held_tl = defaultdict(list)  # holder -> sorted [(ts, origin, seq)]
    for (o, s), ts in created.items():
        held_tl[o].append((ts, o, s))
    for (o, s, n), (ts, _) in stored.items():
        held_tl[n].append((ts, o, s))
    for v in held_tl.values():
        v.sort()
    held_ts = {h: [x[0] for x in v] for h, v in held_tl.items()}

    def reply_recon(h, t):
        """(payload_len, {origin: hi}, {origin: set(extra-as-encoded)}) holder h would serve at t.

        The holder's state is taken at the requester's rx time t (breakdown2's rule); the
        true serve instant is a few ms earlier, so a record stored at h inside that gap is
        mis-included -- caught by the logged-hi cross-check below.
        """
        cut = bisect.bisect_right(held_ts.get(h, []), t)
        held = defaultdict(set)
        for _, o, s in held_tl.get(h, [])[:cut]:
            held[o].add(s)
        prog = {}
        for o, ss in held.items():
            hi = 0
            while hi + 1 in ss:
                hi += 1
            prog[o] = OriginProgress(hi, tuple(sorted(x for x in ss if x > hi)))
        enc = holder_state.encode(prog)
        dec = holder_state.decode(enc)
        return len(enc), {o: d["hi"] for o, d in dec.items()}, {o: set(d["extra"]) for o, d in dec.items()}

    # Reconstruct every reply once; attach bytes to the matched poll.
    # 2026-10-29: the holder state at the requester's rx ts over-includes records stored at m
    # between serve and n's *processing* of the reply (state_rx ts is n's processing time,
    # 100-500 ms after the poll when n's loop is busy) -- 8% of replies in a p=0.2 run showed
    # recon hi > logged hi, always in that direction. Coverage via extra therefore uses the
    # state at the matched poll's send time t_j (a lower bound on the serve time when the
    # reply answers poll j), and only when that lower bound is consistent with the logged hi
    # (recon_hi(t_j) <= logged hi for every origin; otherwise the reply answers an older poll
    # and t_j is not a lower bound). Bytes keep breakdown2's rx-time reconstruction.
    rx_info = {}  # (n, m) -> list aligned with srx: (ts, hi_logged, lower_ok, extra_sets_at_tj, plen, tj)
    for (n, m), lst in srx.items():
        pl = [x[0] for x in polls.get((n, m), [])]
        out = []
        for ts, hi in lst:
            plen, hi_r, _ex_r = reply_recon(m, ts)
            j = bisect.bisect_right(pl, ts) - 1
            tj = pl[j] if j >= 0 else ts
            _pl2, hi_j, ex_j = reply_recon(m, tj)
            lower_ok = all(v <= hi.get(o, 0) for o, v in hi_j.items())
            chk["replies"] += 1
            if {o: v for o, v in hi_r.items() if v} != {o: v for o, v in hi.items() if v}:
                chk["reply_hi_recon_at_rx_mismatch"] += 1
            if not lower_ok:
                chk["reply_hi_recon_at_poll_exceeds_logged"] += 1
            out.append((ts, hi, lower_ok, ex_j, plen, tj))
        rx_info[(n, m)] = out

    # ---- per-decision records ------------------------------------------------------
    # dec_rows[(n, m)][i] = mutable dict; reply bytes attached by "latest poll <= reply ts".
    dec_rows = {}
    for (n, m), lst in polls.items():
        rows = []
        for i, (t, arm, ap, st, c, f) in enumerate(lst):
            gap_end = lst[i + 1][0] if i + 1 < len(lst) else t + ap * 1000.0
            rows.append(dict(
                t=t, arm=arm, ap=ap, status="null" if st is None else st, ctob=ctob(c),
                infl0=(f == 0), prev=(lst[i - 1][1] if i > 0 else "none"),
                gap=gap_end - t, last=(i + 1 == len(lst)),
                int_B=len(names.build_state_name(group, m, n, int(t))), rep_B=0, n_rep=0,
            ))
        pts = [r["t"] for r in rows]
        for ts, _hi, _ok, _ex, plen, _tj in rx_info.get((n, m), []):
            j = bisect.bisect_right(pts, ts) - 1
            if j < 0:
                chk["reply_before_any_poll"] += 1
                continue
            rows[j]["rep_B"] += len(names.build_state_name(group, m, n, int(rows[j]["t"]))) + plen
            rows[j]["n_rep"] += 1
        dec_rows[(n, m)] = rows

    hop_acc = defaultdict(dict)
    dec_acc = defaultdict(dict)
    gapdiff = defaultdict(lambda: [0, 0.0, 0.0])  # arm -> [n, sum(gap-applied), max |gap-applied|]

    def stratum(n, m, r, hop):
        """Key under which a decision row's hop/decision sums are pooled.

        Positional: downstream collapses select dimensions by index, so reordering silently breaks them.
        """
        return (stage, p, r["arm"], hop, r["status"], r["ctob"], r["prev"], r["infl0"])

    for (n, m), rows in dec_rows.items():
        o = NODE[0] if POS[m] < POS[n] else NODE[4]  # the one origin whose records flow m -> n
        hop = abs(POS[n] - POS[o])
        for r in rows:
            if not r["last"]:
                g = gapdiff[r["arm"]]
                d = r["gap"] - r["ap"] * 1000.0
                g[0] += 1
                g[1] += d
                g[2] = max(g[2], abs(d))
            if ws <= r["t"] <= we:
                _add(dec_acc, stratum(n, m, r, hop), n_gap=1, gap_ms=r["gap"], int_B=r["int_B"],
                     rep_B=r["rep_B"], n_rep=r["n_rep"], n_norep=int(r["n_rep"] == 0))

    # ---- per record-hop -----------------------------------------------------------
    pending_batch, batch_n = [], defaultdict(int)
    pair = {}  # (o, s, n) -> my latency (sum of hops) or None if censored
    in_win = {k: c for k, c in created.items() if ws <= c <= we}
    for (o, s), c in in_win.items():
        po = POS[o]
        d = 1 if po == 0 else -1
        chain_ok = True
        cum = 0.0
        for h in range(1, 5):
            n, m = NODE[po + d * h], NODE[po + d * (h - 1)]
            sn = stored.get((o, s, n))
            arrival = c if m == o else (stored.get((o, s, m)) or (None,))[0]
            if arrival is None or arrival > we or not chain_ok:
                # upstream never stored (in-window): this hop is undefined, not a mismatch.
                chk["hop_upstream_censored"] += 1
                pair[(o, s, n)] = None
                chain_ok = False
                continue
            censored = sn is None or sn[0] > we
            if sn is not None and sn[1] != m:
                chk["exc_holder_not_upstream"] += 1
            fp = first_fp.get((n, o, s))
            if fp is not None and fp[1] != m:
                chk["exc_fp_neighbor_not_upstream"] += 1
            # learned: first reply from m after arrival covering (o, s) via logged hi, else
            # via reconstructed extra (only when the reconstruction reproduces the logged hi).
            learned, rule = None, None
            for ts, hi, lower_ok, ex, _pl, tj in rx_info.get((n, m), []):
                if ts <= arrival:
                    continue
                if hi.get(o, 0) >= s:
                    learned, rule = ts, "hi"
                    break
                if lower_ok and arrival <= tj and s in ex.get(o, ()):
                    learned, rule = ts, "extra"
                    # cross-check with the tracker's own notified_max in n's next decision at/after ts
                    dl = dec_ts.get(n, [])
                    k = bisect.bisect_left(dl, ts, key=lambda x: x[0])
                    if k < len(dl) and dl[k][0] == ts:
                        nm = dl[k][1].get("origins", {}).get(o, {}).get("advertised", {}).get(m, {}).get("notified_max")
                        chk["extra_rule_nmax_" + ("agree" if nm is not None and nm >= s else "disagree")] += 1
                    else:
                        chk["extra_rule_nmax_unavailable"] += 1
                    break
            if fp is not None and (learned is None or fp[0] < learned):
                rule = "fp_before_reply" if learned is not None else "fp_only"
                learned = fp[0]
            if learned is not None and fp is not None:
                chk["fp_eq_learned" if fp[0] == learned else "fp_after_learned"] += 1
            pk = f"rule_{rule}" if rule else "rule_none"
            chk[f"{pk}|{p}"] += 1
            rows = dec_rows.get((n, m), [])
            pts = [r["t"] for r in rows]
            i = bisect.bisect_left(pts, arrival) - 1
            if censored:
                chain_ok = False
                pair[(o, s, n)] = None
            if learned is None:
                if not censored:
                    chk["stored_without_learn"] += 1
                    chain_ok = False
                    continue
            wait = (learned - arrival) if learned is not None else None
            fetch = (sn[0] - learned) if (not censored and learned is not None) else None
            if fetch is not None and fetch < 0:
                chk["neg_fetch"] += 1
            if not censored:
                cum += wait + fetch
                pair[(o, s, n)] = cum if chain_ok else None
            if i < 0:
                chk["hop_no_gap_before"] += 1
                continue
            r = rows[i]
            if not (ws <= r["t"] <= we):
                chk["hop_gap_outside_window"] += 1
                continue
            gap_end = r["t"] + r["gap"]
            wait_cap = min(learned if learned is not None else float("inf"), we) - arrival
            key = stratum(n, m, r, h)
            if censored:
                _add(hop_acc, key, n_c=1, waitcap_all=wait_cap)
                continue
            j = bisect.bisect_right(pts, learned) - 1
            spill = (learned - gap_end) if j >= i + 2 else 0.0
            post = max(0.0, learned - gap_end)
            fb = n_fetch_int[(n, o, s)] * len(names.build_record_name(group, m, o, s)) \
                + len(names.build_record_name(group, m, o, s)) + rec_bytes[(o, s)]
            _add(hop_acc, key, n_u=1, wait_u=wait, fetch_u=fetch, spill_u=spill, post_u=post,
                 n_spill=int(j >= i + 2), fbytes_u=fb, waitcap_all=wait_cap, wf2_u=(wait + fetch) ** 2)
            pending_batch.append((key, (n, m, learned)))
            batch_n[(n, m, learned)] += 1

    # Batch size = how many uncensored record-hops n learned from the same reply of m (record-weighted
    # when averaged as batch_u / n_u). Diagnostic for the fetch-time confound: a big batch queues
    # behind max_in_flight_per_neighbor, whatever arm the gap had.
    for key, bk in pending_batch:
        _add(hop_acc, key, batch_u=batch_n[bk])

    # ---- exactness vs discovery.csv --------------------------------------------------
    pair_out = defaultdict(float)
    if disc is None:
        chk["runs_without_discovery_rows"] += 1
    else:
        for (o, s), c in in_win.items():
            for n in nodes:
                if n == o:
                    continue
                dl = disc.get((o, s, n), "missing")
                mine = pair.get((o, s, n))
                H = abs(POS[n] - POS[o])
                if dl == "missing":
                    chk["exact_missing_in_discovery"] += 1
                    continue
                if dl is None or mine is None:
                    if (dl is None) != (mine is None):
                        chk["exact_censor_disagree"] += 1
                    pair_out[("c", H)] += 1
                    continue
                err = abs(mine - dl)
                pair_out[("n", H)] += 1
                pair_out[("lat", H)] += dl
                pair_out[("within1", H)] += int(err <= 1.0)
                chk["exact_n"] += 1
                chk["exact_within1"] += int(err <= 1.0)
                chk["exact_maxerr"] = max(chk["exact_maxerr"], err)
    return {"hop": {k: v for k, v in hop_acc.items()}, "dec": {k: v for k, v in dec_acc.items()},
            "chk": dict(chk), "pair": dict(pair_out), "gapdiff": {k: list(v) for k, v in gapdiff.items()}}
