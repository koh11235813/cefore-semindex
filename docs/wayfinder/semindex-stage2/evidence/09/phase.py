"""Poll phase-alignment check for stage-2 s5* runs (read-only over the store).

Part A reuses evidence/04/trial04.py's sync metric verbatim (sync_rows_for_run
+ the same pooling as its sync table) so the numbers line up with map.md.
Part B looks at retry polls only: a retry is the state_poll that follows a
timed_out state_poll_reschedule of the same (node, neighbour) whose
poll_epoch_ms is the previous poll (policy stages). Baselines have no
reschedule rows, so their analogue is "the poll sent right after a timed-out
poll" (state.last_poll_status == timed_out); policy stages report both.

Only events.jsonl, meta.json, results.json and the agent configs are read.
"""
import argparse, bisect, csv, glob, json, os, statistics, sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "04"))
import trial04, breakdown2  # noqa: E402
from semindex import evalgen  # noqa: E402

WS, WE = breakdown2.WS, breakdown2.WE
SLOT_MS = evalgen.DISCONNECT_SLOT_S * 1000.0
D = 50.0          # coincidence window, ms (map.md uses 50 ms)
UP_HEAD_MS = 500.0
LOCAL_MS = 2500.0


def load(run_dir):
    """Light parse: polls / reschedules per (node, nbr), configs, window, link intervals."""
    meta = json.load(open(f"{run_dir}/meta.json"))
    polls = defaultdict(list)    # (n, nb) -> [(ts, last_status)]
    resch = defaultdict(list)    # (n, nb) -> [(ts, epoch, trigger)]
    t0 = None
    for d in sorted(glob.glob(f"{run_dir}/debug/node_dirs/h*")):
        n = os.path.basename(d)
        with open(f"{d}/semindex/events.jsonl") as f:
            for line in f:
                if '"agent_start"' in line and t0 is None:
                    t0 = json.loads(line)["t0_ms"]
                if '"state_poll' not in line:
                    continue
                e = json.loads(line)
                if e.get("event") != "decision":
                    continue
                a = e["action"]
                if e["kind"] == "state_poll":
                    polls[(n, a["neighbor"])].append((e["ts_ms"], e["state"].get("last_poll_status")))
                elif e["kind"] == "state_poll_reschedule":
                    resch[(n, a["neighbor"])].append((e["ts_ms"], a["poll_epoch_ms"], a["trigger"]))
    for v in list(polls.values()) + list(resch.values()):
        v.sort()
    cfgs = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{breakdown2.run_name_dir(run_dir)}/h*.json")}
    results = json.load(open(f"{run_dir}/results.json"))
    links = breakdown2.link_intervals(results)
    lag = [breakdown2.iso_ms(r["ts"]) - (t0 + r["actual_at"] * 1000) for r in results
           if r.get("op_type") == "event" and r.get("actual_at") is not None]
    return meta, t0, polls, resch, cfgs, links, lag


def retries_of(polls, resch, use_resch):
    """Per pair: sorted retry send times (all, window filter later) and per-poll outcome lookup."""
    out = {}
    for k, pl in polls.items():
        ts = [t for t, _ in pl]
        r = []
        if use_resch:
            rs = resch.get(k, [])
            rts = [x[0] for x in rs]
            for i in range(1, len(pl)):
                j = bisect.bisect_right(rts, ts[i]) - 1
                if j >= 0 and rs[j][2] == "timed_out" and rs[j][1] == ts[i - 1]:
                    r.append(ts[i])
        else:
            r = [t for t, s in pl if s == "timed_out"]
        out[k] = r
    return out


def analyze(job):
    stage, run_dir = job
    meta, t0, polls, resch, cfgs, links, lag = load(run_dir)
    ws, we = t0 + WS, t0 + WE
    policy = bool(meta.get("state_poll_policy"))
    parsed = dict(ws=ws, we=we, cfgs=cfgs, polls={k: [t for t, _ in v] for k, v in polls.items()})
    sync_rows = trial04.sync_rows_for_run(stage, meta, parsed, [D])
    for r in sync_rows:
        r["_drift"] = r["_drift"]  # kept for pooling
    rr = retries_of(polls, resch, policy)
    rr_alt = retries_of(polls, resch, False) if policy else None
    rwin = {k: [t for t in v if ws <= t <= we] for k, v in rr.items()}
    # outcome of each poll from the reschedule row that names it (policy only)
    outcome = {}
    for k, rs in resch.items():
        for _, ep, trig in rs:
            outcome[(k, ep)] = trig

    pairs = sorted(polls)
    ret_rows = []
    allret = sorted((t, k) for k in pairs for t in rwin.get(k, []))
    for k in pairs:
        xs = rwin.get(k, [])
        n, nb = k
        ivs = links.get(frozenset((n, nb)), [])
        ups = sorted(e for s, e in ivs if e != float("inf"))
        others = sorted(t for kk in pairs if kk != k for t in rwin.get(kk, []))
        rev = rwin.get((nb, n), [])
        # split "others": same sender (one node polls both neighbours on one tick) vs other nodes
        oth_same = sorted(t for kk in pairs if kk != k and kk[0] == n for t in rwin.get(kk, []))
        oth_cross = sorted(t for kk in pairs if kk[0] != n for t in rwin.get(kk, []))
        so = trial04.sync_stream(xs, others, ws, we, [D])
        sr = trial04.sync_stream(xs, rev, ws, we, [D])
        # time-coverage chance levels of this link over the window
        down_cov = breakdown2.overlap(ivs, ws, we) / (we - ws)
        head = [(u, u + UP_HEAD_MS) for u in ups]
        head_cov = breakdown2.overlap(head, ws, we) / (we - ws)
        allp = [t for t, _ in polls[k] if ws <= t <= we]
        def cls(t):
            down = breakdown2.is_down(ivs, t)
            j = bisect.bisect_right(ups, t) - 1
            since = t - ups[j] if j >= 0 else None
            inhead = (not down) and since is not None and since < UP_HEAD_MS
            return down, inhead
        rc = [cls(t) for t in xs]
        pc = [cls(t) for t in allp]
        phases = [((t - (t0 + WS)) % SLOT_MS) for t in xs]
        # consecutive-retry phase change, wrapped to (-SLOT/2, SLOT/2]
        dph = []
        for a, b in zip(xs, xs[1:]):
            if b - a < 2.5 * SLOT_MS:   # same outage episode (back-to-back retries)
                x = ((b - a) % SLOT_MS)
                dph.append(abs(x - SLOT_MS if x > SLOT_MS / 2 else x))
        fails = [outcome.get((k, t)) for t in xs] if policy else []
        # local chance: Y density within +-LOCAL_MS of each x, scaled to the +-D window.
        # Retries are confined to outage episodes, so a whole-window shift baseline
        # is too low for sparse retry streams; this one keeps the confinement.
        def local(ys):
            return sum((bisect.bisect_right(ys, x + LOCAL_MS) - bisect.bisect_left(ys, x - LOCAL_MS)) * D / LOCAL_MS
                       for x in xs)
        head_fail = sum(1 for (dn, h), f in zip(rc, fails) if h and f == "timed_out")
        head_n = sum(1 for (dn, h), f in zip(rc, fails) if h and f is not None)
        ret_rows.append(dict(
            stage=stage, p=float(meta["p"]), seed=meta["seed"], x=f"{n}->{nb}",
            n_polls=len(allp), n_retry=len(xs),
            n_retry_alt=len([t for t in rr_alt[k] if ws <= t <= we]) if policy else None,
            oth_hits=so[D]["hits"], oth_base=so[D]["shift_base"],
            rev_hits=sr[D]["hits"], rev_base=sr[D]["shift_base"], rev_n_y=len(rev),
            oth_first_n=so[D]["first_n"], oth_first_hits=so[D]["first_hits"],
            oth_second_n=so[D]["second_n"], oth_second_hits=so[D]["second_hits"],
            ret_down=sum(d for d, _ in rc), ret_head=sum(h for _, h in rc),
            poll_down=sum(d for d, _ in pc), poll_head=sum(h for _, h in pc),
            down_cov=down_cov, head_cov=head_cov,
            phase_head=sum(1 for ph in phases if ph < UP_HEAD_MS),
            _dph=dph, _phases=phases,
            oth_local=local(others), rev_local=local(rev),
            same_hits=trial04.sync_stream(xs, oth_same, ws, we, [D])[D]["hits"], same_local=local(oth_same),
            cross_hits=trial04.sync_stream(xs, oth_cross, ws, we, [D])[D]["hits"], cross_local=local(oth_cross),
            first_poll_ms=(polls[k][0][0] - t0) if polls[k] else None,
            rev_pers_num=sr[D]["pers_num"], rev_pers_den=sr[D]["pers_den"],
            oth_pers_num=so[D]["pers_num"], oth_pers_den=so[D]["pers_den"],
            head_fail=head_fail, head_n=head_n,
            _poll_phases=[((t - (t0 + WS)) % SLOT_MS, t < ws + (we - ws) / 2) for t in allp],
            ret_timed_out=sum(1 for f in fails if f == "timed_out"),
            ret_answered=sum(1 for f in fails if f == "answered"),
        ))
    # cluster sizes across all 8 pairs (single linkage, gap <= D)
    clusters, cur = [], []
    for t, k in allret:
        if cur and t - cur[-1][0] > D:
            clusters.append(cur); cur = []
        cur.append((t, k))
    if cur:
        clusters.append(cur)
    csz = [len({k for _, k in c}) for c in clusters]
    # recovery: link_up -> first poll sent after it, per directed pair of that link
    rec = []
    for k in pairs:
        ivs = links.get(frozenset(k), [])
        ts = [t for t, _ in polls[k]]
        for s, e in ivs:
            if e == float("inf") or not (ws <= e <= we):
                continue
            j = bisect.bisect_right(ts, e)
            if j < len(ts):
                rec.append(ts[j] - e)
    run = dict(stage=stage, p=float(meta["p"]), seed=meta["seed"], n_ret=len(allret),
               csz=csz, rec=rec, lag=lag)
    return sync_rows, ret_rows, run


def pool_sync(sync_rows):
    """Same pooling as trial04.main's sync table (delta 50 only)."""
    sg = defaultdict(list)
    for r in sync_rows:
        sg[(r["stage"], r["p"], r["pair_type"])].append(r)
    out = []
    for (label, p, pt), rs in sorted(sg.items()):
        nx = sum(r["n_x"] for r in rs)
        hits = sum(r["hits"] for r in rs)
        sb = sum(r["shift_base"] * r["n_x"] for r in rs if r["shift_base"] is not None)
        drift = [x for r in rs for x in r["_drift"]]
        fn, fh = sum(r["first_n"] for r in rs), sum(r["first_hits"] for r in rs)
        sn, sh = sum(r["second_n"] for r in rs), sum(r["second_hits"] for r in rs)
        pn, pdn = sum(r["pers_num"] for r in rs), sum(r["pers_den"] for r in rs)
        out.append(dict(stage=label, p=p, pair_type=pt, n_runs=len({r["seed"] for r in rs}), n_polls=nx,
                        coincidence=hits / nx if nx else None, chance_shift=sb / nx if nx else None,
                        excess_ratio=hits / sb if sb else None,
                        first_half=fh / fn if fn else None, second_half=sh / sn if sn else None,
                        persistence=pn / pdn if pdn else None,
                        offset_drift_median_ms=statistics.median(drift) if drift else None))
    return out


def q(v):
    return (statistics.quantiles(v, n=4) if len(v) >= 2 else [None] * 3)


def pool_retry(ret_rows, runs):
    g = defaultdict(list)
    for r in ret_rows:
        g[(r["stage"], r["p"])].append(r)
    rg = defaultdict(list)
    for r in runs:
        rg[(r["stage"], r["p"])].append(r)
    out = []
    for key, rs in sorted(g.items()):
        S = lambda f: sum(r[f] for r in rs if r[f] is not None)
        nr = S("n_retry")
        npl = S("n_polls")
        wcov = lambda f, w: sum(r[f] * r[w] for r in rs) / S(w) if S(w) else None
        dph = [x for r in rs for x in r["_dph"]]
        csz = [x for r in rg[key] for x in r["csz"]]
        rec = [x for r in rg[key] for x in r["rec"]]
        lag = [x for r in rg[key] for x in r["lag"]]
        rq = q(rec)
        out.append(dict(
            stage=key[0], p=key[1], n_runs=len(rg[key]), n_polls=npl, n_retry=nr,
            n_retry_alt=S("n_retry_alt") if rs[0]["n_retry_alt"] is not None else None,
            oth_coinc=S("oth_hits") / nr if nr else None, oth_chance=S("oth_base") * 1.0 / nr if nr else None,
            oth_ratio=S("oth_hits") / S("oth_base") if S("oth_base") else None,
            oth_first=S("oth_first_hits") / S("oth_first_n") if S("oth_first_n") else None,
            oth_second=S("oth_second_hits") / S("oth_second_n") if S("oth_second_n") else None,
            oth_chance_local=S("oth_local") / nr if nr else None,
            oth_ratio_local=S("oth_hits") / S("oth_local") if S("oth_local") else None,
            oth_persist=S("oth_pers_num") / S("oth_pers_den") if S("oth_pers_den") else None,
            rev_chance_local=S("rev_local") / nr if nr else None,
            rev_ratio_local=S("rev_hits") / S("rev_local") if S("rev_local") else None,
            rev_persist=S("rev_pers_num") / S("rev_pers_den") if S("rev_pers_den") else None,
            head_fail_share=S("head_fail") / S("head_n") if S("head_n") else None, head_n=S("head_n"),
            same_coinc=S("same_hits") / nr if nr else None,
            same_ratio_local=S("same_hits") / S("same_local") if S("same_local") else None,
            cross_coinc=S("cross_hits") / nr if nr else None,
            cross_ratio_local=S("cross_hits") / S("cross_local") if S("cross_local") else None,
            head_chance_cond=(1 - S("ret_down") / nr) * UP_HEAD_MS / SLOT_MS if nr else None,
            first_poll_med_ms=statistics.median([r["first_poll_ms"] for r in rs if r["first_poll_ms"] is not None]),
            first_poll_min_ms=min(r["first_poll_ms"] for r in rs if r["first_poll_ms"] is not None),
            first_poll_max_ms=max(r["first_poll_ms"] for r in rs if r["first_poll_ms"] is not None),
            rev_coinc=S("rev_hits") / nr if nr else None, rev_chance=S("rev_base") / nr if nr else None,
            rev_ratio=S("rev_hits") / S("rev_base") if S("rev_base") else None,
            ret_down_share=S("ret_down") / nr if nr else None,
            ret_head_share=S("ret_head") / nr if nr else None,
            poll_down_share=S("poll_down") / npl if npl else None,
            poll_head_share=S("poll_head") / npl if npl else None,
            chance_down=wcov("down_cov", "n_retry"), chance_head=wcov("head_cov", "n_retry"),
            slot_phase_lt500_share=S("phase_head") / nr if nr else None,
            dphase_med_ms=statistics.median(dph) if dph else None, n_dphase=len(dph),
            dphase_lt50_share=sum(1 for x in dph if x <= D) / len(dph) if dph else None,
            retry_timed_out=S("ret_timed_out"), retry_answered=S("ret_answered"),
            cluster_max_pairs=max(csz) if csz else None,
            retries_in_clusters_ge3=(sum(c for c in csz if c >= 3) / sum(csz)) if csz else None,
            recov_n=len(rec), recov_q1=rq[0], recov_med=rq[1], recov_q3=rq[2],
            link_ts_lag_med_ms=statistics.median(lag) if lag else None,
        ))
    return out


def write_csv(path, rows):
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/home/lab_shared/semindex-eval")
    ap.add_argument("--agg", required=True, help="dir holding <stage>/runs.csv from evalagg")
    ap.add_argument("--stages", default="s5short,s5wait,s5b10p043,s5b3p695,s5c")
    ap.add_argument("--ps", default="0,0.2,0.4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    ps = {float(x) for x in a.ps.split(",")}
    jobs = []
    for s in a.stages.split(","):
        for row in csv.DictReader(open(f"{a.agg}/{s}/runs.csv")):
            if row["valid"] != "valid" or float(row["p"]) not in ps:
                continue
            pdir = "p0" if float(row["p"]) == 0 else f"p{float(row['p']):g}"
            jobs.append((s, f"{a.store}/{s}/{pdir}/{int(row['seed'])}"))
    with Pool(a.workers) as pool:
        res = pool.map(analyze, jobs)
    sync_rows = [x for r in res for x in r[0]]
    ret_rows = [x for r in res for x in r[1]]
    runs = [r[2] for r in res]
    write_csv(f"{a.out}/sync_table.csv", pool_sync(sync_rows))
    write_csv(f"{a.out}/retry_pairs.csv", ret_rows)
    write_csv(f"{a.out}/retry_table.csv", pool_retry(ret_rows, runs))
    # slot-phase histogram of retries (0.5 s bins over the 5 s slot)
    hist = defaultdict(lambda: [0] * 10)
    for r in ret_rows:
        for ph in r["_phases"]:
            hist[(r["stage"], r["p"])][min(9, int(ph // 500))] += 1
    with open(f"{a.out}/retry_phase_hist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "p"] + [f"{i*0.5:g}-{(i+1)*0.5:g}s" for i in range(10)])
        for k, v in sorted(hist.items()):
            w.writerow(list(k) + v)
    # slot-phase histogram of all in-window polls, by window half
    ph = defaultdict(lambda: [0] * 10)
    for r in ret_rows:
        for v, first in r["_poll_phases"]:
            ph[(r["stage"], r["p"], "1st" if first else "2nd")][min(9, int(v // 500))] += 1
    with open(f"{a.out}/poll_phase_hist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "p", "half"] + [f"{i*0.5:g}-{(i+1)*0.5:g}s" for i in range(10)])
        for k, v in sorted(ph.items()):
            w.writerow(list(k) + v)
    print(f"{len(jobs)} runs")


if __name__ == "__main__":
    main()
