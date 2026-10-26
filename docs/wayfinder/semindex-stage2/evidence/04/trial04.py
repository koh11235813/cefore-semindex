"""Stage-2 ticket 04 trial analysis: interval table, k table, poll-synchronization
check and mixed-policy sanity, over whatever trial stages exist in the store.

Read-only over the store. Per stage it (1) runs `python -m semindex.evalagg`
into <out>/agg/<stage>/ and reads its CSVs (run validity, KM discovery /
consistency medians, control bytes per record, pooled decision fields), then
(2) walks every run evalagg marked valid with breakdown2.analyze (per-hop
delay split, per-type control bytes) and a few extra per-run passes over the
same parsed events (fetch_plan sizes, poll synchronization, interval choices,
on-disk size). Outputs CSVs + summary.md into --out.

Run (the PYTHONPATH makes the branch's semindex win over the venv's editable main):
  PYTHONPATH=<worktree>/src \
  MPLCONFIGDIR=<scratch>/mpl \
  /home/lab_shared/cefore-semindex/.venv/bin/python3 trial04.py --out <dir>
"""
import argparse, bisect, csv, glob, json, math, os, random, statistics, subprocess, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import breakdown2  # noqa: E402
from semindex import evalagg  # noqa: E402

# Group names are embedded once in every Interest/Data name, so stages whose
# group name differs in length (t2i0p5_p000_seed1 is 17 chars, t2i1_... 15,
# t2mix_... 16, s1_p0.4_seed1 13) carry a per-name byte offset that scales with
# the poll rate. Normalized bytes rebase every name to this length.
REF_GROUP_LEN = 15

DEFAULT_STAGE_GLOB = "t2*"


# ---------------------------------------------------------------- helpers

def fnum(x):
    """float(x), or None for an empty CSV cell."""
    return None if x in (None, "") else float(x)


def mean(v):
    """Arithmetic mean, None for empty."""
    return statistics.fmean(v) if v else None


def read_csv(path):
    """Rows of a CSV as dicts; [] if the file is missing."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    """Write dict rows; header from the first row's keys unless `fields` is given."""
    fields = fields or (list(rows[0].keys()) if rows else ["empty"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def tree_size(path):
    """(apparent bytes, on-disk bytes like `du`) of everything under `path`."""
    app = disk = 0
    for dp, _, fs in os.walk(path):
        for fn in fs:
            st = os.lstat(os.path.join(dp, fn))
            app += st.st_size
            disk += st.st_blocks * 512
    return app, disk


def policy_label(k, choices):
    """Human label of the poll policy recorded in meta.json."""
    if not choices:
        return "default(2s+-20%)"
    if len(choices) == 1:
        return f"fixed {choices[0]:g}s"
    return "uniform{" + ",".join(f"{c:g}" for c in choices) + "}"


# ---------------------------------------------------------------- sync metric

def _nearest_offset(ys, x):
    """Signed x - (nearest y), or None if ys empty."""
    if not ys:
        return None
    i = bisect.bisect_left(ys, x)
    cands = [ys[j] for j in (i - 1, i) if 0 <= j < len(ys)]
    y = min(cands, key=lambda y: abs(x - y))
    return x - y


def _coinc_flags(xs, ys, delta):
    """Per x: whether some y lies within +-delta ms."""
    out = []
    for x in xs:
        d = _nearest_offset(ys, x)
        out.append(d is not None and abs(d) <= delta)
    return out


def sync_stream(xs, ys, ws, we, deltas, n_shifts=200):
    """Synchronization stats of poll stream xs against ys (both sorted, in-window ms).

    Per delta: hits (x with a y within +-delta), shift baseline (mean hit
    fraction after circularly shifting ys by n_shifts seeded uniform offsets -- keeps
    ys' own interval structure, so periodic streams get a fair chance level),
    Poisson baseline 1-exp(-2*delta*n_y/W), hits in the first / second half,
    and persistence counts (pairs of consecutive x both coincident).
    Delta-independent: per-x |change in signed nearest-y offset| (drift).
    """
    W = we - ws
    mid = ws + W / 2
    res = {"n_x": len(xs), "n_y": len(ys), "drift": []}
    offs = [_nearest_offset(ys, x) for x in xs]
    res["drift"] = [abs(b - a) for a, b in zip(offs, offs[1:]) if a is not None and b is not None]
    shifted = []
    # 2026-10-26: evenly spaced shifts (j*W/n) hit only a handful of phases
    # of a periodic stream (600 ms steps cover just 5 phases of a 1 s period),
    # doubling its chance level; seeded uniform shifts cover the phase circle.
    rng = random.Random(0)
    for _ in range(n_shifts):
        s = rng.uniform(0, W)
        shifted.append(sorted(((y - ws + s) % W) + ws for y in ys))
    for d in deltas:
        fl = _coinc_flags(xs, ys, d)
        base = mean([sum(_coinc_flags(xs, sy, d)) / len(xs) for sy in shifted]) if xs else None
        res[d] = dict(
            hits=sum(fl),
            first_n=sum(1 for x in xs if x < mid), first_hits=sum(f for x, f in zip(xs, fl) if x < mid),
            second_n=sum(1 for x in xs if x >= mid), second_hits=sum(f for x, f in zip(xs, fl) if x >= mid),
            pers_num=sum(1 for a, b in zip(fl, fl[1:]) if a and b), pers_den=sum(1 for a in fl[:-1] if a),
            shift_base=base, poisson_base=1 - math.exp(-2 * d * len(ys) / W) if W > 0 else None,
        )
    return res


def sync_rows_for_run(label, meta, parsed, deltas):
    """Rows of sync_pairs.csv for one run: every same-holder and reverse-link stream pair, both directions."""
    ws, we, cfgs = parsed["ws"], parsed["we"], parsed["cfgs"]
    polls = {k: [t for t in v if ws <= t <= we] for k, v in parsed["polls"].items()}
    pairs = []
    for h, cfg in cfgs.items():
        nb = cfg["neighbors"]
        for a in nb:
            for b in nb:
                if a != b:
                    pairs.append(("same_holder", f"{a}->{h}", (a, h), f"{b}->{h}", (b, h)))
        for b in nb:
            pairs.append(("link_reverse", f"{h}->{b}", (h, b), f"{b}->{h}", (b, h)))
    rows = []
    for ptype, xl, xk, yl, yk in pairs:
        r = sync_stream(polls.get(xk, []), polls.get(yk, []), ws, we, deltas)
        for d in deltas:
            rows.append(dict(stage=label, p=float(meta["p"]), seed=meta["seed"], pair_type=ptype,
                             x=xl, y=yl, delta_ms=d, n_x=r["n_x"], n_y=r["n_y"], **r[d],
                             drift_median_ms=statistics.median(r["drift"]) if r["drift"] else None,
                             _drift=r["drift"]))
    return rows


# ---------------------------------------------------------------- per-run pass

def per_run_extras(label, meta, run_dir, parsed):
    """fetch_plan sizes, interval/action_prob draws, decision counts and sizes for one valid run."""
    ws, we = parsed["ws"], parsed["we"]
    fetched, intervals, probs = [], [], Counter()
    n_sp_win = 0
    for evs in parsed["nodes"].values():
        for e in evs:
            if e["event"] != "decision":
                continue
            if e["kind"] == "fetch_plan" and ws <= e["ts_ms"] <= we:
                fetched.append(len(e["action"]["fetched"]))
            elif e["kind"] == "state_poll":
                # every draw counts for the choice sanity check, not only in-window ones
                intervals.append(e["action"]["interval_s"])
                probs[round(e["action_prob"], 6)] += 1
                n_sp_win += ws <= e["ts_ms"] <= we
    app, disk = tree_size(run_dir)
    ev_bytes = sum(os.path.getsize(p) for p in glob.glob(f"{run_dir}/debug/node_dirs/h*/semindex/events.jsonl"))
    info = dict(stage=label, p=float(meta["p"]), seed=meta["seed"], k=meta.get("k") or 1,
                choices=json.dumps(meta.get("state_poll_interval_choices_s")),
                run_dir=run_dir, du_disk_bytes=disk, apparent_bytes=app, events_jsonl_bytes=ev_bytes,
                n_state_poll_window=n_sp_win, n_fetch_plan_window=len(fetched))
    return info, fetched, intervals, probs


# ---------------------------------------------------------------- main

def build_s1_substore(store, out, seeds, ps):
    """Symlink a few stage-1 runs into <out>/s1store/s1sub/ so evalagg aggregates only them."""
    root = os.path.join(out, "s1store")
    for p in ps:
        for s in seeds:
            src = os.path.join(store, "s1", f"p{p}", str(s))
            dst = os.path.join(root, "s1sub", f"p{p}", str(s))
            if os.path.isdir(src) and not os.path.lexists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.symlink(src, dst)
    return root


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="/home/lab_shared/semindex-eval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=None,
                    help=f"comma list of stages; default: every <store>/{DEFAULT_STAGE_GLOB} dir")
    ap.add_argument("--s1-seeds", default="1,2,3", help="stage-1 seeds to include as baseline 's1sub' ('' = none)")
    ap.add_argument("--s1-ps", default="0,0.4", help="stage-1 p dirs for the baseline")
    ap.add_argument("--deltas", default="10,50", help="coincidence windows in ms")
    ap.add_argument("--skip-evalagg", action="store_true", help="reuse <out>/agg/<stage>/ from a previous run")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(out, "mpl"))
    deltas = [float(x) for x in a.deltas.split(",")]
    stages = (a.stages.split(",") if a.stages else
              sorted(os.path.basename(d) for d in glob.glob(f"{a.store}/{DEFAULT_STAGE_GLOB}") if os.path.isdir(d)))
    jobs = [(s, a.store) for s in stages]
    if a.s1_seeds:
        jobs.append(("s1sub", build_s1_substore(a.store, out, a.s1_seeds.split(","), a.s1_ps.split(","))))

    del_rows, byte_rows, run_info, sync_rows = [], [], [], []
    fetch_by, interval_by, prob_by = defaultdict(list), defaultdict(list), defaultdict(Counter)
    agg = {}
    errors = []
    anomalies = defaultdict(int)
    anomalies["link_ts_lag_ms"] = []
    for label, root in jobs:
        adir = os.path.join(out, "agg", label)
        if not a.skip_evalagg:
            r = subprocess.run([sys.executable, "-m", "semindex.evalagg", "--store", root, "--stage", label,
                                "--out", adir], capture_output=True, text=True)
            if r.returncode != 0:
                errors.append(f"{label}: evalagg exit {r.returncode}: {r.stderr.strip()[-300:]}")
        agg[label] = {n: read_csv(os.path.join(adir, f"{n}.csv")) for n in ("runs", "summary", "decisions", "discovery")}
        # The run list is evalagg's own view (valid rows of its runs.csv), so
        # every table below pools exactly the runs evalagg pooled even while
        # the batch keeps adding runs.
        ref = {(r.p, r.seed): r.run_dir for r in evalagg.discover_runs(root, label)}
        for row in agg[label]["runs"]:
            if row["valid"] != "valid":
                continue
            rd = ref.get((float(row["p"]), int(row["seed"])))
            try:
                parsed = breakdown2.analyze(rd, del_rows, byte_rows, anomalies, stage=label)
                if parsed is None:
                    continue
                meta = parsed["meta"]
                info, fetched, intervals, probs = per_run_extras(label, meta, rd, parsed)
                run_info.append(info)
                key = (label, float(meta["p"]))
                fetch_by[key] += fetched
                interval_by[key] += intervals
                prob_by[key] += probs
                sync_rows += sync_rows_for_run(label, meta, parsed, deltas)
                del parsed
            except Exception as exc:  # a root-written dir mid-write must not kill the batch report
                errors.append(f"{rd}: {type(exc).__name__}: {exc}")

    write_csv(f"{out}/deliveries.csv", del_rows)
    write_csv(f"{out}/bytes.csv", byte_rows)
    write_csv(f"{out}/run_info.csv", run_info)
    write_csv(f"{out}/sync_pairs.csv", [{k: v for k, v in r.items() if k != "_drift"} for r in sync_rows])

    info_by = defaultdict(list)
    for r in run_info:
        info_by[(r["stage"], r["p"])].append(r)

    # per-(stage,p) building blocks shared by the interval and k tables
    def base_row(label, p):
        """Common columns: validity counts, KM medians, discovery mean, control bytes."""
        A = agg[label]
        summ = next((r for r in A["summary"] if float(r["p"]) == p), {})
        runs = [r for r in A["runs"] if float(r["p"]) == p]
        valid = {(float(r["p"]), r["seed"]) for r in runs if r["valid"] == "valid"}
        disc = [float(r["latency_ms"]) for r in A["discovery"]
                if float(r["p"]) == p and (p, r["seed"]) in valid and r["censored"] in ("False", "false")]
        bpr = [fnum(r["control_bytes_sent_per_record"]) for r in runs if r["valid"] == "valid"]
        infos = info_by[(label, p)]
        k = infos[0]["k"] if infos else (int(runs[0]["k"]) if runs and runs[0]["k"] else None)
        ch = json.loads(infos[0]["choices"]) if infos else None
        # name-length-normalized sent bytes per record, per run, from bytes.csv
        nb = defaultdict(lambda: [0.0, 0.0, 0, 0])
        for b in byte_rows:
            if b["stage"] == label and float(b["p"]) == p:
                t = nb[b["seed"]]
                names_sent = b["state_req_n"] + b["fetch_int_n"] + b["state_rep_n"] + b["fetch_data_n"]
                t[0] += b["logged_sent"] or 0
                t[1] += (b["group_len"] - REF_GROUP_LEN) * names_sent
                t[2] = b["n_records_created"]
        norm = [(lg - corr) / n for lg, corr, n, _ in nb.values() if n]
        return dict(
            stage=label, p=p, k=k, policy=policy_label(k, ch),
            n_runs=len(runs), n_valid=summ.get("n_valid"), n_flagged=summ.get("n_flagged"),
            n_incomplete=summ.get("n_incomplete"), n_unknown=summ.get("n_unknown"),
            n_pairs=summ.get("n_pairs"), n_censored_pairs=summ.get("n_censored_pairs"),
            disc_km_median_ms=fnum(summ.get("discovery_latency_median_ms")),
            disc_km_q1_ms=fnum(summ.get("discovery_latency_q1_ms")),
            disc_km_q3_ms=fnum(summ.get("discovery_latency_q3_ms")),
            disc_mean_uncensored_ms=mean(disc),
            consistency_km_median_ms=fnum(summ.get("consistency_time_median_ms")),
            ctrl_B_sent_per_rec_median=fnum(summ.get("control_bytes_sent_per_record_median")),
            ctrl_B_sent_per_rec_mean=mean([x for x in bpr if x is not None]),
            ctrl_B_sent_per_rec_norm_mean=mean(norm),
        )

    keys = sorted({(lb, float(r["p"])) for lb in agg for r in agg[lb]["summary"]})

    # ---- 1. interval table
    dl_by = defaultdict(list)
    for r in del_rows:
        dl_by[(r["stage"], float(r["p"]))].append(r)
    interval_rows = []
    for label, p in keys:
        row = base_row(label, p)
        dec = [r for r in agg[label]["decisions"] if float(r["p"]) == p and r["kind"] == "state_poll"]
        n_inc = int(row["n_valid"] or 0)
        cnt = next((int(r["decision_count"]) for r in dec), 0)
        freq = {r["field"].split("=", 1)[1]: fnum(r["frequency"]) for r in dec if r["field"].startswith("last_poll_status=")}
        D = dl_by[(label, p)]
        tot = [r["total"] for r in D]
        down = [r["down_pre"] for r in D]
        pw = [r["poll_wait_after_up"] + r["poll_wait_noflap"] for r in D]
        row.update(
            interval_s=row["policy"],
            n_hop_deliveries=len(D),
            hop_total_mean_ms=mean(tot),
            hop_link_down_mean_ms=mean(down),
            hop_poll_wait_mean_ms=mean(pw),
            hop_poll_wait_after_up_mean_ms=mean([r["poll_wait_after_up"] for r in D]),
            hop_poll_wait_noflap_mean_ms=mean([r["poll_wait_noflap"] for r in D]),
            hop_other_mean_ms=(mean(tot) - mean(down) - mean(pw)) if D else None,
            hop_poll_rtt_mean_ms=mean([r["poll_rtt"] for r in D]),
            state_poll_decisions_per_run=cnt / n_inc if n_inc else None,
            **{f"last_poll_{s}": freq.get(s) for s in ("null", "answered", "timed_out", "pending")},
        )
        interval_rows.append(row)
    write_csv(f"{out}/interval_table.csv", interval_rows)

    # ---- 2. k table
    k_rows = []
    for label, p in keys:
        row = base_row(label, p)
        fc = sorted(fetch_by[(label, p)])
        med, q1, q3 = evalagg.median_q1_q3(fc)
        infos = info_by[(label, p)]
        row.update(
            n_fetch_plans=len(fc), fetched_median=med, fetched_q1=q1, fetched_q3=q3,
            fetched_max=max(fc) if fc else None, fetched_mean=mean(fc),
            share_plans_gt1=(sum(1 for x in fc if x > 1) / len(fc)) if fc else None,
            fetched_hist=json.dumps(dict(sorted(Counter(fc).items()))),
            fetch_plans_per_run=len(fc) / len(infos) if infos else None,
            du_disk_mean_bytes=mean([r["du_disk_bytes"] for r in infos]),
            du_disk_max_bytes=max((r["du_disk_bytes"] for r in infos), default=None),
            events_jsonl_mean_bytes=mean([r["events_jsonl_bytes"] for r in infos]),
            events_jsonl_max_bytes=max((r["events_jsonl_bytes"] for r in infos), default=None),
        )
        k_rows.append(row)
    write_csv(f"{out}/k_table.csv", k_rows)

    # ---- 3. sync table
    sg = defaultdict(list)
    for r in sync_rows:
        sg[(r["stage"], r["p"], r["pair_type"], r["delta_ms"])].append(r)
    sync_table = []
    for (label, p, pt, d), rs in sorted(sg.items()):
        nx = sum(r["n_x"] for r in rs)
        hits = sum(r["hits"] for r in rs)
        sb = sum(r["shift_base"] * r["n_x"] for r in rs if r["shift_base"] is not None)
        pb = sum(r["poisson_base"] * r["n_x"] for r in rs if r["poisson_base"] is not None)
        drift = [x for r in rs for x in r["_drift"]]
        fn, fh = sum(r["first_n"] for r in rs), sum(r["first_hits"] for r in rs)
        sn, sh = sum(r["second_n"] for r in rs), sum(r["second_hits"] for r in rs)
        pn, pdn = sum(r["pers_num"] for r in rs), sum(r["pers_den"] for r in rs)
        per_stream = [r["hits"] / r["n_x"] for r in rs if r["n_x"]]
        sync_table.append(dict(
            stage=label, p=p, pair_type=pt, delta_ms=d, n_runs=len({r["seed"] for r in rs}), n_streams=len(rs),
            n_polls=nx, coincidence=hits / nx if nx else None,
            chance_shift=sb / nx if nx else None, chance_poisson=pb / nx if nx else None,
            excess_ratio=(hits / sb) if sb else None,
            coincidence_first_half=fh / fn if fn else None, coincidence_second_half=sh / sn if sn else None,
            persistence=pn / pdn if pdn else None,
            max_stream_coincidence=max(per_stream) if per_stream else None,
            offset_drift_median_ms=statistics.median(drift) if drift else None,
        ))
    write_csv(f"{out}/sync_table.csv", sync_table)

    # ---- 4. interval-choice sanity
    choice_rows = []
    for (label, p), iv in sorted(interval_by.items()):
        infos = info_by[(label, p)]
        ch = json.loads(infos[0]["choices"]) if infos else None
        probs = ";".join(f"{v:g}:{n}" for v, n in sorted(prob_by[(label, p)].items()))
        if ch:
            c = Counter(iv)
            n = len(iv)
            exp = n / len(ch)
            chi2 = sum((c.get(x, 0) - exp) ** 2 / exp for x in ch)
            off = sorted(set(c) - set(ch))
            for x in ch:
                choice_rows.append(dict(stage=label, p=p, interval_s=x, n=c.get(x, 0), share=c.get(x, 0) / n,
                                        expected_share=1 / len(ch), n_total=n, chi2=chi2, df=len(ch) - 1,
                                        off_choice_values=json.dumps(off), action_prob_counts=probs))
        else:
            choice_rows.append(dict(stage=label, p=p, interval_s="jitter", n=len(iv), share=1.0,
                                    expected_share="", n_total=len(iv), chi2="", df="",
                                    off_choice_values=f"min={min(iv):.3f} max={max(iv):.3f}",
                                    action_prob_counts=probs))
    write_csv(f"{out}/choice_table.csv", choice_rows)

    anomalies.pop("link_ts_lag_ms")
    write_summary(out, interval_rows, k_rows, sync_table, choice_rows, dict(anomalies), errors, deltas)


def _f(x, nd=0):
    """Compact cell formatting for the markdown summary."""
    if x is None or x == "":
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def md_table(rows, cols):
    """Markdown table of `rows` with (header, key, decimals) `cols`."""
    out = ["| " + " | ".join(h for h, _, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_f(r.get(k), nd) for _, k, nd in cols) + " |")
    return "\n".join(out)


def write_summary(out, interval_rows, k_rows, sync_table, choice_rows, anomalies, errors, deltas):
    """summary.md: the four tables (selected columns) plus metric definitions; also printed."""
    main_d = max(deltas)
    L = ["# Stage-2 trial (ticket 04) summary", ""]
    # k>1 rows stay in interval_table.csv; the markdown keeps the k=1 rows so the
    # default-policy t2k1 / s1sub rows read as the same-load jitter baseline.
    cens = [f"{r['stage']} p={r['p']:g} ({r['n_censored_pairs']})" for r in interval_rows
            if r.get("n_censored_pairs") not in (None, "", "0")]
    L += ["## 1. Interval / policy table (ms; per (stage, p); k=1 rows, all rows in interval_table.csv)", "",
          md_table([r for r in interval_rows if r["k"] in (1, None)], [
        ("stage", "stage", 0), ("policy", "policy", 0), ("p", "p", 1), ("k", "k", 0),
        ("valid", "n_valid", 0), ("flag", "n_flagged", 0), ("incompl", "n_incomplete", 0),
        ("disc KM med", "disc_km_median_ms", 0), ("disc mean*", "disc_mean_uncensored_ms", 0),
        ("cens", "n_censored_pairs", 0), ("all-nodes med", "consistency_km_median_ms", 0),
        ("hop total", "hop_total_mean_ms", 0), ("link-down", "hop_link_down_mean_ms", 0),
        ("poll wait", "hop_poll_wait_mean_ms", 0), ("(after-up)", "hop_poll_wait_after_up_mean_ms", 0),
        ("other", "hop_other_mean_ms", 0), ("poll rtt", "hop_poll_rtt_mean_ms", 0),
        ("B/rec med", "ctrl_B_sent_per_rec_median", 0), ("B/rec norm", "ctrl_B_sent_per_rec_norm_mean", 0),
        ("polls/run", "state_poll_decisions_per_run", 0), ("answered", "last_poll_answered", 3),
        ("timed_out", "last_poll_timed_out", 3), ("pending", "last_poll_pending", 3), ("null", "last_poll_null", 3),
    ]), "", ("WARNING: censored discovery pairs, so disc mean* (uncensored only) reads low for: " + ", ".join(cens))
          if cens else "(no censored discovery pairs: disc mean* is unbiased)", ""]
    L += ["## 2. k table", "", md_table([dict(r, _du=(r["du_disk_mean_bytes"] or 0) / 1e6, _ev=(r["events_jsonl_mean_bytes"] or 0) / 1e6)
                              for r in k_rows], [
        ("stage", "stage", 0), ("policy", "policy", 0), ("p", "p", 1), ("k", "k", 0), ("valid", "n_valid", 0),
        ("plans", "n_fetch_plans", 0), ("fetched med", "fetched_median", 1), ("Q1", "fetched_q1", 1),
        ("Q3", "fetched_q3", 1), ("max", "fetched_max", 0), ("share>1", "share_plans_gt1", 3),
        ("disc KM med", "disc_km_median_ms", 0), ("B/rec med", "ctrl_B_sent_per_rec_median", 0),
        ("B/rec norm", "ctrl_B_sent_per_rec_norm_mean", 0), ("du MB/run", "_du", 2), ("events MB/run", "_ev", 2),
    ]), ""]
    L += [f"## 3. Poll synchronization (delta = {main_d:g} ms; all deltas in sync_table.csv)", "", md_table(
        [r for r in sync_table if r["delta_ms"] == main_d], [
            ("stage", "stage", 0), ("p", "p", 1), ("pair", "pair_type", 0), ("runs", "n_runs", 0),
            ("polls", "n_polls", 0), ("coinc", "coincidence", 3), ("chance(shift)", "chance_shift", 3),
            ("chance(Poisson)", "chance_poisson", 3), ("excess", "excess_ratio", 2),
            ("1st half", "coincidence_first_half", 3), ("2nd half", "coincidence_second_half", 3),
            ("persist", "persistence", 3), ("max stream", "max_stream_coincidence", 3),
            ("offset drift med", "offset_drift_median_ms", 0)]), ""]
    L += ["## 4. Interval-choice sanity", "", md_table(choice_rows, [
        ("stage", "stage", 0), ("p", "p", 1), ("interval_s", "interval_s", 1), ("n", "n", 0), ("share", "share", 3),
        ("expected", "expected_share", 3), ("chi2", "chi2", 2), ("df", "df", 0), ("off-choice / range", "off_choice_values", 0),
        ("action_prob:count", "action_prob_counts", 0)]), ""]
    L += ["## Definitions", "",
          "- Runs pooled: evalagg's own valid runs (runs.csv valid=valid); DONE-less runs are n_incomplete.",
          "- disc KM med / all-nodes med: evalagg summary.csv KM medians (discovery per (record,node); time until every non-origin node holds the record). disc mean*: plain mean over uncensored discovery pairs only (biased low when cens > 0).",
          "- Per-hop split (breakdown2, stage-1 method): per (record,node), U = upstream holder stored, P = poll whose reply first advertised the seq, S = this node stored. link-down = down time in [U,P]; poll wait = up-time wait in [U,P] (after-up: last link-up..P; no-flap: link up throughout); other = hop total - link-down - poll wait (pre-poll leftover + poll rtt + fetch). P is the latest poll before the reply (state_rx has no poll epoch), so with several polls in flight poll wait can be overstated; poll rtt shown as the check (ms-scale = attribution fine).",
          f"- B/rec med: evalagg control bytes sent per record (window). B/rec norm: per-run mean of the logged sent bytes with every Interest/Data name rebased to a {REF_GROUP_LEN}-char group name (group name lengths differ by stage), / in-window records.",
          "- polls/run: in-window state_poll decisions / valid runs. answered/timed_out/pending/null: decisions.csv last_poll_status frequencies.",
          "- fetched: len(action.fetched) of each in-window fetch_plan decision. du: on-disk bytes (st_blocks) of the run dir; events: sum of the 5 events.jsonl.",
          "- Sync: streams = in-window state_poll times of one (sender->target). same_holder: the two neighbors polling the same holder (h0&h2->h1 etc.); link_reverse: A->B vs B->A over one link. For stream X vs Y: coinc = fraction of X polls with a Y poll within +-delta; chance(shift) = same after circularly shifting Y by 200 seeded uniform offsets in [0,W) (keeps Y's own periodicity); chance(Poisson) = 1-exp(-2*delta*n_Y/W); excess = coinc/chance(shift) (1 = chance, >>1 = synchronized); 1st/2nd half = coinc by window half; persist = P(next X poll coincident | this one coincident); max stream = most-coincident single stream; offset drift = median |change| of the signed X-to-nearest-Y offset between consecutive X polls (~0 = phase-locked even if the offset is > delta). Absolute phases drift (processing delay accumulates), so lock-step is judged on the relative offset (drift, persist), not on a phase mod the interval.",
          "- Choice sanity: every state_poll decision (whole run), counts per interval_s vs uniform; chi2 = sum (obs-exp)^2/exp (df = choices-1; 5% critical: df2 5.99, df4 9.49). action_prob:count lists distinct values.",
          "", f"anomalies (breakdown2): {anomalies}", f"errors: {errors or 'none'}"]
    text = "\n".join(L) + "\n"
    with open(f"{out}/summary.md", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
