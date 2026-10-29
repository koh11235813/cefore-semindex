"""Print the ope05 CSVs as Markdown tables, straight from the CSVs (no recomputation)."""
import argparse
import csv
import os
from collections import defaultdict

_ap = argparse.ArgumentParser()
_ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)),
                 help="directory holding the ope05 CSVs (default: this script's directory)")
D = _ap.parse_args().dir


def rd(name):
    """All rows of CSV `name` in the input dir, as dicts of raw strings."""
    with open(f"{D}/{name}") as f:
        return list(csv.DictReader(f))


def f(x, nd=0):
    """Fixed-point cell with `nd` decimals; an empty CSV cell (NaN upstream) shows as '-'."""
    if x in ("", None):
        return "-"
    v = float(x)
    return f"{v:.{nd}f}"


def pct(x):
    """Signed whole-percent cell for a relative error; empty shows as '-'."""
    return "-" if x in ("", None) else f"{100 * float(x):+.0f}%"


def table(header, rows):
    """Markdown pipe table from a header list and row lists."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


hb = rd("per_hop_B.csv")
print("## T1 sanity: B stages at p=0, per hop (per_hop_B.csv)\n")
rows = []
for r in hb:
    if float(r["p"]) == 0:
        rows.append([r["stage"], r["hop"], r["n_recordhops"], f(r["wait_ms"]), f(r["wait_over_chosen"], 2),
                     f(r["ingap_wait_ms"]), f(r["post_ms"]), f(r["fetch_ms"]), f(r["poll_B_per_recordhop"]),
                     f(r["fetch_B_per_recordhop"])])
print(table(["stage", "hop", "record-hops", "wait ms", "wait/chosen", "in-gap ms", "post-gap ms", "fetch ms",
             "poll B/rh", "fetch B/rh"], rows))

ck = rd("checks.csv")
c = defaultdict(dict)
for r in ck:
    c[(r["stage"], r["p"])][r["check"]] = float(r["value"])
print("\n## T2 checks (checks.csv)\n")
rows = []
for st in ["s2b0p5", "s2b1p359", "s2b3p695", "s2b10p043", "s2c", "s2a"]:
    a = c[(st, "all")]
    rows.append([st, int(a.get("runs_valid", 0)), int(a.get("runs_invalid", 0)), int(a.get("exact_n", 0)),
                 f(a.get("exact_maxerr", 0), 1), f(a.get("exact_share_within_1ms", 0), 4),
                 int(a.get("exc_holder_not_upstream", 0)), int(a.get("exc_fp_neighbor_not_upstream", 0)),
                 int(a.get("hop_upstream_censored", 0)), int(a.get("hop_gap_outside_window", 0)),
                 int(a.get("stored_without_learn", 0)), int(a.get("neg_fetch", 0)), int(a.get("multi_life_nodes", 0))])
print(table(["stage", "valid runs", "invalid", "pairs checked", "max |err| ms", "share <=1 ms", "holder!=upstream",
             "fetch nbr!=upstream", "hops w/ upstream censored", "hops in out-of-window gap", "stored w/o learn",
             "neg fetch", "multi-life nodes"], rows))
print()
rows = []
for st in ["s2b0p5", "s2b1p359", "s2b3p695", "s2b10p043", "s2c", "s2a"]:
    for p in ["0.0", "0.1", "0.2", "0.3", "0.4"]:
        a = c[(st, p)]
        pk = p if p != "0.0" else "0"
        b = c[(st, pk)] if (st, pk) in c else {}
        rows.append([st, p, f(100 * a.get("rule_hi_share", 0), 2), f(100 * a.get("rule_extra_share", 0), 2),
                     f(100 * a.get("rule_fp_before_reply_share", 0), 2), f(100 * a.get("rule_fp_only_share", 0), 3),
                     f(100 * a.get("rule_none_share", 0), 2), f(100 * b.get("spill_share_of_wait", 0), 1),
                     f(100 * b.get("post_gap_share_of_wait", 0), 1), int(b.get("hops_censored", 0)),
                     int(b.get("hops_uncensored", 0))])
print(table(["stage", "p", "% learned via hi", "% via extra (recon)", "% fp before reply", "% fp only",
             "% never learned", "spill % of wait", "post-gap % of wait", "hops censored", "hops uncensored"], rows))

pv = rd("predicted_vs_B.csv")
print("\n## T3a per-hop components, s2a-predicted (a) vs B-measured (b), hop-averaged relative errors (predicted_vs_B.csv)\n")
rows = []
for arm in ["0.5", "1.359", "3.695", "10.043"]:
    for p in ["0", "0.1", "0.2", "0.3", "0.4"]:
        rr = [r for r in pv if r["arm_s"] == arm and r["p"] == p]
        if not rr:
            continue
        mean = lambda k: sum(float(r[k]) for r in rr) / len(rr)  # noqa: E731
        rows.append([arm, p, sum(int(r["a_n_recordhops"]) for r in rr), f(mean("a_wait_ms")), f(mean("b_wait_ms")),
                     pct(mean("relerr_wait")), pct(mean("relerr_ingap_wait")), f(mean("a_post_ms")),
                     f(mean("b_post_ms")), f(mean("a_fetch_ms")), f(mean("b_fetch_ms")), pct(mean("relerr_fetch")),
                     f(mean("a_batch"), 1), f(mean("b_batch"), 1),
                     f(mean("a_poll_B_per_rh")), f(mean("b_poll_B_per_rh")), pct(mean("relerr_poll_B"))])
print(table(["arm s", "p", "s2a rh (4 hops)", "a wait", "b wait", "err wait", "err in-gap", "a post", "b post",
             "a fetch", "b fetch", "err fetch", "a batch", "b batch", "a poll B/rh", "b poll B/rh", "err poll B"], rows))
print("\n(values are means over hops 1-4 of the per-hop numbers; ms unless noted)\n")

print("## T3b end-to-end, node 4 hops from origin (predicted_vs_B.csv, hop=4)\n")
rows = []
for r in pv:
    if r["hop"] == "4":
        rows.append([r["arm_s"], r["p"], f(r["pred_e2e_ms"]) + " ± " + f(r["pred_e2e_se"]),
                     f(r["b_e2e_sum_of_hop_means_ms"]) + " ± " + f(r["b_e2e_sum_se"]), f(r["b_e2e_direct_ms"]),
                     pct(r["relerr_e2e_vs_sum"]), pct(r["relerr_e2e_vs_direct"]), r["b_e2e_pairs_censored"],
                     f(100 * float(r["cum_censored_share_a"]), 2), f(100 * float(r["cum_censored_share_b"]), 2),
                     f(100 * float(r["e2e_censored_share_a"]), 2), f(100 * float(r["e2e_censored_share_b"]), 2),
                     "UNRELIABLE" if r["unreliable"] == "1" else ""])
print(table(["arm s", "p", "predicted (s2a) ± SE", "B sum of hop means ± SE", "B direct", "err vs sum",
             "err vs direct", "B pairs censored", "s2a hops cens %", "B hops cens %",
             "s2a 4-hop pairs cens %", "B 4-hop pairs cens %", "flag"], rows))

print("\n## T3c end-to-end relative error by hop distance (pred vs B direct)\n")
rows = []
for arm in ["0.5", "1.359", "3.695", "10.043"]:
    for p in ["0", "0.1", "0.2", "0.3", "0.4"]:
        rr = {r["hop"]: r for r in pv if r["arm_s"] == arm and r["p"] == p}
        rows.append([arm, p] + [pct(rr[h]["relerr_e2e_vs_direct"]) for h in "1234"])
print(table(["arm s", "p", "H=1", "H=2", "H=3", "H=4"], rows))

sb = rd("state_buckets.csv")
for P in ["all", "0", "0.2", "0.4"]:
    print(f"\n## T4 state buckets, p={P} (state_buckets.csv)\n")
    rows = []
    for r in sb:
        if r["p"] != P:
            continue
        rows.append([r["last_poll_status"], r["cto_bucket"], r["arm_s"], r["n_gaps"], r["n_recordhops"],
                     f(r["wait_ms"]) + " ± " + f(r["wait_se"]), f(r["poll_B_per_recordhop"]), f(r["backlog_rate"], 2),
                     f(100 * float(r["prev_arm_share_0.5"] or 0)) + "/" + f(100 * float(r["prev_arm_share_1.359"] or 0))
                     + "/" + f(100 * float(r["prev_arm_share_3.695"] or 0)) + "/"
                     + f(100 * float(r["prev_arm_share_10.043"] or 0)),
                     r["rank_wait"], r["rank_l1"], r["rank_l10"]])
    print(table(["last status", "cto", "arm", "gaps", "record-hops", "wait ms ± SE", "poll B/rh", "backlog",
                 "prev arm % (0.5/1.4/3.7/10)", "rank wait", "rank λ=1", "rank λ=10"], rows))
