"""Recount, written apart from attrib.py, of a poll's failure rate and of its continuation, by arm.

A poll k from n toward m "yields" if some state_rx from m reaches n in [t_k, t_{k+1}).
P(fail) = share of polls that do not yield; c = P(poll k+1 fails | poll k failed). Both are split by
the arm chosen at poll k (its interval is the gap to poll k+1); a B stage has one arm, labelled by
the stage name.
Every logged poll is counted, the lead-in before t0 + 10 s included. No link goes down there, so
P(fail) comes out about 8% (relative) below seq_model.py's `dec P(fail)`, which counts only polls
in the window [t0 + 10 s, t0 + 130 s]; restricted to that window this script gave the same values
(checked at p = 0.2, 2026-11-02).
c is unaffected, since every failure falls inside the window.
Usage: contcheck.py <store> <p-dir> stage [stage ...]   e.g. p0.2 s4b0p5 s4b1p359 s4b3p695 s4b10p043 s4a
"""
import bisect, glob, json, sys, collections
root, pdir = sys.argv[1], sys.argv[2]
for stage in sys.argv[3:]:
    nf = collections.Counter(); npoll = collections.Counter(); cc = collections.Counter(); cn = collections.Counter()
    for f in glob.glob(f"{root}/{stage}/{pdir}/*/debug/node_dirs/h*/semindex/events.jsonl"):
        polls = collections.defaultdict(list); rx = collections.defaultdict(list)
        for l in open(f):
            e = json.loads(l)
            if e["event"] == "decision" and e["kind"] == "state_poll":
                polls[e["action"]["neighbor"]].append((e["ts_ms"], e["action"]["interval_s"]))
            elif e["event"] == "state_rx":
                rx[e["neighbor"]].append(e["ts_ms"])
        for m, lst in polls.items():
            lst.sort(); r = sorted(rx[m])
            fail = []
            for (t, iv), (tn, _) in zip(lst, lst[1:]):
                k = bisect.bisect_left(r, t)
                fail.append((not (k < len(r) and r[k] < tn), iv))
            for (fk, ivk), (fk1, _iv1) in zip(fail, fail[1:]):
                arm = round(ivk, 3) if stage.endswith("a") else stage
                npoll[arm] += 1; nf[arm] += fk
                if fk:
                    cn[arm] += 1; cc[arm] += fk1
    for arm in sorted(npoll, key=str):
        print(f"{stage} {pdir} arm={arm}: polls={npoll[arm]} P(fail)={nf[arm]/npoll[arm]:.3f} c=P(next fails|failed)={cc[arm]/cn[arm]:.3f} (n={cn[arm]})")
