"""In-situ fetcher-side processing time of one received record, per step.

Within one Agent.step(): record_stored carries poll()'s post-receive `now`
(read BEFORE on_record's decode/sha256/SQLite INSERT+COMMIT), while an
on_tick fetch_plan decision in the same step carries transport.now() read
AFTER poll() returned, i.e. after the commit. Engine events are drained in
emit order, so a record_stored line immediately followed by a fetch_plan
decision line (no state_rx between -- on_state plans with poll's own `now`)
gives  N1 - N0 = time spent inside poll() after receive(): on_record
(decode + sha256 + SQLite commit + get_progress) + _check_timeouts.

Also reports the next record_stored from the same holder after N1 (time the
loop then waited for / immediately got the next Data): wait = N_next - N1.

Usage: python3 commit_time.py <eval_root> stage [stage ...]
"""
import collections, glob, json, statistics, sys
root = sys.argv[1]
for stage in sys.argv[2:]:
    commit, wait = [], []
    for p in glob.glob(f"{root}/{stage}/*/*/debug/node_dirs/h*/semindex/events.jsonl"):
        ev = [json.loads(l) for l in open(p)]
        for i in range(len(ev) - 1):
            a, b = ev[i], ev[i + 1]
            if a["event"] == "record_stored" and b["event"] == "decision" and b["kind"] == "fetch_plan" \
                    and b["action"]["neighbor"] == a["holder"]:
                d = b["ts_ms"] - a["ts_ms"]
                if 0 <= d < 2000:
                    commit.append(d)
                    for c in ev[i + 2:]:
                        if c["event"] == "record_stored" and c["holder"] == a["holder"]:
                            if c["ts_ms"] - b["ts_ms"] < 2000:
                                wait.append(c["ts_ms"] - b["ts_ms"])
                            break
    def s(xs):
        """Count and nearest-rank quantiles (p10/median/p90/p99) of a list of ms values."""
        xs = sorted(xs)
        q = lambda f: xs[int(f * (len(xs) - 1))]
        return f"n={len(xs)} p10={q(.1)} median={q(.5)} p90={q(.9)} p99={q(.99)}"
    hist = collections.Counter(min(x // 4, 30) for x in commit)
    print(f"== {stage}\n  commit-in-step (N1-N0): {s(commit)}\n  then wait for next Data from same holder: {s(wait)}")
    print("  commit hist (4ms bins): " + " ".join(f"{4*k}:{hist[k]}" for k in sorted(hist)))
