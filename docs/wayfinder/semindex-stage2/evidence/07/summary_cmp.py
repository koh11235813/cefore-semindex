"""Side by side, per stage x p: the FULL main run (s2*) vs the NORMAL re-collection (s4*).

Reads two stage-prefixed summary CSVs (concat_summary.py output: evidence/05 main_summary.csv for
s2, evidence/07 main3_summary.csv for s4) and prints a markdown table of the evalagg medians of
discovery latency and consistency time, control bytes sent per record, and the censored-pair share.
Usage: summary_cmp.py <s2 summary csv> <s4 summary csv> > summary_cmp.md
"""
import csv, sys
old_path, new_path = sys.argv[1], sys.argv[2]
old = {(r["stage"][2:], float(r["p"])): r for r in csv.DictReader(open(old_path))}
new = {(r["stage"][2:], float(r["p"])): r for r in csv.DictReader(open(new_path))}
cols = [("discovery_latency_median_ms", "disc med ms"), ("consistency_time_median_ms", "cons med ms"),
        ("control_bytes_sent_per_record_median", "ctrl B/rec"), ("pairs_censored_fraction", "cens frac")]
hdr = ["stage", "p", "n s2/s4"] + [f"{h} s2 / s4 (Δ%)" for _, h in cols]
print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
for st in ["c", "b0p5", "b1p359", "b3p695", "b10p043", "a"]:
    for p in [0.0, 0.1, 0.2, 0.3, 0.4]:
        o, n = old[(st, p)], new[(st, p)]
        cells = [st, f"{p:g}", f"{o['n_valid']}/{n['n_valid']}"]
        for k, _ in cols:
            a, b = float(o[k]), float(n[k])
            if k == "pairs_censored_fraction":
                cells.append(f"{a*100:.2f}% / {b*100:.2f}%")
            else:
                cells.append(f"{a:.0f} / {b:.0f} ({(b-a)/a*100:+.0f}%)")
        print("| " + " | ".join(cells) + " |")
