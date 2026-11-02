"""Concatenate the evalagg summary.csv of several stages into one CSV with a leading `stage` column.

Usage: concat_summary.py <agg_dir> stage [stage ...] > main3_summary.csv
Rows keep evalagg's text verbatim, stages in argv order. The csv module's default CRLF line ends
are kept on purpose: evidence/05 main_summary.csv was written that way, and this reproduces it
byte for byte from the s2 agg dirs.
"""
import csv
import sys

agg, stages = sys.argv[1], sys.argv[2:]
w = None
for st in stages:
    with open(f"{agg}/{st}/summary.csv", newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        if w is None:
            w = csv.writer(sys.stdout)
            w.writerow(["stage"] + hdr)
        for row in r:
            w.writerow([st] + row)
