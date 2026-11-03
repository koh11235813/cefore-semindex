#!/bin/sh
# Aggregate one stored stage read-only with main 3d89cfe and with the branch, then compare every CSV.
set -e
P=${P:?set P to a scratch directory holding main/src (git archive 3d89cfe src)}
W=$(cd "$(dirname "$0")/../../../../.." && pwd)
PY=/home/lab_shared/cefore-semindex/.venv/bin/python3
STAGE=${1:-t6c}
rm -rf $P/agg/$STAGE && mkdir -p $P/agg/$STAGE
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$P/main/src $PY -m semindex.evalagg --store /home/lab_shared/semindex-eval --stage $STAGE --out $P/agg/$STAGE/main
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$W/src $PY -m semindex.evalagg --store /home/lab_shared/semindex-eval --stage $STAGE --out $P/agg/$STAGE/branch
cd $P/agg/$STAGE
for f in main/*.csv; do
  n=$(basename $f)
  if [ $n = runs.csv ]; then continue; fi
  cmp main/$n branch/$n && echo "$n IDENTICAL"
done
[ -e main/polls.csv ] || echo "polls.csv: new (main has none), $(($(wc -l < branch/polls.csv) - 1)) rows"
# runs.csv: drop the branch's last column with the csv module (cells hold quoted JSON), then compare.
$PY -I - branch/runs.csv main/runs.csv <<'EOP'
import csv, sys
branch = list(csv.reader(open(sys.argv[1], newline="")))
main = list(csv.reader(open(sys.argv[2], newline="")))
assert branch[0][-1] == "state_poll_policy", branch[0][-1]
assert all(row[-1] == "" for row in branch[1:]), "non-empty state_poll_policy cell"
assert [row[:-1] for row in branch] == main
print(f"runs.csv IDENTICAL except the new last column state_poll_policy (empty in all {len(branch) - 1} rows)")
EOP
