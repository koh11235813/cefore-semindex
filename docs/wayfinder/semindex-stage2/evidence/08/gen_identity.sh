#!/bin/sh
# Generate the A/B/C evalgen runs and the gen experiments with main 3d89cfe and with the branch,
# into the same --out path per side (so output_dir and config paths match), then diff byte for byte.
set -e
P=${P:?set P to a scratch directory holding main/src (git archive 3d89cfe src)}
W=$(cd "$(dirname "$0")/../../../../.." && pwd)
PY=/home/lab_shared/cefore-semindex/.venv/bin/python3
cd $P
rm -rf gen && mkdir gen
for side in main branch; do
  if [ $side = main ]; then SRC=$P/main/src; else SRC=$W/src; fi
  for st in A B C; do
    case $st in
      A) extra="--interval-choices 0.5 1.359 3.695 10.043 --choice-jitter 0.1";;
      B) extra="--interval-choices 0.5 --choice-jitter 0.1";;
      C) extra="";;
    esac
    rm -rf $P/gen/work
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SRC $PY -m semindex.evalgen --p 0.2 --seed 1 --k 4 --stage s$st --out $P/gen/work $extra > /dev/null
    mv $P/gen/work $P/gen/$side.$st
  done
  for e in t1 t2a t2b t2c; do
    rm -rf $P/gen/work
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SRC $PY -m semindex.gen $W/experiments/$e.json $P/gen/work > /dev/null
    mv $P/gen/work $P/gen/$side.$e
  done
done
# The only path that differs between the sides is SEMINDEX_SRC_DIR (gen.py, __file__-relative),
# baked into each YAML's host_apps env PYTHONPATH: main's is the git-archive copy.
grep -rl "$P/main/src" $P/gen/main.* | xargs sed -i "s#$P/main/src#$W/src#g"
for d in A B C t1 t2a t2b t2c; do
  diff -r $P/gen/main.$d $P/gen/branch.$d && echo "$d IDENTICAL ($(find $P/gen/branch.$d -type f | wc -l) files)"
done
