#!/bin/bash
# Q24 check: does SQLite synchronous=NORMAL remove the fetcher's tens-of-ms per-record handling?
# FULL baseline passes no flag (exactly today's configs); NORMAL passes --store-synchronous NORMAL.
# One --stage per setting (the flag is not in the store path).
W=/home/lab_shared/wt/semindex-storesync
run() { env PYTHONPATH=$W/src /home/lab_shared/cefore-semindex/.venv/bin/python3 -m semindex.evalrun \
  --store /home/lab_shared/semindex-eval --recefore /home/lab_shared/cefore-emu-sslab "$@"; echo "== exit $? : $*"; }
echo "== commit $(git -C $W rev-parse HEAD) start $(date -Is)"
MIX="--interval-choices 0.5 1.359 3.695 10.043 --choice-jitter 0.1"
run --stage s3fk8   --p 0   --seeds 1-3 --k 8
run --stage s3nk8   --p 0   --seeds 1-3 --k 8 --store-synchronous NORMAL
run --stage s3fmix  --p 0   --seeds 1-3 --k 4 $MIX
run --stage s3nmix  --p 0   --seeds 1-3 --k 4 $MIX --store-synchronous NORMAL
run --stage s3nc32  --p 0.4 --seeds 1-3 --k 4 --max-in-flight 32 --store-synchronous NORMAL
echo "== micro24 finished $(date -Is)"
