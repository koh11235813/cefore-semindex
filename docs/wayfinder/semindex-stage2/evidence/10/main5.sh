#!/bin/bash
# Stage-2 second outage shape: each link's down periods uniform on [8, 12] s, up periods exponential
# with mean 10(1-p)/p s, p = down-time fraction (evalgen --disconnect-duration-s 8 12, map
# 「ほかの切断の形（2026-11-11 決定）」). Same agent settings as main4.sh's s5b* and policies, stage prefix s6.
#   B: each interval fixed (j=0.1). Policies unchanged from s5short / s5wait.
# p 0.2 and 0.4 for all six stages first (the only p where the 5 s slot stages exist to compare
# against), then p 0.5 and 0.6, so an interrupted run still leaves the comparable cells complete.
# All runs from one detached, clean checkout nobody edits. One --stage per setting (the outage
# shape, choices and policy are not in the store path). Resumable: evalrun skips DONE runs and does
# not compare settings, so never reuse a stage name.
# Run as root: sudo bash main5.sh
set -u
W=/home/lab_shared/wt/semindex-main5
R=/home/lab_shared/cefore-emu-sslab
D="--disconnect-duration-s 8 12"
run() { env PYTHONPATH=$W/src /home/lab_shared/cefore-semindex/.venv/bin/python3 -m semindex.evalrun \
  --store /home/lab_shared/semindex-eval --recefore $R "$@"; echo "== exit $? : $*"; }
echo "== commit $(git -C $W rev-parse HEAD) dirty=$(git -C $W status --porcelain | wc -l)" \
  "recefore $(git -c safe.directory=$R -C $R rev-parse HEAD) dirty=$(git -c safe.directory=$R -C $R status --porcelain | wc -l)" \
  "start $(date -Is)"
for P in "0.2 0.4" "0.5 0.6"; do
  for c in 0.5 1.359 3.695 10.043; do
    run --stage "s6b${c/./p}" --p $P --seeds 1-20 --k 4 --store-synchronous NORMAL $D --interval-choices $c --choice-jitter 0.1
  done
  run --stage s6short --p $P --seeds 1-20 --k 4 --store-synchronous NORMAL $D --poll-policy '{"intervals_s": [10.043, 3.695], "jitter_frac": [0.1, 0.1]}'
  run --stage s6wait --p $P --seeds 1-20 --k 4 --store-synchronous NORMAL $D --poll-policy '{"intervals_s": [10.043, 5.0], "jitter_frac": [0.1, 0.0]}'
done
echo "== main5 finished $(date -Is)"
