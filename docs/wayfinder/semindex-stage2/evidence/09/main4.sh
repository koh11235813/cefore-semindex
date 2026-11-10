#!/bin/bash
# Stage-2 re-collection on the re-decision code (merge commit of the re-decision PR), then the two
# failure-reactive state-poll policies, all from one detached, clean checkout nobody edits.
# A/B/C: same settings as main3.sh (s4*), stage prefix s5. k=4, cap default 8, p in {0,.1,.2,.3,.4}.
#   C: stage-1 default policy, 20 seeds. B: each interval fixed (j=0.1), 20 seeds. A: uniform mix of the 4, 100 seeds.
# Policies (taf-grant-2026/analysis/value-check/policies.py), p in {0,.2,.4}, 20 seeds:
#   s5short: 10.043 s after an answer, 3.695 s after a failure, j=0.1 on both rows.
#   s5wait:  10.043 s after an answer (j=0.1), exactly 5 s after a failure (j=0).
# Order C -> B -> A -> policies. One --stage per setting (choices/policy are not in the store path).
# Resumable: evalrun skips DONE runs, and does not compare settings, so never reuse a stage name.
# Run as root: sudo bash main4.sh
set -u
W=/home/lab_shared/wt/semindex-main4
R=/home/lab_shared/cefore-emu-sslab
P="0 0.1 0.2 0.3 0.4"
run() { env PYTHONPATH=$W/src /home/lab_shared/cefore-semindex/.venv/bin/python3 -m semindex.evalrun \
  --store /home/lab_shared/semindex-eval --recefore $R "$@"; echo "== exit $? : $*"; }
echo "== commit $(git -C $W rev-parse HEAD) dirty=$(git -C $W status --porcelain | wc -l)" \
  "recefore $(git -c safe.directory=$R -C $R rev-parse HEAD) dirty=$(git -c safe.directory=$R -C $R status --porcelain | wc -l)" \
  "start $(date -Is)"
run --stage s5c --p $P --seeds 1-20 --k 4 --store-synchronous NORMAL
for c in 0.5 1.359 3.695 10.043; do run --stage "s5b${c/./p}" --p $P --seeds 1-20 --k 4 --store-synchronous NORMAL --interval-choices $c --choice-jitter 0.1; done
run --stage s5a --p $P --seeds 1-100 --k 4 --store-synchronous NORMAL --interval-choices 0.5 1.359 3.695 10.043 --choice-jitter 0.1
run --stage s5short --p 0 0.2 0.4 --seeds 1-20 --k 4 --store-synchronous NORMAL --poll-policy '{"intervals_s": [10.043, 3.695], "jitter_frac": [0.1, 0.1]}'
run --stage s5wait --p 0 0.2 0.4 --seeds 1-20 --k 4 --store-synchronous NORMAL --poll-policy '{"intervals_s": [10.043, 5.0], "jitter_frac": [0.1, 0.0]}'
echo "== main4 finished $(date -Is)"
