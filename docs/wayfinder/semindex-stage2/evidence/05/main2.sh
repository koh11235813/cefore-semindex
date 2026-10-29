#!/bin/bash
# Stage-2 main run: k=4, cap default 8, p in {0,.1,.2,.3,.4}. Order C -> B -> A.
# C: stage-1 default policy, 20 seeds. B: each interval fixed (j=0.1), 20 seeds. A: uniform mix of the 4, 100 seeds.
# One --stage per setting (k/choices/cap are not in the store path). Resumable: evalrun skips DONE runs.
W=/home/lab_shared/wt/semindex-s2cap
P="0 0.1 0.2 0.3 0.4"
run() { env PYTHONPATH=$W/src /home/lab_shared/cefore-semindex/.venv/bin/python3 -m semindex.evalrun \
  --store /home/lab_shared/semindex-eval --recefore /home/lab_shared/cefore-emu-sslab "$@"; echo "== exit $? : $*"; }
echo "== commit $(git -C $W rev-parse HEAD) start $(date -Is)"
run --stage s2c --p $P --seeds 1-20 --k 4
for c in 0.5 1.359 3.695 10.043; do run --stage "s2b${c/./p}" --p $P --seeds 1-20 --k 4 --interval-choices $c --choice-jitter 0.1; done
run --stage s2a --p $P --seeds 1-100 --k 4 --interval-choices 0.5 1.359 3.695 10.043 --choice-jitter 0.1
echo "== main2 finished $(date -Is)"
