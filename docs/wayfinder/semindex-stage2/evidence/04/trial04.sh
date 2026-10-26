#!/bin/bash
# Stage-2 trial (ticket 04). One --stage per setting: k and choices are not in the store path.
W=/home/lab_shared/wt/semindex-s2tickets
run() { env PYTHONPATH=$W/src /home/lab_shared/cefore-semindex/.venv/bin/python3 -m semindex.evalrun \
  --store /home/lab_shared/semindex-eval --recefore /home/lab_shared/cefore-emu-sslab "$@"; echo "== exit $? : $*"; }
for c in 0.5 1 2 4 8; do run --stage "t2i${c/./p}" --p 0 0.4 --seeds 1-3 --interval-choices $c; done
for k in 1 2 4 8; do run --stage "t2k$k" --p 0.4 --seeds 1-3 --k $k; done
run --stage t2mix --p 0 0.4 --seeds 1-3 --interval-choices 0.5 1 2 4 8
echo "== trial04 finished"
