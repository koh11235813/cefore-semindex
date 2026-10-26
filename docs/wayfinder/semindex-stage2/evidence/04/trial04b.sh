#!/bin/bash
# Stage-2 trial, re-run with choices-mode jitter j=0.1 (default) at 0.5*e^n.
# t2i* (no jitter) stay as they are for comparison. One --stage per setting.
W=/home/lab_shared/wt/semindex-s2jitter
run() { env PYTHONPATH=$W/src /home/lab_shared/cefore-semindex/.venv/bin/python3 -m semindex.evalrun \
  --store /home/lab_shared/semindex-eval --recefore /home/lab_shared/cefore-emu-sslab "$@"; echo "== exit $? : $*"; }
for c in 0.5 1.359 3.695 10.043; do run --stage "t2j${c/./p}" --p 0 0.4 --seeds 1-3 --interval-choices $c --choice-jitter 0.1; done
run --stage t2jmix --p 0 0.4 --seeds 1-3 --interval-choices 0.5 1.359 3.695 10.043 --choice-jitter 0.1
echo "== trial04b finished"
