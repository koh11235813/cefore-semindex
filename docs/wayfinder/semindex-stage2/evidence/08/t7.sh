#!/bin/bash
# Trial runs of the two failure-reactive state-poll policies (re-decision on answer/timeout),
# from a detached, clean checkout nobody edits, before the full A/B/C re-collection.
# Run as root: sudo bash t7.sh
set -u
W=/home/lab_shared/wt/semindex-t7
R=/home/lab_shared/cefore-emu-sslab
SPY=/home/lab_shared/cefore-semindex/.venv/bin/python3
echo "== commit $(git -C $W rev-parse HEAD) dirty=$(git -C $W status --porcelain | wc -l) start $(date -Is)"
run() { env PYTHONPATH=$W/src $SPY -m semindex.evalrun --store /home/lab_shared/semindex-eval --recefore $R "$@"; echo "== exit $? : $*"; }
run --stage t7short --p 0 0.2 --seeds 1-2 --k 4 --store-synchronous NORMAL --poll-policy '{"intervals_s": [10.043, 3.695], "jitter_frac": [0.1, 0.1]}'
run --stage t7wait --p 0 0.2 --seeds 1-2 --k 4 --store-synchronous NORMAL --poll-policy '{"intervals_s": [10.043, 5.0], "jitter_frac": [0.1, 0.0]}'
echo "== t7 finished $(date -Is)"
