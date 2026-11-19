#!/bin/bash
# The runs of evidence/17 (spec.md): 20 new seeds for the two rules and the two fixed arms of evidence/16,
# and a third fixed arm at 6.0 s on seeds 1-40. Same outage shape and agent settings as main6.sh, p 0.4.
# New stage names (prefix s8): the stages evidence/10 and evidence/16 were computed from must keep exactly
# the runs they had, and evalrun does not compare settings, so a stage name is never reused.
# All runs from one detached, clean checkout nobody edits. Resumable: evalrun skips DONE runs.
# PYTHONDONTWRITEBYTECODE: root would otherwise leave __pycache__ in the checkout that its owner cannot
# delete (2026-11-19: removing the evidence/16 run checkout needed root for that reason).
# Run as root: sudo bash main7.sh
set -u
W=/home/lab_shared/wt/semindex-main7
R=/home/lab_shared/cefore-emu-sslab
C="--p 0.4 --k 4 --store-synchronous NORMAL --disconnect-duration-s 8 12"
RULE='"i_min_s": 3.695, "factor": 2, "i_max_s": 10.043, "jitter_frac": 0.1'
run() { env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$W/src /home/lab_shared/cefore-semindex/.venv/bin/python3 -m semindex.evalrun \
  --store /home/lab_shared/semindex-eval --recefore $R "$@"; echo "== exit $? : $*"; }
echo "== commit $(git -C $W rev-parse HEAD) dirty=$(git -C $W status --porcelain | wc -l)" \
  "recefore $(git -c safe.directory=$R -C $R rev-parse HEAD) dirty=$(git -c safe.directory=$R -C $R status --porcelain | wc -l)" \
  "start $(date -Is)"
run --stage s8t37 $C --seeds 21-40 --poll-rule "{$RULE, \"on_timeout\": \"keep\"}"
run --stage s8t37r $C --seeds 21-40 --poll-rule "{$RULE, \"on_timeout\": \"reset\"}"
run --stage s8b3p695 $C --seeds 21-40 --interval-choices 3.695 --choice-jitter 0.1
run --stage s8b10p043 $C --seeds 21-40 --interval-choices 10.043 --choice-jitter 0.1
run --stage s8b6p0 $C --seeds 1-40 --interval-choices 6.0 --choice-jitter 0.1
echo "== main7 finished $(date -Is)"
