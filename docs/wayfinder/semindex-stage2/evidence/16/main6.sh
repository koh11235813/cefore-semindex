#!/bin/bash
# One testbed check of the per-neighbor state-poll rule (spec.md). Same outage shape and agent settings as
# main5.sh's s6 stages (../10/main5.sh), p 0.4 only, stage prefix s7.
#   rerun: three existing settings, seed 1, with the code that carries the rule, under new stage names, so
#          that a change of behaviour of the existing policies shows before the rule runs are read.
#   rules: the two rules, seeds 1-20. Start it only after the rerun check of spec.md pass line 4 holds.
# All runs from one detached, clean checkout nobody edits. One --stage per setting (the settings are not in
# the store path). Resumable: evalrun skips DONE runs and does not compare settings, so never reuse a stage
# name.
# Run as root: sudo bash main6.sh rerun   then   sudo bash main6.sh rules
set -u
W=/home/lab_shared/wt/semindex-main6
R=/home/lab_shared/cefore-emu-sslab
C="--p 0.4 --k 4 --store-synchronous NORMAL --disconnect-duration-s 8 12"
run() { env PYTHONPATH=$W/src /home/lab_shared/cefore-semindex/.venv/bin/python3 -m semindex.evalrun \
  --store /home/lab_shared/semindex-eval --recefore $R "$@"; echo "== exit $? : $*"; }
echo "== commit $(git -C $W rev-parse HEAD) dirty=$(git -C $W status --porcelain | wc -l)" \
  "recefore $(git -c safe.directory=$R -C $R rev-parse HEAD) dirty=$(git -c safe.directory=$R -C $R status --porcelain | wc -l)" \
  "start $(date -Is)"
case "${1:-}" in
  rerun)
    run --stage s7short $C --seeds 1 --poll-policy '{"intervals_s": [10.043, 3.695], "jitter_frac": [0.1, 0.1]}'
    run --stage s7b3p695 $C --seeds 1 --interval-choices 3.695 --choice-jitter 0.1
    run --stage s7b10p043 $C --seeds 1 --interval-choices 10.043 --choice-jitter 0.1
    ;;
  rules)
    run --stage s7t37 $C --seeds 1-20 --poll-rule '{"i_min_s": 3.695, "factor": 2, "i_max_s": 10.043, "jitter_frac": 0.1, "on_timeout": "keep"}'
    run --stage s7t37r $C --seeds 1-20 --poll-rule '{"i_min_s": 3.695, "factor": 2, "i_max_s": 10.043, "jitter_frac": 0.1, "on_timeout": "reset"}'
    ;;
  *)
    echo "usage: sudo bash main6.sh rerun|rules"; exit 2
    ;;
esac
echo "== main6 ${1} finished $(date -Is)"
