#!/bin/bash
# The runs of evidence/19 (spec.md): the two on/off loads, three fixed arms and the two timeout rules on
# each. Same outage shape and agent settings as main7.sh, p 0.4. Stage prefix s9: s8 belongs to evidence/17
# and evalrun does not compare settings, so a stage name is never reused.
#   gate  : three w0 seed-1 runs. They say that the workload knob left the default load's runs comparable
#           with the s7 runs of evidence/16 before 200 runs are spent on the new loads.
#   sweep : the 200 on/off runs, seed outer (for each load, for each seed, the five stages). Stopping in the
#           middle then leaves every (seed, stage) pair of the seeds it finished, so a re-run needs no
#           retake of a partial seed. All of wo first, then all of wv: the ticket's order, so that the
#           decision logs of one load are complete before the other starts.
#   dry   : print the evalrun command lines of gate and sweep without running anything, for review.
# All runs from one detached, clean checkout nobody edits while it is loaded. Resumable: evalrun skips DONE
# runs, which is why one evalrun call per (stage, seed) costs nothing.
# Before gate and before sweep, check by hand that no emulator and no Cefore daemon is running.
# PYTHONDONTWRITEBYTECODE: root would otherwise leave __pycache__ in the checkout that its owner cannot
# delete (2026-11-19: removing the evidence/16 run checkout needed root for that reason).
# Run as root: sudo bash main9.sh gate   then   sudo bash main9.sh sweep
set -u
W=/home/lab_shared/wt/semindex-main9
R=/home/lab_shared/cefore-emu-sslab
S=/home/lab_shared/semindex-eval
PY=/home/lab_shared/cefore-semindex/.venv/bin/python3
C="--p 0.4 --k 4 --store-synchronous NORMAL --disconnect-duration-s 8 12"
RULE='"i_min_s": 3.695, "factor": 2, "i_max_s": 10.043, "jitter_frac": 0.1'
DRY=0
# In dry mode an argument that holds a space is printed inside single quotes and every other argument bare,
# so a reviewer reads one line per run with the --poll-rule JSON intact, instead of a line whose quoting was
# lost to word splitting. No argument here contains a single quote, so the printed line is also the line that
# can be pasted back into a shell.
run() {
  if [ "$DRY" = 1 ]; then
    printf 'env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=%s/src %s -m semindex.evalrun --store %s --recefore %s' \
      "$W" "$PY" "$S" "$R"
    local a
    for a in "$@"; do
      case "$a" in
        *[[:space:]]*) printf " '%s'" "$a" ;;
        *) printf ' %s' "$a" ;;
      esac
    done
    printf '\n'
    return 0
  fi
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$W/src $PY -m semindex.evalrun --store $S --recefore $R "$@"
  echo "== exit $? : $*"
}
gate() {
  run --stage s9w0_b3p695 $C --workload w0 --seeds 1 --interval-choices 3.695 --choice-jitter 0.1
  run --stage s9w0_b10p043 $C --workload w0 --seeds 1 --interval-choices 10.043 --choice-jitter 0.1
  run --stage s9w0_t37r $C --workload w0 --seeds 1 --poll-rule "{$RULE, \"on_timeout\": \"reset\"}"
}
sweep() {
  local wl seed
  for wl in wo wv; do
    for seed in $(seq 1 20); do
      run --stage s9${wl}_b3p695 $C --workload $wl --seeds $seed --interval-choices 3.695 --choice-jitter 0.1
      run --stage s9${wl}_b6p0 $C --workload $wl --seeds $seed --interval-choices 6.0 --choice-jitter 0.1
      run --stage s9${wl}_b10p043 $C --workload $wl --seeds $seed --interval-choices 10.043 --choice-jitter 0.1
      run --stage s9${wl}_t37 $C --workload $wl --seeds $seed --poll-rule "{$RULE, \"on_timeout\": \"keep\"}"
      run --stage s9${wl}_t37r $C --workload $wl --seeds $seed --poll-rule "{$RULE, \"on_timeout\": \"reset\"}"
    done
  done
}
# The commit header is printed only for a real run: in dry mode $W does not exist yet (it is cut after the
# merge), and a failing git would bury the command lines the dry mode exists to show.
case "${1:-}" in
  gate)
    echo "== commit $(git -C $W rev-parse HEAD) dirty=$(git -C $W status --porcelain | wc -l)" \
      "recefore $(git -c safe.directory=$R -C $R rev-parse HEAD) dirty=$(git -c safe.directory=$R -C $R status --porcelain | wc -l)" \
      "start $(date -Is)"
    gate
    ;;
  sweep)
    echo "== commit $(git -C $W rev-parse HEAD) dirty=$(git -C $W status --porcelain | wc -l)" \
      "recefore $(git -c safe.directory=$R -C $R rev-parse HEAD) dirty=$(git -c safe.directory=$R -C $R status --porcelain | wc -l)" \
      "start $(date -Is)"
    sweep
    ;;
  dry)
    DRY=1
    echo "== gate (3 runs)"
    gate
    echo "== sweep (200 runs)"
    sweep
    ;;
  *)
    echo "usage: sudo bash main9.sh gate|sweep   (or: bash main9.sh dry)"; exit 2
    ;;
esac
echo "== main9 ${1} finished $(date -Is)"
