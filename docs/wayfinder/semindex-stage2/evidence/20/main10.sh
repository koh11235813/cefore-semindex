#!/bin/bash
# The runs of evidence/20 (spec.md): the baseline stage of ticket 07, on the held-out seeds 121 to 160 of the
# main cell (p 0.2, w0, the s6 outage shape). Three fixed arms, the rule T37r and the pinned formula F, 40
# seeds each, 200 runs. Stage prefix s10: s9 belongs to evidence/19 and evalrun does not compare settings,
# so a stage name is never reused.
#   sweep : the 200 runs, seed outer (for each seed, the five stages). Stopping in the middle then leaves
#           every (seed, stage) pair of the seeds it finished, so a re-run needs no retake of a partial seed.
#   dry   : print the evalrun command lines of the sweep without running anything, for review.
# NO GATE STAGE, unlike main9.sh. A gate exists to catch a code version that moved between two stages that
# will be compared, and `git -C <worktree> diff --stat 03f5ecc d009cfb -- src tests` is empty: 03f5ecc is the
# commit evidence/19 ran from (../19/RUN.txt) and d009cfb is the commit this sweep runs from, so src and
# tests are the same version in both. spec.md records that command and its empty output as the evidence.
# All runs from one detached, clean checkout nobody edits while it is loaded. Resumable: evalrun skips DONE
# runs, which is why one evalrun call per (stage, seed) costs nothing.
# Before the sweep, check by hand that no emulator and no Cefore daemon is running.
# PYTHONDONTWRITEBYTECODE: root would otherwise leave __pycache__ in the checkout that its owner cannot
# delete (2026-11-19: removing the evidence/16 run checkout needed root for that reason).
# Run as root: sudo bash main10.sh sweep
set -u
W=/home/lab_shared/wt/semindex-main10
R=/home/lab_shared/cefore-emu-sslab
S=/home/lab_shared/semindex-eval
PY=/home/lab_shared/cefore-semindex/.venv/bin/python3
C="--p 0.2 --k 4 --store-synchronous NORMAL --workload w0 --disconnect-duration-s 8 12"
# T37r, the second baseline, exactly as the s9 stages launched it.
T37R='{"i_min_s": 3.695, "factor": 2, "i_max_s": 10.043, "jitter_frac": 0.1, "on_timeout": "reset"}'
# The pinned F, copied from the winner line of fpin.txt and not edited here. Changing this string makes the
# sweep measure a member other than the one spec.md names, so it is written out once and used once.
FPIN='{"i_min_s": 3.695, "factor": 1.5, "i_max_s": 10.043, "jitter_frac": 0.1, "on_timeout": "reset"}'
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
# The five stages of one seed, in the order spec.md tables them. One evalrun call per (stage, seed) rather
# than one call with --seeds 121-160 per stage: the seed-outer order is what keeps a sweep that stops part
# way from leaving a seed with some stages run and others not.
sweep() {
  local seed
  for seed in $(seq 121 160); do
    run --stage s10b3p695 $C --seeds $seed --interval-choices 3.695 --choice-jitter 0.1
    run --stage s10b6p0 $C --seeds $seed --interval-choices 6.0 --choice-jitter 0.1
    run --stage s10b10p043 $C --seeds $seed --interval-choices 10.043 --choice-jitter 0.1
    run --stage s10t37r $C --seeds $seed --poll-rule "$T37R"
    run --stage s10f $C --seeds $seed --poll-rule "$FPIN"
  done
}
# The commit header is printed only for a real run: in dry mode $W does not exist yet (it is cut when the
# sweep is set up), and a failing git would bury the command lines the dry mode exists to show.
case "${1:-}" in
  sweep)
    echo "== commit $(git -C $W rev-parse HEAD) dirty=$(git -C $W status --porcelain | wc -l)" \
      "recefore $(git -c safe.directory=$R -C $R rev-parse HEAD) dirty=$(git -c safe.directory=$R -C $R status --porcelain | wc -l)" \
      "start $(date -Is)"
    sweep
    ;;
  dry)
    DRY=1
    echo "== sweep (200 runs)"
    sweep
    ;;
  *)
    echo "usage: sudo bash main10.sh sweep   (or: bash main10.sh dry)"; exit 2
    ;;
esac
echo "== main10 ${1} finished $(date -Is)"
