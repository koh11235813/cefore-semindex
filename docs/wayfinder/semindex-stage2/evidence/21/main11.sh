#!/bin/bash
# The runs of evidence/21 (spec.md): the randomly-drawn stage of ticket 08, with the
# choices mode re-deciding its interval when a poll resolves (--choice-redecide). Three
# arms drawn uniformly, p 0.2 and p 0.4, seeds 1 to 100 each, 200 runs. Stage prefix
# s11: s10 belongs to evidence/20 and evalrun does not compare settings, so a stage name
# is never reused.
#   smoke : the two runs of stage s11smoke, seed 101, that evidence/21/smoke.py reads.
#           Seed 101 is outside the sweep's seeds 1 to 100 and outside the held-out
#           seeds 121 to 160, so neither set is spent on it.
#   sweep : the 200 runs, p outer: all of p 0.2 first, then all of p 0.4. p 0.2 is the
#           main cell, so a sweep that is stopped part way leaves that cell whole.
#   dry   : print the evalrun command lines of both modes without running anything.
# NO GATE STAGE. A gate exists to catch a code version that moved between two stages
# that are compared with each other, and no pass line of this step compares an s11 stage
# with a stage run from another commit: pass line 5 holds these runs against the replay's
# own prediction, not against s10. src did move between d009cfb (evidence/20's runs) and
# the commit below, which is why that comparison is not made and spec.md says so.
# All runs from one detached, clean checkout nobody edits while it is loaded. Resumable:
# evalrun skips DONE runs, which is why one evalrun call per (stage, p, seed) costs
# nothing.
# Before the sweep, check by hand that no emulator and no Cefore daemon is running.
# PYTHONDONTWRITEBYTECODE: root would otherwise leave __pycache__ in the checkout that
# its owner cannot delete (2026-11-19: removing the evidence/16 run checkout needed root
# for that reason).
# Run as root: sudo bash main11.sh smoke   then   sudo bash main11.sh sweep
set -u
W=/home/lab_shared/wt/semindex-main11
R=/home/lab_shared/cefore-emu-sslab
S=/home/lab_shared/semindex-eval
PY=/home/lab_shared/cefore-semindex/.venv/bin/python3
# The settings every run of this step shares. The three arms and the jitter are the
# ticket's own; --choice-redecide is the knob the step exists to exercise, and without it
# these runs would be the plain choices mode that stages 04 to 09 already hold.
C="--k 4 --store-synchronous NORMAL --workload w0 --disconnect-duration-s 8 12"
C="$C --interval-choices 3.695 6.0 10.043 --choice-jitter 0.1 --choice-redecide"
DRY=0
# In dry mode an argument that holds a space is printed inside single quotes and every
# other argument bare, so a reviewer reads one line per run with the settings intact
# instead of a line whose quoting was lost to word splitting. No argument here contains a
# single quote, so the printed line is also the line that can be pasted back into a shell.
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
# The two runs smoke.py reads. They carry the sweep's settings exactly, so what they
# prove about the reschedule rows holds for the sweep; only the stage name and the seed
# differ.
smoke() {
  run --stage s11smoke --p 0.2 --seeds 101 $C
  run --stage s11smoke --p 0.4 --seeds 101 $C
}
# One evalrun call per (p, seed) rather than one call with --seeds 1-100 per p: a sweep
# that stops part way then leaves every finished seed complete, and the call that was
# interrupted is the only one to redo.
sweep() {
  local p seed
  for p in 0.2 0.4; do
    for seed in $(seq 1 100); do
      run --stage s11a --p $p --seeds $seed $C
    done
  done
}
# The commit header is printed only for a real run: in dry mode $W does not exist yet (it
# is cut when the sweep is set up), and a failing git would bury the command lines the dry
# mode exists to show.
header() {
  echo "== commit $(git -C $W rev-parse HEAD) dirty=$(git -C $W status --porcelain | wc -l)" \
    "recefore $(git -c safe.directory=$R -C $R rev-parse HEAD) dirty=$(git -c safe.directory=$R -C $R status --porcelain | wc -l)" \
    "start $(date -Is)"
}
case "${1:-}" in
  smoke)
    header
    smoke
    ;;
  sweep)
    header
    sweep
    ;;
  dry)
    DRY=1
    echo "== smoke (2 runs)"
    smoke
    echo "== sweep (200 runs)"
    sweep
    ;;
  *)
    echo "usage: sudo bash main11.sh smoke | sweep   (or: bash main11.sh dry)"; exit 2
    ;;
esac
echo "== main11 ${1} finished $(date -Is)"
