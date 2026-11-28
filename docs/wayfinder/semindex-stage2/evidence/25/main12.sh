#!/bin/bash
# The runs of evidence/25's live comparison: three fixed arms, the rule T37r, the pinned
# formula F, M (evidence/24's linear decider) and L (the tuned model, through the relay),
# all on the held-out seeds 121 to 160 of the main cell (p 0.2, w0, the s6 outage shape).
# Seven stages, 40 seeds each, 280 runs. Stage prefix s12: s10 belongs to evidence/20 and
# evalrun does not compare settings across stages, so a stage name is never reused.
#   sweep   : the 280 runs, seed outer (for each seed, the seven stages in the order below).
#             Stopping in the middle leaves every (seed, stage) pair already run alone, so a
#             re-run costs nothing on what already finished: evalrun itself skips a (stage,
#             seed) whose store directory already has DONE.
#   --print : echo every evalrun command line this sweep would run, with no evalrun, sudo,
#             mininet, Cefore daemon, ssh or socket touched, for review before `sweep`. An
#             optional seed argument (e.g. `--print 121`) narrows this to one seed's 7 lines.
#
# T37r and F are copied unchanged from evidence/20/main10.sh, so s12t37r and s12f measure the
# same two baselines against the same held-out seeds that evidence/20 already measured them
# against, now sitting beside M and L in one sweep instead of a second, separate one.
#
# Every stage runs from ONE detached clean checkout of the docs commit that holds this file.
# That commit sits on top of the merged src commit, so its src tree is the merged one and the
# decider files M and L load are the ones the same commit records. The assertion below refuses
# the checkout unless it descends from SRC_COMMIT with an src tree byte-equal to it.
# 2026-11-28: a second checkout for the decider files was considered and dropped, because
# nothing pinned which commit it held, and a decider file read from an unpinned tree is not
# the file the seam document describes.
#
# PYTHONDONTWRITEBYTECODE: root would otherwise leave __pycache__ in a checkout its own owner
# cannot delete.
# Run as root: sudo bash main12.sh sweep
set -u

# Every path below can be overridden from the environment (W=... bash main12.sh ...), so this
# script can be pointed at a scratch checkout for review without editing the file.
W="${W:-/home/lab_shared/wt/semindex-main12}"   # the one run checkout, checked below
DECKW="$W"
R="${R:-/home/lab_shared/cefore-emu-sslab}"
# The production store. This script never opens a path under it itself: every byte it causes
# to be written there goes through semindex.evalrun, named on the command line below.
S="${S:-/home/lab_shared/semindex-eval}"
PY="${PY:-/home/lab_shared/cefore-semindex/.venv/bin/python3}"
# Where the s12m and s12l recorder writes its per-node input records, outside every run's own
# node_dirs so parity.py can read them without reaching into a store run directory. Also where
# the L client's own failure counters land (its stats_dir): the two never collide, because the
# recorder writes "<node>.jsonl" and the client writes "<node>.lclient.json".
RECBASE="${RECBASE:-/home/lab_shared/t25rec-s12}"
# Fixed, short, on the shared filesystem: lrun.sh's relay binds this path, and a unix socket
# path over 108 bytes cannot be opened at all. Not an override like the paths above: lrun.sh's
# own LSOCK is hardcoded to this same value, and probing a different socket than the one
# ldecider.py actually connects to would make probe_l_or_die check nothing useful.
SOCKET_PATH=/home/lab_shared/t25sock/l.sock
LRUN="$DECKW/docs/wayfinder/semindex-stage2/evidence/25/lrun.sh"

# The commit this comparison's src change (the decider seam in agent.py/decider.py) was
# reviewed and merged at. Not overridable: this is the value $W is checked against, not a
# setting of this script.
SRC_COMMIT=c333655a52f52a36b0eb4ae3a56c0970317b9e9d

C="--p 0.2 --k 4 --store-synchronous NORMAL --workload w0 --disconnect-duration-s 8 12"
# T37r, copied unchanged from evidence/20/main10.sh.
T37R='{"i_min_s": 3.695, "factor": 2, "i_max_s": 10.043, "jitter_frac": 0.1, "on_timeout": "reset"}'
# The pinned F, copied unchanged from evidence/20/main10.sh's own copy of fpin.txt's winner
# line. Not re-derived here: a second definition of the same string could drift from the one
# evidence/20 already measured.
FPIN='{"i_min_s": 3.695, "factor": 1.5, "i_max_s": 10.043, "jitter_frac": 0.1, "on_timeout": "reset"}'
# ../22/port.py:45's ARMS[1], the interval both M and L assume a pair holds before its first
# resolution (seam.md section 8.2). A literal here, not an import of port.py, because the L
# client (ldecider.py) is kept as small as a socket and a JSON encoder and port.py pulls in a
# whole replay module at import time. This script has no such constraint, so it is the one
# that imports port.py (assert_send_start_or_die, below) and refuses a mismatch before any
# run, in --print as well as sweep.
SEND_START=6.0

PRINT=0

refuse_if_testbed_running() {
  # A bracket around one letter of the pattern keeps this grep from matching its own argv:
  # pgrep's own command line contains the literal text "[c]efnetd", not the three letters
  # "cef" in a row, so it never matches the pattern it is searching for.
  local name
  for name in mininet cefnetd csmgrd; do
    if pgrep -f "[${name:0:1}]${name:1}" >/dev/null 2>&1; then
      echo "REFUSING: a $name process is already running. Stop it before this sweep starts." >&2
      exit 1
    fi
  done
}

assert_src_commit() {
  local head
  head=$(git -c safe.directory="$W" -C "$W" rev-parse HEAD 2>&1) || {
    echo "REFUSING: cannot read $W's HEAD ($head). Set up the src checkout first." >&2
    exit 1
  }
  if ! git -c safe.directory="$W" -C "$W" merge-base --is-ancestor "$SRC_COMMIT" "$head"; then
    echo "REFUSING: $W at $head does not descend from the merged src commit $SRC_COMMIT." >&2
    exit 1
  fi
  if ! git -c safe.directory="$W" -C "$W" diff --quiet "$SRC_COMMIT" "$head" -- src; then
    echo "REFUSING: $W's src differs from the merged src commit $SRC_COMMIT." >&2
    exit 1
  fi
  local status_out
  # 2026-11-28 fix: `git status --porcelain | wc -l` discarded git's own exit code -- a
  # git status that failed (empty stdout, non-zero exit) used to read as "0 dirty files"
  # through the pipe instead of refusing. The status is captured and its own exit code
  # checked first; only a known-good (possibly empty) string is then piped to wc -l.
  status_out=$(git -c safe.directory="$W" -C "$W" status --porcelain) || {
    echo "REFUSING: cannot read $W's git status." >&2
    exit 1
  }
  local dirty=0
  [ -n "$status_out" ] && dirty=$(printf '%s\n' "$status_out" | wc -l)
  if [ "$dirty" != 0 ]; then
    echo "REFUSING: $W has $dirty uncommitted change(s); the src checkout must be clean." >&2
    exit 1
  fi
}

# Resolves the decider files' paths under DECKW and their sha256, once, so every seed's
# --poll-decider JSON below carries the same digest instead of re-hashing per run. Needed by
# both sweep and --print: --print exists so a reviewer can see the exact command lines,
# digest included, not a placeholder standing in for one.
decider_paths_or_die() {
  MDEC_PATH="$DECKW/docs/wayfinder/semindex-stage2/evidence/25/mdecider.py"
  LDEC_PATH="$DECKW/docs/wayfinder/semindex-stage2/evidence/25/ldecider.py"
  COEF_PATH="$DECKW/docs/wayfinder/semindex-stage2/evidence/24/coef_a.npy"
  local f
  for f in "$MDEC_PATH" "$LDEC_PATH" "$COEF_PATH"; do
    if [ ! -f "$f" ]; then
      echo "REFUSING: $f not found. Set up the decider checkout ($DECKW) first." >&2
      exit 1
    fi
  done
  MDEC_SHA=$(sha256sum "$MDEC_PATH" | cut -d' ' -f1)
  LDEC_SHA=$(sha256sum "$LDEC_PATH" | cut -d' ' -f1)
}

# 2026-11-28 fix: seam.md 8.2 and spec.md's pass line 2 both used to say parity.py asserts
# SEND_START == ../22/port.py's ARMS[1]; parity.py never imports port or checks ARMS at all
# (verified by reading the whole file), so nothing actually caught a SEND_START that drifted
# from the choice both M and L assume a pair holds before its first resolution. This is the
# check, run before any evalrun call in both --print and sweep.
assert_send_start_or_die() {
  local port_path="$DECKW/docs/wayfinder/semindex-stage2/evidence/22/port.py"
  if [ ! -f "$port_path" ]; then
    echo "REFUSING: $port_path not found. Set up the decider checkout ($DECKW) first." >&2
    exit 1
  fi
  "$PY" -B -c "
import importlib.util
spec = importlib.util.spec_from_file_location('port', '$port_path')
port = importlib.util.module_from_spec(spec)
spec.loader.exec_module(port)
send_start = $SEND_START
if port.ARMS[1] != send_start:
    raise SystemExit('SEND_START=%r does not equal port.ARMS[1]=%r' % (send_start, port.ARMS[1]))
" || { echo "REFUSING: SEND_START does not equal ../22/port.py's ARMS[1]. See the error above." >&2; exit 1; }
}

# Called only outside --print: a probe that actually opened the relay's socket would no
# longer be something this mode "only echoes".
probe_l_or_die() {
  if ! bash "$LRUN" probe; then
    echo "REFUSING: the L server did not answer the probe through the relay." >&2
    echo "This launcher never starts or restarts it; start it by hand (MAIN12_RUN.md step 2) and re-run." >&2
    exit 1
  fi
}

# 2026-11-28 fix: evalrun's own resume guard only clears the STORE side of a (stage, seed)
# that lacks DONE (_guarded_rmtree, src/semindex/evalrun.py:749-750); the record/stats files
# this script's RECBASE keeps live outside the store, so nothing used to clear them, and a
# retaken s12m/s12l run would append attempt 2's lines onto attempt 1's leftover
# <node>.jsonl (fakes.Recording opens it in append mode on purpose, so a crash mid-run never
# loses lines a first run already wrote) -- making parity.py and arms.py read the two
# attempts as one, mismatched run. Moving the stale directory aside, never deleting it,
# keeps attempt 2's record clean without discarding attempt 1's own evidence. The DONE path
# checked here is evalrun's own (src/semindex/evalrun.py's _store_run_dir: store/stage/p<p>/seed).
stash_stale_record_dir() {
  local stage="$1" seed="$2"
  local done_marker="$S/$stage/p0.2/$seed/DONE"
  local dir="$RECBASE/$stage/$seed"
  if [ -e "$done_marker" ]; then
    return 0
  fi
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    local n=1
    while [ -e "${dir}.attempt-${n}" ]; do
      n=$((n + 1))
    done
    mv "$dir" "${dir}.attempt-${n}" || {
      echo "REFUSING: could not move the stale record dir $dir aside." >&2
      exit 1
    }
    echo "== stashed stale record dir $dir as ${dir}.attempt-${n} (no DONE marker yet for $stage seed $seed)"
  fi
}

m_decider_json() {
  # $1: this (stage, seed)'s record_dir. factory is build(node, coef, record_dir=None,
  # delay_s=None); delay_s is left out, because M answers synchronously and needs no
  # deferring wrapper (mdecider.py's own docstring).
  local record_dir="$1"
  printf '{"path": "%s", "sha256": "%s", "factory": "build", "args": {"coef": "%s", "record_dir": "%s"}}' \
    "$MDEC_PATH" "$MDEC_SHA" "$COEF_PATH" "$record_dir"
}

l_decider_json() {
  # $1: this (stage, seed)'s record_dir, reused as stats_dir (see RECBASE's comment above).
  # factory is build(node, socket_path, send_start, stats_dir, record_dir=None).
  local record_dir="$1"
  printf '{"path": "%s", "sha256": "%s", "factory": "build", "args": {"socket_path": "%s", "send_start": %s, "stats_dir": "%s", "record_dir": "%s"}}' \
    "$LDEC_PATH" "$LDEC_SHA" "$SOCKET_PATH" "$SEND_START" "$record_dir" "$record_dir"
}

# In --print, an argument holding a space is printed inside single quotes and every other
# argument bare, so a reviewer reads one line per run with the --poll-rule/--poll-decider
# JSON intact instead of a line whose quoting was lost to word splitting. No argument here
# holds a single quote, so the printed line is also the line that could be pasted into a shell.
run() {
  if [ "$PRINT" = 1 ]; then
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
  local rc=$?
  echo "== exit $rc : $*"
  # 2026-11-28 fix: evalrun.run_all already catches every per-run exception inside itself,
  # so a non-zero exit here is sweep-level (a DirtySemindexError, a bad flag, a broken venv,
  # a signal), not one run's own failure -- and the sweep used to keep going through all 280
  # calls and finish exit 0 regardless.
  if [ "$rc" != 0 ]; then
    echo "REFUSING: semindex.evalrun exited $rc; stopping the sweep." >&2
    exit "$rc"
  fi
}

# The seven stages of one seed, in the order this comparison tables them. One evalrun call
# per (stage, seed), not one call with --seeds 121-160 per stage, for the same resumability
# reason evidence/20/main10.sh gives: a sweep stopped part way then leaves every seed with
# either all seven stages done or none of them touched yet, never some of each.
sweep() {
  local seed
  for seed in $SEEDS_LIST; do
    run --stage s12b3p695 $C --seeds $seed --interval-choices 3.695 --choice-jitter 0.1
    run --stage s12b6p0 $C --seeds $seed --interval-choices 6.0 --choice-jitter 0.1
    run --stage s12b10p043 $C --seeds $seed --interval-choices 10.043 --choice-jitter 0.1
    run --stage s12t37r $C --seeds $seed --poll-rule "$T37R"
    run --stage s12f $C --seeds $seed --poll-rule "$FPIN"
    [ "$PRINT" = 1 ] || stash_stale_record_dir s12m "$seed"
    run --stage s12m $C --seeds $seed --interval-choices 3.695 6.0 10.043 --choice-jitter 0.1 \
      --choice-redecide --poll-decider "$(m_decider_json "$RECBASE/s12m/$seed")"
    # The probe, not a start: this launcher is told never to bring the L server up itself,
    # only to refuse cleanly when it finds nothing answering on the other end of the relay.
    [ "$PRINT" = 1 ] || probe_l_or_die
    [ "$PRINT" = 1 ] || stash_stale_record_dir s12l "$seed"
    run --stage s12l $C --seeds $seed --interval-choices 3.695 6.0 10.043 --choice-jitter 0.1 \
      --choice-redecide --poll-decider "$(l_decider_json "$RECBASE/s12l/$seed")"
  done
}

case "${1:-}" in
  sweep)
    SEEDS_LIST="$(seq 121 160)"
    refuse_if_testbed_running
    decider_paths_or_die
    assert_send_start_or_die
    assert_src_commit
    echo "== commit $(git -c safe.directory="$W" -C "$W" rev-parse HEAD) dirty=$(git -c safe.directory="$W" -C "$W" status --porcelain | wc -l)" \
      "src-equal-to $SRC_COMMIT" \
      "recefore $(git -c safe.directory="$R" -C "$R" rev-parse HEAD) dirty=$(git -c safe.directory="$R" -C "$R" status --porcelain | wc -l)" \
      "start $(date -Is)"
    sweep
    ;;
  --print)
    PRINT=1
    decider_paths_or_die
    assert_send_start_or_die
    if [ -n "${2:-}" ]; then
      SEEDS_LIST="$2"
      echo "== sweep, seed $2 only (7 commands)"
    else
      SEEDS_LIST="$(seq 121 160)"
      echo "== sweep (280 commands)"
    fi
    sweep
    ;;
  *)
    echo "usage: sudo bash main12.sh sweep   (or: bash main12.sh --print [SEED])"; exit 2
    ;;
esac
echo "== main12 ${1} finished $(date -Is)"
