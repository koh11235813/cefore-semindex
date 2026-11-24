#!/usr/bin/env bash
# Send to the tuning host everything the replay reads, and nothing else.
#
# Runs ON THE LOCAL MACHINE.  Idempotent: it overwrites the same paths with the same
# bytes and re-verifies, so a second run leaves the far side as it was.
#
#   bash push_inputs.sh
#
# 2026-11-24.  Three payloads, kept apart because they have three different reasons to be
# there and three different ways of being wrong:
#   (a) the hindsight labels, which are the training target and are checked by digest;
#   (b) the run inputs of the main cell, FILTERED -- a run directory also holds the
#       agent's own log, its yaml and other subdirectories, and none of that is an input
#       of the replay, so the file list is built name by name rather than by an include
#       pattern that would quietly widen if a run directory gained a file;
#   (c) the code, at the same relative paths as the worktree, because two scripts on the
#       replay path assert their own location: evidence/20's predict.py walks five levels
#       up from itself to find `src` and refuses a `semindex` imported from anywhere else,
#       and evidence/22's port.py asserts the realpath of the engine modules it imports.
#
# 2026-11-24, NOT rsync, and this is a deviation and not a preference: rsync is not
# installed on this machine (no /usr/bin/rsync, no dpkg entry), and installing it would be
# a system change nobody asked for.  `tar -T <list>` over ssh carries the same explicit
# file list to the same relative paths.  What rsync's own checksum pass would have given
# is replaced by something stricter and visible: every transferred file is hashed on BOTH
# sides and the two lists are compared, so a short or mangled transfer stops the script.
set -euo pipefail

WT=/home/lab_shared/wt/semindex-ev23
S2=$WT/docs/wayfinder/semindex-stage2
STORE_SRC=/home/lab_shared/semindex-eval
LEARN_SRC=/home/lab_shared/semindex-learn/ev22

HOST=hpc-debian
ROOT=/home/yamakawa/Development/semindex-learn
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"

STAGE=s6b10p043
PDIR=p0.2                       # `replay.pdir(0.2)`
SEEDS=$(seq 1 20)               # the main cell's logged seeds; 21-100 borrow one of these
NODES="h0 h1 h2 h3 h4"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

$SSH "$HOST" "mkdir -p $ROOT/ev22 $ROOT/store $ROOT/repo"

# One function, so that the three payloads cannot drift into three different ideas of
# "sent and verified".  $1 source root, $2 destination directory, $3 the file list.
send_and_verify() {
    local src=$1 dest=$2 list=$3
    local n bytes
    n=$(wc -l < "$list")
    bytes=$(cd "$src" && tr '\n' '\0' < "$list" | xargs -0 stat -c %s \
            | awk '{s+=$1} END {print s}')
    echo "  files $n, bytes $bytes"
    $SSH "$HOST" "mkdir -p $dest"
    tar -C "$src" -cf - --no-recursion -T "$list" \
        | $SSH "$HOST" "tar -C $dest -xf - --no-same-owner"
    (cd "$src" && tr '\n' '\0' < "$list" | xargs -0 sha256sum) | sort -k2 > "$TMP/near.sha"
    # The list goes to ROOT and not into $dest: a scratch file under ROOT/store would be a
    # fifth kind of file there, and the classification below would then have to know about
    # this script's own leftovers.
    $SSH "$HOST" "cat > $ROOT/.push.list" < "$list"
    $SSH "$HOST" "cd $dest && tr '\n' '\0' < $ROOT/.push.list | xargs -0 sha256sum; \
                  rm -f $ROOT/.push.list" | sort -k2 > "$TMP/far.sha"
    if cmp -s "$TMP/near.sha" "$TMP/far.sha"; then
        echo "  digests: all $n files agree on both sides"
    else
        echo "FAIL: the two sides disagree on these files:" >&2
        diff "$TMP/near.sha" "$TMP/far.sha" >&2 || true
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# (a) the labels
# ---------------------------------------------------------------------------
echo "== (a) labels"
printf '%s\n' labels_p0.2.jsonl.gz labels_p0.4.jsonl.gz > "$TMP/labels.list"
send_and_verify "$LEARN_SRC" "$ROOT/ev22" "$TMP/labels.list"

# The committed digest file names the files by their ABSOLUTE local path, which does not
# exist on the far side, so the check is run on the basenames.  Only the path column is
# rewritten; the hashes are the committed ones and the hashing happens on the far side.
sed -E 's# +/.*/# #' "$S2/evidence/22/labels.sha256" > "$TMP/labels.sha256"
$SSH "$HOST" "cat > $ROOT/ev22/labels.sha256" < "$TMP/labels.sha256"
echo "-- sha256sum -c against evidence/22/labels.sha256, on the far side"
$SSH "$HOST" "cd $ROOT/ev22 && sha256sum -c labels.sha256"

# ---------------------------------------------------------------------------
# (b) the run inputs, by an explicit file list
# ---------------------------------------------------------------------------
echo "== (b) run inputs"
: > "$TMP/store.list"
for s in $SEEDS; do
    d=$STORE_SRC/$STAGE/$PDIR/$s
    for n in $NODES; do
        f=$STAGE/$PDIR/$s/debug/node_dirs/$n/semindex/events.jsonl
        [ -f "$STORE_SRC/$f" ] || { echo "FAIL: missing $STORE_SRC/$f" >&2; exit 1; }
        printf '%s\n' "$f" >> "$TMP/store.list"
    done
    for f in meta.json results.json; do
        [ -f "$d/$f" ] || { echo "FAIL: missing $d/$f" >&2; exit 1; }
        printf '%s\n' "$STAGE/$PDIR/$s/$f" >> "$TMP/store.list"
    done
    # `replay.load_run` takes `glob("*.experiment.json")[0]`, so more than one would make
    # the replay's choice depend on the directory order; one is asserted here instead.
    mapfile -t exp < <(cd "$d" && ls -1 ./*.experiment.json 2>/dev/null | sed 's#^\./##')
    [ "${#exp[@]}" = 1 ] || { echo "FAIL: $d has ${#exp[@]} *.experiment.json" >&2; exit 1; }
    printf '%s\n' "$STAGE/$PDIR/$s/${exp[0]}" >> "$TMP/store.list"
done
send_and_verify "$STORE_SRC" "$ROOT/store" "$TMP/store.list"

# ---------------------------------------------------------------------------
# (c) the code
# ---------------------------------------------------------------------------
echo "== (c) code"
# The worktree's TRACKED files under the two trees the replay path needs, plus this
# directory, which is not committed yet.  Only the kinds a script reads: the modules
# themselves and the small data files beside them (`evidence/22/weight.json` is read at
# import, `labels.sha256` by the label checks, the csv by the analysis helpers).
# 2026-11-24, a deliberate over-send: every such file under `evidence/` goes, about 5 MB,
# rather than the transitive closure of what the parity run touches.  Tracing every
# `open()` of eight scripts to a minimal list would be a second definition of the replay's
# own dependencies, and getting it wrong shows up as an ImportError on the far side only
# for the paths a given run happens to take.
(cd "$WT" && git ls-files -- src/semindex docs/wayfinder/semindex-stage2/evidence \
    | grep -E '\.(py|json|csv|sha256)$') > "$TMP/code.list"
(cd "$WT" && find docs/wayfinder/semindex-stage2/evidence/23 -maxdepth 1 -type f \
    | sort) >> "$TMP/code.list"
send_and_verify "$WT" "$ROOT/repo" "$TMP/code.list"

# ---------------------------------------------------------------------------
# What landed under ROOT/store, kind by kind
# ---------------------------------------------------------------------------
echo "== every file under $ROOT/store, classified"
$SSH "$HOST" "cd $ROOT/store && find . -type f | sort" > "$TMP/far.list"
python3 - "$TMP/far.list" <<'PYEOF'
import re
import sys

# The four kinds `replay.load_run` opens, and nothing else may be there.  Each pattern is
# anchored at both ends: a pattern that only had to appear somewhere in the name would
# accept `.../meta.json.bak` and `.../semindex/events.jsonl.gz`.
KINDS = {
    "events.jsonl": re.compile(r"^\./[^/]+/[^/]+/\d+/debug/node_dirs/h[0-4]/semindex/events\.jsonl$"),
    "meta.json": re.compile(r"^\./[^/]+/[^/]+/\d+/meta\.json$"),
    "results.json": re.compile(r"^\./[^/]+/[^/]+/\d+/results\.json$"),
    "*.experiment.json": re.compile(r"^\./[^/]+/[^/]+/\d+/[^/]+\.experiment\.json$"),
}
names = [ln.rstrip("\n") for ln in open(sys.argv[1]) if ln.strip()]
counts = dict.fromkeys(KINDS, 0)
bad = []
for name in names:
    for kind, pat in KINDS.items():
        if pat.match(name):
            counts[kind] += 1
            break
    else:
        bad.append(name)
for kind, n in counts.items():
    print(f"  {n:4d}  {kind}")
print(f"  {len(names):4d}  total")
if bad:
    for name in bad:
        print("FAIL: not one of the four allowed kinds:", name, file=sys.stderr)
    raise SystemExit(1)
print("OK: every file under the store is one of the four kinds the replay opens")
PYEOF
