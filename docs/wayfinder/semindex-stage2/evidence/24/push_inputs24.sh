#!/usr/bin/env bash
# Send filtered s11a seeds 1-80 run inputs and worktree code to hpc-debian.
#
# Runs ON THE LOCAL MACHINE.  Idempotent: verifies sha256 digests on both sides.
# Use `--code-only` after the initial store push to re-sync code without re-tarring the
# 109 MB store (still verifies that no seed > 80 sits under ROOT/store/s11a).
#
# 2026-11-26: keeps all previous files under ROOT untouched; writes only into
# ROOT/store/s11a and ROOT/repo24.
set -euo pipefail

WT=/home/lab_shared/wt/semindex-ev24
STORE_SRC=/home/lab_shared/semindex-eval
HOST=hpc-debian
ROOT=/home/yamakawa/Development/semindex-learn
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"

STAGE=s11a
P_CELLS="p0.2 p0.4"
SEEDS=$(seq 1 80)
NODES="h0 h1 h2 h3 h4"
CODE_ONLY=0
if [ "${1:-}" = "--code-only" ]; then
    CODE_ONLY=1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

$SSH "$HOST" "mkdir -p $ROOT/store $ROOT/repo24 $ROOT/ev24"

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
    $SSH "$HOST" "cat > $ROOT/.push24.list" < "$list"
    $SSH "$HOST" "cd $dest && tr '\n' '\0' < $ROOT/.push24.list | xargs -0 sha256sum; \
                  rm -f $ROOT/.push24.list" | sort -k2 > "$TMP/far.sha"
    if cmp -s "$TMP/near.sha" "$TMP/far.sha"; then
        echo "  digests: all $n files agree on both sides"
    else
        echo "FAIL: the two sides disagree on these files:" >&2
        diff "$TMP/near.sha" "$TMP/far.sha" >&2 || true
        exit 1
    fi
}

if [ "$CODE_ONLY" -eq 0 ]; then
    echo "== (a) s11a run inputs (seeds 1 to 80 only, filtered to 4 kinds)"
    : > "$TMP/store.list"
    for pdir in $P_CELLS; do
        for s in $SEEDS; do
            d=$STORE_SRC/$STAGE/$pdir/$s
            for n in $NODES; do
                f=$STAGE/$pdir/$s/debug/node_dirs/$n/semindex/events.jsonl
                [ -f "$STORE_SRC/$f" ] || { echo "FAIL: missing $STORE_SRC/$f" >&2; exit 1; }
                printf '%s\n' "$f" >> "$TMP/store.list"
            done
            for f in meta.json results.json; do
                [ -f "$d/$f" ] || { echo "FAIL: missing $d/$f" >&2; exit 1; }
                printf '%s\n' "$STAGE/$pdir/$s/$f" >> "$TMP/store.list"
            done
            mapfile -t exp < <(cd "$d" && ls -1 ./*.experiment.json 2>/dev/null | sed 's#^\./##')
            [ "${#exp[@]}" = 1 ] || { echo "FAIL: $d has ${#exp[@]} *.experiment.json" >&2; exit 1; }
            printf '%s\n' "$STAGE/$pdir/$s/${exp[0]}" >> "$TMP/store.list"
        done
    done
    send_and_verify "$STORE_SRC" "$ROOT/store" "$TMP/store.list"
else
    echo "== (a) s11a run inputs: skipped (--code-only)"
fi

echo "== (b) worktree code into ROOT/repo24"
(cd "$WT" && git ls-files -- src/semindex docs/wayfinder/semindex-stage2/evidence \
    | grep -E '\.(py|json|csv|sha256|txt)$') > "$TMP/code.list"
(cd "$WT" && find docs/wayfinder/semindex-stage2/evidence/24 -maxdepth 1 -type f \
    | grep -E '\.(py|json|csv|sha256|txt)$' | sort) >> "$TMP/code.list"
sort -u "$TMP/code.list" -o "$TMP/code.list"
send_and_verify "$WT" "$ROOT/repo24" "$TMP/code.list"

echo "== verify ROOT/store/s11a contains only seeds 1..80 and 4 allowed file kinds"
$SSH "$HOST" "cd $ROOT/store && find s11a -type f | sort" > "$TMP/far_s11a.list"
python3 - "$TMP/far_s11a.list" <<'PYEOF'
import re
import sys

KINDS = {
    "events.jsonl": re.compile(r"^s11a/p0\.[24]/(\d+)/debug/node_dirs/h[0-4]/semindex/events\.jsonl$"),
    "meta.json": re.compile(r"^s11a/p0\.[24]/(\d+)/meta\.json$"),
    "results.json": re.compile(r"^s11a/p0\.[24]/(\d+)/results\.json$"),
    "*.experiment.json": re.compile(r"^s11a/p0\.[24]/(\d+)/[^/]+\.experiment\.json$"),
}
names = [ln.rstrip("\n") for ln in open(sys.argv[1], encoding="utf-8") if ln.strip()]
counts = dict.fromkeys(KINDS, 0)
seeds = set()
bad = []
for name in names:
    matched = False
    for kind, pat in KINDS.items():
        m = pat.match(name)
        if m:
            s = int(m.group(1))
            if not 1 <= s <= 80:
                bad.append(f"seed out of 1..80: {name}")
            seeds.add(s)
            counts[kind] += 1
            matched = True
            break
    if not matched:
        bad.append(f"disallowed file: {name}")
for kind, n in counts.items():
    print(f"  {n:4d}  {kind}")
print(f"  {len(names):4d}  total, distinct seeds {len(seeds)} ({min(seeds)}..{max(seeds)})")
if bad or len(names) != 1280 or len(seeds) != 80:
    for item in bad[:10]:
        print("FAIL:", item, file=sys.stderr)
    raise SystemExit(1)
print("OK: ROOT/store/s11a holds exactly the 1280 allowed files of seeds 1 to 80")
PYEOF
