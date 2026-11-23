#!/usr/bin/env bash
# Download the base weights of the one-token classifier onto the tuning host.
#
# Runs ON THE TUNING HOST.  Idempotent: a file whose size already matches the hub's is
# left alone, so a second run re-verifies and downloads nothing.
#
#   bash fetch_weights.sh
#
# 2026-11-24.  curl and not huggingface_hub, for three reasons that are each a decision:
#  * the hub client materialises a snapshot as symlinks into its cache, and what the
#    training step needs is a plain directory of files it can read without a cache around
#    it -- a cache that a later `hf` call in this account could garbage-collect;
#  * curl takes the revision in the URL, so every byte is pinned to one commit and a
#    re-run after the hub moves `main` downloads nothing new and still verifies;
#  * it keeps the download anonymous without having to trust a library's token lookup.
#
# 2026-11-24, the download is anonymous ON PURPOSE: the repository is ungated, and the
# token stored in this account belongs to its owner and is not this work's to use.  No
# Authorization header is ever sent, HF_HOME is pointed inside ROOT so that no library
# called later writes into the account's own cache, and the only identity in the request
# is curl's own default User-Agent.
set -euo pipefail

ROOT=/home/yamakawa/Development/semindex-learn
REPO=google/gemma-4-E2B-it
DEST=$ROOT/models/gemma-4-E2B-it
API=https://huggingface.co/api/models/$REPO

# The size the brief states for the weight file, as a literal.  It is here so that a
# repository that silently grew a second shard, or a truncated transfer that the hub's own
# reported size would also describe, stops the script instead of being tidied away.
EXPECT_BYTES=10246621918

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_API_TOKEN || true
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME=$ROOT/hf-home
mkdir -p "$HF_HOME" "$DEST"

# ---------------------------------------------------------------------------
# 1. The commit
# ---------------------------------------------------------------------------
# The revision is recorded on the first run and reused afterwards, because "the commit the
# hub reports now" has to mean one commit for the whole of this work: if `main` moved
# between two runs of this script the directory would hold a mix of two revisions and
# nothing would say so.
REVFILE=$DEST/REVISION
if [ -s "$REVFILE" ]; then
    REV=$(cat "$REVFILE")
    echo "revision (recorded by an earlier run): $REV"
else
    REV=$(curl -sS --fail --max-time 60 "$API" \
          | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')
    [ -n "$REV" ] || { echo "FAIL: the hub reported no revision" >&2; exit 1; }
    printf '%s\n' "$REV" > "$REVFILE"
    echo "revision (read from the hub now): $REV"
fi

# ---------------------------------------------------------------------------
# 2. What the commit holds, and how big each file is
# ---------------------------------------------------------------------------
# Both lists come from the API at the pinned revision, not from a literal: a hand-written
# file list would quietly skip a file the repository gained.
curl -sS --fail --max-time 60 "$API/revision/$REV" > "$DEST/.api_model.json"
mapfile -t FILES < <(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
for s in d["siblings"]:
    print(s["rfilename"])
' "$DEST/.api_model.json")
echo "files in the commit: ${#FILES[@]}"

PATHS_JSON=$(python3 -c '
import json, sys
print(json.dumps({"paths": sys.argv[1:], "expand": True}))
' "${FILES[@]}")
curl -sS --fail --max-time 120 -X POST "$API/paths-info/$REV" \
     -H "Content-Type: application/json" -d "$PATHS_JSON" > "$DEST/.api_paths.json"

# ---------------------------------------------------------------------------
# 3. Fetch each file at that revision
# ---------------------------------------------------------------------------
# `-C -` resumes a part file, so an interrupted 9.5 GiB transfer does not start over; the
# rename happens only after the size matches, so an interrupted run never leaves a short
# file under the real name for a later run to accept.
for f in "${FILES[@]}"; do
    want=$(python3 -c '
import json, sys
for e in json.load(open(sys.argv[1])):
    if e["path"] == sys.argv[2]:
        print(e.get("lfs", {}).get("size") or e["size"])
        break
else:
    raise SystemExit("the hub reported no entry for " + sys.argv[2])
' "$DEST/.api_paths.json" "$f")
    out=$DEST/$f
    mkdir -p "$(dirname "$out")"
    have=$(stat -c %s "$out" 2>/dev/null || echo 0)
    if [ "$have" = "$want" ]; then
        echo "have   $f ($have bytes)"
        continue
    fi
    echo "get    $f ($want bytes)"
    curl -L --fail --silent --show-error --retry 5 --retry-delay 5 --retry-all-errors \
         -C - -o "$out.part" "https://huggingface.co/$REPO/resolve/$REV/$f"
    got=$(stat -c %s "$out.part")
    if [ "$got" != "$want" ]; then
        echo "FAIL: $f came down as $got bytes, the hub says $want" >&2
        exit 1
    fi
    mv "$out.part" "$out"
done

# ---------------------------------------------------------------------------
# 4. The weight file, checked three ways
# ---------------------------------------------------------------------------
W=$DEST/model.safetensors
SIZE=$(stat -c %s "$W")
HUB_SHA=$(python3 -c '
import json, sys
for e in json.load(open(sys.argv[1])):
    if e["path"] == "model.safetensors":
        print(e["lfs"]["oid"])
        break
else:
    raise SystemExit("the hub reported no entry for model.safetensors")
' "$DEST/.api_paths.json")
LOCAL_SHA=$(sha256sum "$W" | cut -d" " -f1)

echo
echo "revision        $REV"
echo "size            $SIZE"
echo "size expected   $EXPECT_BYTES"
echo "sha256 local    $LOCAL_SHA"
echo "sha256 hub lfs  $HUB_SHA"

# Three checks, each able to fail on its own: the brief's literal size, the hub's size as
# reported for this commit, and the content digest against the hub's LFS object id.  They
# are not redundant -- the first catches "the repository is not the file this work was
# planned around", the third catches a transfer that is the right length and wrong bytes.
fail=0
[ "$SIZE" = "$EXPECT_BYTES" ] || { echo "FAIL size: $SIZE != $EXPECT_BYTES" >&2; fail=1; }
[ "$LOCAL_SHA" = "$HUB_SHA" ] || { echo "FAIL sha256: $LOCAL_SHA != $HUB_SHA" >&2; fail=1; }
# A symlink here would mean the directory is a view of a cache that something else owns
# and can empty, which is the thing this script exists to avoid.
for f in "${FILES[@]}"; do
    if [ -L "$DEST/$f" ]; then
        echo "FAIL: $f is a symlink" >&2
        fail=1
    fi
done
[ "$fail" = 0 ] || exit 1
echo "OK: $DEST holds the commit $REV as plain files"
