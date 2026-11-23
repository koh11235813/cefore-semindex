#!/usr/bin/env bash
# Build the one venv this work needs on the tuning host, and prove it can see the GPU.
#
# Runs ON THE TUNING HOST.  Idempotent: an existing venv is reused and `uv pip install`
# leaves a satisfied requirement alone, so a second run installs nothing and re-verifies.
#
#   bash hpc_setup.sh            # the venv, the packages, the two checks
#   bash hpc_setup.sh freeze     # print `uv pip freeze` and nothing else
set -euo pipefail

ROOT=/home/yamakawa/Development/semindex-learn
VENV=$ROOT/.venv
PY=$VENV/bin/python
UV=/home/yamakawa/.local/bin/uv

# 2026-11-24: /usr/bin/python3.13 by absolute path, not `--python 3.13`.  A bare version
# request lets uv DOWNLOAD an interpreter into ~/.local/share/uv, which is neither ROOT nor
# uv's cache, and this work is allowed to write in only those two places.
BASE_PY=/usr/bin/python3.13

# 2026-11-24: the cu132 wheel index, which is a decision with two alternatives rejected.
# PyTorch's install selector (its own data file, published_versions.json, which is what the
# page renders) offers for the current stable 2.15.0 on linux/pip exactly two CUDA options:
# plain PyPI, and `--index-url https://download.pytorch.org/whl/cu134`.  Neither is 13.2.
# The driver on this host reports CUDA 13.2, the cu132 index carries a
# torch-2.15.0+cu132-cp313 manylinux_2_28_x86_64 wheel, and that exact build is already
# proven on THIS card to list sm_120 and open the device.  cu134 would rest on CUDA minor
# version compatibility against an older driver, which nothing here has tested, so the
# index that matches the driver is the one taken.
TORCH_INDEX=https://download.pytorch.org/whl/cu132
TORCH_REQ="torch==2.15.0+cu132"

# numpy is pinned to the version the replay's numbers were measured with on the local
# machine, because a parity check between the two hosts that let numpy drift would be
# comparing two libraries as well as two machines.
NUMPY_REQ="numpy==2.5.4"

# transformers 5.24 is the floor because the gemma4 architecture and the
# `continue_final_message` prefill behaviour this work depends on were read at that tag;
# peft 0.19 is the floor because that is where its gemma4 target-module defaults landed.
# trl, bitsandbytes, datasets and the vision/audio extras are deliberately absent: the base
# is loaded in bf16 with no quantisation, the training loop is text-only, and an extra that
# is not imported is an extra that can still break an import.
PKGS=("transformers>=5.24" "peft>=0.19" "accelerate" "huggingface_hub" "safetensors")

# The download is anonymous and the account's own hub cache is not touched.  Set here as
# well as in fetch_weights.sh, because a transformers import resolves HF_HOME on its own.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_API_TOKEN || true
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME=$ROOT/hf-home

if [ "${1:-}" = "freeze" ]; then
    exec "$UV" pip freeze --python "$PY"
fi

mkdir -p "$ROOT" "$HF_HOME"

# ---------------------------------------------------------------------------
# 0. The GPU has to be free before anything opens a CUDA context
# ---------------------------------------------------------------------------
# This box is shared and has no scheduler, so the only thing standing between this work
# and somebody else's run is this check.  It is first, before the torch probe, because the
# probe itself takes memory on the card.  Display processes (type G) are the desktop's and
# are not compute.
echo "== nvidia-smi"
nvidia-smi
BUSY=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true)
if [ -n "$BUSY" ]; then
    echo "STOP: another compute process is on the GPU; nothing was run." >&2
    echo "$BUSY" >&2
    exit 2
fi
echo "GPU compute processes: none"

# ---------------------------------------------------------------------------
# 1. The venv
# ---------------------------------------------------------------------------
if [ -x "$PY" ]; then
    echo "venv: reusing $VENV ($("$PY" -V))"
else
    "$UV" venv --python "$BASE_PY" "$VENV"
fi

# ---------------------------------------------------------------------------
# 2. The packages, in two calls, because the two indexes cannot be one list
# ---------------------------------------------------------------------------
# uv's own index documentation: `--index-url` REPLACES PyPI as the default index rather
# than adding to it.  So one call with the cu132 index would hide transformers, peft and
# numpy, which that index does not serve, and one call with PyPI alone would hide the
# +cu132 local version.  Hence: torch and its nvidia/triton wheels from the cu132 index,
# everything else from PyPI.  `--extra-index-url` is NOT used to merge them, because an
# extra index outranks the default in uv and would let numpy come from the wheel mirror.
echo "== torch from $TORCH_INDEX"
"$UV" pip install --python "$PY" --index-url "$TORCH_INDEX" "$TORCH_REQ"
echo "== the rest from PyPI"
"$UV" pip install --python "$PY" "${PKGS[@]}" "$NUMPY_REQ"

# The second call resolves against PyPI, where a plain `torch` 2.15.0 also exists, so it
# could in principle replace the local-version build with the default one.  That would
# leave a venv that imports and then cannot see this card, which is exactly the failure
# this whole step exists to rule out, so it is asserted rather than assumed.
GOT_TORCH=$("$PY" -c 'import torch; print(torch.__version__)')
case "$GOT_TORCH" in
    *+cu132) echo "torch: $GOT_TORCH" ;;
    *) echo "FAIL: torch is $GOT_TORCH, not a +cu132 build" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# 3. The two checks the step is judged on
# ---------------------------------------------------------------------------
echo "== checks"
"$PY" - <<'PYEOF'
import sys

import torch

print("torch          ", torch.__version__)
print("torch cuda     ", torch.version.cuda)
arch = torch.cuda.get_arch_list()
print("arch list      ", arch)
print("cuda available ", torch.cuda.is_available())

fail = []
# Each check names one way this venv could be useless, and each can fail alone: a CPU-only
# wheel (no sm_), a wheel built before Blackwell (no sm_120), a driver/library mismatch
# (not available), and a transformers too old for the architecture (no gemma4 module).
if "sm_120" not in arch:
    fail.append(f"sm_120 is not in the arch list: {arch}")
if not torch.cuda.is_available():
    fail.append("torch.cuda.is_available() is False")
else:
    props = torch.cuda.get_device_properties(0)
    print("device         ", props.name, (props.major, props.minor))
    if (props.major, props.minor) != (12, 0):
        fail.append(f"the device reports capability {(props.major, props.minor)}, not (12, 0)")

try:
    import transformers
    import transformers.models.gemma4 as g4
except Exception as exc:                                  # noqa: BLE001
    fail.append(f"import transformers.models.gemma4 failed: {exc!r}")
else:
    print("transformers   ", transformers.__version__)
    print("gemma4 module  ", g4.__name__)

import numpy
print("numpy          ", numpy.__version__)
if numpy.__version__ != "2.5.4":
    fail.append(f"numpy is {numpy.__version__}, not the 2.5.4 the replay was measured with")

import peft, accelerate, huggingface_hub, safetensors
print("peft           ", peft.__version__)
print("accelerate     ", accelerate.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("safetensors    ", safetensors.__version__)

if fail:
    for f in fail:
        print("FAIL:", f, file=sys.stderr)
    raise SystemExit(1)
print("OK: the venv sees this GPU and carries the gemma4 architecture")
PYEOF
