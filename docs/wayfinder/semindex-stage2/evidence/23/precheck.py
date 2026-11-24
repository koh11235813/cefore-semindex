#!/usr/bin/env python3
"""The four measurements the tuning plan of L rests on, on the tuning host.

    cd $ROOT && env PYTHONDONTWRITEBYTECODE=1 STORE=$ROOT/store \
        PYTHONPATH=$ROOT/repo/src HF_HUB_OFFLINE=1 ./.venv/bin/python -B \
        repo/docs/wayfinder/semindex-stage2/evidence/23/precheck.py $ROOT

WHAT IT MEASURES, AND WHY EACH NUMBER IS NEEDED BEFORE THE PLAN IS FROZEN
  1 READ-OUT NOISE.  A node makes single passes; an evaluation of 128 000+ passes would
    like to batch.  smoke.txt showed a batched pass differing from a single pass by up to
    about one logit in bf16 even with NO padding.  If that noise can flip the argmax of
    the three candidates, then batching changes the decisions and the replay score is not
    the score of the decider a node would run.  Measured here on 512 rows, in bf16 and in
    float32, at batch 1 and batch 8, plus whether one process repeats another's bits.
  2 PARALLEL THROUGHPUT at batch 1.  An evaluation that keeps the node's own read-out can
    still run several replay processes against one card.  This says how many passes per
    second K processes get, and what 128 000 passes would then cost.
  3 THE LEGEND LINE.  The rendered observation ends with a fixed legend sentence of about
    86 tokens.  This is what dropping it would save per pass.  No recommendation is drawn
    here; the plan decides.
  4 ONE TUNING STEP.  Seconds per optimiser step and peak memory at batch 8 and batch 16,
    so that the cost of an epoch is arithmetic and not a guess.

WHAT IT DOES NOT DO
  It does not score a tuned model in the replay and does not compute any tuned model's
  agreement with the labels: the plan is frozen first, and this script only measures what
  the plan needs in order to be written.  The adapter of part 4 exists for the duration of
  60 optimiser steps and is never written to disk; nothing tuned is kept.
  It uses seeds 1-80 ONLY (the tuning half).  Seeds 81-100 are the selection half and are
  not touched here; seeds above 100 are never used anywhere.

THE PROMPT WORDING IS STILL PROVISIONAL.  The instruction sentence and the answer prefix
are smoke.py's, and they live in prefill.py, which this script imports rather than copies.
A later step fixes the wording, so every timing and token count below moves with it.

WORKER MODES (the script re-runs itself; see parts 1 and 2)
    precheck.py <ROOT> --worker-identity <tmpdir>
    precheck.py <ROOT> --worker-pass <tmpdir> <index>
"""
import os
import sys

# 2026-11-24: FIRST import, before torch / numpy / transformers.  prefill.py sets the
# thread caps, the offline HF variables, ROOT and the two sys.path entries at ITS import
# time, and holds the wording, the prefill and the left-padding builder that smoke.py and
# this script must share.  A second copy of the prefill would let the smoke report and the
# numbers the plan rests on drift apart without either script noticing.
import prefill as P  # noqa: E402

ROOT = P.ROOT
MODEL_DIR = P.MODEL_DIR
LABELS = P.LABELS

import gzip        # noqa: E402
import json        # noqa: E402
import random      # noqa: E402
import re          # noqa: E402
import shutil      # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time        # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
MAX_TUNE_SEED = 80           # the tuning half; 81-100 select, >100 never
NOISE_ROWS = 512             # rows for part 1
NOISE_BATCH = 8              # the batch size part 1 compares against batch 1
TIMED_PASSES = 300           # parts 1 and 3
WARMUP_PASSES = 10
WORKER_PASSES = 200          # part 2, per process
WORKER_K = (1, 2, 4, 8)      # 8 is run only if the measured footprint allows it
SWEEP = 128_000              # the pass count of one candidate's evaluation (scout, part 4)
LORA_R, LORA_ALPHA = 16, 32
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
# The documented recipe's exclusion.  It is load-bearing and part 4 shows why: the audio
# tower has modules named q_proj/v_proj too, and they are Gemma4ClippableLinear, not
# nn.Linear.  Targeting by bare name alone would reach them.
LORA_EXCLUDE = r"model\.(vision_tower|audio_tower)\..*"
TUNE_STEPS = 30
TUNE_BATCHES = (8, 16)
LR = 2e-4
EPOCH_ROWS = (11_438, 4_600)  # candidate a (every labelled row of seeds 1-80) and b
DEV = "cuda:0"

MODE = sys.argv[2] if len(sys.argv) > 2 else ""
IS_WORKER = MODE.startswith("--worker")

OUT = []


def say(text=""):
    """Print and keep, so the whole report is one stream the caller can redirect."""
    print(text, flush=True)
    OUT.append(text)


def head(title):
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


class Fail(Exception):
    """A check that did not hold.

    An exception type of its own, because every check below is shown firing on a mutated
    input in part 5, and catching `Exception` there would also catch a typo in the
    mutation itself and print it as if the check had worked.
    """


# ---------------------------------------------------------------------------
# the checks, as functions, so that part 5 can feed each one a mutated input
# ---------------------------------------------------------------------------
def check_seeds_tuning(seeds, where):
    """Every seed used here is in the TUNING half, 1-80.

    port.check_seeds gates the measuring seeds (>100) and would let 81-100 through, so it
    cannot stand in for this: 81-100 are the selection half, and a measurement that
    quietly used them would make the later choice between candidates a and b circular.
    """
    bad = sorted({int(s) for s in seeds if int(s) > MAX_TUNE_SEED})
    if bad:
        raise Fail("%s: seeds above %d: %s" % (where, MAX_TUNE_SEED, bad[:8]))
    return sorted({int(s) for s in seeds})


LABEL_ARM = {3.695: 0, 6.0: 1, 10.043: 2}


def target_of(label):
    """The class index of a label.

    A dict and not a nearest-match: a label that is not one of the three arms is a row
    this loss cannot represent, and a KeyError at the first such row is the report.  A
    tolerance here would quietly train on a fourth arm as if it were one of the three.
    """
    return LABEL_ARM[label]


def check_finite(values, where):
    """No loss is NaN or inf."""
    bad = [(i, v) for i, v in enumerate(values) if not (v == v and abs(v) != float("inf"))]
    if bad:
        raise Fail("%s: %d non-finite value(s), first at step %d: %r"
                   % (where, len(bad), bad[0][0], bad[0][1]))
    return True


def check_changed(before, after, where):
    """The adapter weights moved.

    An optimiser that silently updated nothing (a frozen adapter, a detached loss, a zero
    learning rate) would otherwise leave every other number in part 4 looking right.
    """
    import torch  # noqa: F401  unused here; left in place so that no executed line of the script whose output is archived beside it moves
    moved = {k: (after[k] - before[k]).abs().max().item() for k in before}
    still = sorted(k for k, v in moved.items() if v == 0.0)
    if still:
        raise Fail("%s: %d of %d adapter tensors did not move, e.g. %s"
                   % (where, len(still), len(before), still[:3]))
    return moved


def check_targets(names, model_named, n_layers, n_shared, where):
    """The LoRA target set is exactly the language model's own projection modules.

    Three branches, and each rejects a different way of getting this wrong.
      1. A target set that reached into the vision or audio tower, which this text-only
         read-out never runs.
      2. A model whose projection count is not the one this architecture implies.  7 per
         layer is WRONG for E2B: 2026-11-24, the last `num_kv_shared_layers` layers reuse
         the k/v of an earlier layer and transformers 5.24.0 gives them no k_proj and no
         v_proj module at all, so those layers have 5.  The arithmetic is spelled out so
         that a transformers version which changed it would stop here instead of quietly
         training a different set of modules.
      3. A target set that is not that set: a name list or regex that silently matched
         nothing in, say, the MLP would still give a model that trains and a loss that
         falls, just not the documented recipe.
    """
    bad = sorted(n for n in names if not n.startswith("model.language_model.layers."))
    if bad:
        raise Fail("%s: %d module(s) outside the language model, e.g. %s"
                   % (where, len(bad), bad[:3]))
    full = len(LORA_TARGETS)
    want = full * (n_layers - n_shared) + (full - 2) * n_shared
    if len(model_named) != want:
        raise Fail("%s: the language model has %d projection modules, not %d x %d + %d x "
                   "%d = %d" % (where, len(model_named), full, n_layers - n_shared,
                                full - 2, n_shared, want))
    extra = sorted(set(names) - set(model_named))
    missed = sorted(set(model_named) - set(names))
    if extra or missed:
        raise Fail("%s: %d module(s) targeted that are not projections (%s) and %d "
                   "projection(s) not targeted (%s)"
                   % (where, len(extra), extra[:2], len(missed), missed[:2]))
    return True


def check_all_zero(params, where):
    """Every tensor handed in is exactly zero.

    Used on lora_B after the adapter is put back to its initial state: peft starts lora_B
    at exactly zero, so this is what "the next block starts untuned" means.  Without it, a
    reset that silently did nothing would leave the second block's losses continuing the
    first block's and reading like a batch-size effect.
    """
    nz = sorted(n for n, v in params.items() if float(v.abs().max()) != 0.0)
    if nz:
        raise Fail("%s: %d of %d tensors are not zero, e.g. %s"
                   % (where, len(nz), len(params), nz[:2]))
    return len(params)


def check_workers(results, k, where):
    """Exactly K workers came back.

    A worker that died would otherwise turn into a smaller K at a better-looking rate.
    """
    if len(results) != k:
        raise Fail("%s: %d of %d workers reported" % (where, len(results), k))
    return True


def check_identical(a, b, where):
    """Two tensors are equal BIT FOR BIT.

    Not a tolerance: the question part 1 asks about two processes is whether the same
    input gives the same bits, and a tolerance would answer a different question.
    """
    import torch
    if not torch.equal(a, b):
        d = (a.float() - b.float()).abs()
        raise Fail("%s: %d of %d values differ, max %.6g"
                   % (where, int((a != b).sum()), a.numel(), d.max().item()))
    return True


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def load_rows():
    with gzip.open(LABELS, "rb") as fh:
        return [json.loads(line) for line in fh]


def load_model(dtype, with_info=False):
    """The checkpoint, as the composite class.

    Gemma4ForCausalLM is NOT used: smoke.txt recorded it loading this checkpoint with 541
    missing keys, i.e. a partly random text tower that would still print plausible logits.
    """
    import torch  # noqa: F401  unused here; left in place so that no executed line of the script whose output is archived beside it moves
    from transformers import Gemma4ForConditionalGeneration
    if with_info:
        m, info = Gemma4ForConditionalGeneration.from_pretrained(
            MODEL_DIR, dtype=dtype, output_loading_info=True)
        return m.to(DEV).eval(), info
    m = Gemma4ForConditionalGeneration.from_pretrained(MODEL_DIR, dtype=dtype)
    return m.to(DEV).eval()


def forward_last(model, input_ids, attention_mask=None, position_ids=None):
    """Logits of the LAST position only, one forward pass, no cache."""
    import torch
    kwargs = {"input_ids": input_ids, "use_cache": False, "logits_to_keep": 1}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    with torch.inference_mode():
        return model(**kwargs).logits[:, -1, :]


def quantiles(values):
    """median, p99 and max of a list, as the three numbers every spread here reports."""
    v = sorted(values)
    p99 = v[min(len(v) - 1, int(0.99 * len(v)))]
    return statistics.median(v), p99, v[-1]


def free_gb():
    import torch
    free, total = torch.cuda.mem_get_info()
    return free / 2**30, total / 2**30


# ---------------------------------------------------------------------------
# worker modes
# ---------------------------------------------------------------------------
def worker_identity(tmpdir):
    """A SECOND PROCESS doing the same 512 bf16 batch-1 passes the parent did.

    Two passes inside one process share a warmed-up allocator, a chosen kernel and a
    cuBLAS workspace, so their agreeing says little about the evaluation, which would run
    in many processes.  This is the same question asked across a process boundary.
    """
    import torch
    from transformers import AutoTokenizer
    texts = json.load(open(os.path.join(tmpdir, "texts.json")))
    cand = json.load(open(os.path.join(tmpdir, "cand.json")))
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = load_model(torch.bfloat16)
    cand_ids = torch.tensor(cand, device=DEV)
    out = torch.empty(len(texts), 3, dtype=torch.float32)
    for i, text in enumerate(texts):
        ii = torch.tensor([P.ids_of(tok, text)], device=DEV)
        out[i] = forward_last(model, ii)[0, cand_ids].float().cpu()
    torch.save(out, os.path.join(tmpdir, "identity.pt"))


def worker_pass(tmpdir, index):
    """One of K processes making batch-1 passes on the same card at the same time.

    The barrier matters: without it the first worker would be timed while the last was
    still loading 10 GB of weights, and the rate would be the rate of a card that is not
    yet busy.  Each worker reports the wall clock of its own timed window, and the parent
    divides the total passes by the union of the windows.
    """
    import torch
    from transformers import AutoTokenizer
    prompt = open(os.path.join(tmpdir, "prompt.txt")).read()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = load_model(torch.bfloat16)
    ii = torch.tensor([P.ids_of(tok, prompt)], device=DEV)
    for _ in range(WARMUP_PASSES):
        forward_last(model, ii)
    torch.cuda.synchronize()
    open(os.path.join(tmpdir, "ready_%d" % index), "w").write("1")
    go = os.path.join(tmpdir, "go")
    deadline = time.time() + 900
    while not os.path.exists(go):
        if time.time() > deadline:
            raise SystemExit("worker %d: no go file after 900 s" % index)
        time.sleep(0.02)
    times = []
    t_start = time.time()
    for _ in range(WORKER_PASSES):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        forward_last(model, ii)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    t_end = time.time()
    json.dump({"index": index, "t_start": t_start, "t_end": t_end, "times": times},
              open(os.path.join(tmpdir, "res_%d.json" % index), "w"))


if IS_WORKER:
    if MODE == "--worker-identity":
        worker_identity(sys.argv[3])
    elif MODE == "--worker-pass":
        worker_pass(sys.argv[3], int(sys.argv[4]))
    else:
        raise SystemExit("unknown worker mode %r" % MODE)
    raise SystemExit(0)


# ===========================================================================
# PARENT
# ===========================================================================
head("0  THE CARD IS FREE, AND WHAT THIS RUN IS")
say("The prompt wording is STILL PROVISIONAL: the instruction sentence and the answer")
say("prefix are smoke.py's, shared through prefill.py.  Every token count and every")
say("timing below moves when a later step fixes the wording.")
say()
say("Nothing tuned is scored here and no agreement with the labels is computed: this run")
say("measures what the tuning plan needs in order to be written, and the plan is frozen")
say("before anything is tuned for keeps.")
say()

# The host is shared and has no scheduler, so this is the only thing standing between this
# script and someone else's job.  It is asked ONCE, here: from part 1 on, the processes on
# the card are this script's own workers, and a per-load check would then stop on them.
_apps = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
     "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
say("nvidia-smi compute apps: %s" % (_apps if _apps else "(none)"))
if _apps:
    say("STOP: another compute process is on the GPU; nothing was run.")
    raise SystemExit(2)
say("OK: no other compute process on the GPU")

import numpy   # noqa: E402
import torch   # noqa: E402

torch.set_num_threads(8)

import transformers  # noqa: E402
import peft          # noqa: E402
import torch.nn.functional as F  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("FAIL: torch does not see a CUDA device")

say()
say("torch %s (cuda %s), transformers %s, peft %s, numpy %s, python %s"
    % (torch.__version__, torch.version.cuda, transformers.__version__,
       peft.__version__, numpy.__version__, sys.version.split()[0]))
say("gpu %s, %.2f GB free of %.2f GB" % ((torch.cuda.get_device_name(0),) + free_gb()))

TMP = os.path.join(ROOT, "precheck-tmp")
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP)
ROOT_BEFORE = sorted(os.listdir(ROOT))

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
say(P.prepare_padding(tok))

# ---------------------------------------------------------------------------
# the rows: seeds 1-80 only
# ---------------------------------------------------------------------------
head("0b  THE ROWS, AND THE SEED GUARD")

rows = load_rows()
say("label rows      %d from %s" % (len(rows), os.path.basename(LABELS)))
say("seeds in file   min %d max %d" % (min(r["seed"] for r in rows),
                                       max(r["seed"] for r in rows)))

import port as PORT  # noqa: E402  (the replay's seed gate, and the >100 bound)

tune_rows = [r for r in rows if r["seed"] <= MAX_TUNE_SEED]
tune_labelled = [r for r in tune_rows if r["label"] is not None]
say("rows of seeds 1-%d        %d, of which labelled %d"
    % (MAX_TUNE_SEED, len(tune_rows), len(tune_labelled)))
say("the plan's candidate a is %d labelled rows; this file gives %d"
    % (EPOCH_ROWS[0], len(tune_labelled)))
if len(tune_labelled) != EPOCH_ROWS[0]:
    say("DIFFERS from the plan's number: the epoch arithmetic below uses %d, the file's own"
        % len(tune_labelled))

ALL_SEEDS = check_seeds_tuning([r["seed"] for r in tune_rows], "the rows this run reads")
PORT.check_seeds(ALL_SEEDS, "precheck.py")
say("seed range used %d-%d, %d distinct seeds; port.check_seeds passed them"
    % (ALL_SEEDS[0], ALL_SEEDS[-1], len(ALL_SEEDS)))
say("seeds 81-100 (selection) and 121-160 (measuring) appear nowhere: the row list is cut")
say("at seed %d before anything is rendered." % MAX_TUNE_SEED)

# 512 rows spread evenly: a stride over the file, which is seed-major, rather than the
# first 512, which would be two seeds and would read like a measurement over 80.
stride = max(1, len(tune_rows) // NOISE_ROWS)
noise_rows = tune_rows[::stride][:NOISE_ROWS]
if len(noise_rows) != NOISE_ROWS:
    raise SystemExit("FAIL: %d rows picked, not %d" % (len(noise_rows), NOISE_ROWS))
per_seed = {}
for r in noise_rows:
    per_seed[r["seed"]] = per_seed.get(r["seed"], 0) + 1
check_seeds_tuning([r["seed"] for r in noise_rows], "the 512 rows of part 1")
say("part 1 rows     %d, over %d seeds, %d-%d rows per seed (evenly: shown, not asserted)"
    % (len(noise_rows), len(per_seed), min(per_seed.values()), max(per_seed.values())))

t0 = time.time()
NOISE_TEXTS = [P.chat_text(tok, P.rendered(r)) for r in noise_rows]
NOISE_LENS = [len(P.ids_of(tok, t)) for t in NOISE_TEXTS]
say("rendered and tokenized the 512 in %.1f s: %d to %d tokens, median %d"
    % (time.time() - t0, min(NOISE_LENS), max(NOISE_LENS),
       int(statistics.median(NOISE_LENS))))

CAND_IDS_LIST = []
for word in P.CANDIDATES:
    base = P.ids_of(tok, NOISE_TEXTS[0])
    longer = P.ids_of(tok, NOISE_TEXTS[0] + word)
    if longer[:len(base)] != base or len(longer) != len(base) + 1:
        raise SystemExit("FAIL: %r is not one token at this position" % word)
    CAND_IDS_LIST.append(longer[-1])
say("candidate ids   %s for %s" % (CAND_IDS_LIST, list(P.CANDIDATES)))
CAND_IDS = torch.tensor(CAND_IDS_LIST, device=DEV)

# The median-length row, used by the timings of parts 1, 2 and 3.
_med = int(statistics.median(NOISE_LENS))
MED_I = min(range(len(NOISE_LENS)), key=lambda i: abs(NOISE_LENS[i] - _med))
MED_ROW = noise_rows[MED_I]
MED_TEXT = NOISE_TEXTS[MED_I]
say("timing row      seed %d k %d, %d tokens (median of the 512 is %d)"
    % (MED_ROW["seed"], MED_ROW["k"], NOISE_LENS[MED_I], _med))

# The batch-8 grouping is SHUFFLED, not consecutive: consecutive rows of the stride tend
# to sit at the same position of their runs and come out nearly the same length, which
# would hand the batch almost no padding and make the padded comparison vacuous.
ORDER = list(range(NOISE_ROWS))
random.Random(0).shuffle(ORDER)


def pass_batch1(model, texts):
    """The three candidate logits of every text, one pass per text."""
    out = torch.empty(len(texts), 3, dtype=torch.float32)
    for i, text in enumerate(texts):
        ii = torch.tensor([P.ids_of(tok, text)], device=DEV)
        out[i] = forward_last(model, ii)[0, CAND_IDS].float().cpu()
    return out


def pass_batch8(model, texts):
    """The same, in mixed left-padded batches of 8; returns the logits AND the padding share."""
    out = torch.empty(len(texts), 3, dtype=torch.float32)
    pad_tokens = real_tokens = 0
    for s in range(0, len(ORDER), NOISE_BATCH):
        idx = ORDER[s:s + NOISE_BATCH]
        iid, attn, pos, lengths = P.batch_of(tok, [texts[i] for i in idx], DEV)
        last = forward_last(model, iid, attn, pos)
        for j, i in enumerate(idx):
            out[i] = last[j, CAND_IDS].float().cpu()
        real_tokens += int(lengths.sum())
        pad_tokens += int(iid.numel()) - int(lengths.sum())
    return out, pad_tokens / float(pad_tokens + real_tokens)


def timed(model, text, n=TIMED_PASSES):
    ii = torch.tensor([P.ids_of(tok, text)], device=DEV)
    for _ in range(WARMUP_PASSES):
        forward_last(model, ii)
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        forward_last(model, ii)
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t) * 1000.0)
    return out


def timed_pair(model, text_a, text_b, n=TIMED_PASSES):
    """Two prompts timed in ONE ALTERNATING loop.

    2026-11-24: the first run timed the two legend variants in two separate loops minutes
    apart, and the shorter prompt came out slower.  Measured that way, the sign of a
    sub-millisecond difference can come from the clocks, the fan or anything else that
    drifts between the loops.  Alternating puts both prompts in the same conditions, pass
    by pass, so a difference that survives is a difference between the prompts.
    """
    ia = torch.tensor([P.ids_of(tok, text_a)], device=DEV)
    ib = torch.tensor([P.ids_of(tok, text_b)], device=DEV)
    for _ in range(WARMUP_PASSES):
        forward_last(model, ia)
        forward_last(model, ib)
    torch.cuda.synchronize()
    a, b = [], []
    for _ in range(n):
        for ii, acc in ((ia, a), (ib, b)):
            torch.cuda.synchronize()
            t = time.perf_counter()
            forward_last(model, ii)
            torch.cuda.synchronize()
            acc.append((time.perf_counter() - t) * 1000.0)
    return a, b


FAILURES = []


def guard(where, fn):
    """Run a part; a check that fires is reported and the run goes on.

    A failing check must not also skip the cleanup and the rest of the report: that would
    turn one failure into two.  Every failure is collected and the script exits 1.
    """
    try:
        return fn()
    except Fail as exc:
        say("FAIL in %s: %s" % (where, exc))
        FAILURES.append("%s: %s" % (where, exc))
    except torch.cuda.OutOfMemoryError as exc:
        say("FAIL in %s: out of memory -- %s" % (where, str(exc).split("\n")[0]))
        FAILURES.append("%s: out of memory" % where)
        torch.cuda.empty_cache()
    return None


# ===========================================================================
# 1  READ-OUT NOISE
# ===========================================================================
head("1  READ-OUT NOISE: bf16 / float32 x batch 1 / batch 8, on 512 rows")
say("The decision is argmax over the three candidate logits.  What matters is not the")
say("size of the logit difference but whether it reaches the margin, so both are here.")
say()

settings = {}

t0 = time.time()
model = load_model(torch.bfloat16)
say("bf16 model loaded in %.1f s, %.2f GB free of %.2f GB"
    % ((time.time() - t0,) + free_gb()))
say("attention implementation: %s" % model.config._attn_implementation)

t0 = time.time()
settings["bf16 b1"] = pass_batch1(model, NOISE_TEXTS)
say("bf16 batch 1    512 passes in %.1f s" % (time.time() - t0))
t0 = time.time()
settings["bf16 b8"], pad_share = pass_batch8(model, NOISE_TEXTS)
say("bf16 batch 8    64 mixed batches in %.1f s, padding %.1f%% of the tokens"
    % (time.time() - t0, pad_share * 100))

BF16_MS = timed(model, MED_TEXT)
m, p99, mx = quantiles(BF16_MS)
say("bf16 batch 1    single pass over %d passes: median %.2f ms, p99 %.2f ms, max %.2f ms"
    % (TIMED_PASSES, m, p99, mx))

# ---- part 3 runs here, while the bf16 model is the one that is loaded ----
head("3  THE LEGEND LINE")
say("The rendered observation ends with a fixed legend sentence.  This is what it costs")
say("per pass.  No recommendation is drawn: the plan decides whether the reader needs it.")
say()
MED_NOLEG = P.chat_text(tok, P.rendered(MED_ROW, legend=False))
n_with = len(P.ids_of(tok, MED_TEXT))
n_without = len(P.ids_of(tok, MED_NOLEG))
WITH_MS, NOLEG_MS = timed_pair(model, MED_TEXT, MED_NOLEG)
mw, pw, xw = quantiles(WITH_MS)
mn, pn, xn = quantiles(NOLEG_MS)
say("with legend     %d tokens   median %.2f ms  p99 %.2f ms  max %.2f ms"
    % (n_with, mw, pw, xw))
say("without legend  %d tokens   median %.2f ms  p99 %.2f ms  max %.2f ms"
    % (n_without, mn, pn, xn))
say("the legend is %d tokens of this row, %.1f %% of them" % (n_with - n_without,
                                                              100.0 * (n_with - n_without) / n_with))
saved = mw - mn
if saved > 0:
    say("dropping it saves %.2f ms of the median pass (%.1f %%), %.1f min over %d passes"
        % (saved, 100.0 * saved / mw, SWEEP * saved / 1000 / 60, SWEEP))
else:
    say("DROPPING IT DID NOT SHORTEN THE PASS: the median was %.2f ms with the legend and"
        % mw)
    say("%.2f ms without it, so the shorter prompt came out %.2f ms SLOWER." % (mn, -saved))
say("Either way the difference is to be read against this row's own spread, median to p99")
say("%.2f ms with the legend and %.2f ms without: a pass of %d tokens on this card is not"
    % (pw - mw, pn - mn, n_with))
say("paid for by the token at this length.  No recommendation is drawn; the plan decides.")

# ---- the second process, while the bf16 model is still loaded ----
head("1b  IS bf16 BATCH 1 BIT-IDENTICAL ACROSS TWO PROCESSES?")
json.dump(NOISE_TEXTS, open(os.path.join(TMP, "texts.json"), "w"))
json.dump(CAND_IDS_LIST, open(os.path.join(TMP, "cand.json"), "w"))
open(os.path.join(TMP, "prompt.txt"), "w").write(MED_TEXT)
t0 = time.time()
log = open(os.path.join(TMP, "identity.log"), "w")
rc = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), ROOT,
                     "--worker-identity", TMP], stdout=log, stderr=subprocess.STDOUT).returncode
log.close()
if rc != 0:
    say("FAIL: the second process exited %d; its log is %s/identity.log" % (rc, TMP))
    raise SystemExit(3)
other = torch.load(os.path.join(TMP, "identity.pt"))
say("a second process repeated the 512 bf16 batch-1 passes in %.1f s" % (time.time() - t0))
try:
    check_identical(settings["bf16 b1"], other, "two processes, bf16 batch 1")
    say("bit-identical across the two processes: all 512 x 3 values equal")
    CROSS_PROCESS = "identical"
except Fail as exc:
    say("NOT bit-identical across processes: %s" % exc)
    CROSS_PROCESS = str(exc)
settings["bf16 b1 (2nd process)"] = other

del model
torch.cuda.empty_cache()

# ---- float32 ----
head("1c  THE SAME 512 ROWS IN float32")
# TF32 would make a "float32" column that is not float32: the matmuls would run at 10
# mantissa bits.  Both switches are set and printed, so the column is what it says.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
say("matmul precision %r, allow_tf32 matmul %s / cudnn %s"
    % (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
       torch.backends.cudnn.allow_tf32))
t0 = time.time()
model = load_model(torch.float32)
say("float32 model loaded in %.1f s, parameter dtype %s, %.2f GB free of %.2f GB"
    % ((time.time() - t0, next(model.parameters()).dtype) + free_gb()))
t0 = time.time()
settings["fp32 b1"] = pass_batch1(model, NOISE_TEXTS)
say("fp32 batch 1    512 passes in %.1f s" % (time.time() - t0))
t0 = time.time()
settings["fp32 b8"], pad32 = pass_batch8(model, NOISE_TEXTS)
say("fp32 batch 8    64 mixed batches in %.1f s, padding %.1f%% of the tokens"
    % (time.time() - t0, pad32 * 100))
FP32_MS = timed(model, MED_TEXT)
m32, p3299, mx32 = quantiles(FP32_MS)
say("fp32 batch 1    single pass over %d passes: median %.2f ms, p99 %.2f ms, max %.2f ms"
    % (TIMED_PASSES, m32, p3299, mx32))
say("beside bf16 batch 1: median %.2f ms, p99 %.2f ms.  float32 is %.2fx the bf16 median."
    % (m, p99, m32 / m))
if m32 < m:
    say("SURPRISE, and it is not a typo: float32 is the FASTER of the two here, by %.2f ms"
        % (m - m32))
    say("at batch 1, with tf32 off.  A single pass of %d tokens is latency-bound on this"
        % NOISE_LENS[MED_I])
    say("card, so the narrower dtype buys nothing and the bf16 path's own conversions cost")
    say("something.  This is a measurement, not an explanation: nothing here identifies")
    say("the kernel that is slower.  float32 needs %.0f GB more of the card."
        % (20.5 - 10.1))
del model
torch.cuda.empty_cache()

# ---- the comparison ----
head("1d  WHAT THE FOUR SETTINGS DISAGREE ABOUT")
order = ["bf16 b1", "bf16 b8", "fp32 b1", "fp32 b8"]
EXACTER = {"bf16 b1": 1, "bf16 b8": 0, "fp32 b1": 3, "fp32 b8": 2}  # higher = more exact

margins = {}
for name in order:
    v = settings[name]
    top2 = v.topk(2, dim=-1).values
    margins[name] = (top2[:, 0] - top2[:, 1])
    mm, mp99, mmax = quantiles(margins[name].tolist())
    smallest = sorted(margins[name].tolist())[:1][0]
    arm = v.argmax(dim=-1)
    say("%-8s margin median %7.4f  p99 %7.4f  max %7.4f  min %7.4f   arms chosen "
        "short %d mid %d long %d"
        % (name, mm, mp99, mmax, smallest,
           int((arm == 0).sum()), int((arm == 1).sum()), int((arm == 2).sum())))

say()
say("The arm counts are the UNTUNED model's and are not evidence of anything: they say")
say("which word the base model happens to prefer after this provisional instruction.  They")
say("are here only so that the argmax differences below have a denominator a reader can")
say("see.  The margin is what the differences have to be read against.")
say()
say("Each pair: how many of the 512 argmax decisions differ, the spread of the absolute")
say("logit difference over all 512 x 3 values, and how many rows have a margin smaller")
say("than the largest difference that pair showed.  The margin is taken from the MORE")
say("EXACT setting of the pair (float32 over bf16, batch 1 over batch 8), so the count is")
say("not inflated by the noise it is being compared against.")
say()
BIGGEST = 0.0
for i, a in enumerate(order):
    for b in order[i + 1:]:
        d = (settings[a] - settings[b]).abs()
        dm, dp99, dmax = quantiles(d.flatten().tolist())
        flips = int((settings[a].argmax(-1) != settings[b].argmax(-1)).sum())
        ref = a if EXACTER[a] > EXACTER[b] else b
        at_risk = int((margins[ref] < dmax).sum())
        BIGGEST = max(BIGGEST, dmax)
        say("%-8s vs %-8s  argmax differs on %3d of 512   |diff| median %.4f p99 %.4f "
            "max %.4f   rows with margin < %.4f: %d (%.1f %%, margin of %s)"
            % (a, b, flips, dm, dp99, dmax, dmax, at_risk,
               100.0 * at_risk / NOISE_ROWS, ref))
say()
say("largest difference seen anywhere among the pairs: %.4f" % BIGGEST)
say("rows whose bf16 batch-1 margin is below it: %d of 512 (%.1f %%)"
    % (int((margins["bf16 b1"] < BIGGEST).sum()),
       100.0 * int((margins["bf16 b1"] < BIGGEST).sum()) / NOISE_ROWS))

# ===========================================================================
# 2  PARALLEL THROUGHPUT AT BATCH 1
# ===========================================================================
head("2  PARALLEL THROUGHPUT AT BATCH 1, K PROCESSES ON ONE CARD")
say("Every process keeps the node's own read-out (batch 1) and loads its own copy of the")
say("weights.  The parent holds no model while these run.  The aggregate rate divides the")
say("total passes by the union of the workers' own timed windows, so a worker that was")
say("still loading cannot make the rate look better.")
say()
free_now, total_now = free_gb()
say("card before part 2: %.2f GB free of %.2f GB" % (free_now, total_now))

throughput = {}
footprint = None
for k in WORKER_K:
    if k == 8 and footprint is not None and 8 * footprint > free_now:
        say("K = 8 NOT RUN: one worker measured %.2f GB, 8 of them would need %.2f GB and "
            "the card has %.2f GB free." % (footprint, 8 * footprint, free_now))
        continue
    d = os.path.join(TMP, "k%d" % k)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "prompt.txt"), "w").write(MED_TEXT)
    procs, logs = [], []
    for i in range(k):
        lf = open(os.path.join(d, "w%d.log" % i), "w")
        logs.append(lf)
        procs.append(subprocess.Popen(
            [sys.executable, "-B", os.path.abspath(__file__), ROOT, "--worker-pass", d,
             str(i)], stdout=lf, stderr=subprocess.STDOUT))
    deadline = time.time() + 900
    while sum(os.path.exists(os.path.join(d, "ready_%d" % i)) for i in range(k)) < k:
        for p in procs:
            if p.poll() not in (None, 0):
                say("FAIL: a K = %d worker exited %d before the barrier; logs in %s"
                    % (k, p.returncode, d))
                raise SystemExit(3)
        if time.time() > deadline:
            say("FAIL: K = %d workers did not all reach the barrier in 900 s" % k)
            raise SystemExit(3)
        time.sleep(0.2)
    if k == 1:
        # The footprint of ONE worker, measured on the card rather than estimated from the
        # file size, because it decides whether K = 8 is run at all.
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, check=True).stdout.strip()
        mib = [int(x) for x in re.findall(r"(\d+)\s*MiB", q)]
        footprint = max(mib) / 1024.0 if mib else None
        say("one worker's footprint on the card: %s  -> %.2f GB"
            % (q.replace("\n", " | "), footprint or -1))
    open(os.path.join(d, "go"), "w").write("1")
    for p in procs:
        p.wait()
    for lf in logs:
        lf.close()
    res = []
    for i in range(k):
        f = os.path.join(d, "res_%d.json" % i)
        if os.path.exists(f):
            res.append(json.load(open(f)))
    check_workers(res, k, "K = %d" % k)
    window = max(r["t_end"] for r in res) - min(r["t_start"] for r in res)
    rate = k * WORKER_PASSES / window
    worst = max(statistics.median(r["times"]) for r in res)
    best = min(statistics.median(r["times"]) for r in res)
    throughput[k] = rate
    say("K = %d  %6.1f passes/s total   per-worker median pass %.2f-%.2f ms   window %.2f s"
        % (k, rate, best, worst, window))
    say("        %d passes at this rate: %.1f min" % (SWEEP, SWEEP / rate / 60))

say()
if 1 in throughput:
    for k in sorted(throughput):
        say("K = %d is %.2fx the single-process rate" % (k, throughput[k] / throughput[1]))
say("Processes: at most %d workers plus this parent, within the brief's cap of 8."
    % max(throughput) if throughput else "")

# ===========================================================================
# 4  ONE TUNING STEP
# ===========================================================================
head("4  ONE TUNING STEP: LoRA r=%d alpha=%d, 3-way loss on the candidate logits"
     % (LORA_R, LORA_ALPHA))

model, load_info = load_model(torch.bfloat16, with_info=True)
n_layers = model.config.text_config.num_hidden_layers
n_shared = model.config.text_config.num_kv_shared_layers
say("text layers %d, vocab %d, kv-shared layers %d (from index %d)"
    % (n_layers, model.config.text_config.vocab_size, n_shared, n_layers - n_shared))

# What the bare name list alone would reach, i.e. what exclude_modules is for.  Collected
# before the adapter is attached, so part 5 can feed it to the same check that passes on
# the real target set.
suffixes = tuple("." + t for t in LORA_TARGETS)
ALL_NAMED = [n for n, _ in model.named_modules() if n.endswith(suffixes)]
LM_NAMED = [n for n in ALL_NAMED if n.startswith("model.language_model.layers.")]
TOWER_NAMED = [n for n in ALL_NAMED if n not in set(LM_NAMED)]
say("modules whose name ends in one of %s: %d in the whole model, %d of them outside the"
    % (LORA_TARGETS, len(ALL_NAMED), len(TOWER_NAMED)))
say("language model (e.g. %s).  That is what exclude_modules %r is for."
    % (TOWER_NAMED[:2], LORA_EXCLUDE))

# 2026-11-24, and this corrects the recipe this step started from: "7 projections per
# layer" is NOT true of E2B.  The last 20 of the 35 layers reuse the k/v of an earlier
# layer, and transformers 5.24.0 gives those layers no k_proj and no v_proj MODULE, so the
# language model has 205 projections and not 245.  A plan that budgeted the adapter at 7
# per layer would be wrong by 40 modules.
import collections  # noqa: E402

by_name = collections.Counter(n.split(".")[-1] for n in LM_NAMED)
say("language-model projections by name: %s" % dict(sorted(by_name.items())))
say("so %d, not %d x %d: the %d kv-shared layers have no k_proj and no v_proj module."
    % (len(LM_NAMED), len(LORA_TARGETS), n_layers, n_shared))
ck_missing = len(load_info.get("missing_keys") or [])
ck_unexpected = len(load_info.get("unexpected_keys") or [])
say("the checkpoint nevertheless ships k_proj and v_proj weights for all %d layers, and"
    % n_layers)
say("the loader reported missing_keys %d and unexpected_keys %d: %d checkpoint tensors"
    % (ck_missing, ck_unexpected, 2 * n_shared))
say("have no module in this model AND the loader did not say so.  Nothing here is random")
say("-- every module that exists got its weights -- but unexpected_keys cannot be read as")
say("a statement that every tensor of the file was used.")

from peft import LoraConfig, get_peft_model  # noqa: E402

# No task_type: it would wrap the model in PeftModelForCausalLM, whose forward rebuilds
# the kwargs.  Plain PeftModel.forward passes **kwargs straight to the base model
# (peft/peft_model.py), which is what keeps logits_to_keep and position_ids reaching
# Gemma4's forward -- the same call the read-out makes.
lcfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
                  target_modules=LORA_TARGETS, exclude_modules=LORA_EXCLUDE)
pm = get_peft_model(model, lcfg)

from peft.tuners.lora import LoraLayer  # noqa: E402

MATCHED = sorted(n.replace("base_model.model.", "", 1)
                 for n, mod in pm.named_modules() if isinstance(mod, LoraLayer))
guard("the LoRA target set",
      lambda: check_targets(MATCHED, LM_NAMED, n_layers, n_shared, "the LoRA target set"))
trainable, total = pm.get_nb_trainable_parameters()
adapter_dtypes = sorted({str(p.dtype) for n, p in pm.named_parameters() if "lora_" in n})
base_grad = [n for n, p in pm.named_parameters() if p.requires_grad and "lora_" not in n]
say("matched modules %d, exactly the language model's own projections, nothing else"
    % len(MATCHED))
say("e.g. %s" % MATCHED[0])
say("trainable params %d of %d (%.4f %%), adapter dtype %s, base params with grad %d"
    % (trainable, total, 100.0 * trainable / total, adapter_dtypes, len(base_grad)))
if base_grad:
    raise SystemExit("FAIL: %d base parameters are trainable: %s"
                     % (len(base_grad), base_grad[:3]))

tune_pool = list(tune_labelled)
random.Random(1).shuffle(tune_pool)
check_seeds_tuning([r["seed"] for r in tune_pool], "the rows of part 4")
say("tuning rows     %d labelled rows of seeds %d-%d"
    % (len(tune_pool), min(r["seed"] for r in tune_pool),
       max(r["seed"] for r in tune_pool)))

CURSOR = [0]


def next_batch(bs):
    """bs rows and their class indices, cycling through the shuffled labelled rows."""
    taken = []
    while len(taken) < bs:
        if CURSOR[0] >= len(tune_pool):
            CURSOR[0] = 0
        taken.append(tune_pool[CURSOR[0]])
        CURSOR[0] += 1
    texts = [P.chat_text(tok, P.rendered(r)) for r in taken]
    y = torch.tensor([target_of(r["label"]) for r in taken], device=DEV)
    return texts, y


def one_step(bs, opt):
    """One optimiser step; returns the loss value and the seconds it took.

    The forward is the SAME call the read-out makes (use_cache=False, logits_to_keep=1)
    and the loss is a 3-way cross-entropy over the candidate logits at the last position,
    so nothing is trained through a path that inference does not use.
    """
    texts, y = next_batch(bs)
    iid, attn, pos, _ = P.batch_of(tok, texts, DEV)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = pm(input_ids=iid, attention_mask=attn, position_ids=pos,  # noqa: F821  pm is bound at module scope, and the `del pm` at the end unbinds it
                 use_cache=False, logits_to_keep=1)
    loss = F.cross_entropy(out.logits[:, -1, :][:, CAND_IDS].float(), y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return loss.item(), time.perf_counter() - t0


def snapshot():
    return {n: p.detach().float().cpu().clone()
            for n, p in pm.named_parameters() if "lora_" in n and p.requires_grad}  # noqa: F821  pm is bound at module scope, and the `del pm` at the end unbinds it


# peft starts lora_B at exactly zero, so "B moved" is a real statement about the step and
# not about floating point.
def lora_B():
    return {n: p.detach().float().cpu().clone()
            for n, p in pm.named_parameters() if "lora_B" in n}  # noqa: F821  pm is bound at module scope, and the `del pm` at the end unbinds it


INIT_STATE = {n: p.detach().clone() for n, p in pm.named_parameters() if "lora_" in n}
say("adapter tensors at the initial state: %d, of which lora_B exactly zero: %d"
    % (len(INIT_STATE), check_all_zero(lora_B(), "the untuned adapter")))


def reset_adapter():
    """Put the adapter back to the state get_peft_model built.

    2026-11-24: without this the batch-16 block continued the batch-8 block's adapter and
    its printed loss started at 0.81 instead of at the untuned 4.25, which reads like a
    batch-size effect and is not one.  The timings did not depend on it; the loss column
    did.
    """
    with torch.no_grad():
        for n, p in pm.named_parameters():  # noqa: F821  pm is bound at module scope, and the `del pm` at the end unbinds it
            if n in INIT_STATE:
                p.copy_(INIT_STATE[n])


results = {}


def run_batch(bs):
    """30 optimiser steps at one batch size, from the untuned adapter, and what they cost."""
    reset_adapter()
    check_all_zero(lora_B(), "the adapter at the start of the batch %d block" % bs)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=LR)  # noqa: F821  pm is bound at module scope, and the `del pm` at the end unbinds it
    before = snapshot()
    torch.cuda.reset_peak_memory_stats()
    losses, secs = [], []
    for _ in range(TUNE_STEPS):
        lv, sec = one_step(bs, opt)
        losses.append(lv)
        secs.append(sec)
    check_finite(losses, "batch %d losses" % bs)
    moved = check_changed(before, snapshot(), "batch %d adapter" % bs)
    # The first steps pay for cuBLAS workspaces and an allocator that has not seen this
    # shape, so they are not what an epoch of this shape costs.
    steady = sorted(secs[3:])
    med, p99s, mx = quantiles(steady)
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    results[bs] = (med, peak)
    say()
    say("batch %2d  %d steps from the untuned adapter, loss first %.4f last %.4f, min %.4f "
        "max %.4f, all finite"
        % (bs, TUNE_STEPS, losses[0], losses[-1], min(losses), max(losses)))
    say("          seconds per step (steps 4-%d): median %.3f  p99 %.3f  max %.3f"
        % (TUNE_STEPS, med, p99s, mx))
    say("          peak allocated %.2f GB, peak reserved %.2f GB" % (peak, reserved))
    say("          adapter tensors moved: %d of %d, largest move %.3g"
        % (len(moved), len(before), max(moved.values())))
    for n_rows in (len(tune_labelled),) + EPOCH_ROWS[1:]:
        steps = -(-n_rows // bs)
        say("          one epoch over %d rows = %d steps = %.1f min"
            % (n_rows, steps, steps * med / 60))


for bs in TUNE_BATCHES:
    guard("the tuning steps at batch %d" % bs, lambda bs=bs: run_batch(bs))

# ===========================================================================
# 5  EVERY CHECK OF THIS SCRIPT, SHOWN FAILING ON A MUTATED INPUT
# ===========================================================================
head("5  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT")
say("The mutation is of the INPUT, inside this process, and not of a copy of the script:")
say("a per-check script copy would mean one 10-minute run of a shared card per check.")
say("prefill.batch_of has two checks and this script's batches go through both.  The first")
say("(left padding leaves a real token last) is mutated here.  The second (the last")
say("position is length-1) is an invariant of the builder's own arithmetic and no argument")
say("can break it, so it is shown by mutation k of RUN.txt section 7 instead.")
say()


def shows(name, fn):
    """Run a mutated input through a check and print the failure it produced."""
    try:
        fn()
    except Fail as exc:
        say("  %-22s FIRES: %s" % (name, exc))
        return True
    except KeyError as exc:
        say("  %-22s FIRES: KeyError %s" % (name, exc))
        return True
    except SystemExit as exc:
        say("  %-22s FIRES: %s" % (name, exc))
        return True
    say("  %-22s DID NOT FIRE -- the check cannot catch its own mutation" % name)
    return False


fired = []
fired.append(shows("seeds_tuning", lambda: check_seeds_tuning(ALL_SEEDS + [81], "mutated")))
fired.append(shows("port.check_seeds",
                   lambda: PORT.check_seeds([PORT.MAX_SEED + 1], "mutated")))
fired.append(shows("target_of", lambda: target_of(7.0)))
fired.append(shows("finite", lambda: check_finite([0.5, float("nan")], "mutated")))
_s = snapshot()
fired.append(shows("changed", lambda: check_changed(_s, _s, "mutated")))
fired.append(shows("targets/tower",
                   lambda: check_targets(ALL_NAMED, LM_NAMED, n_layers, n_shared, "mutated")))
fired.append(shows("targets/arithmetic",
                   lambda: check_targets(MATCHED, LM_NAMED, n_layers, 0, "mutated")))
fired.append(shows("targets/set",
                   lambda: check_targets(MATCHED[:-1], LM_NAMED, n_layers, n_shared, "mutated")))
fired.append(shows("all_zero (reset)", lambda: check_all_zero(lora_B(), "mutated")))
fired.append(shows("workers", lambda: check_workers([{}], 2, "mutated")))
_one = settings["bf16 b1"].clone()
_one[0, 0] += 1.0
fired.append(shows("identical", lambda: check_identical(settings["bf16 b1"], _one, "mutated")))


def _right_pad():
    """The shortest and the longest of the 512, so the batch really has padding to place."""
    i_short = min(range(NOISE_ROWS), key=lambda i: NOISE_LENS[i])
    i_long = max(range(NOISE_ROWS), key=lambda i: NOISE_LENS[i])
    tok.padding_side = "right"
    try:
        P.batch_of(tok, [NOISE_TEXTS[i_short], NOISE_TEXTS[i_long]], DEV)
    finally:
        tok.padding_side = "left"


fired.append(shows("batch_of/left-pad", _right_pad))
# The OTHER check inside prefill.batch_of, that the last position is length-1, cannot be
# made to fire by mutating an argument: it is an invariant of the builder's own arithmetic.
# It is shown firing by mutation k of RUN.txt section 7, which reverses the position_ids
# inside this same function -- the one prefill.py now holds and both scripts call.
say("  batch_of/position      not mutable from an argument; shown by RUN.txt section 7 k")

# A real step with lr = 0: the one mutation here that is of the TRAINING and not of an
# argument.  It is what an adapter that is attached but not actually optimised looks like.
opt0 = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=0.0)
b0 = snapshot()
one_step(8, opt0)
fired.append(shows("changed (lr=0 step)",
                   lambda: check_changed(b0, snapshot(), "mutated: lr=0")))

say()
if all(fired):
    say("all %d mutations produced a failure" % len(fired))
else:
    say("FAILURE OF THIS SECTION: %d of %d mutations did not produce a failure, so those"
        % (len(fired) - sum(fired), len(fired)))
    say("checks cannot catch what they exist to catch.")

# ===========================================================================
# 6  NOTHING TUNED IS KEPT
# ===========================================================================
head("6  THE ADAPTER IS GONE")
say("save_pretrained was never called: the adapter existed in this process's memory for")
say("%d optimiser steps and nowhere else." % (2 * TUNE_STEPS + 1))
del pm, model
torch.cuda.empty_cache()
shutil.rmtree(TMP, ignore_errors=True)
after = sorted(os.listdir(ROOT))
say("ROOT before this run: %s" % ROOT_BEFORE)
say("ROOT after  this run: %s" % after)
new = [x for x in after if x not in ROOT_BEFORE]
gone = [x for x in ROOT_BEFORE if x not in after]
if new or gone:
    say("CHANGED under ROOT: added %s, removed %s" % (new, gone))
else:
    say("no file was added or removed under ROOT by this run")

head("7  WHAT THIS RUN DOES NOT SAY")
say("- Nothing about which word a tuned model would answer: no tuned model was scored in")
say("  the replay and no agreement with the labels was computed.  That is the next step's")
say("  work, after the plan is frozen.")
say("- Nothing about the final wording: the instruction and the prefix are provisional.")
say("- The 30 steps per batch size say what a step COSTS, not that 30 steps or 2e-4 or")
say("  rank 16 are the right amount of tuning.  The loss falling over 30 steps is not")
say("  evidence of anything; it is a cost measurement that happens to print a loss.")
say("- Part 2's rate is for %d-token prompts of one shape.  A replay process also runs"
    % NOISE_LENS[MED_I])
say("  the simulation between passes, so its real rate is lower than this ceiling.")
say("- Part 1 compares settings against each other.  None of them is a ground truth: the")
say("  float32 read-out is the more exact arithmetic, not the right answer.")
say()
if FAILURES:
    say("THIS RUN FAILED: %d check(s) fired on real input, not on a mutation:"
        % len(FAILURES))
    for f in FAILURES:
        say("  %s" % f)
    say("end of precheck.py")
    raise SystemExit(1)
say("no check fired on real input; the %d failures printed in part 5 are the mutations"
    % sum(fired))
say("end of precheck.py")
