#!/usr/bin/env python3
"""The first forward passes of Gemma 4 E2B on this card: prefill, candidates, timing.

Runs ON THE TUNING HOST, in the venv built by hpc_setup.sh:

    cd $ROOT && PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -B \
        repo/docs/wayfinder/semindex-stage2/evidence/23/smoke.py $ROOT

ROOT comes from argv[1], else $SMOKE_ROOT, else $ROOT.  Everything is read from under
ROOT, so the script needs no network: HF_HUB_OFFLINE=1 is set below, before transformers
is imported, and a stray download would then raise instead of reaching the hub.

WHAT THIS ANSWERS, AND WHAT IT DOES NOT
  It answers whether the read-out works at all: whether the checkpoint loads in bf16 on
  this card, whether the chat template leaves the prefilled answer prefix where we think
  it is, whether the three answer words are one token each AT THAT POSITION, whether one
  forward pass gives a stable logit for each of them, how long a pass takes, and how long
  a sweep of 128 000 or 256 000 passes would take.  The model is UNTUNED here, so nothing
  about WHICH word wins is evidence of anything; only the mechanics are.

THE INSTRUCTION AND THE ANSWER PREFIX ARE PROVISIONAL (they live in prefill.py).
  2026-11-24: a later step fixes the wording (it is part of what tuning and scoring are
  about).  They are kept short here so that the token counts this script prints are
  almost entirely the rendered observation, which is the part that is already fixed.
"""
import os
import sys

# 2026-11-24: FIRST import, before torch / numpy / transformers.  prefill.py sets the
# thread caps, the offline HF variables, ROOT and the two sys.path entries at ITS import
# time, and holds the wording, the prefill, the candidate words and the left-padding
# builder that this script and precheck.py must share.  A second copy of the prefill would
# let the two reports drift apart without either script noticing.
import prefill as P  # noqa: E402

ROOT = P.ROOT
REPO = P.REPO
EV22 = P.EV22
MODEL_DIR = P.MODEL_DIR
LABELS = P.LABELS

import gzip        # noqa: E402
import inspect     # noqa: E402
import json        # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time        # noqa: E402

# ---------------------------------------------------------------------------
# The provisional prompt wording lives in prefill.py (see the module docstring there)
# ---------------------------------------------------------------------------
INSTRUCTION = P.INSTRUCTION
PREFIX = P.PREFIX
CANDIDATES = P.CANDIDATES
SECONDS = P.SECONDS
EXPECTED_IDS = P.EXPECTED_IDS

EXPECT_ROWS = P.EXPECT_ROWS   # labels_p0.2.jsonl.gz, stated by the brief
EXPECT_SEEDS = P.EXPECT_SEEDS  # min and max seed of the main cell's label file
TIMED_PASSES = 300
WARMUP_PASSES = 10
BATCH_SIZES = (8, 32, 64)
BATCH_REPEATS = 20           # timed batched passes per size
SWEEP_SIZES = (128_000, 256_000)
FORWARD_ROWS = 5             # label rows for the single-pass read-out of step 4

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


# ---------------------------------------------------------------------------
# 0. the GPU must be free
# ---------------------------------------------------------------------------
def gpu_is_free():
    """Refuse to run if anyone else's compute process is on the card.

    The host is shared and has no scheduler, so this is the only thing standing between
    this script and someone else's job.  `--query-compute-apps` lists compute contexts
    only: the desktop's gnome-shell and Xwayland are graphics contexts and do not appear,
    which is why the display being on is not a reason to stop.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip()
    say("nvidia-smi compute apps: %s" % (out if out else "(none)"))
    return out == ""


head("0  THE CARD IS FREE")
if not gpu_is_free():
    say("STOP: another compute process is on the GPU; nothing was run.")
    raise SystemExit(2)
say("OK: no other compute process on the GPU")

import numpy  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)   # the same cap as the environment variables above

import transformers  # noqa: E402
import peft          # noqa: E402

DEV = "cuda:0"
if not torch.cuda.is_available():
    raise SystemExit("FAIL: torch does not see a CUDA device")


# ---------------------------------------------------------------------------
# 1. load the tokenizer and the model
# ---------------------------------------------------------------------------
head("1  LOADING")

t0 = time.time()
from transformers import AutoProcessor, AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
say("tokenizer       %s, vocab %d, in %.1f s"
    % (type(tok).__name__, len(tok), time.time() - t0))

# The card's own recipe loads AutoProcessor (the checkpoint is multimodal), so try it and
# say whether its rendering of a text-only chat is the same text as the tokenizer's.  If
# the two disagree the prompt would depend on which object built it, which is exactly the
# kind of difference that is invisible until the tuning and the scoring disagree.
try:
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    say("processor       %s, in %.1f s" % (type(proc).__name__, time.time() - t0))
except Exception as exc:   # noqa: BLE001 - whatever it raises is the finding
    proc = None
    say("processor       NOT loaded: %s: %s" % (type(exc).__name__, exc))


def mem(label):
    free, total = torch.cuda.mem_get_info()
    say("%-15s allocated %.2f GB, reserved %.2f GB, card in use %.2f GB of %.2f GB"
        % (label, torch.cuda.memory_allocated() / 2**30,
           torch.cuda.memory_reserved() / 2**30,
           (total - free) / 2**30, total / 2**30))


def chat_text(body, prefix=PREFIX):
    """prefill.chat_text with this script's tokenizer bound."""
    return P.chat_text(tok, body, prefix)


def ids_of(text):
    """prefill.ids_of with this script's tokenizer bound."""
    return P.ids_of(tok, text)


# --- the text-only load, tried rather than assumed ---------------------------
# The scout report could not establish whether Gemma4ForCausalLM loads this composite
# checkpoint without the vision and audio towers (no _checkpoint_conversion_mapping in
# modeling_gemma4.py).  The failure mode to fear is NOT an exception: from_pretrained
# happily returns a model with randomly initialised weights for keys it did not find, so
# the test is `missing_keys` plus a functional comparison of the candidate logits further
# down -- a load that silently randomises the text tower would print plausible numbers.
TEXT_ONLY_LOGITS = None
text_only_note = ""
try:
    from transformers import Gemma4ForCausalLM
    t0 = time.time()
    tmodel, info = Gemma4ForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, output_loading_info=True)
    took = time.time() - t0
    missing = list(info.get("missing_keys") or [])
    unexpected = list(info.get("unexpected_keys") or [])
    say("text-only load  %s in %.1f s, missing_keys %d, unexpected_keys %d"
        % (type(tmodel).__name__, took, len(missing), len(unexpected)))
    if missing:
        # 2026-11-24: sorted, because the loader hands these back in an order that is not
        # stable between two runs of the same script on the same machine.  This file is
        # archived and compared line by line, so an unordered sample would make the
        # archive differ from a rerun for a reason that says nothing.
        say("                first missing keys: %s" % sorted(missing)[:5])
    if unexpected:
        say("                first unexpected keys: %s" % sorted(unexpected)[:5])
    if missing:
        text_only_note = ("loads but %d weights were NOT in the checkpoint, so the text "
                          "tower would be partly random: not usable" % len(missing))
    else:
        tmodel = tmodel.to(DEV).eval()
        probe = ids_of(chat_text("probe"))
        with torch.inference_mode():
            out = tmodel(input_ids=torch.tensor([probe], device=DEV))
        TEXT_ONLY_LOGITS = out.logits[0, -1].float().cpu()
        text_only_note = "loads with no missing key; logits compared against the full model below"
    del tmodel
    torch.cuda.empty_cache()
except Exception as exc:   # noqa: BLE001 - the exception IS the answer
    text_only_note = "%s: %s" % (type(exc).__name__, exc)
    say("text-only load  FAILED -- %s" % text_only_note)

# --- the model actually used -------------------------------------------------
torch.cuda.reset_peak_memory_stats()
from transformers import Gemma4ForConditionalGeneration  # noqa: E402
t0 = time.time()
model = Gemma4ForConditionalGeneration.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
model = model.to(DEV).eval()
say("model           %s in %.1f s, dtype %s, device %s"
    % (type(model).__name__, time.time() - t0, next(model.parameters()).dtype,
       next(model.parameters()).device))
mem("after loading")
say("text-only:      %s" % text_only_note)

FORWARD_PARAMS = inspect.signature(model.forward).parameters
HAS_LTK = "logits_to_keep" in FORWARD_PARAMS
say("forward takes logits_to_keep: %s" % HAS_LTK)
if not HAS_LTK:
    say("                falling back to the last position of the full logits")


def forward_last(input_ids, attention_mask=None, position_ids=None):
    """Logits of the LAST position only, one forward pass, no cache.

    use_cache=False because nothing here generates a second token; a cache would allocate
    the whole KV block and measure something other than one pass.  logits_to_keep=1 asks
    the model to project only the last position, which at a 262 144-wide vocabulary is the
    difference between a 400x262144 logit tensor and a 1x262144 one -- and at batch 64
    that is the difference between fitting and not.
    """
    kwargs = {"input_ids": input_ids, "use_cache": False}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if HAS_LTK:
        kwargs["logits_to_keep"] = 1
    with torch.inference_mode():
        return model(**kwargs).logits[:, -1, :]


# ---------------------------------------------------------------------------
# the label rows
# ---------------------------------------------------------------------------
head("2  THE PREFILL")

with gzip.open(LABELS, "rb") as fh:
    rows = [json.loads(line) for line in fh]
seeds = [r["seed"] for r in rows]
say("label rows      %d from %s" % (len(rows), os.path.basename(LABELS)))
say("seeds           min %d max %d" % (min(seeds), max(seeds)))
if len(rows) != EXPECT_ROWS:
    raise SystemExit("FAIL rows: %d != %d" % (len(rows), EXPECT_ROWS))
if (min(seeds), max(seeds)) != EXPECT_SEEDS:
    raise SystemExit("FAIL seeds: (%d, %d) != %s"
                     % (min(seeds), max(seeds), EXPECT_SEEDS))


def rendered(row, legend=True):
    """prefill.rendered: the one road from a label row to the text a model reads.

    The first call imports evidence/22's observe.py (and port.py with it), which is why it
    happens here and not at the top of the file.
    """
    return P.rendered(row, legend=legend)


# Five rows that carry a label and lie in the TUNING range (seeds 1-80); seeds above 100
# are never touched by this script at all, and 81-100 are the selection half.  Spread
# across the file rather than the first five, which would all be one seed and one node.
picked = [r for r in rows if r["label"] is not None and r["seed"] <= 80]
step = max(1, len(picked) // FORWARD_ROWS)
picked = picked[::step][:FORWARD_ROWS]
say("rows for step 4 %s"
    % ", ".join("seed %d k %d %s->%s label %s" % (r["seed"], r["k"], r["node"],
                                                  r["neighbor"], r["label"])
                for r in picked))

PROMPT0 = chat_text(rendered(picked[0]))
IDS0 = ids_of(PROMPT0)

say()
say("--- the exact final prompt, special tokens visible (row 1 of the five) ---")
say(PROMPT0)
say("--- end of prompt (nothing after the last character above) ---")
say()

# 2026-11-24: the bos count is checked FIRST, before the two rendering paths are
# compared, because it is a property of IDS0 alone -- checked after the comparison, a
# prompt carrying a second bos would trip the comparison instead and the report would
# name the wrong fault.
bos = tok.bos_token_id
say("bos id %s appears %d time(s), at position(s) %s"
    % (bos, IDS0.count(bos), [i for i, t in enumerate(IDS0) if t == bos][:4]))
if IDS0.count(bos) != 1 or IDS0[0] != bos:
    raise SystemExit("FAIL: expected exactly one bos, at position 0")

# The two ways of building the same prompt must agree.  tokenize=True renders and
# tokenizes inside the template helper; tokenize=False + tok() renders first and
# tokenizes the text.  If they differ, the prompt depends on which call built it.
enc = tok.apply_chat_template(
    [{"role": "user", "content": INSTRUCTION + "\n\n" + rendered(picked[0])},
     {"role": "assistant", "content": PREFIX}],
    tokenize=True, return_dict=True, continue_final_message=True,
    enable_thinking=False)
direct = list(enc["input_ids"])
if direct and isinstance(direct[0], list):
    direct = direct[0]
if direct != IDS0:
    raise SystemExit("FAIL: tokenize=True gives %d ids, tokenize=False+tok() gives %d, "
                     "and they differ" % (len(direct), len(IDS0)))
say("tokenize=True and tokenize=False+tok() agree: %d ids" % len(IDS0))

# Does the prefix survive the template's `trim`?  The rendered text must END with the
# prefix: anything after it would sit between the prompt and the answer position.
if not PROMPT0.endswith(PREFIX):
    raise SystemExit("FAIL: the rendered prompt does not end with the prefill %r; it ends "
                     "%r" % (PREFIX, PROMPT0[-40:]))
say("the prompt ends with the prefill %r, unaltered" % PREFIX)

say()
say("the last 12 tokens of the prompt (id, string):")
for i, tid in enumerate(IDS0[-12:], start=len(IDS0) - 12):
    say("  %4d  %7d  %r" % (i, tid, tok.decode([tid])))

if proc is not None:
    try:
        ptext = proc.apply_chat_template(
            [{"role": "user", "content": INSTRUCTION + "\n\n" + rendered(picked[0])},
             {"role": "assistant", "content": PREFIX}],
            tokenize=False, continue_final_message=True, enable_thinking=False)
        say("processor renders the same text as the tokenizer: %s" % (ptext == PROMPT0))
    except Exception as exc:   # noqa: BLE001
        say("processor cannot render a string-content chat: %s: %s"
            % (type(exc).__name__, exc))


# ---------------------------------------------------------------------------
# 3. the candidates, on the real prompt
# ---------------------------------------------------------------------------
head("3  THE CANDIDATE TOKENS")


def one_token_id(prompt, word):
    """The id `word` adds to `prompt`, provided it adds EXACTLY one token there.

    Checked at the real position and not in isolation: `encode(" short")` having length 1
    says nothing about this prompt, because a BPE merge can join the last prompt token to
    the first answer token.  The property needed is that the prompt's own ids are a prefix
    of the longer sequence AND that the longer one is one token longer.
    """
    base = ids_of(prompt)
    longer = ids_of(prompt + word)
    if longer[:len(base)] != base:
        raise ValueError("%r changes the prompt's own tokens: %d ids shared of %d"
                         % (word, sum(1 for a, b in zip(base, longer) if a == b),
                            len(base)))
    if len(longer) != len(base) + 1:
        pieces = [tok.decode([t]) for t in longer[len(base):]]
        raise ValueError("%r adds %d tokens, not 1: %r"
                         % (word, len(longer) - len(base), pieces))
    return longer[-1]


ids = {}
for word in CANDIDATES:
    tid = one_token_id(PROMPT0, word)
    ids[word] = tid
    expected = EXPECTED_IDS[word]
    say("%-8r one token, id %6d  %s  (%.3f s)"
        % (word, tid,
           "as the vocabulary lookup said" if tid == expected
           else "CORRECTION: the lookup said %d" % expected,
           SECONDS[word]))
# 2026-11-24: counted against CANDIDATES and not against `ids`.  `ids` is keyed by the
# word, so a repeated candidate word collapses into one entry and `len(set(values)) !=
# len(ids)` was then trivially false -- the check passed on CANDIDATES = (" short",
# " short", " long"), which is exactly the input it exists to reject.
if len(set(ids.values())) != len(CANDIDATES):
    raise SystemExit("FAIL: %d candidates give only %d distinct ids: %s"
                     % (len(CANDIDATES), len(set(ids.values())), ids))
say("the %d ids are distinct" % len(CANDIDATES))

CAND_IDS = torch.tensor([ids[w] for w in CANDIDATES], device=DEV)

# The same check on a word that cannot be one token.  This is the check being shown
# failing, on the real tokenizer and the real prompt: the interval strings are several
# tokens each, which is the whole reason the answer is a word and not a number.
say()
for word in (" 10.043", " 6.0"):
    try:
        one_token_id(PROMPT0, word)
    except ValueError as exc:
        say("the one-token check fires: %s" % exc)
    else:
        raise SystemExit("FAIL: %r passed the one-token check, which the vocabulary says "
                         "is impossible" % word)


# ---------------------------------------------------------------------------
# 4. one forward pass per row, twice each
# ---------------------------------------------------------------------------
head("4  ONE FORWARD PASS, FIVE ROWS (the model is UNTUNED)")
say("Nothing is expected of the answers.  What is being shown is that the read-out works")
say("and repeats exactly.")
say()

single = {}
for n, row in enumerate(picked, 1):
    prompt = chat_text(rendered(row))
    ii = torch.tensor([ids_of(prompt)], device=DEV)
    first = forward_last(ii)
    second = forward_last(ii)
    # Bit equality, not a tolerance: the same input through the same kernels in the same
    # shape must give the same bits.  A tolerance here would hide a nondeterministic
    # kernel, which is the one thing this check exists to catch.
    if not torch.equal(first, second):
        d = (first.float() - second.float()).abs().max().item()
        raise SystemExit("FAIL: two identical passes differ, max %.6g" % d)
    logit = first[0, CAND_IDS].float()
    prob = torch.softmax(logit, dim=-1)
    single[n] = first[0].float().cpu()
    say("row %d  seed %d k %d %s->%s  label %s  tokens %d  (two passes identical)"
        % (n, row["seed"], row["k"], row["node"], row["neighbor"], row["label"],
           ii.shape[1]))
    say("        %s" % "  ".join(
        "%s logit %+8.4f p %.4f" % (w.strip(), logit[i].item(), prob[i].item())
        for i, w in enumerate(CANDIDATES)))
    top = torch.topk(first[0].float(), 10)
    say("        top 10 next tokens: %s" % ", ".join(
        "%r %.4f" % (tok.decode([int(t)]), v.item())
        for v, t in zip(top.values, top.indices)))

# Two different rows must NOT give the same logits.  Without this, a read-out that
# returned a constant (a wrong position, a detached prompt) would pass every check above.
if torch.equal(single[1], single[2]):
    raise SystemExit("FAIL: two different rows give identical logits, so the read-out is "
                     "not reading the prompt")
say()
say("rows 1 and 2 give different logits (max difference %.4f), so the read-out does "
    "depend on the prompt" % (single[1] - single[2]).abs().max().item())

if TEXT_ONLY_LOGITS is not None:
    probe = torch.tensor([ids_of(chat_text("probe"))], device=DEV)
    full = forward_last(probe)[0].float().cpu()
    d = (full - TEXT_ONLY_LOGITS).abs().max().item()
    say("text-only vs full model on the same prompt: max logit difference %.6g" % d)


# ---------------------------------------------------------------------------
# 5. token lengths over every label row
# ---------------------------------------------------------------------------
head("5  TOKEN LENGTHS OVER ALL %d LABEL ROWS" % len(rows))

say("instruction alone: %d tokens" % len(ids_of(INSTRUCTION)))

CHUNK = 500
with_legend, without = [], []
t0 = time.time()
for start in range(0, len(rows), CHUNK):
    block = rows[start:start + CHUNK]
    a = [chat_text(rendered(r, legend=True)) for r in block]
    b = [chat_text(rendered(r, legend=False)) for r in block]
    enc_a = tok(a, add_special_tokens=False)["input_ids"]
    enc_b = tok(b, add_special_tokens=False)["input_ids"]
    with_legend += [len(x) for x in enc_a]
    without += [len(x) for x in enc_b]
say("rendered and tokenized %d prompts twice in %.1f s" % (len(rows), time.time() - t0))


def spread(name, values):
    values = sorted(values)
    p99 = values[min(len(values) - 1, int(0.99 * len(values)))]
    say("%-22s min %4d  median %4d  p99 %4d  max %4d"
        % (name, values[0], int(statistics.median(values)), p99, values[-1]))


spread("prompt, with legend", with_legend)
spread("prompt, no legend", without)
say("the legend costs %d tokens at the median"
    % (int(statistics.median(with_legend)) - int(statistics.median(without))))


# ---------------------------------------------------------------------------
# 6. timing
# ---------------------------------------------------------------------------
head("6  TIMING")

median_len = int(statistics.median(with_legend))
pick = min(range(len(rows)), key=lambda i: abs(with_legend[i] - median_len))
MED_ROW = rows[pick]
MED_PROMPT = chat_text(rendered(MED_ROW))
MED_IDS = ids_of(MED_PROMPT)
say("timing prompt: label row %d, seed %d, %d tokens (file median %d)"
    % (pick, MED_ROW["seed"], len(MED_IDS), median_len))

ii = torch.tensor([MED_IDS], device=DEV)
for _ in range(WARMUP_PASSES):
    forward_last(ii)
torch.cuda.synchronize()

times = []
for _ in range(TIMED_PASSES):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    forward_last(ii)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000.0)
times.sort()


def ms(name, values):
    p = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    say("%-22s median %7.2f  p90 %7.2f  p99 %7.2f  max %7.2f  ms"
        % (name, statistics.median(values), p(0.90), p(0.99), values[-1]))


ms("batch 1, %d passes" % TIMED_PASSES, times)
say("beside it, the number the brief names: 100 ms.  No verdict is drawn here.")

BASE1 = forward_last(ii)[0, CAND_IDS].float().cpu()

# --- batched ---------------------------------------------------------------
say()
say("--- batched passes ---")
# LEFT padding, so that the last column of every row is that row's last REAL token and
# one logits_to_keep=1 read gives every row's answer position.  With right padding the
# answer position differs per row, which would force the full 262 144-wide logits for
# every position (13 GB at batch 64) and a gather.
say(P.prepare_padding(tok))


def batch_of(texts):
    """prefill.batch_of: the left-padded batch, with its two read-position checks."""
    return P.batch_of(tok, texts, DEV)


# 2026-11-24: the first run took the FIRST 64 label rows and batch 8 then reported 0.0 %
# padding -- the early rows of a run are all the minimal observation (no origin, no peer,
# no exchange) and come out the same length, so the padding check had nothing to check.
# The rows are now spread over the whole file and sorted by length, which puts the
# SHORTEST row at index 0: that is the row with the most padding, and row 0 is the one
# compared against its own batch-1 pass below.
_spread_rows = rows[::max(1, len(rows) // max(BATCH_SIZES))][:max(BATCH_SIZES)]
rows_for_batch = sorted((chat_text(rendered(r)) for r in _spread_rows),
                        key=lambda t: len(ids_of(t)))
say("batch rows: %d prompts spread over the file, %d to %d tokens, shortest first"
    % (len(rows_for_batch), len(ids_of(rows_for_batch[0])),
       len(ids_of(rows_for_batch[-1]))))
worst = {}
for bs in BATCH_SIZES:
    # (a) a MIXED batch of real prompts of different lengths -- the padding case.
    # 2026-11-24: taken SPREAD over the sorted list and not as its first `bs` entries.
    # The first run took the first `bs`, which handed batch 8 the eight shortest prompts
    # and batch 64 the whole range, so the three throughput numbers were measured on
    # three different length distributions and were not comparable -- batch 8 came out
    # fastest per pass for that reason alone.  The stride keeps index 0 the shortest row,
    # which is what the padding comparison below needs.
    texts = rows_for_batch[::len(rows_for_batch) // bs][:bs]
    iid, attn, pos, lengths = batch_of(texts)
    pad_share = 1.0 - float(lengths.sum()) / float(iid.numel())
    for _ in range(3):
        forward_last(iid, attn, pos)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(BATCH_REPEATS):
        last = forward_last(iid, attn, pos)
    torch.cuda.synchronize()
    per_batch = (time.perf_counter() - t0) / BATCH_REPEATS
    rate = bs / per_batch

    # Row 0 of this batch is label row 0, not the median prompt, so compare it against a
    # batch-1 pass of its own text.
    own = forward_last(torch.tensor([ids_of(texts[0])], device=DEV))[0, CAND_IDS].float()
    mixed_d = (last[0, CAND_IDS].float() - own).abs().max().item()
    mixed_p = (torch.softmax(last[0, CAND_IDS].float(), -1)
               - torch.softmax(own, -1)).abs().max().item()

    # (b) the CONTROL: the same prompt bs times, so there is no padding at all. Its
    # difference from batch 1 is pure kernel-and-shape rounding, which is the yardstick
    # the padded number has to be read against.  Without it a bf16 rounding difference
    # and a real padding bug look the same.
    ctl_iid, ctl_attn, ctl_pos, _ = batch_of([MED_PROMPT] * bs)
    ctl = forward_last(ctl_iid, ctl_attn, ctl_pos)
    # 2026-11-24: ROW 0 of the control, not every row.  The padded number above is a max
    # over one row's three candidate logits; taking the yardstick over all `bs` rows made
    # it a max over up to 3*bs samples, so the threshold was systematically the larger of
    # the two by construction and the comparison leaned towards passing.  Same row count
    # on both sides, so the two numbers are of the same quantity.
    ctl_d = (ctl[0, CAND_IDS].float() - BASE1.to(DEV)).abs().max().item()
    ctl_p = (torch.softmax(ctl[0, CAND_IDS].float(), -1)
             - torch.softmax(BASE1.to(DEV), -1)).abs().max().item()
    worst[bs] = (mixed_d, ctl_d, mixed_p, ctl_p)

    say("batch %2d  %6.1f ms/batch  %7.1f passes/s  padded width %d, mean real length "
        "%.0f, padding %4.1f%% of the tokens"
        % (bs, per_batch * 1000, rate, iid.shape[1],
           float(lengths.float().mean()), pad_share * 100))
    say("          candidate logits vs batch 1:  mixed (padded) %.4f, softmax %.4f"
        "   |   no-padding control %.4f, softmax %.4f"
        % (mixed_d, mixed_p, ctl_d, ctl_p))
    for n in SWEEP_SIZES:
        say("          %7d passes at this rate: %.1f min" % (n, n / rate / 60))

say()
# THE TOLERANCE, and what it is a tolerance ON.  2026-11-24: an absolute tolerance
# against the batch-1 logits is the wrong check, and the first run said so -- the
# NO-PADDING control, where every row is the same prompt and nothing is padded, already
# differs from batch 1 by up to ~1.0, which is eight bf16 steps.  That is batch shape:
# these logits are bf16 and soft-capped at +-30 (one bf16 step is 0.125 there), and a
# different batch width picks a different GEMM kernel and a different reduction order.
# So the control is the EMPIRICAL yardstick and the tolerance is on the EXCESS: padding
# must not add more than one quarter of a logit on top of what batch shape alone costs.
# Without the control this check would have to accept ~1.0 outright and would then not
# notice a real position_ids bug of that size.
EXCESS_TOL = 0.25
bad = {bs: v for bs, v in worst.items() if v[0] > v[1] + EXCESS_TOL}
say("tolerance: the padded batch may exceed the no-padding control by at most %.2f of a"
    % EXCESS_TOL)
say("logit.  The control is what batch shape alone costs, measured and not assumed.")
for bs, (m, c, mp, cp) in sorted(worst.items()):
    say("  batch %2d  padded %.4f, control %.4f, excess %+.4f   (softmax: padded %.4f, "
        "control %.4f)" % (bs, m, c, m - c, mp, cp))
if bad:
    raise SystemExit("FAIL: padding costs more than rounding at batch size(s) %s: %s"
                     % (sorted(bad), {k: round(v[0] - v[1], 4) for k, v in bad.items()}))
say("OK: at every batch size the padded difference is at or below the control's, so the")
say("read position is each row's last real token and the rest is bf16 rounding.")


# ---------------------------------------------------------------------------
# 7. versions
# ---------------------------------------------------------------------------
head("7  VERSIONS AND PEAK MEMORY")
say("torch           %s (cuda %s)" % (torch.__version__, torch.version.cuda))
say("transformers    %s" % transformers.__version__)
say("peft            %s" % peft.__version__)
say("numpy           %s" % numpy.__version__)
say("python          %s" % sys.version.split()[0])
say("gpu             %s" % torch.cuda.get_device_name(0))
say("peak allocated  %.2f GB" % (torch.cuda.max_memory_allocated() / 2**30))
say("peak reserved   %.2f GB" % (torch.cuda.max_memory_reserved() / 2**30))
mem("at the end")
say()
say("OK: the read-out works on this card and repeats exactly.")
