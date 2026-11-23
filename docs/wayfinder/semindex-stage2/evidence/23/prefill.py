#!/usr/bin/env python3
"""The one road from a label row to the tokens Gemma 4 reads, shared by every script here.

Imported by `smoke.py` and by `precheck.py` on the tuning host.  2026-11-24: it exists
because those two scripts must build the SAME prompt.  A second copy of the prefill would
let the smoke report and the measurements that the tuning plan rests on drift apart
without either script noticing -- the wording, the prefill, the candidate words and the
left-padding convention are the things both of them are about.

Importing this module has side effects ON PURPOSE, and they must happen before torch,
numpy or transformers are imported anywhere:
  - the thread caps (a BLAS library opens one thread per core on this 96-core shared box);
  - HF_HUB_OFFLINE and an HF_HOME under ROOT, so nothing reaches the hub and the account's
    own cache is never read;
  - ROOT, and the two sys.path entries the replay code asserts.
So this is the FIRST import of any script that uses it, right after `os` and `sys`.

ROOT comes from argv[1] when that is a directory, else $SMOKE_ROOT, else $ROOT.
"""
import os
import sys

# ---------------------------------------------------------------------------
# Environment, BEFORE torch / numpy / transformers are imported
# ---------------------------------------------------------------------------
# 2026-11-24: a 0.7 s replay ran at 1273 % cpu on this 96-core box -- a BLAS library
# opening one thread per core at import.  The host is shared with ten other accounts and
# has no scheduler, so the brief caps us at 8 processes; a library that fans out to 96
# threads breaks that cap without anyone asking for it.  Set before the import, because
# both OpenMP and OpenBLAS read these once, at load time.
for _name, _value in (("OMP_NUM_THREADS", "8"), ("OPENBLAS_NUM_THREADS", "1"),
                      ("MKL_NUM_THREADS", "1"), ("RAYON_NUM_THREADS", "8"),
                      # The Rust tokenizer forks a thread pool per batch call and warns
                      # about it; one tokenizer call is never the bottleneck here.
                      ("TOKENIZERS_PARALLELISM", "false")):
    os.environ.setdefault(_name, _value)

_argv1 = sys.argv[1] if len(sys.argv) > 1 else ""
ROOT = (_argv1 if os.path.isdir(_argv1)
        else os.environ.get("SMOKE_ROOT") or os.environ.get("ROOT") or "")
if not ROOT or not os.path.isdir(ROOT):
    raise SystemExit("usage: <script> <ROOT> [...]   (or set SMOKE_ROOT)")
ROOT = os.path.abspath(ROOT)

# Anonymous and offline, and the account's own cache is never touched: HF_HOME points
# into ROOT.  HF_TOKEN is removed rather than emptied, because an empty string is still a
# value the hub client would try to send.
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HOME"] = os.path.join(ROOT, "hf-home")
os.environ["HF_HUB_OFFLINE"] = "1"
# The replay's own store: observe.py imports port.py at module scope, and port.py resolves
# the store from this variable.
os.environ.setdefault("STORE", os.path.join(ROOT, "store"))

REPO = os.path.join(ROOT, "repo")
EV22 = os.path.join(REPO, "docs/wayfinder/semindex-stage2/evidence/22")
MODEL_DIR = os.path.join(ROOT, "models/gemma-4-E2B-it")
LABELS = os.path.join(ROOT, "ev22/labels_p0.2.jsonl.gz")

# src first, then evidence/22, the same two entries in the same order the parity run put
# on the path, so that `semindex` is this worktree's and `observe`/`port` are evidence/22's.
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(1, EV22)

# ---------------------------------------------------------------------------
# The prompt wording
# ---------------------------------------------------------------------------
# FROZEN 2026-11-24 by `spec.md`'s "The input text", which is the file that stopped this
# wording being provisional: every token count and every timing in `smoke.txt` and
# `precheck.txt` rests on this exact instruction, so a rewording makes all of them stale.
# One sentence of task plus the three words and their seconds, because the words are the
# answer and the numbers only ever appear in the prompt: the tokenizer has no single token
# for "3.695", "6.0" or "10.043", so the interval itself can never be the answer token.
# `smoke.py` and `precheck.py` ran BEFORE the freeze and their own text still calls the
# wording provisional; that is left alone, because their archived output is the run's and
# editing it would unpick the record.  A later step may still reword, and then every
# number built on this wording is re-measured rather than carried over.
INSTRUCTION = (
    "A node decides how long to wait before polling this peer for state again.\n"
    "Answer with one word: short (3.695 s), mid (6.0 s) or long (10.043 s)."
)
# FROZEN 2026-11-24 by `spec.md`, with the instruction above.  It must end in a NON-space
# character: the Gemma 4 template
# runs the assistant content through `| trim`, so a prefix ending in a space loses it
# silently, and the next token is then the space-prefixed word anyway.  Ending in ":"
# makes that explicit instead of accidental.
PREFIX = "Interval:"
CANDIDATES = (" short", " mid", " long")
# The seconds each word stands for, printed with the logits so a report is readable on its
# own.  Never tokenized.
SECONDS = {" short": 3.695, " mid": 6.0, " long": 10.043}
# The vocabulary lookup of the scout report, to be confirmed or corrected live.
EXPECTED_IDS = {" short": 2822, " mid": 5453, " long": 1440}

EXPECT_ROWS = 16071          # labels_p0.2.jsonl.gz, stated by the brief
EXPECT_SEEDS = (1, 100)      # min and max seed of the main cell's label file

_OBS = None


def obs():
    """evidence/22's `observe` module, imported on first use.

    Lazily, because importing it pulls in `port.py`, which asserts its own location and
    reads the store: a script that only wants the wording (or that has not yet checked
    the card is free) should not pay for that at its first import line.
    """
    global _OBS
    if _OBS is None:
        import observe
        _OBS = observe
    return _OBS


def chat_text(tok, body, prefix=PREFIX):
    """The final prompt TEXT, with the special tokens in it, prefill included.

    `continue_final_message=True` is what makes the assistant turn a prefix instead of a
    finished turn: the template's end-of-turn marker is cut off, so the next token the
    model predicts is the first token of the answer.  `enable_thinking=False` keeps the
    template from injecting a think block, which would put several hundred tokens and a
    different position between the prompt and the answer.
    """
    messages = [{"role": "user", "content": INSTRUCTION + "\n\n" + body},
                {"role": "assistant", "content": prefix}]
    return tok.apply_chat_template(messages, tokenize=False,
                                   continue_final_message=True,
                                   enable_thinking=False)


def ids_of(tok, text):
    """Token ids of an ALREADY rendered prompt.

    add_special_tokens=False because the template emits bos itself; letting the tokenizer
    add another one would put a second bos in the prompt and shift every position.
    """
    return tok(text, add_special_tokens=False)["input_ids"]


def rendered(row, legend=True):
    """The one road from a label row to the text a model reads (evidence/22/observe.py)."""
    OBS = obs()
    record = OBS.observation(row["state"], row["resolved"], row["history"],
                             row["ts_ms"], row["neighbor"])
    return OBS.render(record, legend=legend)


def prepare_padding(tok):
    """LEFT padding, so the last column of every row is that row's last REAL token.

    One `logits_to_keep=1` read then gives every row's answer position.  With right
    padding the answer position differs per row, which would force the full 262 144-wide
    logits for every position (13 GB at batch 64) and a gather.
    """
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return "pad token %r id %s, padding_side %s" % (tok.pad_token, tok.pad_token_id,
                                                    tok.padding_side)


def batch_of(tok, texts, dev):
    """Left-padded batch: input_ids, attention_mask, position_ids, real lengths.

    position_ids are passed explicitly: cumsum-1 over the mask gives every real token the
    position it has in its OWN sequence, and the clamp keeps the pads at 0 (they are
    masked out anyway).  2026-11-24: passing them is NOT what makes left padding correct.
    Running with position_ids=None changes the candidate logits by no more than the
    batch-shape rounding does (measured), because with LEFT padding every real token of a
    row is offset by the same pad count and RoPE is relative, so a constant shift inside a
    row cancels in the attention.  What the explicit ids buy is that the read position is
    stated rather than inferred, and that the two checks below have something to assert
    against.  A non-constant corruption of them DOES move the logits far outside the
    tolerance, which is what keeps that comparison from being a check that cannot fail.
    """
    enc = tok(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()}
    attn = enc["attention_mask"]
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    lengths = attn.sum(-1)
    # These two are what "the read position is each row's last real token" MEANS, so they
    # are asserted and not asserted-in-a-comment.  2026-11-24: SystemExit and not
    # `assert`, which disappears under -O or PYTHONOPTIMIZE and would silently remove the
    # only checks that define the read position while the script still printed its OK lines.
    if not bool(attn[:, -1].all()):
        raise SystemExit("FAIL: left padding did not leave a real token last")
    if not bool((pos[:, -1] == lengths - 1).all()):
        raise SystemExit("FAIL: last position is not length-1")
    return enc["input_ids"], attn, pos, lengths
