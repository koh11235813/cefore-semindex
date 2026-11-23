#!/usr/bin/env python3
"""The two LoRA candidates of L, tuned on the hindsight labels by the frozen recipe.

    cd $ROOT && env PYTHONDONTWRITEBYTECODE=1 STORE=$ROOT/store \
        PYTHONPATH=$ROOT/repo/src HF_HUB_OFFLINE=1 ./.venv/bin/python -B \
        repo/docs/wayfinder/semindex-stage2/evidence/23/tune.py $ROOT \
        > $ROOT/tune.log 2> $ROOT/tune.err

ONE INVOCATION FOR BOTH CANDIDATES, and the reason is the adapter and not the wall clock:
the two runs have to start from the SAME untuned adapter, and in one process that is a
fresh `get_peft_model` under a fixed torch seed per candidate rather than a claim.  A
run of this script that tuned only one candidate would also have to re-derive every row
count, every seed set and every assertion below, which is the part a reader checks.

WHAT IT DOES, IN ORDER
  0. refuses to start if another compute process is on the shared card;
  1. re-counts both candidates' row sets ON THIS HOST from the label file's own bytes and
     stops if any count differs from the committed `rows.txt`.  The counts are the whole
     definition of what is being tuned, and `rows.py` runs on another machine;
  2. builds the adapter and asserts the 205 matched modules, the parameter counts, the
     float32 adapter dtype over the bf16 base and the 205 zero `lora_B` tensors;
  3. 3 epochs per candidate at batch 16, lr 2e-4, constant, one permutation per epoch;
  4. after each epoch: saves the adapter, reads out the candidate's own held-out rows
     (seeds 71 to 80) at the read-out the score will be taken at, and prints the held-out
     cross-entropy with the held-out agreement and that slice's own baseline beside it;
  5. keeps the epoch with the lowest held-out cross-entropy, the earlier epoch on a tie,
     deletes the other two, re-loads the kept adapter from disk and requires it to
     reproduce the numbers it was chosen by;
  6. shows every check failing on a mutated input, and says what the run does not show.

WHAT IT DOES NOT DO
  It reads no seed above 80.  Seeds 71 to 80 choose the epoch and seeds 1 to 70 tune;
  the selection half, 81 to 100, is not opened here at all, and the measuring seeds 121
  to 160 are refused by two guards on every path.  No number printed here says which
  candidate is better: that is the replay score's on the selection half, and the held-out
  numbers below are tuning-half numbers.
"""
import os
import sys

# 2026-11-24: FIRST import, before torch / numpy / transformers.  `prefill.py` sets the
# thread caps, the offline HF variables, ROOT and the two sys.path entries at ITS import
# time, and `decider.py` holds the read-out, the tie rule and L itself, so that the
# held-out pass here and the replay score later are taken by the same call.
import prefill as P   # noqa: E402
import decider as D   # noqa: E402

# `../22`'s own paired-t statistic, through the path `decider.py` resolved from its own
# location: candidate b is selected by that statistic, `../22/stability.py` binned the
# agreement it was measured with on the same function, and a second implementation here
# would be a second definition of what candidate b is.
import label as LBL   # noqa: E402

ROOT = P.ROOT

import gzip        # noqa: E402
import hashlib     # noqa: E402
import json        # noqa: E402
import shutil      # noqa: E402
import statistics  # noqa: E402
import time        # noqa: E402

# ---------------------------------------------------------------------------
# the recipe, every setting of it fixed before any tuned model was scored
# ---------------------------------------------------------------------------
SEED = 20261124              # the one tuning seed; no sweep, because a sweep is a selection
EPOCHS = 3
BATCH = 16                   # 0.0279 s per row against 0.0326 at batch 8, 24.95 GB of 47.27
LR = 2e-4                    # torch's own AdamW defaults otherwise, incl. weight decay 1e-2
MAX_PROMPT_TOKENS = 1024     # a bound no row of this cell reaches; nothing is truncated
LORA_R, LORA_ALPHA = 16, 32
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
# Load-bearing: 353 modules of the whole model end in one of the seven names and 148 of
# them are in the vision and audio towers, which this text-only read-out never runs.
LORA_EXCLUDE = r"model\.(vision_tower|audio_tower)\..*"

MAX_TUNE_SEED = D.MAX_TUNE_SEED      # 80
HELD_FROM = 71                       # seeds 71 to 80 choose the epoch and do not tune
BAR = 3.0                            # the paired-t bar of candidate b's filter
BAR_TOL = 1e-6                       # the tolerance the clump at the bar is measured with
ARMS = D.ARMS
# The arms as they are spelled in the count keys below, so that a key and a committed
# number cannot drift apart through a format string ("%g" prints 6.0 as "6").
ARM_LABEL = {ARMS[0]: "3.695", ARMS[1]: "6.0", ARMS[2]: "10.043"}
CELL = "labels_p0.2.jsonl.gz"
LABELS = os.path.join(ROOT, "ev22", CELL)
SHA_FILE = os.path.join(ROOT, "repo/docs/wayfinder/semindex-stage2/evidence/22",
                        "labels.sha256")
ADAPTERS = os.path.join(ROOT, "adapters")

# Candidate b first, and this is a choice about the cost of a mistake rather than about
# the result: b is 4018 rows against a's 10013, so running it first takes every code path
# below -- three epochs, three held-out passes, three saves, the choice, the two deletions,
# the reload and the digest -- for less than half the card time, on a card shared with
# other accounts.  The order cannot touch the numbers: each candidate starts from a fresh
# base, a fresh adapter under the same torch seed, and its own per-epoch permutation.
ORDER = ("b", "a")

# `rows.txt`, which counted these on the other machine from the same label file.  Every
# one of them is re-counted below and a difference stops the run: the row sets are the
# definition of what is being tuned, and a tuning that quietly tuned on another set of
# rows would print a plausible loss.
EXPECT = {
    "file rows": 16071, "file seeds": 100, "tuning-half rows": 12868,
    "tuning-half ties": 1430,
    "a rows": 11438, "a 3.695": 2581, "a 6.0": 2972, "a 10.043": 5885,
    "b rows": 4586, "b 3.695": 714, "b 6.0": 719, "b 10.043": 3153,
    "b rows, bare float at the bar": 4571,
    "a tune rows": 10013, "a tune 3.695": 2240, "a tune 6.0": 2615, "a tune 10.043": 5158,
    "a held rows": 1425, "a held 3.695": 341, "a held 6.0": 357, "a held 10.043": 727,
    "b tune rows": 4018, "b tune 3.695": 625, "b tune 6.0": 628, "b tune 10.043": 2765,
    "b held rows": 568, "b held 3.695": 89, "b held 6.0": 91, "b held 10.043": 388,
}

OUT = []


def say(text=""):
    """Print and keep, so the whole report is one stream the caller redirects."""
    print(text, flush=True)
    OUT.append(text)


def head(title):
    """A part heading, in this directory's own shape."""
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


Fail = D.Fail


# ---------------------------------------------------------------------------
# this script's own checks, each one a function of plain values
# ---------------------------------------------------------------------------
def check_digest(name, want, got, where):
    """The label file is the one the committed digest names.

    Every count below is only as good as the bytes it came from, the label file lives
    outside the repository, and this host holds a copy that was pushed to it.
    """
    if got != want:
        raise Fail("%s: digest of %s is %s, committed %s" % (where, name, got, want))
    return got


def check_counts(got, want, where):
    """Every re-counted number equals the committed one.

    One function over a dict rather than a line of `assert` per number: the failure has to
    name which count moved, because the interesting failure is one count of twenty-eight
    differing by one.
    """
    bad = sorted(k for k in want if got.get(k) != want[k])
    if bad:
        raise Fail("%s: %d count(s) differ from rows.txt: %s" % (
            where, len(bad), ", ".join("%s %r against %r" % (k, got.get(k), want[k])
                                       for k in bad[:4])))
    return len(want)


def check_roles(roles, where):
    """The three seed roles are inside their ranges and meet the measuring seeds nowhere.

    The roles are tuning 1 to 70, held out 71 to 80 and selection 81 to 100.  This is the
    check that the held-out slice was carved out of the TUNING half: a held-out slice taken
    from 81 to 100 would spend the selection half on the epoch choice and nothing else
    would notice.

    The measuring seeds are tested BEFORE the ranges, although the range test would catch
    every one of them: tested after it, the measuring branch could never fire on any input
    the range test admits, and a branch that cannot fail is not evidence.  In this order
    both branches have a mutation in part 5.
    """
    want = {"tuning": (1, 70), "held out": (71, 80), "selection": (81, 100)}
    union = sorted({s for seeds in roles.values() for s in seeds})
    met = sorted(set(union) & set(range(121, 161)))
    if met:
        raise Fail("%s: the seed set meets the measuring seeds 121-160: %s" % (where, met))
    for role, seeds in sorted(roles.items()):
        lo, hi = want[role]
        bad = sorted(s for s in seeds if not lo <= s <= hi)
        if bad:
            raise Fail("%s: the %s role carries %s, outside %d-%d"
                       % (where, role, bad[:4], lo, hi))
    return union


def check_targets(names, model_named, n_layers, n_shared, where):
    """The LoRA target set is exactly the language model's own projection modules.

    Three branches, each rejecting a different way of getting this wrong: a target set that
    reached into the vision or audio tower; a model whose projection count is not the one
    this architecture implies (7 per layer is WRONG for E2B, because the last
    `num_kv_shared_layers` layers reuse an earlier layer's k/v and this transformers build
    gives them no `k_proj` and no `v_proj` module at all, so those layers have 5); and a
    target set that is simply not that set, which would still give a model that trains and
    a loss that falls, just not this recipe.
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

    Used on `lora_B` before the first step: peft starts it at exactly zero, so this is
    what "this candidate starts untuned" means.  Without it a candidate that silently
    continued the previous candidate's adapter would print a loss that starts low and
    reads like an effect of the row set.
    """
    nz = sorted(n for n, v in params.items() if float(v.abs().max()) != 0.0)
    if nz:
        raise Fail("%s: %d of %d tensors are not zero, e.g. %s"
                   % (where, len(nz), len(params), nz[:2]))
    return len(params)


def check_changed(before, after, where):
    """The adapter weights moved.

    An optimiser that silently updated nothing -- a frozen adapter, a detached loss, a zero
    learning rate -- would leave every other number here looking right.
    """
    moved = {k: (after[k] - before[k]).abs().max().item() for k in before}
    still = sorted(k for k, v in moved.items() if v == 0.0)
    if still:
        raise Fail("%s: %d of %d adapter tensors did not move, e.g. %s"
                   % (where, len(still), len(before), still[:3]))
    return moved


def check_finite(values, where):
    """No loss is NaN or inf."""
    bad = [(i, v) for i, v in enumerate(values) if not (v == v and abs(v) != float("inf"))]
    if bad:
        raise Fail("%s: %d non-finite value(s), first at step %d: %r"
                   % (where, len(bad), bad[0][0], bad[0][1]))
    return True


def check_reload(before, after, where):
    """The adapter read back from disk reproduces the numbers it was chosen by.

    2026-11-24, added because the epoch is chosen from numbers taken while the adapter was
    in memory and every later script of this step reads the adapter from DISK.  If
    `save_pretrained` wrote another epoch's weights, or wrote them in another dtype, the
    cross-entropy that chose the epoch would describe a file nobody will ever load, and the
    cross-check the agreement step is meant to make against this run would compare two
    different models.
    """
    bad = sorted(k for k in before if before[k] != after.get(k))
    if bad:
        raise Fail("%s: %s" % (where, ", ".join(
            "%s %r in memory against %r from disk" % (k, before[k], after.get(k))
            for k in bad)))
    return True


# ---------------------------------------------------------------------------
# reading and classifying the rows
# ---------------------------------------------------------------------------
def committed_digest(name):
    """The sha256 the committed digest file carries for one label file.

    That file names absolute paths of the machine it was written on, which do not exist
    here, so only the basename column is matched; the hash compared is the committed one.
    """
    with open(SHA_FILE, encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and os.path.basename(line.split()[1]) == name:
                return line.split()[0]
    raise Fail("%s names no %s" % (SHA_FILE, name))


def read_rows(path):
    """Every label row of the file, and the sha256 of the bytes they came out of."""
    with open(path, "rb") as fh:
        raw = fh.read()
    got = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(line) for line in gzip.decompress(raw).splitlines() if line.strip()]
    return rows, got


def t_of(row):
    """`(t, sd)` of a row, by `../22/label.py`'s `paired_t` over the row's own passes.

    The statistic is imported and not re-implemented, because the agreement candidate b's
    filter was measured with was binned on that function.
    """
    t, sd, _n = LBL.paired_t(row["pass_scores"], row["passes"])
    return t, sd


def keeps_b(row, tol=BAR_TOL):
    """Candidate b's own rule on one row: `t >= 3 or (sd = 0 and labelled)`.

    `rows.py` holds the same rule and is the file whose count is of record.  It cannot be
    imported here: it reads the committed digest file for an ABSOLUTE path of the other
    machine, and importing it would run that read.  So the rule is restated and the
    restatement is checked the only way it can be -- by reproducing every one of
    `rows.txt`'s counts on this host before anything is tuned.

    The bar is `t >= 3 - 1e-6` and not a bare `>=`: the rows at t exactly 3 are a mass
    point that doubles put within 1e-13 of 3 on either side, the user's decision is that
    all of them are kept, and the bare comparison keeps 15 fewer of them.
    """
    if row["label"] is None:
        return False
    t, sd = t_of(row)
    if sd == 0.0:
        return True
    return t is not None and t >= BAR - tol


def arm_counts(rows):
    """{arm: how many of these rows carry that label}."""
    counts = {arm: 0 for arm in ARMS}
    for r in rows:
        counts[r["label"]] += 1
    return counts


def baseline(counts, n):
    """(the most frequent arm, its share) over a row set's per-arm counts.

    Every agreement below is read against this and never against 1/3: always answering
    10.043 s already agrees with 0.5102 of candidate a's held-out rows.
    """
    top = max(ARMS, key=lambda a: (counts[a], -ARMS.index(a)))
    return top, counts[top] / n


# ===========================================================================
# 0  THE CARD, THE HOST AND WHAT THIS RUN IS
# ===========================================================================
T_START = time.time()
head("0  THE CARD IS FREE, AND WHAT THIS RUN IS")
say("Two tuned candidates of L are produced here and kept.  No seed above 80 is read: the")
say("choice between the candidates is the replay score's on seeds 81 to 100 and is not")
say("made here, and every held-out number below is a tuning-half number.")
say()
_apps = D.nvidia_compute_apps()
say("nvidia-smi compute apps: %s" % (_apps.strip() if _apps.strip() else "(none)"))
D.check_card_free(_apps, "tune.py at start")
say("OK: no other compute process on the GPU")

import numpy   # noqa: E402
import torch   # noqa: E402

torch.set_num_threads(8)

import peft                        # noqa: E402
import torch.nn.functional as F    # noqa: E402
import transformers                # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("FAIL: torch does not see a CUDA device")

say()
say("started         %s" % time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(T_START)))
say("torch %s (cuda %s), transformers %s, peft %s, numpy %s, python %s"
    % (torch.__version__, torch.version.cuda, transformers.__version__,
       peft.__version__, numpy.__version__, sys.version.split()[0]))
_free, _total = torch.cuda.mem_get_info()
say("gpu %s, %.2f GB free of %.2f GB"
    % (torch.cuda.get_device_name(0), _free / 2**30, _total / 2**30))
say("base model      %s" % P.MODEL_DIR)
say("adapters        %s/{%s}" % (ADAPTERS, ",".join(ORDER)))
say("recipe          LoRA r %d alpha %d dropout 0.0 bias none, no task_type; AdamW lr %g,"
    % (LORA_R, LORA_ALPHA, LR))
say("                torch's other defaults (weight decay 1e-2); constant rate, no warmup;")
say("                batch %d, %d epochs, seed %d; no truncation, prompts capped at %d"
    % (BATCH, EPOCHS, SEED, MAX_PROMPT_TOKENS))
say("read-out        bf16 base, float32 adapter, batch 1, the three candidate logits at")
say("                the last real position, first maximum of %s wins a tie"
    % list(P.CANDIDATES))

# ===========================================================================
# 1  THE ROWS, RE-COUNTED ON THIS HOST
# ===========================================================================
head("1  THE ROW SETS, RE-COUNTED HERE AND CHECKED AGAINST rows.txt")
ROWS, GOT = read_rows(LABELS)
WANT = committed_digest(CELL)
check_digest(CELL, WANT, GOT, "the label file")
say("label file      %s" % LABELS)
say("sha256          %s" % GOT)
say("                == evidence/22/labels.sha256, read from %s" % SHA_FILE)

SEEDS_IN_FILE = sorted({int(r["seed"]) for r in ROWS})
say("rows            %d" % len(ROWS))
say("seed column     min %d max %d, %d distinct"
    % (SEEDS_IN_FILE[0], SEEDS_IN_FILE[-1], len(SEEDS_IN_FILE)))
D.PORT.check_seeds(SEEDS_IN_FILE, "tune.py: the seeds the label file carries")
say("port.check_seeds passed them, so nothing above 100 is in this file")

TUNE_HALF = [r for r in ROWS if int(r["seed"]) <= MAX_TUNE_SEED]
HALF_SEEDS = D.check_seeds_tuning([r["seed"] for r in TUNE_HALF], "the rows this run reads")
say("rows of seeds 1-%d  %d, over %d seeds; the row list is cut there before anything is"
    % (MAX_TUNE_SEED, len(TUNE_HALF), len(HALF_SEEDS)))
say("rendered, so seeds 81-100 (selection) and 121-160 (measuring) appear nowhere below")

A_ROWS = [r for r in TUNE_HALF if r["label"] is not None]
B_ROWS = [r for r in TUNE_HALF if keeps_b(r)]
_b_keys = {(r["seed"], r["k"]) for r in B_ROWS}
_a_keys = {(r["seed"], r["k"]) for r in A_ROWS}
if _b_keys - _a_keys:
    raise Fail("candidate b is not a subset of candidate a: %d row(s) outside"
               % len(_b_keys - _a_keys))
say("candidate b is a subset of candidate a: %d of %d rows" % (len(_b_keys), len(_a_keys)))

SETS = {}
for name, rows in (("a", A_ROWS), ("b", B_ROWS)):
    fit = [r for r in rows if int(r["seed"]) < HELD_FROM]
    held = [r for r in rows if int(r["seed"]) >= HELD_FROM]
    SETS[name] = (fit, held)

# The bare-float count of the same bar, in `rows.py`'s own two parts: every row whose t
# clears 3 without a tolerance, plus every row the sd arm of the rule admits.  It is
# counted because the 15 rows the tolerance keeps are the user's "keep every t of exactly
# 3" decision made visible, and a tolerance that had stopped selecting them would show up
# here and nowhere else.
_strict = sum(1 for r in A_ROWS if t_of(r)[0] is not None and t_of(r)[0] >= BAR)
_strict += sum(1 for r in A_ROWS if t_of(r)[1] == 0.0)
COUNTED = {"file rows": len(ROWS), "file seeds": len(SEEDS_IN_FILE),
           "tuning-half rows": len(TUNE_HALF),
           "tuning-half ties": sum(1 for r in TUNE_HALF if r["label"] is None),
           "b rows, bare float at the bar": _strict}
for name, rows in (("a", A_ROWS), ("b", B_ROWS)):
    counts = arm_counts(rows)
    COUNTED["%s rows" % name] = len(rows)
    for arm in ARMS:
        COUNTED["%s %s" % (name, ARM_LABEL[arm])] = counts[arm]
    fit, held = SETS[name]
    for role, part in (("tune", fit), ("held", held)):
        COUNTED["%s %s rows" % (name, role)] = len(part)
        pc = arm_counts(part)
        for arm in ARMS:
            COUNTED["%s %s %s" % (name, role, ARM_LABEL[arm])] = pc[arm]
check_counts(COUNTED, EXPECT, "the recount on this host")
say()
for key in sorted(EXPECT):
    say("  %-32s %6d  == rows.txt" % (key, COUNTED[key]))
say()
say("all %d counts reproduce rows.txt on this host, so the rows this run tunes on are the"
    % len(EXPECT))
say("rows the frozen plan names")

ROLES = {"tuning": sorted({int(r["seed"]) for r in A_ROWS if int(r["seed"]) < HELD_FROM}),
         "held out": sorted({int(r["seed"]) for r in A_ROWS
                             if int(r["seed"]) >= HELD_FROM}),
         "selection": sorted(range(81, 101))}
UNION = check_roles(ROLES, "the seed roles")
say()
for role in ("tuning", "held out", "selection"):
    seeds = ROLES[role]
    say("  %-10s %d seeds, %d to %d%s"
        % (role, len(seeds), seeds[0], seeds[-1],
           "  (named here so the roles are complete; not read by this run)"
           if role == "selection" else ""))
say("  the union of the three meets range(121, 161) in %d seeds"
    % len(set(UNION) & set(range(121, 161))))
D.check_seeds_tuning(ROLES["tuning"] + ROLES["held out"], "the roles that tune")
D.PORT.check_seeds(UNION, "tune.py: the three seed roles")
say("  check_seeds_tuning (>80 refused) passed the two roles this run reads and")
say("  port.check_seeds (>100 refused) passed all three; both are shown refusing in part 5")
say("  this script starts no simulation and builds no port, so the first item of the")
say("  seeds-never-used proof -- every simulation through port.sim on a run dict that")
say("  carries its seed -- is the evaluation script's and not this one's")

for name in ORDER:
    fit, held = SETS[name]
    cf, ch = arm_counts(fit), arm_counts(held)
    top_f, share_f = baseline(cf, len(fit))
    top_h, share_h = baseline(ch, len(held))
    say()
    say("candidate %s   tunes on %d rows, %d steps per epoch; held out %d rows"
        % (name, len(fit), -(-len(fit) // BATCH), len(held)))
    say("            tune     always %.3f s agrees with %.4f" % (top_f, share_f))
    say("            held out always %.3f s agrees with %.4f  <- the bar every agreement"
        % (top_h, share_h))
    say("                     below is read against")

# ===========================================================================
# 2  THE PROMPTS
# ===========================================================================
head("2  THE PROMPTS, RENDERED ONCE FOR EVERY ROW THIS RUN USES")
tok = D.load_tokenizer()
say(P.prepare_padding(tok))
t0 = time.time()
PROMPT = {}
for r in A_ROWS:
    text = D.text_of(r["state"], r["resolved"], r["history"], r["ts_ms"], r["neighbor"],
                     "the rows of candidate a")
    PROMPT[(r["seed"], r["k"])] = P.chat_text(tok, text)
say("rendered %d prompts in %.1f s, each one through decider.text_of, so the no-node-name"
    % (len(PROMPT), time.time() - t0))
say("assertion ran on every row of both candidates (b is a subset of a)")

CAND_IDS_LIST = D.candidate_ids(tok, PROMPT[(A_ROWS[0]["seed"], A_ROWS[0]["k"])])
CAND_IDS = torch.tensor(CAND_IDS_LIST, device=D.DEV)
say("candidate ids   %s for %s, each one token at this position"
    % (CAND_IDS_LIST, list(P.CANDIDATES)))

t0 = time.time()
LENS = {k: len(P.ids_of(tok, v)) for k, v in PROMPT.items()}
LONGEST = D.check_prompt_len(list(LENS.values()), MAX_PROMPT_TOKENS, "the prompts")
say("tokenized them in %.1f s: %d to %d tokens, median %d, bound %d, nothing truncated"
    % (time.time() - t0, min(LENS.values()), LONGEST,
       int(statistics.median(list(LENS.values()))), MAX_PROMPT_TOKENS))

# The fixed row the repeat check reads, chosen as the median-length prompt of candidate b's
# held-out rows: fixed by the data and not by a number typed here, and in both candidates'
# held-out slices, so the two candidates' repeat checks are of the same input.
_held_b = SETS["b"][1]
_med = statistics.median([LENS[(r["seed"], r["k"])] for r in _held_b])
REPEAT_ROW = min(_held_b, key=lambda r: (abs(LENS[(r["seed"], r["k"])] - _med),
                                         r["seed"], r["k"]))
REPEAT_PROMPT = PROMPT[(REPEAT_ROW["seed"], REPEAT_ROW["k"])]
say("repeat-check row  seed %d k %d, %d tokens (the median-length prompt of candidate b's"
    % (REPEAT_ROW["seed"], REPEAT_ROW["k"], LENS[(REPEAT_ROW["seed"], REPEAT_ROW["k"])]))
say("                  held-out slice, which is inside candidate a's as well)")


# ===========================================================================
# the tuning itself
# ===========================================================================
def lora_params(model, which="lora_"):
    """{name: tensor} of the adapter's parameters, detached on the cpu in float32."""
    return {n: p.detach().float().cpu().clone()
            for n, p in model.named_parameters() if which in n}


def read_three(model, prompt):
    """The three candidate logits of one prompt, through the shared read-out."""
    return D.readout(model, tok, CAND_IDS, prompt)


def held_out(model, rows, where):
    """(cross-entropy, agreement count, ties, per-answer counts) on a held-out slice.

    At the read-out the score will be taken at: bf16 base, float32 adapter, batch 1, the
    first maximum of the three candidate logits.  The cross-entropy is what chooses the
    epoch, because it is the quantity being optimised; the agreement is printed beside it
    for the reader and chooses nothing, and on 568 rows it can only move in steps of
    0.0018.
    """
    logits, targets, answers, ties, agree = [], [], {w: 0 for w in P.CANDIDATES}, 0, 0
    for r in rows:
        three = read_three(model, PROMPT[(r["seed"], r["k"])])
        index, tied = D.pick(three)
        D.check_pick(three, index, where)
        logits.append(three)
        targets.append(D.target_of(r["label"]))
        answers[P.CANDIDATES[index]] += 1
        ties += 1 if tied else 0
        agree += 1 if ARMS[index] == r["label"] else 0
    ce = float(F.cross_entropy(torch.tensor(logits, dtype=torch.float32),
                               torch.tensor(targets)))
    return ce, agree, ties, answers


def one_epoch(model, opt, rows, epoch):
    """One pass over a candidate's tuning rows in a seeded permutation.

    The forward is the SAME call the read-out makes (`use_cache=False`,
    `logits_to_keep=1`) and the loss is a three-way cross-entropy over the candidate
    logits at the last real position, so nothing is trained through a path inference does
    not use.  The autocast is the call the step cost was measured with and is therefore
    kept; what it means for the float32 adapter is said in part 6.

    The model is left in `eval()` throughout, as the measured call was: LoRA dropout is 0.0
    and there is no other training-mode layer in the adapter, so the two modes compute the
    same thing and the held-out pass between epochs is then the same call as the training
    forward but for the batch and the autocast.
    """
    rng = numpy.random.default_rng(SEED + epoch)
    order = rng.permutation(len(rows))
    losses = []
    t0 = time.time()
    for start in range(0, len(order), BATCH):
        idx = order[start:start + BATCH]
        batch = [rows[int(i)] for i in idx]
        texts = [PROMPT[(r["seed"], r["k"])] for r in batch]
        y = torch.tensor([D.target_of(r["label"]) for r in batch], device=D.DEV)
        iid, attn, pos, _lengths = P.batch_of(tok, texts, D.DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=iid, attention_mask=attn, position_ids=pos,
                        use_cache=False, logits_to_keep=1)
        loss = F.cross_entropy(out.logits[:, -1, :][:, CAND_IDS].float(), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    check_finite(losses, "candidate epoch %d losses" % (epoch + 1))
    return losses, time.time() - t0


def build(name):
    """A fresh base with a fresh untuned adapter on it, with every assertion printed.

    A fresh base per candidate and not one base with the adapter reset: `get_peft_model`
    wraps the base model's modules in place, so a second candidate on the same object would
    rest on the reset being exact.  A load is 2 s against the 6 to 14 minutes of a
    candidate, so the cheaper thing to trust is the loader.
    """
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer
    t0 = time.time()
    base = D.load_base()
    n_layers = base.config.text_config.num_hidden_layers
    n_shared = base.config.text_config.num_kv_shared_layers
    suffixes = tuple("." + t for t in LORA_TARGETS)
    all_named = [n for n, _ in base.named_modules() if n.endswith(suffixes)]
    lm_named = [n for n in all_named if n.startswith("model.language_model.layers.")]
    # The adapter's own random initialisation (`lora_A`; `lora_B` starts at exactly zero)
    # is the one draw this recipe did not name.  Seeded with the run's own seed here, and
    # per candidate, so that both candidates start from the same untuned adapter and a
    # re-run of this script produces the same one.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = get_peft_model(base, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
        target_modules=LORA_TARGETS, exclude_modules=LORA_EXCLUDE))
    matched = sorted(n.replace("base_model.model.", "", 1)
                     for n, mod in model.named_modules() if isinstance(mod, LoraLayer))
    check_targets(matched, lm_named, n_layers, n_shared, "the LoRA target set of %s" % name)
    trainable, total = model.get_nb_trainable_parameters()
    dtypes = [str(p.dtype) for n, p in model.named_parameters() if "lora_" in n]
    D.check_adapter_dtype(dtypes, "the adapter of candidate %s" % name)
    base_grad = [n for n, p in model.named_parameters()
                 if p.requires_grad and "lora_" not in n]
    if base_grad:
        raise Fail("%d base parameters are trainable: %s" % (len(base_grad), base_grad[:3]))
    zero_b = check_all_zero(lora_params(model, "lora_B"),
                            "the untuned adapter of candidate %s" % name)
    layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    D.check_unmerged(len(layers), [bool(m.merged) for m in layers],
                     "the adapter of candidate %s" % name)
    say("loaded the base and attached a fresh adapter in %.1f s" % (time.time() - t0))
    say("text layers %d, of which kv-shared %d, so the language model has %d projections"
        % (n_layers, n_shared, len(lm_named)))
    say("modules ending in one of the seven target names: %d in the whole model, %d of them"
        % (len(all_named), len(all_named) - len(lm_named)))
    say("in the vision and audio towers, which exclude_modules %r keeps out" % LORA_EXCLUDE)
    say("matched modules %d, exactly those projections and nothing else, e.g. %s"
        % (len(matched), matched[0]))
    say("trainable params %d of %d (%.4f %%), adapter dtype %s, adapter tensors %d of which"
        % (trainable, total, 100.0 * trainable / total, sorted(set(dtypes)), len(dtypes)))
    say("lora_B exactly zero %d, base params carrying a gradient %d, adapter attached and"
        % (zero_b, len(base_grad)))
    say("not merged (%d LoRA layers, none merged)" % len(layers))
    return base, model, matched, lm_named, n_layers, n_shared


RESULTS = {}
LAST = {}

for NAME in ORDER:
    head("3.%d  CANDIDATE %s" % (ORDER.index(NAME) + 1, NAME.upper()))
    FIT, HELD = SETS[NAME]
    HELD_COUNTS = arm_counts(HELD)
    TOP, SHARE = baseline(HELD_COUNTS, len(HELD))
    say("tunes on %d rows of seeds 1-%d, %d steps per epoch at batch %d"
        % (len(FIT), HELD_FROM - 1, -(-len(FIT) // BATCH), BATCH))
    say("held out %d rows of seeds %d-%d, baseline always %.3f s = %.4f"
        % (len(HELD), HELD_FROM, MAX_TUNE_SEED, TOP, SHARE))
    say()
    BASE, PM, MATCHED, LM_NAMED, N_LAYERS, N_SHARED = build(NAME)
    OPT = torch.optim.AdamW([p for p in PM.parameters() if p.requires_grad], lr=LR)
    BEFORE = lora_params(PM)
    torch.cuda.reset_peak_memory_stats()

    EPOCH_ROWS = []
    for E in range(EPOCHS):
        losses, secs = one_epoch(PM, OPT, FIT, E)
        where = "candidate %s, epoch %d" % (NAME, E + 1)
        epoch_dir = os.path.join(ADAPTERS, NAME, "epoch%d" % (E + 1))
        shutil.rmtree(epoch_dir, ignore_errors=True)
        PM.save_pretrained(epoch_dir)
        t0 = time.time()
        ce, agree, ties, answers = held_out(PM, HELD, where)
        EPOCH_ROWS.append({"epoch": E + 1, "ce": ce, "agree": agree, "ties": ties,
                           "answers": dict(answers), "loss_first": losses[0],
                           "loss_last": losses[-1], "loss_mean": statistics.fmean(losses),
                           "steps": len(losses), "train_s": secs,
                           "held_s": time.time() - t0, "dir": epoch_dir})
        say("epoch %d  permutation from default_rng(%d + %d), %d steps in %.1f s"
            % (E + 1, SEED, E, len(losses), secs))
        say("         training loss first %.4f last %.4f mean %.4f, all finite"
            % (losses[0], losses[-1], statistics.fmean(losses)))
        say("         held-out cross-entropy %.6f over %d rows in %.1f s"
            % (ce, len(HELD), time.time() - t0))
        say("         held-out agreement %d/%d = %.4f (baseline always %.3f s %.4f),"
            % (agree, len(HELD), agree / len(HELD), TOP, SHARE))
        say("         ties %d, answers %s" % (ties, dict(answers)))
    MOVED = check_changed(BEFORE, lora_params(PM), "the adapter of candidate %s" % NAME)
    TRAINED_B = lora_params(PM, "lora_B")
    say()
    say("adapter tensors moved: %d of %d, largest move %.4g; peak allocated %.2f GB"
        % (len(MOVED), len(BEFORE), max(MOVED.values()),
           torch.cuda.max_memory_allocated() / 2**30))

    # The epoch with the LOWEST held-out cross-entropy, the EARLIER epoch on a tie.  The
    # rule was fixed before any of these numbers existed, so nothing is chosen here.
    KEPT = min(range(EPOCHS), key=lambda i: (EPOCH_ROWS[i]["ce"], i))
    say()
    say("  epoch   train loss (first/last/mean)      held-out CE   agreement      baseline")
    for row in EPOCH_ROWS:
        say("  %5d   %7.4f %7.4f %7.4f        %11.6f   %4d/%-4d %.4f  %.4f%s"
            % (row["epoch"], row["loss_first"], row["loss_last"], row["loss_mean"],
               row["ce"], row["agree"], len(HELD), row["agree"] / len(HELD), SHARE,
               "  <- kept" if row["epoch"] == KEPT + 1 else ""))
    say()
    say("KEPT %s epoch %d: the lowest held-out cross-entropy of the three (%s), the rule"
        % (NAME, KEPT + 1, ", ".join("%.6f" % r["ce"] for r in EPOCH_ROWS)))
    say("fixed before any of them existed; a tie would have gone to the earlier epoch")

    # The adapter is read back from DISK and required to reproduce what it was chosen by,
    # because every later script of this step loads it from there and not from this process.
    del PM, OPT, BASE
    torch.cuda.empty_cache()
    KEPT_DIR = EPOCH_ROWS[KEPT]["dir"]
    BASE2 = D.load_base()
    RELOADED = D.attach_adapter(BASE2, KEPT_DIR, "the kept adapter of candidate %s" % NAME)
    ce2, agree2, ties2, _answers2 = held_out(RELOADED, HELD, "the reloaded %s" % NAME)
    check_reload({"cross-entropy": EPOCH_ROWS[KEPT]["ce"],
                  "agreement": EPOCH_ROWS[KEPT]["agree"],
                  "ties": EPOCH_ROWS[KEPT]["ties"]},
                 {"cross-entropy": ce2, "agreement": agree2, "ties": ties2},
                 "the kept adapter of candidate %s read back from disk" % NAME)
    say()
    say("read back from %s, attached and unmerged, adapter dtype float32:" % KEPT_DIR)
    say("  cross-entropy %.6f, agreement %d/%d, ties %d -- the three numbers the epoch was"
        % (ce2, agree2, len(HELD), ties2))
    say("  chosen by, to the last bit, so the file on disk is the model that was measured")
    say()
    say("the repeat check, with the kept adapter attached (seed %d k %d):"
        % (REPEAT_ROW["seed"], REPEAT_ROW["k"]))
    if not D.repeat_lines(lambda p: read_three(RELOADED, p), REPEAT_PROMPT, say,  # noqa: F821  RELOADED is bound just above; the `del` below unbinds the name and pyflakes reads that as undefined
                          "the repeat check of candidate %s" % NAME):
        raise Fail("the repeat check's own mutation did not fire for candidate %s" % NAME)
    del RELOADED, BASE2
    torch.cuda.empty_cache()

    # The kept epoch's files move up to ROOT/adapters/<candidate>, which is where the
    # frozen plan says an adapter of this step lives, and the two epochs not kept are
    # deleted there and then: an unkept epoch left on disk is an adapter a later script
    # could load by mistake, and the plan says they go.
    DEST = os.path.join(ADAPTERS, NAME)
    for fn in sorted(os.listdir(KEPT_DIR)):
        os.replace(os.path.join(KEPT_DIR, fn), os.path.join(DEST, fn))
    for row in EPOCH_ROWS:
        shutil.rmtree(row["dir"], ignore_errors=True)
    say()
    say("the kept epoch's files moved to %s and all %d epoch directories removed:"
        % (DEST, EPOCHS))
    DIGESTS = []
    for fn in sorted(os.listdir(DEST)):
        path = os.path.join(DEST, fn)
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        DIGESTS.append((fn, os.path.getsize(path), digest))
        say("  %-28s %9d bytes  sha256 %s" % (fn, os.path.getsize(path), digest))
    RESULTS[NAME] = {"kept": KEPT + 1, "epochs": EPOCH_ROWS, "digests": DIGESTS,
                     "held": len(HELD), "baseline": SHARE, "top": TOP,
                     "ce": ce2, "agree": agree2, "ties": ties2}
    LAST = {"name": NAME, "matched": MATCHED, "lm_named": LM_NAMED, "n_layers": N_LAYERS,
            "n_shared": N_SHARED, "trained_B": TRAINED_B, "before": BEFORE}

# ===========================================================================
# 4  BOTH CANDIDATES, SIDE BY SIDE
# ===========================================================================
head("4  THE TWO KEPT ADAPTERS")
say("Neither number below says which candidate is better: these are tuning-half rows, and")
say("the choice is the replay score's on seeds 81 to 100.")
say()
say("  cand  kept epoch  held-out CE   held-out agreement   baseline  rows  ties")
for name in ("a", "b"):
    r = RESULTS[name]
    say("  %-4s  %10d  %11.6f   %4d/%-4d = %.4f     %.4f   %5d  %4d"
        % (name, r["kept"], r["ce"], r["agree"], r["held"], r["agree"] / r["held"],
           r["baseline"], r["held"], r["ties"]))
say()
say("the agreement of candidate a's kept epoch over its %d held-out rows of seeds %d-%d is"
    % (RESULTS["a"]["held"], HELD_FROM, MAX_TUNE_SEED))
say("%d/%d = %.6f; the later agreement step reads the same rows with the same adapter and"
    % (RESULTS["a"]["agree"], RESULTS["a"]["held"],
       RESULTS["a"]["agree"] / RESULTS["a"]["held"]))
say("has to reproduce it exactly")
say()
say("every file under %s:" % ADAPTERS)
for root, dirs, files in os.walk(ADAPTERS):
    dirs.sort()
    for fn in sorted(files):
        say("  %s" % os.path.relpath(os.path.join(root, fn), ROOT))

# ===========================================================================
# 5  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT
# ===========================================================================
head("5  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT")
say("The mutation is of the INPUT, inside this process, because one script copy per check")
say("would be one half-hour run of a shared card per check.  The checks are functions for")
say("that reason.  First decider.py's own, which need no card and are the ones a reader")
say("can re-run on a machine without one; then this script's.")
say()
say("-- decider.py, with a stub read-out --")
SELF_OK = D.self_check(A_ROWS[0], say)
say()
say("-- tune.py; the module lists and the trained adapter mutated below are candidate %s's,"
    % LAST["name"])
say("   the last one tuned --")
FIRED = []


def shows(name, call):
    """Run a mutated input through a check and print the failure it produced."""
    try:
        call()
    except (Fail, SystemExit, KeyError) as exc:
        say("  %-24s FIRES: %s" % (name, exc))
        return True
    say("  %-24s DID NOT FIRE -- the check cannot catch its own mutation" % name)
    return False


FIRED.append(shows("digest", lambda: check_digest(CELL, "deadbeef", GOT, "mutated")))
FIRED.append(shows("counts", lambda: check_counts(dict(COUNTED, **{"a rows": 11439}),
                                                  EXPECT, "mutated")))
FIRED.append(shows("roles/range",
                   lambda: check_roles(dict(ROLES, tuning=ROLES["tuning"] + [81]),
                                       "mutated")))
FIRED.append(shows("roles/measuring",
                   lambda: check_roles(dict(ROLES, selection=[121]), "mutated")))
FIRED.append(shows("targets/tower",
                   lambda: check_targets(LAST["matched"] + ["model.audio_tower.q_proj"],
                                         LAST["lm_named"], LAST["n_layers"],
                                         LAST["n_shared"], "mutated")))
FIRED.append(shows("targets/arithmetic",
                   lambda: check_targets(LAST["matched"], LAST["lm_named"],
                                         LAST["n_layers"], 0, "mutated")))
FIRED.append(shows("targets/set",
                   lambda: check_targets(LAST["matched"][:-1], LAST["lm_named"],
                                         LAST["n_layers"], LAST["n_shared"], "mutated")))
FIRED.append(shows("all_zero", lambda: check_all_zero(LAST["trained_B"], "mutated")))
FIRED.append(shows("changed",
                   lambda: check_changed(LAST["before"], LAST["before"], "mutated")))
FIRED.append(shows("finite", lambda: check_finite([0.5, float("nan")], "mutated")))
FIRED.append(shows("prompt_len/lowered bound",
                   lambda: D.check_prompt_len(list(LENS.values()), 300, "mutated")))
FIRED.append(shows("reload",
                   lambda: check_reload({"cross-entropy": RESULTS["a"]["ce"]},
                                        {"cross-entropy": RESULTS["a"]["ce"] + 1e-6},
                                        "mutated")))


def _right_pad():
    """The shortest and the longest prompt right-padded, so the batch has padding to place.

    `prefill.batch_of` builds every training batch here and has two checks; this is the
    first.  The second, that the read position is length-1, is an invariant of the
    builder's own arithmetic and no argument can break it, so it is not counted here.
    """
    keys = sorted(LENS, key=lambda k: LENS[k])
    tok.padding_side = "right"
    try:
        P.batch_of(tok, [PROMPT[keys[0]], PROMPT[keys[-1]]], D.DEV)
    finally:
        tok.padding_side = "left"


FIRED.append(shows("batch_of/left-pad", _right_pad))
say("  batch_of/position        not mutable from an argument: it is an invariant of the")
say("                           builder's own arithmetic, and it is shown firing by the")
say("                           position mutation recorded in RUN.txt section 7")
say()
# The keep rule is a rule and not a check, so it is DEMONSTRATED and not counted among the
# mutations: a rule has no failing input, only a wrong answer.
_tied = [{"ce": 1.0}, {"ce": 1.0}, {"ce": 0.5}]
_pick = min(range(3), key=lambda i: (_tied[i]["ce"], i))
_tied2 = [{"ce": 0.5}, {"ce": 0.5}, {"ce": 1.0}]
_pick2 = min(range(3), key=lambda i: (_tied2[i]["ce"], i))
say("the keep rule, demonstrated and not counted as a check: on cross-entropies %s it"
    % [r["ce"] for r in _tied])
say("keeps epoch %d, and on %s it keeps epoch %d, i.e. the earlier of two tied epochs"
    % (_pick + 1, [r["ce"] for r in _tied2], _pick2 + 1))
say()
if all(FIRED) and SELF_OK:
    say("all %d mutations of this script and all of decider.py's produced a failure"
        % len(FIRED))
else:
    say("FAILURE OF THIS SECTION: %d of %d of this script's mutations did not produce a"
        % (len(FIRED) - sum(FIRED), len(FIRED)))
    say("failure, so those checks cannot catch what they exist to catch")

# ===========================================================================
# 6  WHAT THIS RUN DOES NOT SHOW
# ===========================================================================
head("6  WHAT THIS RUN DOES NOT SHOW")
say("- Nothing about which candidate is better.  Every number here is on seeds 1 to 80 and")
say("  the choice between a and b is the mean replay score on seeds 81 to 100.")
say("- Nothing about L in a simulation.  No decider ran, no port was built and the")
say("  send-time convention is exercised here only by decider.py's stub self-check.")
say("- Nothing about the agreement the ticket asks for.  That is on seeds 81 to 100; the")
say("  held-out agreement above is on seeds 71 to 80, which tuned nothing but which the")
say("  epoch was chosen on, so it is not a held-out number in the ordinary sense either.")
say("- The model being scored is not exactly the model that was optimised.  The training")
say("  forward is batch 16 under a bf16 autocast and the read-out is batch 1 with no")
say("  autocast, and the LoRA branch casts its input to the adapter's float32 while the")
say("  autocast would run the same matmul in bf16.  So the kept epoch minimises a")
say("  cross-entropy of the read-out arithmetic over weights fitted in another one.  Both")
say("  calls are the ones the plan froze and the measurements were made with; what this")
say("  difference costs is measured nowhere.")
say("- Nothing about the labels being right.  Two independent labellings of the same")
say("  decisions agreed on 0.7577 of them, so a model that matched these labels perfectly")
say("  would match a teacher that disagrees with itself on a quarter of its own work.")
say("- That the loss falls is not a result.  The held-out cross-entropy is what chose the")
say("  epoch and the agreement beside it is read against the baseline printed with it,")
say("  never against 1/3.")
say("- Nothing about the pass time with an adapter attached.  The repeat check makes two")
say("  passes and times neither.")
say()
say("wall clock    %.1f s (%.1f min) from the first line to this one"
    % (time.time() - T_START, (time.time() - T_START) / 60.0))
say("finished      %s" % time.strftime("%Y-%m-%dT%H:%M:%S%z"))
say("end of tune.py")
if not (all(FIRED) and SELF_OK):
    raise SystemExit(1)
