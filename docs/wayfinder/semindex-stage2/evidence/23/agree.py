#!/usr/bin/env python3
"""The agreement of the three model deciders with the hindsight labels, with no verdict.

    cd $ROOT && env PYTHONDONTWRITEBYTECODE=1 STORE=$ROOT/store \
        PYTHONPATH=$ROOT/repo/src HF_HUB_OFFLINE=1 ./.venv/bin/python -B \
        repo/docs/wayfinder/semindex-stage2/evidence/23/agree.py $ROOT [--dry] \
        > $ROOT/agree.log 2> $ROOT/agree.err

    --dry   seeds 71 to 80, the HELD-OUT slice of the tuning half, in place of 81 to 100.
            That slice is exactly candidate a's 1425 held-out rows, so the dry run carries
            a cross-check for free and asserts it: a's agreement here must equal, to the
            row, the held-out agreement `tune.txt` printed for a's kept epoch.

ONE PROCESS, BATCH 1, WHICH IS THE REFERENCE ARITHMETIC BY CONSTRUCTION, so this script
needs no reference gate and has none: `eval.py` gates K concurrent workers against a
single-process batch-1 read-out precisely so that its scores are read out at the
arithmetic this script and `latency.py` use.  That is said here rather than left to be
inferred from a gate's absence.

NO PASS OR FAIL IS ATTACHED to any number below, by the ticket's own instruction.  What
the agreement is NOT: it is not a score and does not predict one (the choice between the
candidates is the replay score's), it is not accuracy against testbed truth (two
independent labellings of the same decisions agreed on only 0.7577 of them), and it is
not evidence that the state was read (always answering 10.043 s already matches more than
half of these rows, which is why every agreement below is printed beside that baseline
and never beside 1/3).
"""
import os
import sys

# 2026-11-24: FIRST import, before torch / numpy / transformers.  `prefill.py` sets the
# thread caps, the offline HF variables, ROOT and the sys.path entries at ITS import time,
# and `decider.py` holds the read-out, so the agreement and the replay score are taken by
# the same call.
import prefill as P   # noqa: E402
import decider as D   # noqa: E402

import gzip        # noqa: E402
import json        # noqa: E402
import statistics  # noqa: E402
import time        # noqa: E402

ROOT = P.ROOT
PORT = D.PORT
ARMS = D.ARMS

ADAPTERS = os.path.join(ROOT, "adapters")
MODEL_DECIDERS = (("untuned", None),
                  ("a", os.path.join(ADAPTERS, "a")),
                  ("b", os.path.join(ADAPTERS, "b")))

# What the dry mode checks itself against, READ OFF the file that printed it rather than
# transcribed: tune.txt's own kept-epoch block carries candidate a's held-out row count,
# agreement, tie count and cross-entropy, and this run has to reproduce them or stop.
# 2026-11-24: 1425 and 777 were typed here, and the same three logit triples were typed
# into three scripts, so every archive of this step carried a hand-copied number.
KEPT = D.tune_kept()
DRY_ROWS = KEPT["a"]["rows"]
# The one number of the dry cross-check that tune.txt does not print as a count: the
# most-frequent-arm baseline of those rows.  `check_baseline` recomputes it from the rows
# this run reads and refuses a difference, so the literal is a bar and not an input.
DRY_TOP_A = 727
# The real mode's own pre-registered baseline, fixed in the frozen plan BEFORE any model
# was read out on these rows: 1498 of 2841 labelled rows of seeds 81 to 100 carry 10.043 s,
# which is 0.5273.  It is a subtraction of two generated files (`../22/labels.txt`'s
# whole-cell counts minus `rows.txt`'s candidate a counts), and this script recomputes it
# from the rows it reads and stops if it differs.
REAL_ROWS, REAL_TOP, REAL_BASELINE = 2841, 1498, 0.5273

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
def check_slice(labelled, unlabelled, want_rows, where):
    """The slice this run read is the slice the plan names, counted and not assumed.

    The denominator of every share below is `labelled`, so a slice that quietly lost or
    gained rows would move every number in this report while each of them still printed.
    """
    if want_rows is not None and labelled != want_rows:
        raise Fail("%s: %d labelled row(s), not the %d the plan names"
                   % (where, labelled, want_rows))
    if unlabelled < 0:
        raise Fail("%s: %d unlabelled rows" % (where, unlabelled))
    return labelled


def check_baseline(top_count, labelled, want_count, want_share, where):
    """The most-frequent-arm baseline of these rows is the pre-registered one.

    The bar every agreement is read against.  It was fixed before any model was read out
    on these rows, so reproducing it is what says the rows are the rows the bar was fixed
    on; a baseline computed after the fact and quietly different would turn a
    pre-registered prediction into a description.
    """
    if want_count is not None and top_count != want_count:
        raise Fail("%s: the most frequent arm carries %d of %d rows, not %d"
                   % (where, top_count, labelled, want_count))
    got = top_count / labelled
    if want_share is not None and round(got, 4) != want_share:
        raise Fail("%s: the baseline is %.6f, not the pre-registered %.4f"
                   % (where, got, want_share))
    return got


def check_cross(got, want, where):
    """One number reproduces the number another run of this step printed.

    The free cross-check of the dry mode: seeds 71 to 80 are exactly candidate a's
    held-out slice, so a's agreement here is `tune.txt`'s held-out agreement for the kept
    epoch, over the same rows with the same adapter at the same read-out.  A difference
    means the adapter on disk is not the one the tuning measured, or the read-out moved.

    2026-11-24: used for the held-out CROSS-ENTROPY as well, which is the stronger of the
    two because it is the quantity the epoch was chosen by while the agreement was printed
    beside it and chose nothing.  Compared at the six decimals tune.txt prints, which is
    why the caller rounds; the message prints both values through `repr` so that a float
    is not shown as an integer.
    """
    if got != want:
        raise Fail("%s: %r, against the %r already on record" % (where, got, want))
    return got


def check_table(table, labelled, where):
    """The 3 x 3 table accounts for every labelled row exactly once.

    A table whose cells do not sum to the denominator is a table that lost rows between
    the read-out and the count, and every share printed from it would still look like a
    share.
    """
    total = sum(sum(row) for row in table)
    if total != labelled:
        raise Fail("%s: the table holds %d decisions, not %d" % (where, total, labelled))
    return total


# ---------------------------------------------------------------------------
# the rows
# ---------------------------------------------------------------------------
def read_slice(lo, hi):
    """(every row of seeds lo to hi, the labelled ones) from the main cell's label file."""
    with gzip.open(os.path.join(ROOT, "ev22", "labels_p0.2.jsonl.gz"), "rt") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    PORT.check_seeds([r["seed"] for r in rows], "agree.py: the label file's seed column")
    mine = [r for r in rows if lo <= int(r["seed"]) <= hi]
    return rows, mine, [r for r in mine if r["label"] is not None]


def arm_counts(rows):
    """{arm: how many of these rows carry that label}."""
    counts = {arm: 0 for arm in ARMS}
    for r in rows:
        counts[r["label"]] += 1
    return counts


# ===========================================================================
# 0  THE CARD, THE MODE AND WHAT THIS RUN IS
# ===========================================================================
ARGV = sys.argv[1:]
DRY = "--dry" in ARGV
LO, HI = (71, 80) if DRY else (81, 100)

T_START = time.time()
head("0  THE CARD IS FREE, AND WHAT THIS RUN IS")
if DRY:
    say("A DRY RUN.  The rows are the HELD-OUT slice of the tuning half, seeds 71 to 80,")
    say("in place of the selection half.  Those rows chose the kept epoch of each")
    say("candidate, so an agreement on them is not a held-out number in the ordinary")
    say("sense and says NOTHING about which candidate is better -- that is the replay")
    say("score's on seeds 81 to 100.  What the slice does buy is a cross-check this run")
    say("asserts: candidate a's agreement here must equal tune.txt's %d/%d exactly."
        % (KEPT["a"]["agree"], DRY_ROWS))
else:
    say("The label agreement of the three model deciders on seeds 81 to 100 of the main")
    say("cell, against the baseline the frozen plan pre-registered at %.4f before any"
        % REAL_BASELINE)
    say("model was read out on these rows.  NO PASS OR FAIL IS ATTACHED.")
say()
_apps = D.nvidia_compute_apps()
say("nvidia-smi compute apps: %s" % (_apps.strip() if _apps.strip() else "(none)"))
D.check_card_free(_apps, "agree.py at start")
say("OK: no other compute process on the GPU")
say()
# 2026-11-24: printed because tune.txt carries no digest of any code file, so the
# adapters it wrote are bound to no code state and the archived output of every later
# script is the only place a reader can see which code produced a number.
CODE_PARTS = D.code_digests(os.path.abspath(__file__))
D.check_digest_cover(CODE_PARTS, ["agree.py"] + [n for n, _p in D.CODE_FILES],
                     "agree.py: the code digests")
say("the code that produced every number below, one sha256 per file:")
for _n, _sh in CODE_PARTS:
    say("  %-18s %s" % (_n, _sh))
say("combined          %s" % D.combined_sha(CODE_PARTS))
say("WHAT THESE DIGESTS DO NOT COVER:")
for _what in D.CODE_UNCOVERED:
    say("  - %s" % _what)

import numpy   # noqa: E402
import torch   # noqa: E402

torch.set_num_threads(8)

import peft          # noqa: E402
import transformers  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("FAIL: torch does not see a CUDA device")

say()
say("started         %s" % time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(T_START)))
say("torch %s (cuda %s), transformers %s, peft %s, numpy %s, python %s"
    % (torch.__version__, torch.version.cuda, transformers.__version__,
       peft.__version__, numpy.__version__, sys.version.split()[0]))
_FREE, _TOTAL = torch.cuda.mem_get_info()
say("gpu %s, %.2f GB free of %.2f GB"
    % (torch.cuda.get_device_name(0), _FREE / 2**30, _TOTAL / 2**30))
say("rows            seeds %d to %d" % (LO, HI))
say("read-out        bf16 base, float32 adapter, batch 1, the three candidate logits at")
say("                the last real position, first maximum of %s wins a tie"
    % list(P.CANDIDATES))
say("                ONE PROCESS at batch 1, which is eval.py's reference arithmetic by")
say("                construction, so this script needs no gate on concurrent workers")
say("                and has none")

# ===========================================================================
# 1  THE ROWS, THE SEEDS AND THE BASELINE
# ===========================================================================
head("1  THE ROWS, THE SEED ROLES, AND THE BASELINE EVERY AGREEMENT IS READ AGAINST")
ALL_ROWS, MINE, LABELLED = read_slice(LO, HI)
_seeds_in = sorted({int(r["seed"]) for r in ALL_ROWS})
_mine_seeds = sorted({int(r["seed"]) for r in MINE})
say("label file      %s" % os.path.join(ROOT, "ev22", "labels_p0.2.jsonl.gz"))
say("rows            %d, seed column min %d max %d, %d distinct"
    % (len(ALL_ROWS), _seeds_in[0], _seeds_in[-1], len(_seeds_in)))
say("this slice      %d rows of seeds %d to %d (%d distinct), of which %d carry a label"
    % (len(MINE), _mine_seeds[0], _mine_seeds[-1], len(_mine_seeds), len(LABELLED)))
say("                and %d are TIES that carry none.  The ties are excluded from every"
    % (len(MINE) - len(LABELLED)))
say("                denominator below and counted here: a tie is not a weak preference,")
say("                it is two arms that scored the same.")
check_slice(len(LABELLED), len(MINE) - len(LABELLED), DRY_ROWS if DRY else REAL_ROWS,
            "the slice of seeds %d to %d" % (LO, HI))

ROLES = {"tuning": list(range(1, 71)), "held out": list(range(71, 81)),
         "selection": list(range(81, 101)), "read out here": _mine_seeds}
WANT = {"tuning": (1, 70), "held out": (71, 80), "selection": (81, 100),
        "read out here": (LO, HI)}
UNION = D.check_roles(ROLES, WANT, "agree.py: the seed roles")
say()
for role in ("tuning", "held out", "selection", "read out here"):
    seeds = ROLES[role]
    say("  %-15s %2d seeds, %d to %d" % (role, len(seeds), min(seeds), max(seeds)))
say("  the union of the four meets range(121, 161) in %d seeds"
    % len(set(UNION) & set(range(121, 161))))
PORT.check_seeds(UNION, "agree.py: the four seed roles")
if DRY:
    D.check_seeds_tuning(_mine_seeds, "agree.py: the rows of a dry run")
    say("  check_seeds_tuning (>80 refused) passed the rows of this dry run and")
    say("  port.check_seeds (>100 refused) passed all four roles; both are shown refusing")
    say("  in part 4")
else:
    say("  port.check_seeds (>100 refused) passed all four roles and is shown refusing in")
    say("  part 4.  check_seeds_tuning is NOT applied to the rows read out here: these are")
    say("  the selection half, which is what the real mode exists to read")
say("  this script starts no simulation and builds no port, so the first item of the")
say("  seeds-never-used proof is eval.py's and not this one's")

COUNTS = arm_counts(LABELLED)
TOP = max(ARMS, key=lambda a: (COUNTS[a], -ARMS.index(a)))
BASE = check_baseline(COUNTS[TOP], len(LABELLED),
                      DRY_TOP_A if DRY else REAL_TOP,
                      None if DRY else REAL_BASELINE,
                      "the baseline of seeds %d to %d" % (LO, HI))
say()
say("the label shares of these %d rows:" % len(LABELLED))
for arm in ARMS:
    say("  %-9.3f s  %5d  %.4f" % (arm, COUNTS[arm], COUNTS[arm] / len(LABELLED)))
say("always %.3f s (the most frequent arm) agrees with %d/%d = %.4f  <- THE BAR, and every"
    % (TOP, COUNTS[TOP], len(LABELLED), BASE))
say("agreement below is read against it and never against 1/3")
if not DRY:
    say("It reproduces the plan's pre-registered %.4f, which was a subtraction of two"
        % REAL_BASELINE)
    say("already-generated files and was fixed before any model was read out on these rows.")
else:
    say("It is tune.txt's own held-out baseline for candidate a, %d/%d = 0.5102."
        % (DRY_TOP_A, DRY_ROWS))

# ===========================================================================
# 2  THE READ-OUTS
# ===========================================================================
head("2  THE THREE MODEL DECIDERS, READ OUT ON EVERY LABELLED ROW")
TOK = D.load_tokenizer()
say(P.prepare_padding(TOK))
_t0 = time.time()
PROMPT = {}
for r in LABELLED:
    PROMPT[(r["seed"], r["k"])] = P.chat_text(TOK, D.text_of(
        r["state"], r["resolved"], r["history"], r["ts_ms"], r["neighbor"],
        "the rows of seeds %d to %d" % (LO, HI)))
say("rendered %d prompts in %.1f s, each one through decider.text_of, so the no-node-name"
    % (len(PROMPT), time.time() - _t0))
say("assertion ran on every row this run reads out")
CAND_LIST = D.candidate_ids(TOK, PROMPT[(LABELLED[0]["seed"], LABELLED[0]["k"])])
CAND_IDS = torch.tensor(CAND_LIST, device=D.DEV)
say("candidate ids   %s for %s, each one token at this position"
    % (CAND_LIST, list(P.CANDIDATES)))
LENS = {k: len(P.ids_of(TOK, v)) for k, v in PROMPT.items()}
LONGEST = D.check_prompt_len(list(LENS.values()), 1024, "the prompts")
say("prompts         %d to %d tokens, median %d, bound 1024, nothing truncated"
    % (min(LENS.values()), LONGEST, int(statistics.median(list(LENS.values())))))

# The repeat row is the one `tune.py` read twice, so the triples below can be held against
# the ones tune.txt archived.  It is not in this slice in the real mode (it is a seed-71
# row), which is why it is rendered on its own rather than taken from PROMPT.
_repeat = None
for r in ALL_ROWS:
    if int(r["seed"]) == 71 and int(r["k"]) == 72:
        _repeat = r
REPEAT_PROMPT = P.chat_text(TOK, D.text_of(
    _repeat["state"], _repeat["resolved"], _repeat["history"], _repeat["ts_ms"],
    _repeat["neighbor"], "the repeat row"))
# Read off tune.txt and not typed here: the same six numbers were transcribed by hand
# into three scripts, so each of their archives carried a typed number and a re-tune
# would have had to be followed into three files instead of one.
TUNED_TRIPLES = D.tuned_triples()
say("repeat row      seed %d k %d, the row tune.py and eval.py both read twice; the"
    % (_repeat["seed"], _repeat["k"]))
say("                triples it is held against are READ OFF tune.txt, the tuning run's")
say("                own stdout, and not transcribed into this file")

RESULT = {}
for NAME, ADAPTER in MODEL_DECIDERS:
    say()
    say("-- %s --" % NAME)
    _t0 = time.time()
    base = D.load_base()
    model = base if ADAPTER is None else D.attach_adapter(
        base, ADAPTER, "the adapter of %s" % NAME)
    say("loaded in %.1f s%s" % (time.time() - _t0,
                                "" if ADAPTER is None else
                                ", adapter attached and unmerged, dtype float32"))
    if not D.repeat_lines(lambda p: D.readout(model, TOK, CAND_IDS, p), REPEAT_PROMPT,
                          lambda text="": say("  " + text),
                          "the repeat check of %s" % NAME):
        raise Fail("the repeat check's own mutation did not fire for %s" % NAME)
    if NAME in TUNED_TRIPLES:
        got = list(D.readout(model, TOK, CAND_IDS, REPEAT_PROMPT))
        D.check_triple(got, TUNED_TRIPLES[NAME], NAME,
                       "the read-out of seed %d k %d" % (_repeat["seed"], _repeat["k"]))
        say("  %s against tune.txt's %s: equal True" % (got, TUNED_TRIPLES[NAME]))
    _t0 = time.time()
    table = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    ties, answers = 0, [0, 0, 0]
    # 2026-11-24: the logits are kept so that the three-way cross-entropy can be computed
    # from them.  It costs no extra pass and it is the quantity the epoch was CHOSEN by,
    # so it holds this read-out against tune.txt's own held-out number rather than only
    # against the agreement count, and its standard error gives the margin that choice
    # rested on a scale (part 3).
    logits, targets = [], []
    for r in LABELLED:
        three = D.readout(model, TOK, CAND_IDS, PROMPT[(r["seed"], r["k"])])
        index, tied = D.pick(three)
        D.check_pick(three, index, "%s on seed %d k %d" % (NAME, r["seed"], r["k"]))
        table[D.target_of(r["label"])][index] += 1
        answers[index] += 1
        ties += 1 if tied else 0
        logits.append(three)
        targets.append(D.target_of(r["label"]))
    secs = time.time() - _t0
    check_table(table, len(LABELLED), "the table of %s" % NAME)
    agree = sum(table[i][i] for i in range(3))
    # The same call tune.py's `held_out` makes, on the same rows in the same order, so the
    # two numbers are comparable to the last bit and not to a tolerance.
    _per = torch.nn.functional.cross_entropy(
        torch.tensor(logits, dtype=torch.float32), torch.tensor(targets),
        reduction="none")
    ce = float(torch.nn.functional.cross_entropy(
        torch.tensor(logits, dtype=torch.float32), torch.tensor(targets)))
    ce_se = float(_per.std(unbiased=True)) / (len(LABELLED) ** 0.5)
    RESULT[NAME] = {"table": table, "agree": agree, "ties": ties, "answers": answers,
                    "secs": secs, "ce": ce, "ce_se": ce_se}
    say("  %d rows in %.1f s (%.1f ms per row, the render and the tokenizer call included)"
        % (len(LABELLED), secs, 1000.0 * secs / len(LABELLED)))
    say("  agreement %d/%d = %.6f, against the baseline %.4f"
        % (agree, len(LABELLED), agree / len(LABELLED), BASE))
    say("  exact ties resolved by the first-maximum rule: %d of %d decisions"
        % (ties, len(LABELLED)))
    # 2026-11-24: printed BESIDE the count of answers that are not the most frequent
    # arm, because the tie count alone does not show the rule's reach.  Every tie
    # resolves to the FIRST maximum, i.e. to a shorter arm, never to the longest, so the
    # ties are a subset of exactly these answers and the ratio is an upper bound on the
    # share of them the rule decided rather than a strict maximum.
    _notlong = answers[0] + answers[1]
    say("  answers that are not %r: %d, so the tie rule decided at most %.3f of them"
        % (P.CANDIDATES[2], _notlong, ties / _notlong if _notlong else 0.0))
    say("  label \\ answer      short      mid     long   |  per class")
    for i, arm in enumerate(ARMS):
        n = sum(table[i])
        say("  %-9.3f s     %8d %8d %8d   |  %5d  %s"
            % (arm, table[i][0], table[i][1], table[i][2], n,
               "%.4f" % (table[i][i] / n) if n else "--"))
    say("  answers           %8d %8d %8d   |  %5d"
        % (answers[0], answers[1], answers[2], sum(answers)))
    say("  answer shares     %8.4f %8.4f %8.4f"
        % tuple(a / len(LABELLED) for a in answers))
    say("  label shares      %8.4f %8.4f %8.4f"
        % tuple(COUNTS[arm] / len(LABELLED) for arm in ARMS))
    say("  A DECIDER THAT COLLAPSED ONTO ONE WORD IS VISIBLE AS ONE COLUMN above.")
    say("  three-way cross-entropy %.6f over those rows, standard error of the mean"
        % ce)
    say("  %.6f.  This is the quantity the kept epoch was CHOSEN by, recomputed here from"
        % ce_se)
    say("  the same logits the agreement above is counted from, at no extra pass.")
    if DRY and NAME == "a":
        check_cross(agree, KEPT["a"]["agree"],
                    "candidate a's agreement over the %d held-out rows of seeds 71 to 80"
                    % len(LABELLED))
        check_cross(round(ce, 6), KEPT["a"]["ce"],
                    "candidate a's held-out cross-entropy over those %d rows"
                    % len(LABELLED))
        say("  CROSS-CHECK: %d/%d and %.6f are tune.txt's own held-out agreement and"
            % (agree, len(LABELLED), ce))
        say("  cross-entropy for a's kept epoch, to the row and to six decimals, over the")
        say("  same rows with the same adapter at the same read-out.  The cross-entropy is")
        say("  the stronger of the two: it is what chose the epoch, and the agreement is")
        say("  printed beside it and chose nothing.")
    # Rebound to None rather than `del`-ed: the refcount drops either way and the
    # allocator releases the weights, and a `del` of a name bound in this loop makes a
    # static reader call the earlier use of it undefined.
    model = base = None
    torch.cuda.empty_cache()

# ===========================================================================
# 3  THE UNTUNED AND THE TUNED, SIDE BY SIDE
# ===========================================================================
head("3  THE UNTUNED MODEL AND THE TWO CANDIDATES, SIDE BY SIDE")
say("What the ticket's third acceptance item asks for, with no pass or fail attached.")
say()
say("  decider   agreement        share    - baseline   ties   answers short/mid/long")
for NAME, _a in MODEL_DECIDERS:
    r = RESULT[NAME]
    say("  %-8s  %5d/%-5d   %.6f   %+9.6f  %5d   %d/%d/%d"
        % (NAME, r["agree"], len(LABELLED), r["agree"] / len(LABELLED),
           r["agree"] / len(LABELLED) - BASE, r["ties"],
           r["answers"][0], r["answers"][1], r["answers"][2]))
say()
say()
say("  decider   held-out cross-entropy   standard error of the mean")
for NAME, _a in MODEL_DECIDERS:
    say("  %-8s  %22.6f   %.6f" % (NAME, RESULT[NAME]["ce"], RESULT[NAME]["ce_se"]))
say()
say("WHAT CHOSE EACH KEPT EPOCH, AND BY HOW MUCH, from tune.txt's own three numbers:")
for NAME in ("a", "b"):
    kept, up, margin = D.margin_of(KEPT[NAME]["epoch_ce"])
    say("  candidate %s kept epoch %d at %.6f; its runner-up is %.6f, so the margin the"
        % (NAME, KEPT[NAME]["epoch"], kept, up))
    say("  choice rests on is %.6f, which is %.3f %% of the value, over %d rows"
        % (margin, 100.0 * margin / kept, KEPT[NAME]["rows"]))
    # The standard error is only a scale for the margin when it is over the SAME rows the
    # margin was taken on.  In the dry mode candidate a's held-out slice IS these 1425
    # rows, so the ratio is meaningful for a; candidate b's filter keeps 568 of them, so
    # the figure below is over a wider and differently shaped population than b's margin
    # and no ratio is printed for it.  Saying so is the point: a ratio of two numbers from
    # two populations reads like evidence and is not.
    if NAME in RESULT and KEPT[NAME]["rows"] == len(LABELLED):
        say("  for scale, the standard error of this run's own mean per-row cross-entropy")
        say("  over the SAME %d rows is %.6f, i.e. %.2f times that margin"
            % (len(LABELLED), RESULT[NAME]["ce_se"],
               RESULT[NAME]["ce_se"] / margin if margin else float("inf")))
    elif NAME in RESULT:
        say("  NO RATIO IS PRINTED for this candidate: its margin is over its own %d"
            % KEPT[NAME]["rows"])
        say("  filtered rows and this run reads out %d, so the standard error above"
            % len(LABELLED))
        say("  (%.6f) is over a different and differently shaped population."
            % RESULT[NAME]["ce_se"])
say("CANDIDATE a's EPOCH IS NOT DISTINGUISHED BY ANYTHING IN THIS STEP, and that is said")
say("here rather than left to a reader to divide two numbers.  A ratio is printed above")
say("only where this run reads out the very rows the margin was taken on, because a ratio")
say("of two numbers from two populations reads like evidence and is not; and where it is")
say("printed, the standard error is the LARGER of the two, so the margin is a fraction of")
say("the uncertainty on one epoch's own mean rather than a multiple of it.  The quantity")
say("that would settle it is the standard error of the PAIRED per-row cross-entropy")
say("differences between two epochs, which is smaller than the unpaired figure above and")
say("is NOT computable any more: the keep rule deleted the two epochs that were not kept,")
say("so no per-row cross-entropy of epoch 1 or epoch 2 exists on either machine.  So a's")
say("kept epoch is reported as arbitrary among its three, and the number above is its")
say("scale and not its test.")
say()
say("the baseline of these rows is %.4f and the labels' own self-agreement across two" % BASE)
say("independent 8-pass labellings is 0.7577 (../22/stability.txt), so an agreement above")
say("that is agreement with one labelling's noise and not with a better answer.  Neither")
say("edge is a pass line: no pass or fail is attached to any of these numbers.")
if DRY:
    say()
    say("candidate b's own held-out slice is 568 of these %d rows and tune.txt's 408/568 is"
        % len(LABELLED))
    say("NOT the number above: b's filter is a paired-t rule over each row's own passes and")
    say("restating it here would be a third definition of what candidate b is, so this")
    say("cross-check is candidate a's alone.")

# ===========================================================================
# 4  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT
# ===========================================================================
head("4  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT")
say("decider.py's own eighteen are archived in tune.txt part 5 and are not repeated; what")
say("is below is this script's own, plus the two of decider.py's this script is a")
say("load-bearing caller of.")
say()
FIRED = []


def shows(name, call):
    """Run a mutated input through a check and print the failure it produced."""
    try:
        call()
    except (Fail, SystemExit, KeyError, ValueError) as exc:
        say("  %-26s FIRES: %s" % (name, exc))
        return True
    say("  %-26s DID NOT FIRE -- the check cannot catch its own mutation" % name)
    return False


FIRED.append(shows("card_free",
                   lambda: D.check_card_free("4321, python, 10240 MiB", "mutated")))
FIRED.append(shows("seeds_tuning", lambda: D.check_seeds_tuning([71, 81], "mutated")))
FIRED.append(shows("port.check_seeds",
                   lambda: PORT.check_seeds([PORT.MAX_SEED + 1], "mutated")))
FIRED.append(shows("roles/range",
                   lambda: D.check_roles(dict(ROLES, tuning=[81]), WANT, "mutated")))
FIRED.append(shows("roles/measuring",
                   lambda: D.check_roles(dict(ROLES, selection=[121]), WANT, "mutated")))
FIRED.append(shows("slice/one row lost", lambda: check_slice(
    len(LABELLED) - 1, 0, len(LABELLED), "mutated")))
FIRED.append(shows("baseline/count", lambda: check_baseline(
    COUNTS[TOP] + 1, len(LABELLED), COUNTS[TOP], None, "mutated")))
FIRED.append(shows("baseline/share", lambda: check_baseline(
    COUNTS[TOP], len(LABELLED), None, round(BASE + 0.001, 4), "mutated")))
FIRED.append(shows("cross-check/agreement", lambda: check_cross(
    RESULT["a"]["agree"] + 1, RESULT["a"]["agree"], "mutated")))
FIRED.append(shows("cross-check/cross-entropy", lambda: check_cross(
    round(RESULT["a"]["ce"] + 1e-6, 6), round(RESULT["a"]["ce"], 6), "mutated")))
FIRED.append(shows("table/one decision lost", lambda: check_table(
    [[0, 0, 0], [0, 0, 0], [0, 0, 1]], len(LABELLED), "mutated")))
FIRED.append(shows("target_of", lambda: D.target_of(7.0)))
FIRED.append(shows("pick/tie", lambda: D.check_pick((1.0, 0.0, 1.0), 2, "mutated")))
FIRED.append(shows("prompt_len/lowered bound",
                   lambda: D.check_prompt_len(list(LENS.values()), 300, "mutated")))
FIRED.append(shows("triple", lambda: D.check_triple([1.0, 2.0, 3.0], [1.0, 2.0, 4.0],
                                                   "a", "mutated")))
FIRED.append(shows("digest_cover", lambda: D.check_digest_cover(
    CODE_PARTS[1:], ["agree.py"] + [n for n, _p in D.CODE_FILES], "mutated")))
FIRED.append(shows("no_node_name", lambda: D.check_no_node_name(
    REPEAT_PROMPT.replace("p1", "h3", 1), "mutated")))
say()
say("The repeat check's own mutation is in part 2, once per model decider, inside")
say("decider.repeat_lines: one of the two stored logit triples is moved by one unit.")
say()
say("WHAT HAS NO MUTATION, named and not counted: nothing of this script.  The three")
say("read-outs themselves are not a check and carry no pass line.")
say()
if all(FIRED):
    say("all %d mutations of this script produced a failure" % len(FIRED))
else:
    say("FAILURE OF THIS SECTION: %d of %d mutations did not produce a failure"
        % (len(FIRED) - sum(FIRED), len(FIRED)))

# ===========================================================================
# 5  WHAT THIS RUN DOES NOT SHOW
# ===========================================================================
head("5  WHAT THIS RUN DOES NOT SHOW")
if DRY:
    say("- NOTHING about which candidate is better, and nothing the ticket asks for.  These")
    say("  are seeds 71 to 80, the rows that chose each candidate's kept epoch, so they")
    say("  are neither the selection half nor a held-out slice in the ordinary sense.")
say("- Nothing about a score.  The choice between the candidates is the mean replay")
say("  score's and never this number's, and an agreement near the labels and an agreement")
say("  far from them are equally uninformative about that score: the states these labels")
say("  sit on came from a uniformly random roll-in, the label is the best arm under an F")
say("  suffix L does not supply, and in the replay L drives the run and reads its own")
say("  choices back in the history column.")
say("- Nothing about accuracy against testbed truth.  The labels are the replay's own")
say("  hindsight and two independent 8-pass labellings of the same decisions agreed on")
say("  0.7577 of them, with the 0.90 bar fixed in advance never reached.")
say("- Nothing about the state being read rather than the clock.  The rendered text")
say("  carries the history length and the ages of the last exchanges, so the position in")
say("  the run is readable from the input, and the labels' own reliability moves with that")
say("  position (0.5101 in a run's first quarter against 0.9597 in its last).")
say("- Nothing about a class prior being learnt instead of the state.  The baseline above")
say("  is what always answering the most frequent arm already buys.")
say("- Nothing about the pass time.  The ms per row above includes the render and the")
say("  tokenizer call and is not a latency measurement; latency.py is.")
say()
say("wall clock    %.1f s (%.1f min) from the first line to this one"
    % (time.time() - T_START, (time.time() - T_START) / 60.0))
say("finished      %s" % time.strftime("%Y-%m-%dT%H:%M:%S%z"))
say("end of agree.py")
if not all(FIRED):
    raise SystemExit(1)
