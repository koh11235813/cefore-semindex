#!/usr/bin/env python3
"""What one decision's forward pass costs on this card, with the adapter attached.

    cd $ROOT && env PYTHONDONTWRITEBYTECODE=1 STORE=$ROOT/store \
        PYTHONPATH=$ROOT/repo/src HF_HUB_OFFLINE=1 ./.venv/bin/python -B \
        repo/docs/wayfinder/semindex-stage2/evidence/23/latency.py $ROOT <a|b> [--dry] \
        > $ROOT/latency.log 2> $ROOT/latency.err

    --dry   the run is named a dry run in its own output and the candidate is whatever
            the command line says.  The REAL run takes the candidate the replay score
            chose; a dry run takes one before that choice exists, so its candidate name
            carries no claim that this is the chosen one.

WHAT IS TIMED, AND WHAT IS PRINTED BESIDE IT
  The pass, the render and the tokenizer call are timed SEPARATELY and the pass is the
  only one of the three the 100 ms line is printed beside.  They are separated because
  the tuning's held-out pass cost 72 ms per row against the 43.02 ms a pass is on record
  at, and a figure that quoted the sum as "the pass" would not be comparable with that
  number.  A node pays all three, so the sum is printed as well and named as the sum.

  The candidate's kept epoch is attached and UNMERGED, bf16 base and float32 adapter,
  batch 1, one process, which is the arithmetic every score and every agreement of this
  step is read out at.  `merge_and_unload` would change it and would need its own
  agreement check against the unmerged read-out.

  The untuned base is timed in the SAME PROCESS, alternating pass by pass with the
  tuned one, so that what the adapter costs on top is a difference of two medians taken
  under one allocator and one clock rather than a comparison across runs.  That is
  `precheck.txt` part 3's own shape and it is there for its reason: the first run of that
  script timed two variants in two loops minutes apart and let drift supply the sign.

  THE PROMPT IS THE MEDIAN-LENGTH ONE OF THE LABELLED ROWS OF SEEDS 1 TO 80, not of the
  selection half: timing the selection half's median prompt would be a use of those seeds
  for something this step promised they would not be used for, and it would buy a second
  prompt length as well as a second model.

NO VERDICT IS ATTACHED.  100 ms is printed beside the numbers because the ticket asks for
that, and this card is not a node's hardware.
"""
import os
import sys

# 2026-11-24: FIRST import, before torch / numpy / transformers.  `prefill.py` sets the
# thread caps, the offline HF variables, ROOT and the sys.path entries at ITS import time,
# and `decider.py` holds the read-out this script times.
import prefill as P   # noqa: E402
import decider as D   # noqa: E402

import gzip        # noqa: E402
import json        # noqa: E402
import statistics  # noqa: E402
import time        # noqa: E402

ROOT = P.ROOT
PORT = D.PORT

WARMUP = 10
TIMED = 300
BUDGET_MS = 100.0            # the node's loop, printed beside and never compared against
ADAPTERS = os.path.join(ROOT, "adapters")
# `precheck.txt` part 1's own row and `tune.txt`'s own median, both printed beside this
# run's choice so that a reader sees which prompt was timed and how it compares.
PRECHECK_TOKENS, PRECHECK_MEDIAN_MS, PRECHECK_P99_MS = 478, 43.02, 49.10
TUNE_MEDIAN_TOKENS = 473
# Read off tune.txt and not typed here: the same six numbers were transcribed by hand
# into three scripts, so each of their archives carried a typed number.
KEPT = D.tune_kept()
TUNED_TRIPLES = {name: got["triple"] for name, got in KEPT.items()}

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
def check_median(chosen, lengths, where):
    """The timed prompt is the one closest to the median length, and none is closer.

    The plan says "the median-length prompt of the seeds 1 to 80 rows", and over an even
    number of rows the median is an average of two lengths that need not be any row's.
    So the rule is "closest to the median, the earlier (seed, k) winning a tie", and this
    is the check that the row handed to the timer satisfies it: without it a quietly
    mis-selected row would move every number below while all of them still printed.
    """
    median = statistics.median(lengths)
    best = min(abs(n - median) for n in lengths)
    if abs(chosen - median) != best:
        raise Fail("%s: the timed prompt is %d tokens, %r from the median %r, and a row "
                   "%r away exists" % (where, chosen, abs(chosen - median), median, best))
    return median


def check_timed(times, want, where):
    """Exactly `want` timed passes, every one of them a finite positive number of ms.

    A zero would be a pass that was never synchronised (the kernel still queued when the
    clock was read), and a non-finite one a clock that went backwards; either would make
    the median below a median of something other than passes.
    """
    if len(times) != want:
        raise Fail("%s: %d timed pass(es), not %d" % (where, len(times), want))
    bad = [(i, t) for i, t in enumerate(times) if not (t > 0.0 and t == t and
                                                       t != float("inf"))]
    if bad:
        raise Fail("%s: %d pass time(s) are not finite and positive, first at %d: %r"
                   % (where, len(bad), bad[0][0], bad[0][1]))
    return len(times)


def check_cost_order(pass_ms, render_ms, token_ms, where):
    """The three parts of a decision are each a separate measurement, and named as such.

    The one thing a reader may not do with these numbers is read the pass as the cost of a
    decision, so the parts are required to be three distinct measurements rather than one
    number printed three times -- which is what a copy-paste error in the timing loop
    would produce and what nothing else here would catch.
    """
    if pass_ms == render_ms or pass_ms == token_ms or render_ms == token_ms:
        raise Fail("%s: two of the three medians are the same number (%r, %r, %r), so one "
                   "loop timed another loop's work"
                   % (where, pass_ms, render_ms, token_ms))
    return pass_ms + render_ms + token_ms


def quantiles(times):
    """(median, p90, p99, max) of a list of ms, by position in the sorted list."""
    s = sorted(times)
    return (statistics.median(s), s[int(0.90 * (len(s) - 1))],
            s[int(0.99 * (len(s) - 1))], s[-1])


# ===========================================================================
# 0  THE CARD, THE CANDIDATE AND WHAT THIS RUN IS
# ===========================================================================
ARGV = sys.argv[1:]
DRY = "--dry" in ARGV
NAMES = [a for a in ARGV[1:] if not a.startswith("--")]
if len(NAMES) != 1 or NAMES[0] not in ("a", "b"):
    raise SystemExit("usage: latency.py <ROOT> <a|b> [--dry]")
CAND = NAMES[0]
ADAPTER = os.path.join(ADAPTERS, CAND)

T_START = time.time()
head("0  THE CARD IS FREE, AND WHAT THIS RUN IS")
if DRY:
    say("A DRY RUN.  The candidate timed here is %r because the command line said so, and"
        % CAND)
    say("NOT because any score chose it: the choice between a and b is the mean replay")
    say("score's on seeds 81 to 100 and is ana.py's to make.  The pass time itself is a")
    say("real measurement of a real adapter and does not depend on which candidate wins.")
else:
    say("The pass time of one decision with candidate %r's kept epoch attached and" % CAND)
    say("unmerged, beside 100 ms, with NO VERDICT attached.")
say()
_apps = D.nvidia_compute_apps()
say("nvidia-smi compute apps: %s" % (_apps.strip() if _apps.strip() else "(none)"))
D.check_card_free(_apps, "latency.py at start")
say("OK: no other compute process on the GPU, checked by this script and not by hand")
say()
# 2026-11-24: printed because tune.txt carries no digest of any code file, so the archived
# output of every later script is the only place a reader can see which code produced a
# number.
CODE_PARTS = D.code_digests(os.path.abspath(__file__))
D.check_digest_cover(CODE_PARTS, ["latency.py"] + [n for n, _p in D.CODE_FILES],
                     "latency.py: the code digests")
say("the code that produced every figure below, one sha256 per file:")
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
say("candidate       %s, adapter %s" % (CAND, ADAPTER))
say("read-out        bf16 base, float32 adapter, batch 1, ONE PROCESS, attached and not")
say("                merged -- the arithmetic every score and every agreement of this step")
say("                is read out at, so this is the pass those numbers were made of")
say("passes          %d warm-up then %d timed, each bracketed by torch.cuda.synchronize(),"
    % (WARMUP, TIMED))
say("                the tuned and the untuned model alternating pass by pass")

# ===========================================================================
# 1  THE ROW
# ===========================================================================
head("1  THE PROMPT THAT IS TIMED, AND THE SEEDS IT COMES FROM")
with gzip.open(os.path.join(ROOT, "ev22", "labels_p0.2.jsonl.gz"), "rt") as fh:
    ALL_ROWS = [json.loads(line) for line in fh if line.strip()]
PORT.check_seeds([r["seed"] for r in ALL_ROWS], "latency.py: the label file's seed column")
HALF = [r for r in ALL_ROWS if int(r["seed"]) <= D.MAX_TUNE_SEED and r["label"] is not None]
HALF_SEEDS = D.check_seeds_tuning([r["seed"] for r in HALF], "latency.py: the rows it reads")
_seeds_in = sorted({int(r["seed"]) for r in ALL_ROWS})
say("label file      %s" % os.path.join(ROOT, "ev22", "labels_p0.2.jsonl.gz"))
say("rows            %d, seed column min %d max %d, %d distinct"
    % (len(ALL_ROWS), _seeds_in[0], _seeds_in[-1], len(_seeds_in)))
say("labelled rows of seeds 1-%d: %d, over %d seeds.  The selection half is NOT read: the"
    % (D.MAX_TUNE_SEED, len(HALF), len(HALF_SEEDS)))
say("row list is cut at seed %d before anything is rendered." % D.MAX_TUNE_SEED)
# 2026-11-24: the population is PRINTED because the frozen plan says "the median-length
# prompt of the seeds 1 to 80 rows" and this script takes the median over the LABELLED
# ones, which is 11438 of the 12868 rows of that half: the 1430 ties that carry no label
# are left out.  That is the population tune.py rendered and took its own median over, so
# the comparison with tune.txt's 473 is apples to apples -- but it is a narrower set than
# the plan's words, and a reader has to be told which one the figure is of.
say("THE POPULATION THE MEDIAN IS TAKEN OVER: the %d LABELLED rows of seeds 1 to %d, not"
    % (len(HALF), D.MAX_TUNE_SEED))
say("all %d rows of that half.  The %d rows left out are TIES that carry no label, which"
    % (len([r for r in ALL_ROWS if int(r["seed"]) <= D.MAX_TUNE_SEED]),
       len([r for r in ALL_ROWS
            if int(r["seed"]) <= D.MAX_TUNE_SEED and r["label"] is None])))
say("is the population tune.py rendered and took its own median over, so the comparison")
say("with tune.txt's median below is of one population and not two.  The plan's sentence")
say("says \"the seeds 1 to 80 rows\" and this is narrower than that sentence.")

ROLES = {"tuning": list(range(1, 71)), "held out": list(range(71, 81)),
         "selection": list(range(81, 101)), "read here": HALF_SEEDS}
WANT = {"tuning": (1, 70), "held out": (71, 80), "selection": (81, 100),
        "read here": (1, D.MAX_TUNE_SEED)}
UNION = D.check_roles(ROLES, WANT, "latency.py: the seed roles")
say()
for role in ("tuning", "held out", "selection", "read here"):
    seeds = ROLES[role]
    say("  %-12s %2d seeds, %d to %d" % (role, len(seeds), min(seeds), max(seeds)))
say("  the union of the four meets range(121, 161) in %d seeds"
    % len(set(UNION) & set(range(121, 161))))
PORT.check_seeds(UNION, "latency.py: the four seed roles")
say("  check_seeds_tuning (>80 refused) passed the rows this script reads and")
say("  port.check_seeds (>100 refused) passed all four roles; both are shown refusing in")
say("  part 5.  This script starts no simulation and builds no port, so the first item of")
say("  the seeds-never-used proof is eval.py's and not this one's")

TOK = D.load_tokenizer()
say()
say(P.prepare_padding(TOK))
_t0 = time.time()
TEXT = {}
for r in HALF:
    TEXT[(r["seed"], r["k"])] = D.text_of(r["state"], r["resolved"], r["history"],
                                          r["ts_ms"], r["neighbor"], "the rows of 1-80")
say("rendered %d observations in %.1f s, each through decider.text_of, so the no-node-name"
    % (len(TEXT), time.time() - _t0))
say("assertion ran on every one")
PROMPT = {k: P.chat_text(TOK, v) for k, v in TEXT.items()}
LENS = {k: len(P.ids_of(TOK, v)) for k, v in PROMPT.items()}
D.check_prompt_len(list(LENS.values()), 1024, "the prompts")
MEDIAN = statistics.median(list(LENS.values()))
KEY = min(LENS, key=lambda k: (abs(LENS[k] - MEDIAN), k))
check_median(LENS[KEY], list(LENS.values()), "the timed prompt")
ROW = next(r for r in HALF if (r["seed"], r["k"]) == KEY)
TIMED_PROMPT = PROMPT[KEY]
TIMED_IDS = P.ids_of(TOK, TIMED_PROMPT)
say("prompt lengths  %d to %d tokens, median %r (tune.txt's own median over these rows is"
    % (min(LENS.values()), max(LENS.values()), MEDIAN))
say("                %d)" % TUNE_MEDIAN_TOKENS)
say("timed prompt    seed %d k %d, %d tokens, the row closest to that median with the"
    % (ROW["seed"], ROW["k"], LENS[KEY]))
say("                earlier (seed, k) winning a tie.  precheck.txt part 1 timed a")
say("                %d-token row of the same half at %.2f ms median and %.2f p99, so"
    % (PRECHECK_TOKENS, PRECHECK_MEDIAN_MS, PRECHECK_P99_MS))
say("                the figures below are comparable with that one and differ from it in")
say("                the adapter, in %d tokens of prompt, AND in one host-to-device copy:"
    % abs(LENS[KEY] - PRECHECK_TOKENS))
say("                decider.forward_three builds torch.tensor([ids], device=cuda:0)")
say("                INSIDE the timed call, which precheck.py's forward_last did NOT.")
say("                So the untuned median in part 3 is precheck's quantity plus that")
say("                copy, over %d tokens fewer; the two are named side by side there."
    % abs(LENS[KEY] - PRECHECK_TOKENS))

# ===========================================================================
# 2  THE TWO MODELS
# ===========================================================================
head("2  THE TUNED AND THE UNTUNED MODEL, BOTH IN THIS PROCESS")
_t0 = time.time()
BASE_PLAIN = D.load_base()
BASE_TUNED = D.load_base()
TUNED = D.attach_adapter(BASE_TUNED, ADAPTER, "the kept adapter of candidate %s" % CAND)
say("loaded two copies of the base in %.1f s and attached candidate %s's kept adapter to"
    % (time.time() - _t0, CAND))
say("one of them, unmerged, adapter dtype float32 over the bf16 base (both asserted).")
_FREE2, _ = torch.cuda.mem_get_info()
say("gpu memory      %.2f GB free with both models on the card, from %.2f GB before"
    % (_FREE2 / 2**30, _FREE / 2**30))
CAND_LIST = D.candidate_ids(TOK, TIMED_PROMPT)
CAND_IDS = torch.tensor(CAND_LIST, device=D.DEV)
say("candidate ids   %s for %s, each one token at this position"
    % (CAND_LIST, list(P.CANDIDATES)))
say()
_repeat = next(r for r in ALL_ROWS if int(r["seed"]) == 71 and int(r["k"]) == 72)
REPEAT_PROMPT = P.chat_text(TOK, D.text_of(
    _repeat["state"], _repeat["resolved"], _repeat["history"], _repeat["ts_ms"],
    _repeat["neighbor"], "the repeat row"))
say("the repeat check, with the adapter attached (seed %d k %d, the row tune.py read"
    % (_repeat["seed"], _repeat["k"]))
say("twice):")
if not D.repeat_lines(lambda p: D.readout(TUNED, TOK, CAND_IDS, p), REPEAT_PROMPT,
                      lambda text="": say("  " + text),
                      "the repeat check of candidate %s" % CAND):
    raise Fail("the repeat check's own mutation did not fire")
GOT = list(D.readout(TUNED, TOK, CAND_IDS, REPEAT_PROMPT))
D.check_triple(GOT, TUNED_TRIPLES[CAND], CAND,
               "the read-out of seed %d k %d" % (_repeat["seed"], _repeat["k"]))
say("  %s against tune.txt's %s (read off the archive, not typed here): equal True"
    % (GOT, TUNED_TRIPLES[CAND]))
say("  so the adapter timed below is the file the tuning measured, read out the same way")

# ===========================================================================
# 3  THE PASS
# ===========================================================================
head("3  THE PASS, ALTERNATING TUNED AND UNTUNED, %d TIMED EACH" % TIMED)

def one_pass(model):
    """One batch-1 forward pass of the shared read-out, in ms, synchronised both sides."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    D.forward_three(model, CAND_IDS, TIMED_IDS)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


for _ in range(WARMUP):
    one_pass(TUNED)
    one_pass(BASE_PLAIN)
T_TUNED, T_PLAIN = [], []
for _ in range(TIMED):
    T_TUNED.append(one_pass(TUNED))
    T_PLAIN.append(one_pass(BASE_PLAIN))
check_timed(T_TUNED, TIMED, "the tuned passes")
check_timed(T_PLAIN, TIMED, "the untuned passes")
QT, QP = quantiles(T_TUNED), quantiles(T_PLAIN)
say("  model              median      p90      p99      max   |  beside the budget")
say("  %-16s %8.2f %8.2f %8.2f %8.2f   |  %.0f ms"
    % ("candidate " + CAND, QT[0], QT[1], QT[2], QT[3], BUDGET_MS))
say("  %-16s %8.2f %8.2f %8.2f %8.2f   |  %.0f ms"
    % ("untuned base", QP[0], QP[1], QP[2], QP[3], BUDGET_MS))
# Column by column, i.e. the difference of the two medians and the difference of the two
# p99s -- NOT the median or the p99 of a per-pass difference, which would be a third
# quantity and is not what two alternating loops measure.
say("  %-16s %+8.2f %+8.2f %+8.2f %+8.2f   |  the adapter's cost, column by column"
    % ("difference", QT[0] - QP[0], QT[1] - QP[1], QT[2] - QP[2], QT[3] - QP[3]))
say()
say("NO VERDICT.  100 ms is the node's loop and is printed beside these numbers because")
say("the ticket asks for that; this card is an RTX PRO 5000 Blackwell and not a node's")
say("hardware, and the pass alone is not a decision.")
say("precheck.txt part 1 has the untuned pass at %.2f ms median and %.2f p99 on a %d-token"
    % (PRECHECK_MEDIAN_MS, PRECHECK_P99_MS, PRECHECK_TOKENS))
say("row; the untuned row above is this run's own measurement of the same quantity on a")
say("%d-token row, and RUN.txt section 8 recorded what the adapter costs as UNTESTED."
    % LENS[KEY])
say("THE TWO ARE NOT THE SAME CALL: %.2f ms here against precheck's %.2f ms differs by"
    % (QP[0], PRECHECK_MEDIAN_MS))
say("%+.2f ms, and this timed call contains one host-to-device copy of the %d ids that"
    % (QP[0] - PRECHECK_MEDIAN_MS, LENS[KEY]))
say("precheck's did not, over %d tokens fewer of prompt.  Neither difference is"
    % abs(LENS[KEY] - PRECHECK_TOKENS))
say("separated out here: this run times what a decision costs, copy included, because")
say("that is what decider.forward_three does on every decision of this step.")

# ===========================================================================
# 4  WHAT A NODE PAYS BESIDE THE PASS
# ===========================================================================
head("4  THE RENDER AND THE TOKENIZER CALL, NAMED AS NOT PART OF THE PASS")
say("A node builds the record, renders it and tokenizes it before any pass happens.  Both")
say("are timed here on the same host over the same row, %d times each, and NEITHER is"
    % TIMED)
say("part of the pass above.  The render timed is observe.observation plus render, i.e.")
say("the node's own work, without this step's no-node-name assertion on the result.")
say()
T_RENDER, T_TOKEN, T_CHAT = [], [], []
for _ in range(WARMUP):
    P.rendered(ROW)
    P.ids_of(TOK, TIMED_PROMPT)
for _ in range(TIMED):
    t0 = time.perf_counter()
    text = P.rendered(ROW)
    T_RENDER.append((time.perf_counter() - t0) * 1000.0)
    t0 = time.perf_counter()
    prompt = P.chat_text(TOK, text)
    T_CHAT.append((time.perf_counter() - t0) * 1000.0)
    t0 = time.perf_counter()
    P.ids_of(TOK, prompt)
    T_TOKEN.append((time.perf_counter() - t0) * 1000.0)
check_timed(T_RENDER, TIMED, "the renders")
check_timed(T_CHAT, TIMED, "the chat-template calls")
check_timed(T_TOKEN, TIMED, "the tokenizer calls")
QR, QC, QK = quantiles(T_RENDER), quantiles(T_CHAT), quantiles(T_TOKEN)
say("  part                       median      p90      p99      max")
say("  observation + render     %8.3f %8.3f %8.3f %8.3f" % QR)
say("  chat template            %8.3f %8.3f %8.3f %8.3f" % QC)
say("  tokenizer                %8.3f %8.3f %8.3f %8.3f" % QK)
TOTAL = check_cost_order(QT[0], QR[0] + QC[0], QK[0], "the three parts of a decision")
say()
say("one decision, as the sum of its parts: %.2f + %.3f + %.3f + %.3f = %.2f ms at the"
    % (QT[0], QR[0], QC[0], QK[0], TOTAL))
say("median, against the %.0f ms budget.  The sum is the sum and is not a measurement of a"
    % BUDGET_MS)
say("node's loop: a node also keeps a tracker, plans its fetches and runs a transport.")
say("THE TUNING'S OWN 72 ms PER HELD-OUT ROW IS ACCOUNTED FOR HERE: %.2f ms of pass plus"
    % QT[0])
say("%.3f ms of tokenizer plus %.3f ms of chat template, i.e. the adapter and not the"
    % (QK[0], QC[0]))
say("tokenizer is most of what the 43.02 ms on record does not cover.")
say("One pass per decision and not two: the send-time convention answers a send with the")
say("interval the pair already holds, so a decider that called the model at a send would")
say("pay the figure above twice.")

# ===========================================================================
# 5  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT
# ===========================================================================
head("5  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT")
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
FIRED.append(shows("seeds_tuning", lambda: D.check_seeds_tuning([1, 81], "mutated")))
FIRED.append(shows("port.check_seeds",
                   lambda: PORT.check_seeds([PORT.MAX_SEED + 1], "mutated")))
FIRED.append(shows("roles/range",
                   lambda: D.check_roles(dict(ROLES, tuning=[81]), WANT, "mutated")))
FIRED.append(shows("roles/measuring",
                   lambda: D.check_roles(dict(ROLES, selection=[121]), WANT, "mutated")))
FIRED.append(shows("median/a closer row exists", lambda: check_median(
    max(LENS.values()), list(LENS.values()), "mutated")))
FIRED.append(shows("timed/one pass short", lambda: check_timed(
    T_TUNED[:-1], TIMED, "mutated")))
FIRED.append(shows("timed/a zero pass", lambda: check_timed(
    [0.0] + T_TUNED[1:], TIMED, "mutated")))
FIRED.append(shows("cost order/one loop twice", lambda: check_cost_order(
    QT[0], QT[0], QK[0], "mutated")))
FIRED.append(shows("adapter_dtype",
                   lambda: D.check_adapter_dtype(["torch.bfloat16"], "mutated")))
FIRED.append(shows("unmerged", lambda: D.check_unmerged(2, [False, True], "mutated")))
FIRED.append(shows("prompt_len/lowered bound",
                   lambda: D.check_prompt_len(list(LENS.values()), 300, "mutated")))
FIRED.append(shows("triple", lambda: D.check_triple([1.0, 2.0, 3.0], [1.0, 2.0, 4.0],
                                                   "a", "mutated")))
FIRED.append(shows("digest_cover", lambda: D.check_digest_cover(
    CODE_PARTS[1:], ["latency.py"] + [n for n, _p in D.CODE_FILES], "mutated")))
FIRED.append(shows("no_node_name", lambda: D.check_no_node_name(
    TIMED_PROMPT.replace("p1", "h3", 1), "mutated")))
say()
say("The repeat check's own mutation is in part 2: one of the two stored logit triples is")
say("moved by one unit inside decider.repeat_lines.")
say()
say("WHAT HAS NO MUTATION, named and not counted: the card-free check is shown failing on")
say("a mutated nvidia-smi TEXT above and not by staging a second compute process on a")
say("shared card, which cannot be done without disturbing its owner.  The quantiles")
say("themselves are not a check.")
say()
if all(FIRED):
    say("all %d mutations of this script produced a failure" % len(FIRED))
else:
    say("FAILURE OF THIS SECTION: %d of %d mutations did not produce a failure"
        % (len(FIRED) - sum(FIRED), len(FIRED)))

# ===========================================================================
# 6  WHAT THIS RUN DOES NOT SHOW
# ===========================================================================
head("6  WHAT THIS RUN DOES NOT SHOW")
say("- It is NOT a node's hardware.  An RTX PRO 5000 Blackwell is not what a node runs,")
say("  and nothing here says what the pass would cost on one.")
say("- It does not include a node's own loop: no tracker update, no fetch planning, no")
say("  transport.  The sum in part 4 is the three parts timed here and nothing else.")
say("- Nothing about a merged adapter.  `merge_and_unload` changes the arithmetic and")
say("  would need its own agreement check against the unmerged read-out.")
say("- Nothing about contention.  This card held these two models and nothing else; the")
say("  evaluation runs K workers on it and eval.py's own per-worker figures are where")
say("  that cost is.")
say("- Nothing about whether 100 ms is the right budget.  The loop question is the next")
say("  ticket's, and 100 ms is printed beside these numbers with no verdict attached.")
if DRY:
    say("- Nothing about which candidate is better.  Candidate %r was named on the command"
        % CAND)
    say("  line, not chosen by a score.")
say()
say("wall clock    %.1f s (%.1f min) from the first line to this one"
    % (time.time() - T_START, (time.time() - T_START) / 60.0))
say("finished      %s" % time.strftime("%Y-%m-%dT%H:%M:%S%z"))
say("end of latency.py")
if not all(FIRED):
    raise SystemExit(1)
