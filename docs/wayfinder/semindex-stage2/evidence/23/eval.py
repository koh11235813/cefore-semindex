#!/usr/bin/env python3
"""The seven deciders of the comparison through the replay port, one csv row per run.

    cd $ROOT && env PYTHONDONTWRITEBYTECODE=1 STORE=$ROOT/store \
        PYTHONPATH=$ROOT/repo/src HF_HUB_OFFLINE=1 ./.venv/bin/python -B \
        repo/docs/wayfinder/semindex-stage2/evidence/23/eval.py $ROOT [--dry] [--k 4] \
        > $ROOT/eval.log 2> $ROOT/eval.err

    --dry      seeds 1 and 2, streams 0 and 1 (the TUNING half), writing eval_dry.csv
               and eval_dry.txt.  Everything else -- the reference gate, the repeat
               check, the memory assertion, the restart path -- is the real mode's.
    --k N      how many worker processes read out at once (default 4; the fallback the
               plan names if the gate fails is 1).

WHAT A DRY RUN'S SCORES MEAN: NOTHING ABOUT WHICH CANDIDATE IS BETTER.  The dry seeds
are tuning seeds that both candidates were fitted on, so a dry score is a score on
training data.  The real mode is seeds 81 to 100 and is started separately.

WHAT IT DOES, IN ORDER
  0. refuses to start if another compute process is on the shared card, and refuses if
     K copies of the weights would not fit beside a parent that holds none;
  1. prints and asserts the seed roles, and the seed column of every label row it opens;
  2. THE REFERENCE READ-OUT.  One child process, alone on the card, reads out one fixed
     set of 64 label rows of seeds 1 to 80 at batch 1 for each of the three model
     deciders and writes those logits to disk.  That child EXITS before any worker is
     spawned, and the parent holds no copy of the weights at any time: one copy is 10.12
     GB and four workers plus a fifth copy in the parent would not fit on the card;
  3. the four deciders that call no model -- the three fixed arms and the rule F -- in
     this process, with no card;
  4. the three model deciders, K workers at a time.  Every worker reads out the same 64
     rows first and the parent compares its logits for exact equality (`torch.equal`)
     against the reference, not only against the other workers.  Four workers that
     agreed with each other and differed from the single-process read-out would pass a
     worker-against-worker gate, and the score would then be read out at an arithmetic
     the agreement numbers and the latency figure are not;
  5. one csv row per (decider, seed, stream) with `score.py`'s own terms, written when
     each decider finishes, so a run that dies loses at most one decider's work;
  6. every check shown failing on a mutated input, and what the run does not show.

WHAT IT DOES NOT DO
  It never reads a seed above 100 (`port.check_seeds` is on every path, and `port.Engine`
  checks the run dict's own seed), and in `--dry` it never reads a seed above 80.  It
  chooses nothing: the choice between the two candidates is `ana.py`'s, from this csv.
"""
import os
import sys

# 2026-11-24: FIRST import, before torch / numpy / transformers.  `prefill.py` sets the
# thread caps, the offline HF variables, ROOT and the two sys.path entries at ITS import
# time, and `decider.py` holds the read-out, the tie rule and L itself, so that the score
# here, the agreement and the latency are taken by the same call.
import prefill as P   # noqa: E402
import decider as D   # noqa: E402

# `../22`'s own run dict, rule F and scorer, through the path `decider.py` resolved from
# its own location.  `label.run_dict` is what puts the union choice set, the choice jitter
# and `redecide` on a run and where the seed guard sits on this path, so every decider of
# the comparison runs on the same choice set; F is `../22`'s and not a copy.
import label as LBL   # noqa: E402
import score as SCORE  # noqa: E402

import json        # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time        # noqa: E402

ROOT = P.ROOT
PORT = D.PORT
R = PORT.R
ARMS = D.ARMS

MAIN_P = 0.2                     # the main cell: p 0.2, w0, the s6 outage shape
JF = LBL.JF                      # the choice jitter, 0.1, evidence/21's own
WEIGHT = LBL.WEIGHT              # 104.29595912307323 ms per poll, from ../22/weight.json
NREP = LBL.PR.NREP               # 40 replicate streams per seed
# `weight.py`'s convention and not `label.py`'s: the comparison of the candidates against
# the three arms is the weight's, and the arms were re-keyed onto these streams there.
# The same seed string serves both streams; `port.Engine._seed` puts a different kind
# prefix on each, so one string still gives two independent streams.
LAG_PREFIX = "pairlag:%s" % LBL.PR.STREAM_LABEL

DRY_SEEDS = (1, 2)               # the tuning half: a dry score is a score on training data
DRY_REPS = (0, 1)
REAL_SEEDS = tuple(range(81, 101))
REAL_REPS = tuple(range(NREP))

GATE_N = 64                      # label rows of seeds 1-80 the reference gate reads out
GATE_DEADLINE_S = 3600           # a worker that never reaches the gate is a stop, not a hang
WORKER_GB = 10.12                # one copy of the weights on the card (precheck.txt part 2)
PARENT_GB = 0.5                  # the parent's cuda context, which holds no weights

ADAPTERS = os.path.join(ROOT, "adapters")
# The three model deciders, in the plan's own order, with the adapter each one attaches.
MODEL_DECIDERS = (("untuned", None),
                  ("a", os.path.join(ADAPTERS, "a")),
                  ("b", os.path.join(ADAPTERS, "b")))
# The four that call no model.  The names are `parity.py`'s, so that a line of parity.txt
# and a row of this csv name the same decider.
ARM_NAMES = tuple("const%.3f" % arm for arm in ARMS)
ORDER = ARM_NAMES + ("F",) + tuple(n for n, _ in MODEL_DECIDERS)

# 2026-11-24: `k` is the number of worker processes that were reading out while this
# row was produced -- K for a model decider, 1 for the four that run in the parent and
# make no pass.  Added because a run started at --k 4, killed, and restarted at --k 1
# (which is the plan's own fallback) produced one csv whose deciders were scored at two
# concurrency levels with nothing in the csv or the part files recording it.  The gate
# makes the arithmetic equal by construction, so no number was wrong; the record was
# incomplete, and a reader could not tell.
COLUMNS = ("decider", "seed", "stream", "n4", "c4", "lat_ms", "censored_ms", "polls",
           "delay_ms", "score_ms", "score_dropped_ms", "weight_ms_per_poll",
           "reads", "sends", "ties", "n_short", "n_mid", "n_long",
           "ord_poll", "ord_resched", "k", "script_sha", "model_sha")

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
def check_fits(k, free_gb, where):
    """K copies of the weights fit on the card beside a parent that holds none.

    A card-size constraint and not a style: one copy is 10.12 GB and the card has about
    46.7 GB free, so four workers fit at 40.5 GB and a parent holding a fifth copy beside
    them would need 50.6 GB and fail.  K = 8 is refused here on the same arithmetic that
    `precheck.txt` part 2 refused it on, rather than being discovered by an out-of-memory
    error half an hour into a run.
    """
    want = k * WORKER_GB + PARENT_GB
    if want > free_gb:
        raise Fail("%s: %d worker(s) need %.2f GB beside the parent and the card has "
                   "%.2f GB free" % (where, k, want, free_gb))
    return want


def check_choices(run, where):
    """The run dict is the one every decider of the comparison is scored on.

    The three things that make the comparison a comparison: the union choice set (F
    leaves the three arms as soon as it lengthens, so a run that scored F on the three
    arms alone would raise, and a decider scored on a narrower set of options than
    another is not the same experiment), the choice jitter, and the re-decision at a
    resolution without which only the send-time value would ever be asked for.
    """
    want = list(PORT.union_choices())
    if list(run["choices"]) != want:
        raise Fail("%s: the choice set is %r, not the union set %r"
                   % (where, list(run["choices"]), want))
    if run["choice_jitter"] != JF:
        raise Fail("%s: choice_jitter %r, not %r" % (where, run["choice_jitter"], JF))
    if not run["redecide"]:
        raise Fail("%s: redecide is off, so no resolution-time decision is asked for"
                   % where)
    return want


def check_counters(reads, sends, ord_resched, ord_poll, where):
    """One forward pass per resolution-time decision, and none at a send.

    The counted form of the send-time convention, which the latency figure of this step is
    read against: a decision costs ONE pass, so a decider that called the model at a send
    would pay twice and `reads` would exceed the engine's own resolution count.  Both
    counters are the engine's and the decider's, and they are compared rather than
    described because the engine counts calls and the decider counts passes.
    """
    if reads != ord_resched:
        raise Fail("%s: %d forward pass(es) against %d resolution-time decision(s)"
                   % (where, reads, ord_resched))
    if sends != ord_poll:
        raise Fail("%s: %d send-time decision(s) against the engine's %d"
                   % (where, sends, ord_poll))
    return reads


def check_gate(ref, got, where):
    """One worker's 64 x 3 logits are bit-identical to the single-process reference.

    `torch.equal` and not a tolerance: the question is whether K processes on one card
    read out the same bits as one process alone on it, and a tolerance answers a different
    question.  `RUN.txt` section 13 names the gap in two halves -- the K = 2 and K = 4
    workers' logits were never compared with each other OR with batch 1 -- and a
    worker-against-worker gate closes only the first half.
    """
    import torch
    if tuple(ref.shape) != tuple(got.shape):
        raise Fail("%s: shape %s against the reference's %s"
                   % (where, tuple(got.shape), tuple(ref.shape)))
    if not torch.equal(ref, got):
        bad = (ref != got).nonzero()
        i, j = int(bad[0][0]), int(bad[0][1])
        raise Fail("%s: %d of %d values differ, first at row %d word %d: %r against %r"
                   % (where, int(bad.shape[0]), ref.numel(), i, j,
                      float(got[i][j]), float(ref[i][j])))
    return int(ref.numel())


def check_rollin(live, row, k, where):
    """The re-run roll-in drew the arm the label row records at this decision.

    The anchor of the render seam check: without it, a rendering comparison could be
    between the right row and a simulation of the same seed under another stream label,
    which draws other arms and visits other states, and the comparison would then be
    saying nothing while printing "equal".
    """
    if live != row:
        raise Fail("%s: at decision k %d the re-run roll-in drew %r and the label row "
                   "records %r, so this is not the run that row came from"
                   % (where, k, live, row))
    return live


def check_sidecar(want, got, where):
    """A part file on disk was produced by this script and this adapter.

    What makes a restart safe rather than merely possible.  Rows written under another
    script digest or another adapter digest are rows of another experiment, and a csv that
    mixed them would average two deciders under one name while printing a plausible
    number.  A difference refuses the restart instead of resuming it.
    """
    bad = sorted(k for k in want if got.get(k) != want[k])
    if bad:
        raise Fail("%s: %s" % (where, ", ".join(
            "%s %r against %r" % (k, got.get(k), want[k]) for k in bad)))
    return len(want)


def check_complete(keys, want, where):
    """Exactly one row per (seed, stream) of this decider, no row twice and none missing.

    A decider whose worker died after half its runs leaves a part file that looks like a
    smaller experiment, and a mean over it would be a mean over whichever runs happened to
    finish.  Counted against the product the mode declares, not against what was found.
    """
    got, wanted = sorted(keys), sorted(want)
    if len(got) != len(set(got)):
        twice = sorted({k for k in got if got.count(k) > 1})
        raise Fail("%s: %d row(s) appear twice, e.g. %s" % (where, len(twice), twice[:3]))
    if got != wanted:
        missing = sorted(set(wanted) - set(got))
        extra = sorted(set(got) - set(wanted))
        raise Fail("%s: %d missing (e.g. %s) and %d unexpected (e.g. %s) of %d runs"
                   % (where, len(missing), missing[:3], len(extra), extra[:3],
                      len(wanted)))
    return len(got)


# ---------------------------------------------------------------------------
# the digests that say which code and which adapter produced a row
# ---------------------------------------------------------------------------
CODE_PARTS = D.code_digests(os.path.abspath(__file__))
CODE_WANT = ["eval.py"] + [n for n, _p in D.CODE_FILES]
# 2026-11-24: the adapter's CONFIG is digested beside its weights.  `r` and `lora_alpha`
# scale the LoRA output and therefore every logit, so an adapter whose config changed with
# its weights untouched is a different read-out that an earlier digest over
# `adapter_model.safetensors` alone would have resumed without a word.  The two files are
# byte-identical between candidate a and candidate b today -- same recipe, two row sets --
# so the config alone cannot tell them apart and the weights alone cannot tell a re-scaled
# adapter from the one that was measured; the digest is over both, in a fixed order.
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def script_sha():
    """One digest over every file that decides what a row of this csv is.

    This script, the read-out and L (`decider.py`), the prompt (`prefill.py`), the
    renderer (`../22/observe.py`), the engine wrapper and the seed guard
    (`../22/port.py`), the run dict and F (`../22/label.py`), the scorer
    (`../22/score.py`) and the engine itself (`../13/replay.py`).  A change to any of them
    makes the rows of a half-finished run incomparable with the rows of the restart, which
    is what `check_sidecar` refuses on.

    2026-11-24: the last five were added.  The digest covered this script, `decider.py`
    and `prefill.py` only, so a restart after an edit to the engine, the renderer or the
    scorer -- the files that decide what a score IS -- resumed and wrote one csv out of
    rows produced by two experiments under one decider name.  What the digest still does
    not cover is printed beside it in part 2.
    """
    D.check_digest_cover(CODE_PARTS, CODE_WANT, "eval.py: the script digest")
    return D.combined_sha(CODE_PARTS)


def adapter_parts(adapter_dir):
    """`[(filename, sha256)]` of the two adapter files a read-out's arithmetic depends on."""
    return [(n, D.sha_of(os.path.join(adapter_dir, n))) for n in ADAPTER_FILES]


def model_sha(adapter_dir):
    """The digest of the adapter a model decider attaches, or "none" for the base.

    The base weights are pinned by revision and their digest is in `RUN.txt` section 2;
    what varies between the three model deciders is the adapter, so that is what a row
    carries -- its weights AND its config, for the reason at `ADAPTER_FILES`.
    """
    if adapter_dir is None:
        return "none"
    return D.combined_sha(adapter_parts(adapter_dir))


# ---------------------------------------------------------------------------
# the rows the gate reads out, and the one row the repeat check reads twice
# ---------------------------------------------------------------------------
def tuning_half_rows():
    """Every labelled label row of seeds 1 to 80, in file order.

    The row list is cut at seed 80 BEFORE anything is rendered or read out, in both modes,
    so the gate's own rows can never be selection-half rows whichever mode is running.
    """
    import gzip
    with gzip.open(os.path.join(ROOT, "ev22", "labels_p0.2.jsonl.gz"), "rt") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    PORT.check_seeds([r["seed"] for r in rows], "eval.py: the label file's seed column")
    half = [r for r in rows if int(r["seed"]) <= D.MAX_TUNE_SEED and r["label"] is not None]
    D.check_seeds_tuning([r["seed"] for r in half], "eval.py: the rows the gate reads")
    return rows, half


def gate_rows(half):
    """The fixed 64 rows the reference and every worker read out, by a stride.

    A stride over the labelled rows of seeds 1 to 80 rather than a slice: a slice would be
    one or two seeds' worth of decisions, and the question the gate asks is about the
    arithmetic over a spread of prompt lengths.  Fixed by the data and not by a list typed
    here, so the same 64 rows are read in both modes and on a restart.
    """
    stride = max(1, len(half) // GATE_N)
    return half[::stride][:GATE_N]


def repeat_row(half):
    """The one row the repeat check reads twice: seed 71 k 72, the tuning run's own.

    The same row `tune.py` made its repeat check on, so this step's read-out can be held
    against the logit triples the tuning archived for both adapters.  If the row is not in
    the list (it cannot fail to be, but the fallback is written rather than assumed) the
    median-length row of the list is taken.
    """
    for r in half:
        if int(r["seed"]) == 71 and int(r["k"]) == 72:
            return r
    return half[len(half) // 2]


# ---------------------------------------------------------------------------
# the child processes: the reference read-out, and one worker
# ---------------------------------------------------------------------------
def load_for(adapter_dir, where):
    """The base in bf16, with one adapter attached and unmerged when there is one."""
    base = D.load_base()
    if adapter_dir is None:
        return base
    return D.attach_adapter(base, adapter_dir, where)


def stored_reader(first, second):
    """A read-out that hands back two triples already measured, in order.

    `decider.repeat_lines` makes the comparison, prints it and shows its own mutation
    firing, and it takes a callable rather than a model so that the three logits can come
    from anywhere.  The two passes of a worker's repeat check happen in the child; this is
    how the parent runs the SAME comparator over what the child measured instead of
    writing a second one.
    """
    pair = iter((tuple(first), tuple(second)))
    return lambda _prompt: next(pair)


def read_all(model, tok, cand_ids, prompts):
    """(the (n, 3) float32 logits of `prompts`, the mean seconds per read-out)."""
    import torch
    out = torch.empty(len(prompts), 3, dtype=torch.float32)
    t0 = time.time()
    for i, prompt in enumerate(prompts):
        three = D.readout(model, tok, cand_ids, prompt)
        out[i] = torch.tensor(three, dtype=torch.float32)
    return out, (time.time() - t0) / len(prompts)


def child_reference(tmp):
    """THE REFERENCE READ-OUT: one process alone on the card, for all three models.

    All three references are taken here, in one child that then exits, so that the
    reference of every model decider exists before any worker is spawned and before any
    score of this run is computed.  The parent holds no copy of the weights at any time,
    which is a card-size constraint: four workers are 40.5 GB of the 46.7 GB free and a
    fifth copy in the parent would not fit.
    """
    import torch
    prompts = json.load(open(os.path.join(tmp, "gate_prompts.json"), encoding="utf-8"))
    with open(os.path.join(tmp, "repeat_prompt.txt"), encoding="utf-8") as fh:
        repeat_prompt = fh.read()
    tok = D.load_tokenizer()
    cand_ids_list = D.candidate_ids(tok, prompts[0])
    cand_ids = torch.tensor(cand_ids_list, device=D.DEV)
    print("the reference read-out, one process alone on the card")
    print("  candidate ids %s for %s, each one token at this position"
          % (cand_ids_list, list(P.CANDIDATES)))
    info = {"cand_ids": cand_ids_list, "per": {}}
    for name, adapter in MODEL_DECIDERS:
        t0 = time.time()
        model = load_for(adapter, "the reference read-out of %s" % name)
        load_s = time.time() - t0
        logits, per_s = read_all(model, tok, cand_ids, prompts)
        torch.save(logits, os.path.join(tmp, "ref_%s.pt" % name))
        first = D.readout(model, tok, cand_ids, repeat_prompt)
        second = D.readout(model, tok, cand_ids, repeat_prompt)
        arm, tied = D.pick(first)
        D.check_repeat(first, second, arm, D.pick(second)[0],
                       "the reference repeat check of %s" % name)
        info["per"][name] = {"load_s": load_s, "readout_ms": per_s * 1000.0,
                             "first": list(first), "second": list(second),
                             "arm": arm, "tied": tied}
        print("  %-8s loaded in %4.1f s, %d rows at %.1f ms per read-out, repeat row "
              "%s -> %r" % (name, load_s, len(prompts), per_s * 1000.0, list(first),
                            P.CANDIDATES[arm]))
        del model
        torch.cuda.empty_cache()
    json.dump(info, open(os.path.join(tmp, "ref.json"), "w", encoding="utf-8"))
    print("  the reference child now exits, before any worker is spawned")


def child_worker(tmp, index):
    """One of K processes that reads out at batch 1 and drives its share of the runs.

    The gate first and the simulations only after the parent says every worker's 64 rows
    matched the reference: a worker that scored before the comparison would have produced
    rows at an arithmetic the run then refuses.
    """
    import torch
    job = json.load(open(os.path.join(tmp, "job_%d.json" % index), encoding="utf-8"))
    name, adapter = job["decider"], job["adapter"]
    prompts = json.load(open(os.path.join(tmp, "gate_prompts.json"), encoding="utf-8"))
    tok = D.load_tokenizer()
    cand_ids = torch.tensor(job["cand_ids"], device=D.DEV)
    t0 = time.time()
    model = load_for(adapter, "worker %d of %s" % (index, name))
    load_s = time.time() - t0
    logits, gate_ms = read_all(model, tok, cand_ids, prompts)
    torch.save(logits, os.path.join(tmp, "gate_%s_%d.pt" % (name, index)))
    json.dump({"index": index, "load_s": load_s, "gate_ms": gate_ms * 1000.0,
               "pid": os.getpid()},
              open(os.path.join(tmp, "gate_%s_%d.json" % (name, index)), "w",
                   encoding="utf-8"))
    print("worker %d of %s: loaded in %.1f s, 64 gate rows at %.1f ms, pid %d"
          % (index, name, load_s, gate_ms * 1000.0, os.getpid()))

    go = os.path.join(tmp, "go_%s" % name)
    stop = os.path.join(tmp, "stop_%s" % name)
    deadline = time.time() + GATE_DEADLINE_S
    while not os.path.exists(go):
        if os.path.exists(stop):
            print("worker %d of %s: the gate refused this run; nothing was scored"
                  % (index, name))
            return
        if time.time() > deadline:
            raise SystemExit("worker %d of %s: no go file after %d s"
                             % (index, name, GATE_DEADLINE_S))
        time.sleep(0.05)

    acc = {"t": 0.0, "n": 0}

    def read_three(prompt):
        """The shared read-out, timed, so the real run's wall time can be re-estimated."""
        t_call = time.perf_counter()
        three = D.readout(model, tok, cand_ids, prompt)
        acc["t"] += time.perf_counter() - t_call
        acc["n"] += 1
        return three

    rows, t_start = [], time.time()
    for seed, rep in job["runs"]:
        rows.append(one_run(name, seed, rep, lambda: D.L(read_three, tok), job["k"]))
    json.dump({"index": index, "rows": rows, "t_start": t_start, "t_end": time.time(),
               "readout_ms": 1000.0 * acc["t"] / acc["n"] if acc["n"] else 0.0,
               "passes": acc["n"]},
              open(os.path.join(tmp, "res_%s_%d.json" % (name, index)), "w",
                   encoding="utf-8"))
    print("worker %d of %s: %d run(s), %d forward pass(es) at %.1f ms per read-out"
          % (index, name, len(job["runs"]), acc["n"],
             1000.0 * acc["t"] / acc["n"] if acc["n"] else 0.0))


# ---------------------------------------------------------------------------
# one simulation
# ---------------------------------------------------------------------------
class FirstCallEmpty:
    """A decider wrapper that asserts the per-pair memory is empty at the FIRST decision.

    F and L are both built fresh for every simulation, because each keeps one interval per
    ordered pair and one instance reused across two runs would make the second run depend
    on the first.  The MODEL is not rebuilt -- reloading 10 GB of weights 800 times is not
    what "fresh" means -- so the thing that has to be shown empty is this memory, and it is
    shown empty at the instant the claim is about: the simulation's first decision, not
    the moment the object was made.
    """

    def __init__(self, inner, where, keeps):
        """`inner` is the decider; `where` names the run; `keeps` is the caller's claim.

        `keeps` is passed in rather than guessed, because the thing that has to be
        refused is a decider that keeps per-pair state the wrapper cannot see, and a
        wrapper that guesses cannot tell that case from a constant arm.
        """
        self.inner = inner
        self.where = where
        self.keeps = keeps
        self.checked = False

    def memory(self):
        """The per-pair memory of a decider that keeps one, else None.

        `mem` is L's and `ival` is F's.  2026-11-24: a decider that `keeps_memory` calls
        stateful and that carries neither name is REFUSED here rather than passed, because
        the old form returned None and then checked nothing -- so a rename of `mem`, or a
        fourth decider keeping its intervals under another name, would have removed the
        assertion the plan makes load-bearing with nothing in the output saying it was
        gone.  A constant arm declares `keeps` false and is checked for nothing, which is
        correct: it has no state to carry from one run into the next.
        """
        for attr in ("mem", "ival"):
            if hasattr(self.inner, attr):
                return getattr(self.inner, attr)
        if self.keeps:
            raise Fail("%s: this decider keeps per-pair state and carries neither `mem` "
                       "nor `ival`, so the memory assertion of every simulation's first "
                       "decision cannot be made on it" % self.where)
        return None

    def __call__(self, obs):
        """Check once, then hand the decision straight to the decider."""
        if not self.checked:
            self.checked = True
            mem = self.memory()
            if mem is not None:
                D.check_memory_empty(mem, self.where)
        return self.inner(obs)


def keeps_memory(decider):
    """Whether this decider carries state from one decision to the next.

    L and F each keep one interval per ordered pair; `port.const_decider` returns a
    closure over one float and keeps nothing.  Declared by type here, in one place, so
    that the wrapper above refuses a stateful decider it cannot read instead of passing
    it.  A closure over a dict would still pass this, which is why the test is the
    decider's type and the reason is written down rather than inferred.
    """
    return isinstance(decider, (D.L, LBL.F))


def one_run(name, seed, rep, make_decider, k):
    """One (seed, stream) of one decider: the csv row as a dict.

    `label.run_dict` and not a hand-built dict: it is where the union choice set, the
    choice jitter and `redecide` come from, and where the seed guard sits on this path.
    """
    run = LBL.run_dict(MAIN_P, seed)
    check_choices(run, "the run dict of %s seed %d stream %d" % (name, seed, rep))
    inner = make_decider()
    where = "%s seed %d stream %d" % (name, seed, rep)
    run = dict(run, decider=FirstCallEmpty(inner, where, keeps_memory(inner)))
    base = R.rep_seed(LAG_PREFIX, MAIN_P, seed, rep)
    events, engine = PORT.sim(run, None, base, base)
    t = SCORE.terms(events, run["t0"])
    n4, c4, lat, censored, polls = t
    ordp = engine.decider_ord["state_poll"]
    ordr = engine.decider_ord["state_poll_reschedule"]
    row = {"decider": name, "seed": seed, "stream": rep, "n4": n4, "c4": c4,
           "lat_ms": lat, "censored_ms": censored, "polls": polls,
           "delay_ms": SCORE.delay(t), "score_ms": SCORE.delay(t) + WEIGHT * polls,
           "score_dropped_ms": SCORE.delay_dropped(t) + WEIGHT * polls,
           "weight_ms_per_poll": WEIGHT, "ord_poll": ordp, "ord_resched": ordr,
           "k": k,
           "reads": "", "sends": "", "ties": "", "n_short": "", "n_mid": "",
           "n_long": ""}
    if isinstance(inner, D.L):
        check_counters(inner.reads, inner.sends, ordr, ordp, where)
        row.update(reads=inner.reads, sends=inner.sends, ties=inner.ties,
                   n_short=inner.arms[ARMS[0]], n_mid=inner.arms[ARMS[1]],
                   n_long=inner.arms[ARMS[2]])
    return row


# ---------------------------------------------------------------------------
# the csv
# ---------------------------------------------------------------------------
def csv_line(row, sha, msha):
    """One csv line, floats through `repr` so a reader can recompute the score exactly.

    `ana.py` recomputes `delay + weight x polls` from these columns and requires it to
    equal the score column bit for bit, which only holds if the numbers are written at
    full precision.
    """
    out = dict(row, script_sha=sha, model_sha=msha)
    return ",".join(repr(out[c]) if isinstance(out[c], float) else str(out[c])
                    for c in COLUMNS)


def write_part(path, rows, sha, msha):
    """One decider's rows, written when that decider finishes and not before."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(COLUMNS) + "\n")
        for row in sorted(rows, key=lambda r: (r["seed"], r["stream"])):
            fh.write(csv_line(row, sha, msha) + "\n")


def read_part(path):
    """The (seed, stream) keys and the digest columns of a part file already on disk."""
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    cols = lines[0].split(",")
    rows = [dict(zip(cols, ln.split(","))) for ln in lines[1:]]
    keys = [(int(r["seed"]), int(r["stream"])) for r in rows]
    shas = {(r["script_sha"], r["model_sha"], r.get("k", "")) for r in rows}
    return keys, shas, lines[1:]


# ===========================================================================
# the child modes, before the parent's own report starts
# ===========================================================================
ARGV = sys.argv[1:]
if "--reference" in ARGV:
    child_reference(ARGV[ARGV.index("--reference") + 1])
    raise SystemExit(0)
if "--worker" in ARGV:
    _at = ARGV.index("--worker")
    child_worker(ARGV[_at + 1], int(ARGV[_at + 2]))
    raise SystemExit(0)

DRY = "--dry" in ARGV
K = int(ARGV[ARGV.index("--k") + 1]) if "--k" in ARGV else 4
MODE = "dry" if DRY else "real"
SEEDS = list(DRY_SEEDS if DRY else REAL_SEEDS)
REPS = list(DRY_REPS if DRY else REAL_REPS)
RUNS = [(s, r) for s in SEEDS for r in REPS]
TMP = os.path.join(ROOT, "eval-%s-tmp" % MODE)
PARTS = os.path.join(ROOT, "eval-%s-parts" % MODE)
CSV = os.path.join(ROOT, "eval_dry.csv" if DRY else "eval.csv")

# ===========================================================================
# 0  THE CARD, THE MODE AND WHAT THIS RUN IS
# ===========================================================================
T_START = time.time()
head("0  THE CARD IS FREE, AND WHAT THIS RUN IS")
if DRY:
    say("A DRY RUN.  The seeds are 1 and 2, which are TUNING seeds that both candidates")
    say("were fitted on, so every score below is a score on training data and says NOTHING")
    say("about which candidate is better.  The choice between a and b is the mean score on")
    say("seeds 81 to 100, which the real mode produces and this run does not.")
    say("What IS exercised here is every mechanism of the real mode: the reference gate,")
    say("the repeat check, the memory assertion, the counters and the restart path.")
else:
    say("The seven deciders of the comparison on seeds 81 to 100, 40 streams each.  This")
    say("run computes the scores and chooses nothing: the choice between the two")
    say("candidates is ana.py's, from the csv this run writes.")
say()
_apps = D.nvidia_compute_apps()
say("nvidia-smi compute apps: %s" % (_apps.strip() if _apps.strip() else "(none)"))
D.check_card_free(_apps, "eval.py at start")
say("OK: no other compute process on the GPU")

import numpy   # noqa: E402
import torch   # noqa: E402

torch.set_num_threads(8)

import transformers  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("FAIL: torch does not see a CUDA device")

say()
say("started         %s" % time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(T_START)))
say("torch %s (cuda %s), transformers %s, numpy %s, python %s"
    % (torch.__version__, torch.version.cuda, transformers.__version__,
       numpy.__version__, sys.version.split()[0]))
_FREE, _TOTAL = torch.cuda.mem_get_info()
say("gpu %s, %.2f GB free of %.2f GB"
    % (torch.cuda.get_device_name(0), _FREE / 2**30, _TOTAL / 2**30))
say("mode            %s: %d seed(s) x %d stream(s) = %d runs per decider"
    % (MODE, len(SEEDS), len(REPS), len(RUNS)))
say("workers         K = %d, each with its own copy of the weights; the parent holds none"
    % K)
check_fits(K, _FREE / 2**30, "eval.py at start")
say("                %d x %.2f GB of weights plus the parent's %.2f GB context fit in the"
    % (K, WORKER_GB, PARENT_GB))
say("                %.2f GB free; K = 8 would need %.2f GB and is refused by the same"
    % (_FREE / 2**30, 8 * WORKER_GB + PARENT_GB))
say("                arithmetic (shown in part 5)")
say("weight          %r ms per poll, from ../22/weight.json" % WEIGHT)
say("streams         R.rep_seed(%r, %s, seed, stream), as BOTH pair_lag_seed and"
    % (LAG_PREFIX, MAIN_P))
say("                pair_jitter_seed, sim_rng None (weight.py's step-4 convention)")
say("choice set      %s" % (list(PORT.union_choices()),))
say("read-out        bf16 base, float32 adapter, batch 1, the three candidate logits at")
say("                the last real position, first maximum of %s wins a tie"
    % list(P.CANDIDATES))

# ===========================================================================
# 1  THE SEEDS, THE ROLES AND THE ROWS THE GATE READS
# ===========================================================================
head("1  THE SEEDS, THE ROLES, AND THE ROWS THE GATE READS OUT")
ROLES = {"tuning": list(range(1, 71)), "held out": list(range(71, 81)),
         "selection": list(range(81, 101)), "simulated here": SEEDS}
WANT = {"tuning": (1, 70), "held out": (71, 80), "selection": (81, 100),
        "simulated here": (1, D.MAX_TUNE_SEED) if DRY else (81, 100)}
UNION = D.check_roles(ROLES, WANT, "eval.py: the seed roles")
for role in ("tuning", "held out", "selection", "simulated here"):
    seeds = ROLES[role]
    say("  %-15s %2d seeds, %d to %d" % (role, len(seeds), min(seeds), max(seeds)))
say("  the union of the four meets range(121, 161) in %d seeds"
    % len(set(UNION) & set(range(121, 161))))
PORT.check_seeds(UNION, "eval.py: the four seed roles")
if DRY:
    D.check_seeds_tuning(SEEDS, "eval.py: the simulated seeds of a dry run")
    say("  check_seeds_tuning (>80 refused) passed the simulated seeds of this dry run,")
    say("  and port.check_seeds (>100 refused) passed all four roles; both are shown")
    say("  refusing in part 5")
else:
    say("  port.check_seeds (>100 refused) passed all four roles, and it is shown refusing")
    say("  in part 5.  check_seeds_tuning is NOT applied to the simulated seeds here: this")
    say("  is the selection half, which is what the real mode exists to read")
say("  every simulation below goes through port.sim on a run dict from label.run_dict,")
say("  which carries its own seed, so port.Engine's constructor guard is on the one path")

ALL_ROWS, HALF = tuning_half_rows()
_seeds_in = sorted({int(r["seed"]) for r in ALL_ROWS})
say()
say("label file      %s" % os.path.join(ROOT, "ev22", "labels_p0.2.jsonl.gz"))
say("rows            %d, seed column min %d max %d, %d distinct"
    % (len(ALL_ROWS), _seeds_in[0], _seeds_in[-1], len(_seeds_in)))
say("labelled rows of seeds 1-%d: %d, the list the gate's rows are taken from, cut at"
    % (D.MAX_TUNE_SEED, len(HALF)))
say("seed %d before anything is rendered in either mode" % D.MAX_TUNE_SEED)
GATE = gate_rows(HALF)
REPEAT = repeat_row(HALF)
_gate_seeds = sorted({int(r["seed"]) for r in GATE})
say("gate rows       %d, by a stride of %d over that list, seeds %d to %d (%d distinct)"
    % (len(GATE), max(1, len(HALF) // GATE_N), _gate_seeds[0], _gate_seeds[-1],
       len(_gate_seeds)))
say("repeat row      seed %d k %d, the row tune.py read twice, so the triples below can be"
    % (REPEAT["seed"], REPEAT["k"]))
say("                held against the ones tune.txt archived for each adapter")

TOK = D.load_tokenizer()
say(P.prepare_padding(TOK))
_t0 = time.time()
GATE_TEXTS = [D.text_of(r["state"], r["resolved"], r["history"], r["ts_ms"],
                        r["neighbor"], "the gate rows") for r in GATE]
RENDER_MS = 1000.0 * (time.time() - _t0) / len(GATE_TEXTS)
GATE_PROMPTS = [P.chat_text(TOK, t) for t in GATE_TEXTS]
REPEAT_PROMPT = P.chat_text(TOK, D.text_of(
    REPEAT["state"], REPEAT["resolved"], REPEAT["history"], REPEAT["ts_ms"],
    REPEAT["neighbor"], "the repeat row"))
GATE_LENS = [len(P.ids_of(TOK, p)) for p in GATE_PROMPTS]
D.check_prompt_len(GATE_LENS, 1024, "the gate prompts")
say("gate prompts    %d to %d tokens, median %d, bound 1024, nothing truncated; each one"
    % (min(GATE_LENS), max(GATE_LENS), int(statistics.median(GATE_LENS))))
say("                rendered through decider.text_of, so the no-node-name assertion ran")
say("                on every one of them.  Rendering one costs %.1f ms on the cpu, which"
    % RENDER_MS)
say("                is part of a decision and not part of a forward pass")

# ---------------------------------------------------------------------------
# THE SEAM BETWEEN THE TUNING AND THE REPLAY, closed here and nowhere else
# ---------------------------------------------------------------------------
say()
say("THE SEAM BETWEEN THE TUNING'S INPUT AND THIS RUN'S INPUT")
say("The model was tuned on label rows read back out of a gzip-JSON file, and it is read")
say("out below on the LIVE engine's own observation.  Those are two roads to one string,")
say("and if they ever part the tuning and the scoring are of two different inputs while")
say("both print plausible numbers.  decider.py's card-free self-check compares two CALL")
say("SITES over one already-decoded row, which leaves the JSON round trip on both sides")
say("of it and therefore unexamined.  So the comparison is made here, against the live")
say("object: one labelling roll-in of a tuning seed is re-run (no model, no card), its")
say("own captured observation of one decision is rendered, and that is held against the")
say("rendering of the gzip row of the same (seed, k).")
# 2026-11-24: DRY_SEEDS[0] and not SEEDS[0], so the seam check reads the same tuning seed
# in BOTH modes.  In a real run SEEDS[0] is 81, and taking it here would render one label
# row of the selection half, and re-run that seed's labelling roll-in, before the run's
# first score -- which is what spec.md's sentence that the one early read of the selection
# half is the class-share baseline is measured against.  Without this line that sentence
# becomes false in the real mode.  The seam is a gzip-JSON row against a live engine
# object and is not a question about which seed, so a tuning seed loses nothing here.
_SEAM_SEED = DRY_SEEDS[0]
_seam_run = LBL.run_dict(MAIN_P, _SEAM_SEED)
_seam_label = LBL.stream_label_of(MAIN_P, _SEAM_SEED)
_t0 = time.time()
_ev, _eng, _roll = LBL.simulate(_seam_run, _seam_label, _seam_label, None, None, None)
_SEAM_K = len(_roll.capture) // 2
_live = _roll.capture[_SEAM_K]
_SEAM_ROW = next(r for r in ALL_ROWS
                 if int(r["seed"]) == _SEAM_SEED and int(r["k"]) == _SEAM_K)
# The anchor, and it has a failing input of its own: a roll-in re-run under another
# stream label draws other arms, so this equality is what says the simulation above IS
# the run the row came from and not merely a run of the same seed.
check_rollin(_live["rollin_arm"], _SEAM_ROW["rollin_arm"], _SEAM_K, "the seam check")
_live_text = D.text_of(_live["state"], _live["resolved"], _live["history"],
                       _live["ts_ms"], _live["neighbor"], "the live observation")
_row_text = D.text_of(_SEAM_ROW["state"], _SEAM_ROW["resolved"], _SEAM_ROW["history"],
                      _SEAM_ROW["ts_ms"], _SEAM_ROW["neighbor"], "the gzip row")
D.check_render_roundtrip(_live_text, _row_text, "the seam of seed %d k %d"
                         % (_SEAM_SEED, _SEAM_K))
say("  the seed read here is %d, a TUNING seed, in both modes, and the reason is which"
    % _SEAM_SEED)
say("  seed it is NOT: this mode's own first simulated seed is %d, and in the real mode"
    % SEEDS[0])
say("  that is a selection-half seed, whose label row and whose roll-in would then be")
say("  read before this run's first score.  The seam is a JSON row against a live object")
say("  and not a question about which seed, so it is pinned to the tuning half.")
say("  the roll-in of seed %d re-run in %.1f s, %d captured decisions, no model"
    % (_SEAM_SEED, time.time() - _t0, len(_roll.capture)))
say("  decision k %d: the roll-in arm the live run drew is %.3f s and the row's is %.3f s"
    % (_SEAM_K, _live["rollin_arm"], _SEAM_ROW["rollin_arm"]))
say("  the live observation renders to %d characters, the gzip row of the same (seed, k)"
    % len(_live_text))
say("  to %d, and the two strings are equal, so the JSON round trip -- a tuple becoming"
    % len(_row_text))
say("  a list, a non-string dict key becoming a string -- moves no character of the")
say("  model input.  Shown failing on the row of k + 1 in part 5.")

# ===========================================================================
# 2  THE PARTS ALREADY ON DISK
# ===========================================================================
head("2  WHAT IS ALREADY ON DISK, AND WHAT THIS RUN STILL HAS TO DO")
say("Each decider's rows are written when that decider finishes, so a run that dies loses")
say("at most one decider's work.  A part file is resumed only if it was produced by this")
say("script and this adapter; otherwise the run refuses rather than mixing two")
say("experiments under one decider name.")
say()
os.makedirs(PARTS, exist_ok=True)
SHA = script_sha()
say("script digest   %s" % SHA)
say("                one sha256 over the files below, in this order.  Each part is")
say("                printed too, because a combined digest tells a reader that")
say("                something moved and not what:")
for _n, _s in CODE_PARTS:
    say("  %-18s %s" % (_n, _s))
say("WHAT NO DIGEST OF THIS RUN COVERS, so that it is not read as covering everything:")
for _what in D.CODE_UNCOVERED:
    say("  - %s" % _what)
TODO, DONE = [], []
for NAME in ORDER:
    adapter = dict(MODEL_DECIDERS).get(NAME)
    is_model = NAME in dict(MODEL_DECIDERS)
    msha = model_sha(adapter) if is_model else "none"
    path = os.path.join(PARTS, "%s.csv" % NAME)
    # 2026-11-24: K is compared only for the deciders a read-out reaches.  A constant
    # arm's or F's score cannot depend on how many processes were reading out a model
    # beside them -- they run in the parent and make no pass -- and refusing their part
    # files at a different K would shut the plan's own `--k 1` fallback, whose whole point
    # is to restart after a gate failure.
    want = {"script_sha": SHA, "model_sha": msha}
    if is_model:
        want["k"] = str(K)
    if not os.path.exists(path):
        TODO.append(NAME)
        continue
    keys, shas, _lines = read_part(path)
    for got in sorted(shas):
        check_sidecar(want, {"script_sha": got[0], "model_sha": got[1], "k": got[2]},
                      "the part file of %s" % NAME)
    try:
        check_complete(keys, RUNS, "the part file of %s" % NAME)
    except Fail as exc:
        say("  %-12s incomplete and re-run: %s" % (NAME, exc))
        os.remove(path)
        TODO.append(NAME)
        continue
    DONE.append(NAME)
    say("  %-12s %d rows already on disk under this script and this model; skipped"
        % (NAME, len(keys)))
if not DONE:
    say("  nothing on disk for any of the seven deciders, so all seven run here")
say()
say("to run now      %s" % (", ".join(TODO) if TODO else "(nothing)"))
say("K               %d worker process(es); every row carries it, and a resume of a model"
    % K)
say("                decider's part file at another K is refused rather than mixed")
for NAME, ADAPTER in MODEL_DECIDERS:
    say("model digest    %-8s %s" % (NAME, model_sha(ADAPTER)))
    if ADAPTER is not None:
        for _n, _s in adapter_parts(ADAPTER):
            say("                  %-26s %s" % (_n, _s))
say("                the adapter digest is over those two files together: the config")
say("                carries r and lora_alpha, which scale the LoRA output and therefore")
say("                every logit, so weights alone would resume a re-scaled adapter")

# ===========================================================================
# 3  THE REFERENCE READ-OUT, IN A CHILD THAT EXITS
# ===========================================================================
MODEL_TODO = [n for n in TODO if n in dict(MODEL_DECIDERS)]
head("3  THE REFERENCE READ-OUT, MADE BY A CHILD THAT EXITS FIRST")
REF = {}
if not MODEL_TODO:
    say("No model decider is left to run, so no reference is taken and no worker is")
    say("spawned.  The gate is a gate on this run's own read-outs and there are none.")
else:
    say("One child process, alone on the card, reads out the %d fixed gate rows for each of"
        % GATE_N)
    say("the three model deciders and writes those logits to disk.  It EXITS before any")
    say("worker is spawned, and this parent holds no copy of the weights at any time: one")
    say("copy is %.2f GB and four workers plus a fifth copy here would not fit on the card."
        % WORKER_GB)
    say("Taken for all three models before anything is scored, so no score of this run is")
    say("computed before the reference it will be compared against exists.")
    say()
    os.makedirs(TMP, exist_ok=True)
    json.dump(GATE_PROMPTS, open(os.path.join(TMP, "gate_prompts.json"), "w",
                                 encoding="utf-8"))
    with open(os.path.join(TMP, "repeat_prompt.txt"), "w", encoding="utf-8") as fh:
        fh.write(REPEAT_PROMPT)
    _err = open(os.path.join(TMP, "reference.err"), "w", encoding="utf-8")
    _t0 = time.time()
    _rc = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), ROOT,
                          "--reference", TMP], stdout=subprocess.PIPE, stderr=_err,
                         text=True)
    _err.close()
    for line in _rc.stdout.rstrip("\n").split("\n"):
        say("  " + line)
    if _rc.returncode != 0:
        say("FAIL: the reference child exited %d; nothing was scored" % _rc.returncode)
        raise SystemExit(2)
    REF = json.load(open(os.path.join(TMP, "ref.json"), encoding="utf-8"))
    say("  the reference child took %.1f s in all and its weights are off the card" % (
        time.time() - _t0))
    say()
    say("the repeat check of the reference read-out, per model decider (seed %d k %d):"
        % (REPEAT["seed"], REPEAT["k"]))
    for NAME, _a in MODEL_DECIDERS:
        if NAME not in REF["per"]:
            continue
        per = REF["per"][NAME]
        say("  -- %s --" % NAME)
        if not D.repeat_lines(stored_reader(per["first"], per["second"]), REPEAT_PROMPT,
                              lambda text="": say("  " + text),
                              "the repeat check of %s" % NAME):
            raise Fail("the repeat check's own mutation did not fire for %s" % NAME)
    say()
    say("WHERE THIS REPEAT CHECK IS MADE, and where it is not: in the reference child,")
    say("which is one process alone on the card.  No WORKER reads one row twice.  What")
    say("covers the workers is the gate below, every worker's 64 x 3 logits against this")
    say("child's by torch.equal, which is a stronger statement about those rows (two")
    say("processes agreeing) and no statement at all about a worker repeating itself.")
    say()
    say("held against tune.txt's own repeat check on the same row, which is the evidence")
    say("that the adapter on disk is the one the tuning measured and that this step's")
    say("read-out is the tuning's read-out.  The triples are READ OFF tune.txt and not")
    say("typed here: three scripts carried the same hand transcription, so each of their")
    say("archives printed a typed number and a re-tune would have had to be followed into")
    say("three files.  tune.txt is the tuning run's own stdout:")
    TUNED_TRIPLES = D.tuned_triples()
    for NAME in sorted(TUNED_TRIPLES):
        if NAME not in REF["per"]:
            continue
        got = REF["per"][NAME]["first"]
        D.check_triple(got, TUNED_TRIPLES[NAME], NAME,
                       "the reference read-out of seed %d k %d"
                       % (REPEAT["seed"], REPEAT["k"]))
        say("  %-8s %s against tune.txt's %s: equal True"
            % (NAME, got, TUNED_TRIPLES[NAME]))

# ===========================================================================
# 4  THE RUNS
# ===========================================================================
head("4  THE SEVEN DECIDERS THROUGH THE PORT")
say("The four deciders that call no model run in this process, with no card.  F is built")
say("fresh for every simulation, like L, because it keeps one interval per ordered pair")
say("and one instance across two runs would make the second run depend on the first; the")
say("assertion that the memory is empty at a simulation's FIRST decision is what")
say("\"fresh\" means here, and it is made on both.")
say()


def arm_factory(arm):
    """A factory and not an object, so that nothing here has two ways of making one."""
    return lambda: PORT.const_decider(arm)


NO_MODEL = [(n, arm_factory(a)) for n, a in zip(ARM_NAMES, ARMS)] + [("F", LBL.F)]
ROWS = {}
TIMING = {}
for NAME, MAKE in NO_MODEL:
    if NAME not in TODO:
        continue
    _t0 = time.time()
    rows = [one_run(NAME, s, r, MAKE, 1) for s, r in RUNS]
    write_part(os.path.join(PARTS, "%s.csv" % NAME), rows, SHA, "none")
    ROWS[NAME] = rows
    TIMING[NAME] = time.time() - _t0
    say("  %-12s %d runs in %5.1f s, mean polls per run %7.3f, mean score %12.3f ms"
        % (NAME, len(rows), TIMING[NAME],
           statistics.fmean(r["polls"] for r in rows),
           statistics.fmean(r["score_ms"] for r in rows)))

RATE = {}
for NAME, ADAPTER in MODEL_DECIDERS:
    if NAME not in TODO:
        continue
    say()
    say("-- %s, K = %d workers --" % (NAME, K))
    msha = model_sha(ADAPTER)
    shares = [[list(RUNS[i]) for i in range(len(RUNS)) if i % K == w] for w in range(K)]
    for w in range(K):
        json.dump({"decider": NAME, "adapter": ADAPTER, "runs": shares[w],
                   "cand_ids": REF["cand_ids"], "k": K},
                  open(os.path.join(TMP, "job_%d.json" % w), "w", encoding="utf-8"))
    PROCS, LOGS = [], []
    for w in range(K):
        log = open(os.path.join(TMP, "worker_%s_%d.log" % (NAME, w)), "w+",
                   encoding="utf-8")
        err = open(os.path.join(TMP, "worker_%s_%d.err" % (NAME, w)), "w",
                   encoding="utf-8")
        LOGS.append((log, err))
        PROCS.append(subprocess.Popen(
            [sys.executable, "-B", os.path.abspath(__file__), ROOT, "--worker", TMP,
             str(w)], stdout=log, stderr=err, text=True))
    # The gate: every worker's own 64 rows against the reference, before any of them
    # scores anything.  A worker that exits here never reached the gate.
    DEADLINE = time.time() + GATE_DEADLINE_S
    while True:
        ready = [w for w in range(K)
                 if os.path.exists(os.path.join(TMP, "gate_%s_%d.json" % (NAME, w)))]
        dead = [w for w in range(K)
                if PROCS[w].poll() is not None and w not in ready]
        if dead:
            for log, err in LOGS:
                log.close()
                err.close()
            say("FAIL: worker(s) %s exited before the gate; logs in %s" % (dead, TMP))
            raise SystemExit(2)
        if len(ready) == K:
            break
        if time.time() > DEADLINE:
            say("FAIL: the %d workers did not all reach the gate in %d s"
                % (K, GATE_DEADLINE_S))
            raise SystemExit(2)
        time.sleep(0.1)
    REFT = torch.load(os.path.join(TMP, "ref_%s.pt" % NAME))
    GATE_OK = True
    for w in range(K):
        got = torch.load(os.path.join(TMP, "gate_%s_%d.pt" % (NAME, w)))
        meta = json.load(open(os.path.join(TMP, "gate_%s_%d.json" % (NAME, w)),
                              encoding="utf-8"))
        try:
            n = check_gate(REFT, got, "worker %d of %s against the reference" % (w, NAME))
            say("  worker %d: %d x 3 logits bit-identical to the reference "
                "(loaded in %.1f s, %.1f ms per read-out)"
                % (w, n // 3, meta["load_s"], meta["gate_ms"]))
        except Fail as exc:
            GATE_OK = False
            say("  GATE FAILED: %s" % exc)
    if not GATE_OK:
        open(os.path.join(TMP, "stop_%s" % NAME), "w").write("1")
        for proc in PROCS:
            proc.wait()
        for log, err in LOGS:
            log.close()
            err.close()
        say()
        say("The gate refused this run and nothing of %s was scored.  Four workers that" % NAME)
        say("agreed with each other and differed from the single-process read-out would")
        say("pass a worker-against-worker gate, and the score would then be read out at an")
        say("arithmetic the agreement numbers and the latency figure are not.  The")
        say("fallback the plan names is K = 1, measured at 23.1 passes/s; re-run with")
        say("  %s -B %s %s %s--k 1"
            % (sys.executable, os.path.abspath(__file__), ROOT, "--dry " if DRY else ""))
        raise SystemExit(3)
    say("  the gate passed for all %d workers, so they may score" % K)
    open(os.path.join(TMP, "go_%s" % NAME), "w").write("1")
    for proc in PROCS:
        if proc.wait() != 0:
            for log, err in LOGS:
                log.close()
                err.close()
            say("FAIL: a worker of %s exited %d; logs in %s" % (NAME, proc.returncode, TMP))
            raise SystemExit(2)
    for log, err in LOGS:
        log.seek(0)
        for line in log.read().rstrip("\n").split("\n"):
            say("  " + line)
        log.close()
        err.close()
    rows, t_start, t_end, passes, per_ms = [], [], [], 0, []
    for w in range(K):
        res = json.load(open(os.path.join(TMP, "res_%s_%d.json" % (NAME, w)),
                             encoding="utf-8"))
        rows.extend(res["rows"])
        t_start.append(res["t_start"])
        t_end.append(res["t_end"])
        passes += res["passes"]
        per_ms.append(res["readout_ms"])
    check_complete([(r["seed"], r["stream"]) for r in rows], RUNS,
                   "the rows of %s" % NAME)
    write_part(os.path.join(PARTS, "%s.csv" % NAME), rows, SHA, msha)
    ROWS[NAME] = rows
    window = max(t_end) - min(t_start)
    RATE[NAME] = {"passes": passes, "window": window,
                  "rate": passes / window if window else 0.0,
                  "readout_ms": statistics.fmean(per_ms)}
    say("  %d forward passes in a %.1f s window = %.1f passes/s at K = %d, %.1f ms per"
        % (passes, window, passes / window if window else 0.0, K,
           statistics.fmean(per_ms)))
    say("  read-out inside a worker (the tokenizer call and the pass, not the render)")
    if len(RUNS) <= 8:
        say("  per run, as the plan asks so that the real run can be re-estimated:")
        for r in sorted(rows, key=lambda r: (r["seed"], r["stream"])):
            say("    seed %3d stream %2d  decider_ord state_poll_reschedule %3d, "
                "state_poll %3d, passes %3d, ties %2d, answers %d/%d/%d, polls %3d, "
                "score %12.3f ms"
                % (r["seed"], r["stream"], r["ord_resched"], r["ord_poll"], r["reads"],
                   r["ties"], r["n_short"], r["n_mid"], r["n_long"], r["polls"],
                   r["score_ms"]))
    say("  mean polls per run %7.3f, mean score %12.3f ms, ties %d of %d decisions"
        % (statistics.fmean(r["polls"] for r in rows),
           statistics.fmean(r["score_ms"] for r in rows),
           sum(r["ties"] for r in rows), sum(r["reads"] for r in rows)))

# ===========================================================================
# 5  THE CSV
# ===========================================================================
head("5  THE CSV")
with open(CSV, "w", encoding="utf-8") as fh:
    fh.write(",".join(COLUMNS) + "\n")
    for NAME in ORDER:
        with open(os.path.join(PARTS, "%s.csv" % NAME), encoding="utf-8") as part:
            lines = [ln for ln in part][1:]
        for line in lines:
            fh.write(line)
_keys, _shas, _lines = read_part(CSV)
say("%s: %d rows of %d deciders x %d runs, one script digest and %d model digest(s)"
    % (CSV, len(_lines), len(ORDER), len(RUNS), len({s[1] for s in _shas})))
say("columns         %s" % ",".join(COLUMNS))
say("sha256          %s" % D.sha_of(CSV))
say("Floats are written through repr, so ana.py can recompute delay + weight x polls from")
say("these columns and require it to equal the score column bit for bit.  reads, sends,")
say("ties and the three answer counts are empty for the four deciders that call no model.")

# ===========================================================================
# 6  THE WALL TIME OF THE REAL RUN, RE-ESTIMATED FROM THIS RUN
# ===========================================================================
head("6  WHAT THE REAL RUN WILL COST, FROM THIS RUN'S OWN NUMBERS")
if not RATE:
    say("No model decider ran here, so there is nothing to re-estimate from.")
else:
    say("The plan's estimate rests on the roll-in's 160.7 resolution-time decisions per")
    say("run, which is NOT L's: a candidate leaning to ' short' polls more often and costs")
    say("more passes.  Measured here, per model decider:")
    say()
    say("  decider   passes/run (min/mean/max)   mean ms per read-out   800 runs")
    TOTAL = 0.0
    for NAME, _a in MODEL_DECIDERS:
        if NAME not in RATE:
            continue
        per = [r["reads"] for r in ROWS[NAME]]
        mean = statistics.fmean(per)
        TOTAL += 800 * mean
        say("  %-8s  %3d / %7.1f / %3d          %5.1f                   %8d passes"
            % (NAME, min(per), mean, max(per), RATE[NAME]["readout_ms"], 800 * mean))
    REF_MS = statistics.fmean(REF["per"][n]["readout_ms"] for n in RATE)
    K4_MS = statistics.fmean(RATE[n]["readout_ms"] for n in RATE)
    MEAS = statistics.fmean(RATE[n]["rate"] for n in RATE)
    AT_K = TOTAL / MEAS / 60.0 if MEAS else float("nan")
    AT_K1 = TOTAL * (REF_MS + RENDER_MS) / 60000.0
    say()
    say("  the three model deciders together: %d forward passes on seeds 81 to 100"
        % TOTAL)
    say("  measured here at K = %d: %.1f passes/s over the workers' own windows, so"
        % (K, MEAS))
    say("  %d passes take %.1f min (%.1f h)" % (TOTAL, AT_K, AT_K / 60.0))
    say()
    say("  the per-decision cost, measured, in three parts: %.1f ms of read-out (the"
        % K4_MS)
    say("  tokenizer call and the forward pass) inside a worker at K = %d, %.1f ms of the"
        % (K, REF_MS))
    say("  same read-out in the reference child ALONE on the card, and %.1f ms to render"
        % RENDER_MS)
    say("  one observation on the cpu.  K = 1 is therefore an INFERENCE and not a")
    say("  measurement of this run: %d x (%.1f + %.1f) ms = %.1f min (%.1f h), because the"
        % (TOTAL, REF_MS, RENDER_MS, AT_K1, AT_K1 / 60.0))
    say("  render overlaps another worker's pass at K > 1 and overlaps nothing at K = 1.")
    say("  The plan's own figures, for comparison: 385 704 passes at 75.9 passes/s = 84.7")
    say("  min at K = 4 and 278.3 min at K = 1, from 160.7 passes per run.")
    say("  The simulations themselves are negligible beside this: the four deciders that")
    say("  call no model took %s s here."
        % ", ".join("%.1f" % TIMING[n] for n in TIMING))

# ===========================================================================
# 7  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT
# ===========================================================================
head("7  EVERY CHECK, SHOWN FAILING ON A MUTATED INPUT")
say("The mutation is of the INPUT, inside this process.  decider.py's own card-free")
say("mutations are archived in tune.txt part 5 (the pre-review module) and in RUN.txt")
say("section 19 (the current one, in full, because tune.py was not re-run), and are not")
say("repeated here; what is below is")
say("this script's own, plus the two of decider.py's that eval.py is the load-bearing")
say("caller of: the memory assertion and the repeat check.")
say()
FIRED = []


def shows(name, call):
    """Run a mutated input through a check and print the failure it produced."""
    try:
        call()
    except (Fail, SystemExit, KeyError, ValueError) as exc:
        say("  %-24s FIRES: %s" % (name, exc))
        return True
    say("  %-24s DID NOT FIRE -- the check cannot catch its own mutation" % name)
    return False


FIRED.append(shows("card_free",
                   lambda: D.check_card_free("4321, python, 10240 MiB", "mutated")))
FIRED.append(shows("seeds_tuning",
                   lambda: D.check_seeds_tuning([1, 81], "mutated")))
FIRED.append(shows("port.check_seeds",
                   lambda: PORT.check_seeds([PORT.MAX_SEED + 1], "mutated")))
FIRED.append(shows("roles/range",
                   lambda: D.check_roles(dict(ROLES, tuning=[81]), WANT, "mutated")))
FIRED.append(shows("roles/measuring",
                   lambda: D.check_roles(dict(ROLES, selection=[121]), WANT, "mutated")))
FIRED.append(shows("fits/K=8",
                   lambda: check_fits(8, _FREE / 2**30, "mutated")))
_RUN1 = LBL.run_dict(MAIN_P, SEEDS[0])
FIRED.append(shows("choices/narrowed",
                   lambda: check_choices(dict(_RUN1, choices=[ARMS[0]]), "mutated")))
FIRED.append(shows("choices/redecide",
                   lambda: check_choices(dict(_RUN1, redecide=False), "mutated")))
# The port's own refusal of an interval outside the choice set, on a real simulation: the
# value 5.0 is not in the union set, so the first decision raises.  This is the guard every
# decider of the comparison is held by and it belongs to the replay, not to this script.
FIRED.append(shows("port/interval not in choices", lambda: PORT.sim(
    dict(_RUN1, decider=PORT.const_decider(5.0)), None,
    R.rep_seed(LAG_PREFIX, MAIN_P, SEEDS[0], 0),
    R.rep_seed(LAG_PREFIX, MAIN_P, SEEDS[0], 0))))


def _seeded_memory():
    """A fresh L whose per-pair memory was seeded by hand, driven through one decision.

    The mutation the plan asks for, and it goes through the wrapper every simulation above
    goes through rather than calling the check directly: what is being shown is that a
    decider carrying another run's pair intervals cannot reach its first decision.
    """
    dec = D.L(lambda _prompt: (0.0, 0.0, 1.0))
    dec.mem[("h0", "h1")] = ARMS[0]
    obs = {"state": {}, "resolved": None, "history": [],
           "where": {"node": "h0", "neighbor": "h1", "kind": "state_poll",
                     "ts_ms": 0, "ordinal": 0}}
    FirstCallEmpty(dec, "mutated", True)(obs)


def _unreadable_memory():
    """A decider that keeps per-pair state under a name the wrapper does not know.

    What the wrapper used to pass in silence: `memory()` returned None, the assertion ran
    on nothing, and the output said as much about this simulation as about any other.
    """

    class Other:
        """A decider whose per-pair intervals are not `mem` and not `ival`."""

        def __init__(self):
            self.pair_interval = {("h0", "h1"): ARMS[0]}

        def __call__(self, _obs):
            """The arm it holds, which is not what is being tested here."""
            return ARMS[0]

    obs = {"state": {}, "resolved": None, "history": [],
           "where": {"node": "h0", "neighbor": "h1", "kind": "state_poll",
                     "ts_ms": 0, "ordinal": 0}}
    FirstCallEmpty(Other(), "mutated", True)(obs)


def _seam_next_row():
    """The seam check against the label row of the NEXT decision of the same run.

    The only mutation of this check that is a real input: two renderings of two different
    decisions of one run, which is what a round trip that moved a field would look like.
    """
    other = next(r for r in ALL_ROWS
                 if int(r["seed"]) == _SEAM_SEED and int(r["k"]) == _SEAM_K + 1)
    D.check_render_roundtrip(
        _live_text, D.text_of(other["state"], other["resolved"], other["history"],
                              other["ts_ms"], other["neighbor"], "mutated"), "mutated")


FIRED.append(shows("memory_empty/seeded by hand", _seeded_memory))
FIRED.append(shows("memory_empty/a decider it cannot read", _unreadable_memory))
FIRED.append(shows("seam/the row of k + 1", _seam_next_row))
FIRED.append(shows("seam/rollin anchor",
                   lambda: check_rollin(ARMS[0], ARMS[2], _SEAM_K, "mutated")))
FIRED.append(shows("counters/reads", lambda: check_counters(2, 1, 1, 1, "mutated")))
FIRED.append(shows("counters/sends", lambda: check_counters(1, 2, 1, 1, "mutated")))
FIRED.append(shows("sidecar/script digest", lambda: check_sidecar(
    {"script_sha": SHA, "model_sha": "none"},
    {"script_sha": "deadbeef", "model_sha": "none"}, "mutated")))
FIRED.append(shows("sidecar/model digest", lambda: check_sidecar(
    {"script_sha": SHA, "model_sha": model_sha(MODEL_DECIDERS[1][1])},
    {"script_sha": SHA, "model_sha": "none"}, "mutated")))
FIRED.append(shows("sidecar/K", lambda: check_sidecar(
    {"script_sha": SHA, "model_sha": "none", "k": str(K)},
    {"script_sha": SHA, "model_sha": "none", "k": str(K + 1)}, "mutated")))
FIRED.append(shows("complete/one row missing",
                   lambda: check_complete(RUNS[1:], RUNS, "mutated")))
FIRED.append(shows("complete/one row twice",
                   lambda: check_complete(RUNS[:1] + RUNS, RUNS, "mutated")))
FIRED.append(shows("prompt_len/lowered bound",
                   lambda: D.check_prompt_len(GATE_LENS, 300, "mutated")))
if REF:
    _REFT = torch.load(os.path.join(TMP, "ref_%s.pt" % MODEL_TODO[0]))
    _MOVED = _REFT.clone()
    _MOVED[0][0] = _MOVED[0][0] + 1.0
    FIRED.append(shows("gate/one value moved by 1",
                       lambda: check_gate(_REFT, _MOVED, "mutated")))
    FIRED.append(shows("gate/shape",
                       lambda: check_gate(_REFT, _REFT[:8], "mutated")))
else:
    say("  gate                     NOT SHOWN IN THIS INVOCATION and not counted: no")
    say("                           model decider ran here (every part file was already")
    say("                           on disk under this script and this model), so there")
    say("                           is no reference tensor in this process to mutate.")
    say("                           The invocation that produced those rows shows it.")
say()
say("The memory assertion is also exercised POSITIVELY on every simulation above, at the")
say("first decision of each, which is what the wrapper exists for; the mutation here is")
say("the same check on a memory seeded by hand.  The repeat check's own mutation is in")
say("part 3, inside decider.repeat_lines, once per model decider.")
say()
say("WHAT HAS NO MUTATION, named and not counted:")
say("- the two subprocess failure branches (a worker that exits before the gate, a worker")
say("  that never reaches it).  Staging either needs a killed child and the run then has")
say("  no scores, so they are claims here and not evidence.")
say("- decider.py's own card-free mutations.  tune.txt part 5 archives the set the")
say("  pre-review module had; the ones added since (the round trip through L itself, the")
say("  digest cover, the triple, and three on tune.txt's own kept-epoch block) are")
say("  archived in RUN.txt section 19 instead, because tune.py was not re-run.  No count")
say("  is quoted here because this script does not run them: it imports them unchanged.")
say()
if all(FIRED):
    say("all %d mutations of this script produced a failure" % len(FIRED))
else:
    say("FAILURE OF THIS SECTION: %d of %d mutations did not produce a failure"
        % (len(FIRED) - sum(FIRED), len(FIRED)))

# ===========================================================================
# 8  WHAT THIS RUN DOES NOT SHOW
# ===========================================================================
head("8  WHAT THIS RUN DOES NOT SHOW")
if DRY:
    say("- NOTHING about which candidate is better, and nothing about any decider's score.")
    say("  Seeds 1 and 2 are tuning seeds that both candidates were fitted on, so these")
    say("  are scores on training data.  Two seeds and two streams are four runs, which is")
    say("  not a sample of anything.")
say("- Nothing about L on the testbed.  No run of this step touches the emulator.")
say("- Nothing about the difference between L's score and an arm's being the decider's")
say("  difference alone.  A decider whose send-time value differs from its")
say("  resolution-time value skips resolve's identity early return and draws one extra")
say("  tick lag from that pair's stream; the three fixed arms never differ and never draw")
say("  it, F and L do.  So the comparator family consumes the pair streams differently")
say("  from L by construction, in an unknown direction.")
say("- Nothing about the labels being right, and nothing about the agreement with them.")
say("  The states the labels sit on came from a uniformly random roll-in and the label is")
say("  the best arm under an F suffix L does not supply, while here L drives the run.")
say("- Nothing about the read-out's own noise in a score.  Exact ties occur in bf16 and")
say("  the argmax of 13 of 512 rows differs between bf16 and float32 batch 1, so some of")
say("  the decisions above are the arithmetic's and not the model's, and what that does to")
say("  a score is bounded nowhere.")
say("- Nothing about the gain, the band or the choice between a and b.  Those are ana.py's")
say("  and this run computes none of them.")
say("- Nothing about the pass time of a decision.  The read-out times above include the")
say("  tokenizer call and are taken with K workers queueing on one card; latency.py is")
say("  the measurement.")
say()
say("wall clock    %.1f s (%.1f min) from the first line to this one"
    % (time.time() - T_START, (time.time() - T_START) / 60.0))
say("finished      %s" % time.strftime("%Y-%m-%dT%H:%M:%S%z"))
say("end of eval.py")
if not all(FIRED):
    raise SystemExit(1)
