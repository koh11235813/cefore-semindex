#!/usr/bin/env python3
"""L as a port decider, and everything the scripts of this step read out a model with.

Imported, never run: there is no `main` here.  `tune.py`, `eval.py`, `agree.py` and
`latency.py` all decide with the same read-out, so the read-out lives once.  A second copy
of it would let the held-out cross-entropy of the tuning, the agreement with the labels
and the replay score be taken at three slightly different arithmetics while every one of
them printed a plausible number -- and the whole argument for bf16 batch 1 is that the
score, the agreement and the latency are read out the SAME way.

WHAT IS HERE, AND WHY EACH PIECE IS SHARED RATHER THAN LOCAL TO ONE SCRIPT
  - the model loading: the base in bf16 and, optionally, one adapter ATTACHED and NOT
    merged, with the adapter's dtype asserted to be float32 at the read-out;
  - the read-out: batch 1, the three candidate logits at the last real position, and the
    explicit first-maximum tie rule with a tie counter;
  - the repeat check: one fixed row read out twice, the three logits compared bit for bit
    and the two passes required to pick the same arm;
  - `L`, the decider: the send-time convention keyed on the ordered pair `(node,
    neighbor)`, 6.0 s before a pair's first resolution, no model call at a send, a memory
    that is rebuilt per simulation and can be asserted empty, and the assertion that the
    rendered text names no node;
  - the seed guard of the tuning half and the label-to-class map, which three of the four
    scripts need and which must mean the same thing in all of them.

WHAT IS DELIBERATELY NOT HERE
  The LoRA configuration and the 205-module assertion (one script builds an adapter:
  `tune.py`), candidate b's row filter (likewise), and anything that prints a report.
  Every function below either returns a value or raises `Fail`; the only thing that writes
  is `self_check`, and it writes through a `say` the caller hands in.

NO GPU IS NEEDED for the send-time memory, the tie rule, the arm order, the name
assertion, the repeat comparator or the render round-trip, and `self_check` exercises all
of them with a stub read-out, each on a mutated input, so that the parts a card is not
needed for can be checked on a machine that has none.

The prompt, the prefill, the candidate words and the rendering road are `prefill.py`'s;
importing it is what sets the thread caps, the offline HF variables, ROOT and the sys.path
entries, so it is the FIRST import of every script that uses this module.
"""
import json
import os
import sys

import prefill as P  # noqa: E402  (nothing may be imported before it; see the docstring)

# 2026-11-24: `evidence/22` is put on the path HERE as well, by this file's own location,
# and not left to `prefill.py`'s ROOT-derived entry.  `port.py` asserts the realpath of the
# engine modules it imports, so a ROOT whose `repo` is not the real worktree (a local run
# that stands a directory in for the host's ROOT) makes that assertion fail on a path that
# differs from this file's only by a symlink.  Resolving it from `__file__` gives the same
# directory on the host, where this file sits beside `../22` under ROOT/repo anyway.
_EV22 = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "22"))
if _EV22 not in sys.path:
    sys.path.insert(0, _EV22)

import port as PORT   # noqa: E402  the arms, and the >100 seed guard on every path

# The three arms as the port's own floats, not as literals: `../13/replay.py` refuses an
# interval that is not an element of the run's `choices` by exact float identity, so the
# value L returns has to BE the port's float and not one that prints the same.
ARMS = PORT.ARMS
# What a pair is answered with at a send before it has ever resolved (the user's own
# decision 9).  Written as the arm and not as 6.0, so that it cannot drift from the arm.
SEND_START = ARMS[1]
# The five node names of this topology.  A rendered text that contains one of them as a
# substring has leaked a name into the model input, which is the one thing the input rule
# forbids outright.
NODE_NAMES = ("h0", "h1", "h2", "h3", "h4")
MAX_TUNE_SEED = 80          # seeds 1-80 tune, 81-100 select, above 100 never
DEV = "cuda:0"
# The class index of each arm, for a three-way loss and for the agreement count.  A dict
# and not a nearest match: a label that is not one of the three arms is a row this read-out
# cannot represent, and a KeyError at the first such row is the report.
LABEL_ARM = {ARMS[0]: 0, ARMS[1]: 1, ARMS[2]: 2}


class Fail(Exception):
    """A check that did not hold.

    A type of its own, because `self_check` feeds every check below a mutated input and
    requires it to raise: catching `Exception` there would also catch a typo in the
    mutation and print it as if the check had worked.
    """


# ---------------------------------------------------------------------------
# the checks, every one of them a function of plain values
# ---------------------------------------------------------------------------
# They take numbers, strings and dicts rather than a model or a tokenizer ON PURPOSE: that
# is what lets each one be shown failing on a mutated argument in a process with no card,
# and it is why `self_check` can run anywhere.
def check_seeds_tuning(seeds, where):
    """Every seed handed in is in the tuning half, 1 to 80.

    `port.check_seeds` refuses above 100 and lets 81 to 100 through, so it cannot stand in
    for this one: 81 to 100 are the selection half, and a tuning number that quietly used a
    selection seed would make the choice between the two candidates circular.
    """
    bad = sorted({int(s) for s in seeds if int(s) > MAX_TUNE_SEED})
    if bad:
        raise Fail("%s: seeds above %d: %s" % (where, MAX_TUNE_SEED, bad[:8]))
    return sorted({int(s) for s in seeds})


def target_of(label):
    """The class index of a row's label, by exact dict lookup."""
    return LABEL_ARM[label]


def check_roles(roles, want, where):
    """Each seed role is inside its own range and none of them meets 121 to 160.

    2026-11-24: shared here rather than written again in each of the three scripts that
    run after the tuning, because the three roles mean one thing across the step and a
    second copy of the ranges is a second definition of which seeds select.  `want` is
    `{role: (low, high)}` and is the caller's, since a script's roles are not the same
    set in its dry mode as in its real one; what is NOT the caller's is the measuring
    range, which is written here once.

    The measuring seeds are tested BEFORE the ranges although the range test would catch
    every one of them: after it, the measuring branch could never fire on an input the
    range test admits, and a branch that cannot fail is not evidence.
    """
    union = sorted({int(s) for seeds in roles.values() for s in seeds})
    met = sorted(set(union) & set(range(121, 161)))
    if met:
        raise Fail("%s: the seed set meets the measuring seeds 121-160: %s" % (where, met))
    for role, seeds in sorted(roles.items()):
        lo, hi = want[role]
        bad = sorted(int(s) for s in seeds if not lo <= int(s) <= hi)
        if bad:
            raise Fail("%s: the %s role carries %s, outside %d-%d"
                       % (where, role, bad[:4], lo, hi))
    return union


def check_card_free(compute_apps, where):
    """No other compute process is on the card.

    The host is shared and has no scheduler, so this is the only thing between a run of
    ours and someone else's job.  It takes the TEXT `nvidia-smi --query-compute-apps`
    printed rather than running it, which is what gives the branch a mutation at last: a
    second process on a shared card cannot be staged without disturbing its owner, and
    this check went without a negative test for that reason.
    """
    text = (compute_apps or "").strip()
    if text:
        raise Fail("%s: another compute process is on the GPU: %s"
                   % (where, text.replace("\n", " | ")))
    return True


def check_arm_order(seconds, arms, where):
    """The three candidate WORDS stand for the three arms in the same order.

    The read-out returns an index into the candidate order and `L` turns that index into
    an interval by indexing `ARMS`.  If the two orders ever part, every decision is mapped
    to the wrong interval, the port still accepts it (it is a member of `choices`), and
    nothing else in the step notices.  So the correspondence the prompt states in words is
    checked against the floats the port compares.
    """
    want = tuple(arms)
    got = tuple(seconds[w] for w in P.CANDIDATES)
    if got != want:
        raise Fail("%s: the candidate words stand for %s, the arms are %s"
                   % (where, got, want))
    return want


def check_candidate_ids(got, expected, where):
    """The three answer words tokenize to the ids this step's own output recorded."""
    want = [expected[w] for w in P.CANDIDATES]
    if list(got) != want:
        raise Fail("%s: candidate ids %s, expected %s" % (where, list(got), want))
    if len(set(got)) != 3:
        raise Fail("%s: the three candidate ids are not distinct: %s" % (where, list(got)))
    return list(got)


def check_no_node_name(text, where):
    """The rendered model input contains no node name as a substring.

    The reach of this assertion is the TEXT and nothing else: it is not evidence about the
    policy, which may key its own per-pair memory on the pair identity because a live node
    knows which node it is.  What it does cover is the whole of what the model reads.
    """
    hit = [n for n in NODE_NAMES if n in text]
    if hit:
        raise Fail("%s: the rendered text contains the node name(s) %s" % (where, hit))
    return True


def check_render_roundtrip(from_obs, from_row, where):
    """Two renderings of one decision are the same string.

    The seam this comparator exists for: the model is tuned on label rows read back out
    of a gzip-JSON file and it is read out inside a simulation on the live engine's own
    observation.  If the two roads to the text ever differ, the tuning and the scoring are
    of two different inputs, both scripts print plausible numbers, and nothing else in the
    step compares them.

    2026-11-24, WHAT EACH CALLER OF THIS COMPARATOR ACTUALLY COVERS, because the
    comparator itself cannot know which two strings it was handed and an earlier version
    of this docstring claimed the seam was closed wherever it was called:
      - `eval.py` part 1 closes the seam itself.  It runs the LABELLING roll-in of one
        tuning seed, takes the live engine's own captured observation of decision k, and
        compares its rendering with the rendering of the gzip-JSON row of the same
        (seed, k).  That comparison has a failing input -- the JSON round trip turns a
        tuple into a list and a non-string dict key into a string -- and a mutation
        (the row of k + 1) is shown firing beside it.
      - `self_check` below covers LESS, and says so in its own output: it compares the
        text `L.__call__` rendered with the text rendered straight from the same label
        row, so it catches an `L` that reads a different field of the observation and it
        does NOT touch the JSON round trip, which sits on both sides of it.
    """
    if from_obs != from_row:
        n = min(len(from_obs), len(from_row))
        at = next((i for i in range(n) if from_obs[i] != from_row[i]), n)
        raise Fail("%s: the two renderings differ at character %d: %r against %r"
                   % (where, at, from_obs[at:at + 40], from_row[at:at + 40]))
    return len(from_obs)


def check_prompt_len(lengths, bound, where, unit="tokens"):
    """No prompt is longer than `bound` of whatever `unit` the caller counted in.

    Nothing is truncated anywhere in this step, so this is the check that says so: the
    longest prompt of the whole cell is 519 tokens with the legend, and a row above the
    bound would be a row the read-out silently read the wrong position of.

    2026-11-24: `unit` comes from the caller because the card-free self-check has no
    tokenizer and hands this function a CHARACTER count, and the message then read
    "longer than 8 tokens, longest 694" over a length of 694 characters.  The bound and
    the message have to name the same thing the caller counted, or a reader prices a
    prompt in the wrong unit.
    """
    over = sorted((n for n in lengths if n > bound), reverse=True)
    if over:
        raise Fail("%s: %d prompt(s) longer than %d %s, longest %d"
                   % (where, len(over), bound, unit, over[0]))
    return max(lengths) if lengths else 0


def check_adapter_dtype(dtypes, where):
    """The attached adapter's parameters are float32 over the bf16 base.

    peft's `autocast_adapter_dtype` defaults to True on both `get_peft_model` and
    `PeftModel.from_pretrained`, so this is the installed build's own behaviour and not a
    setting any script here passes.  It is asserted rather than trusted BECAUSE it is a
    default: an upgrade that changed it would change the arithmetic every number of this
    step is read out at, with nothing else in the step noticing.
    """
    want = ["torch.float32"]
    got = sorted(set(dtypes))
    if got != want:
        raise Fail("%s: adapter dtype %s, not %s" % (where, got, want))
    return got


def check_unmerged(n_lora_layers, merged_flags, where):
    """The adapter is attached as layers and none of them is merged into the base.

    `merge_and_unload` would change the arithmetic and would need its own agreement check
    against the unmerged read-out, so the read-out this step scores is the unmerged one and
    that is checked and not assumed.
    """
    if not n_lora_layers:
        raise Fail("%s: no adapter layer is attached" % where)
    merged = [i for i, flag in enumerate(merged_flags) if flag]
    if merged:
        raise Fail("%s: %d of %d adapter layers are merged into the base"
                   % (where, len(merged), len(merged_flags)))
    return n_lora_layers


def check_pick(three, index, where):
    """`index` is the FIRST maximum of the three candidate logits.

    The decision is an explicit scan over the fixed candidate order keeping the first
    maximum, so the shortest interval wins an exact tie.  `torch.argmax` is not used: this
    step has not read what the installed build does with equal values, and a decision that
    depends on that is not a decision this step fixed.  Exact ties do occur -- the bf16
    batch-1 margin over 512 label rows has a minimum of 0.0000 -- so the rule has a reach
    and this is the check that the scan implements it.
    """
    if three[index] != max(three):
        raise Fail("%s: picked %d (%r), which is not the maximum of %r"
                   % (where, index, three[index], list(three)))
    earlier = [j for j in range(index) if three[j] == three[index]]
    if earlier:
        raise Fail("%s: picked %d, but position(s) %s tie it and come first"
                   % (where, index, earlier))
    return index


def check_repeat(first, second, arm_first, arm_second, where):
    """Two passes over one input gave bit-identical logits and the same arm.

    The ticket asks this of the read-out that DECIDES, and the untuned evidence on record
    does not cover an attached adapter, so every script of this step that decides with one
    repeats it.  It is the only evidence in the step that the tuned read-out uses no draw.
    Not a tolerance: the question is whether the same input gives the same bits, and a
    tolerance answers a different question.
    """
    bad = [i for i in range(len(first)) if first[i] != second[i]]
    if bad:
        raise Fail("%s: %d of %d logits differ, first at %d: %r against %r"
                   % (where, len(bad), len(first), bad[0], first[bad[0]], second[bad[0]]))
    if arm_first != arm_second:
        raise Fail("%s: the two passes chose %r and %r" % (where, arm_first, arm_second))
    return True


def check_memory_empty(mem, where):
    """A decider's per-pair memory holds nothing yet.

    L's send-time memory is state of the same kind F's is, so one L object reused across
    simulations would carry pair intervals from one run into the next unguarded.  The
    model is kept across runs (reloading 10 GB per run is not what "fresh" means) and the
    memory is rebuilt, so this is what "rebuilt" means, asserted at the first decision of
    a simulation.
    """
    if mem:
        raise Fail("%s: the per-pair memory is not empty: %d entr(ies), e.g. %s"
                   % (where, len(mem), sorted(mem.items())[:2]))
    return True


def check_no_read_at_send(before, after, where):
    """No model pass was made between two counts taken around a send.

    The user's decision 9 is that a send is answered with the interval the pair already
    holds, so one decision costs one forward pass and not two.  The latency figure of this
    step is read as the cost of A decision on that basis, so the claim is counted rather
    than described.
    """
    if after != before:
        raise Fail("%s: %d model pass(es) were made at a send" % (where, after - before))
    return True


# ---------------------------------------------------------------------------
# what code produced a number, and what tune.txt says the tuned read-out gave
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
TUNE_TXT = os.path.join(HERE, "tune.txt")
# 2026-11-24: every module whose BYTES decide what one scored row is, named by its role.
# Shared here so that the scripts digest one list and not three of their own.  A digest
# over the calling script, the read-out and the prompt alone does not see a change to the
# engine, the renderer or the scorer, and a real run that restarted after such a change
# would resume and write one csv out of rows produced by two experiments under one
# decider name.
CODE_FILES = (
    ("decider.py", os.path.join(HERE, "decider.py")),
    ("prefill.py", os.path.join(HERE, "prefill.py")),
    ("../22/observe.py", os.path.join(_EV22, "observe.py")),
    ("../22/port.py", os.path.join(_EV22, "port.py")),
    ("../22/label.py", os.path.join(_EV22, "label.py")),
    ("../22/score.py", os.path.join(_EV22, "score.py")),
    ("../13/replay.py", os.path.realpath(os.path.join(HERE, "..", "13", "replay.py"))),
)
# What the digests do NOT cover, printed beside them rather than left to a reader to
# notice that a digest is narrower than the thing it is read as protecting.  Named, not
# digested, because each is either pinned elsewhere or outside this step's reach.
CODE_UNCOVERED = (
    "the base weights (pinned by revision; the digest is RUN.txt section 2)",
    "the host's venv: torch, transformers, peft and numpy (versions printed instead)",
    "../22/weight.json, whose weight every row carries and ana.py re-checks",
    "the label file (its own sha256 is checked against ../22/labels.sha256)",
    "../20/predict.py and ../21/predict.py, which no row of the csv passes through",
)


def sha_of(path):
    """The sha256 of one file's bytes."""
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def code_digests(script_path):
    """`[(label, sha256)]` for the calling script and every module of `CODE_FILES`.

    The calling script first, because it is the one a reader of the output is holding.
    Every script of this step prints this list, so an archived output names the code that
    produced it: `tune.txt` does not, and the adapters it wrote are therefore bound to no
    code state at all (RUN.txt section 19).
    """
    label = os.path.basename(script_path)
    return [(label, sha_of(script_path))] + [(n, sha_of(p)) for n, p in CODE_FILES]


def combined_sha(parts):
    """One digest over an ordered `[(label, sha)]`, for a csv column or a sidecar."""
    import hashlib
    return hashlib.sha256("".join(s for _n, s in parts).encode()).hexdigest()


def check_digest_cover(parts, want, where):
    """The digest list covers exactly the roles it is read as covering.

    A list that quietly lost a member would still print a plausible combined digest and
    would still refuse a resume -- it would just stop refusing the one change the member
    was there to catch.  So the membership is asserted and not left to the literal above.
    """
    got = [n for n, _s in parts]
    if got != list(want):
        raise Fail("%s: the digest covers %s, not %s" % (where, got, list(want)))
    if len({s for _n, s in parts}) != len(parts):
        raise Fail("%s: two of the digested files have the same sha256: %s"
                   % (where, got))
    return got


def tune_kept(path=None):
    """What `tune.txt` archived about each candidate's KEPT epoch, read off the file."""
    with open(path or TUNE_TXT, encoding="utf-8") as fh:
        return kept_from(fh)


def tuned_triples(path=None):
    """`{candidate: [three logits]}` of the kept adapters' archived repeat check."""
    return {name: got["triple"] for name, got in tune_kept(path).items()}


def kept_from(lines):
    """`{candidate: {epoch, epoch_ce, ce, agree, rows, ties, triple}}` from tune.txt.

    Takes the LINES and not the path, so that a mutated copy of the archive can be run
    through it in a process with no file to write: that is what gives this parser a
    negative test at all.

    2026-11-24: the logit triples here were transcribed BY HAND into eval.py, agree.py
    and latency.py, so each of those three archives carried a typed number and a re-tune
    would have had to be followed into three files instead of one.  `tune.txt` is the
    tuning run's own stdout, archived byte for byte, so it is the one place these values
    exist as a measurement rather than as a transcription.  The held-out cross-entropies
    come out of the same read, so a later script can hold its own read-out against the
    number that CHOSE the epoch and not only against the agreement count.

    Parsed and then checked against itself, so a reformatted or truncated `tune.txt` is a
    failure here and not a silently empty dict: the two passes of the repeat check must
    agree with each other, the kept cross-entropy must be the lowest of the three the keep
    line prints, the kept epoch must be the one the frozen rule picks from those three
    (lowest, earliest on a tie), and exactly the two candidates must be found.
    """
    out, cand, armed, pending = {}, None, False, None
    for line in lines:
        text = line.strip()
        if text.endswith("CANDIDATE A"):
            cand, armed = "a", False
        elif text.endswith("CANDIDATE B"):
            cand, armed = "b", False
        elif text.startswith("KEPT "):
            # "KEPT b epoch 3: the lowest held-out cross-entropy of the three (c1, c2,
            # c3), the rule ..." -- the keep rule's own line, which carries every epoch's
            # cross-entropy and the epoch it chose in one place.
            name, epoch = text.split()[1], int(text.split()[3].rstrip(":"))
            inner = text[text.index("(") + 1:text.index(")")]
            out.setdefault(name, {}).update(
                epoch=epoch, epoch_ce=[float(v) for v in inner.split(",")])
        elif text.startswith("cross-entropy ") and "agreement" in text:
            head_, tail = text.split(",", 1)
            agree, rows = tail.split("agreement", 1)[1].split(",")[0].strip().split("/")
            out.setdefault(cand, {}).update(
                ce=float(head_.split()[1]), agree=int(agree), rows=int(rows),
                ties=int(tail.rsplit("ties", 1)[1].split()[0]))
        elif text.startswith("the repeat check, with the kept adapter attached"):
            armed = True
        elif armed and text.startswith("two passes over one row:"):
            pending = json.loads(text.split(":", 1)[1].strip())
        elif armed and pending is not None and text.startswith("and:"):
            second = json.loads(text.split(":", 1)[1].strip())
            if second != pending:
                raise Fail("tune.txt: candidate %s printed %s and %s for one row"
                           % (cand, pending, second))
            out.setdefault(cand, {})["triple"] = pending
            armed, pending = False, None
    if sorted(out) != ["a", "b"]:
        raise Fail("tune.txt: the kept epoch was found for %s, not for a and b"
                   % sorted(out))
    for name, got in sorted(out.items()):
        if sorted(got) != ["agree", "ce", "epoch", "epoch_ce", "rows", "ties", "triple"]:
            raise Fail("tune.txt: candidate %s's kept epoch is %s, not the seven fields"
                       % (name, sorted(got)))
        if len(got["triple"]) != 3 or not all(isinstance(v, float)
                                              for v in got["triple"]):
            raise Fail("tune.txt: candidate %s's triple is %r, not three floats"
                       % (name, got["triple"]))
        ce = got["epoch_ce"]
        if len(ce) != 3:
            raise Fail("tune.txt: candidate %s's keep line prints %d epochs, not 3"
                       % (name, len(ce)))
        if min(range(3), key=lambda i: (ce[i], i)) + 1 != got["epoch"]:
            raise Fail("tune.txt: candidate %s kept epoch %d where the frozen rule picks "
                       "%d from %s" % (name, got["epoch"],
                                       min(range(3), key=lambda i: (ce[i], i)) + 1, ce))
        if got["ce"] != min(ce):
            raise Fail("tune.txt: candidate %s's kept cross-entropy is %r and the lowest "
                       "of its three is %r" % (name, got["ce"], min(ce)))
    return out


def margin_of(epoch_ce):
    """(the kept cross-entropy, the runner-up's, the margin between them).

    The quantity the epoch choice rests on, computed from `tune.txt`'s own three numbers
    rather than described, because for one candidate it is 0.001850 of 0.893928 and a
    reader who is not shown it reads "the rule did visible work" and stops.
    """
    at = min(range(len(epoch_ce)), key=lambda i: (epoch_ce[i], i))
    rest = sorted(v for i, v in enumerate(epoch_ce) if i != at)
    return epoch_ce[at], rest[0], rest[0] - epoch_ce[at]


def check_triple(got, want, name, where):
    """This run's read-out of the repeat row reproduces `tune.txt`'s triple exactly.

    Not a tolerance: the question is whether the adapter on disk read out through this
    step's code gives the bits the tuning archived, which is the only evidence that the
    adapter being scored is the adapter that was measured.
    """
    if list(got) != list(want):
        raise Fail("%s: the read-out of %s does not reproduce tune.txt's triple: %s "
                   "against %s" % (where, name, list(got), list(want)))
    return list(got)


# ---------------------------------------------------------------------------
# the model, the prompt and the read-out
# ---------------------------------------------------------------------------
def nvidia_compute_apps():
    """What `nvidia-smi` says is computing on the card, as text."""
    import subprocess
    return subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout


def load_tokenizer():
    """The checkpoint's tokenizer, left-padded, as the read-out needs it."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    P.prepare_padding(tok)
    return tok


def load_base(dtype=None):
    """The base checkpoint on the card, in bfloat16, as the composite class.

    `Gemma4ForCausalLM` is not used: it loads this checkpoint with 541 missing weights,
    i.e. a randomly initialised text tower that would still print plausible logits.
    bfloat16 and not float32 although float32 is the faster and the batch-stable one here:
    the tuning forward runs on a bf16 base, and reading out in another arithmetic than the
    tuned one would score a model that was not tuned.
    """
    import torch
    from transformers import Gemma4ForConditionalGeneration
    model = Gemma4ForConditionalGeneration.from_pretrained(
        P.MODEL_DIR, dtype=torch.bfloat16 if dtype is None else dtype)
    return model.to(DEV).eval()


def attach_adapter(base, adapter_dir, where):
    """One saved adapter attached to a loaded base, unmerged, float32 over bf16.

    No `task_type` was used when it was built, so plain `PeftModel.forward` passes
    `logits_to_keep` and `position_ids` through to the base model's forward, which is the
    call the read-out makes.  The two properties the arithmetic depends on are asserted
    here and not left to peft's defaults.
    """
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    dtypes = [str(p.dtype) for n, p in model.named_parameters() if "lora_" in n]
    check_adapter_dtype(dtypes, where)
    layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    check_unmerged(len(layers), [bool(m.merged) for m in layers], where)
    return model


def candidate_ids(tok, text):
    """The three answer words' token ids, derived at this prompt's own last position.

    Derived and not written down: each word has to add exactly ONE token after this
    prompt's ids, which is a property of the word AND of what precedes it.  The expected
    values are then checked, so a tokenizer change is a failure here and not a silent
    change of which three logits are read.
    """
    out = []
    base = P.ids_of(tok, text)
    for word in P.CANDIDATES:
        longer = P.ids_of(tok, text + word)
        if longer[:len(base)] != base or len(longer) != len(base) + 1:
            raise Fail("candidate_ids: %r is not one token at this position" % word)
        out.append(longer[-1])
    return check_candidate_ids(out, P.EXPECTED_IDS, "candidate_ids")


def text_of(state, resolved, history, ts_ms, neighbor, where="text_of"):
    """The model input of one decision, with the no-node-name assertion on it.

    The one road from the five fields to the text, so that a decision inside a simulation
    and a label row of the tuning set are rendered by the same call.  The assertion runs on
    every call rather than on a sample: it costs five substring searches and it is the only
    thing standing between a rendering change and a name in the model input.
    """
    text = P.rendered({"state": state, "resolved": resolved, "history": history,
                       "ts_ms": ts_ms, "neighbor": neighbor})
    check_no_node_name(text, where)
    return text


def prompt_of(tok, text):
    """The chat-templated prompt, prefill included, for one rendered observation."""
    return P.chat_text(tok, text)


def forward_three(model, cand_ids, ids):
    """The three candidate logits of one ALREADY tokenized prompt, batch 1, one pass.

    2026-11-24, split out of `readout` below so that the pass can be timed without the
    tokenizer in it: the plan requires the render and the tokenize to be printed BESIDE
    the pass and not inside it, and the old `readout` tokenized inside the call, so a
    latency figure taken through it would have quoted the sum as "the pass" and would not
    be comparable with the 43.02 ms on record.  The split moves no arithmetic: the same
    ops in the same order on the same ids, which is why the tuned read-out still
    reproduces the logits the tuning archived for one fixed row, bit for bit.

    2026-11-24, CORRECTED after `latency.py` measured the parts.  This docstring first
    said the gap between the tuning's 72 ms per held-out row and the 43.02 ms a pass was
    measured at "is the tokenizer call `readout` makes".  THAT IS WRONG: the tokenizer is
    0.409 ms median and the chat template 0.214 ms, while the attached adapter costs about
    25 ms at batch 1 (69.76 ms against the untuned 44.36), so the gap is the ADAPTER and
    the tokenizer is about a sixtieth of it.  The split is still the right split; the
    reason is the plan's requirement above and not a tokenizer cost.

    Note for a reader of `latency.py`'s figures: the `torch.tensor([ids], device=DEV)
    host-to-device copy below is INSIDE this function and therefore inside the timed
    pass, while `precheck.py`'s `forward_last` excluded it.  That is the whole difference
    between the two quantities beside the five tokens of prompt, and `latency.py` prints
    it rather than leaving a reader to find it here.

    Returns three python floats, which is lossless for float32 values and lets every check
    above compare them in a process with no torch.
    """
    import torch
    ii = torch.tensor([ids], device=DEV)
    with torch.inference_mode():
        out = model(input_ids=ii, use_cache=False, logits_to_keep=1)
    three = out.logits[:, -1, :][0, cand_ids].float().cpu()
    return (float(three[0]), float(three[1]), float(three[2]))


def readout(model, tok, cand_ids, prompt):
    """The three candidate logits of one prompt, batch 1, one forward pass.

    Batch 1 everywhere a decision is produced, and that is the consequential choice: the
    simulation cannot advance until the decider answers, so there is no batch to form, and
    batching in bf16 changes the argmax of 14 of 512 label rows against batch 1.  Taking
    the held-out loss and the agreement at batch 1 as well is what makes the score, the
    agreement and the latency read out the same model.
    """
    return forward_three(model, cand_ids, P.ids_of(tok, prompt))


def pick(three):
    """(index of the winning word, whether the maximum was tied), by an explicit scan.

    The scan keeps the FIRST maximum over the fixed candidate order, so the shortest
    interval wins a tie.  `check_pick` is the statement that it does.
    """
    best = 0
    for i in (1, 2):
        if three[i] > three[best]:
            best = i
    tied = sum(1 for v in three if v == three[best]) > 1
    return best, tied


def repeat_lines(read_one, prompt, say, where):
    """The repeat check made, printed, and then shown failing on a mutated input.

    `read_one` is a callable that returns the three logits of a prompt, so this works for
    a model with an adapter attached, for one without, and for the stub below.  The
    mutation moves ONE of the two stored triples by one unit before the comparison, which
    is the only way to make a check about bit-identity fail without a second model.
    """
    first = read_one(prompt)
    second = read_one(prompt)
    arm_a, tie_a = pick(first)
    arm_b, _tie_b = pick(second)
    check_repeat(first, second, arm_a, arm_b, where)
    say("  two passes over one row: %s" % (list(first),))
    say("                      and: %s" % (list(second),))
    say("  bit-identical, both chose %r (%.3f s), tie at the maximum: %s"
        % (P.CANDIDATES[arm_a], ARMS[arm_a], tie_a))
    moved = (first[0] + 1.0,) + tuple(first[1:])
    try:
        check_repeat(moved, second, arm_a, arm_b, where + ", mutated")
        say("  MUTATION DID NOT FIRE -- the repeat check cannot catch a moved logit")
        return False
    except Fail as exc:
        say("  one logit moved by 1 FIRES: %s" % exc)
    return True


# ---------------------------------------------------------------------------
# L, the decider
# ---------------------------------------------------------------------------
class L:
    """The model as a decider on the port: one forward pass per resolution, and nothing
    at a send.

    One instance per simulation, like F, because it keeps one interval per ordered pair:
    a single instance reused across runs would carry run N's pair intervals into run N+1.
    The MODEL is not per simulation -- reloading 10 GB of weights per run is not what
    "fresh" means here -- so the read-out is injected as a callable and only this object's
    memory is rebuilt.  `assert_empty` is what the caller checks that with.

    The pair is `(node, neighbor)`, and the node name is read for that and for nothing
    else.  It has to be: one decider object serves all five nodes of a run, so `neighbor`
    alone would merge five nodes' memories into one.  Nothing is rendered from the node
    name and no answer is a function of it; the rendered text is asserted to contain no
    name on every call.  `where["ordinal"]` is never read at all.

    At a send the interval the pair already holds is returned, and 6.0 s before the pair
    has ever resolved, which is the user's own decision 9.  No model pass is made at a
    send, so one decision costs one pass; `reads` is the count that says so.
    """

    def __init__(self, read_three, tok=None):
        """`read_three(prompt) -> (three logits)`; `tok` renders the prompt when given.

        `tok` is optional so that a stub read-out can take the rendered observation
        straight, which is what lets the send-time convention, the tie rule and the name
        assertion be checked on a machine with no card.
        """
        self.read_three = read_three
        self.tok = tok
        self.mem = {}            # (node, neighbor) -> the interval that pair last chose
        # 2026-11-24: the text of the LAST resolution-time decision, kept so that the
        # card-free self-check can compare what this object rendered with what the label
        # row renders to.  Without it that comparison was `text_of` against `text_of` on
        # the same five values -- an identity that could not fail for any row -- so an `L`
        # that read the wrong field of the observation was caught by nothing.
        self.last_text = None
        self.reads = 0           # forward passes made, i.e. resolution-time decisions
        self.sends = 0           # send-time decisions, none of which costs a pass
        self.ties = 0            # decisions whose maximum logit was exactly tied
        self.arms = {arm: 0 for arm in ARMS}   # how often each arm was chosen

    def assert_empty(self, where):
        """The per-pair memory holds nothing, as it must at a simulation's first call."""
        return check_memory_empty(self.mem, where)

    def __call__(self, obs):
        """One decision: the held interval at a send, one forward pass at a resolution."""
        w = obs["where"]
        pair = (w["node"], w["neighbor"])
        if w["kind"] != "state_poll_reschedule":
            self.sends += 1
            return self.mem.get(pair, SEND_START)
        text = text_of(obs["state"], obs["resolved"], obs["history"], w["ts_ms"],
                       w["neighbor"], "L at a resolution")
        self.last_text = text
        prompt = text if self.tok is None else prompt_of(self.tok, text)
        three = self.read_three(prompt)
        self.reads += 1
        index, tied = pick(three)
        check_pick(three, index, "L at a resolution")
        self.ties += 1 if tied else 0
        arm = ARMS[index]
        self.arms[arm] += 1
        self.mem[pair] = arm
        return arm


# ---------------------------------------------------------------------------
# the self-check: every card-free check, shown failing on a mutated input
# ---------------------------------------------------------------------------
def _stub_reader(answers):
    """A read-out that returns canned logits and counts its calls.

    Returns `(read, state)`, where `state["n"]` is the call count: the send-time
    convention's claim is that the count does not move at a send, and a stub is the only
    way to count that without a card.
    """
    state = {"n": 0}

    def read(_prompt):
        three = answers[min(state["n"], len(answers) - 1)]
        state["n"] += 1
        return three
    return read, state


def self_check(row, say):
    """Every check of this module that needs no card, each one shown failing.

    `row` is one real label row: the name assertion and the render round trip are about a
    real rendering and a mutation of it, not about a string typed here.  Returns True when
    every mutation produced a failure.
    """
    fired = []

    def shows(name, call):
        """Run a mutated input through a check and require it to refuse."""
        try:
            call()
        except (Fail, SystemExit, KeyError) as exc:
            say("  %-24s FIRES: %s" % (name, exc))
            return True
        say("  %-24s DID NOT FIRE -- this check cannot catch its own mutation" % name)
        return False

    obs = {"state": row["state"], "resolved": row["resolved"], "history": row["history"],
           "where": {"node": row["node"], "neighbor": row["neighbor"],
                     "kind": "state_poll_reschedule", "ts_ms": row["ts_ms"],
                     "ordinal": 0}}
    text = text_of(row["state"], row["resolved"], row["history"], row["ts_ms"],
                   row["neighbor"], "self_check")
    say("one real label row, seed %s k %s: %d characters rendered, no node name in it"
        % (row["seed"], row["k"], len(text)))

    # What L ITSELF rendered, against the text the row renders to.
    #
    # 2026-11-24: this was `text_of(...)` against `text_of(...)` on the same five values
    # of the same row, i.e. one call with its own arguments compared against itself, so
    # the line it printed ("equal: True") was an identity that no input could falsify,
    # while the output called it the check that covers the seam between the tuning and the
    # replay.  It now reads the text L produced on its own road through `L.__call__`, so
    # an L that read another field of the observation fails it, which is the mutation
    # "render_roundtrip/L" below.  What it still does NOT cover is the gzip-JSON round
    # trip, which is on both sides of it; `eval.py` part 1 closes that against the live
    # engine's own observation.  Positive first, because the mutation is only worth
    # anything if the unmutated comparison holds.
    dec_probe = L(lambda _p: (0.0, 0.0, 0.0))
    dec_probe(obs)
    check_render_roundtrip(dec_probe.last_text, text, "self_check, L against the row")
    say("the text L rendered from the port observation of that row: %d characters, equal"
        % len(dec_probe.last_text))
    say("to the text the row renders to: %s.  That is two CALL SITES over one row and NOT"
        % (dec_probe.last_text == text))
    say("the JSON round trip, which is on both sides of it: eval.py part 1 closes that")
    say("seam against the live engine's own captured observation.")

    # The send-time convention, as behaviour and not as a sentence.
    read, state = _stub_reader([(0.0, 0.0, 1.0)])
    dec = L(read)
    dec.assert_empty("a fresh L")
    send = dict(obs, where=dict(obs["where"], kind="state_poll"))
    before = state["n"]
    first_send = dec(send)
    check_no_read_at_send(before, state["n"], "the first send of a pair")
    resolved_arm = dec(obs)
    after_send = dec(send)
    check_no_read_at_send(1, state["n"], "the send after a resolution")
    say("a pair's first send answers %.3f s and makes no model pass" % first_send)
    say("its resolution answers %.3f s and makes one" % resolved_arm)
    say("the next send answers the same %.3f s and makes none, so these %d decisions cost"
        % (after_send, dec.sends + dec.reads))
    say("%d forward pass(es) in all" % dec.reads)
    if first_send != SEND_START:
        raise Fail("the first send of a pair answered %r, not %r" % (first_send,
                                                                     SEND_START))
    if after_send != resolved_arm:
        raise Fail("the send after a resolution answered %r, not the pair's %r"
                   % (after_send, resolved_arm))

    # The tie rule, as behaviour: an exact tie between the first and the last word.
    tied_three = (1.0, 0.0, 1.0)
    index, tied = pick(tied_three)
    check_pick(tied_three, index, "a tied read-out")
    say("logits %s pick index %d (%r, %.3f s) and count as a tie: %s"
        % (list(tied_three), index, P.CANDIDATES[index], ARMS[index], tied))

    say()
    say("the mutations:")
    fired.append(shows("card_free", lambda: check_card_free("1234, python, 10240 MiB",
                                                            "mutated")))
    fired.append(shows("seeds_tuning",
                       lambda: check_seeds_tuning([1, MAX_TUNE_SEED + 1], "mutated")))
    fired.append(shows("port.check_seeds",
                       lambda: PORT.check_seeds([PORT.MAX_SEED + 1], "mutated")))
    fired.append(shows("target_of", lambda: target_of(7.0)))
    fired.append(shows("arm_order",
                       lambda: check_arm_order(dict(zip(P.CANDIDATES, ARMS[::-1])), ARMS,
                                               "mutated")))
    fired.append(shows("candidate_ids",
                       lambda: check_candidate_ids([1, 2, 3], P.EXPECTED_IDS, "mutated")))
    # The name is spliced into a REAL rendering, at the position the polled peer is named,
    # which is where a rendering change would put one.
    fired.append(shows("no_node_name",
                       lambda: check_no_node_name(text.replace("p1", "h3", 1), "mutated")))
    # A word changed in the middle of the rendering, which is the shape a rendering change
    # has; appending a character would only ever be caught at the end of the string.
    fired.append(shows("render_roundtrip",
                       lambda: check_render_roundtrip(text, text.replace("peers", "peer",
                                                                         1), "mutated")))

    def _l_reads_another_field():
        """An L handed an observation whose stamp is not the row's own.

        The shape an L that read the wrong field of the observation would have, which is
        what the positive comparison above exists to catch and what it could not catch
        while both of its sides were built from the row.
        """
        other = L(lambda _p: (0.0, 0.0, 0.0))
        other(dict(obs, where=dict(obs["where"], ts_ms=obs["where"]["ts_ms"] + 60000)))
        check_render_roundtrip(other.last_text, text, "mutated")

    fired.append(shows("render_roundtrip/L", _l_reads_another_field))
    fired.append(shows("prompt_len",
                       lambda: check_prompt_len([len(text)], 8, "mutated", "characters")))
    fired.append(shows("adapter_dtype",
                       lambda: check_adapter_dtype(["torch.bfloat16"], "mutated")))
    fired.append(shows("unmerged/none",
                       lambda: check_unmerged(0, [], "mutated")))
    fired.append(shows("unmerged/merged",
                       lambda: check_unmerged(2, [False, True], "mutated")))
    # Index 2 is the LAST of two tied maxima, which is what a scan that kept the last
    # maximum would return and what `torch.argmax` is not read here to rule out.
    fired.append(shows("pick/tie", lambda: check_pick(tied_three, 2, "mutated")))
    fired.append(shows("pick/max", lambda: check_pick((0.0, 1.0, 0.0), 0, "mutated")))
    fired.append(shows("repeat/logits",
                       lambda: check_repeat((1.0, 2.0, 3.0), (1.0, 2.0, 4.0), 2, 2,
                                            "mutated")))
    fired.append(shows("repeat/arm",
                       lambda: check_repeat((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), 2, 1,
                                            "mutated")))
    fired.append(shows("memory_empty",
                       lambda: check_memory_empty({("h0", "h1"): ARMS[0]}, "mutated")))
    fired.append(shows("no_read_at_send", lambda: check_no_read_at_send(0, 1, "mutated")))
    # The digest list with one member dropped, which is the shape a narrowing of it has:
    # the combined digest stays plausible and stops catching the one change that member
    # was there to see.
    _parts = [(n, "%064d" % i) for i, (n, _p) in enumerate(CODE_FILES)]
    fired.append(shows("digest_cover",
                       lambda: check_digest_cover(_parts[1:],
                                                  [n for n, _p in CODE_FILES],
                                                  "mutated")))
    fired.append(shows("triple", lambda: check_triple([1.0, 2.0, 3.0], [1.0, 2.0, 4.0],
                                                      "a", "mutated")))
    # tune.txt's own repeat block, with candidate b's second pass moved: the parser is
    # what makes those six numbers a reading of the archive instead of a transcription,
    # so it is shown refusing an archive whose two passes disagree.
    def _block(name, epoch, second="19.5"):
        """One candidate's kept-epoch block of tune.txt, in the archive's own shape."""
        return ["3.1  CANDIDATE %s" % name.upper(),
                "KEPT %s epoch %d: the lowest held-out cross-entropy of the three "
                "(0.643996, 0.789407, 0.622174), the rule" % (name, epoch),
                "  cross-entropy 0.622174, agreement 408/568, ties 18 -- the three",
                "the repeat check, with the kept adapter attached (seed 71 k 72):",
                "  two passes over one row: [19.375, 18.75, 19.5]",
                "                      and: [19.375, 18.75, %s]" % second]

    _both = _block("b", 3) + _block("a", 3)
    fired.append(shows("tune.txt/two passes",
                       lambda: kept_from(_block("b", 3, "20.5") + _block("a", 3))))
    fired.append(shows("tune.txt/one candidate", lambda: kept_from(_block("b", 3))))
    fired.append(shows("tune.txt/the keep rule",
                       lambda: kept_from(_block("b", 1) + _block("a", 3))))
    if kept_from(_both)["a"]["triple"] != [19.375, 18.75, 19.5]:
        raise Fail("the tune.txt parser does not read its own archive shape")
    say()
    if all(fired):
        say("all %d mutations produced a failure" % len(fired))
    else:
        say("FAILURE OF THIS SECTION: %d of %d mutations did not produce a failure"
            % (len(fired) - sum(fired), len(fired)))
    return all(fired)


# The arm order is checked at import, where every user of this module pays for it once: it
# is an invariant of two frozen things (the prompt's words and the port's floats) and a
# script that got it wrong would map every decision to the wrong interval.
check_arm_order(P.SECONDS, ARMS, "decider.py at import")
