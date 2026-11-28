#!/usr/bin/env python3
"""L (candidate b) on the prediction's run dicts, on the GPU machine: one row per run.

    cd $ROOT && env PYTHONDONTWRITEBYTECODE=1 STORE=$ROOT/store PYTHONPATH=$ROOT/repo/src \
        HF_HUB_OFFLINE=1 ./.venv/bin/python -B \
        repo/docs/wayfinder/semindex-stage2/evidence/25/lpredict.py $ROOT \
        --input IN.json --input-sha SHA --out OUT.csv (--dry | --held-out-once) [--k 4]

    --dry            the check runs only: the 24 runs of candidate b that ../23/eval_dry.csv
                     and ../23/eval.csv already hold, re-scored from the input file.
    --held-out-once  the check runs, and only if every one equals the scoring's row, the
                     1600 held-out runs (seeds 121 to 160 x 40 streams).
    --stub           no card: a hash of the prompt in place of the model (lserver.py's
                     stub), one process, check runs only.  For rehearsing the plumbing on
                     a machine without the model; its rows mean nothing.
    The output file is refused if it exists, so a held-out run cannot be repeated by
    accident.

WHY THIS FILE AND NOT ../23/eval.py
  L must be scored exactly as the candidate was: ../23/eval.py's reference read-out (one
  process alone on the card, then exits), K workers that each re-read the same 64 label
  rows and are compared bit for bit (`torch.equal`) against it before any of them
  scores, `decider.L` built fresh per run under the first-call memory assertion.  But
  eval.py's parent is module-level code bound to seeds 81 to 100 and to
  `label.run_dict`, which refuses the held-out seeds.  So eval.py is not imported (an
  import would start its real run): its source up to the banner that opens its child
  modes is executed into a namespace -- every definition it has, with nothing of its
  parent -- after its sha256 is checked against the one ../23/eval.txt printed.  The
  child functions (`child_reference`, `child_worker`) run unchanged; the single
  replacement is the `one_run` they call, which takes the run dict from the input file
  and simulates through heldout.py (the same engine, stream and row as predict.py's six
  CPU stages).  What is written here is the parent loop eval.py has at module level
  (spawn, gate, go, collect), for one decider.

WHAT IT DOES NOT SHOW
  Anything about the read-out on this run beyond what the gates say: the 64-row
  bit-identity gate per worker, the repeat row against tune.txt's triple, and the 24
  check runs against the scoring's own b rows.  predict.py --report re-checks the check
  runs, the input digest, the model digest and the code digest from this file's output.
"""
import os
import sys

sys.dont_write_bytecode = True
# The far checkout carries __pycache__ directories from earlier runs and the digests below
# are over source bytes, so cache reads are switched off as lserver.py does.
sys.pycache_prefix = os.devnull

import hashlib   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EVID = os.path.dirname(HERE)
EV23 = os.path.join(EVID, "23")
EVAL_PY = os.path.join(EV23, "eval.py")
EVAL_SHA = "629c4f0986ec25bb67268ada17c5636e0ee157d91acbf97ad9810fb4ca3a4582"
BANNER = "# the child modes, before the parent's own report starts"
L_MODEL_SHA = "6a39ca005019699bef789b6811abd006da7f9445708894ea444d9e8690441cc1"
NAME_CHECK = "b"           # the scoring's row name for candidate b
NAME_HELD = "s12l"         # the testbed stage's name

with open(EVAL_PY, "rb") as _fh:
    _EVAL_BYTES = _fh.read()
if hashlib.sha256(_EVAL_BYTES).hexdigest() != EVAL_SHA:
    raise SystemExit("lpredict: ../23/eval.py is not the file ../23/eval.txt ran")

for _p in (EV23, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import prefill as P    # noqa: E402  first: thread caps, offline HF, ROOT, sys.path
import decider as D    # noqa: E402
import heldout as HO   # noqa: E402

import json            # noqa: E402
import subprocess      # noqa: E402
import time            # noqa: E402

HO.check_pins()


def eval_definitions():
    """Every definition of ../23/eval.py above its child modes, in a namespace of its own.

    Executed under eval.py's own path so that its `__file__`-relative digest
    (`CODE_PARTS`) and its error messages name eval.py, and cut at the banner so that not
    a line of its parent body runs.
    """
    src = _EVAL_BYTES.decode("utf-8")
    at = src.index(BANNER)
    cut = src.rindex("\n# ====", 0, at)
    ns = {"__file__": EVAL_PY, "__name__": "eval23_definitions", "__builtins__": __builtins__}
    exec(compile(src[:cut], EVAL_PY, "exec"), ns)
    return ns


NS = eval_definitions()
ADAPTER = os.path.join(NS["ADAPTERS"], "b")
# The reference child reads out every model decider the namespace names; here that is b.
NS["MODEL_DECIDERS"] = (("b", ADAPTER),)

ARGV = sys.argv[1:]


def arg(flag, default=None):
    """The value after `flag` on the command line, or `default`."""
    return ARGV[ARGV.index(flag) + 1] if flag in ARGV else default


INPUT = arg("--input")
INPUT_SHA = arg("--input-sha")
RUNS = {}
STATE = {"model_sha": None}


def load_input():
    """The input file, refused unless its digest is the one the caller named."""
    if not INPUT or not INPUT_SHA:
        raise SystemExit("lpredict: --input and --input-sha are required")
    got = HO.sha_of(INPUT)
    if got != INPUT_SHA:
        raise SystemExit("lpredict: %s has sha256 %s, not %s" % (INPUT, got, INPUT_SHA))
    payload, runs = HO.read_runs(INPUT)
    if payload.get("format") != "ev25-predict-runs-1":
        raise SystemExit("lpredict: %s is not a predict.py input file" % INPUT)
    RUNS.update(runs)
    return payload


def one_run(name, seed, rep, make_decider, k):
    """eval.py's `one_run` contract on a run dict from the input file.

    Same checks as eval.py's (the choice set, the first-call memory assertion, the
    counters), same row; the run dict is the input file's instead of `label.run_dict`'s,
    and the engine is heldout's, whose guard admits 121 to 160 only when armed.
    """
    run = RUNS[seed]
    where = "%s seed %d stream %d" % (name, seed, rep)
    NS["check_choices"](run, "the run dict of " + where)
    inner = make_decider()
    wrapper = NS["FirstCallEmpty"](inner, where, NS["keeps_memory"](inner))
    row = HO.one_run(name, run, rep, wrapper, inner, k)
    row["model_sha"] = STATE["model_sha"]
    return row


NS["one_run"] = one_run

# ---------------------------------------------------------------------------
# the child modes, which eval.py's own functions run
# ---------------------------------------------------------------------------
if "--reference" in ARGV:
    NS["child_reference"](arg("--reference"))
    raise SystemExit(0)
if "--worker" in ARGV:
    load_input()
    if "--held-out-once" in ARGV:
        HO.arm_held_out()
    STATE["model_sha"] = L_MODEL_SHA
    _at = ARGV.index("--worker")
    NS["child_worker"](ARGV[_at + 1], int(ARGV[_at + 2]))
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# the parent
# ---------------------------------------------------------------------------
def say(text=""):
    """One line of the report, flushed."""
    print(text, flush=True)


def compare_checks(rows, checks):
    """Each check run's simulated columns against the scoring's own strings."""
    want = {(c["seed"], c["stream"]): c["expect"] for c in checks}
    bad = []
    for row in rows:
        got = HO.sim_strings(row)
        exp = want[(row["seed"], row["stream"])]
        for col in HO.SIM_COLUMNS:
            if got[col] != exp[col]:
                bad.append("seed %d stream %d %s: %r against the scoring's %r"
                           % (row["seed"], row["stream"], col, got[col], exp[col]))
    return bad


def _abort_phase(procs, tmp, name, msg):
    """Raise `msg`, but only after every worker of this phase has stopped and been reaped.

    2026-11-28 fix: every raise below used to leave surviving workers running -- still
    loaded on the GPU until they finished their own share, hit GATE_DEADLINE_S, or (once
    past "go") ran to completion on their own -- which is what later made the next phase's
    D.check_card_free refuse a card one of this phase's own workers was still holding.
    stop_<name> gives a worker still waiting at the gate a chance to notice and exit on its
    own; SIGTERM reaches the rest (a worker already scoring, or one already dead, for which
    a signal is simply a no-op); and every one of them is waited on before the SystemExit,
    so the parent never returns control while a sibling worker is still running.
    """
    open(os.path.join(tmp, "stop_%s" % name), "w").write("1")
    for proc in procs:
        proc.terminate()
    for proc in procs:
        proc.wait()
    raise SystemExit(msg)


def phase(name, runs, k, tmp, ref, cand_ids, held):
    """eval.py's parent loop for one decider: spawn, gate, go, collect."""
    import torch
    shares = [[list(runs[i]) for i in range(len(runs)) if i % k == w] for w in range(k)]
    for w in range(k):
        with open(os.path.join(tmp, "job_%d.json" % w), "w", encoding="utf-8") as fh:
            json.dump({"decider": name, "adapter": ADAPTER, "runs": shares[w],
                       "cand_ids": cand_ids, "k": k}, fh)
    procs, logs = [], []
    for w in range(k):
        log = open(os.path.join(tmp, "worker_%s_%d.log" % (name, w)), "w+", encoding="utf-8")
        err = open(os.path.join(tmp, "worker_%s_%d.err" % (name, w)), "w", encoding="utf-8")
        logs.append((log, err))
        cmd = [sys.executable, "-B", os.path.abspath(__file__), P.ROOT, "--worker", tmp,
               str(w), "--input", INPUT, "--input-sha", INPUT_SHA]
        procs.append(subprocess.Popen(cmd + (["--held-out-once"] if held else []),
                                      stdout=log, stderr=err, text=True))
    deadline = time.time() + NS["GATE_DEADLINE_S"]
    while True:
        ready = [w for w in range(k)
                 if os.path.exists(os.path.join(tmp, "gate_%s_%d.json" % (name, w)))]
        dead = [w for w in range(k) if procs[w].poll() is not None and w not in ready]
        if dead:
            _abort_phase(procs, tmp, name,
                        "FAIL: worker(s) %s of %s exited before the gate; logs in %s"
                        % (dead, name, tmp))
        if len(ready) == k:
            break
        if time.time() > deadline:
            _abort_phase(procs, tmp, name, "FAIL: the workers of %s did not reach the gate" % name)
        time.sleep(0.1)
    ok = True
    for w in range(k):
        got = torch.load(os.path.join(tmp, "gate_%s_%d.pt" % (name, w)))
        try:
            n = NS["check_gate"](ref, got, "worker %d of %s against the reference" % (w, name))
            say("  worker %d: %d x 3 logits bit-identical to the reference" % (w, n // 3))
        except NS["Fail"] as exc:
            ok = False
            say("  GATE FAILED: %s" % exc)
    if not ok:
        _abort_phase(procs, tmp, name,
                    "FAIL: the bit-identity gate refused %s; nothing was scored. The "
                    "plan's fallback is --k 1" % name)
    open(os.path.join(tmp, "go_%s" % name), "w").write("1")
    t0 = time.time()
    # 2026-11-28 fix: waiting on procs in order and raising on the first non-zero one used
    # to leave later workers running unobserved after the parent exited. Waiting on all of
    # them first closed that, but let the survivors score for up to an hour after one had
    # already died, for a phase whose rows would be thrown away. So the workers are polled,
    # and the first non-zero exit stops the rest at once through _abort_phase.
    while True:
        returncodes = [proc.poll() for proc in procs]
        bad = [w for w, rc in enumerate(returncodes) if rc not in (None, 0)]
        if bad:
            _abort_phase(procs, tmp, name,
                        "FAIL: worker(s) %s of %s exited non-zero (%s); logs in %s"
                        % (bad, name, [returncodes[w] for w in bad], tmp))
        if all(rc == 0 for rc in returncodes):
            break
        time.sleep(1.0)
    rows, passes = [], 0
    for w in range(k):
        logs[w][0].seek(0)
        for line in logs[w][0].read().rstrip("\n").split("\n"):
            say("  " + line)
        logs[w][0].close()
        logs[w][1].close()
        with open(os.path.join(tmp, "res_%s_%d.json" % (name, w)), encoding="utf-8") as fh:
            res = json.load(fh)
        rows.extend(res["rows"])
        passes += res["passes"]
    NS["check_complete"]([(r["seed"], r["stream"]) for r in rows], [tuple(x) for x in runs],
                         "the rows of %s" % name)
    wall = time.time() - t0
    say("  %s: %d runs, %d forward passes in %.1f s (%.1f passes/s at K = %d)"
        % (name, len(rows), passes, wall, passes / wall if wall else 0.0, k))
    return sorted(rows, key=lambda r: (r["seed"], r["stream"]))


def main():
    """Check the card, the adapter and the input; the check runs; then, if armed, L."""
    t_start = time.time()
    held = "--held-out-once" in ARGV
    stub = "--stub" in ARGV
    if held == ("--dry" in ARGV):
        raise SystemExit("give exactly one of --dry, --held-out-once")
    if held and stub:
        raise SystemExit("--stub never runs the held-out seeds")
    out = arg("--out")
    if not out or os.path.exists(out):
        raise SystemExit("lpredict: --out must name a file that does not exist yet")
    if held:
        # 2026-11-28 fix: the once-only guard above is keyed on --out's name alone, so a
        # held-out run repeated under a new filename bypassed it completely and resimulated
        # all 1600 held-out runs. predict_l.csv is the one name PREDICT_RUN.md ever passes
        # here, so a different basename is refused outright. A predict_l.csv in another
        # directory still passes this check; such a run cannot change a number, because
        # predict.py --report refuses rows whose input or code digest differs and the rows of
        # an identical input and code are the same rows. The .tmp directory is also
        # checked up front, before the card and the model are even touched: a worker that
        # already wrote partial held-out rows into it (a crash after some workers finished,
        # before any of them write) must be looked at, not silently reused or overwritten.
        if os.path.basename(out) != "predict_l.csv":
            raise SystemExit("lpredict: --held-out-once requires --out's basename to be "
                             "predict_l.csv; any other name bypasses the once-only guard")
        if os.path.exists(out + ".tmp"):
            raise SystemExit("lpredict: %s exists; an earlier attempt may have left partial "
                             "held-out rows in it, and it is looked at before it is removed"
                             % (out + ".tmp"))
    k = int(arg("--k", "4"))
    payload = load_input()
    checks = payload["check"]
    check_runs = [(c["seed"], c["stream"]) for c in checks]
    held_runs = [tuple(x) for x in payload["held_out"]]
    if held:
        if len(held_runs) != len(HO.HELD_OUT) * HO.NREP:
            raise SystemExit("lpredict: the input file holds %d held-out runs" % len(held_runs))
        HO.arm_held_out()
    code_sha = HO.combined(HO.L_CODE)
    say("lpredict.py %s, input %s sha256 %s" % ("held-out" if held else "dry", INPUT, INPUT_SHA))
    say("code digest %s over %s" % (code_sha, ", ".join(HO.L_CODE)))
    say("../23/eval.py %s, its definitions loaded without its parent body" % EVAL_SHA)
    say("check runs %d, held-out runs %s" % (len(check_runs), len(held_runs) if held else "not run"))

    if stub:
        import lserver as LS
        STATE["model_sha"] = "stub"
        rows = [NS["one_run"](NAME_CHECK, s, rep, lambda: D.L(LS.stub_read_three), 1)
                for s, rep in check_runs]
        bad = compare_checks(rows, checks)
        say("STUB: %d check rows simulated; %d columns differ from the scoring's b rows, "
            "as a stub must (its logits are a hash)" % (len(rows), len(bad)))
        HO.write_rows(out, rows, code_sha, INPUT_SHA)
        say("wrote %s" % out)
        return 0

    import lserver as LS
    apps = D.nvidia_compute_apps()
    D.check_card_free(apps, "lpredict.py at start")
    LS.check_versions()
    LS.check_adapter_digests(ADAPTER)
    STATE["model_sha"] = NS["model_sha"](ADAPTER)
    if STATE["model_sha"] != L_MODEL_SHA:
        raise SystemExit("lpredict: adapter digest %s, not candidate b's" % STATE["model_sha"])
    import torch
    free = torch.cuda.mem_get_info()[0] / 2**30
    NS["check_fits"](k, free, "lpredict.py at start")
    say("card free, %.2f GB; K = %d workers fit; adapter b %s" % (free, k, STATE["model_sha"]))

    tmp = out + ".tmp"
    os.makedirs(tmp)
    _all, half = NS["tuning_half_rows"]()
    gate = NS["gate_rows"](half)
    rep_row = NS["repeat_row"](half)
    tok = D.load_tokenizer()
    say(P.prepare_padding(tok))
    prompts = [P.chat_text(tok, D.text_of(r["state"], r["resolved"], r["history"],
                                         r["ts_ms"], r["neighbor"], "the gate rows"))
               for r in gate]
    repeat_prompt = P.chat_text(tok, D.text_of(rep_row["state"], rep_row["resolved"],
                                               rep_row["history"], rep_row["ts_ms"],
                                               rep_row["neighbor"], "the repeat row"))
    D.check_prompt_len([len(P.ids_of(tok, p)) for p in prompts], 1024, "the gate prompts")
    with open(os.path.join(tmp, "gate_prompts.json"), "w", encoding="utf-8") as fh:
        json.dump(prompts, fh)
    with open(os.path.join(tmp, "repeat_prompt.txt"), "w", encoding="utf-8") as fh:
        fh.write(repeat_prompt)
    with open(os.path.join(tmp, "reference.err"), "w", encoding="utf-8") as err:
        rc = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), P.ROOT,
                             "--reference", tmp], stdout=subprocess.PIPE, stderr=err, text=True)
    for line in rc.stdout.rstrip("\n").split("\n"):
        say("  " + line)
    if rc.returncode != 0:
        raise SystemExit("FAIL: the reference child exited %d" % rc.returncode)
    with open(os.path.join(tmp, "ref.json"), encoding="utf-8") as fh:
        ref = json.load(fh)
    per = ref["per"]["b"]
    if not D.repeat_lines(NS["stored_reader"](per["first"], per["second"]), repeat_prompt,
                          lambda text="": say("  " + text), "the repeat check of b"):
        raise SystemExit("FAIL: the repeat check's own mutation did not fire")
    D.check_triple(per["first"], D.tuned_triples()["b"], "b",
                   "the reference read-out of seed %d k %d" % (rep_row["seed"], rep_row["k"]))
    say("  repeat row seed %d k %d: %s, equal to tune.txt's triple for b"
        % (rep_row["seed"], rep_row["k"], per["first"]))
    ref_t = torch.load(os.path.join(tmp, "ref_b.pt"))

    say("\nthe check runs")
    rows = phase(NAME_CHECK, check_runs, k, tmp, ref_t, ref["cand_ids"], False)
    bad = compare_checks(rows, checks)
    if bad:
        for line in bad[:10]:
            say("  DIFFERENT " + line)
        raise SystemExit("FAIL: %d check column(s) differ from the scoring's b rows; no "
                         "held-out run was started" % len(bad))
    say("  all %d check runs equal the scoring's b rows in every simulated column" % len(rows))
    if held:
        say("\nthe held-out runs, seeds 121 to 160 x %d streams" % HO.NREP)
        rows += phase(NAME_HELD, held_runs, k, tmp, ref_t, ref["cand_ids"], True)
    HO.write_rows(out, rows, code_sha, INPUT_SHA)
    say("\nwrote %s, %d rows, sha256 %s" % (out, len(rows), HO.sha_of(out)))
    say("wall clock %.1f min" % ((time.time() - t_start) / 60.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
