#!/usr/bin/env python3
"""The checks on the re-decision branch added to ../13/replay.py (2026-11-22).

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<this worktree>/src \
      /home/lab_shared/cefore-semindex/.venv/bin/python3 -B gate.py

Prints one PASS/FAIL line per check and exits non-zero on the first failure count
above zero. It runs before the s11a sweep and is re-run whenever the replay is
touched, so it is kept to a few seconds of one core.

WHAT IT CHECKS
  1. The knob OFF changes nothing. The edited replay and the replay as it stood before
     the edit are run on the same cells of each existing mode -- a fixed choices arm, a
     policy stage, a rule put on a donor run -- in Mode A and Mode B, and every event
     dict has to be equal. The edit is worthless if it moves a stage that is already
     measured against the testbed.
  2. The knob ON over ONE choice is the fixed arm. With `choices = [X]` the re-decision
     re-draws the only value there is, recomputes the due instant the send already set,
     and therefore must not move a single event: the two streams have to agree event for
     event apart from the `state_poll_reschedule` rows the knob adds. This is what says
     the re-decision costs nothing of its own; a difference here would be a timer or an
     RNG draw the branch moved, and every number of the stage would carry it.
  3. The knob ON over the stage's three choices behaves as the source does, in Mode A and
     Mode B: one reschedule row per resolved poll and never two for one poll, every
     resolution that got past the newest-poll guard wrote a row (counted inside the
     engine, since a resolution that wrote nothing leaves nothing in the log to count),
     every interval out of the choice set, the applied interval built from the poll's OWN
     send-time jitter draw, and exactly one choice-stream draw per send plus one per
     reschedule row.

WHAT IT CANNOT SHOW
  Nothing here is a measurement. Both sides are this replay, so a check says the branch
  is the mechanism the source describes, never that the testbed polls at these instants.
  The `state_poll_choice_redecide` runs themselves have no testbed measurement yet; that
  is what the s11a sweep and evidence/21/spec.md's pass line 5 are for.

BASE_REV names the revision the knob-off comparison reads ../13/replay.py from. The
default is a8add2c, the last commit before the edit: a default of HEAD would carry the
edit as soon as it was committed, and check 1 would then compare the file with itself.
The gate also refuses any base source that already holds the edit, for the same reason.
"""
import importlib.util
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EV13 = os.path.normpath(os.path.join(HERE, "..", "13"))
sys.path.insert(0, EV13)
import replay as R  # noqa: E402  the edited replay

assert R.__file__.startswith(EV13), R.__file__

BASE_REV = os.environ.get("BASE_REV", "a8add2c")
REL = "docs/wayfinder/semindex-stage2/evidence/13/replay.py"

# One fixed choices arm, one policy stage and one rule, on cells the store holds (the
# s6 family is p 0.2, 0.4, 0.5 and 0.6, seeds 1 to 20). The rule has no stage of its
# own anywhere in the store, so it is put on a donor run the way evidence/18 puts its
# members on one; that is the same code path a logged rule stage would take.
OFF_CELLS = [("choices arm", "s6b3p695", 0.4, None),
             ("policy stage", "s6short", 0.2, None),
             ("rule on a donor", "s6b10p043", 0.4,
              {"i_min": 3.695, "m": 2.0, "i_max": 10.043, "jf": 0.1})]
OFF_SEEDS = (1, 2)

IDENT_STAGE, IDENT_P = "s6b3p695", 0.4
IDENT_SEEDS = (1, 2, 3)
CHOICES = [3.695, 6.0, 10.043]          # the s11a arms
JF = 0.1                                # the s11a choice jitter


def flat(events, drop_reschedule=False):
    """The event stream as comparable tuples, node by node in name order.

    `repr` rather than the value itself so that a float, a list and a dict all compare
    by their written form: two events are equal here exactly when the two replays would
    write the same line into events.jsonl.
    """
    out = []
    for node in sorted(events):
        for e in events[node]:
            if drop_reschedule and e.get("kind") == "state_poll_reschedule":
                continue
            out.append((node, tuple(sorted((k, repr(v)) for k, v in e.items()))))
    return out


def first_difference(left, right):
    """The first index where two flattened streams differ, with both rows."""
    for i, (u, v) in enumerate(zip(left, right)):
        if u != v:
            return f"    at {i}\n      base {u}\n      edit {v}"
    return f"    lengths {len(left)} and {len(right)}"


def load_base():
    """The replay as of BASE_REV, imported as a second module object.

    Written into a temporary directory and loaded by path: importing it under its own
    name would hand back the edited module that is already in sys.modules, and the
    check would then compare a module with itself.
    """
    root = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, check=True).stdout.strip()
    src = subprocess.run(["git", "-C", root, "show", f"{BASE_REV}:{REL}"],
                         capture_output=True, text=True, check=True).stdout
    with open(os.path.join(EV13, "replay.py"), encoding="utf-8") as fh:
        if fh.read() == src:
            return None
    tmp = tempfile.TemporaryDirectory()
    path = os.path.join(tmp.name, "replay_base.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("replay_base", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["replay_base"] = module
    spec.loader.exec_module(module)
    module._gate_tmp = tmp          # keep the directory alive as long as the module is
    return module


class Probe(R._Engine):
    """The edited engine, counting choice-stream draws and keeping each poll's jitter draw.

    Both are read by check 3 and neither can be recovered from the event log: the draw
    count is what says the re-decision draws once and once only, and the send-time
    jitter draw is what the reschedule row has to be built from. The override is a pure
    pass-through -- it reads `pending` before the base method resolves the poll and
    changes nothing -- so the events it produces are the events `_sim_run` produces.
    """

    def __init__(self, run, mode, lags, sim_rng, rule=None):
        super().__init__(run, mode, lags, sim_rng, rule)
        self.draws = 0
        self.resolutions = 0          # entries to resolve that got past the newest-poll guard
        self.jitter_u = {}            # (node, neighbour, poll_epoch_ms) -> send-time draw
        for node in self.nodes.values():
            node.choice_rng = self._counted(node.choice_rng)

    def _counted(self, rng):
        def draw():
            self.draws += 1
            return rng()
        return draw

    def resolve(self, t, node, nb, sent_at, status, empty=None):
        pend = self.nodes[node].pending.get(nb)
        if pend is not None and pend[0] == sent_at and not pend[2]:
            # The guard's own condition, read before the base method flips pend[2]. A
            # resolution that wrote no reschedule row leaves no trace in the event log,
            # so counting here is the only way check 3 can see one go missing.
            self.resolutions += 1
            self.jitter_u[(node, nb, int(round(sent_at)))] = pend[1]
        return super().resolve(t, node, nb, sent_at, status, empty=empty)


def say(ok, text):
    """One check's line; returns 1 when it failed, so main can total the failures."""
    print(f"{'PASS' if ok else 'FAIL'} {text}")
    return 0 if ok else 1


def check_off(base):
    """Check 1: the edited replay equals BASE_REV's on every mode that exists today."""
    bad = 0
    for label, stage, p, rule in OFF_CELLS:
        for mode in ("A", "B"):
            diffs, n_events = [], 0
            for seed in OFF_SEEDS:
                run, brun = R.load_run(stage, p, seed), base.load_run(stage, p, seed)
                # The frozen scripts (../20/predict.py among them) build their run
                # dicts by hand and were written before the key existed, so what keeps
                # them on the old path is the DEFAULT of `run.get("redecide", ...)` in
                # the engine, not any value load_run wrote. Dropping the key here is the
                # only way this check reaches that default; left in, a wrong default
                # would stay invisible and a frozen script would quietly re-decide.
                run.pop("redecide")
                if mode == "A":
                    got = R._sim_run(run, "A", lags=R.logged_lags(run), rule=rule)
                    want = base._sim_run(brun, "A", lags=base.logged_lags(brun), rule=rule)
                else:
                    seeded = R.rep_seed(stage, p, seed, 0)
                    got = R._sim_run(run, "B", sim_rng=random.Random(seeded), rule=rule)
                    want = base._sim_run(brun, "B", sim_rng=random.Random(seeded), rule=rule)
                left, right = flat(want), flat(got)
                n_events += len(right)
                if left != right:
                    diffs.append((seed, first_difference(left, right)))
            bad += say(not diffs, f"knob off, {label:<16} {stage:<11} p {p} seeds "
                                  f"{OFF_SEEDS[0]} to {OFF_SEEDS[-1]} mode {mode}: "
                                  f"{n_events} events, every dict equal to {BASE_REV}'s")
            for seed, text in diffs:
                print(f"  seed {seed}\n{text}")
    return bad


def check_identity():
    """Check 2: `choices = [X]` with the knob on is the arm X, plus its reschedule rows."""
    bad = 0
    for x in CHOICES:
        for mode in ("A", "B"):
            diffs, rows = [], 0
            for seed in IDENT_SEEDS:
                run = R.load_run(IDENT_STAGE, IDENT_P, seed)
                run.update(choices=[x], choice_jitter=JF, redecide=False)
                on = dict(run, redecide=True)
                if mode == "A":
                    lags = R.logged_lags(run)
                    off_ev, on_ev = R._sim_run(run, "A", lags=lags), R._sim_run(on, "A", lags=lags)
                else:
                    seeded = R.rep_seed(IDENT_STAGE, IDENT_P, seed, 0)
                    off_ev = R._sim_run(run, "B", sim_rng=random.Random(seeded))
                    on_ev = R._sim_run(on, "B", sim_rng=random.Random(seeded))
                rows += sum(1 for evs in on_ev.values() for e in evs
                            if e.get("kind") == "state_poll_reschedule")
                left, right = flat(off_ev), flat(on_ev, drop_reschedule=True)
                if left != right:
                    diffs.append((seed, first_difference(left, right)))
            bad += say(not diffs, f"one choice {x:>6} mode {mode}: seeds "
                                  f"{IDENT_SEEDS[0]} to {IDENT_SEEDS[-1]}, {rows} reschedule "
                                  "rows added and nothing else moved")
            for seed, text in diffs:
                print(f"  seed {seed}\n{text}")
    return bad


def check_three():
    """Check 3: the three choices with the knob on, against what the source does."""
    per_poll, intervals, jitter, draws, resolved = [], [], [], [], []
    # Both modes, because the row-level properties below are the ones the stage's own
    # numbers are built out of and Mode B is the stream the prediction draws with: a
    # branch that behaved differently under the drawn tick lags would otherwise be seen
    # by the knob-off and single-choice checks alone, neither of which reads a row.
    for mode, seed in [(m, s) for m in ("A", "B") for s in IDENT_SEEDS]:
        run = R.load_run(IDENT_STAGE, IDENT_P, seed)
        run.update(choices=list(CHOICES), choice_jitter=JF, redecide=True)
        if mode == "A":
            engine = Probe(run, "A", R.logged_lags(run), None)
        else:
            engine = Probe(run, "B", None, random.Random(R.rep_seed(IDENT_STAGE, IDENT_P, seed, 0)))
        events = engine.go()
        sends, resched = {}, []
        for node, evs in events.items():
            for e in evs:
                if e.get("kind") == "state_poll":
                    sends[(node, e["action"]["neighbor"], e["ts_ms"])] = e["action"]
                elif e.get("kind") == "state_poll_reschedule":
                    resched.append((node, e["action"]))
        seen = set()
        for node, action in resched:
            key = (node, action["neighbor"], action["poll_epoch_ms"])
            per_poll.append(key in sends and key not in seen)
            seen.add(key)
            intervals.append(action["interval_s"] in CHOICES)
            u = engine.jitter_u.get(key)
            send = sends.get(key)
            if u is None or send is None:
                jitter.append(False)
                continue
            # The row's own arithmetic, from the draw the SEND made and kept: the
            # re-decision must not draw a second jitter value. The send row is read
            # back as well, so the draw that was kept is also the draw that was logged.
            jitter.append(
                action["applied_interval_s"] == action["interval_s"] * (1.0 + JF * (2.0 * u - 1.0))
                and abs(send["applied_interval_s"] / send["interval_s"]
                        - action["applied_interval_s"] / action["interval_s"]) < 1e-12)
        draws.append((engine.draws, len(sends) + len(resched), len(sends), len(resched)))
        resolved.append((engine.resolutions, len(resched)))
    bad = say(all(per_poll), f"three choices: {len(per_poll)} reschedule rows, each one "
                             "on a state_poll of its own pair and no poll re-decided twice")
    # The row-side checks all start from a row that exists, so a resolution that took
    # the branch and wrote nothing would pass every one of them. The counter is read
    # independently of the log for exactly that case.
    bad += say(all(r == n for r, n in resolved),
               f"three choices: every one of the {sum(r for r, _n in resolved)} resolved "
               "newest polls wrote a reschedule row, none wrote none")
    bad += say(all(intervals), f"three choices: every interval of the {len(intervals)} "
                               f"reschedule rows is one of {CHOICES}")
    bad += say(all(jitter), "three choices: every reschedule row applies the jitter draw "
                            "its own poll was sent with")
    bad += say(all(d == w for d, w, _s, _r in draws),
               "three choices: choice-stream draws equal sends plus reschedule rows "
               + ", ".join(f"{d} = {s} + {r}" for d, _w, s, r in draws))
    return bad


def main():
    base = load_base()
    if base is None:
        print(f"FAIL base source: {BASE_REV}:{REL} already carries the re-decision "
              "branch, so the knob-off comparison would compare the file with itself. "
              "Re-run with BASE_REV set to the revision before the edit.")
        return 1
    print(f"Base for the knob-off comparison: {BASE_REV}:{REL}\n")
    bad = check_off(base) + check_identity() + check_three()
    print(f"\n{'every check passed' if not bad else str(bad) + ' CHECK(S) FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
