#!/usr/bin/env python3
"""Build and inspect the s11a fitted decision rows (seeds 1 to 80) for candidates a and b.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 STORE=$ROOT/store PYTHONPATH=$ROOT/repo24/src \
      $ROOT/.venv/bin/python -B rows24.py

Importable: `build_all()` builds the rows without running `main`, so `learn24.py` uses
the exact same row builder and guards.  Asserts `1 <= seed <= 80`, `action_prob == 1/3`,
`1.0 <= tau <= 14.0`, and that every fitted row (`10000 <= t_rel_ms <= 102000`) has two
further decisions of its pair and is non-terminal.
"""
import collections
import hashlib
import os
import statistics
import sys

for _name, _value in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
                      ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_name, _value)

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if EV22 not in sys.path:
    sys.path.insert(0, EV22)

import charges as CH    # noqa: E402
import features as FE   # noqa: E402
import m24              # noqa: E402
import port as PORT     # noqa: E402
import precheck as PRE  # noqa: E402

P_CELLS = ("0.2", "0.4")
ARMS = CH.ARMS
CELLS = CH.CELLS
MAX_LEARN_SEED = 80
FIT_SEEDS = tuple(range(1, 71))
HELD_SEEDS = tuple(range(71, 81))
ALL_SEEDS = tuple(range(1, MAX_LEARN_SEED + 1))


class Fail(Exception):
    """A check or row invariant that did not hold."""


def say(text=""):
    """Print one line of output."""
    print(text, flush=True)


def head(title):
    """Print a section heading."""
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


def file_sha256(path):
    """Hex sha256 digest of a file."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def check_learn_seeds(seeds, where="check_learn_seeds"):
    """Refuse any seed that is not a whole number in 1 to 80, and call `port.check_seeds`."""
    bad = sorted({float(s) for s in seeds
                  if float(s) != int(float(s))
                  or not 1 <= int(float(s)) <= MAX_LEARN_SEED})
    if bad:
        raise Fail("%s: learning seeds must be whole numbers 1 to %d; refused %s"
                   % (where, MAX_LEARN_SEED, bad[:8]))
    PORT.check_seeds(seeds, where)
    return sorted({int(s) for s in seeds})


def check_action_prob(rows, where="check_action_prob"):
    """Assert every row carries `action_prob == 1/3` by exact float equality."""
    want = 1.0 / 3.0
    bad = [r for r in rows if r.get("action_prob") != want]
    if bad:
        got = bad[0].get("action_prob")
        raise Fail("%s: %d row(s) with action_prob != 1/3, first %r" % (where, len(bad), got))
    return len(rows)


def check_window_rows(rows, where="check_window_rows"):
    """Assert every fitted row is non-terminal (so it has two further decisions of its pair)."""
    for r in rows:
        if r.get("next_pos") is None:
            raise Fail("%s: fitted row p%s seed %d %s pos %d is terminal"
                       % (where, r["p"], r["seed"], r["pair"], r["pos"]))
    m24.check_successor_key(rows, where=where)
    m24.check_tau_range(rows, where=where)
    return len(rows)


def load_all_rows(p_cells=P_CELLS, seeds=ALL_SEEDS):
    """Load owned decision periods for `(p_cells, seeds)` with `t_rel_ms`, `sends_in_win`,
    `action_prob`, and per-pair last-decision instants relative to `t0`.
    """
    check_learn_seeds(seeds, "load_all_rows")
    all_rows = []
    last_decision_rel = {p: [] for p in p_cells}
    totals = {p: {"decisions": 0, "owned": 0, "tail": 0} for p in p_cells}
    for p in p_cells:
        for seed in seeds:
            events, t0 = CH.load_run(p, seed, max_seed=MAX_LEARN_SEED)
            probs = {}
            for node, evs in events.items():
                for e in evs:
                    if e.get("event") == "decision" and e.get("kind") == "state_poll_reschedule":
                        key = (node, e["action"]["neighbor"], float(e["ts_ms"]))
                        if "action_prob" not in e:
                            raise Fail("p%s seed %d %r missing action_prob" % (p, seed, key))
                        probs[key] = float(e["action_prob"])

            table, run = PRE.table_of_run(p, seed, max_seed=MAX_LEARN_SEED)
            by_pair_table = collections.defaultdict(list)
            for row in table:
                by_pair_table[row["pair"]].append(row)

            for pair, chain in sorted(run["periods"].items()):
                owned_pers = [per for per in chain if per["kind"] == "owned"]
                tail_pers = [per for per in chain if per["kind"] == "tail"]
                totals[p]["owned"] += len(owned_pers)
                totals[p]["tail"] += len(tail_pers)
                totals[p]["decisions"] += len(owned_pers) + len(tail_pers)
                for per in tail_pers:
                    last_decision_rel[p].append(float(per["start"]) - float(run["t0"]))
                pair_rows = by_pair_table.get(pair, [])
                if len(pair_rows) != len(owned_pers):
                    raise Fail("p%s seed %d %s: %d table rows != %d owned periods"
                               % (p, seed, pair, len(pair_rows), len(owned_pers)))
                for row, per in zip(pair_rows, owned_pers):
                    row["t_rel_ms"] = float(per["start"]) - float(run["t0"])
                    row["sends_in_win"] = int(per["sends"])
                    row["sends_any"] = int(per["sends_any"])
                    pkey = (pair[0], pair[1], float(per["start"]))
                    if pkey not in probs:
                        raise Fail("p%s seed %d %r missing in events action_prob map" % (p, seed, pkey))
                    row["action_prob"] = probs[pkey]
            all_rows.extend(table)
    return all_rows, last_decision_rel, totals


def split_candidate(all_rows, p_list):
    """Return dict of fit, held, and criterion slices for one candidate."""
    fit_set = set(FIT_SEEDS)
    held_set = set(HELD_SEEDS)
    p_set = set(p_list)
    sub = [r for r in all_rows if r["p"] in p_set]
    fit = [r for r in sub if r["seed"] in fit_set and m24.WIN_LO <= r["t_rel_ms"] <= m24.WIN_HI]
    held = [r for r in sub if r["seed"] in held_set and m24.WIN_LO <= r["t_rel_ms"] <= m24.WIN_HI]
    crit = [r for r in sub if r["seed"] in held_set and m24.WIN_LO <= r["t_rel_ms"] <= m24.CRIT_HI]
    return {"all": sub, "fit": fit, "held": held, "crit": crit}


def build_all():
    """Load `s11a` seeds 1..80, validate window & action_prob invariants, and return splits."""
    all_rows, last_rel, totals = load_all_rows()
    cand_a = split_candidate(all_rows, ("0.2",))
    cand_b = split_candidate(all_rows, ("0.2", "0.4"))
    for cname, data in (("a", cand_a), ("b", cand_b)):
        check_action_prob(data["fit"], "cand %s fit" % cname)
        check_action_prob(data["held"], "cand %s held" % cname)
        check_window_rows(data["fit"], "cand %s fit" % cname)
        check_window_rows(data["held"], "cand %s held" % cname)
    return {
        "all_rows": all_rows,
        "last_rel": last_rel,
        "totals": totals,
        "a": cand_a,
        "b": cand_b,
    }


def rho_and_cluster_se(rows):
    """Return `(R_fit, ratio_cluster_se, per_run_mean, per_run_se, n_runs)` over `(p, seed)`."""
    by_run = collections.defaultdict(list)
    for r in rows:
        by_run[(r["p"], r["seed"])].append(r)
    sum_c = sum(float(r["c"]) for r in rows)
    sum_tau = sum(float(r["tau"]) for r in rows)
    r_fit = sum_c / sum_tau
    g = len(by_run)
    # Run-clustered delta-method SE for sum(c)/sum(tau)
    mean_tau_g = sum_tau / float(g)
    resid_sq = 0.0
    per_run_rates = []
    for key in sorted(by_run):
        rs = by_run[key]
        cg = sum(float(r["c"]) for r in rs)
        tg = sum(float(r["tau"]) for r in rs)
        per_run_rates.append(cg / tg)
        u_g = (cg - r_fit * tg) / mean_tau_g
        resid_sq += u_g * u_g
    se_cluster = (resid_sq / (float(g) * float(g - 1))) ** 0.5
    per_mean = statistics.fmean(per_run_rates)
    per_se = statistics.stdev(per_run_rates) / (len(per_run_rates) ** 0.5)
    return r_fit, se_cluster, per_mean, per_se, g


def dead_and_near_constant(rows):
    """Return `(dead_names, near_constant_list)` on fit rows."""
    mat = np.stack([r["vec"] for r in rows])
    sd = mat.std(axis=0)
    dead = []
    near = []
    for j, name in enumerate(FE.NAMES):
        if sd[j] == 0.0:
            dead.append(name)
        else:
            vals = mat[:, j]
            cnt = collections.Counter(vals.tolist())
            _mode_val, mode_n = cnt.most_common(1)[0]
            non_mode = len(vals) - mode_n
            if non_mode <= 20:
                near.append((name, non_mode, len(vals)))
    return dead, near


def show_mutation(label, fn):
    """Run `fn()` and confirm it raises `Fail`, `CH.Fail`, or `SystemExit`."""
    try:
        fn()
    except (Fail, CH.Fail, m24.Fail, SystemExit) as exc:
        say("  %-32s FIRES: %s" % (label, exc))
        return 1
    say("  %-32s DID NOT FIRE" % label)
    return 0


def main():
    """Print every required table, guard check, and mutation for seeds 1 to 80."""
    head("0  SCOPE, DIGESTS AND SEED ROLE SEPARATION")
    say("stage            %s, p %s, seeds %d to %d" % (CH.STAGE, " and ".join(P_CELLS), 1, MAX_LEARN_SEED))
    say("charges default  MAX_LOG_SEED = %d (this caller passes cap %d)" % (CH.MAX_LOG_SEED, MAX_LEARN_SEED))
    say("window           %.1f <= t_rel_ms <= %.1f (crit upper %.1f)"
        % (m24.WIN_LO, m24.WIN_HI, m24.CRIT_HI))
    say("code digests (sha256):")
    for fname in ("rows24.py", "m24.py", "charges.py", "precheck.py", "features.py"):
        say("  %-14s %s" % (fname, file_sha256(os.path.join(HERE, fname))))

    data = build_all()
    all_rows = data["all_rows"]
    fit_seeds_seen = sorted({r["seed"] for r in data["b"]["fit"]})
    held_seeds_seen = sorted({r["seed"] for r in data["b"]["held"]})
    all_seen = sorted({r["seed"] for r in all_rows})
    if set(fit_seeds_seen) != set(FIT_SEEDS) or set(held_seeds_seen) != set(HELD_SEEDS):
        raise Fail("unexpected seed split: fit=%s held=%s" % (fit_seeds_seen[:4], held_seeds_seen[:4]))
    if set(fit_seeds_seen) & set(held_seeds_seen):
        raise Fail("fit and held seeds overlap")
    if set(all_seen) & set(range(121, 161)):
        raise Fail("measuring seeds touched")
    say()
    say("seed roles:")
    say("  fit      %2d seeds (%d to %d)" % (len(fit_seeds_seen), fit_seeds_seen[0], fit_seeds_seen[-1]))
    say("  held out %2d seeds (%d to %d)" % (len(held_seeds_seen), held_seeds_seen[0], held_seeds_seen[-1]))
    say("  disjoint from 121-160: True, fit & held disjoint: True")
    say("  data files read: %d runs, seed min %d max %d count %d"
        % (len(P_CELLS) * len(ALL_SEEDS), min(all_seen), max(all_seen), len(all_seen)))

    head("1  LAST-DECISION INSTANTS AND CANDIDATE ROW COUNTS")
    for p in P_CELLS:
        rels = data["last_rel"][p]
        t = data["totals"][p]
        say("  p %s: decisions=%d  owned=%d  tail=%d  last_decision_rel_ms min=%.1f max=%.1f (n=%d pairs*runs)"
            % (p, t["decisions"], t["owned"], t["tail"], min(rels), max(rels), len(rels)))
    say()
    for cname in ("a", "b"):
        d = data[cname]
        say("  candidate %s: total_owned=%d  fit=%d  held=%d  crit=%d"
            % (cname, len(d["all"]), len(d["fit"]), len(d["held"]), len(d["crit"])))
    m02 = sum(1 for r in data["b"]["fit"] if r["p"] == "0.2")
    m04 = sum(1 for r in data["b"]["fit"] if r["p"] == "0.4")
    say("  candidate b fit p0.4 share: %d / (%d + %d) = %.6f" % (m04, m02, m04, m04 / float(m02 + m04)))

    head("2  FITTED ROWS: IN-WINDOW SEND COUNTS BY CELL AND ARM (SPEC SECTION 2)")
    for p in P_CELLS:
        sub = [r for r in data["b"]["fit"] if r["p"] == p]
        say("p %s (fit rows = %d):" % (p, len(sub)))
        say("  %-10s %8s %8s %8s %8s %8s" % ("arm", "0_sends", "1_send", "2_sends", ">2_sends", "total"))
        for arm in ARMS:
            arows = [r for r in sub if r["arm"] == arm]
            c0 = sum(1 for r in arows if r["sends_in_win"] == 0)
            c1 = sum(1 for r in arows if r["sends_in_win"] == 1)
            c2 = sum(1 for r in arows if r["sends_in_win"] == 2)
            cm = sum(1 for r in arows if r["sends_in_win"] > 2)
            say("  %-10.3f %8d %8d %8d %8d %8d" % (arm, c0, c1, c2, cm, len(arows)))
        say("  by scalar state cell:")
        for cell in CELLS:
            crows = [r for r in sub if r["cell"] == cell]
            c0 = sum(1 for r in crows if r["sends_in_win"] == 0)
            c1 = sum(1 for r in crows if r["sends_in_win"] == 1)
            c2 = sum(1 for r in crows if r["sends_in_win"] == 2)
            cm = sum(1 for r in crows if r["sends_in_win"] > 2)
            say("    %-12s %6d %8d %8d %8d %8d" % (cell, c0, c1, c2, cm, len(crows)))

    head("3  FITTED ROWS' LOGGED RATE R_FIT AND RUN-CLUSTERED SE (SEEDS 1 TO 70)")
    for p in P_CELLS:
        sub = [r for r in data["b"]["fit"] if r["p"] == p]
        r_fit, se_cl, per_m, per_se, n_runs = rho_and_cluster_se(sub)
        say("  p %s (%d runs, %d fit rows): R_fit = %.4f  ratio_SE = %.4f  per_run_mean = %.4f  per_run_SE = %.4f"
            % (p, n_runs, len(sub), r_fit, se_cl, per_m, per_se))

    head("4  DEAD AND NEAR-CONSTANT CELLS ON THE FIT ROWS")
    for cname in ("a", "b"):
        dead, near = dead_and_near_constant(data[cname]["fit"])
        live = FE.LENGTH - len(dead)
        say("candidate %s: %d dead cells, %d live cells (3*(1+%d)+%d = %d unknowns, %.2f fit rows/live unknown)"
            % (cname, len(dead), live, live, 1 if cname == "a" else 2,
               3 * (1 + live) + (1 if cname == "a" else 2),
               len(data[cname]["fit"]) / float(3 * (1 + live) + (1 if cname == "a" else 2))))
        say("  dead (%d): %s" % (len(dead), ", ".join(dead)))
        say("  near-constant (0 < non-mode <= 20):")
        for nm, k_nz, n_tot in near:
            say("    %-28s %d / %d" % (nm, k_nz, n_tot))

    head("5  EVERY CHECK SHOWN FAILING ON A MUTATED INPUT")
    sample = data["a"]["fit"][0]
    fired = []
    fired.append(show_mutation(
        "learn seed > 80", lambda: check_learn_seeds([81], "mut/81")))
    fired.append(show_mutation(
        "learn seed 0", lambda: check_learn_seeds([0], "mut/0")))
    fired.append(show_mutation(
        "learn seed fractional", lambda: check_learn_seeds([1.5], "mut/frac")))
    fired.append(show_mutation(
        "charges.load_run cap 80 > 80",
        lambda: CH.load_run("0.2", 81, max_seed=MAX_LEARN_SEED)))
    fired.append(show_mutation(
        "port.check_seeds > 100",
        lambda: PORT.check_seeds([101], "mut/101")))
    fired.append(show_mutation(
        "action_prob != 1/3",
        lambda: check_action_prob([dict(sample, action_prob=0.5)], "mut/ap")))
    fired.append(show_mutation(
        "terminal row in fit window",
        lambda: check_window_rows([dict(sample, next_pos=None)], "mut/term")))
    fired.append(show_mutation(
        "tau > 14.0",
        lambda: m24.check_tau_range([dict(sample, tau=14.1)], "mut/tau")))
    bad_succ_chain = [dict(x) for x in sample["chain"]]
    bad_succ_chain[sample["next_pos"]]["seed"] = 999
    fired.append(show_mutation(
        "successor wrong seed",
        lambda: m24.check_successor_key(
            [dict(sample, chain=bad_succ_chain)], "mut/succ-seed")))
    say()
    say("%d of %d mutations produced a failure" % (sum(fired), len(fired)))
    say()
    say("end of rows24.py")
    return 0 if sum(fired) == len(fired) else 1


if __name__ == "__main__":
    sys.exit(main())
