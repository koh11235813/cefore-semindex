#!/usr/bin/env python3
"""Fit decision maker M (candidates a and b) from s11a logs (seeds 1 to 80).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 STORE=$ROOT/store PYTHONPATH=$ROOT/repo24/src \
      $ROOT/.venv/bin/python -B learn24.py --out $ROOT/ev24

Runs on hpc-debian (CPU only).  Selects `lambda*` by the frozen-tail criterion `Crit`
(lambda_ref = 1e-4, horizon n = 4, 10000 <= t_rel_ms <= 60000) before reading any check
of section 4.7; runs Check A at every penalty and every iteration; checks the held-out
centring gate and prints the rate diagnostic at `lambda*`; compares logged vs 1/3-average
next-action rules at `lambda*`; runs LSPI with rate/norm/cycle bounds; compares candidate
a against b before any replay score; verifies bit-identical reproduction across a fresh
child process (`--check-digest`).
"""
import argparse
import collections
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile

for _name, _value in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
                      ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_name, _value)

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import charges as CH  # noqa: E402
import m24            # noqa: E402
import port as PORT   # noqa: E402
import rows24 as R24  # noqa: E402


class Fail(Exception):
    """A gate or check that did not hold."""


def file_sha256(path):
    """Return hex sha256 digest of a file."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def cluster_mean_se(values, clusters):
    """Return `(mean, cluster_se, n_clusters)` where `values` are averaged within clusters
    or weighted by row count: specifically `SE(mean over all rows)` clustered on runs
    using cluster sum deviations `(S_g - n_g * mean) / mean_n_g`.
    """
    n_tot = len(values)
    mu_all = float(np.mean(values))
    by_g = collections.defaultdict(list)
    for v, g in zip(values, clusters):
        by_g[g].append(float(v))
    g_cnt = len(by_g)
    if g_cnt < 2:
        return mu_all, 0.0, g_cnt
    mean_ng = n_tot / float(g_cnt)
    ss = 0.0
    for g in sorted(by_g):
        xs = by_g[g]
        dev = (sum(xs) - len(xs) * mu_all) / mean_ng
        ss += dev * dev
    se = (ss / (float(g_cnt) * float(g_cnt - 1))) ** 0.5
    return mu_all, se, g_cnt


def frozen_tail_targets(crit_rows, w_ref, rhos_ref, mu, sigma, two_rates):
    """Build frozen-tail `y_i` for `Crit` at `(w_ref, rhos_ref)` and horizon `n = 4`."""
    rho_map = dict(zip(("0.2", "0.4") if two_rates else ("0.2",), rhos_ref))
    y_vals = np.empty(len(crit_rows), dtype=np.float64)
    x_fut = []
    arm_fut = []
    for i, r in enumerate(crit_rows):
        chain = r["chain"]
        pos = int(r["pos"])
        if pos + m24.HORIZON_N >= len(chain):
            raise Fail("crit row p%s seed %d %s pos %d lacks %d future owned rows"
                       % (r["p"], r["seed"], r["pair"], pos, m24.HORIZON_N))
        rho_c = rho_map[r["p"]]
        acc = 0.0
        for j in range(m24.HORIZON_N):
            step = chain[pos + j]
            acc += float(step["c"]) - rho_c * float(step["tau"])
        y_vals[i] = acc
        end_row = chain[pos + m24.HORIZON_N]
        x_fut.append(end_row["vec"])
        arm_fut.append(m24.ARM_INDEX[end_row["arm"]])
    q_fut = m24.q_matrix(np.stack(x_fut), w_ref, mu, sigma)
    y_vals += q_fut[np.arange(len(crit_rows)), np.array(arm_fut, dtype=np.int64)]
    return y_vals


def eval_crit(crit_rows, y_frozen, w, mu, sigma):
    """Return `(crit_mean, crit_se, n_clusters)` for `Crit(lambda)`."""
    x_cur = np.stack([r["vec"] for r in crit_rows])
    a_cur = np.array([m24.ARM_INDEX[r["arm"]] for r in crit_rows], dtype=np.int64)
    q_cur = m24.q_matrix(x_cur, w, mu, sigma)
    pred = q_cur[np.arange(len(crit_rows)), a_cur]
    sq = (pred - y_frozen) ** 2
    clusters = [(r["p"], r["seed"]) for r in crit_rows]
    return cluster_mean_se(sq, clusters)


def one_step_residuals(rows, w, rhos, mu, sigma, two_rates):
    """Return `(res_logged, res_rb, clusters)` on `rows`."""
    rho_map = dict(zip(("0.2", "0.4") if two_rates else ("0.2",), rhos))
    x_cur = np.stack([r["vec"] for r in rows])
    a_cur = np.array([m24.ARM_INDEX[r["arm"]] for r in rows], dtype=np.int64)
    x_nxt = np.stack([r["chain"][r["next_pos"]]["vec"] for r in rows])
    a_nxt = np.array([m24.ARM_INDEX[r["chain"][r["next_pos"]]["arm"]] for r in rows],
                     dtype=np.int64)
    q_cur = m24.q_matrix(x_cur, w, mu, sigma)[np.arange(len(rows)), a_cur]
    q_nxt_all = m24.q_matrix(x_nxt, w, mu, sigma)
    q_nxt_log = q_nxt_all[np.arange(len(rows)), a_nxt]
    q_nxt_rb = q_nxt_all.mean(axis=1)
    c_arr = np.array([float(r["c"]) for r in rows], dtype=np.float64)
    tau_arr = np.array([float(r["tau"]) for r in rows], dtype=np.float64)
    rho_arr = np.array([rho_map[r["p"]] for r in rows], dtype=np.float64)
    res_log = c_arr - rho_arr * tau_arr + q_nxt_log - q_cur
    res_rb = c_arr - rho_arr * tau_arr + q_nxt_rb - q_cur
    clusters = [(r["p"], r["seed"]) for r in rows]
    return res_log, res_rb, clusters


def check_centring(res_vals, clusters, taus, where="centring_gate"):
    """Assert held-out mean residual sits within 2 run-clustered SEs."""
    mean, se, g_cnt = cluster_mean_se(res_vals, clusters)
    mean_tau = float(np.mean(taus))
    det_row = 2.0 * se
    det_rate = det_row / mean_tau
    ok = abs(mean) <= 2.0 * se
    info = {"mean": mean, "se": se, "g": g_cnt, "det_row": det_row,
            "det_rate": det_rate, "mean_tau": mean_tau}
    if not ok:
        raise Fail("%s: |mean %.4f| > 2 * SE (%.4f, g=%d)" % (where, mean, 2.0 * se, g_cnt))
    return info


def check_iter_bounds(rho_k, r_fit_by_p, w_k, w_0_norm, two_rates, where):
    """Return `(ok, reason)` for the iteration rate and weight-norm bounds."""
    cells = ("0.2", "0.4") if two_rates else ("0.2",)
    for p_cell, rho in zip(cells, rho_k):
        r_fit = r_fit_by_p[p_cell]
        if rho <= 0.0:
            return False, "%s: rho(p%s)=%.4f <= 0" % (where, p_cell, rho)
        if abs(rho - r_fit) > m24.RATE_FRAC * r_fit:
            return False, "%s: |rho(p%s) %.4f - R_fit %.4f| > 0.25 * R_fit (%.4f)" % (
                where, p_cell, rho, r_fit, m24.RATE_FRAC * r_fit)
    w_norm = float(np.linalg.norm(w_k))
    if w_norm > m24.NORM_MULT * w_0_norm:
        return False, "%s: ||w_k|| %.4f > 10 * ||w_0|| (%.4f)" % (
            where, w_norm, m24.NORM_MULT * w_0_norm)
    return True, "ok"


def fit_candidate(cname, cand_data, two_rates, say):
    """Run grid selection, checks, LSPI, and return result dict for candidate `cname`."""
    fit_rows = cand_data["fit"]
    held_rows = cand_data["held"]
    crit_rows = cand_data["crit"]
    cells = ("0.2", "0.4") if two_rates else ("0.2",)

    say()
    say("--- candidate %s (two_rates=%s, fit=%d, held=%d, crit=%d) ---"
        % (cname, two_rates, len(fit_rows), len(held_rows), len(crit_rows)))

    x_fit = np.stack([r["vec"] for r in fit_rows])
    mu, sigma = m24.standardiser(x_fit)

    r_fit_by_p = {}
    r_se_by_p = {}
    for p_cell in cells:
        sub = [r for r in fit_rows if r["p"] == p_cell]
        rf, se_cl, _pm, _ps, _ng = R24.rho_and_cluster_se(sub)
        r_fit_by_p[p_cell] = rf
        r_se_by_p[p_cell] = se_cl

    # Reference solve at LAMBDA_REF for Crit
    m_ref, rhs_ref, n_rates = m24.build_system(
        fit_rows, mu, sigma, m24.LAMBDA_REF, next_mode="logged",
        two_rates=two_rates, check=True, where="cand %s ref" % cname)
    w_ref, rhos_ref, stats_ref = m24.solve_system(m_ref, rhs_ref, n_rates, check_diagnostics=True)
    if stats_ref["cond"] > m24.COND_BAR:
        raise Fail("cand %s ref solve cond %.3e > %.3e" % (cname, stats_ref["cond"], m24.COND_BAR))
    y_frozen = frozen_tail_targets(crit_rows, w_ref, rhos_ref, mu, sigma, two_rates)

    # Grid search over logging policy pi_0
    grid_rows = []
    held_taus = np.array([float(r["tau"]) for r in held_rows], dtype=np.float64)
    say()
    say("grid search on logging policy (Crit m''=%d, held m'=%d):" % (len(crit_rows), len(held_rows)))
    say("  %-8s %11s %9s %10s %10s %10s %10s %12s %s"
        % ("lambda", "Crit", "Crit_SE", "res_log", "res_SE", "msq_log", "msq_RB", "cond", "rhos"))
    for lam in m24.GRID:
        m_sys, rhs, n_r = m24.build_system(
            fit_rows, mu, sigma, lam, next_mode="logged",
            two_rates=two_rates, check=True, where="cand %s lam %g" % (cname, lam))
        w_g, rhos_g, st_g = m24.solve_system(m_sys, rhs, n_r, check_diagnostics=True)
        if st_g["cond"] > m24.COND_BAR:
            raise Fail("cand %s k=0 lam=%g cond %.3e > %.3e" % (cname, lam, st_g["cond"], m24.COND_BAR))
        ca_g = m24.check_telescoped_identity(
            fit_rows, w_g, rhos_g, mu, sigma, two_rates, "cand %s grid lam %g" % (cname, lam))
        c_mean, c_se, c_g = eval_crit(crit_rows, y_frozen, w_g, mu, sigma)
        r_log, r_rb, h_cl = one_step_residuals(held_rows, w_g, rhos_g, mu, sigma, two_rates)
        rm_log, rse_log, _ = cluster_mean_se(r_log, h_cl)
        rm_rb, rse_rb, _ = cluster_mean_se(r_rb, h_cl)
        msq_log = float(np.mean(r_log ** 2))
        msq_rb = float(np.mean(r_rb ** 2))
        grid_rows.append({
            "lam": lam, "w": w_g, "rhos": rhos_g, "stats": st_g, "check_a": ca_g,
            "crit": c_mean, "crit_se": c_se, "crit_g": c_g,
            "res_log": (rm_log, rse_log, msq_log), "res_rb": (rm_rb, rse_rb, msq_rb),
            "r_log_arr": r_log, "r_rb_arr": r_rb, "h_cl": h_cl,
        })
        say("  %-8g %11.3f %9.3f %10.4f %10.4f %10.2f %10.2f %12.3e %s"
            % (lam, c_mean, c_se, rm_log, rse_log, msq_log, msq_rb, st_g["cond"],
               [round(x, 4) for x in rhos_g]))

    # Select lambda* = argmin Crit, ties to LARGER lambda
    best_crit = min(g["crit"] for g in grid_rows)
    chosen = max((g for g in grid_rows if g["crit"] == best_crit), key=lambda g: g["lam"])
    lam_star = chosen["lam"]
    say("selected lambda* = %g (Crit = %.3f +- %.3f)" % (lam_star, chosen["crit"], chosen["crit_se"]))

    # Identification rank & cond at lambda*
    st = chosen["stats"]
    say("instrument block at lambda*: shape=%r rank(A+lam*P)=%d rank(inst)=%d sv_min2=(%.4e, %.4e) cond=%.4e"
        % (st["inst_shape"], st["rank_a"], st["rank_inst"], st["sv_min2"][0], st["sv_min2"][1], st["cond"]))

    # Held-out centring gate at lambda* (logged next arm)
    cgate = check_centring(chosen["r_log_arr"], chosen["h_cl"], held_taus, "cand %s lambda*" % cname)
    say("centring gate PASSED at lambda*: mean=%.4f  2*SE=%.4f (g=%d)  detectable=%.2f ms/row (%.2f ms/s)"
        % (cgate["mean"], 2.0 * cgate["se"], cgate["g"], cgate["det_row"], cgate["det_rate"]))

    # Rate diagnostic curve + detailed breakdown at lambda*
    say("rate diagnostic (non-gating) at lambda* and across grid:")
    for g in grid_rows:
        mark = " <- lambda*" if g["lam"] == lam_star else ""
        parts = []
        for p_cell, rho_hat in zip(cells, g["rhos"]):
            gap = rho_hat - r_fit_by_p[p_cell]
            parts.append("p%s: rho_hat=%.4f R_fit=%.4f diff=%+.4f (SE %.4f)"
                         % (p_cell, rho_hat, r_fit_by_p[p_cell], gap, r_se_by_p[p_cell]))
        say("  lam=%-7g %s%s" % (g["lam"], " | ".join(parts), mark))
    for p_cell, rho_hat in zip(cells, chosen["rhos"]):
        ep_vals = chosen["check_a"][p_cell]["ep_vals"]
        n_pos = sum(1 for v in ep_vals if v > 0.0)
        n_neg = sum(1 for v in ep_vals if v < 0.0)
        n_zero = sum(1 for v in ep_vals if v == 0.0)
        say("  lambda* block endpoints (p%s, %d blocks): sum=%.3f mean=%.4f median=%.4f (+/%d -/%d 0/%d) CheckA_gap=%.3e"
            % (p_cell, len(ep_vals), sum(ep_vals), statistics.fmean(ep_vals),
               statistics.median(ep_vals), n_pos, n_neg, n_zero, chosen["check_a"][p_cell]["gap"]))

    # Compare logged vs 1/3-average next-action rule at lambda*
    m_avg, rhs_avg, n_r = m24.build_system(
        fit_rows, mu, sigma, lam_star, next_mode="avg3",
        two_rates=two_rates, check=True, where="cand %s avg3" % cname)
    w_avg, rhos_avg, _ = m24.solve_system(m_avg, rhs_avg, n_r)
    _rl_avg, rrb_avg, h_cl_avg = one_step_residuals(held_rows, w_avg, rhos_avg, mu, sigma, two_rates)
    rm_avg, rse_avg, _ = cluster_mean_se(rrb_avg, h_cl_avg)
    say("next-action rule comparison at lambda*:")
    say("  logged a': rhos=%s  intercepts=%s  held_mean_res=%.4f (SE %.4f)"
        % ([round(x, 4) for x in chosen["rhos"]],
           [round(float(chosen["w"][i]), 4) for i in m24.INTERCEPTS],
           cgate["mean"], cgate["se"]))
    say("  1/3 avg  : rhos=%s  intercepts=%s  held_mean_res(RB)=%.4f (SE %.4f)"
        % ([round(x, 4) for x in rhos_avg],
           [round(float(w_avg[i]), 4) for i in m24.INTERCEPTS],
           rm_avg, rse_avg))

    # Policy iteration (LSPI) at fixed lam_star
    say()
    say("policy iteration at fixed lambda* = %g (cap=%d):" % (lam_star, m24.ITER_CAP))
    say("  %3s %10s %11s %9s %20s %16s %s"
        % ("k", "changed", "||w_k||_2", "cond", "rhos", "arm_shares", "status"))
    logged_arm_idx = np.array([m24.ARM_INDEX[r["arm"]] for r in fit_rows], dtype=np.int64)
    prev_policies = []
    history = []
    w_0 = chosen["w"]
    w_0_norm = float(np.linalg.norm(w_0))
    w_prev = None
    w_final = w_0
    exit_reason = "init"

    for k in range(m24.ITER_CAP):
        if k == 0:
            w_k = chosen["w"]
            rhos_k = chosen["rhos"]
            cond_k = chosen["stats"]["cond"]
            ca_k = chosen["check_a"]
        else:
            m_k, rhs_k, n_rk = m24.build_system(
                fit_rows, mu, sigma, lam_star, next_mode="greedy", policy_w=w_prev,
                two_rates=two_rates, check=True, where="cand %s k=%d" % (cname, k))
            w_k, rhos_k, st_k = m24.solve_system(m_k, rhs_k, n_rk)
            cond_k = st_k["cond"]
            if cond_k > m24.COND_BAR:
                exit_reason = "cond_bar_k_%d" % k
                w_final = w_prev
                say("  %3d %10s %11.3f %9.3e %20s %16s STOP cond > 1e10 -> pi_greedy(w_%d)"
                    % (k, "--", float(np.linalg.norm(w_k)), cond_k,
                       [round(x, 4) for x in rhos_k], "--", k - 1))
                break
            ca_k = m24.check_telescoped_identity(
                fit_rows, w_k, rhos_k, mu, sigma, two_rates,
                "cand %s iter k=%d" % (cname, k), policy_w=w_prev)
            ok_b, msg_b = check_iter_bounds(rhos_k, r_fit_by_p, w_k, w_0_norm, two_rates,
                                            "cand %s k=%d" % (cname, k))
            if not ok_b:
                exit_reason = "bound_breach_k_%d" % k
                w_final = w_0
                say("  %3d %10s %11.3f %9.3e %20s %16s STOP %s -> fallback pi_greedy(w_0)"
                    % (k, "--", float(np.linalg.norm(w_k)), cond_k,
                       [round(x, 4) for x in rhos_k], "--", msg_b))
                break

        pi_next = m24.greedy_indices(m24.q_matrix(x_fit, w_k, mu, sigma))
        ref_pi = logged_arm_idx if k == 0 else prev_policies[-1]
        changed = float(np.mean(pi_next != ref_pi))
        shares = [float(np.mean(pi_next == i)) for i in range(3)]
        pi_bytes = pi_next.tobytes()

        cycle_len = None
        if k > 0:
            for j, old_bytes in enumerate(prev_policies[:-1]):
                if pi_bytes == old_bytes:
                    cycle_len = k - j
                    break

        note = "vs_log" if k == 0 else ("cycle_%d" % cycle_len if cycle_len else "ok")
        say("  %3d %10.6f %11.3f %9.3e %20s %s %s"
            % (k, changed, float(np.linalg.norm(w_k)), cond_k,
               [round(x, 4) for x in rhos_k],
               "/".join("%.3f" % s for s in shares), note))
        history.append({
            "k": k, "changed": changed, "w_norm": float(np.linalg.norm(w_k)),
            "cond": cond_k, "rhos": list(rhos_k), "shares": shares,
            "cycle_len": cycle_len,
            "check_a_max_gap": max(v["gap"] for v in ca_k.values()),
        })

        if k > 0 and changed == 0.0:
            exit_reason = "converged_k_%d" % k
            w_final = w_k
            break
        prev_policies.append(pi_bytes)
        w_prev = w_k
    else:
        exit_reason = "cap_reached_%d" % m24.ITER_CAP
        w_final = w_0

    say("final policy exit: %s" % exit_reason)
    return {
        "name": cname, "two_rates": two_rates, "w": w_final, "w_0": w_0,
        "mu": mu, "sigma": sigma, "lam_star": lam_star, "exit_reason": exit_reason,
        "history": history, "r_fit": r_fit_by_p, "r_se": r_se_by_p,
        "centring": cgate, "crit_grid": [
            {"lam": g["lam"], "crit": g["crit"], "crit_se": g["crit_se"],
             "rhos": list(g["rhos"]), "cond": g["stats"]["cond"]}
            for g in grid_rows
        ],
        "fit_rows_count": len(fit_rows), "held_rows_count": len(held_rows),
        "crit_rows_count": len(crit_rows),
    }


def write_outputs(out_dir, res, say_fn=None):
    """Write `coef_<c>.npy` and `coef_<c>.json` and return `(npy_path, sha256)`."""
    os.makedirs(out_dir, exist_ok=True)
    cname = res["name"]
    npy_path = os.path.join(out_dir, "coef_%s.npy" % cname)
    json_path = os.path.join(out_dir, "coef_%s.json" % cname)
    sha = m24.save_coef(npy_path, res["w"], res["mu"], res["sigma"])
    meta = {
        "candidate": cname,
        "two_rates": res["two_rates"],
        "lambda_star": res["lam_star"],
        "exit_reason": res["exit_reason"],
        "fit_rows": res["fit_rows_count"],
        "held_rows": res["held_rows_count"],
        "crit_rows": res["crit_rows_count"],
        "r_fit": res["r_fit"],
        "r_se": res["r_se"],
        "centring": res["centring"],
        "history": res["history"],
        "crit_grid": res["crit_grid"],
        "npy_sha256": sha,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    if say_fn is not None:
        say_fn("wrote %s (sha256 %s, %d bytes)" % (os.path.basename(npy_path), sha, os.path.getsize(npy_path)))
    return npy_path, sha


def compare_candidates(res_a, res_b, fit_a_rows, say):
    """Print pre-score comparison between candidate a and candidate b."""
    m24.head("3  CANDIDATE A AGAINST CANDIDATE B (BEFORE ANY SCORE)")
    wa = res_a["w"].reshape(3, m24.BLOCK)
    wb = res_b["w"].reshape(3, m24.BLOCK)
    diff = wa - wb
    for j, arm in enumerate(m24.ARMS):
        say("  block %.3f s ||w_a - w_b||_2 = %.6f (intercept diff %+.6f)"
            % (arm, float(np.linalg.norm(diff[j])), float(diff[j, 0])))
    cell_names = ["intercept"] + list(R24.FE.NAMES)
    entries = []
    for j, arm in enumerate(m24.ARMS):
        for k, nm in enumerate(cell_names):
            entries.append((abs(float(diff[j, k])), arm, nm, float(wa[j, k]), float(wb[j, k])))
    entries.sort(reverse=True)
    say("  top 10 coefficient differences:")
    for _ad, arm, nm, va, vb in entries[:10]:
        say("    arm %-6.3f %-28s w_a=%+10.4f  w_b=%+10.4f  diff=%+10.4f"
            % (arm, nm, va, vb, va - vb))

    x_a = np.stack([r["vec"] for r in fit_a_rows])
    pi_a = m24.greedy_indices(m24.q_matrix(x_a, res_a["w"], res_a["mu"], res_a["sigma"]))
    pi_b = m24.greedy_indices(m24.q_matrix(x_a, res_b["w"], res_b["mu"], res_b["sigma"]))
    disagree = float(np.mean(pi_a != pi_b))
    say("  argmin disagreement on candidate a's %d fit rows: %d / %d = %.6f"
        % (len(fit_a_rows), int(np.sum(pi_a != pi_b)), len(fit_a_rows), disagree))


def check_digest_child(out_dir):
    """Child process entry (`--check-digest`): re-read logs and write digests."""
    data = R24.build_all()
    def sink(*_a, **_kw):
        return None  # silent child; only digests go to stdout
    res_a = fit_candidate("a", data["a"], False, sink)
    res_b = fit_candidate("b", data["b"], True, sink)
    _pa, sha_a = write_outputs(out_dir, res_a)
    _pb, sha_b = write_outputs(out_dir, res_b)
    print("a %s" % sha_a, flush=True)
    print("b %s" % sha_b, flush=True)
    return 0


def two_process_check(out_dir, sha_a, sha_b):
    """Spawn a fresh interpreter to re-read logs and re-solve, asserting identical sha256."""
    with tempfile.TemporaryDirectory() as tmpd:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [sys.executable, "-B", os.path.abspath(__file__), "--check-digest", tmpd],
            capture_output=True, text=True, env=env, check=False)
        if proc.returncode != 0:
            raise Fail("child process failed (%d): %s" % (proc.returncode, proc.stderr.strip()))
        got = {}
        for ln in proc.stdout.splitlines():
            parts = ln.strip().split()
            if len(parts) == 2:
                got[parts[0]] = parts[1]
        if got.get("a") != sha_a or got.get("b") != sha_b:
            raise Fail("two-process digest mismatch: parent=(%s, %s), child=(%s, %s)"
                       % (sha_a, sha_b, got.get("a"), got.get("b")))
    return got


def run_mutations(data, res_a, sample_obs, say):
    """Show every check and gate failing on mutated inputs."""
    say("m24.self_check:")
    for ln in m24.self_check():
        say(ln)

    def show_mut(label, fn):
        try:
            fn()
        except (Fail, m24.Fail, R24.Fail, CH.Fail, SystemExit) as exc:
            say("  %-34s FIRES: %s" % (label, exc))
            return 1
        say("  %-34s DID NOT FIRE" % label)
        return 0

    fit_a = data["a"]["fit"]
    held_a = data["a"]["held"]
    fired = []
    fired.append(show_mut(
        "Check A: negated w fails",
        lambda: m24.check_telescoped_identity(
            fit_a, -res_a["w_0"], (res_a["crit_grid"][0]["rhos"][0],),
            res_a["mu"], res_a["sigma"], False, "mut/neg-w")))
    fired.append(show_mut(
        "Check A: shifted rho fails",
        lambda: m24.check_telescoped_identity(
            fit_a, res_a["w_0"], (res_a["crit_grid"][0]["rhos"][0] + 10.0,),
            res_a["mu"], res_a["sigma"], False, "mut/rho-shift")))

    # Off-by-one successor in a shallow copy of fit_a
    bad_rows = [dict(r) for r in fit_a[:20]]
    ch_copy = [dict(x) for x in bad_rows[0]["chain"]]
    if len(ch_copy) > bad_rows[0]["next_pos"] + 1:
        bad_rows[0]["chain"] = ch_copy
        bad_rows[0]["next_pos"] = bad_rows[0]["next_pos"] + 1
    fired.append(show_mut(
        "Check A: off-by-one successor",
        lambda: m24.check_successor_key(bad_rows, "mut/off-by-one")))

    # Centring gate failure
    fired.append(show_mut(
        "centring gate: shifted residual",
        lambda: check_centring(
            np.full(len(held_a), 500.0), [(r["p"], r["seed"]) for r in held_a],
            np.array([float(r["tau"]) for r in held_a]), "mut/centring")))

    # Iteration bounds
    ok_neg, msg_neg = check_iter_bounds((-1.0,), res_a["r_fit"], res_a["w_0"],
                                        float(np.linalg.norm(res_a["w_0"])), False, "mut/neg-rho")
    fired.append(show_mut("iter bound: negative rho",
                          lambda: (_ for _ in ()).throw(Fail(msg_neg)) if not ok_neg else None))
    ok_nrm, msg_nrm = check_iter_bounds(
        (res_a["r_fit"]["0.2"],), res_a["r_fit"], res_a["w_0"] * 20.0,
        float(np.linalg.norm(res_a["w_0"])), False, "mut/norm-10x")
    fired.append(show_mut("iter bound: 20x weight norm",
                          lambda: (_ for _ in ()).throw(Fail(msg_nrm)) if not ok_nrm else None))

    # Repeat check mutation
    dec_a = m24.M(res_a["w"].reshape(3, m24.BLOCK), res_a["mu"], res_a["sigma"])
    fired.append(show_mut(
        "repeat check: 1 value moved",
        lambda: m24.repeat_check(dec_a, sample_obs, "mut/repeat", mutate=True)))

    # Seed guards
    fired.append(show_mut("learn seed > 80", lambda: R24.check_learn_seeds([81], "mut/81")))
    fired.append(show_mut("port.check_seeds > 100", lambda: PORT.check_seeds([101], "mut/101")))

    say()
    say("%d of %d learn24 mutations produced a failure" % (sum(fired), len(fired)))
    return sum(fired) == len(fired)


def main():
    """Run full learning pipeline, two-process reproducibility gate, and mutations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=HERE)
    parser.add_argument("--check-digest", default=None)
    args = parser.parse_args()

    if args.check_digest is not None:
        return check_digest_child(args.check_digest)

    out_dir = os.path.abspath(args.out)
    lines = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    def head(title):
        say()
        say("=" * 72)
        say(title)
        say("=" * 72)

    m24.head = head
    head("0  SCOPE, DIGESTS AND SETTINGS")
    say("numpy %s, python %s" % (np.__version__, sys.version.split()[0]))
    say("grid=%s  lambda_ref=%g  horizon_n=%d  cond_bar=%.1e  iter_cap=%d"
        % (list(m24.GRID), m24.LAMBDA_REF, m24.HORIZON_N, m24.COND_BAR, m24.ITER_CAP))
    say("code digests (sha256):")
    for fname in ("learn24.py", "m24.py", "rows24.py", "charges.py", "precheck.py", "features.py"):
        say("  %-14s %s" % (fname, file_sha256(os.path.join(HERE, fname))))

    head("1  BUILD ROWS (SEEDS 1 TO 80)")
    data = R24.build_all()
    say("cand a: fit=%d held=%d crit=%d" % (len(data["a"]["fit"]), len(data["a"]["held"]), len(data["a"]["crit"])))
    say("cand b: fit=%d held=%d crit=%d" % (len(data["b"]["fit"]), len(data["b"]["held"]), len(data["b"]["crit"])))

    head("2  FIT CANDIDATES A AND B")
    res_a = fit_candidate("a", data["a"], False, say)
    res_b = fit_candidate("b", data["b"], True, say)

    compare_candidates(res_a, res_b, data["a"]["fit"], say)

    head("4  COEFFICIENT FILES, REPEAT CHECK AND TWO-PROCESS REPRODUCIBILITY")
    _pa, sha_a = write_outputs(out_dir, res_a, say)
    _pb, sha_b = write_outputs(out_dir, res_b, say)
    sample_obs = m24.first_logged_resched_obs("0.2", 1, max_seed=R24.MAX_LEARN_SEED)
    for cname, res in (("a", res_a), ("b", res_b)):
        dec = m24.M(res["w"].reshape(3, m24.BLOCK), res["mu"], res["sigma"])
        vals, arm = m24.repeat_check(dec, sample_obs, "cand %s" % cname)
        say("  repeat check cand %s (p0.2 seed 1 first resched): values=%s -> arm %.3f"
            % (cname, [round(x, 6) for x in vals], arm))
    child_shas = two_process_check(out_dir, sha_a, sha_b)
    say("  two-process sha256 identical: a=%s b=%s" % (child_shas["a"], child_shas["b"]))

    head("5  EVERY CHECK SHOWN FAILING ON A MUTATED INPUT")
    ok_mut = run_mutations(data, res_a, sample_obs, say)
    say()
    say("end of learn24.py")

    with open(os.path.join(out_dir, "learn24.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0 if ok_mut else 1


if __name__ == "__main__":
    sys.exit(main())
