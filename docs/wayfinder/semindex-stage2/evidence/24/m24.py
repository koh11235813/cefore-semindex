#!/usr/bin/env python3
"""Shared learner, construction checks and replay decider M.

Not a standalone script (no `main`).  Provides:
- the block feature map phi(x) in R^262 and Phi(x, a) in R^786;
- the closed-form LSTD-Q linear system builder and solver for candidate a (787 x 787)
  and candidate b (788 x 788, one rate per cell);
- Check A (the telescoped block-endpoint identity walked from the period chains) and the
  four direct construction assertions (penalty matrix, normalisation row, tau unit/range,
  successor key);
- class M for the replay port, carrying `mem` for `FirstCallEmpty` and asserting exact
  float identity against `port.ARMS`;
- the `ast` source check on `../23/decider.py` confirming the send-time default 6.0 s;
- `self_check` showing every card-free check firing on a mutated input.
"""
import ast
import hashlib
import os
import sys

for _name, _value in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
                      ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_name, _value)

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EV22 = os.path.normpath(os.path.join(HERE, "..", "22"))
EV23 = os.path.normpath(os.path.join(HERE, "..", "23"))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if EV22 not in sys.path:
    sys.path.insert(0, EV22)

import charges as CH   # noqa: E402
import features as FE  # noqa: E402
import port as PORT    # noqa: E402

ARMS = PORT.ARMS
SEND_START = ARMS[1]               # 6.0 s, matching the previous learned decider
ARM_INDEX = {arm: i for i, arm in enumerate(ARMS)}

D = FE.LENGTH                      # 261 feature cells
BLOCK = 1 + D                      # 262 per arm block
WIDTH = 3 * BLOCK                  # 786 total linear weight unknowns
INTERCEPTS = (0, BLOCK, 2 * BLOCK)

GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
LAMBDA_REF = 1e-4
HORIZON_N = 4
WIN_LO = 10000.0
WIN_HI = 102000.0
CRIT_HI = 60000.0
TAU_LO = 1.0
TAU_HI = 14.0
MEDIAN_TAU_LO = 1.364
MEDIAN_TAU_HI = 12.172
COND_BAR = 1e10
ITER_CAP = 20
RATE_FRAC = 0.25
NORM_MULT = 10.0
CHECK_A_TOL = 1e-6


class Fail(Exception):
    """A check or invariant that did not hold."""


def file_sha256(path):
    """Return hex sha256 digest of a file."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def unit_u():
    """Return the indicator vector of the three block intercepts in R^786."""
    u = np.zeros(WIDTH, dtype=np.float64)
    for idx in INTERCEPTS:
        u[idx] = 1.0
    return u


def penalty_matrix(lam):
    """Return the 786 x 786 diagonal ridge penalty matrix scaled by `lam`."""
    diag = np.full(WIDTH, float(lam), dtype=np.float64)
    for idx in INTERCEPTS:
        diag[idx] = 0.0
    return np.diag(diag)


def standardiser(vecs):
    """Return (mu, sigma) over `vecs` `(m, 261)`, replacing zero std by 1.0."""
    mat = np.asarray(vecs, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] != D:
        raise Fail("standardiser expects shape (m, %d), got %r" % (D, mat.shape))
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    sigma = np.where(sd > 0.0, sd, 1.0)
    return mu, sigma


def phi_matrix(vecs, mu, sigma):
    """Map raw feature vectors `(m, 261)` to `[1, (x - mu) / sigma]` in `(m, 262)`."""
    mat = np.asarray(vecs, dtype=np.float64)
    scaled = (mat - mu) / sigma
    ones = np.ones((len(mat), 1), dtype=np.float64)
    return np.concatenate([ones, scaled], axis=1)


def phi_one(vec, mu, sigma):
    """Map one raw feature vector `(261,)` to `[1, (x - mu) / sigma]` in `(262,)`."""
    out = np.empty(BLOCK, dtype=np.float64)
    out[0] = 1.0
    out[1:] = (np.asarray(vec, dtype=np.float64) - mu) / sigma
    return out


def block_embed(phi, arm_indices):
    """Place each row's `phi` `(m, 262)` into its chosen arm block in `(m, 786)`."""
    m = len(phi)
    out = np.zeros((m, WIDTH), dtype=np.float64)
    for b_idx in range(3):
        mask = arm_indices == b_idx
        if np.any(mask):
            out[mask, b_idx * BLOCK:(b_idx + 1) * BLOCK] = phi[mask]
    return out


def q_matrix(vecs, w, mu, sigma):
    """Return `(m, 3)` Q-values `w_{j(a)} . phi(x)` for all three arms."""
    phi = phi_matrix(vecs, mu, sigma)
    w_blocks = np.asarray(w, dtype=np.float64).reshape(3, BLOCK)
    return phi @ w_blocks.T


def greedy_indices(q_mat):
    """Return the first-minimum arm index per row in fixed ARMS order."""
    return np.argmin(q_mat, axis=1)


def pick_min(three):
    """Return `(first_min_index, is_tied)` for a 3-tuple of Q-values."""
    m = min(three)
    idx = three.index(m)
    tied = sum(1 for v in three if v == m) > 1
    return idx, tied


def check_pick(three, idx, where="check_pick"):
    """Assert `idx` is the first minimum of `three`."""
    want_idx, _tied = pick_min(three)
    if idx != want_idx:
        raise Fail("%s: picked arm index %d, expected first minimum %d for %r"
                   % (where, idx, want_idx, three))
    return idx


def first_logged_resched_obs(p="0.2", seed=1, max_seed=20):
    """Build the port-shaped obs dict for the first `state_poll_reschedule` of (p, seed)."""
    import observe as OBS  # local to avoid circular top-level path issues
    events, _t0 = CH.load_run(p, seed, max_seed=max_seed)
    for node in sorted(events):
        got = list(OBS.from_logged_rows(events[node]))
        if got:
            nb, state, resolved, history, now = got[0]
            return {
                "where": {"kind": "state_poll_reschedule", "node": node,
                          "neighbor": nb, "ordinal": 0, "ts_ms": now},
                "state": state,
                "resolved": resolved,
                "history": history,
            }
    raise Fail("no state_poll_reschedule row found in p%s seed %d" % (p, seed))


def check_penalty_matrix(p_mat, lam, u, where="check_penalty_matrix"):
    """Assert `p_mat` is 0 on the 3 block intercepts, `lam` elsewhere, and `u^T P = 0`."""
    want_diag = np.full(WIDTH, float(lam), dtype=np.float64)
    for idx in INTERCEPTS:
        want_diag[idx] = 0.0
    if p_mat.shape != (WIDTH, WIDTH):
        raise Fail("%s: shape %r != (%d, %d)" % (where, p_mat.shape, WIDTH, WIDTH))
    off = p_mat - np.diag(np.diag(p_mat))
    if not np.array_equal(off, np.zeros_like(off)):
        raise Fail("%s: off-diagonal entries are non-zero" % where)
    if not np.array_equal(np.diag(p_mat), want_diag):
        raise Fail("%s: diagonal differs from 0 on intercepts / %g elsewhere" % (where, lam))
    up = u @ p_mat
    if not np.array_equal(up, np.zeros(WIDTH, dtype=np.float64)):
        raise Fail("%s: u^T P is not zero (max abs %.3e)" % (where, np.max(np.abs(up))))
    return 0


def check_norm_row(row, rhs_val, n_rates, u, where="check_norm_row"):
    """Assert the normalisation row equals `u` on `w`, `0` on rates, and `0` on RHS."""
    if len(row) != WIDTH + n_rates:
        raise Fail("%s: length %d != %d" % (where, len(row), WIDTH + n_rates))
    if not np.array_equal(row[:WIDTH], u):
        raise Fail("%s: weight slice of normalisation row is not u" % where)
    if not np.array_equal(row[WIDTH:], np.zeros(n_rates, dtype=np.float64)):
        raise Fail("%s: rate entries of normalisation row are non-zero" % where)
    if rhs_val != 0.0:
        raise Fail("%s: normalisation RHS %r != 0.0" % (where, rhs_val))
    return 0


def check_tau_range(rows, where="check_tau_range"):
    """Assert every fitted `tau` is in `[1.0, 14.0]` s and arm medians match `charges.txt`."""
    taus = [float(r["tau"]) for r in rows]
    bad = [t for t in taus if not TAU_LO <= t <= TAU_HI]
    if bad:
        raise Fail("%s: %d tau(s) outside [%.1f, %.1f], e.g. %.4f"
                   % (where, len(bad), TAU_LO, TAU_HI, bad[0]))
    for arm in ARMS:
        arm_taus = sorted(float(r["tau"]) for r in rows if r["arm"] == arm)
        if not arm_taus:
            raise Fail("%s: no rows for arm %.3f" % (where, arm))
        n = len(arm_taus)
        med = arm_taus[n // 2] if n % 2 else 0.5 * (arm_taus[n // 2 - 1] + arm_taus[n // 2])
        if not MEDIAN_TAU_LO <= med <= MEDIAN_TAU_HI:
            raise Fail("%s: arm %.3f median tau %.3f outside [%.3f, %.3f]"
                       % (where, arm, med, MEDIAN_TAU_LO, MEDIAN_TAU_HI))
    return len(taus)


def check_successor_key(rows, where="check_successor_key"):
    """Assert every row's successor exists at `pos + 1` on the same `(p, seed, pair)`."""
    for r in rows:
        npos = r.get("next_pos")
        if npos is None:
            raise Fail("%s: fitted row p%s seed %d %s pos %d is terminal"
                       % (where, r["p"], r["seed"], r["pair"], r["pos"]))
        succ = r["chain"][npos]
        if int(succ["pos"]) != int(r["pos"]) + 1:
            raise Fail("%s: successor pos %d != pos + 1 (%d)"
                       % (where, succ["pos"], r["pos"] + 1))
        if (succ["p"], succ["seed"], succ["pair"]) != (r["p"], r["seed"], r["pair"]):
            raise Fail("%s: successor key %r != row key %r"
                       % (where, (succ["p"], succ["seed"], succ["pair"]),
                          (r["p"], r["seed"], r["pair"])))
    return len(rows)


def check_telescoped_identity(rows, w, rhos, mu, sigma, two_rates, where,
                              policy_w=None):
    """Check A: verify the average-cost Bellman identity by walking chains, not `z_i`.

    For the logging policy (`policy_w is None`), walks each `(p, seed, pair)` block's
    first row and the successor record of its last fitted row to compute the telescoped
    endpoint sum `sum_blocks [Q_w(first) - Q_w(succ of last)]`.  For greedy iterations
    (`policy_w is not None`), evaluates `Q_w(x_i, a_i) - Q_w(x'_i, pi_k(x'_i))` by
    walking each row's `chain[next_pos]` directly.  Runs per cell for candidate b.
    """
    cells = ("0.2", "0.4") if two_rates else ("0.2",)
    if len(rhos) != len(cells):
        raise Fail("%s: expected %d rho value(s), got %d" % (where, len(cells), len(rhos)))
    rho_map = dict(zip(cells, rhos))
    out = {}
    for p_cell in cells:
        sub = [r for r in rows if r["p"] == p_cell]
        if not sub:
            raise Fail("%s: no rows for cell p %s" % (where, p_cell))
        sum_tau = sum(float(r["tau"]) for r in sub)
        sum_c = sum(float(r["c"]) for r in sub)
        sum_abs_c = sum(abs(float(r["c"])) for r in sub)
        tol = CHECK_A_TOL * sum_abs_c
        rho_hat = float(rho_map[p_cell])

        if policy_w is None:
            blocks = {}
            for r in sub:
                blocks.setdefault((r["p"], r["seed"], r["pair"]), []).append(r)
            ep_vals = []
            for key in sorted(blocks):
                blk = sorted(blocks[key], key=lambda item: int(item["pos"]))
                poses = [int(item["pos"]) for item in blk]
                if poses != list(range(poses[0], poses[0] + len(poses))):
                    raise Fail("%s: block %r has non-contiguous pos %r" % (where, key, poses[:6]))
                first = blk[0]
                last = blk[-1]
                succ = last["chain"][last["next_pos"]]
                q_first = float(q_matrix([first["vec"]], w, mu, sigma)[0, ARM_INDEX[first["arm"]]])
                q_succ = float(q_matrix([succ["vec"]], w, mu, sigma)[0, ARM_INDEX[succ["arm"]]])
                ep_vals.append(q_first - q_succ)
            ep_sum = sum(ep_vals)
            lhs = rho_hat * sum_tau + ep_sum
            gap = abs(lhs - sum_c)
            if gap > tol:
                raise Fail("%s (p %s): telescoped Check A gap %.6e > tol %.6e"
                           % (where, p_cell, gap, tol))
            out[p_cell] = {"gap": gap, "tol": tol, "ep_vals": ep_vals,
                           "ep_sum": ep_sum, "sum_tau": sum_tau, "sum_c": sum_c}
        else:
            x_cur = np.stack([r["vec"] for r in sub])
            a_cur = np.array([ARM_INDEX[r["arm"]] for r in sub], dtype=np.int64)
            x_nxt = np.stack([r["chain"][r["next_pos"]]["vec"] for r in sub])
            q_cur = q_matrix(x_cur, w, mu, sigma)
            q_nxt = q_matrix(x_nxt, w, mu, sigma)
            pi_nxt = greedy_indices(q_matrix(x_nxt, policy_w, mu, sigma))
            idx_row = np.arange(len(sub))
            diff_sum = float(np.sum(q_cur[idx_row, a_cur] - q_nxt[idx_row, pi_nxt]))
            lhs = rho_hat * sum_tau + diff_sum
            gap = abs(lhs - sum_c)
            if gap > tol:
                raise Fail("%s (p %s): iteration Check A gap %.6e > tol %.6e"
                           % (where, p_cell, gap, tol))
            out[p_cell] = {"gap": gap, "tol": tol, "diff_sum": diff_sum,
                           "sum_tau": sum_tau, "sum_c": sum_c}
    return out


def build_system(rows, mu, sigma, lam, next_mode="logged", policy_w=None,
                 two_rates=False, check=True, where="build_system"):
    """Build the square LSTD-Q linear system `(M_sys, rhs, n_rates)` for one evaluation.

    `next_mode`:
      - `"logged"`: `Phi(x'_i, a'_i)` using the arm the log drew at `chain[next_pos]`;
      - `"avg3"`: `(1/3) * sum_a Phi(x'_i, a)` (the Rao-Blackwellised logging rule);
      - `"greedy"`: `Phi(x'_i, pi_greedy(x'_i; policy_w))`.
    """
    m = len(rows)
    if m == 0:
        raise Fail("%s: empty fit row set" % where)
    x_cur = np.stack([r["vec"] for r in rows])
    a_cur = np.array([ARM_INDEX[r["arm"]] for r in rows], dtype=np.int64)
    x_nxt = np.stack([r["chain"][r["next_pos"]]["vec"] for r in rows])
    taus = np.array([float(r["tau"]) for r in rows], dtype=np.float64)
    cs = np.array([float(r["c"]) for r in rows], dtype=np.float64)

    phi_cur = phi_matrix(x_cur, mu, sigma)
    psi = block_embed(phi_cur, a_cur)

    phi_nxt = phi_matrix(x_nxt, mu, sigma)
    if next_mode == "logged":
        a_nxt = np.array([ARM_INDEX[r["chain"][r["next_pos"]]["arm"]] for r in rows],
                         dtype=np.int64)
        phi_next_blk = block_embed(phi_nxt, a_nxt)
    elif next_mode == "avg3":
        phi_next_blk = np.concatenate([phi_nxt, phi_nxt, phi_nxt], axis=1) / 3.0
    elif next_mode == "greedy":
        if policy_w is None:
            raise Fail("%s: policy_w required for next_mode='greedy'" % where)
        a_nxt = greedy_indices(q_matrix(x_nxt, policy_w, mu, sigma))
        phi_next_blk = block_embed(phi_nxt, a_nxt)
    else:
        raise Fail("%s: unknown next_mode %r" % (where, next_mode))

    z = psi - phi_next_blk
    a_mat = (psi.T @ z) / float(m)
    p_mat = penalty_matrix(lam)
    u = unit_u()
    b_vec = (psi.T @ cs) / float(m)

    if check:
        check_penalty_matrix(p_mat, lam, u, where=where)

    if not two_rates:
        n_rates = 1
        dim = WIDTH + 1
        m_sys = np.zeros((dim, dim), dtype=np.float64)
        rhs = np.zeros(dim, dtype=np.float64)
        m_sys[:WIDTH, :WIDTH] = a_mat + p_mat
        m_sys[:WIDTH, WIDTH] = (psi.T @ taus) / float(m)
        m_sys[WIDTH, :WIDTH] = u
        rhs[:WIDTH] = b_vec
        if check:
            check_norm_row(m_sys[WIDTH, :], rhs[WIDTH], n_rates, u, where=where)
    else:
        n_rates = 2
        dim = WIDTH + 2
        is02 = np.array([r["p"] == "0.2" for r in rows], dtype=bool)
        is04 = np.array([r["p"] == "0.4" for r in rows], dtype=bool)
        if not np.any(is02) or not np.any(is04):
            raise Fail("%s: two_rates=True requires both p 0.2 and p 0.4 rows" % where)
        m_sys = np.zeros((dim, dim), dtype=np.float64)
        rhs = np.zeros(dim, dtype=np.float64)
        m_sys[:WIDTH, :WIDTH] = a_mat + p_mat
        m_sys[:WIDTH, WIDTH] = (psi[is02].T @ taus[is02]) / float(m)
        m_sys[:WIDTH, WIDTH + 1] = (psi[is04].T @ taus[is04]) / float(m)
        rhs[:WIDTH] = b_vec
        m_sys[WIDTH, :WIDTH] = z[is04].sum(axis=0) / float(m)
        m_sys[WIDTH, WIDTH + 1] = taus[is04].sum() / float(m)
        rhs[WIDTH] = cs[is04].sum() / float(m)
        m_sys[WIDTH + 1, :WIDTH] = u
        if check:
            check_norm_row(m_sys[WIDTH + 1, :], rhs[WIDTH + 1], n_rates, u, where=where)

    return m_sys, rhs, n_rates


def solve_system(m_sys, rhs, n_rates, check_diagnostics=False):
    """Solve `m_sys @ theta = rhs` and return `(w, rhos, stats)`."""
    cond = float(np.linalg.cond(m_sys))
    stats = {"cond": cond}
    if check_diagnostics:
        inst = m_sys[:-1, :]
        sv_inst = np.linalg.svd(inst, compute_uv=False)
        a_blk = m_sys[:-1, :-n_rates]
        rank_a = int(np.linalg.matrix_rank(a_blk))
        rank_inst = int(np.linalg.matrix_rank(inst))
        stats.update({
            "inst_shape": inst.shape,
            "rank_a": rank_a,
            "rank_inst": rank_inst,
            "sv_min2": (float(sv_inst[-2]), float(sv_inst[-1])),
        })
    theta = np.linalg.solve(m_sys, rhs)
    w = theta[:WIDTH].copy()
    rhos = tuple(float(x) for x in theta[WIDTH:])
    return w, rhos, stats


def save_coef(path, w, mu, sigma):
    """Save `(5, 262)` float64 coefficient array `[w_0, w_1, w_2, [0, mu], [0, sigma]]`."""
    arr = np.zeros((5, BLOCK), dtype=np.float64)
    arr[:3, :] = np.asarray(w, dtype=np.float64).reshape(3, BLOCK)
    arr[3, 1:] = np.asarray(mu, dtype=np.float64)
    arr[4, 1:] = np.asarray(sigma, dtype=np.float64)
    np.save(path, arr)
    return file_sha256(path)


def load_coef(path):
    """Load `(5, 262)` float64 array and assert the leading 0.0 and positive sigma."""
    arr = np.load(path)
    if arr.shape != (5, BLOCK) or arr.dtype != np.float64:
        raise Fail("%s: expected shape (5, %d) float64, got %r %s"
                   % (path, BLOCK, arr.shape, arr.dtype))
    if not np.all(np.isfinite(arr)):
        raise Fail("%s: non-finite values in coefficient array" % path)
    if arr[3, 0] != 0.0 or arr[4, 0] != 0.0:
        raise Fail("%s: leading cell of mu/sigma rows must be 0.0 (got %r, %r)"
                   % (path, arr[3, 0], arr[4, 0]))
    if np.any(arr[4, 1:] <= 0.0):
        raise Fail("%s: non-positive standard deviation in row 4" % path)
    w_blocks = arr[:3, :].copy()
    mu = arr[3, 1:].copy()
    sigma = arr[4, 1:].copy()
    return w_blocks, mu, sigma


class M:
    """Linear value-based decider M for the replay port.

    Reads the observation vector at a resolution (`state_poll_reschedule`), stores the
    chosen arm per ordered `(node, neighbor)` pair in `self.mem`, and returns the stored
    interval (or `SEND_START = 6.0` before a pair's first resolution) at a send.
    """

    def __init__(self, w_blocks, mu, sigma):
        """Initialise with loaded `(3, 262)` `w_blocks`, `(261,)` `mu`, and `(261,)` `sigma`."""
        self.w_blocks = np.asarray(w_blocks, dtype=np.float64)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        if self.w_blocks.shape != (3, BLOCK):
            raise Fail("M: w_blocks shape %r != (3, %d)" % (self.w_blocks.shape, BLOCK))
        self.mem = {}
        self.reads = 0
        self.sends = 0
        self.ties = 0
        self.arms = {arm: 0 for arm in ARMS}
        self.last_res = {}
        self.mind_changes = 0

    @classmethod
    def from_file(cls, path):
        """Construct M from a `.npy` coefficient file."""
        w_blocks, mu, sigma = load_coef(path)
        return cls(w_blocks, mu, sigma)

    def _values(self, vec):
        """Return the 3-tuple `(Q(x, 3.695), Q(x, 6.0), Q(x, 10.043))` as Python floats."""
        phi = phi_one(vec, self.mu, self.sigma)
        vals = self.w_blocks @ phi
        return (float(vals[0]), float(vals[1]), float(vals[2]))

    def __call__(self, obs):
        """Return an element of `ARMS` by exact float identity."""
        w = obs["where"]
        pair = (w["node"], w["neighbor"])
        if w["kind"] != "state_poll_reschedule":
            self.sends += 1
            arm = self.mem.get(pair, SEND_START)
            if arm not in ARMS:
                raise Fail("M send-time arm %r not in ARMS" % (arm,))
            return arm
        rec = FE.record_from_port_obs(obs)
        vec = FE.vector(rec)
        three = self._values(vec)
        idx, tied = pick_min(three)
        arm = ARMS[idx]
        if arm not in ARMS:
            raise Fail("M resolution arm %r not in ARMS" % (arm,))
        prev_applied = self.last_res.get(pair)
        if prev_applied is not None and arm != prev_applied:
            self.mind_changes += 1
        self.last_res[pair] = arm
        self.reads += 1
        if tied:
            self.ties += 1
        self.arms[arm] += 1
        self.mem[pair] = arm
        return arm


def check_l_convention_ast(decider_path=None):
    """Parse `../23/decider.py` via `ast` (no import) and verify `SEND_START = ARMS[1]`.

    Asserts both the top-level assignment `SEND_START = ARMS[1]` and that `L.__call__`'s
    non-reschedule branch returns `self.mem.get(pair, SEND_START)`.
    """
    path = decider_path or os.path.join(EV23, "decider.py")
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src, filename=path)

    send_start_ok = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "SEND_START":
                    val = node.value
                    if (isinstance(val, ast.Subscript)
                            and isinstance(val.value, ast.Name)
                            and val.value.id == "ARMS"
                            and isinstance(val.slice, ast.Constant)
                            and val.slice.value == 1):
                        send_start_ok = True
    if not send_start_ok:
        raise Fail("%s: top-level SEND_START = ARMS[1] not found" % os.path.basename(path))

    send_branch_ok = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "L":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__call__":
                    for stmt in item.body:
                        if isinstance(stmt, ast.If):
                            dump = ast.dump(stmt)
                            if ("state_poll_reschedule" in dump
                                    and "Return" in dump
                                    and "SEND_START" in dump
                                    and "mem" in dump):
                                send_branch_ok = True
    if not send_branch_ok:
        raise Fail("%s: L.__call__ non-reschedule branch returning mem.get(..., SEND_START) not found"
                   % os.path.basename(path))
    return SEND_START


def repeat_check(decider, obs, where="repeat_check", mutate=False):
    """Run one observation twice through `decider` and assert bit-identical Q-values and arm."""
    rec = FE.record_from_port_obs(obs)
    vec = FE.vector(rec)
    v1 = decider._values(vec)
    v2 = decider._values(vec)
    if mutate:
        v2 = (v2[0] + 1.0, v2[1], v2[2])
    if v1 != v2:
        raise Fail("%s: two passes differed: %r against %r" % (where, v1, v2))
    a1 = ARMS[pick_min(v1)[0]]
    a2 = ARMS[pick_min(v2)[0]]
    if a1 != a2:
        raise Fail("%s: two passes chose %r against %r" % (where, a1, a2))
    return v1, a1


def self_check():
    """Run card-free construction, convention, and mutation checks on synthetic values."""
    u = unit_u()
    p_mat = penalty_matrix(1e-2)
    check_penalty_matrix(p_mat, 1e-2, u, "self_check")
    check_norm_row(np.concatenate([u, [0.0]]), 0.0, 1, u, "self_check")
    send_default = check_l_convention_ast()
    if send_default != SEND_START:
        raise Fail("self_check: AST SEND_START %r != %r" % (send_default, SEND_START))

    w_zero = np.zeros((3, BLOCK), dtype=np.float64)
    w_zero[1, 0] = -1.0
    m_obj = M(w_zero, np.zeros(D), np.ones(D))
    send_obs = {"where": {"kind": "state_poll", "node": "h0", "neighbor": "h1",
                          "ordinal": 0, "ts_ms": 1000.0}}
    if m_obj(send_obs) is not SEND_START:
        raise Fail("self_check: initial send did not return SEND_START by identity")

    def show_mut(label, fn):
        try:
            fn()
        except Fail as exc:
            return "  %-30s FIRES: %s" % (label, exc)
        raise Fail("self_check: mutation %s did not fire" % label)

    lines = [
        show_mut("penalty on intercept",
                 lambda: check_penalty_matrix(
                     np.eye(WIDTH) * 1e-2, 1e-2, u, "mut/p-intercept")),
        show_mut("norm row not u",
                 lambda: check_norm_row(
                     np.eye(1, WIDTH + 1, 0).ravel(), 0.0, 1, u, "mut/norm-e0")),
        show_mut("norm rhs non-zero",
                 lambda: check_norm_row(
                     np.concatenate([u, [0.0]]), 1.0, 1, u, "mut/norm-rhs")),
        show_mut("tau out of range",
                 lambda: check_tau_range([{"tau": 14.5, "arm": ARMS[0]}], "mut/tau-hi")),
        show_mut("tie pick order",
                 lambda: check_pick((1.0, 1.0, 2.0), 1, "mut/tie")),
        show_mut("AST check missing",
                 lambda: check_l_convention_ast(os.path.join(HERE, "features.py"))),
    ]
    return lines
