"""LRWS-QAOA window solver: prepare (classical) -> circuit -> decode.

A window is (u; c_1..c_k; M) with M the (k+1)x(k+1) cost matrix, index 0 = u.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
from scipy.optimize import linear_sum_assignment
from . import qubo, warmstart, sim, ramp, routing


@lru_cache(maxsize=8)
def _bits(n: int) -> np.ndarray:
    return sim.bit_matrix(n)


@lru_cache(maxsize=8)
def _valid_mask(k: int) -> np.ndarray:
    X = _bits(k * k).reshape(-1, k, k)
    return (X.sum(axis=1) == 1).all(axis=1) & (X.sum(axis=2) == 1).all(axis=1)


@dataclass
class Window:
    M: np.ndarray
    k: int
    Q: np.ndarray
    P: float
    const: float
    x_lp: np.ndarray
    x_clip: np.ndarray
    theta: np.ndarray
    cfg: dict = field(default_factory=dict)
    _E: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.k * self.k

    @property
    def E_norm(self) -> np.ndarray:
        """x^T Q x / P for every basis state (the constant is a global phase and omitted)."""
        if self._E is None:
            self._E = sim.energies(self.Q, _bits(self.n)) / self.P
        return self._E


def make_window(M, lam=1.5, eps=0.25, lp_cost="surrogate", x_lp=None) -> Window:
    M = np.asarray(M, dtype=float)
    k = M.shape[0] - 1
    Q, P, const = qubo.build_qubo(M, lam)
    if x_lp is None:
        x_lp = warmstart.lp_relaxation(M, lp_cost)
    theta, xc = warmstart.warm_start_angles(np.asarray(x_lp), eps)
    return Window(M=M, k=k, Q=Q, P=P, const=const, x_lp=np.asarray(x_lp), x_clip=xc, theta=theta,
                  cfg={"lam": lam, "eps": eps, "lp_cost": lp_cost})


def choose_angles(win: Window, p: int, rcfg: dict) -> dict:
    """Angles of the linear ramp. mode='optimize': (Dg, Db) minimise <E> on the exact simulator;
    mode='fixed': use the configured (Dg, Db) with no simulation."""
    if rcfg["ramp_mode"] == "fixed":
        dg, db, e, ev = rcfg["delta_gamma"], rcfg["delta_beta"], None, 0
        src = "linear_ramp_fixed"
    else:
        r = ramp.optimise_ramp(win.E_norm, win.theta, p, grid=rcfg["ramp_grid"], maxiter=rcfg["ramp_maxiter"])
        dg, db, e, ev = r["delta_gamma"], r["delta_beta"], r["energy"], r["evaluations"]
        src = "linear_ramp_optimized"
    g, b = ramp.ramp_angles(p, dg, db)
    return {"delta_gamma": dg, "delta_beta": db, "gammas": g.tolist(), "betas": b.tolist(),
            "source": src, "energy_norm": e, "evaluations": ev}


def ideal_probabilities(win: Window, gammas, betas) -> np.ndarray:
    return sim.probabilities(win.E_norm, win.theta, gammas, betas)


# ----------------------------------------------------------------- decoding
def marginals_from_probs(win: Window, probs: np.ndarray) -> np.ndarray:
    return (probs @ _bits(win.n)).reshape(win.k, win.k)          # m[i, t]


def counts_to_vector(win: Window, counts: dict, reverse_bits: bool = False) -> np.ndarray:
    """Counts (Qiskit key order unless reverse_bits) -> empirical probability vector (2^n)."""
    n = win.n
    v = np.zeros(2 ** n)
    for key, c in counts.items():
        key = str(key).replace(" ", "")
        if len(key) != n:
            raise ValueError(f"count key '{key}' has length {len(key)}, expected {n}")
        if reverse_bits:
            key = key[::-1]
        v[int(key, 2)] += c
    return v / v.sum()


def decode_order(m: np.ndarray) -> list[int]:
    """Permutation maximising sum_t m[order[t], t] (Hungarian). Ties -> deterministic."""
    rows, cols = linear_sum_assignment(-m.T)                     # rows: position t, cols: candidate
    return [int(c) for c in cols[np.argsort(rows)]]


def interpret(win: Window, probs: np.ndarray, ideal: np.ndarray | None = None, rule: str = "marginal",
              min_prob: float = 0.0) -> dict:
    """Decode an (empirical or ideal) distribution and compute window diagnostics.

    rule = "marginal"   : order maximising the sum of marginals m[i, t] (default, uses the whole distribution)
    rule = "best_valid" : lowest-cost valid permutation among states with probability >= min_prob
                          (min_prob = 1/shots emulates the shot resolution); falls back to "marginal"
    """
    k = win.k
    X = _bits(win.n)
    valid = _valid_mask(k)
    m = marginals_from_probs(win, probs)
    perms, costs = routing.all_order_costs(win.M)
    opt, worst = float(costs.min()), float(costs.max())
    raw_E = win.E_norm * win.P + win.const
    d = {"p_valid": float(probs[valid].sum()), "p_valid_chance": len(perms) / 2 ** win.n,
         "objective_mean": float(probs @ raw_E), "warm_start_order": decode_order(win.x_clip)}
    cand = np.nonzero(valid & (probs > 0) & (probs >= min_prob - 1e-12))[0]
    if len(cand):
        best = None
        for s_ in cand:
            x = X[s_].reshape(k, k)
            o = [int(np.argmax(x[:, t])) for t in range(k)]
            c = routing.window_order_cost(win.M, o)
            if best is None or c < best[0] - 1e-12:
                best = (c, o)
        d["best_valid_cost"], d["best_valid_order"] = best
    order = decode_order(m)
    d["marginal_order"] = order
    if rule == "best_valid" and "best_valid_order" in d:
        order = d["best_valid_order"]
    cost = routing.window_order_cost(win.M, order)
    d.update({"order": order, "marginals": m.tolist(), "cost": cost, "exact_cost": opt, "worst_cost": worst,
              "regret_norm": (cost - opt) / (worst - opt) if worst > opt else 0.0,
              "equals_exact": bool(np.isclose(cost, opt)), "rule": rule})
    d["equals_warm_start_decode"] = d["order"] == d["warm_start_order"]
    if ideal is not None:
        d["tvd_vs_ideal"] = float(0.5 * np.abs(probs - ideal).sum())
        d["ideal_order"] = decode_order(marginals_from_probs(win, ideal))
        d["equals_ideal_decode"] = d["order"] == d["ideal_order"]
        d["p_valid_ideal"] = float(ideal[valid].sum())
    return d
