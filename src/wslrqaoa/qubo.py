"""QUBO / Ising formulation of the k-node sub-tour window.

Variables x_{i,t} (candidate i at position t), qubit index a = i*k + t.
E(x) = f(x) + P g(x),  f = path cost,  g = sum of squared one-hot violations.
x^T Q x + const == E(x) with Q upper triangular.
"""
from __future__ import annotations
import numpy as np


def idx(i: int, t: int, k: int) -> int:
    return i * k + t


def penalty_weight(M: np.ndarray, lam: float) -> float:
    """P = lam * k * m, m = largest edge cost in the window. For costs >= 0 and lam > 1
    every minimiser of E is a permutation (Proposition 1 in the paper)."""
    k = M.shape[0] - 1
    m = float(M[0, 1:].max())
    if k > 1:
        off = M[1:, 1:][~np.eye(k, dtype=bool)]
        m = max(m, float(off.max()))
    return lam * k * m if m > 0 else 1.0


def build_qubo(M: np.ndarray, lam: float = 1.5):
    """Return (Q, P, const). Q is upper triangular (diag = linear terms)."""
    k = M.shape[0] - 1
    n = k * k
    P = penalty_weight(M, lam)
    Q = np.zeros((n, n))
    for i in range(k):                                   # first-step edge  u -> c_i
        Q[idx(i, 0, k), idx(i, 0, k)] += M[0, 1 + i]
    for t in range(k - 1):                               # transitions c_i -> c_j
        for i in range(k):
            for j in range(k):
                if i != j:
                    a, b = sorted((idx(i, t, k), idx(j, t + 1, k)))
                    Q[a, b] += M[1 + i, 1 + j]
    for a in range(n):                                   # -2P per variable (rows + columns)
        Q[a, a] += -2.0 * P
    for i in range(k):                                   # row pairs  (same candidate, two positions)
        for t1 in range(k):
            for t2 in range(t1 + 1, k):
                Q[idx(i, t1, k), idx(i, t2, k)] += 2.0 * P
    for t in range(k):                                   # column pairs (same position, two candidates)
        for i1 in range(k):
            for i2 in range(i1 + 1, k):
                Q[idx(i1, t, k), idx(i2, t, k)] += 2.0 * P
    return Q, P, 2.0 * k * P


def ising(Q: np.ndarray):
    """x = (1 - Z)/2  =>  x^T Q x = off + sum_a h_a Z_a + sum_{a<b} J_ab Z_a Z_b."""
    d = np.diag(Q).copy()
    U = np.triu(Q, 1)
    h = -d / 2.0 - (U + U.T).sum(axis=1) / 4.0
    J = U / 4.0
    off = d.sum() / 2.0 + U.sum() / 4.0
    return h, J, off
