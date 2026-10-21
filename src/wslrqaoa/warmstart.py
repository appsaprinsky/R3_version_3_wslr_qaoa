"""LP-relaxation warm start (Egger, Marecek, Woerner 2021) for the window QUBO."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog


def _assignment_constraints(k):
    n = k * k
    A, b = [], []
    for i in range(k):
        row = np.zeros(n); row[i * k:(i + 1) * k] = 1; A.append(row); b.append(1.0)
    for t in range(k):
        row = np.zeros(n); row[t::k] = 1; A.append(row); b.append(1.0)
    return np.array(A), np.array(b)


def lp_relaxation(M: np.ndarray, mode: str = "linearized", return_value: bool = False):
    """Return x^LP as a (k, k) array, x[i, t] in [0, 1].

    linearized : LP relaxation of the quadratic cost f, obtained by linearising
                 x_{i,t} x_{j,t+1} with flow variables y_{t,i,j}; its optimum is a
                 lower bound on the best window cost.
    surrogate  : linear surrogate cost c_(i,t) = M[u,c_i][t=0] + 1/2 [t<=k-2] sum_{j!=i} M[c_i,c_j]
    Falls back to x = 1/k if HiGHS does not return an optimal solution.
    """
    k = M.shape[0] - 1
    n = k * k
    if k == 1:
        return (np.ones((1, 1)), float(M[0, 1])) if return_value else np.ones((1, 1))
    A_as, b_as = _assignment_constraints(k)
    if mode == "surrogate":
        c = np.zeros(n)
        for i in range(k):
            for t in range(k):
                c[i * k + t] = (M[0, 1 + i] if t == 0 else 0.0)
                if t <= k - 2:
                    c[i * k + t] += 0.5 * sum(M[1 + i, 1 + j] for j in range(k) if j != i)
        res = linprog(c, A_eq=A_as, b_eq=b_as, bounds=(0, 1), method="highs")
        ok = res.status == 0
        x = res.x if ok else None
        val = res.fun if ok else None
    else:
        pairs = [(t, i, j) for t in range(k - 1) for i in range(k) for j in range(k) if i != j]
        m = len(pairs)
        c = np.concatenate([[M[0, 1 + i] if t == 0 else 0.0 for i in range(k) for t in range(k)],
                            [M[1 + i, 1 + j] for (_, i, j) in pairs]])
        A = np.zeros((A_as.shape[0] + 2 * (k - 1) * k, n + m)); b = np.zeros(A.shape[0])
        A[:A_as.shape[0], :n] = A_as; b[:A_as.shape[0]] = b_as
        r = A_as.shape[0]
        for t in range(k - 1):
            for i in range(k):               # sum_j y_{t,i,j} = x_{i,t}
                A[r, i * k + t] = -1
                for q, (tt, ii, jj) in enumerate(pairs):
                    if tt == t and ii == i: A[r, n + q] = 1
                r += 1
            for j in range(k):               # sum_i y_{t,i,j} = x_{j,t+1}
                A[r, j * k + t + 1] = -1
                for q, (tt, ii, jj) in enumerate(pairs):
                    if tt == t and jj == j: A[r, n + q] = 1
                r += 1
        res = linprog(c, A_eq=A, b_eq=b, bounds=(0, 1), method="highs")
        ok = res.status == 0
        x = res.x[:n] if ok else None
        val = res.fun if ok else None
    if x is None:
        out = np.full((k, k), 1.0 / k)
        return (out, None) if return_value else out
    out = np.asarray(x).reshape(k, k)
    return (out, float(val)) if return_value else out


def warm_start_angles(x_lp: np.ndarray, eps: float):
    """Clip to [eps, 1-eps] and return (theta flat, x_clipped); theta = 2 arcsin(sqrt(x))."""
    xc = np.clip(x_lp, eps, 1.0 - eps)
    return 2.0 * np.arcsin(np.sqrt(xc)).reshape(-1), xc
