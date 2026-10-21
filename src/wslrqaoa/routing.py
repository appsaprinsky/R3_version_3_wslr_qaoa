"""Classical routing utilities: open-route cost, asymmetric-correct 2-opt, exact window solver."""
from __future__ import annotations
import itertools
import numpy as np


def open_cost(C: np.ndarray, tour) -> float:
    """Open-route cost  C(pi) = sum_{t=0}^{n-2} C[pi_t, pi_{t+1}]  (no return leg).

    This is the single cost function used for Amazon, every run and every baseline.
    """
    t = np.asarray(tour, dtype=int)
    return float(C[t[:-1], t[1:]].sum()) if len(t) > 1 else 0.0


def two_opt(C: np.ndarray, tour, max_moves: int | None = None):
    """Open-path 2-opt with the depot (tour[0]) fixed.

    Reversing tour[i..j] changes the boundary edges *and*, for asymmetric C, the
    direction of every internal edge. The exact cost change is therefore
        delta = C[a,pj] + B(i,j) + C[pi,b] - C[a,pi] - F(i,j) - C[pj,b]
    with a = tour[i-1], b = tour[j+1] (absent if j is the last position), F the
    forward and B the backward cost of the segment. Prefix sums give F, B in O(1).
    """
    tour = np.array(tour, dtype=int)
    n = len(tour)
    if n < 3:
        return tour.tolist()
    cap = max_moves if max_moves is not None else max(100, 50 * n)
    for _ in range(cap):
        fwd = np.concatenate([[0.0], np.cumsum(C[tour[:-1], tour[1:]])])   # PF[t] = sum_{s<t} C[pi_s, pi_{s+1}]
        bwd = np.concatenate([[0.0], np.cumsum(C[tour[1:], tour[:-1]])])   # PB[t] = sum_{s<t} C[pi_{s+1}, pi_s]
        best = (-1e-9, None, None)
        for i in range(1, n - 1):
            js = np.arange(i + 1, n)
            a, pi = tour[i - 1], tour[i]
            pj = tour[js]
            new = C[a, pj] + (bwd[js] - bwd[i])
            old = C[a, pi] + (fwd[js] - fwd[i])
            has_next = js < n - 1
            b = tour[np.minimum(js + 1, n - 1)]
            new = new + np.where(has_next, C[pi, b], 0.0)
            old = old + np.where(has_next, C[pj, b], 0.0)
            delta = new - old
            m = int(np.argmin(delta))
            if delta[m] < best[0]:
                best = (float(delta[m]), i, int(js[m]))
        if best[1] is None:
            break
        _, i, j = best
        tour[i:j + 1] = tour[i:j + 1][::-1].copy()
    return tour.tolist()


def window_order_cost(M: np.ndarray, order) -> float:
    """Cost of visiting window candidates `order` (0-based candidate indices) from u.
    M is the (k+1)x(k+1) window matrix, index 0 = u, index 1+i = candidate i."""
    cost = M[0, 1 + order[0]]
    for a, b in zip(order[:-1], order[1:]):
        cost += M[1 + a, 1 + b]
    return float(cost)


def all_order_costs(M: np.ndarray):
    k = M.shape[0] - 1
    perms = list(itertools.permutations(range(k)))
    costs = np.array([window_order_cost(M, p) for p in perms])
    return perms, costs


def exact_window_order(M: np.ndarray):
    """Exact optimum of the window (brute force over k! orders; ties -> lexicographically first)."""
    perms, costs = all_order_costs(M)
    return list(perms[int(np.argmin(costs))])


def nearest_neighbor_tour(C: np.ndarray, depot: int = 0):
    """Greedy nearest-neighbour open tour from the depot (ties -> lower index)."""
    n = len(C)
    tour, left = [depot], set(range(n)) - {depot}
    while left:
        u = tour[-1]
        v = min(left, key=lambda x: (C[u, x], x))
        tour.append(v)
        left.remove(v)
    return tour
