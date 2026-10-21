"""Receding-horizon construction (Algorithm 1): candidate selection, commit, finalisation."""
from __future__ import annotations
import hashlib
import numpy as np
from . import routing


def stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:12], 16)


def initial_state(n_nodes: int, depot: int = 0) -> dict:
    return {"tour": [depot], "unvisited": [i for i in range(n_nodes) if i != depot], "iteration": 0}


def plan_step(state: dict, C: np.ndarray, k: int, b: int, rho: float, seed: int, run_key: str):
    """Next window or None when every node is committed.
    Candidates: n_near closest unvisited nodes (ties -> lower index) + n_explore uniform draws."""
    U = list(state["unvisited"])
    if not U:
        return None
    u = state["tour"][-1]
    kb = min(k, len(U))
    n_explore = int(np.floor(rho * kb))
    if kb > 1:
        n_explore = min(n_explore, kb - 1)
    n_near = kb - n_explore
    ranked = sorted(U, key=lambda x: (C[u, x], x))
    near = ranked[:n_near]
    rest = ranked[n_near:]
    explore = []
    if n_explore > 0:
        rng = np.random.default_rng(np.random.SeedSequence([seed, stable_int(run_key), state["iteration"]]))
        explore = [int(x) for x in rng.choice(rest, size=n_explore, replace=False)]
    cands = [int(x) for x in near] + explore
    return {"u": int(u), "cands": cands, "k": kb, "commit": min(b, kb)}


def window_matrix(C: np.ndarray, spec: dict) -> np.ndarray:
    nodes = [spec["u"]] + list(spec["cands"])
    return C[np.ix_(nodes, nodes)]


def apply_order(state: dict, spec: dict, order: list[int]) -> dict:
    """Commit the first `commit` nodes of the decoded order (order = candidate indices)."""
    nodes = [spec["cands"][i] for i in order]
    commit = nodes[:spec["commit"]]
    state["tour"] = state["tour"] + commit
    state["unvisited"] = [x for x in state["unvisited"] if x not in set(commit)]
    state["iteration"] += 1
    return state


def finalize(C: np.ndarray, tour: list[int]) -> dict:
    pre = routing.open_cost(C, tour)
    final = routing.two_opt(C, tour)
    return {"tour_pre_2opt": list(tour), "cost_pre_2opt": pre, "tour": final, "cost": routing.open_cost(C, final)}
