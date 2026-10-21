"""Exact / best-found open-TSP baseline with Google OR-Tools CP-SAT.

Open TSP with fixed start d: add a dummy node z; arcs z->d (cost 0) and i->z (cost 0, any end node);
AddCircuit forces one Hamiltonian circuit over the n+1 nodes, i.e. a Hamiltonian path from d.
Costs are scaled to integers (cost_scale) for the solver; the reported cost is recomputed from the raw matrix.
"""
from __future__ import annotations
import time
import numpy as np
from . import routing


def solve_open_tsp(C: np.ndarray, depot: int = 0, time_limit_s: float = 120.0, workers: int = 8,
                   hint_tour=None, cost_scale: int = 100) -> dict:
    from ortools.sat.python import cp_model
    n = len(C)
    if n <= 2:
        tour = [depot] + [i for i in range(n) if i != depot]
        return {"cost": routing.open_cost(C, tour), "tour": tour, "status": "OPTIMAL", "lower_bound": routing.open_cost(C, tour),
                "optimal": True, "wall_time_s": 0.0, "time_limit_s": time_limit_s, "gap_pct": 0.0}
    model = cp_model.CpModel()
    z = n
    Ci = np.rint(C * cost_scale).astype(np.int64)
    lit, arcs, obj = {}, [], []
    for i in range(n):
        for j in range(n):
            if i == j or j == depot:
                continue
            v = model.NewBoolVar(f"a{i}_{j}"); lit[(i, j)] = v
            arcs.append((i, j, v)); obj.append(int(Ci[i, j]) * v)
    for i in range(n):
        if i == depot:
            continue
        v = model.NewBoolVar(f"e{i}"); lit[(i, z)] = v; arcs.append((i, z, v))
    v = model.NewBoolVar("s"); lit[(z, depot)] = v; arcs.append((z, depot, v))
    model.Add(v == 1)
    model.AddCircuit(arcs)
    model.Minimize(sum(obj))
    if hint_tour is not None:
        t = list(hint_tour)
        used = {(t[x], t[x + 1]) for x in range(len(t) - 1)} | {(t[-1], z), (z, depot)}
        for key, var in lit.items():
            model.AddHint(var, 1 if key in used else 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_workers = int(workers)
    solver.parameters.random_seed = 2026
    t0 = time.time()
    status = solver.Solve(model)
    wall = time.time() - t0
    name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"cost": None, "tour": None, "status": name, "lower_bound": None, "optimal": False,
                "wall_time_s": wall, "time_limit_s": time_limit_s, "gap_pct": None}
    nxt = {i: j for (i, j), var in lit.items() if solver.Value(var)}
    tour, cur = [depot], depot
    while True:
        cur = nxt[cur]
        if cur == z:
            break
        tour.append(cur)
    cost = routing.open_cost(C, tour)
    lb = solver.BestObjectiveBound() / cost_scale
    return {"cost": cost, "tour": tour, "status": name, "lower_bound": lb, "optimal": status == cp_model.OPTIMAL,
            "wall_time_s": wall, "time_limit_s": time_limit_s, "gap_pct": 100.0 * (cost - lb) / cost if cost else 0.0}
