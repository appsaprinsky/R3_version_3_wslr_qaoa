"""OR-Tools CP-SAT for the open TSP: used for the full-route baseline (ORtoolsOutput) and, on tiny sub-instances,
as the window solver of AlgorithmWithORtoolCore. One model builder serves both, so both approaches use the same code path.

Open TSP with fixed start d: add a dummy node z; arcs z->d (cost 0, forced) and i->z (cost 0, any end node);
AddCircuit forces one Hamiltonian circuit over the n+1 nodes, i.e. a Hamiltonian path from d.
Costs are scaled to integers (cost_scale) for the solver; the reported cost is always recomputed from the raw matrix.
No start solution is given unless hint_tour is passed (the experiment never passes one).
"""
from __future__ import annotations
import time
import numpy as np
from . import routing


def _trace_callback(cp_model, scale):
    class Trace(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.trace = []

        def on_solution_callback(self):
            self.trace.append([float(self.WallTime()), float(self.UserTime()), float(self.ObjectiveValue()) / scale])
    return Trace()


def solve_open_tsp(C: np.ndarray, depot: int = 0, time_limit_s: float = 120.0, workers: int = 8,
                   hint_tour=None, cost_scale: int = 100, seed: int = 2026,
                   relative_gap_limit: float | None = None, anytime: bool = False) -> dict:
    """Solve the open TSP from `depot`. Returns cost (raw matrix), tour, status, lower bound, gap, timings and,
    if anytime=True, the improvement trace [[wall_s, cpu_s, objective], ...] of the solver (objective in scaled costs)."""
    from ortools.sat.python import cp_model
    n = len(C)
    if n <= 2:
        tour = [depot] + [i for i in range(n) if i != depot]
        c = routing.open_cost(C, tour)
        return {"cost": c, "tour": tour, "status": "OPTIMAL", "lower_bound": c, "optimal": True, "wall_time_s": 0.0,
                "solver_wall_s": 0.0, "solver_user_s": 0.0, "time_limit_s": time_limit_s, "gap_pct": 0.0, "trace": []}
    t_all = time.perf_counter()
    model = cp_model.CpModel()
    z = n
    Ci = np.rint(C * cost_scale).astype(np.int64)
    lit, arcs, obj = {}, [], []
    for i in range(n):
        for j in range(n):
            if i == j or j == depot:
                continue
            v = model.NewBoolVar(f"a{i}_{j}")
            lit[(i, j)] = v
            arcs.append((i, j, v))
            obj.append(int(Ci[i, j]) * v)
    for i in range(n):
        if i == depot:
            continue
        v = model.NewBoolVar(f"e{i}")
        lit[(i, z)] = v
        arcs.append((i, z, v))
    v = model.NewBoolVar("s")
    lit[(z, depot)] = v
    arcs.append((z, depot, v))
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
    solver.parameters.random_seed = int(seed)
    if relative_gap_limit is not None:
        solver.parameters.relative_gap_limit = float(relative_gap_limit)
    cb = _trace_callback(cp_model, cost_scale) if anytime else None
    status = solver.Solve(model, cb) if cb is not None else solver.Solve(model)
    wall = time.perf_counter() - t_all
    name = solver.StatusName(status)
    out = {"status": name, "wall_time_s": wall, "solver_wall_s": float(solver.WallTime()),
           "solver_user_s": float(solver.UserTime()), "time_limit_s": time_limit_s, "trace": cb.trace if cb else []}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out.update({"cost": None, "tour": None, "lower_bound": None, "optimal": False, "gap_pct": None})
        return out
    nxt = {i: j for (i, j), var in lit.items() if solver.Value(var)}
    tour, cur = [depot], depot
    while True:
        cur = nxt[cur]
        if cur == z:
            break
        tour.append(cur)
    cost = routing.open_cost(C, tour)
    lb = solver.BestObjectiveBound() / cost_scale
    out.update({"cost": cost, "tour": tour, "lower_bound": lb, "optimal": status == cp_model.OPTIMAL,
                "gap_pct": 100.0 * (cost - lb) / cost if cost else 0.0})
    return out


def solve_window(M: np.ndarray, time_limit_s: float = 10.0, cost_scale: int = 1000, workers: int = 1,
                 seed: int = 2026) -> dict:
    """Best visiting order of a window: M is the (k+1)x(k+1) window matrix (index 0 = start node u, 1+i = candidate i).
    Same CP-SAT model as the full-route baseline. Returns order (candidate indices), its raw cost, status, solver time."""
    k = M.shape[0] - 1
    if k == 1:
        return {"order": [0], "cost": float(M[0, 1]), "status": "OPTIMAL", "solver_wall_s": 0.0}
    res = solve_open_tsp(M, 0, time_limit_s, workers, None, cost_scale, seed)
    if res["tour"] is None:
        return {"order": None, "cost": None, "status": res["status"], "solver_wall_s": res["solver_wall_s"]}
    order = [i - 1 for i in res["tour"][1:]]
    return {"order": order, "cost": routing.window_order_cost(M, order), "status": res["status"],
            "solver_wall_s": res["solver_wall_s"]}
