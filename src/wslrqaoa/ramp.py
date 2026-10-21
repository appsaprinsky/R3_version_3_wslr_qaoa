"""Two-parameter linear-ramp schedule and its classical optimisation."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from . import sim


def ramp_angles(p: int, delta_gamma: float, delta_beta: float):
    """gamma_l = s_l * Dg,  beta_l = (1 - s_l) * Db,  s_l = (l - 1/2)/p,  l = 1..p."""
    s = (np.arange(1, p + 1) - 0.5) / p
    return delta_gamma * s, delta_beta * (1.0 - s)


def optimise_ramp(E_norm, theta, p, grid=7, g_range=(0.2, 4.0), b_range=(0.2, 3.0), maxiter=100,
                  x0=(1.0, 1.0)):
    """Minimise <E> over (Delta_gamma, Delta_beta): coarse grid, then COBYLA from the best point.
    Only two parameters are optimised for any depth p."""
    trace = []

    def obj(v):
        g, b = ramp_angles(p, float(v[0]), float(v[1]))
        val = float(sim.probabilities(E_norm, theta, g, b) @ E_norm)
        trace.append((float(v[0]), float(v[1]), val))
        return val

    best = None
    for dg in np.linspace(*g_range, grid):
        for db in np.linspace(*b_range, grid):
            v = obj((dg, db))
            if best is None or v < best[0]:
                best = (v, dg, db)
    res = minimize(obj, x0=[best[1], best[2]], method="COBYLA",
                   options={"maxiter": maxiter, "rhobeg": 0.25})
    v = [float(res.x[0]), float(res.x[1])]
    val = float(obj(v))
    if best[0] < val:                      # keep the better of grid / COBYLA
        v, val = [best[1], best[2]], best[0]
    return {"delta_gamma": v[0], "delta_beta": v[1], "energy": val, "evaluations": len(trace)}
