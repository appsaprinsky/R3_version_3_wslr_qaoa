#!/usr/bin/env python3
"""Pilot behind Remark 1 of the paper: does the QAOA change the decoded order of a window?

Ideal simulation on random synthetic k=3 windows (no Amazon data, no Quantum Inspire). For several
warm-start regularisations epsilon it compares, as excess cost over the exact window optimum:
  QAOA decode | warm-start-only decode | nearest-first order.
Usage: python scripts/pilot_warmstart_bias.py        (about 1 minute)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from wslrqaoa import routing, window as W

K, N, P = 3, 120, 2
rng = np.random.default_rng(11)


def random_window(k):
    pts = rng.random((k + 1, 2)) * 1000
    M = np.linalg.norm(pts[:, None] - pts[None], axis=2) * (1 + 0.2 * rng.random((k + 1, k + 1)))
    np.fill_diagonal(M, 0)
    o = np.argsort(M[0, 1:])
    idx = np.concatenate([[0], 1 + o])
    return M[np.ix_(idx, idx)]


wins = [random_window(K) for _ in range(N)]
exact = np.array([routing.all_order_costs(M)[1].min() for M in wins])
nn = np.array([routing.window_order_cost(M, list(range(K))) for M in wins])
rc = {"ramp_mode": "optimize", "ramp_grid": 7, "ramp_maxiter": 60, "delta_gamma": 1.0, "delta_beta": 1.0}
excess = lambda c: 100 * (np.mean(c) - exact.mean()) / exact.mean()
print(f"{N} random windows, k={K}, p={P}, ideal simulation, LP cost = surrogate")
print(f"nearest-first order: excess over exact {excess(nn):.2f}%")
print("epsilon | QAOA decode | warm-start-only | QAOA decode == warm-start decode")
for eps in (0.25, 0.4, 0.46, 0.5):
    q, w, same = [], [], []
    for M in wins:
        win = W.make_window(M, 1.5, eps, "surrogate")
        a = W.choose_angles(win, P, rc)
        d = W.interpret(win, W.ideal_probabilities(win, a["gammas"], a["betas"]))
        q.append(d["cost"]); w.append(routing.window_order_cost(M, d["warm_start_order"]))
        same.append(d["equals_warm_start_decode"])
    print(f"{eps:7.2f} | {excess(q):10.2f}% | {excess(w):14.2f}% | {np.mean(same):.2f}")
