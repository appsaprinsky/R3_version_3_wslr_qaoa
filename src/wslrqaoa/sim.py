"""Exact statevector simulator for the warm-start QAOA window circuit (numpy).

Qubit a <-> bit a of the basis-state index (little-endian, as in Qiskit).
"""
from __future__ import annotations
from functools import reduce
import numpy as np


def bit_matrix(n: int) -> np.ndarray:
    """(2^n, n) array of the bits of every basis state, column a = qubit a."""
    s = np.arange(2 ** n, dtype=np.int64)
    return ((s[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)


def energies(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """x^T Q x for every basis state (Q upper triangular)."""
    return np.einsum("si,ij,sj->s", X, Q, X, optimize=True)


def initial_state(theta: np.ndarray) -> np.ndarray:
    qs = [np.array([np.cos(t / 2), np.sin(t / 2)], dtype=complex) for t in theta]
    return reduce(np.kron, qs[::-1])              # qubit n-1 is the most significant factor


def mixer_matrix(theta: float, beta: float) -> np.ndarray:
    """RY(theta) RZ(-2 beta) RY(-theta) = exp(i beta (sin(theta) X + cos(theta) Z))."""
    c, s = np.cos(beta), np.sin(beta)
    return np.array([[c + 1j * s * np.cos(theta), 1j * s * np.sin(theta)],
                     [1j * s * np.sin(theta), c - 1j * s * np.cos(theta)]])


def probabilities(E_norm: np.ndarray, theta: np.ndarray, gammas, betas) -> np.ndarray:
    """|psi|^2 for  prod_l U_M(beta_l) U_C(gamma_l) |phi*>,  U_C = exp(-i gamma E_norm(x))."""
    n = len(theta)
    psi = initial_state(theta)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * E_norm)
        for a in range(n):
            U = mixer_matrix(theta[a], b)
            psi = np.einsum("ij,pjq->piq", U, psi.reshape(2 ** (n - 1 - a), 2, 2 ** a)).reshape(-1)
    p = np.abs(psi) ** 2
    return p / p.sum()
