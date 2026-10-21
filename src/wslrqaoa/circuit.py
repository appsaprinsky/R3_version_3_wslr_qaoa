"""Qiskit circuits: LRWS-QAOA window circuit and the bit-order calibration probe."""
from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit
from . import qubo


def build_circuit(win, gammas, betas, name: str = "window") -> QuantumCircuit:
    """|psi> = prod_l U_M(beta_l) U_C(gamma_l) |phi*>, measured in the computational basis.

    U_C(g) = prod_a RZ(2 g h_a/P) prod_{a<b} RZZ(2 g J_ab/P)      (Ising form of x^T Q x)
    U_M(b) = prod_a RY(theta_a) RZ(-2 b) RY(-theta_a)             (ground state of H_M = warm start)
    """
    n = win.n
    h, J, _ = qubo.ising(win.Q)
    h, J = h / win.P, J / win.P
    qc = QuantumCircuit(n, n, name=name)
    for a in range(n):
        qc.ry(float(win.theta[a]), a)
    for g, b in zip(gammas, betas):
        for a in range(n):
            if h[a] != 0.0:
                qc.rz(float(2 * g * h[a]), a)
        for a in range(n):
            for c in range(a + 1, n):
                if J[a, c] != 0.0:
                    qc.rzz(float(2 * g * J[a, c]), a, c)
        for a in range(n):
            qc.ry(float(-win.theta[a]), a)
            qc.rz(float(-2 * b), a)
            qc.ry(float(win.theta[a]), a)
    qc.measure(range(n), range(n))
    return qc


def unmeasured(qc: QuantumCircuit) -> QuantumCircuit:
    return qc.remove_final_measurements(inplace=False)


def calibration_circuit(n: int, name: str = "calibration") -> QuantumCircuit:
    """X on qubit 0, measure all: Qiskit key order expects '0...01'."""
    qc = QuantumCircuit(n, n, name=name)
    qc.x(0)
    qc.measure(range(n), range(n))
    return qc
