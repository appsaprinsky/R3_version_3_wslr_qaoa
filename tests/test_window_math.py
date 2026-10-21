import itertools
import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from wslrqaoa import qubo, warmstart, sim, ramp, routing, window as W, circuit as CIR

RNG = np.random.default_rng(5)


def rand_M(k, zero_some=False):
    pts = RNG.random((k + 1, 2)) * 1000
    M = np.linalg.norm(pts[:, None] - pts[None], axis=2) * (1 + 0.2 * RNG.random((k + 1, k + 1)))
    np.fill_diagonal(M, 0)
    if zero_some:
        M[RNG.random(M.shape) < 0.3] = 0.0
    return M


def direct_energy(M, P, x):
    k = M.shape[0] - 1
    x = x.reshape(k, k)
    f = sum(M[0, 1 + i] * x[i, 0] for i in range(k))
    f += sum(M[1 + i, 1 + j] * x[i, t] * x[j, t + 1] for t in range(k - 1) for i in range(k) for j in range(k) if i != j)
    g = ((x.sum(1) - 1) ** 2).sum() + ((x.sum(0) - 1) ** 2).sum()
    return f + P * g


@pytest.mark.parametrize("k", [2, 3, 4])
def test_qubo_equals_cost_plus_penalty(k):
    M = rand_M(k)
    Q, P, const = qubo.build_qubo(M, 1.5)
    X = sim.bit_matrix(k * k)
    E = sim.energies(Q, X) + const
    for s in RNG.integers(0, 2 ** (k * k), 300):
        assert E[s] == pytest.approx(direct_energy(M, P, X[s]), rel=1e-9, abs=1e-6)


def test_self_term_not_double_counted():
    """Regression for the old bug: the -2P diagonal must survive; an all-zero state must cost 2kP."""
    M = rand_M(3)
    Q, P, const = qubo.build_qubo(M, 1.5)
    assert np.allclose(np.diag(Q)[1:], np.diag(Q)[1:])        # sanity
    for a in range(9):
        assert Q[a, a] <= -2 * P + M.max() + 1e-9
    E0 = const + 0.0                                          # x = 0  ->  x^T Q x = 0
    assert E0 == pytest.approx(2 * 3 * P)
    x1 = np.zeros(9); x1[0] = 1                                # one variable set: still violated
    assert (x1 @ Q @ x1 + const) > P - 1e-9


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("zeros", [False, True])
def test_penalty_proposition(k, zeros):
    """Proposition 1: every minimiser of E is a permutation matrix attaining the exact window optimum."""
    M = rand_M(k, zeros)
    Q, P, const = qubo.build_qubo(M, 1.5)
    X = sim.bit_matrix(k * k)
    E = sim.energies(Q, X) + const
    opt = routing.all_order_costs(M)[1].min()
    gs = np.nonzero(np.isclose(E, E.min()))[0]
    for s in gs:
        x = X[s].reshape(k, k)
        assert (x.sum(0) == 1).all() and (x.sum(1) == 1).all()
    assert E.min() == pytest.approx(opt, abs=1e-6)
    viol = ~W._valid_mask(k)
    assert E[viol].min() > opt                                 # every violated state is strictly worse


@pytest.mark.parametrize("k", [2, 3])
def test_ising_mapping(k):
    M = rand_M(k)
    Q, P, const = qubo.build_qubo(M, 1.5)
    h, J, off = qubo.ising(Q)
    X = sim.bit_matrix(k * k)
    Z = 1 - 2 * X
    E = off + Z @ h + np.einsum("si,ij,sj->s", Z, J, Z)
    assert np.allclose(E, sim.energies(Q, X))


def test_lp_surrogate_hand_computed():
    # k = 2: c(i,0) = M[u,c_i] + 0.5*M[c_i,c_j];  c(i,1) = 0.
    M = np.array([[0, 1.0, 5.0], [1.0, 0, 2.0], [5.0, 4.0, 0]])
    x = warmstart.lp_relaxation(M, "surrogate")
    # candidate 0 first: 1 + 0.5*2 = 2 ;  candidate 1 first: 5 + 0.5*4 = 7  ->  choose candidate 0 at position 0
    assert np.allclose(x, [[1, 0], [0, 1]])


@pytest.mark.parametrize("k", [2, 3, 4])
def test_lp_linearized_is_lower_bound(k):
    for _ in range(8):
        M = rand_M(k)
        x, val = warmstart.lp_relaxation(M, "linearized", return_value=True)
        assert np.allclose(x.sum(0), 1) and np.allclose(x.sum(1), 1)
        assert val <= routing.all_order_costs(M)[1].min() + 1e-7      # relaxation value is a lower bound


def test_clipping_and_angles():
    th, xc = warmstart.warm_start_angles(np.array([[1.0, 0.0], [0.0, 1.0]]), 0.25)
    assert xc.min() == 0.25 and xc.max() == 0.75
    assert np.allclose(np.sin(th / 2) ** 2, xc.reshape(-1))


def test_lp_fallback_uniform(monkeypatch):
    class R: status = 1; x = None
    monkeypatch.setattr(warmstart, "linprog", lambda *a, **k: R())
    assert np.allclose(warmstart.lp_relaxation(rand_M(3)), 1 / 3)


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("lp", ["surrogate", "linearized"])
def test_circuit_matches_numpy_and_qiskit(k, lp):
    M = rand_M(k)
    w = W.make_window(M, 1.5, 0.25, lp)
    rc = {"ramp_mode": "optimize", "ramp_grid": 3, "ramp_maxiter": 20, "delta_gamma": 1, "delta_beta": 1}
    a = W.choose_angles(w, 3, rc)
    qc = CIR.build_circuit(w, a["gammas"], a["betas"])
    sv = Statevector(CIR.unmeasured(qc)).probabilities()
    mine = W.ideal_probabilities(w, a["gammas"], a["betas"])
    assert np.abs(sv - mine).max() < 1e-10


def test_mixer_leaves_warm_start_invariant():
    w = W.make_window(rand_M(3), 1.5, 0.25)
    base = W.ideal_probabilities(w, [0.0], [0.0])
    assert np.abs(W.ideal_probabilities(w, [0.0, 0.0], [0.8, -0.3]) - base).max() < 1e-12


def test_linear_ramp_formula():
    g, b = ramp.ramp_angles(4, 2.0, 3.0)
    s = (np.arange(1, 5) - 0.5) / 4
    assert np.allclose(g, 2.0 * s) and np.allclose(b, 3.0 * (1 - s))
    assert np.all(np.diff(g) > 0) and np.all(np.diff(b) < 0)
    g1, b1 = ramp.ramp_angles(1, 2.0, 3.0)
    assert g1[0] == pytest.approx(1.0) and b1[0] == pytest.approx(1.5)


def test_ramp_optimisation_lowers_energy():
    w = W.make_window(rand_M(3), 1.5, 0.25)
    e0 = float(W.ideal_probabilities(w, [0.0], [0.0]) @ w.E_norm)
    r = ramp.optimise_ramp(w.E_norm, w.theta, 2, grid=5, maxiter=40)
    assert r["energy"] < e0


def test_decode_marginals_and_bit_order():
    w = W.make_window(rand_M(2), 1.5, 0.25)          # n = 4, bit a = i*2 + t
    # candidate 1 at position 0, candidate 0 at position 1  -> x(1,0)=1 (a=2), x(0,1)=1 (a=1) -> state 0b0110
    counts = {"0110": 90, "0000": 10}
    v = W.counts_to_vector(w, counts)
    d = W.interpret(w, v)
    assert d["order"] == [1, 0]
    # reversed key order is normalised back
    v2 = W.counts_to_vector(w, {"0110"[::-1]: 90, "0000": 10}, reverse_bits=True)
    assert np.allclose(v, v2)
    assert d["p_valid"] == pytest.approx(0.9)
    with pytest.raises(ValueError):
        W.counts_to_vector(w, {"011": 5})


def test_decode_tie_is_deterministic():
    m = np.full((3, 3), 0.5)
    assert W.decode_order(m) == W.decode_order(m)
    assert sorted(W.decode_order(m)) == [0, 1, 2]


def test_best_valid_rule_picks_cheapest_valid():
    M = np.array([[0, 1.0, 5.0], [1.0, 0, 2.0], [5.0, 4.0, 0]])
    w = W.make_window(M, 1.5, 0.25)
    # both valid states appear; order [0,1] costs 3, order [1,0] costs 9
    counts = {"1001": 40, "0110": 60}                     # 1001 -> x(0,0), x(1,1)  = order [0,1]
    d = W.interpret(w, W.counts_to_vector(w, counts), None, "best_valid")
    assert d["order"] == [0, 1] and d["best_valid_cost"] == pytest.approx(3.0)
