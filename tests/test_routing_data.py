import itertools
import json
import numpy as np
import pytest
from wslrqaoa import routing, amazon, ortools_solver
from make_fixture import make_fixture

RNG = np.random.default_rng(2)


def test_open_cost_has_no_return_leg():
    C = np.array([[0, 1, 9], [9, 0, 2], [3, 9, 0]], float)
    assert routing.open_cost(C, [0, 1, 2]) == 3.0


def test_two_opt_asymmetric_exact_and_local_optimum():
    n = 30
    C = RNG.random((n, n)) * 100; np.fill_diagonal(C, 0)
    tour = [0] + list(RNG.permutation(np.arange(1, n)))
    t2 = routing.two_opt(C, tour)
    assert t2[0] == 0 and sorted(t2) == list(range(n))
    assert routing.open_cost(C, t2) <= routing.open_cost(C, tour)
    best = routing.open_cost(C, t2)
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            t = list(t2); t[i:j + 1] = t[i:j + 1][::-1]
            assert routing.open_cost(C, t) >= best - 1e-7


def test_exact_window_matches_bruteforce():
    M = RNG.random((5, 5)) * 10
    o = routing.exact_window_order(M)
    best = min(routing.window_order_cost(M, list(p)) for p in itertools.permutations(range(4)))
    assert routing.window_order_cost(M, o) == pytest.approx(best)


def test_ortools_matches_bruteforce():
    n = 8
    C = RNG.random((n, n)) * 100; np.fill_diagonal(C, 0)
    res = ortools_solver.solve_open_tsp(C, 0, 20, 4, cost_scale=10000)
    best = min(routing.open_cost(C, [0] + list(p)) for p in itertools.permutations(range(1, n)))
    assert res["status"] == "OPTIMAL" and res["cost"] == pytest.approx(best, rel=1e-3)


def test_sampling_reproducible_and_seed_dependent(tmp_path):
    root = make_fixture(tmp_path, n_routes=6)
    rd, tt, ac = amazon.load_raw(root)
    a, _ = amazon.sample_routes(rd, tt, ac, 3, 2026, log=lambda *_: None)
    b, _ = amazon.sample_routes(rd, tt, ac, 3, 2026, log=lambda *_: None)
    c, _ = amazon.sample_routes(rd, tt, ac, 3, 7, log=lambda *_: None)
    assert [r["route_id"] for r in a] == [r["route_id"] for r in b]
    assert [r["route_id"] for r in a] != [r["route_id"] for r in c]
    r = a[0]
    C = np.array(r["cost_matrix"])
    assert r["amazon_cost"] == pytest.approx(routing.open_cost(C, r["amazon_sequence"]))
    assert r["amazon_sequence"][0] == 0 and sorted(r["amazon_sequence"]) == list(range(r["n_nodes"]))


def test_stop_filter(tmp_path):
    root = make_fixture(tmp_path, n_routes=6)
    rd, tt, ac = amazon.load_raw(root)
    routes, _ = amazon.sample_routes(rd, tt, ac, 2, 1, min_stops=0, max_stops=10, log=lambda *_: None)
    assert all(r["n_nodes"] <= 10 for r in routes)


def test_sequence_repair(tmp_path):
    root = make_fixture(tmp_path, n_routes=1, sizes=(6,))
    rd, tt, ac = amazon.load_raw(root)
    rid = next(iter(rd)); ids = list(rd[rid]["stops"])
    ac[rid]["actual"] = {ids[2]: 0, ids[0]: 1, ids[3]: 2, ids[4]: 3}      # depot not first, two stops missing
    rec, why = amazon.build_route(rid, rd, tt, ac)
    assert why is None and rec["repairs"]
    assert rec["amazon_sequence"][0] == 0 and sorted(rec["amazon_sequence"]) == list(range(6))


def test_missing_travel_time_skips_route(tmp_path):
    root = make_fixture(tmp_path, n_routes=1, sizes=(6,))
    rd, tt, ac = amazon.load_raw(root)
    rid = next(iter(tt)); a = list(tt[rid])[1]; b = list(tt[rid])[2]
    del tt[rid][a][b]
    rec, why = amazon.build_route(rid, rd, tt, ac)
    assert rec is None and "missing travel time" in why
