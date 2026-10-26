"""Tests of the three-approach experiment: solver equivalence, fairness invariants, separate outputs, nesting."""
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wslrqaoa import engine as E, experiment, ortools_solver, routing, heuristic
from wslrqaoa.config import (ConfigError, algorithm_window, approach_fingerprint, load_config)
from make_fixture import make_fixture
import test_engine as T

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(9)


def rand_window(k):
    pts = RNG.random((k + 1, 2)) * 1000
    M = np.linalg.norm(pts[:, None] - pts[None], axis=2) * (1 + 0.2 * RNG.random((k + 1, k + 1)))
    np.fill_diagonal(M, 0)
    return M


# ---------------------------------------------------------------- the window solver is exact
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_cpsat_window_equals_bruteforce(k):
    for _ in range(6):
        M = rand_window(k)
        res = ortools_solver.solve_window(M, 10.0, 1000, 1, 2026)
        assert res["status"] == "OPTIMAL"
        assert sorted(res["order"]) == list(range(k))
        best = routing.window_order_cost(M, routing.exact_window_order(M))
        assert res["cost"] == pytest.approx(best, abs=k / 1000 + 1e-9)


def test_baseline_has_no_hint_and_a_consistent_anytime_trace():
    n = 12
    C = RNG.random((n, n)) * 100
    np.fill_diagonal(C, 0)
    res = ortools_solver.solve_open_tsp(C, 0, 20, 2, None, 100, 2026, 0.0, True)
    assert res["status"] == "OPTIMAL" and res["optimal"]
    tr = res["trace"]
    assert tr and all(a[0] <= b[0] and a[2] >= b[2] for a, b in zip(tr, tr[1:]))     # time up, objective down
    assert tr[-1][2] == pytest.approx(res["cost"], abs=n / 100)                      # last point = final (rounded) cost
    assert res["lower_bound"] <= res["cost"] + 1e-6
    assert res["cost"] == pytest.approx(routing.open_cost(C, res["tour"]))           # reported on the unrounded matrix


def test_anytime_best_and_paired_test():
    trace = [[0.1, 0.2, 100.0], [0.5, 0.9, 80.0], [2.0, 3.0, 70.0]]
    assert experiment.anytime_best(trace, 0.05, 0) is None
    assert experiment.anytime_best(trace, 0.6, 0) == 80.0
    assert experiment.anytime_best(trace, 5.0, 1) == 70.0
    assert experiment.anytime_best(trace, None, 0) is None
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 5.0, 5.0], "y": [2.0, 2.0, 4.0, 4.0, 6.0]})
    t = experiment.paired_test(df, "x", "y")
    assert (t["n"], t["x_wins"], t["ties"], t["y_wins"]) == (5, 3, 1, 1)
    assert t["wilcoxon_p"] is not None and 0 < t["wilcoxon_p"] <= 1


def test_comparison_rows_gap_definition():
    routes = {"r1": {"route_id": "r1", "n_nodes": 10, "amazon_cost": 120.0}}
    mk = lambda mode, name, cost, pre=None, **kw: {"route_id": "r1", "mode": mode, "name": name, "status": "finished",
                                                  "result": {"cost": cost, "cost_pre_2opt": pre or cost, **kw}, "timing": {"compute_s": 1.0}, "windows": []}
    full = {"r1": {**mk("ortools_full", "ortools_full", 100.0, cost_with_2opt=100.0), "ortools": {"status": "OPTIMAL", "lower_bound": 100.0, "trace": [[0.1, 0.1, 100.0]]}}}
    core = {"r1": mk("ortools_core", "ortools_core", 110.0, 130.0)}
    qc = {"r1": mk("qpu", "Tuna-17", 121.0, 200.0)}
    df = experiment.comparison_rows(routes, full, core, qc, {})
    r = df.iloc[0]
    assert r["best_known_cost"] == 100.0
    assert (r["ortools_gap_pct"], r["core_gap_pct"], r["qc_gap_pct"]) == pytest.approx((0.0, 10.0, 21.0))
    assert r["amazon_gap_pct"] == pytest.approx(20.0)


# ---------------------------------------------------------------- the two algorithm approaches differ ONLY in the window solver
def test_core_equals_exact_window_run_and_shares_k_b(tmp_path):
    routes = T.make_routes(tmp_path, size=19)
    cfg = T.make_cfg()
    cfg["quantum_inspire"]["backends"] = ["sim:Tuna-17"]
    k, b = algorithm_window(cfg)
    assert (k, b) == (4, 4)
    qc = T.sim_engine(tmp_path, routes, cfg, ("Tuna-17",), sub="qc")          # includes the brute-force 'exact' control at the same k, b
    qc.run_all()
    core = E.Engine(cfg, routes, {}, tmp_path / "core", log=lambda *_: None, approach="AlgorithmWithORtoolCore")
    core.create_runs()
    core.run_all()
    exact = [r for r in qc.runs if r["name"] == f"exact_k{k}"][0]
    cr = core.runs[0]
    assert cr["status"] == "finished" and (cr["k"], cr["b"]) == (k, b) == (exact["k"], exact["b"])
    assert cr["result"]["tour"] == exact["result"]["tour"]                       # same candidates, same optimal windows, same 2-opt
    assert cr["result"]["cost_pre_2opt"] == pytest.approx(exact["result"]["cost_pre_2opt"])
    assert all(w["ortools_status"] == "OPTIMAL" for w in cr["windows"] if not w.get("trivial"))
    assert cr["timing"]["subsolver_s"] > 0


def test_core_run_fails_loudly_when_a_window_is_not_proven_optimal(tmp_path, monkeypatch):
    routes = T.make_routes(tmp_path, size=11)
    cfg = T.make_cfg()
    cfg["quantum_inspire"]["backends"] = ["sim:Tuna-17"]
    monkeypatch.setattr(ortools_solver, "solve_window", lambda *a, **k: {"order": [0, 1], "cost": 1.0, "status": "FEASIBLE", "solver_wall_s": 0.0})
    eng = E.Engine(cfg, routes, {}, tmp_path / "core", log=lambda *_: None, approach="AlgorithmWithORtoolCore")
    eng.create_runs()
    eng.run_all()
    assert eng.runs[0]["status"] == "failed" and "optimal" in eng.runs[0]["failure"]


def test_baseline_engine_run(tmp_path):
    routes = T.make_routes(tmp_path, size=14)
    cfg = T.make_cfg()
    cfg["ortools_full"].update(time_limit_s=20.0, workers=2)
    eng = E.Engine(cfg, routes, {}, tmp_path / "full", log=lambda *_: None, approach="ORtoolsOutput")
    eng.create_runs()
    eng.run_all()
    r = eng.runs[0]
    C = routes[0]["C"]
    assert r["status"] == "finished" and r["ortools"]["hint"] == "none" and r["ortools"]["status"] == "OPTIMAL"
    assert r["result"]["cost"] == pytest.approx(routing.open_cost(C, r["result"]["tour"]))
    assert r["result"]["cost_with_2opt"] <= r["result"]["cost"] + 1e-9                  # sensitivity column never worse
    assert r["result"]["cost_pre_2opt"] == r["result"]["cost"]                          # the baseline is raw output
    assert r["timing"]["subsolver_s"] > 0 and r["timing"]["cpu_s"] > 0


def test_qc_windows_are_checked_with_cpsat_and_diagnostics_are_not_timed(tmp_path):
    routes = T.make_routes(tmp_path, size=15)
    cfg = T.make_cfg()
    eng = T.sim_engine(tmp_path, routes, cfg, ("Tuna-9",), sub="qcv")
    eng.run_all()
    run = [r for r in eng.runs if r["name"] == "sim-Tuna-9"][0]
    ws = [w for w in run["windows"] if not w.get("trivial")]
    assert ws and all(w["ortools_matches_bruteforce"] and w["ortools_window_cost"] <= w["window_cost"] + 1e-6 for w in ws)
    assert run["timing"]["diag_s"] > 0 and run["timing"]["compute_s"] >= 0
    nn = [r for r in eng.runs if r["name"] == "nn"][0]
    assert nn["timing"]["diag_s"] == 0                                                   # only QC windows are verified


def test_same_candidates_and_commit_rule_for_both_algorithm_arms():
    C = RNG.random((30, 30)) * 100
    np.fill_diagonal(C, 0)
    st = heuristic.initial_state(30)
    a = heuristic.plan_step(st, C, 4, 4, 0.0, 2026, "AlgorithmWithQC/x")
    b = heuristic.plan_step(st, C, 4, 4, 0.0, 2026, "AlgorithmWithORtoolCore/x")
    assert a == b                                                                         # no randomness without exploration


# ---------------------------------------------------------------- config
def test_new_config_rules(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text("[prepare]\nmin_stops = 40\n")
    with pytest.raises(ValueError, match="min_stops"):
        load_config(f)
    f.write_text('[quantum_inspire]\nbackends = ["Tuna-9", "Tuna-17"]\n')
    with pytest.raises(ValueError, match="exactly one"):
        load_config(f)
    f.write_text('[experiment]\napproaches = ["Nonsense"]\n')
    with pytest.raises(ValueError):
        load_config(f)


def test_approach_fingerprints_are_independent():
    cfg = load_config(ROOT / "config.toml")
    base = {a: approach_fingerprint(cfg, a) for a in cfg["experiment"]["approaches"]}
    cfg2 = load_config(ROOT / "config.toml")
    cfg2["ortools_full"]["time_limit_s"] = 7.0
    changed = {a: approach_fingerprint(cfg2, a) for a in base}
    assert changed["ORtoolsOutput"] != base["ORtoolsOutput"]
    assert changed["AlgorithmWithQC"] == base["AlgorithmWithQC"] and changed["AlgorithmWithORtoolCore"] == base["AlgorithmWithORtoolCore"]
    cfg3 = load_config(ROOT / "config.toml")
    cfg3["qaoa"]["p"] = 3
    assert approach_fingerprint(cfg3, "AlgorithmWithQC") != base["AlgorithmWithQC"]
    assert approach_fingerprint(cfg3, "AlgorithmWithORtoolCore") == base["AlgorithmWithORtoolCore"]       # p is irrelevant for the CP-SAT core


# ---------------------------------------------------------------- whole programme: folders, independence, nesting
CONFIG = '''
[prepare]
data_dir = "{data}"
input_dir = "input"
num_routes = {n}
seed = 5
exclude_route_ids = {excl}
[experiment]
approaches = {approaches}
[ortools_full]
time_limit_s = 10.0
workers = 2
[ortools_core]
window_time_limit_s = 10.0
[qaoa]
k = "auto"
p = 1
shots = 256
ramp_mode = "fixed"
delta_gamma = 0.84
delta_beta = 0.65
[quantum_inspire]
backends = ["sim:Tuna-17"]
poll_interval_s = 1
[compare]
nearest_neighbor = true
[output]
output_dir = "output"
figures = false
visualise_subroute = false
'''


def _setup(tmp_path, monkeypatch, n, approaches, excl="[]", data=None):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(CONFIG.format(data=data, n=n, excl=excl, approaches=json.dumps(approaches)))


def _load(name):
    sys.path.insert(0, str(ROOT))
    return importlib.import_module(name)


def test_three_folders_independent_runs_nesting_and_tamper_checks(tmp_path, monkeypatch, capsys):
    data = make_fixture(tmp_path / "raw", n_routes=6, sizes=(12, 9, 14, 11, 10, 13)).parent.parent
    prep, main = _load("prepare_input"), _load("main")
    # (1) only the baseline first: other folders must not appear
    _setup(tmp_path, monkeypatch, 2, ["ORtoolsOutput"], data=data)
    assert prep.main() == 0 and main.main() == 0
    out = tmp_path / "output"
    assert (out / "ORtoolsOutput" / "results" / "results.csv").exists()
    assert not (out / "AlgorithmWithQC").exists() and not (out / "AlgorithmWithORtoolCore").exists()
    ids = sorted(p.stem for p in (tmp_path / "input").glob("route_*.json"))
    assert len(ids) == 2
    base_runs = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in (out / "ORtoolsOutput" / "runs").glob("*.json")}
    # (2) add the other two approaches: the finished baseline is not recomputed
    _setup(tmp_path, monkeypatch, 2, ["ORtoolsOutput", "AlgorithmWithORtoolCore", "AlgorithmWithQC"], data=data)
    assert main.main() == 0
    for a in ("ORtoolsOutput", "AlgorithmWithORtoolCore", "AlgorithmWithQC"):
        assert (out / a / "results").exists() and (out / a / "experiment.json").exists()
    assert (out / "Comparison" / "paired_comparison.csv").exists() and (out / "Comparison" / "statistics.md").exists()
    assert {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in (out / "ORtoolsOutput" / "runs").glob("*.json")} == base_runs
    cmp = pd.read_csv(out / "Comparison" / "paired_comparison.csv")
    assert len(cmp) == 2 and cmp[["ortools_cost", "core_cost", "qc_cost"]].notna().all().all()
    # (3) nested sample: more routes only ADD routes; existing runs are untouched
    qc_before = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in (out / "AlgorithmWithQC" / "runs").glob("*.json")}
    _setup(tmp_path, monkeypatch, 4, ["ORtoolsOutput", "AlgorithmWithORtoolCore", "AlgorithmWithQC"], data=data)
    assert prep.main() == 0
    ids4 = sorted(p.stem for p in (tmp_path / "input").glob("route_*.json"))
    assert len(ids4) == 4 and set(ids) <= set(ids4)
    assert main.main() == 0
    qc_after = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in (out / "AlgorithmWithQC" / "runs").glob("*.json")}
    assert all(qc_after[n] == h for n, h in qc_before.items()) and len(qc_after) == 2 * len(qc_before)
    assert len(pd.read_csv(out / "Comparison" / "paired_comparison.csv")) == 4
    # (4) a changed route file or changed settings must be refused
    f = sorted((tmp_path / "input").glob("route_*.json"))[0]
    d = json.loads(f.read_text()); d["amazon_cost"] += 1.0
    f.write_text(json.dumps(d))
    capsys.readouterr()
    assert main.main() == 1 and "changed" in capsys.readouterr().out
    d["amazon_cost"] -= 1.0
    f.write_text(json.dumps(d))
    (tmp_path / "config.toml").write_text(CONFIG.format(data=data, n=4, excl="[]", approaches=json.dumps(["ORtoolsOutput"])).replace("time_limit_s = 10.0\nworkers = 2", "time_limit_s = 11.0\nworkers = 2"))
    assert main.main() == 1


def test_excluded_routes_are_never_selected(tmp_path, monkeypatch):
    data = make_fixture(tmp_path / "raw", n_routes=6, sizes=(12, 9, 14, 11, 10, 13)).parent.parent
    prep = _load("prepare_input")
    _setup(tmp_path, monkeypatch, 3, ["ORtoolsOutput"], data=data)
    assert prep.main() == 0
    first = sorted(p.stem.replace("route_", "") for p in (tmp_path / "input").glob("route_*.json"))
    import shutil
    shutil.rmtree(tmp_path / "input")
    _setup(tmp_path, monkeypatch, 3, ["ORtoolsOutput"], excl=json.dumps(first[:1]), data=data)
    assert prep.main() == 0
    second = sorted(p.stem.replace("route_", "") for p in (tmp_path / "input").glob("route_*.json"))
    assert first[0] not in second and len(second) == 3
