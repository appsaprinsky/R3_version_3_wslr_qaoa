import copy
import numpy as np
import pytest
from wslrqaoa import amazon, engine as E, heuristic, routing
from wslrqaoa.config import load_config, parse_backend, window_size
from wslrqaoa.gateways import SimGateway, QuantumInspireGateway, AuthError
from wslrqaoa.io_utils import read_json
from make_fixture import make_fixture
import fakes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_cfg(**qaoa):
    cfg = load_config(ROOT / "config.toml")
    cfg["qaoa"].update({"ramp_grid": 3, "ramp_maxiter": 25, "shots": 512, **qaoa})
    cfg["quantum_inspire"]["poll_interval_s"] = 0
    return cfg


def make_routes(tmp_path, size=7):
    root = make_fixture(tmp_path / "raw", n_routes=1, sizes=(size,))
    rd, tt, ac = amazon.load_raw(root)
    recs, _ = amazon.sample_routes(rd, tt, ac, 1, 1, log=lambda *_: None)
    for r in recs:
        r["C"] = np.array(r["cost_matrix"])
    return recs


def sim_engine(tmp_path, routes, cfg, backends=("Tuna-5",), reverse=False, sub="out"):
    gws = {}
    for b in backends:
        g = SimGateway(b, parse_backend(b)[1], 1, reverse)
        g.connect(); g.k = window_size(cfg["qaoa"]["k"], g.qubits); gws[f"sim-{b}"] = g
    eng = E.Engine(cfg, routes, gws, tmp_path / sub, log=lambda *_: None)
    eng.idle_sleep = 0
    eng.create_runs()
    return eng


def test_end_to_end_sim_valid_tours(tmp_path):
    routes = make_routes(tmp_path)
    cfg = make_cfg()
    eng = sim_engine(tmp_path, routes, cfg, ("Tuna-5", "Tuna-9"))
    eng.run_all()
    n = routes[0]["n_nodes"]
    for r in eng.runs:
        assert r["status"] == "finished"
        t = r["result"]["tour"]
        assert t[0] == 0 and sorted(t) == list(range(n))
        assert r["result"]["cost"] <= r["result"]["cost_pre_2opt"] + 1e-9
        assert r["result"]["cost"] == pytest.approx(routing.open_cost(routes[0]["C"], t))
    names = {r["name"] for r in eng.runs}
    assert {"sim-Tuna-5", "sim-Tuna-9", "ideal_k2_p2", "exact_k3", "warm_k2", "nn"} <= names


def test_resume_gives_identical_result(tmp_path):
    routes = make_routes(tmp_path, size=11)
    cfg = make_cfg()
    full = sim_engine(tmp_path, routes, cfg, sub="full"); full.run_all()
    part = sim_engine(tmp_path, routes, cfg, sub="part")
    run = [r for r in part.runs if r["name"] == "sim-Tuna-5"][0]
    for _ in range(2):                                   # interrupt after a few steps
        part.step(run)
    assert run["status"] == "running"
    resumed = sim_engine(tmp_path, routes, cfg, sub="part")      # fresh process: reads the saved state files
    resumed.run_all()
    a = [r for r in full.runs if r["name"] == "sim-Tuna-5"][0]["result"]["tour"]
    b = [r for r in resumed.runs if r["name"] == "sim-Tuna-5"][0]["result"]["tour"]
    assert a == b


def test_reversed_bit_order_is_calibrated_away(tmp_path):
    routes = make_routes(tmp_path)
    cfg = make_cfg()
    a = sim_engine(tmp_path, routes, cfg, sub="a"); a.run_all()
    b = sim_engine(tmp_path, routes, cfg, reverse=True, sub="b"); b.run_all()
    ra = [r for r in a.runs if r["name"] == "sim-Tuna-5"][0]
    rb = [r for r in b.runs if r["name"] == "sim-Tuna-5"][0]
    assert ra["bit_order"] == "qiskit" and rb["bit_order"] == "reversed"
    assert ra["result"]["tour"] == rb["result"]["tour"]


def test_inconclusive_calibration_stops_run(tmp_path):
    routes = make_routes(tmp_path)
    cfg = make_cfg()
    eng = sim_engine(tmp_path, routes, cfg, sub="c")
    gw = eng.gws["sim-Tuna-5"]
    orig = gw.submit

    def bad(jobs, shots):
        out = orig(jobs, shots)
        for j, o in zip(jobs, out):
            if j["role"] == "calibration":
                o["result"]["counts"] = {"0110": shots}
        return out
    gw.submit = bad
    run = [r for r in eng.runs if r["name"] == "sim-Tuna-5"][0]
    for _ in range(3):
        eng.step(run)
    assert run["status"] == "failed" and "calibration" in run["failure"]


def test_calibration_decision_with_readout_noise():
    # 16 qubits: the exact key is only 46% of the shots (as on Tuna-17) but the order is still clear
    n = 16
    counts = {"0" * 15 + "1": 470, "0" * 14 + "11": 60, "0" * 13 + "101": 55, "0" * 12 + "1001": 50, "1" + "0" * 14 + "1": 25,
              "0" * 16: 60, "0" * 15 + "1"[:0] + "0": 0, "0" * 8 + "1" + "0" * 6 + "1": 40, "0" * 10 + "1" + "0" * 4 + "1": 40,
              "0" * 6 + "1" + "0" * 8 + "1": 40, "0" * 4 + "1" + "0" * 10 + "1": 38, "0" * 2 + "1" + "0" * 12 + "1": 40,
              "0" * 15 + "0": 106}
    order, det = E.bit_order_from_calibration(counts)
    assert order == "qiskit" and det["p_exact_key"] < 0.5
    rev = {k[::-1]: v for k, v in counts.items()}
    assert E.bit_order_from_calibration(rev)[0] == "reversed"
    assert E.bit_order_from_calibration({"0" * 16: 100})[0] is None            # no excitation at all
    assert E.bit_order_from_calibration({"1" * 16: 100})[0] is None            # everything excited
    assert E.bit_order_from_calibration({"1" + "0" * 14 + "1": 100})[0] is None  # both ends excited


def test_exploration_is_seeded_and_state_based():
    C = np.random.default_rng(0).random((12, 12)); np.fill_diagonal(C, 0)
    st = heuristic.initial_state(12)
    a = heuristic.plan_step(st, C, 3, 3, 0.4, 2026, "r/x")
    b = heuristic.plan_step(st, C, 3, 3, 0.4, 2026, "r/x")
    c = heuristic.plan_step(st, C, 3, 3, 0.4, 7, "r/x")
    assert a == b and len(set(a["cands"])) == 3
    assert a["cands"][:2] == sorted(range(1, 12), key=lambda x: (C[0, x], x))[:2]
    assert c["cands"][:2] == a["cands"][:2]


def test_last_window_smaller_than_k():
    C = np.random.default_rng(0).random((5, 5)); np.fill_diagonal(C, 0)
    st = {"tour": [0, 1, 2], "unvisited": [3, 4], "iteration": 2}
    s = heuristic.plan_step(st, C, 4, 4, 0.0, 1, "x")
    assert s["k"] == 2 and len(s["cands"]) == 2


# ------------------------------------------------------------- Quantum Inspire path with fakes
def qi_engine(tmp_path, routes, cfg, monkeypatch, server, backends=("Tuna-5",)):
    fakes.install(monkeypatch, server)
    gws = {}
    for b in backends:
        g = QuantumInspireGateway(b, cfg["quantum_inspire"])
        info = g.connect(); g.k = window_size(cfg["qaoa"]["k"], info["qubits"]); gws[b] = g
    eng = E.Engine(cfg, routes, gws, tmp_path / "qi", log=lambda *_: None)
    eng.idle_sleep = 0
    eng.create_runs()
    return eng


def test_quantum_inspire_flow_with_fakes(tmp_path, monkeypatch):
    routes = make_routes(tmp_path)
    cfg = make_cfg()
    server = fakes.Server()
    eng = qi_engine(tmp_path, routes, cfg, monkeypatch, server, ("Tuna-5", "Tuna-9"))
    eng.run_all()
    n = routes[0]["n_nodes"]
    for lab in ("Tuna-5", "Tuna-9"):
        run = [r for r in eng.runs if r["name"] == lab][0]
        assert run["status"] == "finished" and sorted(run["result"]["tour"]) == list(range(n))
        assert run["bit_order"] == "qiskit"
        for w in run["windows"]:
            if w.get("trivial"):
                continue
            job = w["job"]
            assert isinstance(job["job_id"], int) and job["transpiled"]["depth"] > 0
            assert w["tvd_vs_ideal"] < 0.2          # the transpiled circuit samples the intended distribution


def test_failed_job_is_resubmitted(tmp_path, monkeypatch):
    routes = make_routes(tmp_path)
    cfg = make_cfg()
    server = fakes.Server(fail_first=1)
    eng = qi_engine(tmp_path, routes, cfg, monkeypatch, server)
    eng.run_all()
    run = [r for r in eng.runs if r["name"] == "Tuna-5"][0]
    assert run["status"] == "finished"
    assert any(w.get("job", {}).get("history") for w in run["windows"])


def test_job_failing_forever_marks_run_failed(tmp_path, monkeypatch):
    routes = make_routes(tmp_path)
    cfg = make_cfg(); cfg["quantum_inspire"]["max_job_retries"] = 1
    server = fakes.Server(fail_first=10 ** 6)
    eng = qi_engine(tmp_path, routes, cfg, monkeypatch, server)
    eng.run_all()
    run = [r for r in eng.runs if r["name"] == "Tuna-5"][0]
    assert run["status"] == "failed" and "failed" in run["failure"]


def test_waits_until_results_return(tmp_path, monkeypatch):
    """A job that stays 'running' for many polls must not make the run fail or time out."""
    routes = make_routes(tmp_path)
    cfg = make_cfg()
    server = fakes.Server()
    eng = qi_engine(tmp_path, routes, cfg, monkeypatch, server)
    run = [r for r in eng.runs if r["name"] == "Tuna-5"][0]
    server.never_ready = True
    for _ in range(40):
        eng.step(run)
    assert run["status"] == "running" and run["pending"] is not None
    server.never_ready = False
    eng.run_all()
    assert run["status"] == "finished"


def test_connectivity_guard_rejects_uncoupled_pair(monkeypatch):
    from qiskit import QuantumCircuit
    fakes.install(monkeypatch, fakes.Server())
    g = QuantumInspireGateway("Tuna-5", load_config(ROOT / "config.toml")["quantum_inspire"])
    g.connect()
    qc = QuantumCircuit(5); qc.cz(0, 1)               # 0-1 is not a coupler of the star
    with pytest.raises(RuntimeError, match="uncoupled"):
        g._check_connectivity(qc)
    qc = QuantumCircuit(5); qc.cz(2, 1)
    g._check_connectivity(qc)


@pytest.mark.parametrize("exc", [Exception("401 Unauthorized: token expired"),
                                 FileNotFoundError("No configuration file found. Please connect to Quantum Inspire using the CLI.")])
def test_auth_error_message(monkeypatch, exc):
    import qiskit_quantuminspire.qi_provider as qp

    class Boom:
        def __init__(self): raise exc
    monkeypatch.setattr(qp, "QIProvider", Boom)
    with pytest.raises(AuthError, match="qi login"):
        QuantumInspireGateway("Tuna-9", load_config(ROOT / "config.toml")["quantum_inspire"]).connect()


def test_auth_error_during_polling_stops_the_program(tmp_path, monkeypatch):
    routes = make_routes(tmp_path)
    cfg = make_cfg()
    server = fakes.Server()
    eng = qi_engine(tmp_path, routes, cfg, monkeypatch, server)
    import quantuminspire.managers.resource_manager as rm

    class Expired:
        def get_job(self, job_id): raise Exception("401 Unauthorized")
    run = [r for r in eng.runs if r["name"] == "Tuna-5"][0]
    eng.step(run)                                       # submits the first window
    monkeypatch.setattr(rm, "ResourceManager", Expired)
    with pytest.raises(AuthError):
        for _ in range(5):
            eng.step(run)
    assert run["pending"] is not None                    # state kept for the resume


def test_unreachable_service_is_a_clear_error(monkeypatch):
    import qiskit_quantuminspire.qi_provider as qp

    class Boom:
        def __init__(self): raise ConnectionError("Name or service not known")
    monkeypatch.setattr(qp, "QIProvider", Boom)
    with pytest.raises(RuntimeError, match="Could not reach Quantum Inspire"):
        QuantumInspireGateway("Tuna-9", load_config(ROOT / "config.toml")["quantum_inspire"]).connect()


def test_config_rejects_unknown_keys_and_backends(tmp_path):
    bad = tmp_path / "c.toml"
    bad.write_text("[qaoa]\nnonsense = 1\n")
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(bad)
    bad.write_text('[quantum_inspire]\nbackends = ["Tuna-7"]\n')
    with pytest.raises(ValueError, match="not allowed"):
        load_config(bad)
    bad.write_text('[qaoa]\nk = 5\n')
    cfg = load_config(bad)
    with pytest.raises(ValueError, match="needs 25 qubits"):
        window_size(cfg["qaoa"]["k"], 17)


# ------------------------------------------------------------- real-hardware guard, platform info, timing
def test_emulator_is_refused_on_the_hardware_path(monkeypatch):
    fakes.install(monkeypatch, fakes.Server(hardware=False))
    g = QuantumInspireGateway("Tuna-5", load_config(ROOT / "config.toml")["quantum_inspire"])
    with pytest.raises(RuntimeError, match="NOT confirmed real quantum hardware"):
        g.connect()


def test_hardware_is_recorded_in_backend_info(monkeypatch):
    fakes.install(monkeypatch, fakes.Server(hardware=True))
    g = QuantumInspireGateway("Tuna-5", load_config(ROOT / "config.toml")["quantum_inspire"])
    info = g.connect()
    assert info["is_hardware"] is True and info["backend_status"] == "idle"


def test_platform_info_and_timing_are_recorded(tmp_path, monkeypatch):
    routes = make_routes(tmp_path)
    cfg = make_cfg()
    eng = qi_engine(tmp_path, routes, cfg, monkeypatch, fakes.Server())
    eng.run_all()
    run = [r for r in eng.runs if r["name"] == "Tuna-5"][0]
    w = [x for x in run["windows"] if x.get("job")][0]
    plat = w["job"]["platform"]
    assert plat["batch_backend_is_hardware"] is True and plat["batch_backend_type_name"] == "Tuna-fake"
    assert plat["job_created_on"] and plat["batch_reserved_at"]
    assert w["job"]["backend_is_hardware"] is True and w["job"]["completed_seen_utc"]
    tm = run["timing"]
    assert tm["compute_s"] > 0 and tm["submit_s"] > 0 and tm["first_activity_utc"]
    for r in eng.runs:                                   # every run, also the classical ones, has a compute time
        assert r["timing"]["compute_s"] >= 0 and r["status"] == "finished"


def test_sim_backend_is_marked_not_hardware(tmp_path):
    routes = make_routes(tmp_path)
    eng = sim_engine(tmp_path, routes, make_cfg())
    assert eng.gws["sim-Tuna-5"].info["is_hardware"] is False


def test_verify_hardware_script(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import verify_hardware as V
    import quantuminspire.managers.resource_manager as rm
    routes = make_routes(tmp_path)
    cfg = make_cfg()
    server = fakes.Server()
    eng = qi_engine(tmp_path, routes, cfg, monkeypatch, server)
    eng.run_all()
    runs = V.collect(tmp_path / "qi")
    assert runs
    off = V.offline_report(runs)
    assert all(r["all_job_ids_are_integers"] and not r["any_simulator_id"] and r["sum_exec_time_s"] > 0 for r in off)
    rows, err = V.platform_report(runs, rm.ResourceManager)
    assert err is None and rows and all(r["is_hardware"] is True for r in rows)
    server.hardware = False
    rows, err = V.platform_report(runs, rm.ResourceManager)
    assert all(r["is_hardware"] is False for r in rows)
