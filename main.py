#!/usr/bin/env python3
"""Step 2: run the experiment.  Usage: python main.py

Reads input/ (made by prepare_input.py) and config.toml. Submits the windows to the Quantum Inspire QPUs,
waits (polls) until the jobs return, advances the algorithm and writes the results to output/results/.
Everything is saved after every step: if you stop it (Ctrl+C, closed laptop), run `python main.py` again.
"""
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from wslrqaoa import report
from wslrqaoa.config import ConfigError, load_config, parse_backend, run_fingerprint, window_size
from wslrqaoa.engine import Engine
from wslrqaoa.gateways import AuthError, QuantumInspireGateway, SimGateway
from wslrqaoa.io_utils import input_fingerprint, load_routes, read_json, write_json_atomic


def versions():
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "qiskit", "qiskit-quantuminspire", "quantuminspire", "ortools", "pandas"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def main() -> int:
    try:
        cfg = load_config("config.toml")
        routes = load_routes(cfg["prepare"]["input_dir"])
    except (ConfigError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        return 1
    out = Path(cfg["output"]["output_dir"])
    exp_path = out / "experiment.json"
    fp, ifp = run_fingerprint(cfg), input_fingerprint(cfg["prepare"]["input_dir"])
    if exp_path.exists():
        exp = read_json(exp_path)
        if exp["run_fingerprint"] != fp or exp["input_fingerprint"] != ifp:
            print("ERROR: config.toml or input/ changed since this experiment was started.\n"
                  f"Restore the old settings to resume, or delete '{out}/' to start a new experiment.")
            return 1
    else:
        exp = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "run_fingerprint": fp, "input_fingerprint": ifp, "versions": versions()}

    try:                                                   # fail fast on an impossible window size
        for name in cfg["quantum_inspire"]["backends"]:
            window_size(cfg["qaoa"]["k"], parse_backend(name)[1])
    except ConfigError as e:
        print(f"ERROR: {e}")
        return 1

    # ---- connect to the back-ends (Quantum Inspire or local simulation)
    gws, infos = {}, {}
    try:
        for name in cfg["quantum_inspire"]["backends"]:
            base, qubits, is_sim = parse_backend(name)
            label = f"sim-{base}" if is_sim else base
            gw = SimGateway(base, qubits, cfg["qaoa"]["seed"]) if is_sim else QuantumInspireGateway(base, cfg["quantum_inspire"])
            print(f"Connecting to {'local simulator' if is_sim else 'Quantum Inspire'} backend {base} ...")
            info = gw.connect()
            if info["qubits"] != qubits:
                print(f"  WARNING: platform reports {info['qubits']} qubits for {base} (expected {qubits}); using {info['qubits']}.")
            gw.k = window_size(cfg["qaoa"]["k"], info["qubits"])
            gws[label], infos[label] = gw, info
            kind = (f"REAL QUANTUM HARDWARE (platform is_hardware = True, status: {info.get('backend_status')})"
                    if info.get("is_hardware") else "SIMULATION - NOT a quantum computer")
            print(f"  {info['name']}: {info['qubits']} qubits -> window size k = {gw.k} ({gw.k ** 2} qubits used)\n"
                  f"  -> {kind}")
            if info.get("is_hardware") and info.get("backend_status") in ("offline", "maintenance"):
                print(f"  WARNING: backend status is '{info['backend_status']}'; jobs will wait until it is available.")
    except AuthError as e:
        print(f"ERROR: {e}")
        return 1
    except (ConfigError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return 1
    write_json_atomic(out / "backends.json", infos)
    write_json_atomic(exp_path, exp)

    eng = Engine(cfg, routes, gws, out)
    eng.create_runs()
    for r in eng.runs:                                    # retry runs that stopped on a failed job
        if r["status"] == "failed":
            print(f"Retrying failed run {r['name']}: {r['failure']}")
            r["status"], r["failure"] = "running", None
            if r["pending"]:
                for j in r["pending"]["jobs"]:
                    if not j["done"]:
                        j["attempt"] = 1
            eng.save(r)
    n_unfinished = sum(r["status"] == "running" for r in eng.runs)
    print(f"\n{len(routes)} route(s), {len(eng.runs)} run(s), {n_unfinished} still to do. "
          f"Jobs are polled every {cfg['quantum_inspire']['poll_interval_s']} s; "
          "this program waits until they return (Ctrl+C is safe, `python main.py` resumes).\n")
    try:
        eng.run_all()
    except AuthError as e:
        print(f"\nERROR: {e}\nProgress is saved. After logging in, run: python main.py")
        return 1
    except KeyboardInterrupt:
        print("\nStopped. All progress and job IDs are saved. To resume, run: python main.py")
        return 0

    results_dir, warn = report.build_report(cfg, routes, eng.runs, out, infos)
    failed = [r for r in eng.runs if r["status"] == "failed"]
    if failed:
        print("\nATTENTION: some runs did not complete:")
        for r in failed:
            print(f"  {r['name']}: {r['failure']}")
        print(f"Partial results: {results_dir}/\nAfter fixing the cause, run: python main.py")
        return 2
    print(f"\nFINISHED\n\nAll runs are complete. Results: {results_dir}/")
    print("  summary.md  comparison.csv  windows.csv  jobs_ledger.csv  results.json  figures/")
    if warn:
        print(f"WARNING: {warn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
