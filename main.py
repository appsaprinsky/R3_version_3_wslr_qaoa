#!/usr/bin/env python3
"""Step 2: run the experiment.  Usage: python main.py

Runs the approaches listed in config.toml [experiment] one after another (never at the same time, so the timings are not
disturbed), each on the same routes from input/ and each with its own folder and state:

  <output_dir>/ORtoolsOutput/            the whole route solved by OR-Tools CP-SAT
  <output_dir>/AlgorithmWithORtoolCore/  my algorithm, every window solved by OR-Tools CP-SAT
  <output_dir>/AlgorithmWithQC/          my algorithm, every window sampled on the Tuna-17 quantum computer
  <output_dir>/Comparison/               paired tables, statistics, figures across the approaches

Everything is saved after every step: if you stop it (Ctrl+C, closed laptop), run `python main.py` again.
A finished approach is not recomputed; routes added to input/ later are added to every approach.
"""
import math
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent / "src"))
from wslrqaoa import experiment, report
from wslrqaoa.config import ConfigError, algorithm_window, approach_fingerprint, load_config, parse_backend, window_size
from wslrqaoa.engine import Engine
from wslrqaoa.gateways import AuthError, QuantumInspireGateway, SimGateway
from wslrqaoa.io_utils import load_routes, read_json, route_hashes, write_json_atomic


def versions():
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "qiskit", "qiskit-quantuminspire", "quantuminspire", "ortools", "pandas"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def settings_for(cfg, approach):
    """The configuration sections that define one approach (stored with its results for provenance)."""
    if approach == "ORtoolsOutput":
        return {"ortools_full": cfg["ortools_full"]}
    k, b = algorithm_window(cfg)
    if approach == "AlgorithmWithORtoolCore":
        return {"k": k, "b": b, "exploration": cfg["qaoa"]["exploration"], "seed": cfg["qaoa"]["seed"], "ortools_core": cfg["ortools_core"]}
    return {"qaoa": cfg["qaoa"], "quantum_inspire": cfg["quantum_inspire"], "compare": cfg["compare"], "ortools_core": cfg["ortools_core"]}


def check_experiment_file(cfg, approach, out, hashes):
    """Create or validate <out>/experiment.json. Settings must not change once an approach has started; routes may be
    added, but an existing route file must not change."""
    path = out / "experiment.json"
    fp = approach_fingerprint(cfg, approach)
    if path.exists():
        exp = read_json(path)
        if exp["fingerprint"] != fp:
            raise ConfigError(f"The settings of {approach} changed since it was started. Restore the old config.toml values, "
                              f"or delete '{out}/' to start this approach again.")
        changed = [rid for rid, h in hashes.items() if rid in exp["route_hashes"] and exp["route_hashes"][rid] != h]
        if changed:
            raise ConfigError(f"Route file(s) changed since {approach} was started: {changed[:3]}. Restore input/ or delete '{out}/'.")
        exp["route_hashes"].update(hashes)
    else:
        exp = {"schema_version": 2, "approach": approach, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "fingerprint": fp, "route_hashes": dict(hashes), "settings": settings_for(cfg, approach), "versions": versions()}
    write_json_atomic(path, exp)


def qc_labels(cfg):
    base, qubits, is_sim = parse_backend(cfg["quantum_inspire"]["backends"][0])
    return base, qubits, is_sim, (f"sim-{base}" if is_sim else base)


def qc_all_finished(cfg, routes, out) -> bool:
    """True if every AlgorithmWithQC run of every route is already finished (then no connection is needed)."""
    _, _, _, label = qc_labels(cfg)
    k, _ = algorithm_window(cfg)
    p, comp = cfg["qaoa"]["p"], cfg["compare"]
    names = [label] + ([f"ideal_k{k}_p{p}"] if comp["ideal_simulation"] else []) + ([f"exact_k{k}"] if comp["exact_window"] else []) \
        + ([f"warm_k{k}"] if comp["warm_start_only"] else []) + (["nn"] if comp["nearest_neighbor"] else [])
    for r in routes:
        for nm in names:
            f = out / "runs" / f"{r['route_id']}__{nm}.json"
            if not f.exists() or read_json(f).get("status") != "finished":
                return False
    return (out / "backends.json").exists()


def connect_qc(cfg, out):
    """Connect to Tuna-17 (or the local 'sim:' sampler). Returns ({label: gateway}, {label: info})."""
    base, qubits, is_sim, label = qc_labels(cfg)
    gw = SimGateway(base, qubits, cfg["qaoa"]["seed"]) if is_sim else QuantumInspireGateway(base, cfg["quantum_inspire"])
    print(f"Connecting to {'local simulator' if is_sim else 'Quantum Inspire'} backend {base} ...")
    info = gw.connect()
    if info["qubits"] != qubits:
        print(f"  WARNING: platform reports {info['qubits']} qubits for {base} (expected {qubits}); using {info['qubits']}.")
    gw.k = window_size(cfg["qaoa"]["k"], info["qubits"])
    kind = (f"REAL QUANTUM HARDWARE (platform is_hardware = True, status: {info.get('backend_status')})"
            if info.get("is_hardware") else "SIMULATION - NOT a quantum computer")
    print(f"  {info['name']}: {info['qubits']} qubits -> window size k = {gw.k} ({gw.k ** 2} qubits used)\n  -> {kind}")
    if info.get("is_hardware") and info.get("backend_status") in ("offline", "maintenance"):
        print(f"  WARNING: backend status is '{info['backend_status']}'; jobs will wait until it is available.")
    write_json_atomic(out / "backends.json", {label: info})
    return {label: gw}, {label: info}


def preflight(cfg, approach, routes, k, b):
    n = [r["n_nodes"] for r in routes]
    if approach == "ORtoolsOutput":
        f = cfg["ortools_full"]
        print(f"  {len(routes)} route(s) ({min(n)}-{max(n)} nodes); at most {len(routes) * f['time_limit_s'] / 60:.0f} min "
              f"({f['time_limit_s']:g} s per route, {f['workers']} workers, no start solution).")
    elif approach == "AlgorithmWithORtoolCore":
        w = sum(math.ceil((m - 1) / b) for m in n)
        print(f"  {len(routes)} route(s) ({min(n)}-{max(n)} nodes), k = {k}, b = {b}: {w} windows solved by CP-SAT.")
    else:
        w = sum(math.ceil((m - 1) / b) for m in n)
        print(f"  {len(routes)} route(s) ({min(n)}-{max(n)} nodes), k = {k}, b = {b}: about {w} quantum jobs in total, "
              f"{max(math.ceil((m - 1) / b) for m in n)} sequential jobs on the longest route.")


def run_approach(cfg, routes, hashes, approach, root):
    """Returns (state, engine) with state in finished | failed | stopped | error."""
    out = root / approach
    k, b = algorithm_window(cfg)
    print(f"\n=== {approach} ===")
    preflight(cfg, approach, routes, k, b)
    try:
        check_experiment_file(cfg, approach, out, hashes)
    except ConfigError as e:
        print(f"ERROR: {e}")
        return "error", None
    gws, infos = {}, {}
    if approach == "AlgorithmWithQC":
        try:
            if qc_all_finished(cfg, routes, out):
                base, qubits, is_sim, label = qc_labels(cfg)
                infos = read_json(out / "backends.json")
                gws = {label: SimpleNamespace(k=k, name=base, kind="offline", info=infos.get(label, {}))}
            else:
                gws, infos = connect_qc(cfg, out)
        except AuthError as e:
            print(f"ERROR: {e}")
            return "error", None
        except (ConfigError, RuntimeError) as e:
            print(f"ERROR: {e}")
            return "error", None
    eng = Engine(cfg, routes, gws, out, approach=approach)
    eng.create_runs()
    for r in eng.runs:                                    # retry runs that stopped on a failure
        if r["status"] == "failed":
            print(f"Retrying failed run {r['name']}: {r['failure']}")
            r["status"], r["failure"] = "running", None
            if r["pending"]:
                for j in r["pending"]["jobs"]:
                    if not j["done"]:
                        j["attempt"] = 1
            eng.save(r)
    todo = sum(r["status"] == "running" for r in eng.runs)
    print(f"  {len(eng.runs)} run(s), {todo} still to do." + (
        f" Jobs are polled every {cfg['quantum_inspire']['poll_interval_s']} s; this program waits until they return "
        "(Ctrl+C is safe, `python main.py` resumes)." if approach == "AlgorithmWithQC" and todo else ""))
    try:
        eng.run_all()
    except AuthError as e:
        print(f"\nERROR: {e}\nProgress is saved. After logging in, run: python main.py")
        return "error", eng
    except KeyboardInterrupt:
        print("\nStopped. All progress and job IDs are saved. To resume, run: python main.py")
        return "stopped", eng
    # ---- results of this approach, in its own folder
    if approach == "ORtoolsOutput":
        experiment.build_ortools_report(cfg, routes, eng.runs, out)
    elif approach == "AlgorithmWithORtoolCore":
        experiment.build_core_report(cfg, routes, eng.runs, out)
    else:
        baseline = experiment.ortools_by_route(experiment.load_runs(root / "ORtoolsOutput"))
        report.build_report(cfg, routes, eng.runs, out, infos, ortools=baseline)
    failed = [r for r in eng.runs if r["status"] == "failed"]
    for r in failed:
        print(f"ATTENTION {approach}: {r['name']} on {r['route_id'][:22]}: {r['failure']}")
    print(f"  results: {out / 'results'}/")
    return ("failed" if failed else "finished"), eng


def main() -> int:
    try:
        cfg = load_config("config.toml")
        routes = load_routes(cfg["prepare"]["input_dir"])
        algorithm_window(cfg)                              # fail fast on an impossible window size
    except (ConfigError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        return 1
    root = Path(cfg["output"]["output_dir"])
    hashes = route_hashes(cfg["prepare"]["input_dir"])
    print(f"{len(routes)} route(s) from {cfg['prepare']['input_dir']}/; approaches: {', '.join(cfg['experiment']['approaches'])}")
    any_failed = False
    for approach in cfg["experiment"]["approaches"]:
        state, _ = run_approach(cfg, routes, hashes, approach, root)
        if state in ("error", "stopped"):
            return 1 if state == "error" else 0
        any_failed |= state == "failed"
    comp = experiment.build_comparison(cfg, routes, root)
    if comp:
        print(f"\nComparison across approaches: {comp}/  (paired_comparison.csv, statistics.md, figures/)")
    if any_failed:
        print("\nATTENTION: some runs did not complete. After fixing the cause, run: python main.py")
        return 2
    print("\nFINISHED\n\nAll requested approaches are complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
