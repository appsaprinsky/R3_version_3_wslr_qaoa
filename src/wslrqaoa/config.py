"""config.toml loading and validation. Unknown keys are errors; every key is used."""
from __future__ import annotations
import hashlib
import json
import math
import tomllib
from pathlib import Path

HARDWARE_QUBITS = {"tuna-5": 5, "tuna-9": 9, "tuna-17": 17}

APPROACHES = ("ORtoolsOutput", "AlgorithmWithORtoolCore", "AlgorithmWithQC")

DEFAULTS = {
    # No route-size filters exist on purpose: the sample must not favour any approach.
    "prepare": {"data_dir": "data/almrrc2021-data-training", "input_dir": "input", "num_routes": 3,
                "seed": 2027, "exclude_route_ids": []},
    "experiment": {"approaches": list(APPROACHES)},
    # Approach 1: the whole route solved by CP-SAT. No start solution (hint) is ever given.
    "ortools_full": {"time_limit_s": 120.0, "workers": 8, "seed": 2026, "relative_gap_limit": 0.0, "cost_scale": 100},
    # Approach 2: the window solver of my algorithm is CP-SAT (exact on k <= 4 nodes). verify_qc_windows: also solve
    # every window visited by the QC run with CP-SAT (diagnostic only, excluded from the QC timing).
    "ortools_core": {"window_time_limit_s": 10.0, "cost_scale": 1000, "workers": 1, "seed": 2026,
                     "verify_qc_windows": True},
    "qaoa": {"k": "auto", "p": 2, "batch_size": 0, "exploration": 0.0, "shots": 1024, "seed": 2026,
             "penalty_lambda": 1.5, "epsilon": 0.25, "lp_cost": "surrogate", "decode_rule": "marginal",
             "ramp_mode": "optimize", "delta_gamma": 1.0, "delta_beta": 1.0,
             "ramp_grid": 7, "ramp_maxiter": 100},
    "quantum_inspire": {"backends": ["Tuna-17"], "poll_interval_s": 30,
                        "max_job_retries": 3, "optimization_level": 3, "transpile_seed": 7},
    # Optional controls, computed inside the AlgorithmWithQC folder. nearest_neighbor = nearest neighbour + 2-opt.
    "compare": {"ideal_simulation": False, "exact_window": False, "warm_start_only": False,
                "nearest_neighbor": True},
    "output": {"output_dir": "output", "figures": True, "visualise_subroute": True, "figure_max_routes": 5},
}


class ConfigError(ValueError):
    pass


def parse_backend(name: str):
    """'Tuna-9' -> ('Tuna-9', 9, False); 'sim:Tuna-9' -> ('Tuna-9', 9, True)."""
    sim = name.lower().startswith("sim:")
    base = name[4:] if sim else name
    q = HARDWARE_QUBITS.get(base.lower())
    if q is None:
        raise ConfigError(f"Backend '{name}' is not allowed. Use Tuna-5, Tuna-9 or Tuna-17 "
                          f"(optionally prefixed 'sim:' for a local simulation).")
    return base, q, sim


def window_size(cfg_k, qubits: int) -> int:
    k_max = math.isqrt(qubits)
    if cfg_k == "auto":
        return k_max
    if not isinstance(cfg_k, int) or cfg_k < 1:
        raise ConfigError("qaoa.k must be 'auto' or a positive integer")
    if cfg_k * cfg_k > qubits:
        raise ConfigError(f"qaoa.k = {cfg_k} needs {cfg_k**2} qubits but the backend has {qubits}. "
                          f"Set qaoa.k = 'auto' or a smaller value.")
    return cfg_k


def load_config(path: str | Path = "config.toml") -> dict:
    path = Path(path)
    user = {}
    if path.exists():
        with open(path, "rb") as f:
            user = tomllib.load(f)
    else:
        raise ConfigError(f"{path} not found. Run this command from the project folder.")
    cfg = {}
    for sec, defaults in DEFAULTS.items():
        got = user.get(sec, {})
        unknown = set(got) - set(defaults)
        if unknown:
            hint = ""
            if sec == "prepare" and unknown & {"min_stops", "max_stops"}:
                hint = " (min_stops/max_stops were removed on purpose: a route-size filter would bias the comparison)"
            if sec == "prepare" and unknown & {"ortools_time_limit_s", "ortools_workers"}:
                hint += " (the OR-Tools baseline now runs inside main.py; see [ortools_full])"
            raise ConfigError(f"Unknown key(s) in [{sec}]: {sorted(unknown)}. Allowed: {sorted(defaults)}{hint}")
        cfg[sec] = {**defaults, **got}
    extra = set(user) - set(DEFAULTS)
    if extra:
        raise ConfigError(f"Unknown section(s): {sorted(extra)}")
    q = cfg["qaoa"]
    if q["decode_rule"] not in ("marginal", "best_valid"):
        raise ConfigError("qaoa.decode_rule must be 'marginal' or 'best_valid'")
    if q["lp_cost"] not in ("surrogate", "linearized"):
        raise ConfigError("qaoa.lp_cost must be 'surrogate' or 'linearized'")
    if q["ramp_mode"] not in ("optimize", "fixed"):
        raise ConfigError("qaoa.ramp_mode must be 'optimize' or 'fixed'")
    if not (0.0 < q["epsilon"] <= 0.5):
        raise ConfigError("qaoa.epsilon must be in (0, 0.5]")
    if q["p"] < 1 or q["shots"] < 1 or q["penalty_lambda"] <= 1.0:
        raise ConfigError("qaoa.p >= 1, qaoa.shots >= 1 and qaoa.penalty_lambda > 1 are required")
    if not (0.0 <= q["exploration"] < 1.0):
        raise ConfigError("qaoa.exploration must be in [0, 1)")
    for b in cfg["quantum_inspire"]["backends"]:
        parse_backend(b)
    if not cfg["quantum_inspire"]["backends"]:
        raise ConfigError("quantum_inspire.backends must not be empty")
    ap = cfg["experiment"]["approaches"]
    if not ap or any(a not in APPROACHES for a in ap) or len(set(ap)) != len(ap):
        raise ConfigError(f"experiment.approaches must be a non-empty list without repeats from {list(APPROACHES)}")
    if len(cfg["quantum_inspire"]["backends"]) != 1:
        raise ConfigError("quantum_inspire.backends must contain exactly one backend (the experiment uses Tuna-17 only)")
    if not isinstance(cfg["prepare"]["exclude_route_ids"], list):
        raise ConfigError("prepare.exclude_route_ids must be a list of route ids")
    of, oc = cfg["ortools_full"], cfg["ortools_core"]
    if of["time_limit_s"] <= 0 or of["workers"] < 1 or of["cost_scale"] < 1 or of["relative_gap_limit"] < 0:
        raise ConfigError("ortools_full: time_limit_s > 0, workers >= 1, cost_scale >= 1, relative_gap_limit >= 0 required")
    if oc["window_time_limit_s"] <= 0 or oc["workers"] < 1 or oc["cost_scale"] < 1:
        raise ConfigError("ortools_core: window_time_limit_s > 0, workers >= 1, cost_scale >= 1 required")
    for key in ("figures", "visualise_subroute"):
        if not isinstance(cfg["output"][key], bool):
            raise ConfigError(f"output.{key} must be true or false (lowercase, TOML syntax)")
    if not isinstance(oc["verify_qc_windows"], bool):
        raise ConfigError("ortools_core.verify_qc_windows must be true or false")
    if not isinstance(cfg["output"]["figure_max_routes"], int) or cfg["output"]["figure_max_routes"] < 1:
        raise ConfigError("output.figure_max_routes must be an integer >= 1")
    if cfg["prepare"]["num_routes"] < 1:
        raise ConfigError("prepare.num_routes must be >= 1")
    return cfg


def algorithm_window(cfg: dict):
    """(k, b) shared by AlgorithmWithORtoolCore and AlgorithmWithQC: window size from the Tuna-17 qubit count
    (k = floor(sqrt(qubits)) when qaoa.k = "auto") and commit depth b (0 = k)."""
    qubits = parse_backend(cfg["quantum_inspire"]["backends"][0])[1]
    k = window_size(cfg["qaoa"]["k"], qubits)
    b = min(cfg["qaoa"]["batch_size"] or k, k)
    return k, b


def approach_fingerprint(cfg: dict, approach: str) -> str:
    """Hash of every setting that defines one approach (not of the poll interval, figure options or the route set)."""
    k, b = algorithm_window(cfg)
    if approach == "ORtoolsOutput":
        blob = {"ortools_full": cfg["ortools_full"]}
    elif approach == "AlgorithmWithORtoolCore":
        q = cfg["qaoa"]
        blob = {"k": k, "b": b, "exploration": q["exploration"], "seed": q["seed"], "ortools_core": cfg["ortools_core"]}
    else:
        qi = {key: v for key, v in cfg["quantum_inspire"].items() if key != "poll_interval_s"}
        blob = {"qaoa": cfg["qaoa"], "qi": qi, "compare": cfg["compare"], "ortools_core": cfg["ortools_core"]}
    return hashlib.sha256(json.dumps({"approach": approach, **blob}, sort_keys=True).encode()).hexdigest()[:16]
