"""config.toml loading and validation. Unknown keys are errors; every key is used."""
from __future__ import annotations
import hashlib
import json
import math
import tomllib
from pathlib import Path

HARDWARE_QUBITS = {"tuna-5": 5, "tuna-9": 9, "tuna-17": 17}

DEFAULTS = {
    "prepare": {"data_dir": "data/almrrc2021-data-training", "input_dir": "input", "num_routes": 1,
                "seed": 2026, "min_stops": 0, "max_stops": 0,
                "ortools_time_limit_s": 120, "ortools_workers": 8},
    "qaoa": {"k": "auto", "p": 2, "batch_size": 0, "exploration": 0.0, "shots": 1024, "seed": 2026,
             "penalty_lambda": 1.5, "epsilon": 0.25, "lp_cost": "surrogate", "decode_rule": "marginal",
             "ramp_mode": "optimize", "delta_gamma": 1.0, "delta_beta": 1.0,
             "ramp_grid": 7, "ramp_maxiter": 100},
    "quantum_inspire": {"backends": ["Tuna-5", "Tuna-9", "Tuna-17"], "poll_interval_s": 30,
                        "max_job_retries": 3, "optimization_level": 3, "transpile_seed": 7},
    "compare": {"ideal_simulation": True, "exact_window": True, "warm_start_only": True,
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
            raise ConfigError(f"Unknown key(s) in [{sec}]: {sorted(unknown)}. Allowed: {sorted(defaults)}")
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
    for key in ("figures", "visualise_subroute"):
        if not isinstance(cfg["output"][key], bool):
            raise ConfigError(f"output.{key} must be true or false (lowercase, TOML syntax)")
    if not isinstance(cfg["output"]["figure_max_routes"], int) or cfg["output"]["figure_max_routes"] < 1:
        raise ConfigError("output.figure_max_routes must be an integer >= 1")
    if cfg["prepare"]["num_routes"] < 1:
        raise ConfigError("prepare.num_routes must be >= 1")
    return cfg


def run_fingerprint(cfg: dict) -> str:
    """Hash of everything that defines the experiment (not of the poll interval/output options)."""
    q = dict(cfg["qaoa"])
    qi = {k: v for k, v in cfg["quantum_inspire"].items() if k != "poll_interval_s"}
    blob = json.dumps({"qaoa": q, "qi": qi, "compare": cfg["compare"]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
