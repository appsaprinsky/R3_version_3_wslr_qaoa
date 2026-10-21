"""Atomic JSON writes and input loading."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def write_json_atomic(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_routes(input_dir) -> list[dict]:
    d = Path(input_dir)
    files = sorted(d.glob("route_*.json"))
    if not files:
        raise FileNotFoundError(f"No routes in '{d}/'. Run: python prepare_input.py")
    routes = []
    for f in files:
        r = read_json(f)
        r["C"] = np.array(r["cost_matrix"], dtype=float)
        routes.append(r)
    return routes


def input_fingerprint(input_dir) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(input_dir).glob("route_*.json")):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()[:16]
