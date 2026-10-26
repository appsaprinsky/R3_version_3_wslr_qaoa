"""Amazon Last Mile Routing Research Challenge (ALMRRC 2021) data -> prepared route records."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from . import routing

RAW_FILES = ("route_data.json", "travel_times.json", "actual_sequences.json")


def find_raw_dir(data_dir: str | Path) -> Path:
    d = Path(data_dir)
    for cand in (d, d / "model_build_inputs", d / "almrrc2021-data-training" / "model_build_inputs"):
        if all((cand / f).exists() for f in RAW_FILES):
            return cand
    raise FileNotFoundError(
        f"Could not find {', '.join(RAW_FILES)} under '{data_dir}'. "
        f"Set [prepare] data_dir in config.toml to the folder that contains them "
        f"(usually .../almrrc2021-data-training/model_build_inputs).")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(raw_dir: Path):
    out = []
    for name in RAW_FILES:
        with open(raw_dir / name) as f:
            out.append(json.load(f))
    return out  # route_data, travel_times, actual_sequences


def build_route(route_id, route_data, travel_times, actual_sequences):
    """Return (record, None) or (None, reason). Nodes are re-indexed so the depot is index 0."""
    rd = route_data[route_id]["stops"]
    depots = [s for s, v in rd.items() if v.get("type") == "Station"]
    if not depots:
        return None, "no Station stop"
    depot = depots[0]
    tt = travel_times[route_id]
    others = sorted(s for s in rd if s != depot)
    ids = [depot] + others
    n = len(ids)
    if any(s not in tt for s in ids):
        return None, "stop missing in travel_times"
    C = np.zeros((n, n))
    for a, sa in enumerate(ids):
        row = tt[sa]
        for b, sb in enumerate(ids):
            if a != b:
                if sb not in row:
                    return None, f"missing travel time {sa}->{sb}"
                C[a, b] = float(row[sb])
    seq = actual_sequences.get(route_id, {}).get("actual")
    if not seq:
        return None, "no planned sequence in actual_sequences.json"
    pos = {s: float(v) for s, v in seq.items() if s in ids}
    planned = [s for s, _ in sorted(pos.items(), key=lambda kv: kv[1])]
    repairs = []
    seen, dedup = set(), []
    for s in planned:
        if s in seen:
            repairs.append(f"duplicate {s} removed")
        else:
            seen.add(s); dedup.append(s)
    for s in ids:
        if s not in seen:
            repairs.append(f"missing {s} appended"); dedup.append(s)
    if dedup[0] != depot:
        repairs.append("sequence rotated so the depot leads")
        dedup.remove(depot); dedup.insert(0, depot)
    index = {s: i for i, s in enumerate(ids)}
    seq_idx = [index[s] for s in dedup]
    coords = [[float(rd[s]["lat"]), float(rd[s]["lng"])] for s in ids]
    rec = {
        "schema_version": 1, "route_id": route_id, "n_nodes": n, "stop_ids": ids, "depot_index": 0,
        "cost_matrix": C.tolist(), "amazon_sequence": seq_idx,
        "amazon_cost": routing.open_cost(C, seq_idx), "amazon_sequence_source": "actual_sequences.json",
        "repairs": repairs, "coordinates": coords,
    }
    return rec, None


def sample_routes(route_data, travel_times, actual_sequences, num_routes, seed, min_stops=0, max_stops=0,
                  log=print, exclude=()):
    """Uniform sample without replacement (seeded). Invalid routes are skipped and the next one in the
    seeded permutation is used, so the same (seed, exclusion list) always gives the same routes, and the sample for
    N routes is the first N of the sample for any larger N (nested). min_stops/max_stops exist for exploratory use
    only; the experiment configuration does not expose them. `exclude` removes route ids (e.g. the pilot route)."""
    ids = sorted((set(travel_times) & set(route_data)) - set(exclude))
    if min_stops or max_stops:
        ids = [r for r in ids if (not min_stops or len(route_data[r]["stops"]) >= min_stops)
               and (not max_stops or len(route_data[r]["stops"]) <= max_stops)]
    if not ids:
        raise ValueError("No route satisfies the stop-count filter.")
    order = np.random.default_rng(seed).permutation(len(ids))
    picked, skipped = [], []
    for o in order:
        rid = ids[int(o)]
        rec, why = build_route(rid, route_data, travel_times, actual_sequences)
        if rec is None:
            skipped.append((rid, why)); log(f"  skipped {rid}: {why}"); continue
        picked.append(rec)
        if len(picked) == num_routes:
            break
    if len(picked) < num_routes:
        raise ValueError(f"Only {len(picked)} valid routes found, {num_routes} requested.")
    return picked, skipped
