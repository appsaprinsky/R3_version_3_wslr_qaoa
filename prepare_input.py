#!/usr/bin/env python3
"""Step 1: select random Amazon routes (seeded) and save them to input/.  Usage: python prepare_input.py

Reads [prepare] from config.toml. Saves per route: cost matrix, Amazon planned sequence and cost,
coordinates (plots only) and the OR-Tools CP-SAT solution. main.py reads only from input/.
"""
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
from wslrqaoa import amazon, ortools_solver, routing
from wslrqaoa.config import load_config, ConfigError
from wslrqaoa.io_utils import write_json_atomic


def main() -> int:
    try:
        cfg = load_config("config.toml")["prepare"]
        raw_dir = amazon.find_raw_dir(cfg["data_dir"])
    except (ConfigError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        return 1
    out = Path(cfg["input_dir"])
    if out.exists() and any(out.glob("route_*.json")):
        print(f"ERROR: '{out}/' already contains routes. Delete the folder (and output/) to prepare a new sample.")
        return 1
    print(f"Reading Amazon data from {raw_dir} (large files, this can take a minute) ...")
    t0 = time.time()
    route_data, travel_times, actual = amazon.load_raw(raw_dir)
    print(f"  loaded {len(travel_times)} routes in {time.time() - t0:.0f} s")
    routes, skipped = amazon.sample_routes(route_data, travel_times, actual, cfg["num_routes"], cfg["seed"],
                                           cfg["min_stops"], cfg["max_stops"])
    del route_data, travel_times, actual
    out.mkdir(parents=True, exist_ok=True)
    for rec in routes:
        C = np.array(rec["cost_matrix"])
        print(f"Route {rec['route_id']}: {rec['n_nodes']} nodes, Amazon planned cost = {rec['amazon_cost']:.1f}")
        if rec["repairs"]:
            print("  repairs:", "; ".join(rec["repairs"]))
        print(f"  OR-Tools CP-SAT (time limit {cfg['ortools_time_limit_s']} s) ...")
        nn2 = routing.two_opt(C, routing.nearest_neighbor_tour(C))          # start solution for the solver
        hint = min((rec["amazon_sequence"], nn2), key=lambda t: routing.open_cost(C, t))
        res = ortools_solver.solve_open_tsp(C, 0, cfg["ortools_time_limit_s"], cfg["ortools_workers"], hint_tour=hint)
        res["hint"] = "amazon" if hint is rec["amazon_sequence"] else "nearest_neighbor_2opt"
        rec["ortools"] = res
        if res["cost"] is not None:
            print(f"  OR-Tools cost = {res['cost']:.1f} ({res['status']}, lower bound {res['lower_bound']:.1f})")
        write_json_atomic(out / f"route_{rec['route_id']}.json", rec)
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg["seed"], "num_routes": cfg["num_routes"], "min_stops": cfg["min_stops"],
        "max_stops": cfg["max_stops"], "route_ids": [r["route_id"] for r in routes],
        "skipped": [{"route_id": r, "reason": w} for r, w in skipped],
        "raw_dir": str(raw_dir), "raw_sha256": {f: amazon.sha256_file(raw_dir / f) for f in amazon.RAW_FILES},
        "python": platform.python_version(),
    }
    write_json_atomic(out / "manifest.json", manifest)
    print(f"\nSaved {len(routes)} route(s) to {out}/. Next: python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
