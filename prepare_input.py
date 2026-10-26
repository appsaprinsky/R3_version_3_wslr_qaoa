#!/usr/bin/env python3
"""Step 1: select random Amazon routes (seeded) and save them to input/.  Usage: python prepare_input.py

Reads [prepare] from config.toml. Saves per route: cost matrix, Amazon planned sequence and cost, coordinates (plots only).
The sample has no size filter. It is nested: with the same seed and exclusion list, a larger num_routes only ADDS routes,
so you can start with a few routes and extend later; re-running this command adds the missing routes to input/.
Solving is not done here: the OR-Tools baseline is an approach of main.py (same framework, same timing).
"""
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from wslrqaoa import amazon
from wslrqaoa.config import load_config, ConfigError
from wslrqaoa.io_utils import write_json_atomic, read_json


def main() -> int:
    try:
        cfg = load_config("config.toml")["prepare"]
        raw_dir = amazon.find_raw_dir(cfg["data_dir"])
    except (ConfigError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        return 1
    out = Path(cfg["input_dir"])
    manifest_path = out / "manifest.json"
    have = sorted(out.glob("route_*.json")) if out.exists() else []
    old = None
    if have:
        if not manifest_path.exists():
            print(f"ERROR: '{out}/' has routes but no manifest.json. Delete the folder to prepare a new sample.")
            return 1
        old = read_json(manifest_path)
        if old["seed"] != cfg["seed"] or sorted(old.get("excluded", [])) != sorted(cfg["exclude_route_ids"]):
            print(f"ERROR: '{out}/' was prepared with a different seed or exclusion list. Delete the folder (and the "
                  "output) to prepare a new sample, or restore the old settings to extend it.")
            return 1
        if len(have) >= cfg["num_routes"]:
            print(f"'{out}/' already holds {len(have)} route(s) (num_routes = {cfg['num_routes']}). Nothing to do. "
                  "Increase num_routes to extend the sample.")
            return 0
    print(f"Reading Amazon data from {raw_dir} (large files, this can take a minute) ...")
    t0 = time.time()
    route_data, travel_times, actual = amazon.load_raw(raw_dir)
    print(f"  loaded {len(travel_times)} routes in {time.time() - t0:.0f} s")
    routes, skipped = amazon.sample_routes(route_data, travel_times, actual, cfg["num_routes"], cfg["seed"],
                                           exclude=cfg["exclude_route_ids"])
    del route_data, travel_times, actual
    ids = [r["route_id"] for r in routes]
    if old is not None and ids[:len(old["route_ids"])] != old["route_ids"]:
        print("ERROR: the existing routes are not a prefix of the sample for this seed (raw data changed?). Aborting.")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    new = 0
    for rec in routes:
        path = out / f"route_{rec['route_id']}.json"
        if path.exists():
            continue
        print(f"Route {rec['route_id']}: {rec['n_nodes']} nodes, Amazon planned cost = {rec['amazon_cost']:.1f}")
        if rec["repairs"]:
            print("  repairs:", "; ".join(rec["repairs"]))
        write_json_atomic(path, rec)
        new += 1
    manifest = {
        "schema_version": 2, "created_utc": (old or {}).get("created_utc") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg["seed"], "excluded": list(cfg["exclude_route_ids"]), "num_routes": cfg["num_routes"],
        "route_ids": ids, "n_nodes": {r["route_id"]: r["n_nodes"] for r in routes},
        "skipped": [{"route_id": r, "reason": w} for r, w in skipped],
        "raw_dir": str(raw_dir), "raw_sha256": {f: amazon.sha256_file(raw_dir / f) for f in amazon.RAW_FILES},
        "python": platform.python_version(),
    }
    write_json_atomic(manifest_path, manifest)
    print(f"\nSaved {new} new route(s) to {out}/ ({len(ids)} in total). Next: python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
