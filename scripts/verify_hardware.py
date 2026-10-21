#!/usr/bin/env python3
"""Check that the QPU runs in output/ really ran on real Quantum Inspire hardware.

Usage: python scripts/verify_hardware.py            (needs `qi login` for the platform check; works offline too)

Offline evidence (from the saved run files): backend name, integer job ids (simulator ids start with 'sim-'),
platform-reported execution time > 0, shots done, counts that differ from the ideal distribution.
Platform check (online): for every saved batch id the platform is asked which backend type executed it and whether
that backend type is real hardware.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wslrqaoa.config import load_config
from wslrqaoa.io_utils import read_json


def collect(output_dir: Path):
    out = []
    for f in sorted((output_dir / "runs").glob("*.json")):
        r = read_json(f)
        if r.get("mode") != "qpu":
            continue
        jobs = [w["job"] for w in r["windows"] if w.get("job")]
        out.append((r, jobs))
    return out


def offline_report(runs):
    rows = []
    for r, jobs in runs:
        ids = [j["job_id"] for j in jobs]
        rows.append({
            "run": r["name"], "route": r["route_id"][:24], "windows_on_device": len(jobs),
            "backend": sorted({j.get("backend") for j in jobs}),
            "all_job_ids_are_integers": all(isinstance(i, int) for i in ids),
            "any_simulator_id": any(str(i).startswith("sim-") for i in ids),
            "sum_exec_time_s": round(sum(j.get("execution_time_s") or 0 for j in jobs), 2),
            "shots_done_ok": all((j.get("shots_done") or 0) >= (j.get("shots") or 1) for j in jobs),
            "mean_tvd_vs_ideal": round(sum(w["tvd_vs_ideal"] for w in r["windows"] if "tvd_vs_ideal" in w) /
                                       max(1, sum("tvd_vs_ideal" in w for w in r["windows"])), 3),
        })
    return rows


def platform_report(runs, resource_manager_cls=None):
    """Ask the platform which backend type ran each batch. Returns (rows, error)."""
    try:
        if resource_manager_cls is None:
            from quantuminspire.managers.resource_manager import ResourceManager as resource_manager_cls
        from compute_api_client import BatchJobsApi
    except Exception as e:                                           # noqa: BLE001
        return [], f"Quantum Inspire SDK not available ({type(e).__name__}: {e})"
    rows, cache = [], {}
    batches = sorted({(r["name"], j["batch_job_id"]) for r, jobs in runs for j in jobs if j.get("batch_job_id")})
    try:
        rm = resource_manager_cls()
        for run_name, bid in batches:
            page = resource_manager_cls._invoke(BatchJobsApi, "read_batch_jobs_batch_jobs_get", id=int(bid))
            b = page.items[0]
            if b.backend_type_id not in cache:
                cache[b.backend_type_id] = rm.get_backend_type(b.backend_type_id)
            bt = cache[b.backend_type_id]
            rows.append({"run": run_name, "batch_job_id": bid, "backend_type": bt.name, "is_hardware": bt.is_hardware,
                         "queued_at": str(b.queued_at), "reserved_at": str(b.reserved_at), "finished_at": str(b.finished_at)})
    except Exception as e:                                           # noqa: BLE001
        return rows, f"platform query failed ({type(e).__name__}: {e}); run `qi login` and try again"
    return rows, None


def main() -> int:
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.toml")
    out = Path(__file__).resolve().parents[1] / cfg["output"]["output_dir"]
    runs = collect(out)
    if not runs:
        print(f"No QPU runs found in {out}/runs/.")
        return 1
    print("OFFLINE EVIDENCE (saved run files)")
    for row in offline_report(runs):
        print("  ", row)
    print("\nPLATFORM CHECK (Quantum Inspire)")
    rows, err = platform_report(runs)
    for row in rows:
        print("  ", row)
    if err:
        print("  ", err)
        return 2
    if rows and all(r["is_hardware"] is True for r in rows):
        print(f"\nVERDICT: all {len(rows)} batches ran on real quantum hardware ({sorted({r['backend_type'] for r in rows})}).")
        return 0
    print("\nVERDICT: NOT all batches are confirmed hardware - inspect the table above.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
