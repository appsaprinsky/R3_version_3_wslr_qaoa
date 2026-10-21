"""Create a small synthetic ALMRRC-like dataset (same JSON structure) for offline tests/dry runs."""
import json
import sys
from pathlib import Path
import numpy as np


def make_fixture(root, n_routes=6, sizes=(9, 14, 20), seed=0):
    root = Path(root) / "almrrc2021-data-training" / "model_build_inputs"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    route_data, travel, actual = {}, {}, {}
    for r in range(n_routes):
        n = int(sizes[r % len(sizes)])
        rid = f"RouteID_{r:04d}_synthetic"
        pts = rng.random((n, 2)) * 0.05 + np.array([47.6, -122.3])      # lat/lng-like cluster
        ids = [f"S{r}{i:03d}" for i in range(n)]
        stops = {ids[i]: {"lat": float(pts[i, 0]), "lng": float(pts[i, 1]),
                          "type": "Station" if i == 0 else "Dropoff", "zone_id": f"Z{i % 4}"} for i in range(n)}
        route_data[rid] = {"station_code": "DXX1", "route_score": "High", "stops": stops}
        d = np.linalg.norm(pts[:, None] - pts[None], axis=2) * 20000 * (1 + 0.15 * rng.random((n, n)))
        np.fill_diagonal(d, 0)
        travel[rid] = {ids[i]: {ids[j]: float(d[i, j]) for j in range(n)} for i in range(n)}
        # "Amazon" sequence: noisy angular sweep from the depot
        ang = np.arctan2(pts[1:, 0] - pts[0, 0], pts[1:, 1] - pts[0, 1])
        order = [0] + list(1 + np.argsort(ang + 0.3 * rng.standard_normal(n - 1)))
        actual[rid] = {"actual": {ids[o]: p for p, o in enumerate(order)}}
    for name, obj in (("route_data.json", route_data), ("travel_times.json", travel), ("actual_sequences.json", actual)):
        with open(root / name, "w") as f:
            json.dump(obj, f)
    return root


if __name__ == "__main__":
    print(make_fixture(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fixture"))
