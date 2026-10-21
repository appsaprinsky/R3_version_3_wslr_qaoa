"""Quantum back-ends behind one interface.

submit(jobs, shots) -> [{"job_id", "batch_job_id", "transpiled", "result"?}]   (returns immediately)
status(job_id)      -> 'planned' | 'running' | 'completed' | 'failed' | 'cancelled'
fetch(job_id)       -> {"counts": {bitstring: int}, "meta": {...}}  or None if not available yet

QuantumInspireGateway reuses the calls of the reference scripts:
  QIProvider() / backend.run(transpile(qc, backend, optimization_level=3, seed_transpiler=7), shots)
  ResourceManager().get_results(job_id)[0].results
and never blocks on job.result().
"""
from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
from .heuristic import stable_int


class AuthError(RuntimeError):
    pass


def _looks_like_auth(exc: Exception) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return any(w in s for w in ("401", "unauthor", "login", "token", "credential", "authenticat", "forbidden", "403",
                                "no configuration file", "connect to quantum inspire", "configuration file"))


AUTH_MSG = "Quantum Inspire is not logged in or the login expired. Run: qi login"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class QuantumInspireGateway:
    kind = "quantum_inspire"

    def __init__(self, name: str, qi_cfg: dict):
        self.name, self.cfg = name, qi_cfg
        self.backend = None
        self.info = {}

    # ---- connection
    def connect(self) -> dict:
        try:
            from qiskit_quantuminspire.qi_provider import QIProvider
            backends = QIProvider().backends()
        except ImportError as e:
            raise RuntimeError("Quantum Inspire SDK missing. Run: pip install -r requirements.txt") from e
        except Exception as e:
            if _looks_like_auth(e):
                raise AuthError(AUTH_MSG) from e
            raise RuntimeError(f"Could not reach Quantum Inspire: {type(e).__name__}: {e}") from e
        names = [b.name for b in backends]
        match = [b for b in backends if b.name.lower() == self.name.lower()] or \
                [b for b in backends if self.name.lower() in b.name.lower()]
        if not match:
            raise RuntimeError(f"Backend '{self.name}' not found on Quantum Inspire. Available: {names}")
        self.backend = match[0]
        # ---- hard guard: only REAL hardware is accepted on this path (emulators must be requested as "sim:...")
        try:
            bt = self.backend.get_backend_type()
        except Exception as e:                                       # noqa: BLE001
            raise RuntimeError(f"Could not confirm that '{self.name}' is real quantum hardware "
                               f"({type(e).__name__}: {e}). Refusing to run.") from e
        is_hw = getattr(bt, "is_hardware", None)
        if is_hw is not True:
            raise RuntimeError(f"Backend '{self.backend.name}' on Quantum Inspire is NOT confirmed real quantum hardware "
                               f"(is_hardware={is_hw}). Refusing to run: choose Tuna-5, Tuna-9 or Tuna-17.")
        status = getattr(bt, "status", None)
        cmap = getattr(self.backend, "coupling_map", None)
        edges = sorted({tuple(sorted(e)) for e in cmap.get_edges()}) if cmap is not None else []
        self.info = {"name": self.backend.name, "qubits": int(self.backend.num_qubits),
                     "coupling_edges": [list(e) for e in edges],
                     "operation_names": sorted(self.backend.operation_names),
                     "max_shots": int(getattr(self.backend, "max_shots", 0) or 0),
                     "is_hardware": True, "backend_type_id": getattr(bt, "id", None),
                     "infrastructure": getattr(bt, "infrastructure", None), "identifier": getattr(bt, "identifier", None),
                     "backend_status": str(getattr(status, "value", status)).lower(),
                     "retrieved_utc": _utc()}
        return self.info

    # ---- submission
    def _check_connectivity(self, qc) -> None:
        edges = {tuple(e) for e in self.info["coupling_edges"]}
        ops = set(self.info["operation_names"]) | {"measure", "barrier", "delay"}
        for inst in qc.data:
            name = inst.operation.name
            if name not in ops:
                raise RuntimeError(f"Compiled circuit uses unsupported operation '{name}' on {self.name}")
            if len(inst.qubits) == 2 and edges:
                a, b = (qc.find_bit(q).index for q in inst.qubits)
                if tuple(sorted((a, b))) not in edges:
                    raise RuntimeError(f"Compiled circuit uses uncoupled pair ({a},{b}) on {self.name}")

    def submit(self, jobs: list[dict], shots: int) -> list[dict]:
        from qiskit import transpile
        tq = []
        for j in jobs:
            qc = transpile(j["circuit"], self.backend, optimization_level=self.cfg["optimization_level"],
                           seed_transpiler=self.cfg["transpile_seed"])
            qc.name = j["circuit"].name
            self._check_connectivity(qc)
            tq.append(qc)
        shots = min(int(shots), self.info.get("max_shots") or int(shots))
        try:
            job = self.backend.run(tq, shots=shots)          # returns immediately; no job.result()
        except Exception as e:
            if _looks_like_auth(e):
                raise AuthError(AUTH_MSG) from e
            raise
        out = []
        for qc, data in zip(tq, job.circuits_run_data):
            try:
                layout = qc.layout.final_index_layout() if qc.layout is not None else None
            except Exception:
                layout = None
            out.append({"job_id": int(data.job_id), "batch_job_id": int(job.batch_job_id) if job.batch_job_id else None,
                        "shots": shots, "submitted_utc": _utc(),
                        "transpiled": {"depth": int(qc.depth()), "two_qubit_gates": int(qc.num_nonlocal_gates()),
                                       "ops": {k: int(v) for k, v in qc.count_ops().items()},
                                       "qubit_mapping": [int(x) for x in layout] if layout else None}})
        return out

    # ---- retrieval (as quantum_inspire_job_id.py)
    def status(self, job_id) -> str:
        from quantuminspire.managers.resource_manager import ResourceManager
        try:
            job = ResourceManager().get_job(int(job_id))
        except Exception as e:
            if _looks_like_auth(e):
                raise AuthError(AUTH_MSG) from e
            raise
        return str(getattr(job.status, "value", job.status)).lower()

    def fetch(self, job_id):
        from quantuminspire.managers.resource_manager import ResourceManager
        try:
            results = ResourceManager().get_results(int(job_id))
        except Exception as e:
            if _looks_like_auth(e):
                raise AuthError(AUTH_MSG) from e
            raise
        if isinstance(results, list):
            if not results:
                return None
            result = results[0]
        else:
            result = results
        raw = result.results if hasattr(result, "results") else result["results"]
        counts = {str(k).replace(" ", ""): int(v) for k, v in raw.items()}
        meta = {f: (str(getattr(result, f)) if f == "created_on" else getattr(result, f))
                for f in ("id", "job_id", "created_on", "execution_time_in_seconds", "shots_requested", "shots_done")
                if hasattr(result, f)}
        meta["platform"] = self._platform_info(job_id)
        return {"counts": counts, "meta": meta}

    def _platform_info(self, job_id) -> dict:
        """Platform-side evidence and timestamps of one job (best effort; never raises).
        job: created_on / queued_at / finished_at.  batch: backend type (+ is_hardware), queued / reserved / finished."""
        from quantuminspire.managers.resource_manager import ResourceManager
        info: dict = {}
        batch_id = None
        try:
            job = ResourceManager().get_job(int(job_id))
            batch_id = getattr(job, "batch_job_id", None)
            info.update({"job_created_on": str(getattr(job, "created_on", None)),
                         "job_queued_at": str(getattr(job, "queued_at", None)),
                         "job_finished_at": str(getattr(job, "finished_at", None)),
                         "batch_job_id": batch_id})
        except Exception as e:                                       # noqa: BLE001
            info["job_lookup_error"] = f"{type(e).__name__}: {e}"
        if batch_id is not None:
            try:
                from compute_api_client import BatchJobsApi
                page = ResourceManager._invoke(BatchJobsApi, "read_batch_jobs_batch_jobs_get", id=int(batch_id))
                b = page.items[0]
                bt = ResourceManager().get_backend_type(b.backend_type_id)
                info.update({"batch_backend_type_id": b.backend_type_id, "batch_backend_id": getattr(b, "backend_id", None),
                             "batch_backend_type_name": getattr(bt, "name", None),
                             "batch_backend_is_hardware": getattr(bt, "is_hardware", None),
                             "batch_queued_at": str(getattr(b, "queued_at", None)),
                             "batch_reserved_at": str(getattr(b, "reserved_at", None)),
                             "batch_finished_at": str(getattr(b, "finished_at", None))})
            except Exception as e:                                   # noqa: BLE001
                info["batch_lookup_error"] = f"{type(e).__name__}: {e}"
        return info


class SimGateway:
    """Local, noise-free sampler with the same interface (dry runs and tests). Results are returned at
    submission time, so nothing is held in memory between invocations."""
    kind = "simulator"

    def __init__(self, name: str, qubits: int, seed: int = 2026, reverse_bits: bool = False):
        self.name, self.qubits, self.seed, self.reverse_bits = name, qubits, seed, reverse_bits
        self.info = {"name": f"sim:{name}", "qubits": qubits, "coupling_edges": [], "operation_names": [],
                     "max_shots": 100000, "is_hardware": False, "backend_status": "simulation", "retrieved_utc": _utc()}
        self._n = 0

    def connect(self) -> dict:
        return self.info

    def submit(self, jobs, shots):
        out = []
        for j in jobs:
            probs = j["probs"]
            n = j["circuit"].num_qubits
            rng = np.random.default_rng([self.seed, stable_int(j["tag"])])
            c = rng.multinomial(shots, probs)
            counts = {}
            for s in np.nonzero(c)[0]:
                key = format(int(s), f"0{n}b")
                counts[key[::-1] if self.reverse_bits else key] = int(c[s])
            self._n += 1
            out.append({"job_id": f"sim-{stable_int(j['tag']) % 10**8}", "batch_job_id": None, "shots": shots,
                        "submitted_utc": _utc(), "transpiled": None,
                        "result": {"counts": counts, "meta": {"shots_requested": shots, "shots_done": shots,
                                                              "execution_time_in_seconds": 0.0}}})
        return out

    def status(self, job_id):
        return "completed"

    def fetch(self, job_id):
        return None
