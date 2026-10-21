"""Fakes for offline tests: a Quantum-Inspire-like backend/provider/resource manager built on Aer."""
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import GenericBackendV2


class Server:
    def __init__(self, fail_first=0, never_ready=False, hardware=True):
        self.jobs, self.next_id, self.calls = {}, 1000, {}
        self.fail_first, self.failed_so_far, self.never_ready = fail_first, 0, False
        self.hardware = hardware                  # False -> backends report is_hardware=False (an emulator)


class FakeResult:
    def __init__(self, counts, shots):
        self.results = counts
        self.id, self.job_id = 1, 1
        self.created_on = "2026-10-18T10:00:00Z"
        self.execution_time_in_seconds = 0.5
        self.shots_requested = shots
        self.shots_done = shots


class FakeCircuitData:
    def __init__(self, job_id):
        self.job_id = job_id


class FakeJob:
    def __init__(self, ids, batch):
        self.circuits_run_data = [FakeCircuitData(i) for i in ids]
        self.batch_job_id = batch


class _BT:
    def __init__(self, hardware):
        self.id, self.name, self.is_hardware = 7, "Tuna-fake", hardware
        self.infrastructure, self.identifier = "fake", "tuna-fake"
        self.status = type("S", (), {"value": "idle"})()


class FakeQIBackend(GenericBackendV2):
    def __init__(self, name, nq, edges, server):
        both = [list(e) for e in edges] + [[b, a] for a, b in edges]
        super().__init__(nq, basis_gates=["rx", "ry", "rz", "cz"], coupling_map=both, noise_info=False, seed=1)
        self.name = name
        self.server = server
        self.max_shots = 8192

    def get_backend_type(self):
        return _BT(self.server.hardware)

    def run(self, circuits, shots=1024, **kw):
        circuits = [circuits] if not isinstance(circuits, list) else circuits
        ids = []
        for qc in circuits:
            counts = AerSimulator().run(qc, shots=shots, seed_simulator=11).result().get_counts()
            jid = self.server.next_id
            self.server.next_id += 1
            self.server.jobs[jid] = {"counts": dict(counts), "shots": shots, "polls": 0,
                                     "fail": self.server.failed_so_far < self.server.fail_first}
            if self.server.jobs[jid]["fail"]:
                self.server.failed_so_far += 1
            ids.append(jid)
        return FakeJob(ids, batch=jid + 5000)


STAR5 = [(2, 0), (2, 1), (2, 3), (2, 4)]
GRID9 = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8)]


def install(monkeypatch, server):
    import qiskit_quantuminspire.qi_provider as qp
    import quantuminspire.managers.resource_manager as rm

    class FakeProvider:
        def backends(self, *a, **k):
            return [FakeQIBackend("Tuna-5", 5, STAR5, server), FakeQIBackend("Tuna-9", 9, GRID9, server)]

    class FakeRM:
        def get_job(self, job_id):
            j = server.jobs[job_id]
            j["polls"] += 1
            st = "failed" if j["fail"] else ("completed" if j["polls"] >= 3 and not server.never_ready else
                                              ("running" if j["polls"] == 2 else "planned"))
            return type("J", (), {"status": type("S", (), {"value": st})(), "batch_job_id": job_id + 5000,
                                  "created_on": "2026-10-20 06:36:40+00:00", "queued_at": "2026-10-20 06:36:41+00:00",
                                  "finished_at": "2026-10-20 06:41:10+00:00"})()

        @staticmethod
        def _invoke(api_class, method_name, *a, **k):
            b = type("B", (), {"backend_type_id": 7, "backend_id": 3, "queued_at": "2026-10-20 06:36:41+00:00",
                               "reserved_at": "2026-10-20 06:40:30+00:00", "finished_at": "2026-10-20 06:41:10+00:00"})()
            return type("P", (), {"items": [b]})()

        def get_backend_type(self, backend_type_id):
            return _BT(server.hardware)

        def get_results(self, job_id):
            j = server.jobs[job_id]
            if j["polls"] < 3 or j["fail"]:
                return []
            return [FakeResult(j["counts"], j["shots"])]

    monkeypatch.setattr(qp, "QIProvider", FakeProvider)
    monkeypatch.setattr(rm, "ResourceManager", FakeRM)
