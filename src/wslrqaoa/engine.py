"""Experiment engine: runs = (route x algorithm variant). State lives in output/runs/*.json, so the
program can be stopped at any time and `python main.py` resumes where it left off.

Run modes
  qpu    LRWS-QAOA windows sampled on a Quantum Inspire backend (or local 'sim:' sampler)
  ideal  same algorithm with exact circuit probabilities (no device)
  exact  every window solved exactly (brute force)               -- classical-window ablation
  warm   decode the warm start only (no QAOA layers)             -- what does the QAOA add?
  nn     nearest neighbour (window size 1) + 2-opt
"""
from __future__ import annotations
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from . import heuristic, routing, window as W, circuit as CIR
from .gateways import AuthError
from .io_utils import write_json_atomic, read_json


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _Timed:
    """Transparent proxy that adds the time spent in gateway calls to a shared clock:
    submit (transpile + upload) and poll (status + fetch). Everything else is forwarded unchanged."""

    def __init__(self, gw, clock: dict):
        self._gw, self._clock = gw, clock

    def __getattr__(self, name):
        attr = getattr(self._gw, name)
        bucket = {"submit": "submit", "status": "poll", "fetch": "poll"}.get(name)
        if bucket is None or not callable(attr):
            return attr

        def wrapped(*a, **k):
            t = time.perf_counter()
            try:
                return attr(*a, **k)
            finally:
                self._clock[bucket] += time.perf_counter() - t
        return wrapped


class Engine:
    def __init__(self, cfg: dict, routes: list[dict], gateways: dict, out_dir: Path, log=print):
        self.cfg, self.q, self.qi = cfg, cfg["qaoa"], cfg["quantum_inspire"]
        self.routes = {r["route_id"]: r for r in routes}
        self._clock = {"submit": 0.0, "poll": 0.0}          # seconds spent inside gateway calls
        self.gws = {k: _Timed(g, self._clock) for k, g in gateways.items()}    # label -> gateway
        self.out = Path(out_dir)
        self.log = lambda m: log(f"[{_now()}] {m}")
        self.runs: list[dict] = []
        self._next_poll: dict[str, float] = {}
        self._last_heartbeat = 0.0
        self.idle_sleep = 2.0

    # ------------------------------------------------------------------ run set
    def run_path(self, run) -> Path:
        return self.out / "runs" / f"{run['route_id']}__{run['name']}.json"

    def save(self, run) -> None:
        write_json_atomic(self.run_path(run), run)

    def _new_run(self, route_id, name, mode, k, p, backend=None, label=None) -> dict:
        n = self.routes[route_id]["n_nodes"]
        b = 1 if mode == "nn" else (self.q["batch_size"] or k)
        b = min(b, k)
        return {"schema_version": 1, "route_id": route_id, "name": name, "key": f"{route_id}/{name}", "mode": mode,
                "backend": backend, "label": label, "k": k, "p": p, "b": b, "status": "running",
                "state": heuristic.initial_state(n), "pending": None, "bit_order": None,
                "windows": [], "result": None, "failure": None,
                "timing": {"compute_s": 0.0, "submit_s": 0.0, "poll_s": 0.0, "first_activity_utc": None},
                "windows_total": int(np.ceil((n - 1) / b)) if n > 1 else 0}

    def create_runs(self) -> None:
        comp = self.cfg["compare"]
        names = set()
        for rid in self.routes:
            specs = []
            for label, gw in self.gws.items():
                k = gw.k
                specs.append((label, "qpu", k, self.q["p"], gw.name, label))
                if comp["ideal_simulation"]:
                    specs.append((f"ideal_k{k}_p{self.q['p']}", "ideal", k, self.q["p"], None, None))
                if comp["exact_window"]:
                    specs.append((f"exact_k{k}", "exact", k, 0, None, None))
                if comp["warm_start_only"]:
                    specs.append((f"warm_k{k}", "warm", k, 0, None, None))
            if comp["nearest_neighbor"]:
                specs.append(("nn", "nn", 1, 0, None, None))
            for name, mode, k, p, backend, label in specs:
                if (rid, name) in names:
                    continue
                names.add((rid, name))
                path = self.out / "runs" / f"{rid}__{name}.json"
                self.runs.append(read_json(path) if path.exists() else self._new_run(rid, name, mode, k, p, backend, label))
        for r in self.runs:
            self.save(r)

    # ------------------------------------------------------------------ main loop
    def run_all(self) -> None:
        while True:
            active = [r for r in self.runs if r["status"] == "running"]
            if not active:
                return
            progressed = False
            for r in active:
                progressed |= self.step(r)
            if not progressed:
                self._heartbeat(active)
                time.sleep(self.idle_sleep)

    def _heartbeat(self, active) -> None:
        if time.time() - self._last_heartbeat < 300:
            return
        self._last_heartbeat = time.time()
        waiting = []
        for r in active:
            p = r["pending"]
            if p:
                st = ",".join(sorted({str(j.get("last_status", "queued")) for j in p["jobs"] if not j["done"]}))
                waiting.append(f"{r['label']} window {p['iteration'] + 1}/{r['windows_total']} ({st})")
        if waiting:
            self.log("waiting for Quantum Inspire: " + "; ".join(waiting) + "  (Ctrl+C is safe; rerun python main.py to resume)")

    # ------------------------------------------------------------------ one step of one run
    def step(self, run) -> bool:
        """One step of one run, with timing. compute_s = time in this program's own computation (candidate selection,
        LP, angle optimisation on the exact simulator, circuit building, decoding, 2-opt) -- it excludes time inside
        gateway calls (submit_s: transpile + upload, poll_s: status/fetch) and, of course, all waiting."""
        if run["status"] != "running":
            return False
        tm = run.setdefault("timing", {"compute_s": 0.0, "submit_s": 0.0, "poll_s": 0.0, "first_activity_utc": None})
        tm.setdefault("first_activity_utc", None)
        if tm["first_activity_utc"] is None:
            tm["first_activity_utc"] = _utc_now()
        t0, c0 = time.perf_counter(), dict(self._clock)
        progressed = self._step(run)
        d_sub, d_poll = self._clock["submit"] - c0["submit"], self._clock["poll"] - c0["poll"]
        tm["submit_s"] += d_sub
        tm["poll_s"] += d_poll
        tm["compute_s"] += max(time.perf_counter() - t0 - d_sub - d_poll, 0.0)
        tm["last_activity_utc"] = _utc_now()
        if progressed:
            self.save(run)
        return progressed

    def _step(self, run) -> bool:
        if run["status"] != "running":
            return False
        route = self.routes[run["route_id"]]
        C = route["C"]
        if run["pending"] is not None:
            return self._poll(run, route)
        spec = heuristic.plan_step(run["state"], C, run["k"], run["b"], self.q["exploration"], self.q["seed"], run["key"])
        if spec is None:
            return self._finish(run, C)
        if spec["k"] == 1:
            self._commit(run, spec, [0], {"trivial": True})
            return True
        M = heuristic.window_matrix(C, spec)
        mode = run["mode"]
        if mode == "exact":
            order = routing.exact_window_order(M)
            self._commit(run, spec, order, {"mode": "exact"})
        elif mode == "nn":
            self._commit(run, spec, [0], {"trivial": True})
        elif mode == "warm":
            win = W.make_window(M, self.q["penalty_lambda"], self.q["epsilon"], self.q["lp_cost"])
            self._commit(run, spec, W.decode_order(win.x_clip), {"mode": "warm"})
        elif mode == "ideal":
            win = W.make_window(M, self.q["penalty_lambda"], self.q["epsilon"], self.q["lp_cost"])
            ang = W.choose_angles(win, run["p"], self.q)
            probs = W.ideal_probabilities(win, ang["gammas"], ang["betas"])
            d = W.interpret(win, probs, None, self.q["decode_rule"], 1.0 / self.q["shots"])
            self._commit(run, spec, d["order"], {"mode": "ideal", "angles": ang, **_slim(d)})
        else:
            return self._submit(run, spec, M)
        return True

    # ------------------------------------------------------------------ qpu path
    def _submit(self, run, spec, M) -> bool:
        gw = self.gws[run["label"]]
        win = W.make_window(M, self.q["penalty_lambda"], self.q["epsilon"], self.q["lp_cost"])
        ang = W.choose_angles(win, run["p"], self.q)
        it = run["state"]["iteration"]
        tag = f"{run['key']}/{it:03d}"
        qc = CIR.build_circuit(win, ang["gammas"], ang["betas"], name=f"wslr-{run['label']}-w{it:03d}")
        jobs = [{"role": "window", "circuit": qc, "tag": tag + "/window",
                 "probs": W.ideal_probabilities(win, ang["gammas"], ang["betas"]) if gw.kind == "simulator" else None}]
        if run["bit_order"] is None:
            probs = np.zeros(2 ** win.n); probs[1] = 1.0
            jobs.append({"role": "calibration", "circuit": CIR.calibration_circuit(win.n, f"wslr-cal-{run['label']}"),
                         "tag": tag + "/calibration", "probs": probs if gw.kind == "simulator" else None})
        subs = gw.submit(jobs, self.q["shots"])
        pending = {"iteration": it, "spec": spec, "x_lp": win.x_lp.tolist(), "angles": ang, "jobs": []}
        for j, s in zip(jobs, subs):
            res = s.get("result")
            pending["jobs"].append({"role": j["role"], "job_id": s["job_id"], "batch_job_id": s.get("batch_job_id"),
                                    "attempt": 1, "history": [], "submitted_utc": s["submitted_utc"], "shots": s["shots"],
                                    "transpiled": s.get("transpiled"), "result": res, "done": res is not None})
        run["pending"] = pending
        self.save(run)
        ids = ", ".join(str(j["job_id"]) for j in pending["jobs"])
        self.log(f"{run['label']}: route {run['route_id'][:18]} window {it + 1}/{run['windows_total']} "
                 f"({win.n} qubits, p={run['p']}) submitted; job id(s): {ids}")
        if all(j["done"] for j in pending["jobs"]):
            return self._complete(run, self.routes[run["route_id"]])
        self._next_poll[run["key"]] = time.time() + min(self.qi["poll_interval_s"], 10)
        return True

    def _poll(self, run, route) -> bool:
        if time.time() < self._next_poll.get(run["key"], 0):
            return False
        gw = self.gws[run["label"]]
        pend = run["pending"]
        changed = False
        for job in pend["jobs"]:
            if job["done"]:
                continue
            try:
                st = gw.status(job["job_id"])
                job["last_status"] = st
                if st == "completed":
                    res = gw.fetch(job["job_id"])
                    if res:
                        job["result"], job["done"], changed = res, True, True
                        job["completed_seen_utc"] = _utc_now()
                elif st in ("failed", "cancelled"):
                    self._resubmit(run, job, st)
                    changed = True
                    if run["status"] == "failed":
                        return True
            except AuthError:
                raise
            except Exception as e:                                  # transient network/API problem: keep waiting
                self.log(f"{run['label']}: could not query job {job['job_id']} ({type(e).__name__}: {e}); will retry")
        self._next_poll[run["key"]] = time.time() + self.qi["poll_interval_s"]
        if all(j["done"] for j in pend["jobs"]):
            return self._complete(run, route)
        if changed:
            self.save(run)
        return changed

    def _resubmit(self, run, job, why) -> None:
        pend = run["pending"]
        if job["attempt"] > self.qi["max_job_retries"]:
            run["status"], run["failure"] = "failed", f"job {job['job_id']} {why} after {job['attempt']} attempts"
            self.log(f"ATTENTION {run['label']}: {run['failure']}")
            self.save(run)
            return
        route = self.routes[run["route_id"]]
        M = heuristic.window_matrix(route["C"], pend["spec"])
        win = W.make_window(M, self.q["penalty_lambda"], self.q["epsilon"], self.q["lp_cost"], x_lp=np.array(pend["x_lp"]))
        a = pend["angles"]
        qc = (CIR.build_circuit(win, a["gammas"], a["betas"], name=f"wslr-{run['label']}-w{pend['iteration']:03d}")
              if job["role"] == "window" else CIR.calibration_circuit(win.n, f"wslr-cal-{run['label']}"))
        sub = self.gws[run["label"]].submit([{"role": job["role"], "circuit": qc, "probs": None,
                                              "tag": f"{run['key']}/{pend['iteration']:03d}/{job['role']}/r{job['attempt']}"}],
                                            self.q["shots"])[0]
        job["history"].append({"job_id": job["job_id"], "status": why})
        self.log(f"{run['label']}: job {job['job_id']} {why}; resubmitted as {sub['job_id']} (attempt {job['attempt'] + 1})")
        job.update({"job_id": sub["job_id"], "batch_job_id": sub.get("batch_job_id"), "attempt": job["attempt"] + 1,
                    "submitted_utc": sub["submitted_utc"], "transpiled": sub.get("transpiled"), "result": None, "done": False})

    def _complete(self, run, route) -> bool:
        pend = run["pending"]
        M = heuristic.window_matrix(route["C"], pend["spec"])
        win = W.make_window(M, self.q["penalty_lambda"], self.q["epsilon"], self.q["lp_cost"], x_lp=np.array(pend["x_lp"]))
        ang = pend["angles"]
        jobs = {j["role"]: j for j in pend["jobs"]}
        if "calibration" in jobs:
            order, cal = bit_order_from_calibration(jobs["calibration"]["result"]["counts"])
            run["calibration"] = cal
            if order is None:
                run["status"] = "failed"
                run["failure"] = (f"bit-order calibration inconclusive (P(first key bit = 1) = {cal['p_first']:.2f}, "
                                  f"P(last key bit = 1) = {cal['p_last']:.2f}); not decoding")
                self.log(f"ATTENTION {run['label']}: {run['failure']}")
                self.save(run)
                return True
            run["bit_order"] = order
            self.log(f"{run['label']}: bit-order calibration: {order} (P(last key bit = 1) = {cal['p_last']:.2f}, "
                     f"P(first key bit = 1) = {cal['p_first']:.2f}, exact key {cal['p_exact_key']:.0%} of shots)")
        wj = jobs["window"]
        counts, meta = wj["result"]["counts"], wj["result"].get("meta", {})
        reverse = run["bit_order"] == "reversed"
        probs = W.counts_to_vector(win, counts, reverse)
        ideal = W.ideal_probabilities(win, ang["gammas"], ang["betas"])
        d = W.interpret(win, probs, ideal, self.q["decode_rule"], 1.0 / self.q["shots"])
        info = {"mode": "qpu", "angles": ang, **_slim(d),
                "job": {"job_id": wj["job_id"], "batch_job_id": wj["batch_job_id"], "attempts": wj["attempt"],
                        "submitted_utc": wj["submitted_utc"], "shots": wj["shots"], "transpiled": wj["transpiled"],
                        "execution_time_s": meta.get("execution_time_in_seconds"), "shots_done": meta.get("shots_done"),
                        "created_on": meta.get("created_on"), "history": wj["history"],
                        "completed_seen_utc": wj.get("completed_seen_utc"), "platform": meta.get("platform"),
                        "backend": self.gws[run["label"]].info["name"],
                        "backend_is_hardware": self.gws[run["label"]].info.get("is_hardware"),
                        "error_mitigation": "none"},
                "n_qubits": win.n, "bit_order": run["bit_order"]}
        run["pending"] = None
        self._commit(run, pend["spec"], d["order"], info)
        return True

    # ------------------------------------------------------------------ commit / finish
    def _commit(self, run, spec, order, info) -> None:
        route = self.routes[run["route_id"]]
        M = heuristic.window_matrix(route["C"], spec)
        rec = {"iteration": run["state"]["iteration"], "u": spec["u"], "cands": spec["cands"], "k": spec["k"],
               "commit": spec["commit"], "order": order, "window_cost": routing.window_order_cost(M, order),
               "exact_cost": routing.window_order_cost(M, routing.exact_window_order(M)) if spec["k"] > 1 else 0.0,
               **info}
        run["windows"].append(rec)
        run["state"] = heuristic.apply_order(run["state"], spec, order)
        self.save(run)
        if run["mode"] == "qpu":
            self.log(f"{run['label']}: window {rec['iteration'] + 1}/{run['windows_total']} decoded; "
                     f"{len(run['state']['unvisited'])} stops left")

    def _finish(self, run, C) -> bool:
        res = heuristic.finalize(C, run["state"]["tour"])
        tour = res["tour"]
        assert sorted(tour) == list(range(len(C))) and tour[0] == 0, "final tour is not a valid open route"
        run["result"], run["status"] = res, "finished"
        self.save(run)
        if run["mode"] == "qpu":
            self.log(f"{run['label']}: FINISHED route {run['route_id'][:18]}: cost {res['cost']:.1f} "
                     f"(before 2-opt {res['cost_pre_2opt']:.1f}); Amazon {self.routes[run['route_id']]['amazon_cost']:.1f}")
        return True


def bit_order_from_calibration(counts: dict):
    """Decide the key order from the known-answer circuit (X on qubit 0, all qubits measured).

    Qiskit order puts qubit 0 in the LAST character of the key, the reverse order in the FIRST. The decision uses the
    single-bit frequencies of the first and last key character, which stay clear with readout noise on many qubits
    (the frequency of the one exact key falls with the number of qubits). Returns ('qiskit'|'reversed'|None, details).
    """
    total = sum(counts.values())
    keys = [(str(k).replace(" ", ""), v) for k, v in counts.items()]
    n = len(keys[0][0])
    p_first = sum(v for k, v in keys if k[0] == "1") / total
    p_last = sum(v for k, v in keys if k[-1] == "1") / total
    exact = counts.get("0" * (n - 1) + "1", 0) / total
    det = {"p_first": p_first, "p_last": p_last, "p_exact_key": exact, "n_bits": n, "shots": total}
    if p_last >= 0.6 and p_last - p_first >= 0.4:
        return "qiskit", det
    if p_first >= 0.6 and p_first - p_last >= 0.4:
        return "reversed", det
    return None, det


def _slim(d: dict) -> dict:
    keys = ("p_valid", "p_valid_chance", "p_valid_ideal", "objective_mean", "regret_norm", "equals_exact",
            "equals_warm_start_decode", "equals_ideal_decode", "tvd_vs_ideal", "best_valid_cost",
            "warm_start_order", "ideal_order", "best_valid_order", "marginal_order", "rule")
    return {k: d[k] for k in keys if k in d}
