"""Final results: tables (CSV/JSON/Markdown) and figures."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from .io_utils import write_json_atomic
from . import routing, figures


# --------------------------------------------------------------------------- timing
def _parse(ts):
    if ts in (None, "", "None", "nan"):
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def _secs(a, b):
    try:
        return None if a is None or b is None else (b - a).total_seconds()
    except TypeError:                                   # naive vs aware timestamps
        return None


def job_times(job: dict) -> dict:
    """Timing of one saved window job. exec_s is the platform-reported execution time of the job;
    turnaround_s = platform job creation -> platform job finished (queue + execution), falling back to our submit
    time -> result creation time for runs saved by older versions."""
    plat = job.get("platform") or {}
    created = _parse(plat.get("job_created_on")) or _parse(job.get("submitted_utc"))
    finished = _parse(plat.get("job_finished_at")) or _parse(job.get("created_on"))
    out = {"exec_s": job.get("execution_time_s"), "turnaround_s": _secs(created, finished),
           "queue_wait_platform_s": _secs(_parse(plat.get("batch_queued_at")), _parse(plat.get("batch_reserved_at"))),
           "created": created, "finished": finished}
    return out


def run_timing(run) -> dict:
    """Per-run timing. compute_s: this program's own computation; submit_s: transpile + upload;
    qpu_exec_s: sum of platform-reported job execution times; qpu_turnaround_s: sum of job creation->finished
    (= includes queue waiting); qpu_wait_s = turnaround - exec."""
    t = run.get("timing") or {}
    out = {"compute_s": t.get("compute_s"), "submit_s": t.get("submit_s"), "poll_s": t.get("poll_s")}
    if run["mode"] == "qpu":
        ex = ta = qw = 0.0
        n = n_ta = n_qw = 0
        for w in run["windows"]:
            job = w.get("job")
            if not job:
                continue
            jt = job_times(job)
            n += 1
            ex += jt["exec_s"] or 0.0
            if jt["turnaround_s"] is not None:
                ta += jt["turnaround_s"]; n_ta += 1
            if jt["queue_wait_platform_s"] is not None:
                qw += jt["queue_wait_platform_s"]; n_qw += 1
        out.update({"n_jobs": n, "qpu_exec_s": ex if n else None, "qpu_turnaround_s": ta if n_ta else None,
                    "qpu_wait_s": max(ta - ex, 0.0) if n_ta else None,
                    "qpu_queue_wait_platform_s": qw if n_qw else None})
        c, sub = out["compute_s"], out["submit_s"]
        have = c is not None and sub is not None
        out["time_incl_waiting_s"] = (c + sub + ta) if have and n_ta else None
        out["time_excl_waiting_s"] = (c + sub + ex) if have and n else None
    return out


def _runs_by(runs):
    d = {}
    for r in runs:
        d.setdefault(r["route_id"], {})[r["name"]] = r
    return d


def _cost(run):
    return run["result"]["cost"] if run and run.get("result") else None


def _pre(run):
    return run["result"]["cost_pre_2opt"] if run and run.get("result") else None


def _pct(a, b):
    return None if a is None or b is None or not a else 100.0 * (a - b) / a


def comparison_table(routes, runs, labels_k, ortools=None):
    """One row per route; columns per QPU label. `ortools`: {route_id: {cost, status, lower_bound, wall_time_s}} from the
    ORtoolsOutput approach (if it has been run)."""
    by = _runs_by(runs)
    rows = []
    for rid, route in routes.items():
        rr = by.get(rid, {})
        orr = (ortools or {}).get(rid) or route.get("ortools") or {}
        row = {"route_id": rid, "n_stops": route["n_nodes"], "amazon_cost": route["amazon_cost"],
               "ortools_cost": orr.get("cost"), "ortools_status": orr.get("status"),
               "ortools_lower_bound": orr.get("lower_bound"), "ortools_wall_s": orr.get("wall_time_s"),
               "nn_2opt_cost": _cost(rr.get("nn")),
               "nn_2opt_compute_s": (run_timing(rr["nn"])["compute_s"] if rr.get("nn") else None)}
        for label, (k, p) in labels_k.items():
            q = rr.get(label)
            row[f"{label}_k"] = k
            row[f"{label}_qubits_used"] = k * k
            row[f"{label}_cost"] = _cost(q)
            row[f"{label}_cost_pre_2opt"] = _pre(q)
            row[f"{label}_ideal_sim_cost"] = _cost(rr.get(f"ideal_k{k}_p{p}"))
            row[f"{label}_exact_window_cost"] = _cost(rr.get(f"exact_k{k}"))
            row[f"{label}_warm_start_only_cost"] = _cost(rr.get(f"warm_k{k}"))
            row[f"{label}_vs_amazon_pct"] = _pct(route["amazon_cost"], _cost(q))
            row[f"{label}_gap_to_ortools_pct"] = (None if orr.get("cost") is None or _cost(q) is None
                                                  else 100.0 * (_cost(q) - orr["cost"]) / orr["cost"])
            if q:
                tq = run_timing(q)
                for key in ("compute_s", "submit_s", "qpu_exec_s", "qpu_turnaround_s", "qpu_wait_s",
                            "time_incl_waiting_s", "time_excl_waiting_s"):
                    row[f"{label}_{key}"] = tq.get(key)
            for suffix, name in (("ideal_sim", f"ideal_k{k}_p{p}"), ("exact_window", f"exact_k{k}"),
                                 ("warm_start_only", f"warm_k{k}")):
                row[f"{label}_{suffix}_compute_s"] = run_timing(rr[name])["compute_s"] if rr.get(name) else None
        rows.append(row)
    return pd.DataFrame(rows)


def window_table(runs):
    rows = []
    for r in runs:
        for w in r["windows"]:
            if w.get("trivial"):
                continue
            job = w.get("job") or {}
            tr = job.get("transpiled") or {}
            ang = w.get("angles") or {}
            jt = job_times(job) if job else {}
            plat = job.get("platform") or {}
            rows.append({"route_id": r["route_id"], "run": r["name"], "mode": r["mode"], "k": w["k"],
                         "n_qubits": w.get("n_qubits"), "p": r["p"], "iteration": w["iteration"],
                         "delta_gamma": ang.get("delta_gamma"), "delta_beta": ang.get("delta_beta"),
                         "window_cost": w["window_cost"], "exact_cost": w["exact_cost"],
                         "regret_norm": w.get("regret_norm"), "equals_exact": w.get("equals_exact"),
                         "p_valid": w.get("p_valid"), "p_valid_ideal": w.get("p_valid_ideal"),
                         "p_valid_chance": w.get("p_valid_chance"), "objective_mean": w.get("objective_mean"),
                         "tvd_vs_ideal": w.get("tvd_vs_ideal"), "equals_ideal_decode": w.get("equals_ideal_decode"),
                         "equals_warm_start_decode": w.get("equals_warm_start_decode"),
                         "best_valid_cost": w.get("best_valid_cost"), "bit_order": w.get("bit_order"),
                         "job_id": job.get("job_id"), "batch_job_id": job.get("batch_job_id"),
                         "attempts": job.get("attempts"), "submitted_utc": job.get("submitted_utc"),
                         "created_on": job.get("created_on"), "execution_time_s": job.get("execution_time_s"),
                         "shots_done": job.get("shots_done"), "transpiled_depth": tr.get("depth"),
                         "two_qubit_gates": tr.get("two_qubit_gates"),
                         "turnaround_s": jt.get("turnaround_s"), "queue_wait_platform_s": jt.get("queue_wait_platform_s"),
                         "job_created_platform": plat.get("job_created_on"), "job_finished_platform": plat.get("job_finished_at"),
                         "ortools_window_cost": w.get("ortools_window_cost"),
                         "ortools_window_status": w.get("ortools_window_status"),
                         "ortools_matches_bruteforce": w.get("ortools_matches_bruteforce"),
                         "excess_over_ortools_pct": (100.0 * (w["window_cost"] - w["ortools_window_cost"]) / w["ortools_window_cost"]
                                                     if w.get("ortools_window_cost") else None),
                         "backend": job.get("backend"), "backend_is_hardware": job.get("backend_is_hardware"),
                         "platform_backend_is_hardware": plat.get("batch_backend_is_hardware"),
                         "platform_backend_type": plat.get("batch_backend_type_name")})
    return pd.DataFrame(rows)


def _fmt(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def summary_markdown(cmp_df, win_df, labels_k, cfg, runs, backends_info) -> str:
    L = ["# Results summary", "",
         f"Routes: {len(cmp_df)} · p = {cfg['qaoa']['p']} · shots = {cfg['qaoa']['shots']} · "
         f"Linear-ramp mode: {cfg['qaoa']['ramp_mode']} · ε = {cfg['qaoa']['epsilon']} · "
         f"LP cost: {cfg['qaoa']['lp_cost']}", "",
         "All costs are open-route travel times (seconds, no return leg), computed by one function. "
         "Lower is better. *Improvement* = 100·(Amazon − algorithm)/Amazon.", "",
         "## 0. What ran on what", ""]
    for lab, info in backends_info.items():
        hw = info.get("is_hardware")
        L.append(f"* **{lab}**: {info.get('name')}, {info.get('qubits')} qubits - "
                 + ("**REAL QUANTUM HARDWARE** (platform flag is_hardware = True)" if hw else
                    "**SIMULATION (not a quantum computer)**"))
    L += ["* *ideal simulation*, *exact windows*, *warm start only*, *nearest neighbour* and *OR-Tools* are classical "
          "computations; they never touch a quantum computer.", "", "## 1. Main comparison", ""]
    head = ["Route", "Stops", "Amazon planned", "OR-Tools (best found)", "OR-Tools status", "NN + 2-opt"]
    for lab in labels_k:
        head += [f"{lab} final", f"{lab} vs Amazon %"]
    L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in cmp_df.iterrows():
        cells = [str(r["route_id"])[:28], str(int(r["n_stops"])), _fmt(r["amazon_cost"]), _fmt(r["ortools_cost"]),
                 str(r["ortools_status"]), _fmt(r["nn_2opt_cost"])]
        for lab in labels_k:
            cells += [_fmt(r.get(f"{lab}_cost")), _fmt(r.get(f"{lab}_vs_amazon_pct"), 2)]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "OR-Tools CP-SAT status OPTIMAL means a proven global minimum; FEASIBLE means the best solution "
              "found within the time limit (see `ortools_lower_bound` in `comparison.csv`).", "",
          "## 2. Quantum device vs. classical counterparts at the same window size k", ""]
    head = ["Route", "Backend", "k", "qubits", "QPU final", "ideal simulation", "exact windows", "warm start only", "QPU − ideal"]
    L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in cmp_df.iterrows():
        for lab, (k, p) in labels_k.items():
            q, i = r.get(f"{lab}_cost"), r.get(f"{lab}_ideal_sim_cost")
            L.append("| " + " | ".join([str(r["route_id"])[:20], lab, str(k), str(k * k), _fmt(q), _fmt(i),
                                         _fmt(r.get(f"{lab}_exact_window_cost")), _fmt(r.get(f"{lab}_warm_start_only_cost")),
                                         _fmt(None if q is None or i is None else q - i)]) + " |")
    if len(win_df):
        L += ["", "## 3. Window-level diagnostics (QPU runs)", "",
              "| Backend | windows | qubits | mean valid-shot fraction | ideal valid fraction | chance | mean TVD to ideal | "
              "windows = exact optimum (QPU / ideal-sim) | decode = warm-start decode |", "|" + "---|" * 9]
        for lab in labels_k:
            w = win_df[win_df["run"] == lab]
            if not len(w):
                continue
            kk = labels_k[lab][0]
            ideal_runs = win_df[(win_df["mode"] == "ideal") & (win_df["k"] == kk)]
            L.append("| " + " | ".join([lab, str(len(w)), str(kk * kk), _fmt(w["p_valid"].mean(), 3),
                                         _fmt(w["p_valid_ideal"].mean(), 3), _fmt(w["p_valid_chance"].mean(), 5),
                                         _fmt(w["tvd_vs_ideal"].mean(), 3),
                                         f"{w['equals_exact'].mean():.2f} / " + (f"{ideal_runs['equals_exact'].mean():.2f}" if len(ideal_runs) else "–"),
                                         _fmt(w['equals_warm_start_decode'].mean(), 2)]) + " |")
    L += ["", "## 4. Timing", "",
          "Seconds. *compute* = this program's own computation (candidate selection, LP, angle optimisation on the exact "
          "simulator, circuit building, decoding, 2-opt); *transpile+upload* = compiling and submitting the circuits; "
          "*QPU exec* = execution time reported by the Quantum Inspire platform for the jobs (sum over windows); "
          "*turnaround* = platform job creation to job finished (queue + execution, sum over windows); "
          "*queue wait* = turnaround - QPU exec. **Incl. waiting** = compute + transpile/upload + turnaround; "
          "**excl. waiting** = compute + transpile/upload + QPU exec.", "",
          "| Route | Run | compute | transpile+upload | QPU exec | turnaround | queue wait | **total incl. waiting** | **total excl. waiting** |",
          "|---|---|---|---|---|---|---|---|---|"]
    by = _runs_by(runs)
    for rid in cmp_df["route_id"]:
        for name in sorted(by.get(rid, {})):
            r = by[rid][name]
            t = run_timing(r)
            L.append("| " + " | ".join([str(rid)[:20], name, _fmt(t["compute_s"], 1), _fmt(t.get("submit_s"), 1),
                                         _fmt(t.get("qpu_exec_s"), 1), _fmt(t.get("qpu_turnaround_s"), 1),
                                         _fmt(t.get("qpu_wait_s"), 1), _fmt(t.get("time_incl_waiting_s"), 1),
                                         _fmt(t.get("time_excl_waiting_s"), 1)]) + " |")
    L += ["", "OR-Tools wall time (solver limit): " + ", ".join(
        f"{str(r['route_id'])[:20]}: {_fmt(r['ortools_wall_s'], 1)} s" for _, r in cmp_df.iterrows()),
          "", "'–' means the value was not recorded (for example classical compute time of runs saved before timing was added).", "",
          "## 5. How to read this", "",
          "* The window size k grows with the backend (Tuna-5: k=2, Tuna-9: k=3, Tuna-17: k=4), so a difference between "
          "backends mixes the effect of a larger window with the effect of the hardware. The hardware effect is the "
          "difference to the *ideal simulation* at the same k.",
          "* *Warm start only* decodes the LP warm start without any QAOA layer; if the QPU and ideal results equal it, "
          "the QAOA layers did not change the decoded orders.",
          "* No quantum advantage is claimed. Raw counts are used; no error mitigation is applied.", ""]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- figures
def make_figures(routes, runs, labels_k, out_dir: Path, cfg: dict, log=print, ortools=None):
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    by = _runs_by(runs)
    limit = cfg["output"]["figure_max_routes"]
    for n_done, (rid, route) in enumerate(routes.items()):
        rr = by.get(rid, {})
        bars = [("Amazon\nplanned", route["amazon_cost"], figures.PATH_COLORS["amazon"])]
        orr = (ortools or {}).get(rid) or route.get("ortools") or {}
        route = {**route, "ortools": orr}
        if orr.get("cost") is not None:
            bars.append(("OR-Tools", orr["cost"], figures.PATH_COLORS["ortools"]))
        if rr.get("nn") and rr["nn"].get("result"):
            bars.append(("NN+2opt", _cost(rr["nn"]), figures.PATH_COLORS["nn"]))
        for lab in labels_k:
            if _cost(rr.get(lab)) is not None:
                bars.append((lab, _cost(rr[lab]), figures.RED))
        figures.cost_bars(route, bars, fig_dir)
        if n_done >= limit:
            continue                                                   # route pictures only for the first routes
        figures.route_figures_for(route, rr, fig_dir, log)
        if cfg["output"]["visualise_subroute"]:
            for name, run in rr.items():
                if run["mode"] == "qpu" and run["windows"]:
                    figures.subroute_figures_for(route, run, fig_dir)
    win = window_table(runs)
    if len(win):
        rows = []
        for lab, (k, p) in labels_k.items():
            w = win[win["run"] == lab]
            if len(w):
                rows.append((k * k, lab, w["p_valid"].mean(), w["p_valid_ideal"].mean(), w["p_valid_chance"].mean(),
                             w["regret_norm"].mean(),
                             win[(win["mode"] == "ideal") & (win["k"] == k)]["regret_norm"].mean()))
        if rows:
            figures.scaling_plot(rows, fig_dir)


def build_report(cfg, routes_list, runs, out_dir, backends_info, log=print, ortools=None):
    out = Path(out_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)
    routes = {r["route_id"]: r for r in routes_list}
    labels_k = {}
    for r in runs:
        if r["mode"] == "qpu":
            labels_k[r["name"]] = (r["k"], r["p"])
    cmp_df = comparison_table(routes, runs, labels_k, ortools)
    win_df = window_table(runs)
    cmp_df.to_csv(out / "comparison.csv", index=False)
    win_df.to_csv(out / "windows.csv", index=False)
    if len(win_df):
        win_df[win_df["mode"] == "qpu"][["route_id", "run", "iteration", "job_id", "batch_job_id", "submitted_utc",
                                         "created_on", "execution_time_s", "shots_done", "attempts"]].to_csv(out / "jobs_ledger.csv", index=False)
    write_json_atomic(out / "results.json", {"comparison": cmp_df.to_dict(orient="records"),
                                              "runs": [{k: v for k, v in r.items() if k not in ("windows",)} for r in runs]})
    (out / "summary.md").write_text(summary_markdown(cmp_df, win_df, labels_k, cfg, runs, backends_info))
    warn = None
    if cfg["output"]["figures"]:
        try:
            make_figures(routes, runs, labels_k, out, cfg, log, ortools)
        except Exception as e:                                           # figures must never lose results
            warn = f"figures were not created ({type(e).__name__}: {e})"
    return out, warn
