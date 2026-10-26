"""Reports for the three approaches and their paired comparison.

  ORtoolsOutput            the whole route solved by OR-Tools CP-SAT (baseline)
  AlgorithmWithORtoolCore  my algorithm, every window solved by OR-Tools CP-SAT
  AlgorithmWithQC          my algorithm, every window sampled on Tuna-17 (report.py)

Every approach writes into its own folder; Comparison/ only reads them. Cost is always the open-route cost of
routing.open_cost on the unrounded matrix.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from . import figures, report
from .io_utils import read_json, write_json_atomic

TIE_REL = 1e-9


def load_runs(approach_dir) -> list[dict]:
    d = Path(approach_dir) / "runs"
    return [read_json(f) for f in sorted(d.glob("*.json"))] if d.exists() else []


def _fmt(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def ortools_by_route(runs) -> dict:
    """{route_id: {cost, status, lower_bound, wall_time_s, tour, trace}} from finished ORtoolsOutput runs."""
    out = {}
    for r in runs:
        if r.get("mode") == "ortools_full" and r.get("result") and r.get("ortools"):
            o = r["ortools"]
            out[r["route_id"]] = {"cost": r["result"]["cost"], "tour": r["result"]["tour"], "status": o["status"],
                                  "lower_bound": o["lower_bound"], "wall_time_s": o["solver_wall_s"], "trace": o["trace"]}
    return out


def timing_block(run) -> dict:
    """Timing of one run with identical keys for every approach (seconds).
    compute_s     own computation (candidate selection, window preparation, decoding, 2-opt ...)
    subsolver_s   time inside CP-SAT (ORtoolsOutput: the whole solve; AlgorithmWithORtoolCore: all windows); QC: 0
    submit_s      QC only: transpile + upload;   qpu_exec_s: platform-reported execution time;   queue_wait_s: turnaround - exec
    cpu_s         process CPU time of the run (all threads; diagnostics excluded)
    wall_excl_waiting_s = compute + subsolver + submit + qpu_exec;   wall_incl_waiting_s = compute + subsolver + submit + turnaround"""
    t = run.get("timing") or {}
    out = {"compute_s": t.get("compute_s"), "subsolver_s": t.get("subsolver_s") or 0.0, "submit_s": t.get("submit_s") or 0.0,
           "cpu_s": t.get("cpu_s"), "qpu_exec_s": None, "queue_wait_s": None}
    if run.get("mode") == "qpu":
        rt = report.run_timing(run)
        out.update({"qpu_exec_s": rt.get("qpu_exec_s"), "queue_wait_s": rt.get("qpu_wait_s"),
                    "wall_incl_waiting_s": rt.get("time_incl_waiting_s"), "wall_excl_waiting_s": rt.get("time_excl_waiting_s")})
    else:
        c = t.get("compute_s")
        w = None if c is None else c + out["subsolver_s"]
        out.update({"wall_incl_waiting_s": w, "wall_excl_waiting_s": w})
    return out


def anytime_best(trace, budget_s, col):
    """Best CP-SAT objective found within `budget_s` (col 0 = wall clock, col 1 = CPU time); None if none yet."""
    if budget_s is None or not trace:
        return None
    ok = [p[2] for p in trace if p[col] <= budget_s]
    return min(ok) if ok else None


# ===================================================================================== ORtoolsOutput
def build_ortools_report(cfg, routes_list, runs, out_dir, log=print):
    out = Path(out_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)
    routes = {r["route_id"]: r for r in routes_list}
    rows, trace_rows = [], []
    for r in runs:
        if r.get("mode") != "ortools_full" or not r.get("result"):
            continue
        route, res, o, tm = routes[r["route_id"]], r["result"], r["ortools"], timing_block(r)
        tr = o["trace"]
        rows.append({
            "route_id": r["route_id"], "n_nodes": route["n_nodes"], "amazon_cost": route["amazon_cost"],
            "ortools_cost": res["cost"], "status": o["status"], "proven_optimal": o["optimal"],
            "lower_bound": o["lower_bound"], "gap_to_bound_pct": o["gap_pct"],
            "vs_amazon_pct": report._pct(route["amazon_cost"], res["cost"]),
            "cost_with_2opt_sensitivity": res["cost_with_2opt"],
            "time_limit_s": o["time_limit_s"], "workers": o["workers"], "seed": o["seed"], "hint": o["hint"],
            "solver_wall_s": o["solver_wall_s"], "solver_cpu_s": o["solver_user_s"],
            "wall_incl_waiting_s": tm["wall_incl_waiting_s"], "cpu_s": tm["cpu_s"],
            "time_to_best_solver_wall_s": tr[-1][0] if tr else None, "n_improvements": len(tr)})
        trace_rows += [{"route_id": r["route_id"], "solver_wall_s": a, "solver_cpu_s": b, "objective": c} for a, b, c in tr]
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(out / "anytime_trace.csv", index=False)
    if len(df):
        s = df.iloc[0]
        L = ["# ORtoolsOutput: the whole route solved by OR-Tools CP-SAT", "",
             f"Settings: time limit {s['time_limit_s']:g} s per route, {int(s['workers'])} workers, seed {int(s['seed'])}, "
             "relative gap limit as configured, **no start solution**. Costs are open-route travel times in seconds "
             "(no return leg). CP-SAT optimises costs rounded to the configured scale; reported costs use the unrounded matrix.", "",
             "| Route | Nodes | Amazon | OR-Tools | Status | Lower bound | vs Amazon % | solver wall s | CPU s | OR-Tools + 2-opt (sensitivity) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in df.iterrows():
            L.append(f"| {str(r['route_id'])[:24]} | {int(r['n_nodes'])} | {_fmt(r['amazon_cost'])} | {_fmt(r['ortools_cost'])} | "
                     f"{r['status']} | {_fmt(r['lower_bound'])} | {_fmt(r['vs_amazon_pct'], 2)} | {_fmt(r['solver_wall_s'])} | "
                     f"{_fmt(r['cpu_s'])} | {_fmt(r['cost_with_2opt_sensitivity'])} |")
        L += ["", "OPTIMAL = proven global minimum. FEASIBLE = best found within the time limit; the lower bound shows how far "
                  "from optimal it can be. `anytime_trace.csv` lists every improvement (solver wall time, CPU time, objective)."]
        (out / "summary.md").write_text("\n".join(L) + "\n")
    if cfg["output"]["figures"]:
        _route_pictures(cfg, routes, runs, Path(out_dir) / "results", log)
    return out


# ===================================================================================== AlgorithmWithORtoolCore
def build_core_report(cfg, routes_list, runs, out_dir, log=print):
    out = Path(out_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)
    routes = {r["route_id"]: r for r in routes_list}
    rows, wrows = [], []
    for r in runs:
        if r.get("mode") != "ortools_core":
            continue
        for w in r["windows"]:
            if w.get("trivial"):
                continue
            wrows.append({"route_id": r["route_id"], "iteration": w["iteration"], "k": w["k"], "window_cost": w["window_cost"],
                          "brute_force_cost": w["exact_cost"], "equals_brute_force": abs(w["window_cost"] - w["exact_cost"]) <= 1e-6,
                          "ortools_status": w.get("ortools_status"), "ortools_solver_wall_s": w.get("ortools_solver_wall_s")})
        if not r.get("result"):
            continue
        route, res, tm = routes[r["route_id"]], r["result"], timing_block(r)
        rows.append({"route_id": r["route_id"], "n_nodes": route["n_nodes"], "k": r["k"], "b": r["b"],
                     "windows_solved": sum(1 for w in r["windows"] if not w.get("trivial")),
                     "amazon_cost": route["amazon_cost"], "cost_pre_2opt": res["cost_pre_2opt"], "cost": res["cost"],
                     "vs_amazon_pct": report._pct(route["amazon_cost"], res["cost"]),
                     "compute_s": tm["compute_s"], "subsolver_s": tm["subsolver_s"], "cpu_s": tm["cpu_s"],
                     "wall_incl_waiting_s": tm["wall_incl_waiting_s"]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    pd.DataFrame(wrows).to_csv(out / "windows.csv", index=False)
    if len(df):
        s = df.iloc[0]
        L = ["# AlgorithmWithORtoolCore: my algorithm, every window solved by OR-Tools CP-SAT", "",
             f"Window size k = {int(s['k'])}, commit depth b = {int(s['b'])}, candidates = the k nearest unvisited stops, closing 2-opt. "
             "Identical to AlgorithmWithQC except for the window solver. Each window is solved to proven optimality "
             "(a run fails if CP-SAT cannot prove it).", "",
             "| Route | Nodes | windows | Amazon | before 2-opt | final | vs Amazon % | compute s | CP-SAT s | CPU s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in df.iterrows():
            L.append(f"| {str(r['route_id'])[:24]} | {int(r['n_nodes'])} | {int(r['windows_solved'])} | {_fmt(r['amazon_cost'])} | "
                     f"{_fmt(r['cost_pre_2opt'])} | {_fmt(r['cost'])} | {_fmt(r['vs_amazon_pct'], 2)} | {_fmt(r['compute_s'], 2)} | "
                     f"{_fmt(r['subsolver_s'], 2)} | {_fmt(r['cpu_s'], 2)} |")
        wdf = pd.DataFrame(wrows)
        if len(wdf):
            L += ["", f"Windows solved: {len(wdf)}; equal to the brute-force optimum: {wdf['equals_brute_force'].mean():.1%}."]
        (out / "summary.md").write_text("\n".join(L) + "\n")
    if cfg["output"]["figures"]:
        _route_pictures(cfg, routes, runs, Path(out_dir) / "results", log)
    return out


def _route_pictures(cfg, routes, runs, results_dir, log):
    by = {}
    for r in runs:
        by.setdefault(r["route_id"], {})[r["name"]] = r
    fig_dir = Path(results_dir) / "figures"
    for n_done, (rid, route) in enumerate(routes.items()):
        if n_done >= cfg["output"]["figure_max_routes"] or rid not in by:
            continue
        try:
            figures.route_figures_for({**route, "ortools": None}, by[rid], fig_dir, log)
        except Exception as e:                                           # figures must never lose results
            log(f"WARNING: figures for {rid} were not created ({type(e).__name__}: {e})")


# ===================================================================================== Comparison
def comparison_rows(routes, full, core, qc, nn):
    """One row per route over the approaches that have a finished run for it."""
    rows = []
    for rid, route in routes.items():
        f, c, q, n = full.get(rid), core.get(rid), qc.get(rid), nn.get(rid)
        fr = (f or {}).get("result") or {}
        row = {"route_id": rid, "n_nodes": route["n_nodes"], "amazon_cost": route["amazon_cost"],
               "ortools_cost": fr.get("cost"), "ortools_status": (f or {}).get("ortools", {}).get("status"),
               "ortools_lower_bound": (f or {}).get("ortools", {}).get("lower_bound"),
               "ortools_cost_with_2opt": fr.get("cost_with_2opt"),
               "core_cost_pre_2opt": ((c or {}).get("result") or {}).get("cost_pre_2opt"),
               "core_cost": ((c or {}).get("result") or {}).get("cost"),
               "qc_cost_pre_2opt": ((q or {}).get("result") or {}).get("cost_pre_2opt"),
               "qc_cost": ((q or {}).get("result") or {}).get("cost"),
               "nn_cost": ((n or {}).get("result") or {}).get("cost")}
        known = [row[k] for k in ("ortools_cost", "ortools_cost_with_2opt", "core_cost", "qc_cost", "nn_cost") if row[k] is not None]
        bk = min(known) if known else None
        row["best_known_cost"] = bk
        for key in ("amazon", "ortools", "core", "qc", "nn"):
            v = row[f"{key}_cost"]
            row[f"{key}_gap_pct"] = None if bk is None or v is None else 100.0 * (v - bk) / bk
        if row["ortools_cost"] and row["ortools_lower_bound"]:
            row["ortools_gap_to_bound_pct"] = 100.0 * (row["ortools_cost"] - row["ortools_lower_bound"]) / row["ortools_cost"]
        for tag, run in (("ortools", f), ("core", c), ("qc", q)):
            if run:
                for k2, v in timing_block(run).items():
                    row[f"{tag}_{k2}"] = v
        # OR-Tools at the budget the algorithms used (read off its improvement trace; no re-run)
        trace = ((f or {}).get("ortools") or {}).get("trace") or []
        row["ortools_cost_at_core_wall"] = anytime_best(trace, row.get("core_wall_excl_waiting_s"), 0)
        row["ortools_cost_at_core_cpu"] = anytime_best(trace, row.get("core_cpu_s"), 1)
        row["ortools_cost_at_qc_wall_excl"] = anytime_best(trace, row.get("qc_wall_excl_waiting_s"), 0)
        row["ortools_cost_at_qc_cpu"] = anytime_best(trace, row.get("qc_cpu_s"), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_test(df, x, y, tie_rel=TIE_REL) -> dict:
    """Paired comparison of cost columns x and y over routes where both exist. x 'wins' if its cost is lower."""
    d = df[[x, y]].dropna()
    out = {"x": x, "y": y, "n": int(len(d))}
    if not len(d):
        return out
    diff = (d[x] - d[y]).to_numpy()
    scale = np.maximum(np.abs(d[y].to_numpy()), 1e-12)
    tie = np.abs(diff) <= tie_rel * scale
    out.update({"x_wins": int(((diff < 0) & ~tie).sum()), "ties": int(tie.sum()), "y_wins": int(((diff > 0) & ~tie).sum()),
                "median_diff": float(np.median(diff)), "median_rel_diff_pct": float(np.median(100 * diff / scale)),
                "wilcoxon_p": None})
    nz = diff[~tie]
    if len(nz) >= 1:
        try:
            from scipy.stats import wilcoxon
            out["wilcoxon_p"] = float(wilcoxon(nz, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
        except ValueError:
            pass
    return out


def _describe(series) -> dict:
    s = series.dropna()
    if not len(s):
        return {}
    return {"n": int(len(s)), "median": float(s.median()), "q1": float(s.quantile(0.25)), "q3": float(s.quantile(0.75)),
            "mean": float(s.mean()), "max": float(s.max())}


def build_comparison(cfg, routes_list, out_root, log=print):
    root = Path(out_root)
    routes = {r["route_id"]: r for r in routes_list}
    full_runs, core_runs, qc_runs = (load_runs(root / a) for a in ("ORtoolsOutput", "AlgorithmWithORtoolCore", "AlgorithmWithQC"))
    fin = lambda runs, mode=None, name=None: {r["route_id"]: r for r in runs if r.get("status") == "finished" and r.get("result")
                                              and (mode is None or r["mode"] == mode) and (name is None or r["name"] == name)}
    full, core = fin(full_runs, "ortools_full"), fin(core_runs, "ortools_core")
    qc, nn = fin(qc_runs, "qpu"), fin(qc_runs, name="nn")
    present = [n for n, d in (("ORtoolsOutput", full), ("AlgorithmWithORtoolCore", core), ("AlgorithmWithQC", qc)) if d]
    if len(present) < 1:
        return None
    out = root / "Comparison"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = comparison_rows({k: v for k, v in routes.items() if k in set(full) | set(core) | set(qc)}, full, core, qc, nn)
    df.to_csv(out / "paired_comparison.csv", index=False)

    complete = df.dropna(subset=[c for c, d in (("ortools_cost", full), ("core_cost", core), ("qc_cost", qc)) if d])
    stats = {"approaches_present": present, "routes_total": int(len(df)), "routes_complete_for_all_present": int(len(complete)),
             "descriptive_gap_to_best_known_pct": {k: _describe(complete[f"{k}_gap_pct"]) for k in ("ortools", "core", "qc", "nn")
                                                    if f"{k}_gap_pct" in complete},
             "paired": []}
    pairs = [("qc_cost", "core_cost"), ("core_cost", "ortools_cost"), ("qc_cost", "ortools_cost"), ("qc_cost", "nn_cost"),
             ("qc_cost_pre_2opt", "core_cost_pre_2opt")]
    for x, y in pairs:
        if x in df and y in df:
            stats["paired"].append(paired_test(df, x, y))
    # window level: QC decode against the OR-Tools optimum on the very windows the QC run visited
    wdf = report.window_table(qc_runs) if qc_runs else pd.DataFrame()
    wq = wdf[wdf["mode"] == "qpu"] if len(wdf) else wdf
    if len(wq):
        stats["window_level_qc"] = {
            "windows": int(len(wq)), "equals_exact_fraction": float(wq["equals_exact"].astype(float).mean()),
            "mean_excess_over_ortools_pct": (float(wq["excess_over_ortools_pct"].dropna().mean())
                                             if wq["excess_over_ortools_pct"].notna().any() else None),
            "mean_valid_shot_fraction": float(wq["p_valid"].mean()), "mean_chance_level": float(wq["p_valid_chance"].mean()),
            "ortools_matches_bruteforce_fraction": (float(wq["ortools_matches_bruteforce"].dropna().astype(float).mean())
                                                    if wq["ortools_matches_bruteforce"].notna().any() else None)}
    write_json_atomic(out / "statistics.json", stats)
    (out / "statistics.md").write_text(_stats_markdown(stats, df, cfg))
    if cfg["output"]["figures"]:
        try:
            _comparison_figures(df, full, out / "figures", cfg)
        except Exception as e:                                           # figures must never lose results
            log(f"WARNING: comparison figures were not created ({type(e).__name__}: {e})")
    return out


def _stats_markdown(stats, df, cfg) -> str:
    L = ["# Comparison of the three approaches", "",
         f"Approaches with results: {', '.join(stats['approaches_present'])}. Routes: {stats['routes_total']} "
         f"({stats['routes_complete_for_all_present']} with every present approach finished). One run per route and approach.", "",
         "**Cost** = open-route travel time in seconds (no return leg), the same function for every approach. "
         "**Best known** = lowest cost any approach (or the OR-Tools + 2-opt sensitivity, or nearest neighbour) found on that route; "
         "**gap** = 100 (cost - best known) / best known. Final costs include the closing 2-opt for both algorithm approaches; "
         "ORtoolsOutput is raw OR-Tools output.", "",
         "## 1. Gap to the best known cost (%)", "", "| Approach | routes | median | Q1 | Q3 | mean | max |", "|---|---|---|---|---|---|---|"]
    names = {"ortools": "ORtoolsOutput", "core": "AlgorithmWithORtoolCore", "qc": "AlgorithmWithQC", "nn": "nearest neighbour + 2-opt (control)"}
    for k, d in stats["descriptive_gap_to_best_known_pct"].items():
        if d:
            L.append(f"| {names[k]} | {d['n']} | {d['median']:.2f} | {d['q1']:.2f} | {d['q3']:.2f} | {d['mean']:.2f} | {d['max']:.2f} |")
    L += ["", "## 2. Paired comparisons (x vs y; x wins when its cost is lower)", "",
          "| x | y | n | x wins | ties | y wins | median diff (s) | median rel. diff % | Wilcoxon p (two-sided) |", "|---|---|---|---|---|---|---|---|---|"]
    for t in stats["paired"]:
        if t["n"]:
            p = "–" if t["wilcoxon_p"] is None else f"{t['wilcoxon_p']:.3g}"
            L.append(f"| {t['x']} | {t['y']} | {t['n']} | {t['x_wins']} | {t['ties']} | {t['y_wins']} | {t['median_diff']:.1f} | "
                     f"{t['median_rel_diff_pct']:.2f} | {p} |")
    L += ["", "With few routes the Wilcoxon test has little power (n = 5 cannot give p below 0.0625); read the win/tie/loss counts "
              "and the effect sizes first.", ""]
    w = stats.get("window_level_qc")
    if w:
        L += ["## 3. Window level (AlgorithmWithQC, the windows the QC run actually visited)", "",
              f"* windows: {w['windows']}", f"* decoded order equals the optimum: {w['equals_exact_fraction']:.1%}",
              f"* mean excess of the decoded window over the OR-Tools window optimum: "
              f"{'–' if w['mean_excess_over_ortools_pct'] is None else format(w['mean_excess_over_ortools_pct'], '.2f') + ' %'}",
              f"* valid-shot fraction {w['mean_valid_shot_fraction']:.4f} vs chance {w['mean_chance_level']:.4f}",
              f"* CP-SAT optimum equals brute force on these windows: "
              f"{'–' if w['ortools_matches_bruteforce_fraction'] is None else format(w['ortools_matches_bruteforce_fraction'], '.1%')}", ""]
    L += ["## 4. Runtime (median over routes, seconds)", "",
          "Definitions: *compute* = own computation; *CP-SAT* = time inside OR-Tools; *wall excl. waiting* = compute + CP-SAT + "
          "transpile/upload + platform-reported QPU execution time; *wall incl. waiting* = the same with the queue turnaround instead of "
          "the execution time; *CPU* = process CPU time over all threads. ORtoolsOutput uses the configured number of workers, "
          "the algorithms are single-threaded, so compare CPU as well as wall time.", "",
          "| Approach | compute | CP-SAT | QPU exec | queue wait | wall excl. waiting | wall incl. waiting | CPU |", "|---|---|---|---|---|---|---|---|"]
    for tag, nm in (("ortools", "ORtoolsOutput"), ("core", "AlgorithmWithORtoolCore"), ("qc", "AlgorithmWithQC")):
        def med(col):
            return df[f"{tag}_{col}"].median() if f"{tag}_{col}" in df and df[f"{tag}_{col}"].notna().any() else None
        if f"{tag}_compute_s" in df:
            L.append(f"| {nm} | {_fmt(med('compute_s'), 2)} | {_fmt(med('subsolver_s'), 2)} | {_fmt(med('qpu_exec_s'), 1)} | "
                     f"{_fmt(med('queue_wait_s'), 1)} | {_fmt(med('wall_excl_waiting_s'), 1)} | {_fmt(med('wall_incl_waiting_s'), 1)} | "
                     f"{_fmt(med('cpu_s'), 2)} |")
    L += ["", "`paired_comparison.csv` also holds `ortools_cost_at_*`: the OR-Tools cost reached within the wall or CPU time an algorithm "
              "run used (read off the OR-Tools improvement trace, rounded-cost objective). `-` = OR-Tools had no solution yet.", ""]
    return "\n".join(L)


def _comparison_figures(df, full, fig_dir, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.sort_values("n_nodes").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(d) + 3), 4.6))
    x = np.arange(len(d))
    for key, nm, col, mk in (("ortools", "ORtoolsOutput", figures.PATH_COLORS["ortools"], "o"),
                             ("core", "AlgorithmWithORtoolCore", "#f9ab00", "s"), ("qc", "AlgorithmWithQC", figures.RED, "^"),
                             ("nn", "NN + 2-opt (control)", figures.PATH_COLORS["nn"], "x")):
        if f"{key}_gap_pct" in d and d[f"{key}_gap_pct"].notna().any():
            ax.plot(x, d[f"{key}_gap_pct"], mk + "-", color=col, label=nm, lw=1.0, ms=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n)}" for n in d["n_nodes"]])
    ax.set_xlabel("route (number of nodes)")
    ax.set_ylabel("gap to best known cost (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"gap_to_best_known.{ext}", dpi=200)
    plt.close(fig)
    # cost against time: OR-Tools improvement curve vs the algorithm runs (single points)
    rows = d.head(cfg["output"]["figure_max_routes"])
    if not len(rows):
        return
    ncol = min(3, len(rows))
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.9 * nrow), squeeze=False)
    for ax, (_, r) in zip(axs.ravel(), rows.iterrows()):
        run = full.get(r["route_id"])
        tr = ((run or {}).get("ortools") or {}).get("trace") or []
        if tr:
            t_end = max([r.get(f"{t}_wall_excl_waiting_s") or 0 for t in ("core", "qc")] + [tr[-1][0]])
            ax.step([p[0] for p in tr] + [t_end], [p[2] for p in tr] + [tr[-1][2]], where="post",
                    color=figures.PATH_COLORS["ortools"], label="OR-Tools (anytime)")
        for tag, nm, col, mk in (("core", "AlgorithmWithORtoolCore", "#f9ab00", "s"), ("qc", "AlgorithmWithQC", figures.RED, "^")):
            t, c = r.get(f"{tag}_wall_excl_waiting_s"), r.get(f"{tag}_cost")
            if pd.notna(t) and pd.notna(c):
                ax.plot([max(t, 1e-3)], [c], mk, color=col, ms=9, label=f"{nm} (excl. waiting)")
            t2 = r.get(f"{tag}_wall_incl_waiting_s")
            if tag == "qc" and pd.notna(t2) and pd.notna(c):
                ax.plot([max(t2, 1e-3)], [c], mk, color=col, mfc="none", ms=9, label="AlgorithmWithQC (incl. waiting)")
        ax.set_xscale("log")
        ax.set_title(f"{int(r['n_nodes'])} nodes", fontsize=10)
        ax.set_xlabel("wall time (s)")
        ax.set_ylabel("cost (s)")
        ax.grid(alpha=0.3)
    handles, labels = axs.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"cost_vs_time.{ext}", dpi=200)
    plt.close(fig)
