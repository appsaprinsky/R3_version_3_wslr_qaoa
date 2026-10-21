"""Final results: tables (CSV/JSON/Markdown) and figures."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .io_utils import write_json_atomic
from . import routing


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


def comparison_table(routes, runs, labels_k):
    """One row per route; columns per QPU label."""
    by = _runs_by(runs)
    rows = []
    for rid, route in routes.items():
        rr = by.get(rid, {})
        orr = route.get("ortools") or {}
        row = {"route_id": rid, "n_stops": route["n_nodes"], "amazon_cost": route["amazon_cost"],
               "ortools_cost": orr.get("cost"), "ortools_status": orr.get("status"),
               "ortools_lower_bound": orr.get("lower_bound"),
               "nn_2opt_cost": _cost(rr.get("nn"))}
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
                         "two_qubit_gates": tr.get("two_qubit_gates")})
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
         "## 1. Main comparison", ""]
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
    L += ["", "## 4. How to read this", "",
          "* The window size k grows with the backend (Tuna-5: k=2, Tuna-9: k=3, Tuna-17: k=4), so a difference between "
          "backends mixes the effect of a larger window with the effect of the hardware. The hardware effect is the "
          "difference to the *ideal simulation* at the same k.",
          "* *Warm start only* decodes the LP warm start without any QAOA layer; if the QPU and ideal results equal it, "
          "the QAOA layers did not change the decoded orders.",
          "* No quantum advantage is claimed. Raw counts are used; no error mitigation is applied.", ""]
    for lab, info in backends_info.items():
        L.append(f"* Backend {lab}: {info.get('name')} ({info.get('qubits')} qubits)")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- figures
def _plot_route(ax, coords, seq, depot, with_depot, title, color):
    pts = np.asarray(coords)
    xy = pts[:, ::-1]                                   # x = lng, y = lat
    ids = [i for i in seq if with_depot or i != depot]
    mask = np.ones(len(xy), bool)
    if with_depot:
        ax.scatter(xy[:, 0], xy[:, 1], s=10, c="#9aa0a6", zorder=1)
        ax.scatter(*xy[depot], marker="*", s=220, c="gold", edgecolors="k", zorder=5, label="depot")
    else:
        mask[depot] = False
        ax.scatter(xy[mask, 0], xy[mask, 1], s=10, c="#9aa0a6", zorder=1)
    P = xy[ids]
    ax.plot(P[:, 0], P[:, 1], "-", color=color, lw=1.3, alpha=1.0, zorder=3)
    step = max(1, len(ids) // 8)
    for a in range(0, len(ids) - 1, step):
        ax.annotate("", xy=P[a + 1], xytext=P[a], arrowprops=dict(arrowstyle="->", color=color, lw=1.2), zorder=4)
    ax.scatter(*P[0], marker="o", s=60, c="limegreen", edgecolors="k", zorder=6, label="start")
    ax.scatter(*P[-1], marker="s", s=60, c="crimson", edgecolors="k", zorder=6, label="end")
    ax.set_title(title, fontsize=8)
    pad_x = 0.05 * max(np.ptp(xy[:, 0]), 1e-9); pad_y = 0.05 * max(np.ptp(xy[:, 1]), 1e-9)
    ax.set_xlim(xy[:, 0].min() - pad_x, xy[:, 0].max() + pad_x)
    ax.set_ylim(xy[:, 1].min() - pad_y, xy[:, 1].max() + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=7)


def make_figures(routes, runs, labels_k, out_dir: Path, log=print):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    by = _runs_by(runs)
    for rid, route in routes.items():
        C = route["C"]
        rr = by.get(rid, {})
        # cost bars
        bars = [("Amazon\nplanned", route["amazon_cost"], "#34a853")]
        orr = route.get("ortools") or {}
        if orr.get("cost") is not None:
            bars.append(("OR-Tools", orr["cost"], "#1a73e8"))
        if rr.get("nn") and rr["nn"].get("result"):
            bars.append(("NN+2opt", _cost(rr["nn"]), "#9aa0a6"))
        for lab in labels_k:
            if _cost(rr.get(lab)) is not None:
                bars.append((lab, _cost(rr[lab]), "#ea4335"))
        fig, ax = plt.subplots(figsize=(1.2 * len(bars) + 2, 4))
        ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars])
        for x, b in enumerate(bars):
            ax.text(x, b[1], f"{b[1]:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("open-route cost (s)")
        ax.set_title(f"{rid[:40]} · {route['n_nodes']} stops")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"cost_comparison__{rid[:40]}.{ext}", dpi=200)
        plt.close(fig)
        # route maps
        for lab in labels_k:
            q = rr.get(lab)
            if not q or not q.get("result"):
                continue
            tracks = [("Amazon planned", route["amazon_sequence"], "#34a853")]
            if orr.get("tour"):
                tracks.append(("OR-Tools", orr["tour"], "#1a73e8"))
            tracks.append((f"{lab} (final)", q["result"]["tour"], "#ea4335"))
            for with_depot in (True, False):
                fig, axs = plt.subplots(1, len(tracks), figsize=(4.6 * len(tracks), 5.2))
                axs = np.atleast_1d(axs)
                for ax, (name, seq, col) in zip(axs, tracks):
                    c = routing.open_cost(C, seq)
                    t = f"{name}\ncost {c:,.0f}"
                    if not with_depot:
                        t += f"  (stop-to-stop, depot leg excluded: {routing.open_cost(C, seq[1:]):,.0f})"
                    _plot_route(ax, route["coordinates"], seq, 0, with_depot, t, col)
                imp = _pct(route["amazon_cost"], q["result"]["cost"])
                fig.suptitle(f"{rid[:40]} · {route['n_nodes']} stops · {lab} · p={q['p']} · "
                             f"improvement vs Amazon {imp:+.2f}%", fontsize=10)
                fig.tight_layout()
                tag = "with_depot" if with_depot else "without_depot"
                for ext in ("png", "pdf"):
                    fig.savefig(fig_dir / f"route__{rid[:40]}__{lab}__{tag}.{ext}", dpi=200)
                plt.close(fig)
    # scaling plot: valid-shot fraction and window regret vs qubits used
    win = window_table(runs)
    if len(win):
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
        rows = []
        for lab, (k, p) in labels_k.items():
            w = win[win["run"] == lab]
            if len(w):
                rows.append((k * k, lab, w["p_valid"].mean(), w["p_valid_ideal"].mean(), w["p_valid_chance"].mean(),
                             w["regret_norm"].mean(),
                             win[(win["mode"] == "ideal") & (win["k"] == k)]["regret_norm"].mean()))
        if rows:
            rows.sort()
            x = [r[0] for r in rows]
            axs[0].plot(x, [r[2] for r in rows], "o-", c="#ea4335", label="device")
            axs[0].plot(x, [r[3] for r in rows], "s--", c="#1a73e8", label="ideal simulation")
            axs[0].plot(x, [r[4] for r in rows], "^:", c="k", label="chance")
            axs[0].set_yscale("log"); axs[0].set_xlabel("qubits used (k²)"); axs[0].set_ylabel("valid-shot fraction"); axs[0].legend()
            axs[1].plot(x, [r[5] for r in rows], "o-", c="#ea4335", label="device")
            axs[1].plot(x, [r[6] for r in rows], "s--", c="#1a73e8", label="ideal simulation")
            axs[1].set_xlabel("qubits used (k²)"); axs[1].set_ylabel("mean normalised window regret"); axs[1].legend()
            for a, r in zip(x, rows):
                axs[0].annotate(r[1], (a, r[2]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(fig_dir / f"scaling.{ext}", dpi=200)
        plt.close(fig)


def build_report(cfg, routes_list, runs, out_dir, backends_info, log=print):
    out = Path(out_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)
    routes = {r["route_id"]: r for r in routes_list}
    labels_k = {}
    for r in runs:
        if r["mode"] == "qpu":
            labels_k[r["name"]] = (r["k"], r["p"])
    cmp_df = comparison_table(routes, runs, labels_k)
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
            make_figures(routes, runs, labels_k, out, log)
        except Exception as e:                                           # figures must never lose results
            warn = f"figures were not created ({type(e).__name__}: {e})"
    return out, warn
