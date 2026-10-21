"""Figures.

* route_figure      one picture per path (Amazon, OR-Tools, every algorithm run), with and without depot.
* subroute_figure   one picture per QPU window: the order decoded from the QPU (blue) and the part that was
                    committed to the route (red).
* cost_bars, scaling_plot   summary plots.

Geometry: stop coordinates are converted to kilometres (equirectangular, east/north) so that distances are not
distorted, and every canvas is sized from the exact bounds of what it shows, so the route fills the picture.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

BLUE, RED, GREY = "#1a73e8", "#d93025", "#9aa0a6"
PATH_COLORS = {"amazon": "#188038", "ortools": "#1a73e8", "nn": "#5f6368"}


def safe_name(name, n: int = 60) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))[:n]


def to_km(coords) -> np.ndarray:
    """(lat, lng) rows -> (east, north) in km around the centroid."""
    pts = np.asarray(coords, dtype=float)
    lat0 = pts[:, 0].mean()
    x = (pts[:, 1] - pts[:, 1].mean()) * np.cos(np.radians(lat0)) * 111.32
    y = (pts[:, 0] - lat0) * 110.57
    return np.c_[x, y]


def _canvas(points: np.ndarray, pad_frac: float = 0.05, base_w: float = 12.0, min_span: float = 0.02):
    """Figure whose single axes has exactly the aspect ratio of the data bounds (no empty margins)."""
    import matplotlib.pyplot as plt
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    w, h = max(xmax - xmin, min_span), max(ymax - ymin, min_span)
    pad = pad_frac * max(w, h)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    w, h = w + 2 * pad, h + 2 * pad
    left, right, bottom, top = 0.95, 0.25, 0.80, 1.00                     # margins in inches
    ax_w = base_w - left - right
    ax_h = ax_w * h / w
    fig_h = ax_h + bottom + top
    if fig_h > 16:                                                          # very tall route: limit the height
        ax_h = 16 - bottom - top
        ax_w = ax_h * w / h
        base_w, fig_h = ax_w + left + right, 16.0
    elif fig_h < 5:                                                         # very wide route
        fig_h = 5.0
        ax_h = fig_h - bottom - top
    fig = plt.figure(figsize=(base_w, fig_h))
    ax = fig.add_axes([left / base_w, bottom / fig_h, ax_w / base_w, ax_h / fig_h])
    ax.set_xlim(cx - w / 2, cx + w / 2)
    ax.set_ylim(cy - h / 2, cy + h / 2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("East (km)", fontsize=10)
    ax.set_ylabel("North (km)", fontsize=10)
    return fig, ax, fig_h


def _title(fig, fig_h, text: str, size: float = 12.5):
    fig.text(0.5, 1 - 0.12 / fig_h, text, ha="center", va="top", fontsize=size)


# ------------------------------------------------------------------------------------------ one path per picture
def route_figure(route: dict, path_name: str, seq, with_depot: bool, color: str, out_base: Path,
                 cost: float, amazon_cost: float | None = None, dpi: int = 220) -> None:
    import matplotlib.pyplot as plt
    xy = to_km(route["coordinates"])
    n = len(xy)
    ids = [i for i in seq if with_depot or i != 0]
    view = xy if with_depot else xy[1:]
    fig, ax, fig_h = _canvas(view)
    dot = float(np.clip(70 - 0.35 * n, 14, 60))
    ax.scatter(view[:, 0], view[:, 1], s=dot, c=GREY, edgecolors="white", linewidths=0.4, zorder=2)
    P = xy[ids]
    ax.plot(P[:, 0], P[:, 1], "-", color=color, lw=1.9, zorder=3, solid_joinstyle="round")
    step = max(1, len(ids) // 10)
    for a in range(0, len(ids) - 1, step):
        ax.annotate("", xy=P[a + 1], xytext=P[a], zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, mutation_scale=15, shrinkA=0, shrinkB=0))
    lab = max(1, len(ids) // 12)
    for j in range(0, len(ids), lab):
        ax.annotate(str(j if with_depot else j + 1), P[j], xytext=(5, 5), textcoords="offset points", fontsize=8,
                    color="#202124", zorder=6)
    if with_depot:
        ax.scatter(*xy[0], marker="*", s=420, c="gold", edgecolors="k", linewidths=1.0, zorder=7, label="depot (start)")
    else:
        ax.scatter(*P[0], marker="o", s=150, c="limegreen", edgecolors="k", zorder=7, label="first stop")
    ax.scatter(*P[-1], marker="s", s=150, c=RED, edgecolors="k", zorder=7, label="last stop")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    txt = f"{route['route_id'][:42]}  ·  {path_name}  ·  {len(ids) - (1 if with_depot else 0)} stops"
    cost_line = f"open-route cost {cost:,.0f} s"
    if with_depot:
        if amazon_cost:
            cost_line += f"   ·   vs Amazon planned {100 * (amazon_cost - cost) / amazon_cost:+.2f}%"
        title = f"{txt}\n{cost_line}   (with depot)"
    else:
        from .routing import open_cost
        stop_cost = open_cost(route["C"], ids)
        title = f"{txt}\nstop-to-stop cost {stop_cost:,.0f} s (depot leg removed)   (without depot)"
    _title(fig, fig_h, title)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix("." + ext), dpi=dpi)
    plt.close(fig)


def route_figures_for(route: dict, runs_of_route: dict, out_dir: Path, log=print) -> int:
    """All path pictures of one route: Amazon, OR-Tools and every finished run, with and without depot."""
    from .routing import open_cost
    C = route["C"]
    paths = [("amazon_planned", route["amazon_sequence"], PATH_COLORS["amazon"])]
    orr = route.get("ortools") or {}
    if orr.get("tour"):
        paths.append(("ortools", orr["tour"], PATH_COLORS["ortools"]))
    palette = ["#ea4335", "#fa7b17", "#a142f4", "#12b5cb", "#e52592", "#7baaf7", "#c26401"]
    for k, (name, run) in enumerate(sorted(runs_of_route.items())):
        if run.get("result"):
            col = PATH_COLORS["nn"] if name == "nn" else palette[k % len(palette)]
            paths.append((name, run["result"]["tour"], col))
    base = Path(out_dir) / "routes" / safe_name(route["route_id"])
    count = 0
    for name, seq, col in paths:
        cost = open_cost(C, seq)
        for with_depot in (True, False):
            tag = "with_depot" if with_depot else "without_depot"
            route_figure(route, name, seq, with_depot, col, base / f"{safe_name(name)}__{tag}", cost, route["amazon_cost"])
            count += 1
    return count


# ------------------------------------------------------------------------------------------ one window per picture
def subroute_figure(route: dict, run: dict, w: dict, tour_so_far: list, out_path: Path, dpi: int = 140) -> None:
    """Blue = full order decoded from the QPU for this window; red = the part committed to the route."""
    import matplotlib.pyplot as plt
    xy = to_km(route["coordinates"])
    u = w["u"]
    nodes = [w["cands"][i] for i in w["order"]]
    committed = nodes[:w["commit"]]
    focus = xy[[u] + nodes]
    fig, ax, fig_h = _canvas(focus, pad_frac=0.35, base_w=9.0, min_span=0.15)
    ax.scatter(xy[:, 0], xy[:, 1], s=22, c="#d0d3d8", edgecolors="white", linewidths=0.4, zorder=1)
    tail = tour_so_far[-6:]
    if len(tail) > 1:
        ax.plot(xy[tail, 0], xy[tail, 1], ls=":", c="#80868b", lw=1.6, zorder=2, label="route so far (tail)")
    B = xy[[u] + nodes]
    ax.plot(B[:, 0], B[:, 1], color=BLUE, lw=9, alpha=0.45, solid_capstyle="round", zorder=3,
            label=f"QPU-decoded order ({len(nodes)} stops)")
    R = xy[[u] + committed]
    ax.plot(R[:, 0], R[:, 1], color=RED, lw=2.6, zorder=4, label=f"committed to route (first {len(committed)})")
    for pos, node in enumerate(nodes, 1):
        c = RED if node in committed else BLUE
        ax.scatter(*xy[node], s=230, c=c, edgecolors="k", linewidths=0.9, zorder=5)
        ax.text(xy[node, 0], xy[node, 1], str(pos), color="white", ha="center", va="center", fontsize=9,
                fontweight="bold", zorder=6)
    ax.scatter(*xy[u], marker="D", s=190, c="black", edgecolors="white", zorder=5, label="start of window")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    pv = w.get("p_valid")
    pv_txt = f" · valid shots {pv:.3f}" if pv is not None else ""
    eq = " (= exact optimum)" if w.get("equals_exact") else ""
    title = (f"{route['route_id'][:30]} · {run['name']} · window {w['iteration'] + 1}/{run['windows_total']}\n"
             f"k={w['k']}, committed b={w['commit']}{pv_txt} · window cost {w['window_cost']:,.0f} s, "
             f"exact optimum {w['exact_cost']:,.0f} s{eq}")
    _title(fig, fig_h, title, size=10.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def subroute_figures_for(route: dict, run: dict, out_dir: Path) -> int:
    base = Path(out_dir) / "subroutes" / safe_name(route["route_id"]) / safe_name(run["name"])
    tour, count = [0], 0
    for w in run["windows"]:
        if w.get("trivial") or w["k"] < 2:
            tour += [w["cands"][i] for i in w["order"]][:w["commit"]] if w.get("cands") else []
            continue
        subroute_figure(route, run, w, tour, base / f"window_{w['iteration'] + 1:03d}.png")
        tour += [w["cands"][i] for i in w["order"]][:w["commit"]]
        count += 1
    return count


# ------------------------------------------------------------------------------------------ summary plots
def cost_bars(route: dict, bars: list, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.3 * len(bars) + 2, 4.6))
    ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars])
    for x, b in enumerate(bars):
        ax.text(x, b[1], f"{b[1]:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("open-route cost (s)")
    ax.set_title(f"{route['route_id'][:40]} · {route['n_nodes']} nodes")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(out_dir) / f"cost_comparison__{safe_name(route['route_id'])}.{ext}", dpi=200)
    plt.close(fig)


def scaling_plot(rows: list, out_dir: Path) -> None:
    """rows: (qubits, label, p_valid_device, p_valid_ideal, p_valid_chance, regret_device, regret_ideal)."""
    import matplotlib.pyplot as plt
    rows = sorted(rows)
    x = [r[0] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    axs[0].plot(x, [r[2] for r in rows], "o-", c=RED, label="device")
    axs[0].plot(x, [r[3] for r in rows], "s--", c=BLUE, label="ideal simulation")
    axs[0].plot(x, [r[4] for r in rows], "^:", c="k", label="chance")
    axs[0].set_yscale("log"); axs[0].set_xlabel("qubits used (k²)"); axs[0].set_ylabel("valid-shot fraction"); axs[0].legend()
    axs[1].plot(x, [r[5] for r in rows], "o-", c=RED, label="device")
    axs[1].plot(x, [r[6] for r in rows], "s--", c=BLUE, label="ideal simulation")
    axs[1].set_xlabel("qubits used (k²)"); axs[1].set_ylabel("mean normalised window regret"); axs[1].legend()
    for a, r in zip(x, rows):
        axs[0].annotate(r[1], (a, r[2]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(out_dir) / f"scaling.{ext}", dpi=200)
    plt.close(fig)
