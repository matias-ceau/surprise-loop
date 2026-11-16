"""Plot an experiment's CSV tables.

    uv run --group plot python scripts/plot_results.py EXPERIMENT_DIR [--smooth N]

EXPERIMENT_DIR is the `DIR/<experiment name>/` folder `surprise-loop experiment` writes. The
script reads only its CSVs (results.csv, pe_trace.csv, stability.csv) and writes three figures
into EXPERIMENT_DIR/plots/:

- metrics_by_agent.png: one panel per behavioural metric, one row per agent, each seed a point
  (hollow when the value is censored), the seed mean a black bar; the panel title carries the
  metric's ICC(1) from stability.csv.
- pe_trace.png: unsigned prediction error per world step, mean over seeds, one panel per agent,
  smoothed by a centred N-step moving average inside each phase (N=1 turns it off).
- cumulative_record.png: cumulative active presses per world step, one panel per agent, each
  seed a thin line and the seed mean a thick one, the other agents' means in grey behind.

Agents keep one colour across the three figures, in the order they first appear in results.csv.
Exits 2 with a one-line message on stderr when the directory or a CSV is missing.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REQUIRED = ("results.csv", "pe_trace.csv", "stability.csv")
DPI = 150

# Categorical palette, light mode, in fixed order (validated for adjacent-pair CVD separation).
# More agents than slots reuse the hues with a dashed line style, never generated hues.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e4e3df"
CONTEXT = "#cfcec9"

# (column, panel title, x-axis label, censored-flag column or None)
METRICS = (
    (
        "breakpoint",
        "Breakpoint (PR)",
        "last ratio completed before quitting",
        "breakpoint_censored",
    ),
    (
        "extinction_persistence",
        "Extinction persistence",
        "active presses before quitting",
        "extinction_censored",
    ),
    ("response_rate_fr", "Response rate, FR", "active presses per step", None),
    ("response_rate_pr", "Response rate, PR", "active presses per step", None),
    ("response_rate_ext", "Response rate, extinction", "active presses per step", None),
    ("inactive_rate", "Inactive-lever rate", "inactive presses per step", None),
    ("mean_abs_pe", "Mean |reward PE|", "mean |reward - expected| per step", None),
)
PHASE_LABELS = {"fixed_ratio": "FR", "progressive_ratio": "PR", "extinction": "EXT"}


class PlotError(Exception):
    pass


# --- reading --------------------------------------------------------------------------------------


def read_table(directory: Path, name: str) -> list[dict[str, str]]:
    path = directory / name
    if not path.is_file():
        raise PlotError(f"{path} not found (run `surprise-loop experiment` first)")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise PlotError(f"{path} has no rows")
    return rows


def agent_order(results: list[dict[str, str]]) -> list[str]:
    return list(dict.fromkeys(r["agent"] for r in results))


def agent_style(agents: list[str]) -> dict[str, tuple[str, str]]:
    """Colour and line style per agent, stable across figures."""
    return {
        a: (PALETTE[i % len(PALETTE)], "-" if i < len(PALETTE) else "--")
        for i, a in enumerate(agents)
    }


def phase_starts(trace: list[dict[str, str]]) -> list[tuple[int, int, str]]:
    """(first step, last step, kind) of each phase_index reached, in order."""
    spans: dict[int, list] = {}
    for r in trace:
        index, step = int(r["phase_index"]), int(r["step"])
        span = spans.setdefault(index, [step, step, r["phase_kind"]])
        span[0], span[1] = min(span[0], step), max(span[1], step)
    return [tuple(spans[i]) for i in sorted(spans)]  # type: ignore[misc]


def by_run(trace: list[dict[str, str]]) -> dict[str, dict[str, list[dict[str, str]]]]:
    """agent -> seed -> rows in step order."""
    runs: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for r in trace:
        runs[r["agent"]][r["seed"]].append(r)
    for seeds in runs.values():
        for rows in seeds.values():
            rows.sort(key=lambda r: int(r["step"]))
    return runs


def seed_mean(series: list[dict[int, float]]) -> tuple[list[int], list[float]]:
    """Mean over seeds at each step any seed reached."""
    steps = sorted({s for one in series for s in one})
    means = []
    for s in steps:
        values = [one[s] for one in series if s in one]
        means.append(sum(values) / len(values))
    return steps, means


def smooth_within_phases(
    steps: list[int], values: list[float], phases: list[int], window: int
) -> list[float]:
    """Centred moving average of `window` steps that never crosses a phase boundary."""
    if window <= 1:
        return list(values)
    half = window // 2
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        kept = [values[j] for j in range(lo, hi) if phases[j] == phases[i]]
        out.append(sum(kept) / len(kept))
    return out


# --- figure helpers -------------------------------------------------------------------------------


def style_axes(ax: Axes, grid_axis: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def mark_phases(ax: Axes, phases: list[tuple[int, int, str]], label: bool) -> None:
    for i, (first, last, kind) in enumerate(phases):
        if i > 0:
            ax.axvline(first - 0.5, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
        if label:
            ax.text(
                (first + last) / 2,
                1.0,
                PHASE_LABELS.get(kind, kind),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK_SECONDARY,
            )


def grid_shape(panels: int) -> tuple[int, int]:
    cols = 1 if panels == 1 else 2
    return (panels + cols - 1) // cols, cols


def new_figure(rows: int, cols: int, width: float, height: float, **kw):  # type: ignore[no-untyped-def]
    fig, axes = plt.subplots(
        rows, cols, figsize=(width, height), squeeze=False, facecolor=SURFACE, **kw
    )
    return fig, [ax for row in axes for ax in row]


def save(fig, path: Path) -> None:  # type: ignore[no-untyped-def]
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


# --- figures --------------------------------------------------------------------------------------


def plot_metrics(
    results: list[dict[str, str]],
    stability: list[dict[str, str]],
    agents: list[str],
    styles: dict[str, tuple[str, str]],
    title: str,
    path: Path,
) -> None:
    icc = {r["metric"]: r["icc1"] for r in stability}
    seeds = list(dict.fromkeys(r["seed"] for r in results))
    offsets = {
        s: (i - (len(seeds) - 1) / 2) * (0.5 / max(len(seeds) - 1, 1)) for i, s in enumerate(seeds)
    }
    metrics = [m for m in METRICS if m[0] in results[0]]
    cols = 4
    rows = (len(metrics) + 1 + cols - 1) // cols  # one extra slot for the key
    fig, axes = new_figure(
        rows, cols, 3.6 * cols, 0.32 * len(agents) * rows + 1.2 * rows + 0.6, sharey=True
    )
    ypos = {a: i for i, a in enumerate(agents)}

    for ax, (column, name, unit, flag) in zip(axes, metrics, strict=False):
        style_axes(ax, "x")
        for r in results:
            colour = styles[r["agent"]][0]
            censored = flag is not None and r[flag] == "True"
            ax.plot(
                float(r[column]),
                ypos[r["agent"]] + offsets[r["seed"]],
                marker="o",
                markersize=6,
                markerfacecolor=SURFACE if censored else colour,
                markeredgecolor=colour if censored else SURFACE,
                markeredgewidth=1.4 if censored else 0.8,
                linestyle="none",
                zorder=3,
            )
        for a in agents:
            values = [float(r[column]) for r in results if r["agent"] == a]
            mean = sum(values) / len(values)
            ax.plot([mean, mean], [ypos[a] - 0.34, ypos[a] + 0.34], color=INK, lw=2, zorder=4)
        value = icc.get(column, "nan")
        icc_text = "n/a" if value == "nan" else f"{float(value):.2f}"
        ax.set_title(f"{name}\nICC(1) = {icc_text}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel(unit, fontsize=8, color=INK_SECONDARY)
        ax.margins(x=0.08)

    axes[0].set_yticks(range(len(agents)), agents)
    axes[0].set_ylim(len(agents) - 0.5, -0.5)
    for i, ax in enumerate(axes):
        if i % cols == 0:
            ax.tick_params(axis="y", labelsize=8, colors=INK)

    key = axes[len(metrics)]
    key.axis("off")
    handles = [
        Line2D([], [], marker="o", ls="none", mfc=INK_SECONDARY, mec=SURFACE, ms=6, label="seed"),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            mfc=SURFACE,
            mec=INK_SECONDARY,
            mew=1.4,
            ms=6,
            label="seed, censored\n(quit criterion not met)",
        ),
        Line2D([], [], color=INK, lw=2, marker="|", ms=12, ls="none", mew=2, label="mean of seeds"),
    ]
    key.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        fontsize=8,
        title="Key",
        title_fontproperties={"size": 9},
    )
    key.text(
        0.0,
        0.08,
        "ICC(1): share of variance that lies\nbetween agents rather than across seeds.",
        transform=key.transAxes,
        fontsize=7.5,
        color=INK_SECONDARY,
        va="bottom",
    )
    for ax in axes[len(metrics) + 1 :]:
        ax.axis("off")

    fig.suptitle(
        f"{title}: behavioural metrics by trait profile ({len(seeds)} seeds each)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, path)


def plot_series(
    series: dict[str, list[dict[int, float]]],
    phase_of_step: dict[int, int],
    phases: list[tuple[int, int, str]],
    agents: list[str],
    styles: dict[str, tuple[str, str]],
    *,
    ylabel: str,
    title: str,
    path: Path,
    window: int,
    show_seeds: bool,
) -> None:
    rows, cols = grid_shape(len(agents))
    fig, axes = new_figure(rows, cols, 6.2 * cols, 2.0 * rows + 0.8, sharex=True, sharey=True)
    means = {}
    for a in agents:
        steps, values = seed_mean(series[a])
        values = smooth_within_phases(steps, values, [phase_of_step[s] for s in steps], window)
        means[a] = (steps, values)

    for i, (ax, a) in enumerate(zip(axes, agents, strict=False)):
        style_axes(ax, "y")
        colour, ls = styles[a]
        for other in agents:
            if other != a:
                ax.plot(*means[other], color=CONTEXT, lw=1.0, zorder=1)
        if show_seeds:
            for one in series[a]:
                steps = sorted(one)
                ax.plot(steps, [one[s] for s in steps], color=colour, lw=0.7, alpha=0.45, ls=ls)
        ax.plot(*means[a], color=colour, lw=2.0, ls=ls, zorder=3)
        ax.set_title(a, fontsize=9, color=INK, loc="left", pad=12)
        mark_phases(ax, phases, label=True)
        if i % cols == 0:
            ax.set_ylabel(ylabel, fontsize=8, color=INK_SECONDARY)
        ax.tick_params(labelbottom=True)
        if i + cols >= len(agents):  # the lowest panel of its column
            ax.set_xlabel("world step", fontsize=8, color=INK_SECONDARY)
        ax.set_xlim(phases[0][0] - 0.5, phases[-1][1] + 0.5)
        ax.set_ylim(bottom=0)
    for ax in axes[len(agents) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, path)


# --- entry point ----------------------------------------------------------------------------------


def make_plots(directory: Path, window: int = 9) -> list[Path]:
    if not directory.is_dir():
        raise PlotError(f"{directory} is not a directory")
    results = read_table(directory, "results.csv")
    trace = read_table(directory, "pe_trace.csv")
    stability = read_table(directory, "stability.csv")
    for column in ("active_press", "unsigned", "phase_index", "phase_kind"):
        if column not in trace[0]:
            raise PlotError(f"{directory / 'pe_trace.csv'} has no {column} column")

    agents = agent_order(results)
    styles = agent_style(agents)
    phases = phase_starts(trace)
    phase_of_step = {int(r["step"]): int(r["phase_index"]) for r in trace}
    runs = by_run(trace)
    n_seeds = len({r["seed"] for r in results})
    name = directory.name
    out = directory / "plots"
    out.mkdir(exist_ok=True)

    pe = {
        a: [{int(r["step"]): float(r["unsigned"]) for r in rows} for rows in runs[a].values()]
        for a in agents
    }
    cumulative: dict[str, list[dict[int, float]]] = {}
    for a in agents:
        cumulative[a] = []
        for rows in runs[a].values():
            total, one = 0, {}
            for r in rows:
                total += r["active_press"] == "True"
                one[int(r["step"])] = float(total)
            cumulative[a].append(one)

    paths = [out / "metrics_by_agent.png", out / "pe_trace.png", out / "cumulative_record.png"]
    plot_metrics(results, stability, agents, styles, name, paths[0])
    smoothing = (
        f"centred {window}-step moving average within each phase" if window > 1 else "unsmoothed"
    )
    plot_series(
        pe,
        phase_of_step,
        phases,
        agents,
        styles,
        ylabel="unsigned PE\n(confidence-weighted)",
        title=f"{name}: unsigned prediction error, mean of {n_seeds} seeds ({smoothing});"
        " grey = other agents",
        path=paths[1],
        window=window,
        show_seeds=False,
    )
    plot_series(
        cumulative,
        phase_of_step,
        phases,
        agents,
        styles,
        ylabel="cumulative\nactive presses",
        title=f"{name}: cumulative record (thin = each of {n_seeds} seeds, thick = mean;"
        " grey = other agents' means)",
        path=paths[2],
        window=1,
        show_seeds=True,
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment_dir", type=Path, help="DIR/<experiment name>/ with the CSVs")
    parser.add_argument(
        "--smooth", type=int, default=9, help="PE moving-average window in steps (1 = none)"
    )
    args = parser.parse_args(argv)
    try:
        paths = make_plots(args.experiment_dir, max(args.smooth, 1))
    except PlotError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
