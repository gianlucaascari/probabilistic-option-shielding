"""Plot episodic discounted return and cumulative safety violations.

    python scripts/plot_results.py --map large_map [--logdir logs] [--out figures]

Reads ``<logdir>/<map>/<method>/seed_*/progress.csv`` and draws every figure listed in
``configs/<map>/plots.yaml``: the mean over seeds with a shaded standard
error, smoothed with a centred moving average.
"""
import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PANELS = {
    "Episodic Discounted Return (↑)": "rollout/ep_disc_rew_mean",
    "Cumulative Safety Violations (↓)": "rollout/cum_total_violations",
}
REFERENCE_PANEL = "Episodic Discounted Return (↑)"
N_POINTS, SMOOTH_WINDOW = 200, 5


def load_run(run_dir, tag):
    """(steps, values) of ``tag`` from a run's ``progress.csv``."""
    with open(os.path.join(run_dir, "progress.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r.get(tag)]
    steps = np.array([float(r["time/total_timesteps"]) for r in rows])
    order = np.argsort(steps, kind="stable")
    return steps[order], np.array([float(r[tag]) for r in rows])[order]


def align(runs, tail_tolerance=0.01):
    """Interpolate runs onto a common grid; a run is NaN past its last step.
    Runs ending within ``tail_tolerance`` of the longest count as complete."""
    start = max(steps[0] for steps, _ in runs)
    longest = max(steps[-1] for steps, _ in runs)
    end = min(steps[-1] for steps, _ in runs if steps[-1] >= longest - tail_tolerance * (longest - start))
    grid = np.linspace(start, end, N_POINTS)
    matrix = np.stack([np.where(grid <= s[-1], np.interp(grid, s, v), np.nan) for s, v in runs])
    return grid, matrix


def smooth(values, window=SMOOTH_WINDOW):
    """Centred moving average that ignores NaNs."""
    if window <= 1 or len(values) < window:
        return values
    left = window // 2
    padded = np.pad(values, (left, window - 1 - left), mode="edge")
    valid = np.isfinite(padded)
    sums = np.cumsum(np.insert(np.where(valid, padded, 0.0), 0, 0.0))
    counts = np.cumsum(np.insert(valid.astype(float), 0, 0.0))
    total, n = sums[window:] - sums[:-window], counts[window:] - counts[:-window]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def draw_figure(name, figure, reference, run_root, out_dir):
    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    handles = {}
    for ax, (title, tag) in zip(axes, PANELS.items()):
        for arm, style in figure["arms"].items():
            runs = [load_run(d, tag) for d in sorted(glob.glob(os.path.join(run_root, arm, "seed_*")))]
            if not runs:
                print(f"  [skip] no runs for {arm}")
                continue
            grid, matrix = align(runs)
            mean = smooth(np.nanmean(matrix, axis=0))
            n_alive = np.maximum(np.isfinite(matrix).sum(axis=0), 1)
            sem = smooth(np.nanstd(matrix, axis=0) / np.sqrt(n_alive))
            line, = ax.plot(grid, mean, color=style["color"], linestyle=style["linestyle"], linewidth=2,
                            zorder=style["zorder"], label=style["label"])
            ax.fill_between(grid, mean - sem, mean + sem, color=style["color"], alpha=figure["band_alpha"],
                            zorder=style["zorder"] - 0.5)
            handles[arm] = line
        if title == REFERENCE_PANEL:
            handles["reference"] = ax.axhline(reference["y"], color="#8a8a8a", linestyle=(0, (1.5, 2.5)),
                                              linewidth=2, zorder=6, label=reference["label"])
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Environment Steps")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(14)

    # One legend column per group, padded with invisible entries.
    columns = [[k for k in column if k in handles] for column in figure["legend_columns"]]
    height = max(len(c) for c in columns)
    legend_handles, legend_labels = [], []
    for column in columns:
        for key in column:
            legend_handles.append(handles[key])
            legend_labels.append(handles[key].get_label())
        for _ in range(height - len(column)):
            legend_handles.append(Line2D([], [], linestyle="none"))
            legend_labels.append("")
    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(columns), frameon=False, handlelength=4.0, fontsize=16)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", required=True, choices=["large_map", "stochastic_map"])
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo, "configs", args.map, "plots.yaml")) as f:
        config = yaml.safe_load(f)
    for name, figure in config["figures"].items():
        draw_figure(name, figure, config["reference"], os.path.join(args.logdir, args.map), args.out)


if __name__ == "__main__":
    main()
