"""Reproduce slide 15 histograms from all 20 final test RMSEs per method."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[2]
SOURCE = ROOT / (
    "results/synthetic/"
    "synthetic_N200_D8_seeds20_maxiter300_cobyla_log_ratio_analytic_"
    "reps2_gatesry-x_entlinear_20260318_085932/per_seed_metrics.csv"
)
# Shared equal-width bins and count limits keep the figures comparable.
# Both use the same marked axis break across bins with zero observations.
# Matching panel geometry also gives every bin the same displayed width.
EDGES = np.arange(8, 49, dtype=float) / 100
FIGURE_SIZE = (9, 4.8)
EXPORT_DPI = 300
METHODS = [
    ("closed_form", "Direct MAP", "#2855F4", "fig1_direct_map_rmse_histogram"),
    ("vqbr", "VQBR", "#D13E06", "fig2_vqbr_rmse_histogram"),
]


def main():
    with SOURCE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    values_by_method = {}
    for method, _, _, _ in METHODS:
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["seed"]),
        )
        assert [int(row["seed"]) for row in selected] == list(range(20))
        values = np.array([float(row["test_rmse"]) for row in selected])
        assert np.isfinite(values).all()
        assert np.histogram(values, bins=EDGES)[0].sum() == 20
        values_by_method[method] = values
    assert np.allclose(np.diff(EDGES), 0.01)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 20,
        "text.color": "#062449", "axes.labelcolor": "#062449",
        "xtick.color": "#062449", "ytick.color": "#062449",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    maximum_count = max(
        np.histogram(values, bins=EDGES)[0].max()
        for values in values_by_method.values()
    )
    bin_rows = []
    for method, label, color, filename in METHODS:
        fig = plt.figure(figsize=FIGURE_SIZE)
        fig.subplots_adjust(left=0.105, right=0.95, bottom=0.23, top=0.95)
        # Identical panels align the bin widths and axis break side by side.
        grid = fig.add_gridspec(1, 2, width_ratios=[6, 2], wspace=0.38)
        left = fig.add_subplot(grid[0])
        right = fig.add_subplot(grid[1], sharey=left)
        axes = [left, right]
        limits = [(0.08, 0.14), (0.45, 0.47)]
        ticks = [np.arange(8, 15) / 100, np.arange(45, 48) / 100]
        assert not ((values_by_method[method] >= 0.14)
                    & (values_by_method[method] < 0.45)).any()

        counts, _ = np.histogram(values_by_method[method], bins=EDGES)
        for ax, (low, high), tick_values in zip(axes, limits, ticks):
            ax.hist(values_by_method[method], bins=EDGES,
                    color=color, edgecolor="white", linewidth=1.8)
            for center, count in zip((EDGES[:-1] + EDGES[1:]) / 2, counts):
                if count and low < center < high:
                    ax.text(center, count + 0.15, str(int(count)),
                            ha="center", va="bottom", fontsize=26,
                            fontweight="bold")
            ax.set(xlim=(low, high), ylim=(0, maximum_count + 1))
            ax.set_xticks(tick_values)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.tick_params(axis="both", labelsize=20, pad=9, length=5,
                           width=1.2)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#DFE4EB", linewidth=1.0)
            ax.spines[["top", "right"]].set_visible(False)
            for spine in ("bottom", "left"):
                ax.spines[spine].set_color("#718096")
                ax.spines[spine].set_linewidth(1.2)
        axes[0].set_ylabel("Count", fontsize=24, labelpad=12)
        right.spines["left"].set_visible(False)
        right.tick_params(axis="y", left=False, labelleft=False)
        # Slash marks explicitly identify the discontinuous x-axis.
        for ax, x in ((left, 1), (right, 0)):
            ax.plot([x, x], [0, 0], transform=ax.transAxes,
                    marker=[(-1, -1), (1, 1)], markersize=14,
                    linestyle="none", color="#062449", mew=1.8,
                    clip_on=False)
        fig.text(0.5275, 0.035, "Test RMSE", ha="center", fontsize=24)
        for extension in ("png", "pdf", "svg"):
            output_path = OUTPUT / f"{filename}.{extension}"
            fig.savefig(output_path, dpi=EXPORT_DPI,
                        facecolor="white", bbox_inches=None)
            if extension == "svg":
                output_path.write_text(
                    "\n".join(line.rstrip() for line in output_path.read_text().splitlines()) + "\n"
                )
        plt.close(fig)
        bin_rows.extend((label, left, right, int(count)) for left, right, count
                        in zip(EDGES[:-1], EDGES[1:], counts))
        print(f"{label}: {int(counts.sum())} observations; "
              f"range {values_by_method[method].min():.6f}–"
              f"{values_by_method[method].max():.6f}")

    with (OUTPUT / "histogram_bin_counts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["method", "bin_left", "bin_right", "count"])
        writer.writerows(bin_rows)
    with (OUTPUT / "test_rmse_all_seeds.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["seed", "direct_map_test_rmse", "vqbr_test_rmse"])
        writer.writerows(zip(range(20), values_by_method["closed_form"],
                             values_by_method["vqbr"]))


if __name__ == "__main__":
    main()
