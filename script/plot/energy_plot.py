from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9})


METRIC_ORDER = (
    "cosine_similarity",
    "euclidean_distance",
    "train_mse",
    "test_mse",
)
METHOD_ORDER = ("vqbr_cobyla", "vqbr_spsa", "vqbr")
METHOD_LABELS = {
    "vqbr_cobyla": "VQBR (COBYLA)",
    "vqbr_spsa": "VQBR (SPSA)",
    "vqbr": "VQBR",
}
PRIOR_ORDER = ("heteroscedastic",)
PRIOR_LABELS = {
    "heteroscedastic": "Heteroscedastic",
}
METRIC_LABELS = {
    "cosine_similarity": r"Cosine Similarity $\uparrow$",
    "euclidean_distance": r"Euclidean Distance $\downarrow$",
    "train_mse": r"Train MSE $\downarrow$",
    "test_mse": r"Test MSE $\downarrow$",
}


def _format_mean_std(mean: float, std: float, precision: int) -> str:
    return f"{mean:.{precision}f}$\\pm${std:.{precision}f}"


def _build_latex_table(data: Dict[str, Any], precision: int) -> str:
    prior_cases = data.get("prior_cases", {})
    prior_block = prior_cases.get("heteroscedastic")
    if prior_block is None:
        raise ValueError("Missing heteroscedastic results under prior_cases['heteroscedastic'].")
    aggregate = prior_block.get("aggregate", {})

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Energy dataset results over random seeds (mean $\pm$ std).}"
    )
    lines.append(r"\begin{tabular}{ll" + ("c" * len(METRIC_ORDER)) + r"}")
    lines.append(r"\toprule")
    lines.append(
        "Prior & Method & "
        + " & ".join(METRIC_LABELS[m] for m in METRIC_ORDER)
        + r" \\"
    )
    lines.append(r"\midrule")

    for prior in PRIOR_ORDER:
        prior_label_printed = False
        for method in METHOD_ORDER:
            if method not in aggregate:
                continue
            cells = [
                PRIOR_LABELS.get(prior, prior) if not prior_label_printed else "",
                METHOD_LABELS.get(method, method),
            ]
            for metric in METRIC_ORDER:
                stats = aggregate[method].get(metric)
                if stats is None:
                    cells.append("--")
                    continue
                cells.append(
                    _format_mean_std(
                        mean=float(stats["mean"]),
                        std=float(stats["std"]),
                        precision=precision,
                    )
                )
            lines.append(" & ".join(cells) + r" \\")
            prior_label_printed = True

    if lines[-1] == r"\midrule":
        raise ValueError("No aggregate VQBR metrics were found in the result file.")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _stack_histories(histories: List[List[float]]) -> np.ndarray:
    if not histories:
        return np.empty((0, 0), dtype=float)
    max_len = max(len(h) for h in histories)
    arr = np.full((len(histories), max_len), np.nan, dtype=float)
    for i, h in enumerate(histories):
        if not h:
            continue
        vec = np.asarray(h, dtype=float)
        arr[i, : vec.size] = vec
    return arr


def _extract_batch_loss_history(vqbr_info: Dict[str, Any]) -> List[float]:
    for key in ("batch_loss_history", "loss_history"):
        values = vqbr_info.get(key)
        if isinstance(values, list) and values:
            return [float(x) for x in values]
    return []


def _plot_vqbr_convergence(
    data: Dict[str, Any],
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
) -> None:
    prior_cases = data.get("prior_cases", {})
    prior_block = prior_cases.get("heteroscedastic")
    if prior_block is None:
        raise ValueError("Missing heteroscedastic results under prior_cases['heteroscedastic'].")

    per_seed = prior_block.get("per_seed", [])
    method_histories: Dict[str, List[List[float]]] = {}
    for method in METHOD_ORDER:
        histories: List[List[float]] = []
        for seed_result in per_seed:
            vqbr_info = seed_result.get("methods", {}).get(method)
            if not isinstance(vqbr_info, dict):
                continue
            history = _extract_batch_loss_history(vqbr_info)
            if history:
                histories.append(history)
        if histories:
            method_histories[method] = histories

    if not method_histories:
        raise ValueError(
            "No per-iteration VQBR batch loss history found. Expected one of "
            "methods.vqbr_cobyla.batch_loss_history or methods.vqbr_spsa.batch_loss_history."
        )

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    method_colors = {
        "vqbr_cobyla": "#d62728",
        "vqbr_spsa": "#1f77b4",
        "vqbr": "#d62728",
    }
    max_iter = 1
    for method in METHOD_ORDER:
        histories = method_histories.get(method, [])
        if not histories:
            continue
        color = method_colors.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)
        for history in histories:
            x = np.arange(1, len(history) + 1)
            ax.plot(x, history, color=color, alpha=0.16, linewidth=0.7)

        hist_arr = _stack_histories(histories)
        mean_curve = np.nanmean(hist_arr, axis=0)
        std_curve = np.nanstd(hist_arr, axis=0, ddof=0)
        x = np.arange(1, mean_curve.size + 1)
        ax.plot(x, mean_curve, color=color, linewidth=1.6, label=label)
        if len(histories) > 1:
            ax.fill_between(
                x,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.16,
                color=color,
                linewidth=0.0,
            )
        max_iter = max(max_iter, int(mean_curve.size))

    if max_iter == 200:
        xticks = np.array([1, 50, 100, 150, 200], dtype=float)
        xlabels = ["1", "50", "100", "150", "200"]
    else:
        xticks = np.linspace(1.0, float(max_iter), num=5)
        xlabels = [str(int(round(v))) for v in xticks]
        xlabels[0] = "1"
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Batch Loss", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load heteroscedastic-prior energy experiment results and produce a LaTeX table "
            "plus VQBR convergence plot."
        )
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results/energy/energy_experiment_results.json",
    )
    parser.add_argument(
        "--table-path",
        type=str,
        default="results/energy/energy_results_table.tex",
    )
    parser.add_argument(
        "--convergence-plot-path",
        type=str,
        default="results/energy/energy_vqbr_convergence.png",
    )
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--fig-width", type=float, default=3.0)
    parser.add_argument("--fig-height", type=float, default=2.5)
    parser.add_argument("--dpi", type=int, default=1000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results_path = Path(args.results_path).resolve()
    table_path = Path(args.table_path).resolve()
    convergence_plot_path = Path(args.convergence_plot_path).resolve()

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    latex_table = _build_latex_table(data, precision=args.precision)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8") as f:
        f.write(latex_table)
    print(latex_table.strip())
    print("")
    print(f"LaTeX table saved to: {table_path}")

    _plot_vqbr_convergence(
        data=data,
        output_path=convergence_plot_path,
        width=args.fig_width,
        height=args.fig_height,
        dpi=args.dpi,
    )
    print(f"Convergence plot saved to: {convergence_plot_path}")


if __name__ == "__main__":
    main()
