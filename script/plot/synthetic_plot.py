from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9})


METRIC_ORDER = (
    "cosine_similarity",
    "relative_l2_distance",
    "euclidean_distance",
    "train_mse",
    "test_mse",
)
# Keep plotting focused on VQBR methods; table can include additional baselines.
METHOD_ORDER = ("vqbr_cobyla", "vqbr_spsa", "vqbr")
TABLE_METHOD_ORDER = ("closed_form",) + METHOD_ORDER
METHOD_LABELS = {
    "vqbr_cobyla": "VQBR (COBYLA)",
    "vqbr_spsa": "VQBR (SPSA)",
    "vqbr": "VQBR",
}
TABLE_METHOD_LABELS = {
    "closed_form": "Closed-form",
    "vqbr_cobyla": "VQBR (COBYLA, reconstructed)",
    "vqbr_spsa": "VQBR (SPSA, reconstructed)",
    "vqbr": "VQBR (reconstructed)",
}
CONVERGENCE_METHOD = "vqbr_cobyla"
CONVERGENCE_PAIR_ORDER = ("N16_D8", "N40_D8", "N80_D8")
CONVERGENCE_PAIR_STYLES = {
    "N16_D8": {"color": "#d62728", "marker": "o", "label": "N=16, D=8"},
    "N40_D8": {"color": "#1f77b4", "marker": "s", "label": "N=40, D=8"},
    "N80_D8": {"color": "#2ca02c", "marker": "^", "label": "N=80, D=8"},
}
PRIOR_KEY = "heteroscedastic"
METRIC_LABELS = {
    "cosine_similarity": r"Cosine Similarity $\uparrow$",
    "relative_l2_distance": r"Relative L2 Distance $\downarrow$",
    "euclidean_distance": r"Euclidean Distance $\downarrow$",
    "train_mse": r"Train MSE $\downarrow$",
    "test_mse": r"Test MSE $\downarrow$",
}


def _format_mean_std(mean: float, std: float, precision: int) -> str:
    return f"{mean:.{precision}f}$\\pm${std:.{precision}f}"


def _pair_sort_key(pair_name: str, pair_meta: Dict[str, Any]) -> Tuple[int, int, str]:
    N = pair_meta.get("N")
    D = pair_meta.get("D")
    if isinstance(N, (int, float)) and isinstance(D, (int, float)):
        return int(D), int(N), pair_name
    return (10**9, 10**9, pair_name)


def _iter_pair_blocks(data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    pair_results = data.get("pair_results", {})
    if not isinstance(pair_results, dict) or not pair_results:
        raise ValueError("Missing non-empty 'pair_results' in synthetic result JSON.")

    items: List[Tuple[str, Dict[str, Any]]] = []
    for pair_name, pair_block in pair_results.items():
        if isinstance(pair_name, str) and isinstance(pair_block, dict):
            items.append((pair_name, pair_block))

    items.sort(key=lambda x: _pair_sort_key(x[0], x[1].get("pair", {})))
    return items


def _build_latex_table(data: Dict[str, Any], precision: int) -> str:
    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Synthetic dataset sweep results over random seeds (mean $\pm$ std).}"
    )
    lines.append(r"\begin{tabular}{lll" + ("c" * len(METRIC_ORDER)) + r"}")
    lines.append(r"\toprule")
    lines.append(
        "Pair & $(N, D)$ & Method & "
        + " & ".join(METRIC_LABELS[m] for m in METRIC_ORDER)
        + r" \\")
    lines.append(r"\midrule")

    wrote_any = False
    for pair_name, pair_block in _iter_pair_blocks(data):
        pair_meta = pair_block.get("pair", {})
        prior_cases = pair_block.get("prior_cases", {})
        prior_block = prior_cases.get(PRIOR_KEY)
        if not isinstance(prior_block, dict):
            continue
        aggregate = prior_block.get("aggregate", {})
        if not isinstance(aggregate, dict):
            continue

        N = pair_meta.get("N", "?")
        D = pair_meta.get("D", "?")
        nd_text = f"({N}, {D})"

        pair_label_printed = False
        for method in TABLE_METHOD_ORDER:
            if method not in aggregate:
                continue
            method_metrics = aggregate.get(method, {})
            row = [
                pair_name if not pair_label_printed else "",
                nd_text if not pair_label_printed else "",
                TABLE_METHOD_LABELS.get(method, METHOD_LABELS.get(method, method)),
            ]
            for metric in METRIC_ORDER:
                stats = method_metrics.get(metric)
                if not isinstance(stats, dict):
                    row.append("--")
                    continue
                row.append(
                    _format_mean_std(
                        mean=float(stats.get("mean", np.nan)),
                        std=float(stats.get("std", np.nan)),
                        precision=precision,
                    )
                )
            lines.append(" & ".join(row) + r" \\")
            pair_label_printed = True
            wrote_any = True

    if not wrote_any:
        raise ValueError("No aggregate metrics found under pair_results.*.prior_cases.")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
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
    selected_pairs = set(CONVERGENCE_PAIR_ORDER)
    pair_histories: Dict[str, List[List[float]]] = {pair: [] for pair in CONVERGENCE_PAIR_ORDER}

    for pair_name, pair_block in _iter_pair_blocks(data):
        if pair_name not in selected_pairs:
            continue
        prior_cases = pair_block.get("prior_cases", {})
        prior_block = prior_cases.get(PRIOR_KEY)
        if not isinstance(prior_block, dict):
            continue
        per_seed = prior_block.get("per_seed", [])
        if not isinstance(per_seed, list):
            continue

        for seed_result in per_seed:
            if not isinstance(seed_result, dict):
                continue
            vqbr_info = seed_result.get("methods", {}).get(CONVERGENCE_METHOD)
            if not isinstance(vqbr_info, dict):
                continue
            history = _extract_batch_loss_history(vqbr_info)
            if history:
                pair_histories[pair_name].append(history)

    missing_pairs = [pair for pair in CONVERGENCE_PAIR_ORDER if not pair_histories.get(pair)]
    if missing_pairs:
        raise ValueError(
            "Missing convergence histories for pairs: "
            + ", ".join(missing_pairs)
            + f" using method={CONVERGENCE_METHOD}."
        )

    if not any(pair_histories.values()):
        raise ValueError(
            "No per-iteration VQBR batch loss history found in pair_results.*.prior_cases.*.per_seed."
        )

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    max_iter = 1
    for pair_name in CONVERGENCE_PAIR_ORDER:
        histories = pair_histories[pair_name]
        hist_arr = _stack_histories(histories)
        mean_curve = np.nanmean(hist_arr, axis=0)
        std_curve = np.nanstd(hist_arr, axis=0, ddof=0)
        x = np.arange(1, mean_curve.size + 1)
        style = CONVERGENCE_PAIR_STYLES[pair_name]
        mark_every = max(1, int(round(mean_curve.size / 8)))
        ax.plot(
            x,
            mean_curve,
            color=style["color"],
            marker=style["marker"],
            markevery=mark_every,
            markersize=3.5,
            linewidth=1.7,
            label=style["label"],
        )
        if len(histories) > 1:
            ax.fill_between(
                x,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.12,
                color=style["color"],
                linewidth=0.0,
            )
        max_iter = max(max_iter, int(mean_curve.size))

    if max_iter <= 5:
        xticks = np.arange(1, max_iter + 1)
    elif max_iter <= 20:
        xticks = np.array(sorted(set([1, 5, 10, 15, max_iter])), dtype=float)
    else:
        candidates = [1, 50, 100, 150, max_iter]
        xticks = np.array(
            sorted({int(v) for v in candidates if 1 <= int(v) <= max_iter}),
            dtype=float,
        )
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(round(v))) for v in xticks])

    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel(r"$\widehat{\widetilde{\mathcal{L}}}(\boldsymbol{\theta})$", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_test_mse_sweep(
    data: Dict[str, Any],
    output_path: Path,
    width: float,
    height: float,
    dpi: int,
) -> None:
    # Curves by (method, sample multiplier m where N = m*D), x-axis=D, y-axis=test MSE.
    points: Dict[Tuple[str, int], List[Tuple[int, float, float]]] = {}

    for pair_name, pair_block in _iter_pair_blocks(data):
        pair_meta = pair_block.get("pair", {})
        D_raw = pair_meta.get("D")
        N_raw = pair_meta.get("N")
        if not isinstance(D_raw, (int, float)) or not isinstance(N_raw, (int, float)):
            continue
        D = int(D_raw)
        N = int(N_raw)
        if D <= 0:
            continue
        multiplier = int(round(N / D))

        prior_cases = pair_block.get("prior_cases", {})
        prior_block = prior_cases.get(PRIOR_KEY)
        if not isinstance(prior_block, dict):
            continue
        aggregate = prior_block.get("aggregate", {})
        if not isinstance(aggregate, dict):
            continue

        for method in METHOD_ORDER:
            method_metrics = aggregate.get(method)
            if not isinstance(method_metrics, dict):
                continue
            stats = method_metrics.get("test_mse")
            if not isinstance(stats, dict):
                continue
            mean = float(stats.get("mean", np.nan))
            std = float(stats.get("std", np.nan))
            if not np.isfinite(mean):
                continue
            points.setdefault((method, multiplier), []).append((D, mean, std))

    if not points:
        raise ValueError("No test_mse aggregate points found for synthetic sweep plot.")

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    method_colors = {
        "vqbr_cobyla": "#d62728",
        "vqbr_spsa": "#1f77b4",
        "vqbr": "#d62728",
    }
    style_by_multiplier = {
        2: "-",
        5: "--",
        10: ":",
    }

    for method in METHOD_ORDER:
        for (m_method, multiplier), triples in sorted(points.items(), key=lambda x: (x[0][1], x[0][0])):
            if m_method != method:
                continue
            triples_sorted = sorted(triples, key=lambda t: t[0])
            D = np.array([t[0] for t in triples_sorted], dtype=float)
            mean = np.array([t[1] for t in triples_sorted], dtype=float)
            std = np.array([t[2] for t in triples_sorted], dtype=float)
            color = method_colors.get(method, "#333333")
            linestyle = style_by_multiplier.get(int(multiplier), "-")
            label = f"{METHOD_LABELS.get(method, method)}, N={multiplier}D"
            ax.plot(D, mean, linestyle=linestyle, marker="o", linewidth=1.4, color=color, label=label)
            ax.fill_between(D, mean - std, mean + std, color=color, alpha=0.12, linewidth=0.0)

    unique_D = sorted({int(d) for curves in points.values() for d, _, _ in curves})
    if unique_D:
        ax.set_xticks(unique_D)
    ax.set_xlabel("Feature dimension D", fontsize=9)
    ax.set_ylabel("Test MSE", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load synthetic experiment results and produce a LaTeX summary table, "
            "a VQBR convergence plot, and a test-MSE sweep plot."
        )
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results/synthetic/synthetic_experiemnt_results.json",
    )
    parser.add_argument(
        "--table-path",
        type=str,
        default="results/synthetic/synthetic_results_table.tex",
    )
    parser.add_argument(
        "--convergence-plot-path",
        type=str,
        default="results/synthetic/synthetic_vqbr_convergence.png",
    )
    parser.add_argument(
        "--sweep-plot-path",
        type=str,
        default="results/synthetic/synthetic_test_mse_sweep.png",
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
    sweep_plot_path = Path(args.sweep_plot_path).resolve()

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

    _plot_test_mse_sweep(
        data=data,
        output_path=sweep_plot_path,
        width=args.fig_width,
        height=args.fig_height,
        dpi=args.dpi,
    )
    print(f"Sweep plot saved to: {sweep_plot_path}")


if __name__ == "__main__":
    main()
