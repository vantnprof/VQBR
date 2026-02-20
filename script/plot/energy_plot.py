from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9})


METRIC_ORDER = (
    "cosine_similarity",
    "relative_l2_distance",
    "train_mse",
    "test_mse",
)
METRIC_LABELS = {
    "cosine_similarity": r"Cosine Similarity $\uparrow$",
    "relative_l2_distance": r"Relative L2 Distance $\downarrow$",
    "train_mse": r"Train MSE $\downarrow$",
    "test_mse": r"Test MSE $\downarrow$",
}


def _format_mean_std(mean: float, std: float, precision: int) -> str:
    return f"{mean:.{precision}f}$\\pm${std:.{precision}f}"


def _normalize_su2_text(raw: str) -> str:
    return ",".join([tok.strip().lower() for tok in str(raw).split(",") if tok.strip()])


def _load_summary(summary_csv_path: Path) -> pd.DataFrame:
    if not summary_csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv_path}")
    df = pd.read_csv(summary_csv_path)
    required_cols = {
        "prior_case",
        "config_id",
        "shots",
        "maxiter",
        "reps",
        "su2_gates",
        "entanglement",
        "loss",
        "test_mse_mean",
        "test_mse_std",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(
            f"Summary CSV is missing columns: {missing}. "
            "Re-run energy_experiment.py to regenerate outputs."
        )
    return df


def _print_config_list(df: pd.DataFrame, prior_case: str, limit: int) -> None:
    subset = df[df["prior_case"] == prior_case].copy()
    if subset.empty:
        raise ValueError(f"No summary rows found for prior_case={prior_case!r}.")
    if "rank_by_test_mse" in subset.columns:
        subset = subset.sort_values(by=["rank_by_test_mse", "config_id"], kind="stable")
    else:
        subset = subset.sort_values(by=["test_mse_mean", "config_id"], kind="stable")
    cols = [
        "config_id",
        "shots",
        "maxiter",
        "reps",
        "su2_gates",
        "entanglement",
        "loss",
        "test_mse_mean",
        "test_mse_std",
    ]
    if "rank_by_test_mse" in subset.columns:
        cols = ["rank_by_test_mse"] + cols
    to_show = subset[cols].head(int(limit))
    print(to_show.to_string(index=False))


def _select_configuration(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    subset = df[df["prior_case"] == args.prior_case].copy()
    if subset.empty:
        raise ValueError(f"No rows found for prior_case={args.prior_case!r}.")

    explicit_filters: List[str] = []
    if args.config_id is not None:
        subset = subset[subset["config_id"] == args.config_id]
        explicit_filters.append("config_id")
    if args.shots is not None:
        subset = subset[subset["shots"] == int(args.shots)]
        explicit_filters.append("shots")
    if args.maxiter is not None:
        subset = subset[subset["maxiter"] == int(args.maxiter)]
        explicit_filters.append("maxiter")
    if args.reps is not None:
        subset = subset[subset["reps"] == int(args.reps)]
        explicit_filters.append("reps")
    if args.su2_gates is not None:
        target_su2 = _normalize_su2_text(args.su2_gates)
        subset = subset[subset["su2_gates"].map(_normalize_su2_text) == target_su2]
        explicit_filters.append("su2_gates")
    if args.entanglement is not None:
        subset = subset[subset["entanglement"].astype(str) == str(args.entanglement)]
        explicit_filters.append("entanglement")
    if args.loss is not None:
        subset = subset[subset["loss"].astype(str) == str(args.loss)]
        explicit_filters.append("loss")

    if subset.empty:
        raise ValueError("No configuration matched the provided selection filters.")

    subset = subset.sort_values(by=["test_mse_mean", "config_id"], kind="stable").reset_index(drop=True)
    if len(subset) > 1 and explicit_filters:
        sample_ids = ", ".join(subset["config_id"].head(10).astype(str).tolist())
        raise ValueError(
            "Selection is ambiguous. Provide --config-id or refine filters. "
            f"First matches: {sample_ids}"
        )
    return subset.iloc[0]


def _build_latex_table(selected_row: pd.Series, precision: int) -> str:
    prior_label = str(selected_row["prior_case"])
    config_label = (
        f"{selected_row['config_id']} "
        f"(shots={int(selected_row['shots'])}, "
        f"maxiter={int(selected_row['maxiter'])}, reps={int(selected_row['reps'])}, "
        f"su2={selected_row['su2_gates']}, ent={selected_row['entanglement']}, "
        f"loss={selected_row['loss']})"
    )
    config_label_escaped = config_label.replace("_", "\\_")

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Energy dataset results (mean $\pm$ std over random seeds) "
        + "for selected configuration: "
        + config_label_escaped
        + ".}"
    )
    lines.append(r"\begin{tabular}{ll" + ("c" * len(METRIC_ORDER)) + r"}")
    lines.append(r"\toprule")
    lines.append("Prior & Method & " + " & ".join(METRIC_LABELS[m] for m in METRIC_ORDER) + r" \\")
    lines.append(r"\midrule")

    closed_train_mean = float(selected_row.get("closed_form_train_mse_mean", np.nan))
    closed_train_std = float(selected_row.get("closed_form_train_mse_std", np.nan))
    closed_test_mean = float(selected_row.get("closed_form_test_mse_mean", np.nan))
    closed_test_std = float(selected_row.get("closed_form_test_mse_std", np.nan))
    closed_row = [
        prior_label,
        "Closed-form",
        "--",
        "--",
        _format_mean_std(closed_train_mean, closed_train_std, precision),
        _format_mean_std(closed_test_mean, closed_test_std, precision),
    ]
    lines.append(" & ".join(closed_row) + r" \\")

    vqbr_row = ["", "VQBR (COBYLA)"]
    for metric in METRIC_ORDER:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        vqbr_row.append(
            _format_mean_std(
                float(selected_row.get(mean_key, np.nan)),
                float(selected_row.get(std_key, np.nan)),
                precision,
            )
        )
    lines.append(" & ".join(vqbr_row) + r" \\")
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


def _parse_history_cell(raw: Any) -> List[float]:
    if isinstance(raw, str):
        token = raw.strip()
        if token == "":
            return []
        try:
            parsed = json.loads(token)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    return []


def _load_histories(
    training_csv_path: Path,
    prior_case: str,
    config_id: str,
) -> List[List[float]]:
    if not training_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {training_csv_path}")
    df = pd.read_csv(training_csv_path)
    required_cols = {"prior_case", "config_id", "batch_loss_history_json"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(
            f"Training CSV is missing columns: {missing}. "
            "Re-run energy_experiment.py to regenerate outputs."
        )
    rows = df[(df["prior_case"] == prior_case) & (df["config_id"] == config_id)]
    if rows.empty:
        raise ValueError(
            f"No training rows found for prior_case={prior_case!r}, config_id={config_id!r}."
        )
    histories: List[List[float]] = []
    for raw in rows["batch_loss_history_json"].tolist():
        h = _parse_history_cell(raw)
        if h:
            histories.append(h)
    if not histories:
        raise ValueError("No non-empty batch_loss_history_json values found for selected configuration.")
    return histories


def _plot_vqbr_convergence(
    *,
    histories: List[List[float]],
    output_path: Path,
    title: str,
    width: float,
    height: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    color = "#d62728"

    for history in histories:
        x = np.arange(1, len(history) + 1)
        ax.plot(x, history, color=color, alpha=0.14, linewidth=0.7)

    hist_arr = _stack_histories(histories)
    mean_curve = np.nanmean(hist_arr, axis=0)
    std_curve = np.nanstd(hist_arr, axis=0, ddof=0)
    x = np.arange(1, mean_curve.size + 1)
    ax.plot(x, mean_curve, color=color, linewidth=1.6, label="VQBR (COBYLA)")
    if len(histories) > 1:
        ax.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )

    max_iter = int(mean_curve.size)
    if max_iter <= 10:
        xticks = np.arange(1, max_iter + 1, dtype=float)
    elif max_iter <= 50:
        xticks = np.array(sorted(set([1, 10, 20, 30, 40, max_iter])), dtype=float)
    else:
        xticks = np.array([1, max_iter // 4, max_iter // 2, (3 * max_iter) // 4, max_iter], dtype=float)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(v)) for v in xticks])

    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel(r"$\widehat{\widetilde{\mathcal{L}}}(\boldsymbol{\theta})$", fontsize=9)
    ax.set_title(title, fontsize=8)
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
            "Load energy grid-search CSV outputs, select one configuration, "
            "render a LaTeX table row set, and plot convergence."
        )
    )
    parser.add_argument(
        "--summary-csv-path",
        type=str,
        default="results/energy/energy_grid_summary.csv",
    )
    parser.add_argument(
        "--training-csv-path",
        type=str,
        default="results/energy/energy_grid_training_info.csv",
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
    parser.add_argument("--prior-case", type=str, default="heteroscedastic")
    parser.add_argument("--config-id", type=str, default=None)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument(
        "--su2-gates",
        type=str,
        default=None,
        help="Gate set string (e.g. 'ry' or 'rx,y').",
    )
    parser.add_argument("--entanglement", type=str, default=None)
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configurations from summary CSV and exit.",
    )
    parser.add_argument("--list-limit", type=int, default=20)
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--fig-width", type=float, default=3.0)
    parser.add_argument("--fig-height", type=float, default=2.5)
    parser.add_argument("--dpi", type=int, default=1000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summary_csv_path = Path(args.summary_csv_path).resolve()
    training_csv_path = Path(args.training_csv_path).resolve()
    table_path = Path(args.table_path).resolve()
    convergence_plot_path = Path(args.convergence_plot_path).resolve()

    summary_df = _load_summary(summary_csv_path)
    if args.list_configs:
        _print_config_list(summary_df, prior_case=args.prior_case, limit=args.list_limit)
        return

    selected_row = _select_configuration(summary_df, args=args)
    selected_config_id = str(selected_row["config_id"])
    selected_prior = str(selected_row["prior_case"])
    print(
        "Selected configuration: "
        f"{selected_config_id} "
        f"(shots={int(selected_row['shots'])}, "
        f"maxiter={int(selected_row['maxiter'])}, "
        f"reps={int(selected_row['reps'])}, "
        f"su2_gates={selected_row['su2_gates']}, "
        f"entanglement={selected_row['entanglement']}, "
        f"loss={selected_row['loss']})"
    )

    latex_table = _build_latex_table(selected_row=selected_row, precision=args.precision)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8") as f:
        f.write(latex_table)
    print(latex_table.strip())
    print("")
    print(f"LaTeX table saved to: {table_path}")

    histories = _load_histories(
        training_csv_path=training_csv_path,
        prior_case=selected_prior,
        config_id=selected_config_id,
    )
    _plot_vqbr_convergence(
        histories=histories,
        output_path=convergence_plot_path,
        title=f"{selected_config_id} | mean over seeds",
        width=args.fig_width,
        height=args.fig_height,
        dpi=args.dpi,
    )
    print(f"Convergence plot saved to: {convergence_plot_path}")


if __name__ == "__main__":
    main()
