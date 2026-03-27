from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({"font.size": 10})

_METHOD_ORDER = (
    "vqbr_sim",
    "vqbr_ibm_no_zne",
    "vqbr_ibm_zne",
)
_METHOD_COLORS = {
    "vqbr_sim": "#d62728",
    "vqbr_ibm_no_zne": "#1f77b4",
    "vqbr_ibm_zne": "#2ca02c",
}
_CONVERGENCE_LEGEND_LABELS = {
    "vqbr_sim": r"$\mathrm{sim}$",
    "vqbr_ibm_no_zne": r"$\mathtt{ibm\_kingston},\ \mathrm{w/o\ ZNE}$",
    "vqbr_ibm_zne": r"$\mathtt{ibm\_kingston},\ \mathrm{w/\ ZNE}$",
}
_BAR_LABELS = {
    "vqbr_sim": r"$\mathrm{sim}$",
    "vqbr_ibm_no_zne": r"$\mathtt{ibm\_kingston}$" "\n" r"$\mathrm{w/o\ ZNE}$",
    "vqbr_ibm_zne": r"$\mathtt{ibm\_kingston}$" "\n" r"$\mathrm{w/\ ZNE}$",
}


def _stack_histories(histories: List[List[float]]) -> np.ndarray:
    if not histories:
        return np.empty((0, 0), dtype=float)
    max_len = max(len(history) for history in histories)
    arr = np.full((len(histories), max_len), np.nan, dtype=float)
    for idx, history in enumerate(histories):
        vec = np.asarray(history, dtype=float).reshape(-1)
        if vec.size > 0:
            arr[idx, : vec.size] = vec
    return arr


def _load_histories_from_training_csv(training_csv_path: Path) -> Dict[str, Dict[str, Any]]:
    if not training_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {training_csv_path}")

    grouped: Dict[str, Dict[str, Any]] = {}
    with training_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        iteration_key = "iteration" if "iteration" in fieldnames else "objective_eval"
        required = {"seed", "method_key", "method_label", "batch_loss", iteration_key}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"Training CSV is missing required columns: {sorted(missing)}. "
                "Re-run ablation_real_hardware.py to regenerate outputs."
            )

        rows_by_method_seed: Dict[str, Dict[int, List[tuple[int, float]]]] = {}
        labels: Dict[str, str] = {}
        for row in reader:
            method_key = str(row["method_key"])
            labels[method_key] = str(row["method_label"])
            seed = int(row["seed"])
            iteration = int(row[iteration_key])
            loss = float(row["batch_loss"])
            rows_by_method_seed.setdefault(method_key, {}).setdefault(seed, []).append(
                (iteration, loss)
            )

    for method_key, per_seed_entries in rows_by_method_seed.items():
        histories: Dict[int, List[float]] = {}
        for seed, entries in per_seed_entries.items():
            ordered = sorted(entries, key=lambda item: item[0])
            histories[seed] = [float(loss) for _, loss in ordered]
        grouped[method_key] = {
            "method_key": str(method_key),
            "method_label": str(labels.get(method_key, method_key)),
            "histories_by_seed": histories,
        }
    return grouped


def _load_histories_from_results_json(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    methods = payload.get("methods", [])
    if not isinstance(methods, list):
        raise ValueError("Expected results JSON field 'methods' to be a list.")

    for method in methods:
        if not isinstance(method, dict):
            continue
        method_key = str(method.get("method_key", ""))
        if not method_key:
            continue
        metrics = method.get("metrics", {})
        history = metrics.get("objective_history", []) if isinstance(metrics, dict) else []
        if not isinstance(history, list):
            history = []
        grouped[method_key] = {
            "method_key": method_key,
            "method_label": str(method.get("method_label", method_key)),
            "histories_by_seed": {int(method.get("seed", 0)): [float(x) for x in history]},
        }
    return grouped


def _choose_histories(
    *,
    payload: Dict[str, Any],
    training_csv_path: Path,
    prefer_training_csv: bool,
) -> Dict[str, Dict[str, Any]]:
    if prefer_training_csv and training_csv_path.exists():
        return _load_histories_from_training_csv(training_csv_path)
    return _load_histories_from_results_json(payload)


def _load_cosine_table_from_csv(table_csv_path: Path) -> List[Dict[str, Any]]:
    if not table_csv_path.exists():
        raise FileNotFoundError(f"Cosine-table CSV not found: {table_csv_path}")
    rows: List[Dict[str, Any]] = []
    with table_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"method_key", "method_label", "status", "cosine_similarity_to_classical"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Cosine-table CSV is missing required columns: {sorted(missing)}. "
                "Re-run ablation_real_hardware.py to regenerate outputs."
            )
        for row in reader:
            rows.append(
                {
                    "method_key": str(row["method_key"]),
                    "method_label": str(row["method_label"]),
                    "status": str(row["status"]),
                    "cosine_similarity_to_classical": float(
                        row["cosine_similarity_to_classical"]
                    ),
                }
            )
    return rows


def _load_cosine_table_from_results_json(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("cosine_table", [])
    if not isinstance(rows, list):
        raise ValueError("Expected results JSON field 'cosine_table' to be a list.")
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "method_key": str(row.get("method_key", "")),
                "method_label": str(row.get("method_label", "")),
                "status": str(row.get("status", "")),
                "cosine_similarity_to_classical": float(
                    row.get("cosine_similarity_to_classical", np.nan)
                ),
            }
        )
    return out


def _choose_cosine_table(
    *,
    payload: Dict[str, Any],
    table_csv_path: Path,
    prefer_table_csv: bool,
) -> List[Dict[str, Any]]:
    if prefer_table_csv and table_csv_path.exists():
        return _load_cosine_table_from_csv(table_csv_path)
    return _load_cosine_table_from_results_json(payload)


def _resolve_table_output_path(
    raw_path: str,
    results_path: Path,
    fallback_name: str,
) -> Path | None:
    raw = str(raw_path).strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _plot_convergence(
    *,
    histories_by_method: Dict[str, Dict[str, Any]],
    output_path: Path,
    fig_width: float,
    fig_height: float,
    dpi: int,
    show_seed_traces: bool,
    title: str,
) -> None:
    ordered_keys = [key for key in _METHOD_ORDER if key in histories_by_method]
    if not ordered_keys:
        raise ValueError("No VQBR method histories were found to plot.")

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    y_values: List[float] = []
    max_iter = 0
    mean_curves_by_method: Dict[str, np.ndarray] = {}

    for method_key in ordered_keys:
        info = histories_by_method[method_key]
        histories_by_seed = dict(info.get("histories_by_seed", {}))
        seed_history_pairs = [
            (seed, histories_by_seed[seed])
            for seed in sorted(histories_by_seed.keys())
            if histories_by_seed[seed]
        ]
        if not seed_history_pairs:
            continue
        histories = [history for _, history in seed_history_pairs]
        hist_arr = _stack_histories(histories)
        mean_curve = np.nanmean(hist_arr, axis=0)
        std_curve = np.nanstd(hist_arr, axis=0, ddof=0)
        mean_curves_by_method[method_key] = np.asarray(mean_curve, dtype=float)
        x = np.arange(1, mean_curve.size + 1)
        max_iter = max(max_iter, int(mean_curve.size))
        y_values.extend(list(mean_curve[np.isfinite(mean_curve)]))

        color = _METHOD_COLORS.get(method_key, "#333333")
        if show_seed_traces and len(seed_history_pairs) > 1:
            for _, history in seed_history_pairs:
                x_seed = np.arange(1, len(history) + 1)
                ax.plot(
                    x_seed,
                    np.asarray(history, dtype=float),
                    color=color,
                    linewidth=0.8,
                    alpha=0.12,
                )

        ax.plot(
            x,
            mean_curve,
            color=color,
            linewidth=2.0,
            label=_CONVERGENCE_LEGEND_LABELS.get(method_key, str(info.get("method_label", method_key))),
        )
        if len(histories) > 1:
            ax.fill_between(
                x,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )

    ax.set_xlabel(r"$\mathrm{Iteration}$")
    ax.set_ylabel(r"$\widehat{\mathcal{J}}_{\log}(\boldsymbol{\theta})$")
    if str(title).strip():
        ax.set_title(title)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    if max_iter > 0:
        xticks = np.array(
            sorted(
                set(
                    [
                        1,
                        max(1, max_iter // 4),
                        max(1, max_iter // 2),
                        max(1, (3 * max_iter) // 4),
                        max_iter,
                    ]
                )
            ),
            dtype=float,
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(v)) for v in xticks])
        ax.set_xlim(-0.5, float(max_iter) + 1.5)

    if y_values:
        y_arr = np.asarray(y_values, dtype=float)
        y_min = float(np.nanmin(y_arr))
        y_max = float(np.nanmax(y_arr))
        if np.isfinite(y_min) and np.isfinite(y_max):
            y_pad = max(0.02, 0.08 * max(y_max - y_min, 1e-6))
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

    inset_start_iter = 25
    if max_iter >= inset_start_iter and mean_curves_by_method:
        axins = ax.inset_axes([0.40, 0.18, 0.57, 0.40])
        inset_y_values: List[float] = []
        inset_x_min = float(inset_start_iter) - 0.5
        inset_x_max = float(max_iter) + 0.5

        for method_key in ordered_keys:
            curve = np.asarray(mean_curves_by_method.get(method_key, []), dtype=float)
            if curve.size == 0:
                continue
            start_idx = max(int(inset_start_iter) - 1, 0)
            if start_idx >= curve.size:
                continue
            x_zoom = np.arange(start_idx + 1, curve.size + 1, dtype=float)
            y_zoom = curve[start_idx:]
            finite = np.isfinite(y_zoom)
            if not np.any(finite):
                continue
            inset_y_values.extend(list(y_zoom[finite]))
            axins.plot(
                x_zoom,
                y_zoom,
                color=_METHOD_COLORS.get(method_key, "#333333"),
                linewidth=1.0,
            )

        if inset_y_values:
            inset_y = np.asarray(inset_y_values, dtype=float)
            inset_min = float(np.nanmin(inset_y))
            inset_max = float(np.nanmax(inset_y))
            if np.isfinite(inset_min) and np.isfinite(inset_max):
                inset_pad = max(0.01, 0.08 * max(inset_max - inset_min, 1e-6))
                axins.set_ylim(inset_min - inset_pad, inset_max + inset_pad)
        axins.set_xlim(inset_x_min, inset_x_max)
        axins.set_xticks(
            np.array(
                sorted(
                    {
                        int(inset_start_iter),
                        int(max_iter // 2) if max_iter // 2 >= inset_start_iter else int(inset_start_iter),
                        int(max_iter),
                    }
                ),
                dtype=float,
            )
        )
        axins.tick_params(axis="both", labelsize=5.0, pad=1)
        axins.grid(alpha=0.25, linestyle="--", linewidth=0.5)
        if hasattr(ax, "indicate_inset_zoom"):
            ax.indicate_inset_zoom(axins, edgecolor="#777777", alpha=0.8, linewidth=0.6)

    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _ordered_cosine_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {str(row.get("method_key", "")): row for row in rows}
    return [by_key[key] for key in _METHOD_ORDER if key in by_key]


def _build_cosine_gap_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_key = {
        str(row.get("method_key", "")): float(row.get("cosine_similarity_to_classical", np.nan))
        for row in rows
    }
    sim_cos = float(by_key.get("vqbr_sim", np.nan))
    hardware_no_zne_cos = float(by_key.get("vqbr_ibm_no_zne", np.nan))
    hardware_zne_cos = float(by_key.get("vqbr_ibm_zne", np.nan))

    gap_without_zne = (
        float(abs(hardware_no_zne_cos - sim_cos))
        if np.isfinite(sim_cos) and np.isfinite(hardware_no_zne_cos)
        else float("nan")
    )
    gap_with_zne = (
        float(abs(hardware_zne_cos - sim_cos))
        if np.isfinite(sim_cos) and np.isfinite(hardware_zne_cos)
        else float("nan")
    )
    gap_reduction = (
        float(gap_without_zne - gap_with_zne)
        if np.isfinite(gap_without_zne) and np.isfinite(gap_with_zne)
        else float("nan")
    )
    relative_gap_reduction_pct = (
        float(100.0 * gap_reduction / gap_without_zne)
        if np.isfinite(gap_reduction) and gap_without_zne > 0.0
        else float("nan")
    )

    summary_text = ""
    if np.isfinite(gap_without_zne) and np.isfinite(gap_with_zne):
        summary_text = (
            "ZNE reduces the hardware-simulation cosine gap "
            f"from {gap_without_zne:.4f} to {gap_with_zne:.4f}."
        )

    return {
        "simulation_cosine_similarity": float(sim_cos),
        "hardware_no_zne_cosine_similarity": float(hardware_no_zne_cos),
        "hardware_zne_cosine_similarity": float(hardware_zne_cos),
        "gap_without_zne": float(gap_without_zne),
        "gap_with_zne": float(gap_with_zne),
        "gap_reduction": float(gap_reduction),
        "relative_gap_reduction_pct": float(relative_gap_reduction_pct),
        "summary_text": str(summary_text),
    }


def _plot_cosine_bars(
    *,
    cosine_rows: List[Dict[str, Any]],
    output_path: Path,
    fig_width: float,
    fig_height: float,
    dpi: int,
    title: str,
) -> Dict[str, Any]:
    ordered_rows = _ordered_cosine_rows(cosine_rows)
    if not ordered_rows:
        raise ValueError("No cosine rows were found for the simulator / hardware comparison.")

    summary = _build_cosine_gap_summary(ordered_rows)
    method_keys = [str(row["method_key"]) for row in ordered_rows]
    values = np.asarray(
        [float(row.get("cosine_similarity_to_classical", np.nan)) for row in ordered_rows],
        dtype=float,
    )
    labels = [_BAR_LABELS.get(key, key) for key in method_keys]
    colors = [_METHOD_COLORS.get(key, "#333333") for key in method_keys]
    x = 0.70 * np.arange(len(method_keys), dtype=float)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    bars = ax.bar(x, values, color=colors, width=0.40, edgecolor="none")
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)

    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + 0.0025,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.0,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
                "pad": 0.18,
            },
        )

    finite_values = values[np.isfinite(values)]
    if finite_values.size > 0:
        label_top = float(np.max(finite_values)) + 0.005
        y_min = max(0.84, float(np.min(finite_values)) - 0.025)
        y_max = min(1.0015, max(1.0008, label_top))
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(0.0, 1.05)

    ax.set_xlim(float(x[0]) - 0.22, float(x[-1]) + 0.22)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=8.0, pad=3)
    for tick in ax.get_xticklabels():
        tick.set_ha("center")
        tick.set_multialignment("center")
        tick.set_linespacing(0.82)
    ax.tick_params(axis="y", labelsize=9.0, pad=2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_ylabel("Cosine similarity", fontsize=9.0, labelpad=3)
    if str(title).strip():
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.24, right=0.995, bottom=0.31, top=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return summary


def _write_cosine_table_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method_key", "method_label", "status", "cosine_similarity_to_classical"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_cosine_table_tex(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Cosine similarity to the classical closed-form solution for the real-hardware ablation.}",
        r"\label{tab:real_hardware_cosine}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}lc@{}}",
        r"\toprule",
        r"Method & Cosine similarity \\",
        r"\midrule",
    ]
    for row in rows:
        value = float(row.get("cosine_similarity_to_classical", np.nan))
        value_tex = r"\textit{--}" if not np.isfinite(value) else f"{value:.6f}"
        label_tex = str(row.get("method_label", "")).replace("_", r"\_")
        lines.append(f"{label_tex} & {value_tex} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_table(rows: List[Dict[str, Any]]) -> None:
    print("Cosine similarity to the classical solution:")
    for row in rows:
        value = float(row.get("cosine_similarity_to_classical", np.nan))
        label = str(row.get("method_label", ""))
        status = str(row.get("status", ""))
        value_str = "nan" if not np.isfinite(value) else f"{value:.6f}"
        print(f"  {label}: {value_str} [{status}]")


def generate_artifacts(
    *,
    results_path: Path,
    training_csv_path: Path,
    objective_plot_path: Path,
    cosine_bar_plot_path: Path,
    fig_width: float,
    fig_height: float,
    dpi: int,
    show_seed_traces: bool,
    prefer_training_csv: bool,
    prefer_table_csv: bool,
    cosine_table_input_csv_path: Path | None = None,
    cosine_table_output_csv_path: Path | None = None,
    table_tex_path: Path | None = None,
) -> Dict[str, Any]:
    with results_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    table_input_csv_path = (
        cosine_table_input_csv_path
        if cosine_table_input_csv_path is not None
        else results_path.parent / "cosine_table.csv"
    )
    histories_by_method = _choose_histories(
        payload=payload,
        training_csv_path=training_csv_path,
        prefer_training_csv=bool(prefer_training_csv),
    )
    cosine_table_rows = _choose_cosine_table(
        payload=payload,
        table_csv_path=table_input_csv_path,
        prefer_table_csv=bool(prefer_table_csv),
    )
    if not cosine_table_rows:
        raise ValueError(
            "No cosine-table rows were found. "
            "Ensure ablation_real_hardware.py completed successfully."
        )

    config = payload.get("config", {})
    vqbr_config = config.get("vqbr", {}) if isinstance(config, dict) else {}
    ibm_config = config.get("ibm", {}) if isinstance(config, dict) else {}
    n_samples = config.get("n_samples", "?")
    n_features = config.get("n_features", "?")
    reps = vqbr_config.get("reps", "?") if isinstance(vqbr_config, dict) else "?"
    shots = ibm_config.get("shots", "?") if isinstance(ibm_config, dict) else "?"
    convergence_title = ""

    _plot_convergence(
        histories_by_method=histories_by_method,
        output_path=objective_plot_path,
        fig_width=float(fig_width),
        fig_height=float(fig_height),
        dpi=int(dpi),
        show_seed_traces=bool(show_seed_traces),
        title=convergence_title,
    )
    cosine_gap_summary = _plot_cosine_bars(
        cosine_rows=cosine_table_rows,
        output_path=cosine_bar_plot_path,
        fig_width=float(fig_width),
        fig_height=float(fig_height),
        dpi=int(dpi),
        title="",
    )

    if cosine_table_output_csv_path is not None:
        _write_cosine_table_csv(cosine_table_output_csv_path, cosine_table_rows)
    if table_tex_path is not None:
        _write_cosine_table_tex(table_tex_path, cosine_table_rows)

    return {
        "artifacts": {
            "objective_convergence_plot": str(objective_plot_path),
            "cosine_similarity_bar_plot": str(cosine_bar_plot_path),
        },
        "cosine_gap_summary": cosine_gap_summary,
        "cosine_table_rows": cosine_table_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the convergence figure and cosine-similarity bar chart for the "
            "real-hardware VQBR ablation."
        )
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results/synthetic/ablation_real_hardware_results.json",
    )
    parser.add_argument(
        "--training-csv-path",
        type=str,
        default="results/synthetic/ablation_real_hardware_training_log.csv",
    )
    parser.add_argument(
        "--cosine-table-input-csv-path",
        type=str,
        default="",
        help="Optional precomputed cosine-table CSV. If omitted, the plotter reads from results.json.",
    )
    parser.add_argument(
        "--objective-plot-path",
        type=str,
        default="results/synthetic/ablation_real_hardware_objective_convergence.png",
    )
    parser.add_argument(
        "--cosine-bar-plot-path",
        type=str,
        default="results/synthetic/ablation_real_hardware_cosine_similarity_bar.png",
    )
    parser.add_argument(
        "--cosine-table-output-csv-path",
        type=str,
        default="",
        help="Optional normalized cosine-table CSV output path.",
    )
    parser.add_argument(
        "--table-tex-path",
        type=str,
        default="",
        help="Optional LaTeX table output path.",
    )
    parser.add_argument("--fig-width", type=float, default=3.0)
    parser.add_argument("--fig-height", type=float, default=2.5)
    parser.add_argument("--dpi", type=int, default=1000)

    parser.set_defaults(show_seed_traces=False)
    parser.add_argument("--show-seed-traces", action="store_true")
    parser.add_argument("--hide-seed-traces", dest="show_seed_traces", action="store_false")

    parser.set_defaults(prefer_training_csv=True)
    parser.add_argument("--prefer-training-csv", action="store_true")
    parser.add_argument("--prefer-json-history", dest="prefer_training_csv", action="store_false")

    parser.set_defaults(prefer_table_csv=True)
    parser.add_argument("--prefer-table-csv", action="store_true")
    parser.add_argument("--prefer-json-table", dest="prefer_table_csv", action="store_false")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results_path = Path(args.results_path).expanduser().resolve()
    training_csv_path = Path(args.training_csv_path).expanduser().resolve()
    table_input_csv_path = (
        Path(args.cosine_table_input_csv_path).expanduser().resolve()
        if str(args.cosine_table_input_csv_path).strip()
        else None
    )
    objective_plot_path = Path(args.objective_plot_path).expanduser().resolve()
    cosine_bar_plot_path = Path(args.cosine_bar_plot_path).expanduser().resolve()
    cosine_table_output_csv_path = _resolve_table_output_path(
        args.cosine_table_output_csv_path,
        results_path,
        "cosine_table.csv",
    )
    table_tex_path = _resolve_table_output_path(
        args.table_tex_path,
        results_path,
        "cosine_table.tex",
    )

    output = generate_artifacts(
        results_path=results_path,
        training_csv_path=training_csv_path,
        objective_plot_path=objective_plot_path,
        cosine_bar_plot_path=cosine_bar_plot_path,
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
        dpi=int(args.dpi),
        show_seed_traces=bool(args.show_seed_traces),
        prefer_training_csv=bool(args.prefer_training_csv),
        prefer_table_csv=bool(args.prefer_table_csv),
        cosine_table_input_csv_path=table_input_csv_path,
        cosine_table_output_csv_path=cosine_table_output_csv_path,
        table_tex_path=table_tex_path,
    )

    print(f"Saved objective plot to: {objective_plot_path}")
    print(f"Saved cosine bar plot to: {cosine_bar_plot_path}")
    if cosine_table_output_csv_path is not None:
        print(f"Saved cosine-table CSV to: {cosine_table_output_csv_path}")
    if table_tex_path is not None:
        print(f"Saved cosine-table TeX to: {table_tex_path}")
    cosine_table_rows = list(output.get("cosine_table_rows", []))
    _print_table(cosine_table_rows)
    summary = dict(output.get("cosine_gap_summary", {}))
    summary_text = str(summary.get("summary_text", "")).strip()
    if summary_text:
        print(summary_text)


if __name__ == "__main__":
    main()
