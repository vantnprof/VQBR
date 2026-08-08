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
import matplotlib.patheffects as pe

plt.rcParams.update({"font.size": 10})

_METHOD_ORDER = ("vqbr", "cg")
_METHOD_COLORS = {
    "vqbr": "#d62728",
    "cg": "#000000",
}
_METHOD_LABELS = {
    "vqbr": "VQBR",
    "cg": "CG",
}
_METHOD_LINESTYLES = {
    "vqbr": "-",
    "cg": (0, (6, 2)),
}
_METHOD_LINEWIDTHS = {
    "vqbr": 2.0,
    "cg": 2.6,
}
_METHOD_ZORDERS = {
    "vqbr": 3,
    "cg": 5,
}


def _stack_histories(histories: List[List[float]]) -> np.ndarray:
    if not histories:
        return np.empty((0, 0), dtype=float)
    max_len = max(len(h) for h in histories)
    arr = np.full((len(histories), max_len), np.nan, dtype=float)
    for idx, history in enumerate(histories):
        if not history:
            continue
        vec = np.asarray(history, dtype=float).reshape(-1)
        arr[idx, : vec.size] = vec
    return arr


def _load_vqbr_histories_from_training_csv(training_csv_path: Path) -> Dict[int, List[float]]:
    if not training_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {training_csv_path}")

    rows_by_seed: Dict[int, List[tuple[int, float]]] = {}
    with training_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        iteration_key = "iteration" if "iteration" in fieldnames else "objective_eval"
        required_columns = {"seed", "batch_loss", iteration_key}
        missing = required_columns - fieldnames
        if missing:
            raise ValueError(
                f"Training CSV is missing required columns: {sorted(missing)}. "
                "Re-run synthetic_experiemnt.py to regenerate logs."
            )
        for row in reader:
            seed = int(row["seed"])
            iteration = int(row[iteration_key])
            loss = float(row["batch_loss"])
            rows_by_seed.setdefault(seed, []).append((iteration, loss))

    histories: Dict[int, List[float]] = {}
    for seed, entries in rows_by_seed.items():
        ordered = sorted(entries, key=lambda item: item[0])
        histories[seed] = [float(loss) for _, loss in ordered]
    return histories


def _load_vqbr_histories_from_results_json(payload: Dict[str, Any]) -> Dict[int, List[float]]:
    histories: Dict[int, List[float]] = {}
    per_seed = payload.get("per_seed", [])
    if not isinstance(per_seed, list):
        raise ValueError("Expected results JSON field 'per_seed' to be a list.")

    for seed_entry in per_seed:
        if not isinstance(seed_entry, dict):
            continue
        seed = int(seed_entry.get("seed", -1))
        vqbr_info = seed_entry.get("methods", {}).get("vqbr", {})
        objective_history = vqbr_info.get("objective_history", [])
        if not isinstance(objective_history, list):
            continue
        histories[seed] = [float(x) for x in objective_history]
    return histories


def _choose_vqbr_histories(
    *,
    payload: Dict[str, Any],
    training_csv_path: Path,
    prefer_training_csv: bool,
) -> Dict[int, List[float]]:
    if prefer_training_csv and training_csv_path.exists():
        return _load_vqbr_histories_from_training_csv(training_csv_path)
    return _load_vqbr_histories_from_results_json(payload)


def _load_cg_histories_from_training_csv(training_csv_path: Path) -> Dict[int, List[float]]:
    if not training_csv_path.exists():
        raise FileNotFoundError(f"CG training CSV not found: {training_csv_path}")

    rows_by_seed: Dict[int, List[tuple[int, float]]] = {}
    with training_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        iteration_key = "iteration" if "iteration" in fieldnames else "objective_eval"
        objective_key = (
            "reduced_objective"
            if "reduced_objective" in fieldnames
            else ("objective" if "objective" in fieldnames else "batch_loss")
        )
        required_columns = {"seed", iteration_key, objective_key}
        missing = required_columns - fieldnames
        if missing:
            raise ValueError(
                f"CG training CSV is missing required columns: {sorted(missing)}. "
                "Re-run synthetic_experiemnt.py in CG reuse mode to regenerate logs."
            )
        for row in reader:
            seed = int(row["seed"])
            iteration = int(row[iteration_key])
            loss = float(row[objective_key])
            rows_by_seed.setdefault(seed, []).append((iteration, loss))

    histories: Dict[int, List[float]] = {}
    for seed, entries in rows_by_seed.items():
        ordered = sorted(entries, key=lambda item: item[0])
        histories[seed] = [float(loss) for _, loss in ordered]
    return histories


def _load_cg_histories_from_results_json(payload: Dict[str, Any]) -> Dict[int, List[float]]:
    histories: Dict[int, List[float]] = {}
    per_seed = payload.get("per_seed", [])
    if not isinstance(per_seed, list):
        raise ValueError("Expected CG results JSON field 'per_seed' to be a list.")

    for seed_entry in per_seed:
        if not isinstance(seed_entry, dict):
            continue
        seed = int(seed_entry.get("seed", -1))
        cg_info = seed_entry.get("methods", {}).get("cg", {})
        objective_history = cg_info.get("objective_history", [])
        if not isinstance(objective_history, list):
            continue
        histories[seed] = [float(x) for x in objective_history]
    return histories


def _choose_cg_histories(
    *,
    payload: Dict[str, Any],
    training_csv_path: Path | None,
    prefer_training_csv: bool,
) -> Dict[int, List[float]]:
    if prefer_training_csv and training_csv_path is not None and training_csv_path.exists():
        return _load_cg_histories_from_training_csv(training_csv_path)
    return _load_cg_histories_from_results_json(payload)


def _resolve_optional_companion_path(
    raw_path: str,
    primary_results_path: Path,
    fallback_name: str,
) -> Path | None:
    raw = str(raw_path).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    candidate = primary_results_path.with_name(fallback_name)
    return candidate if candidate.exists() else None


def _resolve_output_path(
    raw_path: str,
    primary_results_path: Path,
    fallback_name: str,
    legacy_default_name: str,
) -> Path:
    raw = str(raw_path).strip()
    if not raw:
        return primary_results_path.with_name(fallback_name)
    candidate = Path(raw).expanduser().resolve()
    if candidate.name == str(legacy_default_name):
        return primary_results_path.with_name(fallback_name)
    return candidate


def _resolve_optional_output_path(
    raw_path: str,
    primary_results_path: Path,
    fallback_name: str,
) -> Path | None:
    raw = str(raw_path).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    candidate = primary_results_path.with_name(fallback_name)
    return candidate


def _truncate_histories(
    histories_by_seed: Dict[int, List[float]],
    max_iterations: int | None,
) -> Dict[int, List[float]]:
    if max_iterations is None or int(max_iterations) <= 0:
        return {
            int(seed): [float(x) for x in history]
            for seed, history in histories_by_seed.items()
        }
    limit = int(max_iterations)
    return {
        int(seed): [float(x) for x in history[:limit]]
        for seed, history in histories_by_seed.items()
    }


def _plot_convergence(
    *,
    histories_by_method: Dict[str, Dict[int, List[float]]],
    output_path: Path,
    fig_width: float,
    fig_height: float,
    dpi: int,
    show_seed_traces: bool,
    use_log_y: bool,
    title: str,
) -> None:
    ordered_keys = [key for key in _METHOD_ORDER if key in histories_by_method]
    if not ordered_keys:
        raise ValueError("No convergence histories were found to plot.")

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    max_iter = 0
    positive_values: List[float] = []

    for method_key in ordered_keys:
        histories_by_seed = histories_by_method[method_key]
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
        x = np.arange(1, mean_curve.size + 1)
        max_iter = max(max_iter, int(mean_curve.size))
        color = _METHOD_COLORS[method_key]
        linewidth = _METHOD_LINEWIDTHS.get(method_key, 2.0)
        linestyle = _METHOD_LINESTYLES.get(method_key, "-")
        zorder = _METHOD_ZORDERS.get(method_key, 3)
        markevery = max(1, mean_curve.size // 12)

        if show_seed_traces:
            for _, history in seed_history_pairs:
                x_seed = np.arange(1, len(history) + 1)
                ax.plot(
                    x_seed,
                    np.asarray(history, dtype=float),
                    color=color,
                    linewidth=0.8,
                    alpha=0.12,
                    zorder=1,
                )

        marker = "o" if method_key == "cg" else None
        markerfacecolor = "white" if method_key == "cg" else color
        markeredgecolor = color
        (line,) = ax.plot(
            x,
            mean_curve,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=_METHOD_LABELS[method_key],
            marker=marker,
            markevery=markevery if marker is not None else None,
            markersize=3.0 if marker is not None else 0.0,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            markeredgewidth=0.9 if marker is not None else 0.0,
            zorder=zorder,
        )
        if method_key == "cg":
            line.set_path_effects(
                [
                    pe.Stroke(linewidth=linewidth + 1.6, foreground="white"),
                    pe.Normal(),
                ]
            )
        if len(histories) > 1:
            ax.fill_between(
                x,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=color,
                alpha=0.10 if method_key == "cg" else 0.14,
                linewidth=0.0,
                zorder=max(0, zorder - 1),
            )

        positive_values.extend(
            [
                float(value)
                for value in np.asarray(mean_curve, dtype=float).reshape(-1)
                if np.isfinite(value) and float(value) > 0.0
            ]
        )

    xticks = np.array(
        sorted(
            set([1, max(1, max_iter // 4), max(1, max_iter // 2), max(1, (3 * max_iter) // 4), max_iter])
        ),
        dtype=float,
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(v)) for v in xticks])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\widehat{\mathcal{J}}_{\log}(\boldsymbol{\theta})$")
    ax.set_title(title)
    x_edge_pad = 1.5
    ax.set_xlim(1.0 - x_edge_pad, float(max_iter) + x_edge_pad)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    if use_log_y and positive_values:
        ax.set_yscale("log")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _print_aggregate_summary(
    payload: Dict[str, Any],
    cg_payload: Dict[str, Any] | None = None,
) -> None:
    aggregate = payload.get("aggregate", {})
    if not isinstance(aggregate, dict):
        return

    def _fmt(stats: Dict[str, Any], scientific: bool) -> str:
        mean = float(stats.get("mean", np.nan))
        std = float(stats.get("std", np.nan))
        if not np.isfinite(mean) or not np.isfinite(std):
            return "nan +/- nan"
        if scientific:
            return f"{mean:.6e} +/- {std:.6e}"
        return f"{mean:.6f} +/- {std:.6f}"

    vqbr = aggregate.get("vqbr", {})
    closed_form = aggregate.get("closed_form", {})
    runtime = aggregate.get("runtime_seconds", {})

    if isinstance(vqbr, dict):
        print("VQBR summary:")
        for metric, scientific in (
            ("cosine_similarity", False),
            ("feature_cosine_similarity", False),
            ("train_rmse", True),
            ("test_rmse", True),
            ("train_r2", False),
            ("test_r2", False),
        ):
            if metric in vqbr:
                print(f"  {metric}: {_fmt(vqbr[metric], scientific=scientific)}")

    if isinstance(closed_form, dict):
        print("Closed-form summary:")
        for metric, scientific in (
            ("train_rmse", True),
            ("test_rmse", True),
            ("train_r2", False),
            ("test_r2", False),
        ):
            if metric in closed_form:
                print(f"  {metric}: {_fmt(closed_form[metric], scientific=scientific)}")

    if isinstance(runtime, dict) and "vqbr_fit" in runtime:
        print(f"VQBR runtime (s): {_fmt(runtime['vqbr_fit'], scientific=True)}")

    if cg_payload is None:
        return

    cg_aggregate = cg_payload.get("aggregate", {})
    cg = cg_aggregate.get("cg", {}) if isinstance(cg_aggregate, dict) else {}
    cg_runtime = cg_aggregate.get("runtime_seconds", {}) if isinstance(cg_aggregate, dict) else {}
    if isinstance(cg, dict) and cg:
        print("CG summary:")
        for metric, scientific in (
            ("cosine_similarity", False),
            ("train_rmse", True),
            ("test_rmse", True),
            ("train_r2", False),
            ("test_r2", False),
            ("relative_residual", True),
            ("final_objective", True),
        ):
            if metric in cg:
                print(f"  {metric}: {_fmt(cg[metric], scientific=scientific)}")
    if isinstance(cg_runtime, dict) and "cg_fit" in cg_runtime:
        print(f"CG runtime (s): {_fmt(cg_runtime['cg_fit'], scientific=True)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot VQBR and optional CG convergence trajectories for the synthetic experiment."
        )
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results/synthetic/synthetic_experiment_results.json",
    )
    parser.add_argument(
        "--training-csv-path",
        type=str,
        default="results/synthetic/synthetic_experiment_training_log.csv",
    )
    parser.add_argument(
        "--cg-results-path",
        type=str,
        default="",
        help=(
            "Optional companion CG results JSON. "
            "If omitted, the plotter auto-loads sibling cg_results.json when present."
        ),
    )
    parser.add_argument(
        "--cg-training-csv-path",
        type=str,
        default="",
        help=(
            "Optional companion CG training CSV. "
            "If omitted, the plotter auto-loads sibling cg_training_log.csv when present."
        ),
    )
    parser.add_argument(
        "--objective-plot-path",
        type=str,
        default="results/synthetic/synthetic_vqbr_objective.png",
    )
    parser.add_argument(
        "--cg-objective-plot-path",
        type=str,
        default="",
        help=(
            "Optional standalone CG convergence plot path. "
            "If omitted and CG results are available, the plotter writes a sibling "
            "synthetic_cg_objective.png file."
        ),
    )
    parser.add_argument(
        "--cg-standalone-max-iterations",
        type=int,
        default=50,
        help=(
            "Maximum iterations shown in the standalone CG-only plot. "
            "Set <= 0 to keep the full CG history."
        ),
    )
    parser.add_argument("--fig-width", type=float, default=3.0)
    parser.add_argument("--fig-height", type=float, default=2.5)
    parser.add_argument("--dpi", type=int, default=1000)
    parser.add_argument("--log-y", action="store_true")

    parser.set_defaults(show_seed_traces=False)
    parser.add_argument("--show-seed-traces", action="store_true")
    parser.add_argument("--hide-seed-traces", dest="show_seed_traces", action="store_false")

    parser.set_defaults(prefer_training_csv=True)
    parser.add_argument("--prefer-training-csv", action="store_true")
    parser.add_argument("--prefer-json-history", dest="prefer_training_csv", action="store_false")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results_path = Path(args.results_path).expanduser().resolve()
    training_csv_path = Path(args.training_csv_path).expanduser().resolve()
    objective_plot_path = _resolve_output_path(
        args.objective_plot_path,
        results_path,
        "synthetic_vqbr_objective.png",
        "synthetic_vqbr_objective.png",
    )
    cg_objective_plot_path = _resolve_optional_output_path(
        args.cg_objective_plot_path,
        results_path,
        "synthetic_cg_objective.png",
    )

    with results_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    vqbr_histories = _choose_vqbr_histories(
        payload=payload,
        training_csv_path=training_csv_path,
        prefer_training_csv=bool(args.prefer_training_csv),
    )
    if not vqbr_histories:
        raise ValueError(
            "No VQBR objective histories were found. "
            "Ensure synthetic_experiemnt.py completed and wrote training logs."
        )

    histories_by_method: Dict[str, Dict[int, List[float]]] = {"vqbr": vqbr_histories}
    cg_payload: Dict[str, Any] | None = None
    cg_results_path = _resolve_optional_companion_path(
        args.cg_results_path,
        results_path,
        "cg_results.json",
    )
    cg_training_csv_path = _resolve_optional_companion_path(
        args.cg_training_csv_path,
        results_path,
        "cg_training_log.csv",
    )
    if cg_results_path is not None:
        with cg_results_path.open("r", encoding="utf-8") as f:
            cg_payload = json.load(f)
        cg_histories = _choose_cg_histories(
            payload=cg_payload,
            training_csv_path=cg_training_csv_path,
            prefer_training_csv=bool(args.prefer_training_csv),
        )
        if cg_histories:
            histories_by_method["cg"] = cg_histories

    config = payload.get("config", {})
    n_samples = config.get("n_samples", "?")
    n_features = config.get("n_features", "?")
    title = rf"$N={n_samples},\ D={n_features}$"

    _plot_convergence(
        histories_by_method=histories_by_method,
        output_path=objective_plot_path,
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
        dpi=int(args.dpi),
        show_seed_traces=bool(args.show_seed_traces),
        use_log_y=bool(args.log_y),
        title=title,
    )
    print(f"Saved objective plot to: {objective_plot_path}")
    if "cg" in histories_by_method and cg_objective_plot_path is not None:
        cg_only_histories = _truncate_histories(
            histories_by_method["cg"],
            int(args.cg_standalone_max_iterations),
        )
        _plot_convergence(
            histories_by_method={"cg": cg_only_histories},
            output_path=cg_objective_plot_path,
            fig_width=float(args.fig_width),
            fig_height=float(args.fig_height),
            dpi=int(args.dpi),
            show_seed_traces=bool(args.show_seed_traces),
            use_log_y=bool(args.log_y),
            title=title,
        )
        print(f"Saved CG-only objective plot to: {cg_objective_plot_path}")
    _print_aggregate_summary(payload, cg_payload=cg_payload)


if __name__ == "__main__":
    main()
