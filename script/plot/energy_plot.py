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

plt.rcParams.update({"font.size": 10})


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


def _load_histories_from_training_csv(training_csv_path: Path) -> Dict[int, List[float]]:
    if not training_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {training_csv_path}")

    rows_by_seed: Dict[int, List[tuple[int, float]]] = {}
    with training_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"seed", "iteration", "batch_loss"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Training CSV is missing required columns: {sorted(missing)}. "
                "Re-run energy_experiment.py to regenerate logs."
            )
        for row in reader:
            seed = int(row["seed"])
            iteration = int(row["iteration"])
            loss = float(row["batch_loss"])
            rows_by_seed.setdefault(seed, []).append((iteration, loss))

    histories: Dict[int, List[float]] = {}
    for seed, entries in rows_by_seed.items():
        ordered = sorted(entries, key=lambda item: item[0])
        histories[seed] = [float(loss) for _, loss in ordered]
    return histories


def _load_histories_from_results_json(payload: Dict[str, Any]) -> Dict[int, List[float]]:
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


def _choose_histories(
    *,
    payload: Dict[str, Any],
    training_csv_path: Path,
    prefer_training_csv: bool,
) -> Dict[int, List[float]]:
    if prefer_training_csv and training_csv_path.exists():
        return _load_histories_from_training_csv(training_csv_path)
    return _load_histories_from_results_json(payload)


def _plot_vqbr_objective(
    *,
    histories_by_seed: Dict[int, List[float]],
    output_path: Path,
    fig_width: float,
    fig_height: float,
    dpi: int,
    show_seed_traces: bool,
    use_log_y: bool,
    title: str,
) -> None:
    seed_history_pairs = [
        (seed, histories_by_seed[seed])
        for seed in sorted(histories_by_seed.keys())
        if histories_by_seed[seed]
    ]
    histories = [history for _, history in seed_history_pairs]
    if not histories:
        raise ValueError("No non-empty objective histories found to plot.")

    hist_arr = _stack_histories(histories)
    mean_curve = np.nanmean(hist_arr, axis=0)
    std_curve = np.nanstd(hist_arr, axis=0, ddof=0)
    x = np.arange(1, mean_curve.size + 1)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    if show_seed_traces:
        for seed, history in seed_history_pairs:
            x_seed = np.arange(1, len(history) + 1)
            ax.plot(
                x_seed,
                np.asarray(history, dtype=float),
                color="#1f77b4",
                linewidth=0.8,
                alpha=0.20,
                label=f"seed={seed}" if len(histories) == 1 else None,
            )

    ax.plot(x, mean_curve, color="#d62728", linewidth=2.0, label="VQBR mean objective")
    if len(histories) > 1:
        ax.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color="#d62728",
            alpha=0.20,
            linewidth=0.0,
            label="mean +/- std",
        )

    max_iter = int(mean_curve.size)
    xticks = np.array(
        sorted(set([1, max(1, max_iter // 4), max(1, max_iter // 2), max(1, (3 * max_iter) // 4), max_iter])),
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
    if use_log_y and np.all(mean_curve > 0.0):
        ax.set_yscale("log")
    fig.tight_layout(pad=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _print_aggregate_summary(payload: Dict[str, Any]) -> None:
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
            ("train_mse", True),
            ("test_mse", True),
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
            ("train_mse", True),
            ("test_mse", True),
            ("train_rmse", True),
            ("test_rmse", True),
            ("train_r2", False),
            ("test_r2", False),
        ):
            if metric in closed_form:
                print(f"  {metric}: {_fmt(closed_form[metric], scientific=scientific)}")

    if isinstance(runtime, dict) and "vqbr_fit" in runtime:
        print(f"VQBR runtime (s): {_fmt(runtime['vqbr_fit'], scientific=True)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot VQBR objective trajectories for the energy experiment from "
            "energy_experiment.py outputs."
        )
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results/energy/energy_experiment_results.json",
    )
    parser.add_argument(
        "--training-csv-path",
        type=str,
        default="results/energy/energy_experiment_training_log.csv",
    )
    parser.add_argument(
        "--objective-plot-path",
        type=str,
        default="results/energy/energy_vqbr_objective.png",
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
    objective_plot_path = Path(args.objective_plot_path).expanduser().resolve()

    with results_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    histories_by_seed = _choose_histories(
        payload=payload,
        training_csv_path=training_csv_path,
        prefer_training_csv=bool(args.prefer_training_csv),
    )
    if not histories_by_seed:
        raise ValueError(
            "No VQBR objective histories were found. "
            "Ensure energy_experiment.py completed and wrote training logs."
        )

    dataset_info = payload.get("dataset", {})
    n_samples = dataset_info.get("n_samples", "?")
    n_features = dataset_info.get("n_features", "?")
    title = rf"$N={n_samples},\ D={n_features}$"

    _plot_vqbr_objective(
        histories_by_seed=histories_by_seed,
        output_path=objective_plot_path,
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
        dpi=int(args.dpi),
        show_seed_traces=bool(args.show_seed_traces),
        use_log_y=bool(args.log_y),
        title=title,
    )
    print(f"Saved objective plot to: {objective_plot_path}")
    _print_aggregate_summary(payload)


if __name__ == "__main__":
    main()
