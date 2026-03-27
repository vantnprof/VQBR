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


def _load_curve_from_csv(curve_csv_path: Path) -> List[Dict[str, float]]:
    if not curve_csv_path.exists():
        raise FileNotFoundError(f"Curve-summary CSV not found: {curve_csv_path}")

    rows: List[Dict[str, float]] = []
    with curve_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"reps", "cosine_similarity_mean", "cosine_similarity_std"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Curve-summary CSV is missing required columns: {sorted(missing)}. "
                "Re-run ablation_ansatz_expressibility.py to regenerate outputs."
            )
        for row in reader:
            rows.append(
                {
                    "reps": float(row["reps"]),
                    "cosine_similarity_mean": float(row["cosine_similarity_mean"]),
                    "cosine_similarity_std": float(row["cosine_similarity_std"]),
                    "cosine_similarity_n_valid": float(row.get("cosine_similarity_n_valid", np.nan)),
                }
            )
    return rows


def _load_curve_from_results_json(payload: Dict[str, Any]) -> List[Dict[str, float]]:
    rows = payload.get("curve_summary", [])
    if not isinstance(rows, list):
        raise ValueError("Expected results JSON field 'curve_summary' to be a list.")

    curve_rows: List[Dict[str, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        curve_rows.append(
            {
                "reps": float(row.get("reps", np.nan)),
                "cosine_similarity_mean": float(row.get("cosine_similarity_mean", np.nan)),
                "cosine_similarity_std": float(row.get("cosine_similarity_std", np.nan)),
                "cosine_similarity_n_valid": float(row.get("cosine_similarity_n_valid", np.nan)),
            }
        )
    return curve_rows


def _choose_curve_rows(
    *,
    payload: Dict[str, Any],
    curve_csv_path: Path,
    prefer_curve_csv: bool,
) -> List[Dict[str, float]]:
    if prefer_curve_csv and curve_csv_path.exists():
        return _load_curve_from_csv(curve_csv_path)
    return _load_curve_from_results_json(payload)


def _plot_cosine_curve(
    *,
    curve_rows: List[Dict[str, float]],
    output_path: Path,
    fig_width: float,
    fig_height: float,
    dpi: int,
    title: str,
    use_log_y: bool,
) -> None:
    valid_rows = [
        row
        for row in curve_rows
        if np.isfinite(float(row["reps"])) and np.isfinite(float(row["cosine_similarity_mean"]))
    ]
    if not valid_rows:
        raise ValueError("No valid cosine-similarity curve rows were found to plot.")

    valid_rows = sorted(valid_rows, key=lambda row: float(row["reps"]))
    reps = np.asarray([float(row["reps"]) for row in valid_rows], dtype=float)
    mean = np.asarray([float(row["cosine_similarity_mean"]) for row in valid_rows], dtype=float)
    std = np.asarray([float(row["cosine_similarity_std"]) for row in valid_rows], dtype=float)
    std = np.where(np.isfinite(std), std, 0.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.plot(
        reps,
        mean,
        color="#d62728",
        linewidth=2.0,
        marker="o",
        markersize=3.6,
        label="VQBR mean cosine",
    )
    if reps.size > 1 and np.any(np.isfinite(std)):
        ax.fill_between(
            reps,
            mean - std,
            mean + std,
            color="#d62728",
            alpha=0.20,
            linewidth=0.0,
            label="mean +/- std",
        )

    ax.set_xticks(reps)
    ax.set_xticklabels([str(int(round(v))) for v in reps])
    ax.set_xlabel(r"$p$")
    ax.set_ylabel("Cosine similarity")
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.6)

    y_min = float(np.nanmin(mean - std))
    y_max = float(np.nanmax(mean + std))
    if np.isfinite(y_min) and np.isfinite(y_max):
        y_pad = max(0.02, 0.08 * max(y_max - y_min, 1e-6))
        # Leave some headroom above near-unity cosine scores so the top of the
        # curve does not collapse into the plot boundary.
        ax.set_ylim(max(-1.0, y_min - y_pad), y_max + y_pad)

    if reps.size == 1:
        ax.set_xlim(reps[0] - 0.5, reps[0] + 0.5)
    else:
        ax.set_xlim(float(reps[0]) - 0.25, float(reps[-1]) + 0.25)

    if use_log_y:
        all_positive = np.all((mean > 0.0) & ((mean - std) > 0.0))
        if all_positive:
            ax.set_yscale("log")
        else:
            scale_reference = np.abs(np.concatenate([mean, mean - std, mean + std]))
            positive_reference = scale_reference[scale_reference > 0.0]
            linthresh = (
                float(max(np.min(positive_reference) * 0.5, 1e-3))
                if positive_reference.size > 0
                else 1e-3
            )
            ax.set_yscale("symlog", linthresh=linthresh)

    fig.tight_layout(pad=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _resolve_log_curve_plot_path(
    curve_plot_path: Path,
    log_curve_plot_path_raw: str,
) -> Path:
    raw = str(log_curve_plot_path_raw).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    suffix = curve_plot_path.suffix or ".png"
    return curve_plot_path.with_name(f"{curve_plot_path.stem}_log{suffix}")


def _print_curve_summary(curve_rows: List[Dict[str, float]]) -> None:
    valid_rows = [
        row
        for row in curve_rows
        if np.isfinite(float(row["reps"])) and np.isfinite(float(row["cosine_similarity_mean"]))
    ]
    if not valid_rows:
        return

    valid_rows = sorted(valid_rows, key=lambda row: float(row["reps"]))
    print("Cosine similarity by reps:")
    for row in valid_rows:
        print(
            f"  reps={int(round(float(row['reps'])))}: "
            f"{float(row['cosine_similarity_mean']):.6f} +/- "
            f"{float(row['cosine_similarity_std']):.6f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cosine similarity versus EfficientSU2 circuit repetitions for the "
            "ansatz expressibility ablation."
        )
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results/synthetic/ablation_ansatz_expressibility_results.json",
    )
    parser.add_argument(
        "--curve-summary-csv-path",
        type=str,
        default="results/synthetic/ablation_ansatz_expressibility_curve_summary.csv",
    )
    parser.add_argument(
        "--curve-plot-path",
        type=str,
        default="results/synthetic/ablation_ansatz_expressibility_cosine_curve.png",
    )
    parser.add_argument(
        "--log-curve-plot-path",
        type=str,
        default="",
        help=(
            "Optional output path for the log-scale companion plot. "
            "Defaults to <curve-plot-path stem>_log<suffix>."
        ),
    )
    parser.add_argument("--fig-width", type=float, default=3.0)
    parser.add_argument("--fig-height", type=float, default=2.5)
    parser.add_argument("--dpi", type=int, default=1000)

    parser.set_defaults(prefer_curve_csv=True)
    parser.add_argument("--prefer-curve-csv", action="store_true")
    parser.add_argument("--prefer-json", dest="prefer_curve_csv", action="store_false")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results_path = Path(args.results_path).expanduser().resolve()
    curve_csv_path = Path(args.curve_summary_csv_path).expanduser().resolve()
    curve_plot_path = Path(args.curve_plot_path).expanduser().resolve()
    log_curve_plot_path = _resolve_log_curve_plot_path(
        curve_plot_path=curve_plot_path,
        log_curve_plot_path_raw=args.log_curve_plot_path,
    )
    payload: Dict[str, Any] = {}
    if not (bool(args.prefer_curve_csv) and curve_csv_path.exists()):
        with results_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    elif results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

    curve_rows = _choose_curve_rows(
        payload=payload,
        curve_csv_path=curve_csv_path,
        prefer_curve_csv=bool(args.prefer_curve_csv),
    )
    if not curve_rows:
        raise ValueError(
            "No curve-summary rows were found. "
            "Ensure ablation_ansatz_expressibility.py completed successfully."
        )

    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    n_samples = config.get("n_samples", "?")
    n_features = config.get("n_features", "?")
    title = rf"$N={n_samples},\ D={n_features}$"

    _plot_cosine_curve(
        curve_rows=curve_rows,
        output_path=curve_plot_path,
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
        dpi=int(args.dpi),
        title=title,
        use_log_y=False,
    )
    _plot_cosine_curve(
        curve_rows=curve_rows,
        output_path=log_curve_plot_path,
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
        dpi=int(args.dpi),
        title=title,
        use_log_y=True,
    )
    print(f"Saved cosine-similarity curve to: {curve_plot_path}")
    print(f"Saved log-scale cosine curve to: {log_curve_plot_path}")
    _print_curve_summary(curve_rows)


if __name__ == "__main__":
    main()
