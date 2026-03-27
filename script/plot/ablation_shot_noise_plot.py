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


def _load_curve_from_csv(curve_csv_path: Path) -> List[Dict[str, Any]]:
    if not curve_csv_path.exists():
        raise FileNotFoundError(f"Curve-summary CSV not found: {curve_csv_path}")

    rows: List[Dict[str, Any]] = []
    with curve_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {
            "setting_index",
            "setting_label",
            "display_label",
            "is_analytic",
            "shots",
            "cosine_similarity_mean",
            "cosine_similarity_std",
        }
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Curve-summary CSV is missing required columns: {sorted(missing)}. "
                "Re-run ablation_shot_noise.py to regenerate outputs."
            )
        for row in reader:
            shot_raw = str(row.get("shots", "")).strip()
            rows.append(
                {
                    "setting_index": int(row["setting_index"]),
                    "setting_label": str(row["setting_label"]),
                    "display_label": str(row["display_label"]),
                    "is_analytic": str(row["is_analytic"]).strip().lower() == "true",
                    "shots": (None if not shot_raw else int(float(shot_raw))),
                    "cosine_similarity_mean": float(row["cosine_similarity_mean"]),
                    "cosine_similarity_std": float(row["cosine_similarity_std"]),
                    "cosine_similarity_n_valid": float(row.get("cosine_similarity_n_valid", np.nan)),
                }
            )
    return rows


def _load_curve_from_results_json(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("curve_summary", [])
    if not isinstance(rows, list):
        raise ValueError("Expected results JSON field 'curve_summary' to be a list.")

    curve_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        shot_value = row.get("shots")
        curve_rows.append(
            {
                "setting_index": int(row.get("setting_index", 0)),
                "setting_label": str(row.get("setting_label", "")),
                "display_label": str(row.get("display_label", "")),
                "is_analytic": bool(row.get("is_analytic", False)),
                "shots": None if shot_value is None else int(shot_value),
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
) -> List[Dict[str, Any]]:
    if prefer_curve_csv and curve_csv_path.exists():
        return _load_curve_from_csv(curve_csv_path)
    return _load_curve_from_results_json(payload)


def _resolve_logx_curve_plot_path(
    curve_plot_path: Path,
    logx_curve_plot_path_raw: str,
) -> Path:
    raw = str(logx_curve_plot_path_raw).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    suffix = curve_plot_path.suffix or ".png"
    return curve_plot_path.with_name(f"{curve_plot_path.stem}_logx{suffix}")


def _categorical_sort_key(row: Dict[str, Any]) -> tuple[int, int]:
    if bool(row["is_analytic"]):
        return (1, int(1e18))
    shots = row["shots"]
    return (0, int(shots) if shots is not None else int(1e18))


def _format_shot_power_label(row: Dict[str, Any]) -> str:
    if bool(row["is_analytic"]):
        return "analytic"
    shots = row["shots"]
    if shots is None or int(shots) <= 0:
        return str(row["display_label"])
    log2_shots = np.log2(float(shots))
    if np.isfinite(log2_shots) and np.isclose(log2_shots, round(log2_shots)):
        return rf"$2^{{{int(round(log2_shots))}}}$"
    return str(row["display_label"])


def _plot_categorical_curve(
    *,
    curve_rows: List[Dict[str, Any]],
    output_path: Path,
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    valid_rows = [
        row
        for row in curve_rows
        if np.isfinite(float(row["cosine_similarity_mean"]))
    ]
    if not valid_rows:
        raise ValueError("No valid cosine-similarity curve rows were found to plot.")

    valid_rows = sorted(valid_rows, key=_categorical_sort_key)
    x = np.arange(len(valid_rows), dtype=float)
    mean = np.asarray([float(row["cosine_similarity_mean"]) for row in valid_rows], dtype=float)
    std = np.asarray([float(row["cosine_similarity_std"]) for row in valid_rows], dtype=float)
    std = np.where(np.isfinite(std), std, 0.0)
    labels = [_format_shot_power_label(row) for row in valid_rows]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.plot(
        x,
        mean,
        color="#d62728",
        linewidth=2.0,
        marker="o",
        markersize=3.8,
        label="VQBR mean cosine",
    )
    if len(valid_rows) > 1:
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color="#d62728",
            alpha=0.20,
            linewidth=0.0,
            label="mean +/- std",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"$S$")
    ax.set_ylabel("Cosine similarity")
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.6)

    y_min = float(np.nanmin(mean - std))
    y_max = float(np.nanmax(mean + std))
    if np.isfinite(y_min) and np.isfinite(y_max):
        y_pad = max(0.02, 0.08 * max(y_max - y_min, 1e-6))
        ax.set_ylim(max(-1.0, y_min - y_pad), y_max + y_pad)

    ax.set_xlim(-0.25, float(len(valid_rows) - 1) + 0.25)
    fig.tight_layout(pad=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_logx_curve(
    *,
    curve_rows: List[Dict[str, Any]],
    output_path: Path,
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    analytic_rows = [row for row in curve_rows if bool(row["is_analytic"])]
    finite_rows = [
        row
        for row in curve_rows
        if not bool(row["is_analytic"]) and row["shots"] is not None
    ]
    finite_rows = [
        row
        for row in finite_rows
        if np.isfinite(float(row["shots"])) and np.isfinite(float(row["cosine_similarity_mean"]))
    ]
    if not finite_rows:
        raise ValueError("No finite-shot rows were found for the log-x plot.")

    finite_rows = sorted(finite_rows, key=lambda row: int(row["shots"]))
    shots = np.asarray([float(row["shots"]) for row in finite_rows], dtype=float)
    mean = np.asarray([float(row["cosine_similarity_mean"]) for row in finite_rows], dtype=float)
    std = np.asarray([float(row["cosine_similarity_std"]) for row in finite_rows], dtype=float)
    std = np.where(np.isfinite(std), std, 0.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.plot(
        shots,
        mean,
        color="#1f77b4",
        linewidth=2.0,
        marker="o",
        markersize=3.8,
        label="Finite-shot mean cosine",
    )
    if len(finite_rows) > 1:
        ax.fill_between(
            shots,
            mean - std,
            mean + std,
            color="#1f77b4",
            alpha=0.20,
            linewidth=0.0,
            label="finite-shot mean +/- std",
        )

    if analytic_rows:
        analytic_row = sorted(analytic_rows, key=lambda row: int(row["setting_index"]))[0]
        analytic_mean = float(analytic_row["cosine_similarity_mean"])
        analytic_std = float(analytic_row["cosine_similarity_std"])
        if np.isfinite(analytic_mean):
            ax.axhline(
                analytic_mean,
                color="#d62728",
                linestyle="--",
                linewidth=1.6,
                label="Analytic mean",
            )
            if np.isfinite(analytic_std):
                ax.axhspan(
                    analytic_mean - analytic_std,
                    analytic_mean + analytic_std,
                    color="#d62728",
                    alpha=0.12,
                    linewidth=0.0,
                )

    ax.set_xscale("log")
    ax.set_xticks(shots)
    ax.set_xticklabels([_format_shot_power_label(row) for row in finite_rows])
    ax.set_xlabel(r"$S$")
    ax.set_ylabel("Cosine similarity")
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.6)

    y_all = np.concatenate([mean - std, mean + std])
    if analytic_rows and np.isfinite(float(analytic_rows[0]["cosine_similarity_mean"])):
        analytic_mean = float(analytic_rows[0]["cosine_similarity_mean"])
        analytic_std = float(analytic_rows[0]["cosine_similarity_std"])
        y_all = np.concatenate(
            [
                y_all,
                np.asarray([analytic_mean - analytic_std, analytic_mean + analytic_std], dtype=float),
            ]
        )
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size > 0:
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        y_pad = max(0.02, 0.08 * max(y_max - y_min, 1e-6))
        ax.set_ylim(max(-1.0, y_min - y_pad), y_max + y_pad)

    fig.tight_layout(pad=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _print_curve_summary(curve_rows: List[Dict[str, Any]]) -> None:
    valid_rows = [
        row
        for row in curve_rows
        if np.isfinite(float(row["cosine_similarity_mean"]))
    ]
    if not valid_rows:
        return

    valid_rows = sorted(valid_rows, key=lambda row: int(row["setting_index"]))
    print("Cosine similarity by shot setting:")
    for row in valid_rows:
        label = "analytic" if bool(row["is_analytic"]) else f"shots={int(row['shots'])}"
        print(
            f"  {label}: "
            f"{float(row['cosine_similarity_mean']):.6f} +/- "
            f"{float(row['cosine_similarity_std']):.6f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cosine similarity versus shot budget for the shot-noise ablation."
        )
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results/synthetic/ablation_shot_noise_results.json",
    )
    parser.add_argument(
        "--curve-summary-csv-path",
        type=str,
        default="results/synthetic/ablation_shot_noise_curve_summary.csv",
    )
    parser.add_argument(
        "--curve-plot-path",
        type=str,
        default="results/synthetic/ablation_shot_noise_cosine_curve.png",
    )
    parser.add_argument(
        "--logx-curve-plot-path",
        type=str,
        default="",
        help=(
            "Optional output path for the finite-shot log-x companion plot. "
            "Defaults to <curve-plot-path stem>_logx<suffix>."
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
    logx_curve_plot_path = _resolve_logx_curve_plot_path(
        curve_plot_path=curve_plot_path,
        logx_curve_plot_path_raw=args.logx_curve_plot_path,
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
            "Ensure ablation_shot_noise.py completed successfully."
        )

    _plot_categorical_curve(
        curve_rows=curve_rows,
        output_path=curve_plot_path,
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
        dpi=int(args.dpi),
    )
    _plot_logx_curve(
        curve_rows=curve_rows,
        output_path=logx_curve_plot_path,
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
        dpi=int(args.dpi),
    )
    print(f"Saved shot-noise curve to: {curve_plot_path}")
    print(f"Saved finite-shot log-x curve to: {logx_curve_plot_path}")
    _print_curve_summary(curve_rows)


if __name__ == "__main__":
    main()
