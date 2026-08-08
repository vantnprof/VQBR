from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shlex
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
SYNTHETIC_EXPERIMENT_PATH = ROOT_DIR / "script" / "exp" / "synthetic_experiemnt.py"
DEFAULT_RESULTS_ROOT = ROOT_DIR / "results" / "synthetic"


def _load_synthetic_experiment_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_synthetic_experiment_helpers",
        SYNTHETIC_EXPERIMENT_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load helper module from: {SYNTHETIC_EXPERIMENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SYN = _load_synthetic_experiment_module()

ClosedFormMAPBayesianRegression = _SYN.ClosedFormMAPBayesianRegression
VariationalQuantumBayesianRegression = _SYN.VariationalQuantumBayesianRegression

ENFORCED_VQBR_OPTIMIZER = _SYN.ENFORCED_VQBR_OPTIMIZER
ENFORCED_VQBR_LOSS = _SYN.ENFORCED_VQBR_LOSS
ENFORCED_VQBR_SU2_GATES = _SYN.ENFORCED_VQBR_SU2_GATES
LOG_RATIO_LOSS_FORMULA = _SYN.LOG_RATIO_LOSS_FORMULA
EPS = _SYN.EPS

_aggregate_results = _SYN._aggregate_results
_estimate_prior_from_prior_split = _SYN._estimate_prior_from_prior_split
_evaluate_regression_metrics = _SYN._evaluate_regression_metrics
_extract_vqbr_history = _SYN._extract_vqbr_history
_feature_direction_from_final_overlaps = _SYN._feature_direction_from_final_overlaps
_format_mean_std = _SYN._format_mean_std
_generate_synthetic_data = _SYN._generate_synthetic_data
_normalize_state = _SYN._normalize_state
_normalized_closed_form_state = _SYN._normalized_closed_form_state
_reconstruct_weights_from_formula = _SYN._reconstruct_weights_from_formula
_resolve_seeds = _SYN._resolve_seeds
_safe_cosine = _SYN._safe_cosine
_select_snapshot_for_solution = _SYN._select_snapshot_for_solution
_slugify_token = _SYN._slugify_token
_snapshot_at_final_theta = _SYN._snapshot_at_final_theta
_split_dataset_indices = _SYN._split_dataset_indices
_state_to_feature_direction = _SYN._state_to_feature_direction


def _resolve_shot_budgets(shots_list: str | None) -> List[int]:
    if shots_list is None:
        return [256, 1024, 4096, 16384]
    tokens = [token.strip() for token in str(shots_list).split(",") if token.strip()]
    if not tokens:
        raise ValueError("--shots-list was provided but no valid integers were found.")
    shot_budgets = [int(token) for token in tokens]
    if any(shots <= 0 for shots in shot_budgets):
        raise ValueError("All finite-shot budgets must be positive integers.")
    return sorted(set(shot_budgets))


def _build_shot_settings(
    shot_budgets: Sequence[int],
    include_analytic: bool,
) -> List[Dict[str, Any]]:
    settings: List[Dict[str, Any]] = []
    if include_analytic:
        settings.append(
            {
                "setting_label": "analytic",
                "display_label": "analytic",
                "is_analytic": True,
                "use_shot_noise": False,
                "shots": None,
            }
        )
    for shots in shot_budgets:
        settings.append(
            {
                "setting_label": f"shots_{int(shots)}",
                "display_label": str(int(shots)),
                "is_analytic": False,
                "use_shot_noise": True,
                "shots": int(shots),
            }
        )
    if not settings:
        raise ValueError("At least one shot-noise setting is required.")
    return settings


def _shot_settings_tag(settings: Sequence[Dict[str, Any]]) -> str:
    labels: List[str] = []
    for setting in settings:
        if bool(setting["is_analytic"]):
            labels.append("analytic")
        else:
            labels.append(f"shots{int(setting['shots'])}")
    return "-".join(labels)


def _default_run_tag(
    *,
    args: argparse.Namespace,
    seeds: Sequence[int],
    optimizer: str,
    loss: str,
    su2_gates: Sequence[str],
    settings: Sequence[Dict[str, Any]],
) -> str:
    gates_tag = "-".join(_slugify_token(g) for g in su2_gates)
    ent_tag = _slugify_token(str(args.vqbr_entanglement))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"ablation_shot_noise_N{int(args.n_samples)}_D{int(args.n_features)}_"
        f"seeds{int(len(seeds))}_maxiter{int(args.vqbr_maxiter)}_"
        f"{str(optimizer).lower()}_{_slugify_token(loss)}_"
        f"{_shot_settings_tag(settings)}_gates{gates_tag}_ent{ent_tag}_{timestamp}"
    )


def _to_filename(raw_path: str, fallback_name: str) -> str:
    raw = str(raw_path).strip()
    if not raw:
        return str(fallback_name)
    return Path(raw).name or str(fallback_name)


def _resolve_run_paths(
    *,
    args: argparse.Namespace,
    seeds: Sequence[int],
    optimizer: str,
    loss: str,
    su2_gates: Sequence[str],
    settings: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    cached = getattr(args, "_resolved_run_paths", None)
    if isinstance(cached, dict) and cached:
        return cached

    run_tag = (
        str(args.run_tag).strip()
        if str(args.run_tag).strip()
        else _default_run_tag(
            args=args,
            seeds=seeds,
            optimizer=optimizer,
            loss=loss,
            su2_gates=su2_gates,
            settings=settings,
        )
    )
    run_dir = (
        Path(str(args.run_dir).strip()).expanduser().resolve()
        if str(args.run_dir).strip()
        else Path(args.results_root).expanduser().resolve() / run_tag
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    results_name = _to_filename(args.output_path, "results.json")
    training_name = _to_filename(args.training_csv_path, "training_log.csv")
    per_seed_name = _to_filename(args.per_seed_csv_path, "per_seed_metrics.csv")
    summary_name = _to_filename(args.summary_csv_path, "summary_metrics.csv")
    curve_name = _to_filename(args.curve_summary_csv_path, "curve_summary.csv")
    run_config_name = _to_filename(args.run_config_path, "run_config.json")
    console_log_name = (
        _to_filename(args.console_log_path, "run.log")
        if str(args.console_log_path).strip()
        else ""
    )

    resolved = {
        "run_tag": str(run_tag),
        "run_dir": str(run_dir),
        "results_json": str(run_dir / results_name),
        "training_csv": str(run_dir / training_name),
        "per_seed_csv": str(run_dir / per_seed_name),
        "summary_csv": str(run_dir / summary_name),
        "curve_summary_csv": str(run_dir / curve_name),
        "run_config_json": str(run_dir / run_config_name),
        "console_log": (str(run_dir / console_log_name) if console_log_name else ""),
    }

    args.output_path = resolved["results_json"]
    args.training_csv_path = resolved["training_csv"]
    args.per_seed_csv_path = resolved["per_seed_csv"]
    args.summary_csv_path = resolved["summary_csv"]
    args.curve_summary_csv_path = resolved["curve_summary_csv"]
    args.run_config_path = resolved["run_config_json"]
    args.console_log_path = resolved["console_log"]
    args.resolved_run_tag = str(run_tag)
    args.resolved_run_dir = str(run_dir)
    args._resolved_run_paths = resolved
    return resolved


def _build_training_logger(
    *,
    seed: int,
    setting_display: str,
    maxiter: int,
    log_every_iter: int,
) -> Any:
    if log_every_iter <= 0:
        raise ValueError("--log-every-iter must be positive.")
    prev_loss: float | None = None

    def _format_delta(current: float, previous: float | None) -> str:
        if previous is None:
            return "delta=--"
        if not np.isfinite(current) or not np.isfinite(previous):
            return "delta=n/a"
        diff = float(current - previous)
        if diff > 0.0:
            return f"delta=+{abs(diff):.6e}"
        if diff < 0.0:
            return f"delta=-{abs(diff):.6e}"
        return "delta=0.000000e+00"

    def _callback(iteration: int, snapshot: Any) -> None:
        nonlocal prev_loss
        current_loss = float(getattr(snapshot, "batch_L_tilde", np.nan))
        if not np.isfinite(current_loss):
            current_loss = float(getattr(snapshot, "L_tilde", np.nan))
        should_log = (
            iteration == 1
            or iteration == maxiter
            or (iteration % log_every_iter == 0)
        )
        if should_log:
            print(
                f"    [seed={seed}, setting={setting_display}] eval {iteration:>4}/{maxiter:<4} | "
                f"objective={current_loss:.6e}, {_format_delta(current_loss, prev_loss)}",
                flush=True,
            )
        prev_loss = current_loss

    return _callback


def _write_training_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "setting_label",
        "display_label",
        "is_analytic",
        "shots",
        "seed",
        "objective_eval",
        "batch_loss",
        "L_tilde",
        "full_L_tilde",
        "a_hat",
        "c_hat",
        "d_hat",
        "e_hat",
        "h_hat",
        "epoch",
        "batch_position",
        "batch_size",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_per_seed_metrics_csv(path: Path, per_settings: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "setting_label",
        "display_label",
        "is_analytic",
        "shots",
        "seed",
        "method",
        "cosine_similarity",
        "feature_cosine_similarity",
        "state_overlap_abs",
        "train_rmse",
        "test_rmse",
        "train_r2",
        "test_r2",
        "t_hat",
        "final_objective",
        "objective_evaluations",
        "optimizer_nfev",
        "optimizer_iterations",
        "runtime_seconds_method",
        "runtime_seconds_seed_total",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for setting_entry in per_settings:
            setting = dict(setting_entry["setting"])
            shot_value = "" if setting["shots"] is None else int(setting["shots"])
            for seed_result in setting_entry.get("per_seed", []):
                seed = int(seed_result["seed"])
                runtime_seed_total = float(seed_result["runtime_seconds"].get("total", np.nan))
                runtime_vqbr = float(seed_result["runtime_seconds"].get("vqbr_fit", np.nan))
                runtime_closed_form = float(seed_result["runtime_seconds"].get("closed_form_fit", np.nan))

                vqbr_metrics = dict(seed_result["methods"]["vqbr"])
                writer.writerow(
                    {
                        "setting_label": str(setting["setting_label"]),
                        "display_label": str(setting["display_label"]),
                        "is_analytic": bool(setting["is_analytic"]),
                        "shots": shot_value,
                        "seed": seed,
                        "method": "vqbr",
                        "cosine_similarity": float(vqbr_metrics.get("cosine_similarity", np.nan)),
                        "feature_cosine_similarity": float(
                            vqbr_metrics.get("feature_cosine_similarity", np.nan)
                        ),
                        "state_overlap_abs": float(vqbr_metrics.get("state_overlap_abs", np.nan)),
                        "train_rmse": float(vqbr_metrics.get("train_rmse", np.nan)),
                        "test_rmse": float(vqbr_metrics.get("test_rmse", np.nan)),
                        "train_r2": float(vqbr_metrics.get("train_r2", np.nan)),
                        "test_r2": float(vqbr_metrics.get("test_r2", np.nan)),
                        "t_hat": float(vqbr_metrics.get("t_hat", np.nan)),
                        "final_objective": float(vqbr_metrics.get("final_objective", np.nan)),
                        "objective_evaluations": float(vqbr_metrics.get("objective_evaluations", np.nan)),
                        "optimizer_nfev": float(vqbr_metrics.get("optimizer_nfev", np.nan)),
                        "optimizer_iterations": float(vqbr_metrics.get("optimizer_iterations", np.nan)),
                        "runtime_seconds_method": float(runtime_vqbr),
                        "runtime_seconds_seed_total": float(runtime_seed_total),
                    }
                )

                closed_form_metrics = dict(seed_result["methods"]["closed_form"])
                writer.writerow(
                    {
                        "setting_label": str(setting["setting_label"]),
                        "display_label": str(setting["display_label"]),
                        "is_analytic": bool(setting["is_analytic"]),
                        "shots": shot_value,
                        "seed": seed,
                        "method": "closed_form",
                        "cosine_similarity": float("nan"),
                        "feature_cosine_similarity": float("nan"),
                        "state_overlap_abs": float("nan"),
                        "train_rmse": float(closed_form_metrics.get("train_rmse", np.nan)),
                        "test_rmse": float(closed_form_metrics.get("test_rmse", np.nan)),
                        "train_r2": float(closed_form_metrics.get("train_r2", np.nan)),
                        "test_r2": float(closed_form_metrics.get("test_r2", np.nan)),
                        "t_hat": float("nan"),
                        "final_objective": float("nan"),
                        "objective_evaluations": float("nan"),
                        "optimizer_nfev": float("nan"),
                        "optimizer_iterations": float("nan"),
                        "runtime_seconds_method": float(runtime_closed_form),
                        "runtime_seconds_seed_total": float(runtime_seed_total),
                    }
                )


def _write_summary_csv(path: Path, per_settings: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "setting_label",
        "display_label",
        "is_analytic",
        "shots",
        "category",
        "method",
        "metric",
        "mean",
        "std",
        "n_valid",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for setting_entry in per_settings:
            setting = dict(setting_entry["setting"])
            aggregate = dict(setting_entry["aggregate"])
            shot_value = "" if setting["shots"] is None else int(setting["shots"])
            for method in ("vqbr", "closed_form"):
                for metric, stats in aggregate.get(method, {}).items():
                    writer.writerow(
                        {
                            "setting_label": str(setting["setting_label"]),
                            "display_label": str(setting["display_label"]),
                            "is_analytic": bool(setting["is_analytic"]),
                            "shots": shot_value,
                            "category": "method_metric",
                            "method": method,
                            "metric": metric,
                            "mean": float(stats.get("mean", np.nan)),
                            "std": float(stats.get("std", np.nan)),
                            "n_valid": int(stats.get("n_valid", 0.0)),
                        }
                    )
            for metric, stats in aggregate.get("runtime_seconds", {}).items():
                writer.writerow(
                    {
                        "setting_label": str(setting["setting_label"]),
                        "display_label": str(setting["display_label"]),
                        "is_analytic": bool(setting["is_analytic"]),
                        "shots": shot_value,
                        "category": "runtime_seconds",
                        "method": "runtime",
                        "metric": metric,
                        "mean": float(stats.get("mean", np.nan)),
                        "std": float(stats.get("std", np.nan)),
                        "n_valid": int(stats.get("n_valid", 0.0)),
                    }
                )
            for metric, stats in aggregate.get("iterations", {}).items():
                writer.writerow(
                    {
                        "setting_label": str(setting["setting_label"]),
                        "display_label": str(setting["display_label"]),
                        "is_analytic": bool(setting["is_analytic"]),
                        "shots": shot_value,
                        "category": "iterations",
                        "method": "vqbr",
                        "metric": metric,
                        "mean": float(stats.get("mean", np.nan)),
                        "std": float(stats.get("std", np.nan)),
                        "n_valid": int(stats.get("n_valid", 0.0)),
                    }
                )


def _build_curve_summary(per_settings: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for setting_index, setting_entry in enumerate(per_settings):
        setting = dict(setting_entry["setting"])
        aggregate = dict(setting_entry["aggregate"])
        vqbr = dict(aggregate.get("vqbr", {}))
        runtime = dict(aggregate.get("runtime_seconds", {}))
        iterations = dict(aggregate.get("iterations", {}))

        cosine_stats = dict(vqbr.get("cosine_similarity", {}))
        feature_cosine_stats = dict(vqbr.get("feature_cosine_similarity", {}))
        state_overlap_stats = dict(vqbr.get("state_overlap_abs", {}))
        train_rmse_stats = dict(vqbr.get("train_rmse", {}))
        test_rmse_stats = dict(vqbr.get("test_rmse", {}))
        runtime_stats = dict(runtime.get("vqbr_fit", {}))
        iteration_stats = dict(iterations.get("vqbr_objective_evaluations", {}))

        rows.append(
            {
                "setting_index": int(setting_index),
                "setting_label": str(setting["setting_label"]),
                "display_label": str(setting["display_label"]),
                "is_analytic": bool(setting["is_analytic"]),
                "shots": None if setting["shots"] is None else int(setting["shots"]),
                "cosine_similarity_mean": float(cosine_stats.get("mean", np.nan)),
                "cosine_similarity_std": float(cosine_stats.get("std", np.nan)),
                "cosine_similarity_n_valid": int(cosine_stats.get("n_valid", 0.0)),
                "feature_cosine_similarity_mean": float(
                    feature_cosine_stats.get("mean", np.nan)
                ),
                "feature_cosine_similarity_std": float(
                    feature_cosine_stats.get("std", np.nan)
                ),
                "state_overlap_abs_mean": float(state_overlap_stats.get("mean", np.nan)),
                "state_overlap_abs_std": float(state_overlap_stats.get("std", np.nan)),
                "train_rmse_mean": float(train_rmse_stats.get("mean", np.nan)),
                "train_rmse_std": float(train_rmse_stats.get("std", np.nan)),
                "test_rmse_mean": float(test_rmse_stats.get("mean", np.nan)),
                "test_rmse_std": float(test_rmse_stats.get("std", np.nan)),
                "vqbr_fit_runtime_mean_seconds": float(runtime_stats.get("mean", np.nan)),
                "vqbr_fit_runtime_std_seconds": float(runtime_stats.get("std", np.nan)),
                "objective_evaluations_mean": float(iteration_stats.get("mean", np.nan)),
                "objective_evaluations_std": float(iteration_stats.get("std", np.nan)),
            }
        )
    return rows


def _write_curve_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "setting_index",
        "setting_label",
        "display_label",
        "is_analytic",
        "shots",
        "cosine_similarity_mean",
        "cosine_similarity_std",
        "cosine_similarity_n_valid",
        "feature_cosine_similarity_mean",
        "feature_cosine_similarity_std",
        "state_overlap_abs_mean",
        "state_overlap_abs_std",
        "train_rmse_mean",
        "train_rmse_std",
        "test_rmse_mean",
        "test_rmse_std",
        "vqbr_fit_runtime_mean_seconds",
        "vqbr_fit_runtime_std_seconds",
        "objective_evaluations_mean",
        "objective_evaluations_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if out["shots"] is None:
                out["shots"] = ""
            writer.writerow(out)


def _fit_vqbr_for_setting(
    *,
    args: argparse.Namespace,
    seed: int,
    setting: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    m0: np.ndarray,
    Sigma0: np.ndarray,
    V: float,
    w_closed: np.ndarray,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], float]:
    batch_size = int(args.vqbr_batch_size) if args.vqbr_batch_size is not None else int(X_train.shape[0])
    batch_size = min(max(batch_size, 1), int(X_train.shape[0]))
    shots_for_model = int(setting["shots"]) if setting["shots"] is not None else int(args.vqbr_shots)

    vqbr = VariationalQuantumBayesianRegression(
        shots=shots_for_model,
        batch_size=batch_size,
        reps=int(args.vqbr_reps),
        optimizer=ENFORCED_VQBR_OPTIMIZER,
        maxiter=int(args.vqbr_maxiter),
        cobyla_tol=float(args.vqbr_cobyla_tol),
        su2_gates=ENFORCED_VQBR_SU2_GATES,
        entanglement=str(args.vqbr_entanglement),
        loss=ENFORCED_VQBR_LOSS,
        random_state=int(seed),
        use_shot_noise=bool(setting["use_shot_noise"]),
        verbose=bool(args.vqbr_verbose),
    )

    callback = None
    if bool(args.log_training):
        setting_display = (
            "analytic"
            if bool(setting["is_analytic"])
            else f"shots={int(setting['shots'])}"
        )
        callback = _build_training_logger(
            seed=int(seed),
            setting_display=setting_display,
            maxiter=int(args.vqbr_maxiter),
            log_every_iter=int(args.log_every_iter),
        )

    vqbr_start = time.perf_counter()
    vqbr_state = np.asarray(
        vqbr.fit(
            X_train,
            y_train,
            m0,
            Sigma0,
            V,
            iteration_callback=callback,
        ),
        dtype=complex,
    ).reshape(-1)
    vqbr_runtime = float(time.perf_counter() - vqbr_start)
    vqbr_state = _normalize_state(vqbr_state)

    history_rows, history_series = _extract_vqbr_history(int(seed), vqbr.history_)
    history_rows_with_setting: List[Dict[str, Any]] = []
    for row in history_rows:
        row_with_setting = dict(row)
        row_with_setting["setting_label"] = str(setting["setting_label"])
        row_with_setting["display_label"] = str(setting["display_label"])
        row_with_setting["is_analytic"] = bool(setting["is_analytic"])
        row_with_setting["shots"] = (
            "" if setting["shots"] is None else int(setting["shots"])
        )
        history_rows_with_setting.append(row_with_setting)

    result_fun = float(getattr(vqbr.result_, "fun", np.nan))
    final_snapshot = _snapshot_at_final_theta(vqbr, n_samples=int(X_train.shape[0]))
    if final_snapshot is None:
        final_snapshot = _select_snapshot_for_solution(vqbr.history_, target_objective=result_fun)

    phi_from_overlaps = _feature_direction_from_final_overlaps(vqbr, X_train)
    if phi_from_overlaps is not None:
        phi = phi_from_overlaps
        phi_source = "overlap_lstsq"
    else:
        phi = _state_to_feature_direction(vqbr_state, n_features=int(args.n_features))
        phi_source = "statevector_phase_gauge"

    t_hat, w_hat, reconstruction_terms = _reconstruct_weights_from_formula(phi, final_snapshot)
    regression_metrics = _evaluate_regression_metrics(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        w=w_hat,
    )

    w_closed_norm = float(np.linalg.norm(w_closed))
    w_closed_unit = (
        np.asarray(w_closed, dtype=float) / w_closed_norm
        if w_closed_norm > EPS
        else np.zeros_like(w_closed, dtype=float)
    )
    feature_cosine_similarity = _safe_cosine(phi, w_closed_unit)
    cosine_similarity = _safe_cosine(w_hat, w_closed)
    closed_form_state = _normalized_closed_form_state(w_closed)
    state_overlap_abs = float(np.abs(np.vdot(closed_form_state, vqbr_state)))

    objective_history = history_series["objective_history"]
    finite_objectives = [x for x in objective_history if np.isfinite(x)]
    if np.isfinite(result_fun):
        final_objective = float(result_fun)
    elif finite_objectives:
        final_objective = float(np.min(finite_objectives))
    else:
        final_objective = float("nan")

    optimizer_nfev_raw = getattr(vqbr.result_, "nfev", np.nan)
    try:
        optimizer_nfev = float(optimizer_nfev_raw)
    except Exception:
        optimizer_nfev = float("nan")

    optimizer_iterations_raw = getattr(vqbr.result_, "nit", np.nan)
    try:
        optimizer_iterations = float(optimizer_iterations_raw)
    except Exception:
        optimizer_iterations = float("nan")

    vqbr_metrics = {
        "cosine_similarity": float(cosine_similarity),
        "feature_cosine_similarity": float(feature_cosine_similarity),
        "state_overlap_abs": float(state_overlap_abs),
        "train_rmse": float(regression_metrics["train_rmse"]),
        "test_rmse": float(regression_metrics["test_rmse"]),
        "train_r2": float(regression_metrics["train_r2"]),
        "test_r2": float(regression_metrics["test_r2"]),
        "t_hat": float(t_hat),
        "final_objective": float(final_objective),
        "objective_evaluations": int(len(objective_history)),
        "optimizer_nfev": float(optimizer_nfev),
        "optimizer_iterations": float(optimizer_iterations),
        "reconstruction_terms": reconstruction_terms,
        "objective_history": [float(x) for x in objective_history],
        "a_hat_history": [float(x) for x in history_series["a_hat_history"]],
        "c_hat_history": [float(x) for x in history_series["c_hat_history"]],
        "d_hat_history": [float(x) for x in history_series["d_hat_history"]],
        "e_hat_history": [float(x) for x in history_series["e_hat_history"]],
        "h_hat_history": [float(x) for x in history_series["h_hat_history"]],
        "optimizer_success": bool(getattr(vqbr.result_, "success", False)),
        "optimizer_message": str(getattr(vqbr.result_, "message", "")),
        "simulator_backend": str(getattr(vqbr, "simulator_backend_", "")),
        "simulator_device": str(getattr(vqbr, "simulator_device_", "")),
        "feature_direction_source": str(phi_source),
        "shots": None if setting["shots"] is None else int(setting["shots"]),
        "use_shot_noise": bool(setting["use_shot_noise"]),
    }
    return vqbr_metrics, history_rows_with_setting, float(vqbr_runtime)


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if args.n_samples <= 0:
        raise ValueError("--N/--n-samples must be positive.")
    if args.n_features <= 0:
        raise ValueError("--D/--n-features must be positive.")
    if args.noise_std < 0.0:
        raise ValueError("--noise-std must be non-negative.")
    if args.w_true_std <= 0.0:
        raise ValueError("--w-true-std must be positive.")
    if args.vqbr_reps <= 0:
        raise ValueError("--vqbr-reps must be positive.")
    if args.vqbr_maxiter <= 0:
        raise ValueError("--vqbr-maxiter must be positive.")
    if args.vqbr_cobyla_tol <= 0.0:
        raise ValueError("--vqbr-cobyla-tol must be positive.")
    if args.vqbr_batch_size is not None and args.vqbr_batch_size <= 0:
        raise ValueError("--vqbr-batch-size must be positive when provided.")
    if args.vqbr_shots <= 0:
        raise ValueError("--vqbr-shots must be positive.")
    if args.log_every_iter <= 0:
        raise ValueError("--log-every-iter must be positive.")

    seeds = _resolve_seeds(
        num_seeds=int(args.num_seeds),
        seed_offset=int(args.seed_offset),
        explicit_seeds=args.seeds,
    )
    shot_budgets = _resolve_shot_budgets(args.shots_list)
    settings = _build_shot_settings(
        shot_budgets=shot_budgets,
        include_analytic=bool(args.include_analytic),
    )

    effective_optimizer = ENFORCED_VQBR_OPTIMIZER
    effective_loss = ENFORCED_VQBR_LOSS
    effective_su2_gates = ENFORCED_VQBR_SU2_GATES
    effective_loss_formula = LOG_RATIO_LOSS_FORMULA

    if str(args.vqbr_loss).strip().lower() != str(effective_loss).lower():
        print(
            f"Forcing VQBR loss to '{effective_loss}' (ignoring requested '{args.vqbr_loss}')."
        )

    run_paths = _resolve_run_paths(
        args=args,
        seeds=seeds,
        optimizer=effective_optimizer,
        loss=effective_loss,
        su2_gates=effective_su2_gates,
        settings=settings,
    )

    print("=== Shot-Noise Ablation ===")
    print(
        f"N={int(args.n_samples)}, D={int(args.n_features)}, "
        f"noise_std={float(args.noise_std):.6f}, w_true_std={float(args.w_true_std):.6f}"
    )
    print(
        "Split ratios: "
        f"prior={float(args.prior_ratio):.3f}, "
        f"train={float(args.train_ratio):.3f}, "
        f"test={float(args.test_ratio):.3f}"
    )
    print(f"Seeds ({len(seeds)}): {', '.join(str(s) for s in seeds)}")
    print(
        "Shot settings: "
        + ", ".join(
            str(setting["display_label"]) if not bool(setting["is_analytic"]) else "analytic"
            for setting in settings
        )
    )
    print(
        f"VQBR config: optimizer={effective_optimizer}, reps={int(args.vqbr_reps)}, "
        f"su2_gates={effective_su2_gates}, entanglement={args.vqbr_entanglement}, "
        f"maxiter={int(args.vqbr_maxiter)}, cobyla_tol={float(args.vqbr_cobyla_tol):.3e}, "
        f"loss={effective_loss}"
    )
    print(f"VQBR loss formula: {effective_loss_formula}")
    print(f"Run directory: {run_paths['run_dir']}")

    training_rows: List[Dict[str, Any]] = []
    per_setting_seed_results: Dict[str, List[Dict[str, Any]]] = {
        str(setting["setting_label"]): [] for setting in settings
    }
    setting_runtime_totals: Dict[str, float] = {
        str(setting["setting_label"]): 0.0 for setting in settings
    }
    experiment_start = time.perf_counter()
    split_sizes_reference: Dict[str, int] | None = None

    for seed_idx, seed in enumerate(seeds, start=1):
        print("")
        print(f"[seed={seed}] ({seed_idx}/{len(seeds)})", flush=True)

        X, y, w_true = _generate_synthetic_data(
            n_samples=int(args.n_samples),
            n_features=int(args.n_features),
            noise_std=float(args.noise_std),
            w_true_std=float(args.w_true_std),
            seed=int(seed),
        )
        split = _split_dataset_indices(
            n_samples=int(args.n_samples),
            prior_ratio=float(args.prior_ratio),
            train_ratio=float(args.train_ratio),
            test_ratio=float(args.test_ratio),
            seed=int(seed),
        )

        X_prior = X[split.prior_indices]
        y_prior = y[split.prior_indices]
        X_train = X[split.train_indices]
        y_train = y[split.train_indices]
        X_test = X[split.test_indices]
        y_test = y[split.test_indices]

        if split_sizes_reference is None:
            split_sizes_reference = {
                "prior": int(X_prior.shape[0]),
                "train": int(X_train.shape[0]),
                "test": int(X_test.shape[0]),
            }

        prior_start = time.perf_counter()
        m0, Sigma0, V, prior_info = _estimate_prior_from_prior_split(
            X_prior,
            y_prior,
            seed=int(seed),
            bootstrap_samples=int(args.prior_bootstrap_samples),
            ridge=float(args.prior_ridge),
            sigma_floor=float(args.prior_sigma_floor),
            v_floor=float(args.V_floor),
        )
        prior_runtime = float(time.perf_counter() - prior_start)
        print(
            f"  prior estimate: V={float(V):.6e}, "
            f"prior_fit_mse={float(prior_info['prior_fit_mse']):.6e}, "
            f"mean_sigma0_diag={float(prior_info['mean_sigma0_diag']):.6e}"
        )

        closed_form_start = time.perf_counter()
        closed_form_model = ClosedFormMAPBayesianRegression()
        closed_form_model.fit(X_train, y_train, m0, Sigma0, V)
        w_closed = np.asarray(closed_form_model.get_weights(), dtype=float).reshape(-1)
        closed_form_runtime = float(time.perf_counter() - closed_form_start)
        closed_form_metrics = _evaluate_regression_metrics(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            w=w_closed,
        )
        print(
            "  closed-form: "
            f"train_rmse={closed_form_metrics['train_rmse']:.6e}, "
            f"test_rmse={closed_form_metrics['test_rmse']:.6e}, "
            f"train_R2={closed_form_metrics['train_r2']:.6f}, "
            f"test_R2={closed_form_metrics['test_r2']:.6f}"
        )

        for setting in settings:
            setting_label = str(setting["setting_label"])
            setting_display = (
                "analytic"
                if bool(setting["is_analytic"])
                else f"shots={int(setting['shots'])}"
            )
            print(f"  setting={setting_display}", flush=True)
            print(
                "    objective mode: "
                + (
                    "analytic (no shot noise)"
                    if bool(setting["is_analytic"])
                    else f"finite-shot with S={int(setting['shots'])}"
                )
            )

            vqbr_metrics, setting_training_rows, vqbr_runtime = _fit_vqbr_for_setting(
                args=args,
                seed=int(seed),
                setting=setting,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                m0=m0,
                Sigma0=Sigma0,
                V=float(V),
                w_closed=w_closed,
            )
            training_rows.extend(setting_training_rows)

            runtime_info = {
                "prior_estimation": float(prior_runtime),
                "closed_form_fit": float(closed_form_runtime),
                "vqbr_fit": float(vqbr_runtime),
                "total": float(prior_runtime + closed_form_runtime + vqbr_runtime),
            }
            setting_runtime_totals[setting_label] += float(runtime_info["total"])

            print(
                "    vqbr: "
                f"cos={vqbr_metrics['cosine_similarity']:.6f}, "
                f"feature_cos={vqbr_metrics['feature_cosine_similarity']:.6f}, "
                f"train_rmse={vqbr_metrics['train_rmse']:.6e}, "
                f"test_rmse={vqbr_metrics['test_rmse']:.6e}, "
                f"train_R2={vqbr_metrics['train_r2']:.6f}, "
                f"test_R2={vqbr_metrics['test_r2']:.6f}, "
                f"t_hat={vqbr_metrics['t_hat']:.6e}"
            )
            print(
                "    runtime (s): "
                f"prior={runtime_info['prior_estimation']:.3f}, "
                f"closed_form={runtime_info['closed_form_fit']:.3f}, "
                f"vqbr_fit={runtime_info['vqbr_fit']:.3f}, "
                f"total={runtime_info['total']:.3f}"
            )

            per_setting_seed_results[setting_label].append(
                {
                    "seed": int(seed),
                    "setting_label": str(setting["setting_label"]),
                    "display_label": str(setting["display_label"]),
                    "is_analytic": bool(setting["is_analytic"]),
                    "shots": None if setting["shots"] is None else int(setting["shots"]),
                    "dataset": {
                        "n_samples": int(args.n_samples),
                        "n_features": int(args.n_features),
                        "noise_std": float(args.noise_std),
                        "w_true_std": float(args.w_true_std),
                        "w_true_norm": float(np.linalg.norm(w_true)),
                    },
                    "split_sizes": {
                        "prior": int(X_prior.shape[0]),
                        "train": int(X_train.shape[0]),
                        "test": int(X_test.shape[0]),
                    },
                    "prior_estimate": {
                        "V": float(V),
                        "m0_norm": float(np.linalg.norm(m0)),
                        "mean_sigma0_diag": float(np.mean(np.diag(Sigma0))),
                        "prior_fit_mse": float(prior_info["prior_fit_mse"]),
                        "bootstrap_samples": int(args.prior_bootstrap_samples),
                        "ridge": float(args.prior_ridge),
                        "sigma_floor": float(args.prior_sigma_floor),
                        "v_floor": float(args.V_floor),
                    },
                    "runtime_seconds": runtime_info,
                    "methods": {
                        "closed_form": {
                            "train_rmse": float(closed_form_metrics["train_rmse"]),
                            "test_rmse": float(closed_form_metrics["test_rmse"]),
                            "train_r2": float(closed_form_metrics["train_r2"]),
                            "test_r2": float(closed_form_metrics["test_r2"]),
                        },
                        "vqbr": vqbr_metrics,
                    },
                }
            )

    per_settings_payload: List[Dict[str, Any]] = []
    for setting in settings:
        setting_label = str(setting["setting_label"])
        aggregate = _aggregate_results(per_setting_seed_results[setting_label])
        per_settings_payload.append(
            {
                "setting": {
                    "setting_label": str(setting["setting_label"]),
                    "display_label": str(setting["display_label"]),
                    "is_analytic": bool(setting["is_analytic"]),
                    "shots": None if setting["shots"] is None else int(setting["shots"]),
                    "use_shot_noise": bool(setting["use_shot_noise"]),
                },
                "per_seed": per_setting_seed_results[setting_label],
                "aggregate": aggregate,
                "runtime_seconds_total": float(setting_runtime_totals[setting_label]),
            }
        )

    total_runtime = float(time.perf_counter() - experiment_start)
    curve_summary = _build_curve_summary(per_settings_payload)

    print("")
    print("=== Aggregate Results By Shot Setting (mean +/- std over seeds) ===")
    for setting_entry in per_settings_payload:
        setting = dict(setting_entry["setting"])
        aggregate = dict(setting_entry["aggregate"])
        heading = "analytic" if bool(setting["is_analytic"]) else f"shots={int(setting['shots'])}"
        print(heading)
        print(
            "  VQBR state cosine:   "
            f"{_format_mean_std(aggregate['vqbr']['cosine_similarity'], scientific=False)}"
        )
        print(
            "  VQBR feature cosine: "
            f"{_format_mean_std(aggregate['vqbr']['feature_cosine_similarity'], scientific=False)}"
        )
        print(
            "  VQBR test RMSE:      "
            f"{_format_mean_std(aggregate['vqbr']['test_rmse'], scientific=True)}"
        )
        print(
            "  VQBR fit runtime (s): "
            f"{_format_mean_std(aggregate['runtime_seconds']['vqbr_fit'], scientific=True)}"
        )

    cosine_means = np.asarray(
        [float(row["cosine_similarity_mean"]) for row in curve_summary],
        dtype=float,
    )
    if np.any(np.isfinite(cosine_means)):
        best_index = int(np.nanargmax(cosine_means))
        best_row = curve_summary[best_index]
        best_label = (
            "analytic"
            if bool(best_row["is_analytic"])
            else f"shots={int(best_row['shots'])}"
        )
        print("")
        print(
            "Best mean cosine similarity: "
            f"{best_label}, "
            f"cos={float(best_row['cosine_similarity_mean']):.6f} +/- "
            f"{float(best_row['cosine_similarity_std']):.6f}"
        )
    else:
        print("")
        print("Best mean cosine similarity: unavailable (all runs returned NaN).")
    print(f"Total wall-clock runtime: {total_runtime:.3f} s")

    output_path = Path(args.output_path).expanduser().resolve()
    training_csv_path = Path(args.training_csv_path).expanduser().resolve()
    per_seed_csv_path = Path(args.per_seed_csv_path).expanduser().resolve()
    summary_csv_path = Path(args.summary_csv_path).expanduser().resolve()
    curve_summary_csv_path = Path(args.curve_summary_csv_path).expanduser().resolve()
    run_config_path = Path(args.run_config_path).expanduser().resolve()

    _write_training_csv(training_csv_path, training_rows)
    _write_per_seed_metrics_csv(per_seed_csv_path, per_settings_payload)
    _write_summary_csv(summary_csv_path, per_settings_payload)
    _write_curve_summary_csv(curve_summary_csv_path, curve_summary)

    payload: Dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": str(getattr(args, "resolved_run_tag", "")),
        "run_dir": str(getattr(args, "resolved_run_dir", "")),
        "config": {
            "n_samples": int(args.n_samples),
            "n_features": int(args.n_features),
            "noise_std": float(args.noise_std),
            "w_true_std": float(args.w_true_std),
            "split_ratios": {
                "prior": float(args.prior_ratio),
                "train": float(args.train_ratio),
                "test": float(args.test_ratio),
            },
            "seeds": [int(seed) for seed in seeds],
            "num_seeds": int(len(seeds)),
            "seed_mode": "explicit_list" if args.seeds is not None else "offset_range",
            "prior_estimation": {
                "bootstrap_samples": int(args.prior_bootstrap_samples),
                "ridge": float(args.prior_ridge),
                "sigma_floor": float(args.prior_sigma_floor),
                "v_floor": float(args.V_floor),
            },
            "ablation": {
                "name": "shot_noise",
                "swept_parameter": "vqbr.use_shot_noise/vqbr.shots",
                "settings": [
                    {
                        "setting_label": str(setting["setting_label"]),
                        "display_label": str(setting["display_label"]),
                        "is_analytic": bool(setting["is_analytic"]),
                        "shots": None if setting["shots"] is None else int(setting["shots"]),
                    }
                    for setting in settings
                ],
            },
            "vqbr": {
                "ansatz": "EfficientSU2",
                "optimizer": effective_optimizer,
                "reps": int(args.vqbr_reps),
                "su2_gates": [str(g) for g in effective_su2_gates],
                "entanglement": str(args.vqbr_entanglement),
                "maxiter": int(args.vqbr_maxiter),
                "cobyla_tol": float(args.vqbr_cobyla_tol),
                "batch_size": (
                    None
                    if args.vqbr_batch_size is None
                    else int(args.vqbr_batch_size)
                ),
                "analytic_placeholder_shots": int(args.vqbr_shots),
                "loss": effective_loss,
                "loss_formula": effective_loss_formula,
            },
        },
        "split_sizes": split_sizes_reference if split_sizes_reference is not None else {},
        "curve_summary": curve_summary,
        "per_settings": per_settings_payload,
        "runtime_seconds_total": float(total_runtime),
        "artifacts": {
            "run_dir": str(getattr(args, "resolved_run_dir", "")),
            "run_config_path": str(run_config_path),
            "training_csv_path": str(training_csv_path),
            "per_seed_csv_path": str(per_seed_csv_path),
            "summary_csv_path": str(summary_csv_path),
            "curve_summary_csv_path": str(curve_summary_csv_path),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=int(args.json_indent))

    run_config_payload: Dict[str, Any] = {
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(token) for token in sys.argv),
        "args": dict(vars(args)),
        "resolved": {
            "seeds": [int(seed) for seed in seeds],
            "num_seeds": int(len(seeds)),
            "seed_mode": "explicit_list" if args.seeds is not None else "offset_range",
            "settings": payload["config"]["ablation"]["settings"],
            "split_ratios": {
                "prior": float(args.prior_ratio),
                "train": float(args.train_ratio),
                "test": float(args.test_ratio),
            },
            "split_sizes": split_sizes_reference if split_sizes_reference is not None else {},
            "prior_estimation": payload["config"]["prior_estimation"],
            "vqbr": payload["config"]["vqbr"],
            "outputs": {
                "results_json": str(output_path),
                "run_config_json": str(run_config_path),
                "training_csv": str(training_csv_path),
                "per_seed_csv": str(per_seed_csv_path),
                "summary_csv": str(summary_csv_path),
                "curve_summary_csv": str(curve_summary_csv_path),
                "console_log": str(args.console_log_path).strip(),
            },
        },
    }
    run_config_path.parent.mkdir(parents=True, exist_ok=True)
    with run_config_path.open("w", encoding="utf-8") as f:
        json.dump(run_config_payload, f, indent=int(args.json_indent))

    print(f"Saved JSON results to:    {output_path}")
    print(f"Saved run config JSON:    {run_config_path}")
    print(f"Saved training logs CSV:  {training_csv_path}")
    print(f"Saved per-seed metrics:   {per_seed_csv_path}")
    print(f"Saved aggregate summary:  {summary_csv_path}")
    print(f"Saved curve summary CSV:  {curve_summary_csv_path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic ablation of finite-shot noise on the (N=200, D=8) dataset by "
            "sweeping shot budgets alongside the analytic baseline."
        )
    )
    parser.add_argument("--N", "--n-samples", dest="n_samples", type=int, default=200)
    parser.add_argument("--D", "--n-features", dest="n_features", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--w-true-std", type=float, default=1.0)

    parser.add_argument("--prior-ratio", type=float, default=0.2)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--test-ratio", type=float, default=0.2)

    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated explicit seed list. Overrides --num-seeds/--seed-offset.",
    )

    parser.add_argument("--prior-bootstrap-samples", type=int, default=64)
    parser.add_argument("--prior-ridge", type=float, default=1e-6)
    parser.add_argument("--prior-sigma-floor", type=float, default=1e-4)
    parser.add_argument("--V-floor", type=float, default=1e-8)

    parser.set_defaults(include_analytic=True)
    parser.add_argument("--include-analytic", action="store_true")
    parser.add_argument("--no-analytic", dest="include_analytic", action="store_false")
    parser.add_argument(
        "--shots-list",
        type=str,
        default=None,
        help="Optional comma-separated finite-shot budgets. Default is 256,1024,4096,16384.",
    )

    parser.add_argument("--vqbr-reps", type=int, default=2)
    parser.add_argument("--vqbr-entanglement", type=str, default="linear")
    parser.add_argument("--vqbr-maxiter", type=int, default=300)
    parser.add_argument("--vqbr-cobyla-tol", type=float, default=1e-8)
    parser.add_argument(
        "--vqbr-batch-size",
        type=int,
        default=None,
        help="Batch size used by VQBR. Default uses the full training batch.",
    )
    parser.add_argument(
        "--vqbr-shots",
        type=int,
        default=2048,
        help="Placeholder shots value used when the analytic baseline constructs the VQBR object.",
    )
    parser.add_argument("--vqbr-loss", type=str, default="log_ratio")
    parser.add_argument("--vqbr-verbose", action="store_true")

    parser.set_defaults(log_training=True)
    parser.add_argument("--log-training", action="store_true")
    parser.add_argument("--no-log-training", dest="log_training", action="store_false")
    parser.add_argument("--log-every-iter", type=int, default=1)
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
        help=(
            "Root directory for auto-organized runs. "
            "Each run is saved under <results-root>/<run-tag>/."
        ),
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional run folder name. If omitted, a config-based tag is generated automatically.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Optional explicit run directory. Overrides --results-root/--run-tag when set.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="results.json",
        help="Results JSON filename (saved inside the run directory).",
    )
    parser.add_argument(
        "--training-csv-path",
        type=str,
        default="training_log.csv",
        help="Training CSV filename (saved inside the run directory).",
    )
    parser.add_argument(
        "--per-seed-csv-path",
        type=str,
        default="per_seed_metrics.csv",
        help="Per-seed CSV filename (saved inside the run directory).",
    )
    parser.add_argument(
        "--summary-csv-path",
        type=str,
        default="summary_metrics.csv",
        help="Summary CSV filename (saved inside the run directory).",
    )
    parser.add_argument(
        "--curve-summary-csv-path",
        type=str,
        default="curve_summary.csv",
        help="Curve-summary CSV filename (saved inside the run directory).",
    )
    parser.add_argument(
        "--run-config-path",
        type=str,
        default="run_config.json",
        help="Run-config JSON filename (saved inside the run directory).",
    )
    parser.add_argument(
        "--console-log-path",
        type=str,
        default="run.log",
        help=(
            "Console log filename (saved inside the run directory). "
            "Set empty string to disable file logging."
        ),
    )
    parser.add_argument("--json-indent", type=int, default=2)
    return parser


class _TeeStream:
    """Mirror writes to multiple stream-like objects."""

    def __init__(self, *streams: object):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    seeds = _resolve_seeds(
        num_seeds=int(args.num_seeds),
        seed_offset=int(args.seed_offset),
        explicit_seeds=args.seeds,
    )
    shot_budgets = _resolve_shot_budgets(args.shots_list)
    settings = _build_shot_settings(
        shot_budgets=shot_budgets,
        include_analytic=bool(args.include_analytic),
    )
    _resolve_run_paths(
        args=args,
        seeds=seeds,
        optimizer=ENFORCED_VQBR_OPTIMIZER,
        loss=ENFORCED_VQBR_LOSS,
        su2_gates=ENFORCED_VQBR_SU2_GATES,
        settings=settings,
    )

    console_log_raw = str(args.console_log_path).strip()
    if not console_log_raw:
        run_experiment(args)
        return

    console_log_path = Path(console_log_raw).expanduser().resolve()
    console_log_path.parent.mkdir(parents=True, exist_ok=True)
    with console_log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        tee_stdout = _TeeStream(sys.__stdout__, log_file)
        tee_stderr = _TeeStream(sys.__stderr__, log_file)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            print(f"Console log file: {console_log_path}")
            run_experiment(args)


if __name__ == "__main__":
    main()
