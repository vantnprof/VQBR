from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from openpyxl import load_workbook

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

# Make local method modules importable when running from repo root.
for module_dir in ("closed_form", "vqbr"):
    module_path = SRC_DIR / module_dir
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from closed_form import ClosedFormMAPBayesianRegression  # noqa: E402
from vqbr import VariationalQuantumBayesianRegression  # noqa: E402


EPS = 1e-12
TARGET_CHOICES = ("Y1", "Y2")
VQBR_METRIC_KEYS = (
    "cosine_similarity",
    "feature_cosine_similarity",
    "state_overlap_abs",
    "train_mse",
    "test_mse",
    "train_rmse",
    "test_rmse",
    "train_r2",
    "test_r2",
    "t_hat",
    "final_objective",
)
CLOSED_FORM_METRIC_KEYS = (
    "train_mse",
    "test_mse",
    "train_rmse",
    "test_rmse",
    "train_r2",
    "test_r2",
)

ENFORCED_VQBR_OPTIMIZER = "COBYLA"
ENFORCED_VQBR_LOSS = "log_ratio"
ENFORCED_VQBR_SU2_GATES: tuple[str, ...] = ("ry", "x")
LOG_RATIO_LOSS_FORMULA = "log(a_hat + d_hat + eps) - log((c_hat + e_hat)^2 + eps)"
DEFAULT_RESULTS_ROOT = ROOT_DIR / "results" / "energy"


@dataclass
class SplitIndices:
    prior_indices: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray


def _resolve_seeds(num_seeds: int, seed_offset: int, explicit_seeds: str | None) -> List[int]:
    if explicit_seeds is not None:
        tokens = [token.strip() for token in explicit_seeds.split(",") if token.strip()]
        if not tokens:
            raise ValueError("--seeds was provided but no valid integers were found.")
        return [int(token) for token in tokens]

    if num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    return [int(seed_offset) + i for i in range(int(num_seeds))]


def _parse_su2_gates(raw: str) -> tuple[str, ...]:
    tokens = [token.strip().lower() for token in str(raw).split(",") if token.strip()]
    if not tokens:
        raise ValueError("--vqbr-su2-gates must contain at least one gate token.")
    return tuple(tokens)


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=float).reshape(-1) - np.asarray(y_true, dtype=float).reshape(-1)
    return float(np.mean(err * err))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(max(_mse(y_true, y_pred), 0.0)))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= EPS:
        return 1.0 if ss_res <= EPS else 0.0
    return float(1.0 - (ss_res / ss_tot))


def _safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= eps and nb <= eps:
        return 1.0
    if na <= eps or nb <= eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pad_to_pow2(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec)
    if vec.ndim != 1:
        raise ValueError("Expected a 1D vector.")
    if vec.size == 0:
        raise ValueError("Vector must not be empty.")
    n = int(np.ceil(np.log2(vec.size)))
    m = 2**n
    if m == vec.size:
        return vec.copy()
    return np.pad(vec, (0, m - vec.size), mode="constant")


def _normalize_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(state))
    if norm <= EPS:
        raise RuntimeError("Zero-norm state encountered.")
    return state / norm


def _state_to_feature_direction(
    state: np.ndarray,
    n_features: int,
    eps: float = EPS,
) -> np.ndarray:
    state = np.asarray(state, dtype=complex).reshape(-1)
    feature_state = state[: int(n_features)]
    if feature_state.size != int(n_features):
        raise ValueError("n_features exceeds state dimension.")
    if feature_state.size == 0:
        return np.zeros(0, dtype=float)

    feature_norm = float(np.linalg.norm(feature_state))
    if feature_norm <= eps:
        return np.zeros(n_features, dtype=float)

    pivot = int(np.argmax(np.abs(feature_state)))
    phase = np.exp(-1j * np.angle(feature_state[pivot]))
    aligned = feature_state * phase
    direction_real = np.real_if_close(aligned, tol=1000)
    if np.iscomplexobj(direction_real):
        direction_real = np.real(aligned)

    direction = np.asarray(direction_real, dtype=float).reshape(-1)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= eps:
        return np.zeros(n_features, dtype=float)
    return direction / direction_norm


def _feature_direction_from_final_overlaps(
    vqbr_model: VariationalQuantumBayesianRegression,
    X_train: np.ndarray,
    eps: float = EPS,
) -> np.ndarray | None:
    """Recover feature direction from signed final overlaps r_i * s_i = x_i^T phi."""
    try:
        theta = np.asarray(getattr(vqbr_model, "theta_"), dtype=float).reshape(-1)
        n_samples = int(np.asarray(X_train, dtype=float).shape[0])
        if n_samples <= 0:
            return None

        overlaps = np.array(
            [
                float(
                    vqbr_model._estimate_overlap(
                        vqbr_model._Uxi_gates[i],
                        theta,
                        use_shot_noise=vqbr_model.use_shot_noise,
                    )
                )
                for i in range(n_samples)
            ],
            dtype=float,
        )
        row_norms = np.asarray(getattr(vqbr_model, "_row_norms"), dtype=float).reshape(-1)
        if row_norms.shape[0] != n_samples:
            return None

        z = row_norms * overlaps
        phi_ls, *_ = np.linalg.lstsq(np.asarray(X_train, dtype=float), z, rcond=None)
        phi_ls = np.asarray(phi_ls, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(phi_ls))
        if norm <= eps:
            return None
        return phi_ls / norm
    except Exception:
        return None


def _normalized_closed_form_state(w_closed: np.ndarray) -> np.ndarray:
    state = _pad_to_pow2(np.asarray(w_closed, dtype=complex).reshape(-1))
    norm = float(np.linalg.norm(state))
    if norm <= EPS:
        out = np.zeros_like(state, dtype=complex)
        out[0] = 1.0 + 0.0j
        return out
    return state / norm


def _maybe_subsample_rows(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples <= 0:
        return X, y

    n_samples = int(X.shape[0])
    if n_samples <= int(max_samples):
        return X, y

    rng = np.random.default_rng(seed + 31_415)
    selected = np.sort(rng.choice(n_samples, size=int(max_samples), replace=False))
    return X[selected], y[selected]


def _fit_feature_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D for feature standardization.")
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=0)
    std_safe = np.where(std > EPS, std, 1.0)
    return mean, std_safe


def _apply_feature_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D for feature standardization.")
    if X.shape[1] != mean.shape[0] or X.shape[1] != std.shape[0]:
        raise ValueError("Feature dimensions do not match fitted standardizer.")
    return (X - mean[None, :]) / std[None, :]


def _fit_target_standardizer(y: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=float).reshape(-1)
    mean = float(np.mean(y))
    std = float(np.std(y, ddof=0))
    std_safe = std if std > EPS else 1.0
    return mean, std_safe


def _apply_target_standardizer(y: np.ndarray, mean: float, std: float) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    return (y - float(mean)) / float(std)


def _inverse_target_standardizer(y_scaled: np.ndarray, mean: float, std: float) -> np.ndarray:
    y_scaled = np.asarray(y_scaled, dtype=float).reshape(-1)
    return float(mean) + float(std) * y_scaled


def _split_dataset_indices(
    n_samples: int,
    prior_ratio: float,
    train_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitIndices:
    if n_samples <= 2:
        raise ValueError("n_samples must be > 2.")

    ratio_sum = float(prior_ratio + train_ratio + test_ratio)
    if not np.isclose(ratio_sum, 1.0, atol=1e-12):
        raise ValueError(
            f"Split ratios must sum to 1.0. Got prior+train+test={ratio_sum:.12f}."
        )
    if prior_ratio <= 0.0 or train_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError("All split ratios must be positive.")

    n_prior = int(np.floor(prior_ratio * n_samples))
    n_train = int(np.floor(train_ratio * n_samples))
    n_test = int(n_samples - n_prior - n_train)
    if n_prior < 1 or n_train < 1 or n_test < 1:
        raise ValueError(
            "Invalid split sizes. Increase n_samples or adjust ratios so prior/train/test are non-empty."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    prior_indices = perm[:n_prior]
    train_indices = perm[n_prior : n_prior + n_train]
    test_indices = perm[n_prior + n_train :]
    return SplitIndices(
        prior_indices=prior_indices,
        train_indices=train_indices,
        test_indices=test_indices,
    )


def _ridge_solution(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    d = int(X.shape[1])
    gram = X.T @ X + float(ridge) * np.eye(d, dtype=float)
    rhs = X.T @ y
    return np.linalg.solve(gram, rhs)


def _estimate_prior_from_prior_split(
    X_prior: np.ndarray,
    y_prior: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int,
    ridge: float,
    sigma_floor: float,
    v_floor: float,
) -> tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
    if bootstrap_samples <= 0:
        raise ValueError("--prior-bootstrap-samples must be positive.")
    if ridge < 0.0:
        raise ValueError("--prior-ridge must be non-negative.")
    if sigma_floor <= 0.0:
        raise ValueError("--prior-sigma-floor must be positive.")
    if v_floor <= 0.0:
        raise ValueError("--V-floor must be positive.")

    X_prior = np.asarray(X_prior, dtype=float)
    y_prior = np.asarray(y_prior, dtype=float).reshape(-1)
    n_prior, d = X_prior.shape
    if y_prior.shape[0] != n_prior:
        raise ValueError("Prior split X/y shape mismatch.")

    rng = np.random.default_rng(seed + 77_777)
    weights = np.zeros((bootstrap_samples, d), dtype=float)
    for i in range(bootstrap_samples):
        idx = rng.integers(0, n_prior, size=n_prior)
        weights[i] = _ridge_solution(X_prior[idx], y_prior[idx], ridge=ridge)

    m0 = np.mean(weights, axis=0)
    sigma_diag = np.var(weights, axis=0, ddof=0)
    sigma_diag = np.maximum(sigma_diag, float(sigma_floor))
    Sigma0 = np.diag(sigma_diag)

    y_prior_pred = np.asarray(X_prior, dtype=float) @ np.asarray(m0, dtype=float)
    prior_fit_mse = _mse(y_prior, y_prior_pred)
    V = max(prior_fit_mse, float(v_floor))

    prior_info = {
        "bootstrap_samples": float(bootstrap_samples),
        "ridge": float(ridge),
        "sigma_floor": float(sigma_floor),
        "v_floor": float(v_floor),
        "V": float(V),
        "prior_fit_mse": float(prior_fit_mse),
        "mean_sigma0_diag": float(np.mean(sigma_diag)),
    }
    return m0, Sigma0, float(V), prior_info


def _build_training_logger(
    *,
    seed: int,
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
                f"    [seed={seed}] eval {iteration:>4}/{maxiter:<4} | "
                f"objective={current_loss:.6e}, {_format_delta(current_loss, prev_loss)}",
                flush=True,
            )
        prev_loss = current_loss

    return _callback


def _extract_vqbr_history(
    seed: int,
    history: Sequence[Any],
) -> tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    rows: List[Dict[str, Any]] = []
    series: Dict[str, List[float]] = {
        "objective_history": [],
        "L_tilde_history": [],
        "full_L_tilde_history": [],
        "a_hat_history": [],
        "c_hat_history": [],
        "d_hat_history": [],
        "e_hat_history": [],
        "h_hat_history": [],
    }
    for iteration, snapshot in enumerate(history, start=1):
        batch_loss = float(getattr(snapshot, "batch_L_tilde", np.nan))
        if not np.isfinite(batch_loss):
            batch_loss = float(getattr(snapshot, "L_tilde", np.nan))
        L_tilde = float(getattr(snapshot, "L_tilde", np.nan))
        full_L_tilde = float(getattr(snapshot, "full_L_tilde", np.nan))
        a_hat = float(getattr(snapshot, "a_hat", np.nan))
        c_hat = float(getattr(snapshot, "c_hat", np.nan))
        d_hat = float(getattr(snapshot, "d_hat", np.nan))
        e_hat = float(getattr(snapshot, "e_hat", np.nan))
        h_hat = float(getattr(snapshot, "h_hat", np.nan))
        epoch = int(getattr(snapshot, "epoch", 0))
        batch_position = int(getattr(snapshot, "batch_position", 0))
        batch_indices = np.asarray(getattr(snapshot, "batch_indices", []), dtype=int).reshape(-1)
        batch_size = int(batch_indices.size)

        rows.append(
            {
                "seed": int(seed),
                "objective_eval": int(iteration),
                "batch_loss": float(batch_loss),
                "L_tilde": float(L_tilde),
                "full_L_tilde": float(full_L_tilde),
                "a_hat": float(a_hat),
                "c_hat": float(c_hat),
                "d_hat": float(d_hat),
                "e_hat": float(e_hat),
                "h_hat": float(h_hat),
                "epoch": int(epoch),
                "batch_position": int(batch_position),
                "batch_size": int(batch_size),
            }
        )
        series["objective_history"].append(float(batch_loss))
        series["L_tilde_history"].append(float(L_tilde))
        series["full_L_tilde_history"].append(float(full_L_tilde))
        series["a_hat_history"].append(float(a_hat))
        series["c_hat_history"].append(float(c_hat))
        series["d_hat_history"].append(float(d_hat))
        series["e_hat_history"].append(float(e_hat))
        series["h_hat_history"].append(float(h_hat))
    return rows, series


def _snapshot_objective(snapshot: Any) -> float:
    objective_value = float(getattr(snapshot, "batch_L_tilde", np.nan))
    if not np.isfinite(objective_value):
        objective_value = float(getattr(snapshot, "L_tilde", np.nan))
    return objective_value


def _select_snapshot_for_solution(
    history: Sequence[Any],
    target_objective: float | None = None,
) -> Any | None:
    if not history:
        return None
    target = float(target_objective) if target_objective is not None else float("nan")
    candidates: List[tuple[Any, float]] = []
    for snapshot in history:
        objective_value = _snapshot_objective(snapshot)
        a_hat = float(getattr(snapshot, "a_hat", np.nan))
        c_hat = float(getattr(snapshot, "c_hat", np.nan))
        d_hat = float(getattr(snapshot, "d_hat", np.nan))
        e_hat = float(getattr(snapshot, "e_hat", np.nan))
        if (
            np.isfinite(objective_value)
            and np.isfinite(a_hat)
            and np.isfinite(c_hat)
            and np.isfinite(d_hat)
            and np.isfinite(e_hat)
        ):
            candidates.append((snapshot, float(objective_value)))

    if candidates:
        if np.isfinite(target):
            best_snapshot, _ = min(
                candidates,
                key=lambda item: abs(item[1] - target),
            )
            return best_snapshot
        best_snapshot, _ = min(candidates, key=lambda item: item[1])
        return best_snapshot

    for snapshot in reversed(history):
        a_hat = float(getattr(snapshot, "a_hat", np.nan))
        c_hat = float(getattr(snapshot, "c_hat", np.nan))
        d_hat = float(getattr(snapshot, "d_hat", np.nan))
        e_hat = float(getattr(snapshot, "e_hat", np.nan))
        if np.isfinite(a_hat) and np.isfinite(c_hat) and np.isfinite(d_hat) and np.isfinite(e_hat):
            return snapshot
    return None


def _snapshot_at_final_theta(
    vqbr_model: VariationalQuantumBayesianRegression,
    n_samples: int,
) -> Any | None:
    """Evaluate objective terms at final optimizer parameters (result.x/theta_)."""
    try:
        theta = np.asarray(getattr(vqbr_model, "theta_"), dtype=float).reshape(-1)
        batch_indices = np.arange(int(n_samples), dtype=int)
        return vqbr_model._objective_on_batch(
            theta,
            batch_indices,
            use_shot_noise=vqbr_model.use_shot_noise,
            scale_batch_to_full=False,
        )
    except Exception:
        return None


def _reconstruct_weights_from_formula(
    phi: np.ndarray,
    final_snapshot: Any | None,
) -> tuple[float, np.ndarray, Dict[str, float]]:
    phi = np.asarray(phi, dtype=float).reshape(-1)
    if final_snapshot is None:
        terms = {
            "a_hat": float("nan"),
            "c_hat": float("nan"),
            "d_hat": float("nan"),
            "e_hat": float("nan"),
            "numerator": float("nan"),
            "denominator": float("nan"),
        }
        return float("nan"), np.full(phi.shape, np.nan, dtype=float), terms

    a_hat = float(getattr(final_snapshot, "a_hat", np.nan))
    c_hat = float(getattr(final_snapshot, "c_hat", np.nan))
    d_hat = float(getattr(final_snapshot, "d_hat", np.nan))
    e_hat = float(getattr(final_snapshot, "e_hat", np.nan))
    numerator = float(c_hat + e_hat)
    denominator = float(a_hat + d_hat)

    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or abs(denominator) <= EPS
    ):
        t_hat = float("nan")
        w_hat = np.full(phi.shape, np.nan, dtype=float)
    else:
        t_hat = float(numerator / denominator)
        w_hat = float(t_hat) * phi

    terms = {
        "a_hat": float(a_hat),
        "c_hat": float(c_hat),
        "d_hat": float(d_hat),
        "e_hat": float(e_hat),
        "numerator": float(numerator),
        "denominator": float(denominator),
    }
    return t_hat, w_hat, terms


def _predict_raw_from_standardized(
    X_standardized: np.ndarray,
    w_standardized: np.ndarray,
    y_train_mean: float,
    y_train_std: float,
) -> np.ndarray:
    X_standardized = np.asarray(X_standardized, dtype=float)
    w_standardized = np.asarray(w_standardized, dtype=float).reshape(-1)
    pred_standardized = X_standardized @ w_standardized
    return _inverse_target_standardizer(pred_standardized, y_train_mean, y_train_std)


def _evaluate_regression_metrics(
    *,
    X_train_standardized: np.ndarray,
    y_train_raw: np.ndarray,
    X_test_standardized: np.ndarray,
    y_test_raw: np.ndarray,
    w_standardized: np.ndarray,
    y_train_mean: float,
    y_train_std: float,
) -> Dict[str, float]:
    w_standardized = np.asarray(w_standardized, dtype=float).reshape(-1)
    if w_standardized.size == 0 or not np.all(np.isfinite(w_standardized)):
        return {
            "train_mse": float("nan"),
            "test_mse": float("nan"),
            "train_rmse": float("nan"),
            "test_rmse": float("nan"),
            "train_r2": float("nan"),
            "test_r2": float("nan"),
        }

    train_pred = _predict_raw_from_standardized(
        X_standardized=X_train_standardized,
        w_standardized=w_standardized,
        y_train_mean=y_train_mean,
        y_train_std=y_train_std,
    )
    test_pred = _predict_raw_from_standardized(
        X_standardized=X_test_standardized,
        w_standardized=w_standardized,
        y_train_mean=y_train_mean,
        y_train_std=y_train_std,
    )
    return {
        "train_mse": _mse(y_train_raw, train_pred),
        "test_mse": _mse(y_test_raw, test_pred),
        "train_rmse": _rmse(y_train_raw, train_pred),
        "test_rmse": _rmse(y_test_raw, test_pred),
        "train_r2": _r2_score(y_train_raw, train_pred),
        "test_r2": _r2_score(y_test_raw, test_pred),
    }


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n_valid": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "n_valid": float(arr.size),
    }


def _aggregate_results(per_seed_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {
        "vqbr": {},
        "closed_form": {},
        "runtime_seconds": {},
        "iterations": {},
    }

    for metric_key in VQBR_METRIC_KEYS:
        values = [
            float(seed_result["methods"]["vqbr"].get(metric_key, np.nan))
            for seed_result in per_seed_results
        ]
        aggregate["vqbr"][metric_key] = _mean_std(values)

    for metric_key in CLOSED_FORM_METRIC_KEYS:
        values = [
            float(seed_result["methods"]["closed_form"].get(metric_key, np.nan))
            for seed_result in per_seed_results
        ]
        aggregate["closed_form"][metric_key] = _mean_std(values)

    for runtime_key in ("prior_estimation", "closed_form_fit", "vqbr_fit", "total"):
        values = [
            float(seed_result["runtime_seconds"].get(runtime_key, np.nan))
            for seed_result in per_seed_results
        ]
        aggregate["runtime_seconds"][runtime_key] = _mean_std(values)

    iteration_aggregate_keys = (
        ("vqbr_objective_evaluations", "objective_evaluations"),
        ("vqbr_optimizer_nfev", "optimizer_nfev"),
        ("vqbr_optimizer_iterations", "optimizer_iterations"),
    )
    for aggregate_key, metric_key in iteration_aggregate_keys:
        values = [
            float(seed_result["methods"]["vqbr"].get(metric_key, np.nan))
            for seed_result in per_seed_results
        ]
        aggregate["iterations"][aggregate_key] = _mean_std(values)
    return aggregate


def _write_training_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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


def _write_per_seed_metrics_csv(path: Path, per_seed_results: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "method",
        "cosine_similarity",
        "feature_cosine_similarity",
        "state_overlap_abs",
        "train_mse",
        "test_mse",
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
        for seed_result in per_seed_results:
            seed = int(seed_result["seed"])
            runtime_seed_total = float(seed_result["runtime_seconds"].get("total", np.nan))
            runtime_vqbr = float(seed_result["runtime_seconds"].get("vqbr_fit", np.nan))
            runtime_closed_form = float(seed_result["runtime_seconds"].get("closed_form_fit", np.nan))

            vqbr_metrics = dict(seed_result["methods"]["vqbr"])
            writer.writerow(
                {
                    "seed": seed,
                    "method": "vqbr",
                    "cosine_similarity": float(vqbr_metrics.get("cosine_similarity", np.nan)),
                    "feature_cosine_similarity": float(
                        vqbr_metrics.get("feature_cosine_similarity", np.nan)
                    ),
                    "state_overlap_abs": float(vqbr_metrics.get("state_overlap_abs", np.nan)),
                    "train_mse": float(vqbr_metrics.get("train_mse", np.nan)),
                    "test_mse": float(vqbr_metrics.get("test_mse", np.nan)),
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
                    "seed": seed,
                    "method": "closed_form",
                    "cosine_similarity": float("nan"),
                    "feature_cosine_similarity": float("nan"),
                    "state_overlap_abs": float("nan"),
                    "train_mse": float(closed_form_metrics.get("train_mse", np.nan)),
                    "test_mse": float(closed_form_metrics.get("test_mse", np.nan)),
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


def _write_summary_csv(path: Path, aggregate: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "method", "metric", "mean", "std", "n_valid"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in ("vqbr", "closed_form"):
            for metric, stats in aggregate[method].items():
                writer.writerow(
                    {
                        "category": "method_metric",
                        "method": method,
                        "metric": metric,
                        "mean": float(stats.get("mean", np.nan)),
                        "std": float(stats.get("std", np.nan)),
                        "n_valid": int(stats.get("n_valid", 0.0)),
                    }
                )
        for metric, stats in aggregate["runtime_seconds"].items():
            writer.writerow(
                {
                    "category": "runtime_seconds",
                    "method": "runtime",
                    "metric": metric,
                    "mean": float(stats.get("mean", np.nan)),
                    "std": float(stats.get("std", np.nan)),
                    "n_valid": int(stats.get("n_valid", 0.0)),
                }
            )
        for metric, stats in aggregate["iterations"].items():
            writer.writerow(
                {
                    "category": "iterations",
                    "method": "vqbr",
                    "metric": metric,
                    "mean": float(stats.get("mean", np.nan)),
                    "std": float(stats.get("std", np.nan)),
                    "n_valid": int(stats.get("n_valid", 0.0)),
                }
            )


def _format_mean_std(stats: Dict[str, float], *, scientific: bool) -> str:
    mean = float(stats.get("mean", np.nan))
    std = float(stats.get("std", np.nan))
    if not np.isfinite(mean) or not np.isfinite(std):
        return "nan +/- nan"
    if scientific:
        return f"{mean:.6e} +/- {std:.6e}"
    return f"{mean:.6f} +/- {std:.6f}"


def _slugify_token(raw: str) -> str:
    out: List[str] = []
    for ch in str(raw):
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("-")
    slug = "".join(out).strip("-_")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "na"


def _default_run_tag(
    *,
    args: argparse.Namespace,
    seeds: Sequence[int],
    optimizer: str,
    loss: str,
    use_shot_noise: bool,
    su2_gates: Sequence[str],
) -> str:
    mode = "shot" if bool(use_shot_noise) else "analytic"
    gates_tag = "-".join(_slugify_token(g) for g in su2_gates)
    ent_tag = _slugify_token(str(args.vqbr_entanglement))
    dataset_tag = _slugify_token(Path(str(args.dataset_path)).stem)
    target_tag = _slugify_token(str(args.target_col).upper())
    max_samples_tag = (
        f"maxs{int(args.max_samples)}"
        if args.max_samples is not None and int(args.max_samples) > 0
        else "maxsall"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"energy_{dataset_tag}_{target_tag}_{max_samples_tag}_"
        f"seeds{int(len(seeds))}_maxiter{int(args.vqbr_maxiter)}_"
        f"{str(optimizer).lower()}_{_slugify_token(loss)}_{mode}_"
        f"reps{int(args.vqbr_reps)}_gates{gates_tag}_ent{ent_tag}_{timestamp}"
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
    use_shot_noise: bool,
    su2_gates: Sequence[str],
) -> Dict[str, Any]:
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
            use_shot_noise=use_shot_noise,
            su2_gates=su2_gates,
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
    run_config_name = _to_filename(args.run_config_path, "run_config.json")
    console_log_name = (
        _to_filename(args.console_log_path, "run.log")
        if str(args.console_log_path).strip()
        else ""
    )
    seed_artifacts_dir = run_dir / "seed_artifacts"
    seed_artifacts_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        "run_tag": str(run_tag),
        "run_dir": str(run_dir),
        "results_json": str(run_dir / results_name),
        "training_csv": str(run_dir / training_name),
        "per_seed_csv": str(run_dir / per_seed_name),
        "summary_csv": str(run_dir / summary_name),
        "run_config_json": str(run_dir / run_config_name),
        "console_log": (str(run_dir / console_log_name) if console_log_name else ""),
        "seed_artifacts_dir": str(seed_artifacts_dir),
    }

    args.output_path = resolved["results_json"]
    args.training_csv_path = resolved["training_csv"]
    args.per_seed_csv_path = resolved["per_seed_csv"]
    args.summary_csv_path = resolved["summary_csv"]
    args.run_config_path = resolved["run_config_json"]
    args.console_log_path = resolved["console_log"]
    args.seed_artifacts_dir = resolved["seed_artifacts_dir"]
    args.resolved_run_tag = str(run_tag)
    args.resolved_run_dir = str(run_dir)
    args._resolved_run_paths = resolved
    return resolved


def load_seed_artifacts(npz_path: str | Path) -> Dict[str, np.ndarray]:
    """Load a saved per-seed artifact .npz into a plain dict."""
    path = Path(npz_path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def _save_seed_artifacts(
    *,
    seed_artifact_path: Path,
    seed: int,
    split: SplitIndices,
    X_prior_raw: np.ndarray,
    y_prior_raw: np.ndarray,
    X_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
    X_prior_std: np.ndarray,
    X_train_std: np.ndarray,
    X_test_std: np.ndarray,
    y_prior_std: np.ndarray,
    y_train_std: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    y_train_mean: float,
    y_train_std_value: float,
    m0: np.ndarray,
    Sigma0: np.ndarray,
    V: float,
    w_closed: np.ndarray,
    phi: np.ndarray,
    w_hat: np.ndarray,
    t_hat: float,
    theta_final: np.ndarray,
    vqbr_state: np.ndarray,
    objective_history: Sequence[float],
    a_hat_history: Sequence[float],
    c_hat_history: Sequence[float],
    d_hat_history: Sequence[float],
    e_hat_history: Sequence[float],
    h_hat_history: Sequence[float],
) -> None:
    seed_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        seed_artifact_path,
        seed=np.asarray([int(seed)], dtype=np.int64),
        split_prior_indices=np.asarray(split.prior_indices, dtype=np.int64),
        split_train_indices=np.asarray(split.train_indices, dtype=np.int64),
        split_test_indices=np.asarray(split.test_indices, dtype=np.int64),
        X_prior_raw=np.asarray(X_prior_raw, dtype=float),
        y_prior_raw=np.asarray(y_prior_raw, dtype=float),
        X_train_raw=np.asarray(X_train_raw, dtype=float),
        y_train_raw=np.asarray(y_train_raw, dtype=float),
        X_test_raw=np.asarray(X_test_raw, dtype=float),
        y_test_raw=np.asarray(y_test_raw, dtype=float),
        X_prior_standardized=np.asarray(X_prior_std, dtype=float),
        X_train_standardized=np.asarray(X_train_std, dtype=float),
        X_test_standardized=np.asarray(X_test_std, dtype=float),
        y_prior_standardized=np.asarray(y_prior_std, dtype=float),
        y_train_standardized=np.asarray(y_train_std, dtype=float),
        feature_mean=np.asarray(feature_mean, dtype=float),
        feature_std=np.asarray(feature_std, dtype=float),
        y_train_mean=np.asarray([float(y_train_mean)], dtype=float),
        y_train_std=np.asarray([float(y_train_std_value)], dtype=float),
        m0=np.asarray(m0, dtype=float),
        Sigma0=np.asarray(Sigma0, dtype=float),
        V=np.asarray([float(V)], dtype=float),
        w_closed=np.asarray(w_closed, dtype=float),
        phi=np.asarray(phi, dtype=float),
        w_hat=np.asarray(w_hat, dtype=float),
        t_hat=np.asarray([float(t_hat)], dtype=float),
        theta_final=np.asarray(theta_final, dtype=float),
        vqbr_state=np.asarray(vqbr_state, dtype=np.complex128),
        objective_history=np.asarray(objective_history, dtype=float),
        a_hat_history=np.asarray(a_hat_history, dtype=float),
        c_hat_history=np.asarray(c_hat_history, dtype=float),
        d_hat_history=np.asarray(d_hat_history, dtype=float),
        e_hat_history=np.asarray(e_hat_history, dtype=float),
        h_hat_history=np.asarray(h_hat_history, dtype=float),
    )


def _load_energy_dataset(
    dataset_path: Path,
    target_col: str,
    sheet_name: str | None,
    feature_cols_raw: str | None,
) -> tuple[np.ndarray, np.ndarray, List[str], str, int]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    target_col_normalized = str(target_col).strip().upper()
    if target_col_normalized not in TARGET_CHOICES:
        raise ValueError(
            f"Unsupported --target-col '{target_col}'. Supported values: {list(TARGET_CHOICES)}"
        )

    workbook = load_workbook(dataset_path, read_only=True, data_only=True)
    dropped_rows = 0
    try:
        candidate_sheet_names: List[str]
        if sheet_name is not None and str(sheet_name).strip() != "":
            requested_sheet = str(sheet_name)
            if requested_sheet not in workbook.sheetnames:
                raise ValueError(
                    f"Sheet '{requested_sheet}' not found in workbook. "
                    f"Available sheets: {workbook.sheetnames}"
                )
            candidate_sheet_names = [requested_sheet]
        else:
            candidate_sheet_names = list(workbook.sheetnames)

        selected_sheet_name = ""
        headers: List[str] = []
        matrix = np.empty((0, 0), dtype=float)

        for candidate_name in candidate_sheet_names:
            ws = workbook[candidate_name]
            header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None)
            if header_row is None:
                continue

            kept_col_indices: List[int] = []
            local_headers: List[str] = []
            for idx, raw_header in enumerate(header_row):
                if raw_header is None:
                    continue
                header = str(raw_header).strip()
                if not header:
                    continue
                local_headers.append(header)
                kept_col_indices.append(idx)

            if not local_headers:
                continue

            normalized_headers = [h.strip().upper() for h in local_headers]
            if target_col_normalized not in normalized_headers:
                continue

            rows: List[List[float]] = []
            for row in ws.iter_rows(min_row=2, values_only=True):
                values = [row[idx] if idx < len(row) else None for idx in kept_col_indices]
                if all(v is None or str(v).strip() == "" for v in values):
                    continue

                parsed_row: List[float] = []
                for value in values:
                    if value is None or str(value).strip() == "":
                        parsed_row.append(float("nan"))
                    else:
                        try:
                            parsed_row.append(float(value))
                        except Exception:
                            parsed_row.append(float("nan"))
                rows.append(parsed_row)

            if not rows:
                continue

            selected_sheet_name = candidate_name
            headers = local_headers
            matrix = np.asarray(rows, dtype=float)
            break

        if selected_sheet_name == "":
            if sheet_name:
                raise ValueError(
                    f"No numeric tabular data with target '{target_col_normalized}' found in sheet '{sheet_name}'."
                )
            raise ValueError(
                f"No sheet contains target column '{target_col_normalized}'. "
                f"Workbook sheets: {workbook.sheetnames}"
            )

        header_to_index = {
            str(header).strip().upper(): idx for idx, header in enumerate(headers)
        }
        target_index = int(header_to_index[target_col_normalized])

        if feature_cols_raw is None or str(feature_cols_raw).strip() == "":
            inferred_features = [
                h for h in headers if h.strip().upper().startswith("X") and h.strip().upper() != target_col_normalized
            ]
            if not inferred_features:
                inferred_features = [
                    h for h in headers if h.strip().upper() != target_col_normalized
                ]
            feature_cols = inferred_features
        else:
            requested = [token.strip() for token in str(feature_cols_raw).split(",") if token.strip()]
            if not requested:
                raise ValueError("--feature-cols was provided but no valid column names were found.")
            missing = [col for col in requested if col.strip().upper() not in header_to_index]
            if missing:
                raise ValueError(
                    f"Unknown feature columns: {missing}. Available headers: {headers}"
                )
            if any(col.strip().upper() == target_col_normalized for col in requested):
                raise ValueError("--feature-cols must not include the target column.")
            feature_cols = requested

        if not feature_cols:
            raise ValueError("No feature columns were selected.")

        feature_indices = [int(header_to_index[col.strip().upper()]) for col in feature_cols]
        X = matrix[:, feature_indices]
        y = matrix[:, target_index]

        finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        dropped_rows = int(np.size(finite_mask) - int(np.sum(finite_mask)))
        X = X[finite_mask]
        y = y[finite_mask]

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("Invalid dataset shape after parsing workbook.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Feature matrix and target vector have inconsistent lengths.")
        if X.shape[0] < 3:
            raise ValueError("Dataset must contain at least 3 usable rows.")

        return X, y, list(feature_cols), selected_sheet_name, dropped_rows
    finally:
        workbook.close()


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if args.vqbr_maxiter <= 0:
        raise ValueError("--vqbr-maxiter must be positive.")
    if args.vqbr_cobyla_tol <= 0.0:
        raise ValueError("--vqbr-cobyla-tol must be positive.")
    if args.vqbr_batch_size is not None and args.vqbr_batch_size <= 0:
        raise ValueError("--vqbr-batch-size must be positive when provided.")
    if args.log_every_iter <= 0:
        raise ValueError("--log-every-iter must be positive.")

    requested_su2_gates = _parse_su2_gates(args.vqbr_su2_gates)
    effective_su2_gates = ENFORCED_VQBR_SU2_GATES
    seeds = _resolve_seeds(
        num_seeds=int(args.num_seeds),
        seed_offset=int(args.seed_offset),
        explicit_seeds=args.seeds,
    )
    effective_optimizer = ENFORCED_VQBR_OPTIMIZER
    effective_loss = ENFORCED_VQBR_LOSS
    effective_use_shot_noise = bool(args.vqbr_use_shot_noise)
    run_paths = _resolve_run_paths(
        args=args,
        seeds=seeds,
        optimizer=effective_optimizer,
        loss=effective_loss,
        use_shot_noise=effective_use_shot_noise,
        su2_gates=effective_su2_gates,
    )
    effective_loss_formula = (
        LOG_RATIO_LOSS_FORMULA
        if effective_loss == "log_ratio"
        else "n/a"
    )
    if str(args.vqbr_loss).strip().lower() != effective_loss:
        print(
            f"Forcing VQBR loss to '{effective_loss}' (ignoring requested '{args.vqbr_loss}')."
        )
    print(
        "Using objective evaluation mode: "
        + ("finite-shot (shot noise enabled)." if effective_use_shot_noise else "analytic (no shot noise).")
    )
    if tuple(requested_su2_gates) != tuple(effective_su2_gates):
        print(
            f"Forcing VQBR su2_gates={effective_su2_gates} "
            f"(ignoring requested '{args.vqbr_su2_gates}')."
        )

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    X_all, y_all, feature_cols, selected_sheet_name, dropped_rows = _load_energy_dataset(
        dataset_path=dataset_path,
        target_col=args.target_col,
        sheet_name=args.sheet_name,
        feature_cols_raw=args.feature_cols,
    )
    n_samples_original = int(X_all.shape[0])
    sample_seed = int(seeds[0] if seeds else int(args.seed_offset))
    max_samples = None if args.max_samples is None or args.max_samples <= 0 else int(args.max_samples)
    X, y = _maybe_subsample_rows(X_all, y_all, max_samples=max_samples, seed=sample_seed)

    n_samples = int(X.shape[0])
    n_features = int(X.shape[1])
    if n_samples <= 2:
        raise ValueError("Need at least 3 samples after optional subsampling.")
    if n_features <= 0:
        raise ValueError("No feature columns are available.")

    print("=== Energy VQBR Experiment ===")
    print(f"Dataset: {dataset_path}")
    print(f"Sheet: {selected_sheet_name}")
    print(f"Target: {str(args.target_col).upper()}")
    print(f"Shape (used): N={n_samples}, D={n_features}")
    if max_samples is not None and n_samples < n_samples_original:
        print(
            f"Subsampling: using {n_samples}/{n_samples_original} rows "
            f"(max_samples={max_samples})"
        )
    if dropped_rows > 0:
        print(f"Dropped non-numeric or missing rows: {dropped_rows}")
    print(
        "Split ratios: "
        f"prior={float(args.prior_ratio):.3f}, "
        f"train={float(args.train_ratio):.3f}, "
        f"test={float(args.test_ratio):.3f}"
    )
    print(f"Seeds ({len(seeds)}): {', '.join(str(s) for s in seeds)}")
    print(
        f"VQBR config: optimizer={effective_optimizer}, "
        f"reps={int(args.vqbr_reps)}, su2_gates={effective_su2_gates}, "
        f"entanglement={args.vqbr_entanglement}, "
        f"maxiter={int(args.vqbr_maxiter)}, cobyla_tol={float(args.vqbr_cobyla_tol):.3e}, "
        f"loss={effective_loss}, use_shot_noise={effective_use_shot_noise}"
    )
    print(f"VQBR loss formula: {effective_loss_formula}")
    print(
        "Preprocessing: fit feature/target standardization on TRAIN split only, "
        "then apply the same transform to PRIOR and TEST."
    )
    print(f"Run directory: {run_paths['run_dir']}")

    training_rows: List[Dict[str, Any]] = []
    per_seed_results: List[Dict[str, Any]] = []
    seed_artifact_files: List[str] = []
    experiment_start = time.perf_counter()
    split_sizes_reference: Dict[str, int] | None = None

    for seed_idx, seed in enumerate(seeds, start=1):
        print("")
        print(f"[seed={seed}] ({seed_idx}/{len(seeds)})", flush=True)
        seed_start = time.perf_counter()

        split = _split_dataset_indices(
            n_samples=n_samples,
            prior_ratio=float(args.prior_ratio),
            train_ratio=float(args.train_ratio),
            test_ratio=float(args.test_ratio),
            seed=int(seed),
        )

        X_prior_raw = np.asarray(X[split.prior_indices], dtype=float)
        y_prior_raw = np.asarray(y[split.prior_indices], dtype=float)
        X_train_raw = np.asarray(X[split.train_indices], dtype=float)
        y_train_raw = np.asarray(y[split.train_indices], dtype=float)
        X_test_raw = np.asarray(X[split.test_indices], dtype=float)
        y_test_raw = np.asarray(y[split.test_indices], dtype=float)

        if split_sizes_reference is None:
            split_sizes_reference = {
                "prior": int(X_prior_raw.shape[0]),
                "train": int(X_train_raw.shape[0]),
                "test": int(X_test_raw.shape[0]),
            }

        feature_mean, feature_std = _fit_feature_standardizer(X_train_raw)
        y_train_mean, y_train_std = _fit_target_standardizer(y_train_raw)

        X_prior = _apply_feature_standardizer(X_prior_raw, feature_mean, feature_std)
        X_train = _apply_feature_standardizer(X_train_raw, feature_mean, feature_std)
        X_test = _apply_feature_standardizer(X_test_raw, feature_mean, feature_std)

        y_prior = _apply_target_standardizer(y_prior_raw, y_train_mean, y_train_std)
        y_train_std_values = _apply_target_standardizer(y_train_raw, y_train_mean, y_train_std)

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
        closed_form_model.fit(X_train, y_train_std_values, m0, Sigma0, V)
        w_closed = np.asarray(closed_form_model.get_weights(), dtype=float).reshape(-1)
        closed_form_runtime = float(time.perf_counter() - closed_form_start)
        closed_form_metrics = _evaluate_regression_metrics(
            X_train_standardized=X_train,
            y_train_raw=y_train_raw,
            X_test_standardized=X_test,
            y_test_raw=y_test_raw,
            w_standardized=w_closed,
            y_train_mean=y_train_mean,
            y_train_std=y_train_std,
        )
        print(
            "  closed-form: "
            f"train_rmse={closed_form_metrics['train_rmse']:.6e}, "
            f"test_rmse={closed_form_metrics['test_rmse']:.6e}, "
            f"train_R2={closed_form_metrics['train_r2']:.6f}, "
            f"test_R2={closed_form_metrics['test_r2']:.6f}"
        )

        batch_size = int(args.vqbr_batch_size) if args.vqbr_batch_size is not None else int(X_train.shape[0])
        batch_size = min(max(batch_size, 1), int(X_train.shape[0]))
        vqbr = VariationalQuantumBayesianRegression(
            shots=int(args.vqbr_shots),
            batch_size=batch_size,
            reps=int(args.vqbr_reps),
            optimizer=effective_optimizer,
            maxiter=int(args.vqbr_maxiter),
            cobyla_tol=float(args.vqbr_cobyla_tol),
            su2_gates=effective_su2_gates,
            entanglement=str(args.vqbr_entanglement),
            loss=effective_loss,
            random_state=int(seed),
            use_shot_noise=effective_use_shot_noise,
            verbose=bool(args.vqbr_verbose),
        )

        callback = None
        if bool(args.log_training):
            callback = _build_training_logger(
                seed=int(seed),
                maxiter=int(args.vqbr_maxiter),
                log_every_iter=int(args.log_every_iter),
            )

        vqbr_start = time.perf_counter()
        vqbr_state = np.asarray(
            vqbr.fit(
                X_train,
                y_train_std_values,
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
        training_rows.extend(history_rows)
        result_fun = float(getattr(vqbr.result_, "fun", np.nan))
        final_snapshot = _snapshot_at_final_theta(vqbr, n_samples=int(X_train.shape[0]))
        if final_snapshot is None:
            final_snapshot = _select_snapshot_for_solution(vqbr.history_, target_objective=result_fun)

        phi_from_overlaps = _feature_direction_from_final_overlaps(vqbr, X_train)
        if phi_from_overlaps is not None:
            phi = phi_from_overlaps
            phi_source = "overlap_lstsq"
        else:
            phi = _state_to_feature_direction(vqbr_state, n_features=n_features)
            phi_source = "statevector_phase_gauge"
        t_hat, w_hat, reconstruction_terms = _reconstruct_weights_from_formula(phi, final_snapshot)
        vqbr_regression_metrics = _evaluate_regression_metrics(
            X_train_standardized=X_train,
            y_train_raw=y_train_raw,
            X_test_standardized=X_test,
            y_test_raw=y_test_raw,
            w_standardized=w_hat,
            y_train_mean=y_train_mean,
            y_train_std=y_train_std,
        )

        w_closed_norm = float(np.linalg.norm(w_closed))
        w_closed_unit = (
            np.asarray(w_closed, dtype=float) / w_closed_norm
            if w_closed_norm > EPS
            else np.zeros_like(w_closed, dtype=float)
        )
        # Feature-direction cosine can differ by a global sign that is later absorbed by t_hat.
        # Use reconstructed-weight cosine as the primary signed similarity metric.
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
        theta_final = np.asarray(
            getattr(vqbr, "theta_", getattr(vqbr.result_, "x", np.array([], dtype=float))),
            dtype=float,
        ).reshape(-1)

        vqbr_metrics = {
            "cosine_similarity": float(cosine_similarity),
            "feature_cosine_similarity": float(feature_cosine_similarity),
            "state_overlap_abs": float(state_overlap_abs),
            "train_mse": float(vqbr_regression_metrics["train_mse"]),
            "test_mse": float(vqbr_regression_metrics["test_mse"]),
            "train_rmse": float(vqbr_regression_metrics["train_rmse"]),
            "test_rmse": float(vqbr_regression_metrics["test_rmse"]),
            "train_r2": float(vqbr_regression_metrics["train_r2"]),
            "test_r2": float(vqbr_regression_metrics["test_r2"]),
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
        }
        seed_artifact_path = Path(args.seed_artifacts_dir).expanduser().resolve() / f"seed_{int(seed):04d}.npz"
        _save_seed_artifacts(
            seed_artifact_path=seed_artifact_path,
            seed=int(seed),
            split=split,
            X_prior_raw=X_prior_raw,
            y_prior_raw=y_prior_raw,
            X_train_raw=X_train_raw,
            y_train_raw=y_train_raw,
            X_test_raw=X_test_raw,
            y_test_raw=y_test_raw,
            X_prior_std=X_prior,
            X_train_std=X_train,
            X_test_std=X_test,
            y_prior_std=y_prior,
            y_train_std=y_train_std_values,
            feature_mean=feature_mean,
            feature_std=feature_std,
            y_train_mean=y_train_mean,
            y_train_std_value=y_train_std,
            m0=m0,
            Sigma0=Sigma0,
            V=V,
            w_closed=w_closed,
            phi=phi,
            w_hat=w_hat,
            t_hat=t_hat,
            theta_final=theta_final,
            vqbr_state=vqbr_state,
            objective_history=objective_history,
            a_hat_history=history_series["a_hat_history"],
            c_hat_history=history_series["c_hat_history"],
            d_hat_history=history_series["d_hat_history"],
            e_hat_history=history_series["e_hat_history"],
            h_hat_history=history_series["h_hat_history"],
        )
        seed_artifact_files.append(str(seed_artifact_path))
        print(
            "  vqbr: "
            f"cos={vqbr_metrics['cosine_similarity']:.6f}, "
            f"feature_cos={vqbr_metrics['feature_cosine_similarity']:.6f}, "
            f"train_rmse={vqbr_metrics['train_rmse']:.6e}, "
            f"test_rmse={vqbr_metrics['test_rmse']:.6e}, "
            f"train_R2={vqbr_metrics['train_r2']:.6f}, "
            f"test_R2={vqbr_metrics['test_r2']:.6f}, "
            f"t_hat={vqbr_metrics['t_hat']:.6e}"
        )

        seed_runtime = float(time.perf_counter() - seed_start)
        runtime_info = {
            "prior_estimation": float(prior_runtime),
            "closed_form_fit": float(closed_form_runtime),
            "vqbr_fit": float(vqbr_runtime),
            "total": float(seed_runtime),
        }
        print(
            "  runtime (s): "
            f"prior={runtime_info['prior_estimation']:.3f}, "
            f"closed_form={runtime_info['closed_form_fit']:.3f}, "
            f"vqbr_fit={runtime_info['vqbr_fit']:.3f}, "
            f"total={runtime_info['total']:.3f}"
        )

        per_seed_results.append(
            {
                "seed": int(seed),
                "dataset": {
                    "n_samples": int(n_samples),
                    "n_features": int(n_features),
                    "target_col": str(args.target_col).upper(),
                },
                "split_sizes": {
                    "prior": int(X_prior.shape[0]),
                    "train": int(X_train.shape[0]),
                    "test": int(X_test.shape[0]),
                },
                "preprocessing": {
                    "feature_standardization": {
                        "fitted_on": "train_split_only",
                        "mean": [float(x) for x in feature_mean],
                        "std": [float(x) for x in feature_std],
                    },
                    "target_standardization": {
                        "fitted_on": "train_split_only",
                        "mean": float(y_train_mean),
                        "std": float(y_train_std),
                    },
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
                "artifacts": {
                    "seed_npz_path": str(seed_artifact_path),
                },
                "methods": {
                    "closed_form": {
                        "train_mse": float(closed_form_metrics["train_mse"]),
                        "test_mse": float(closed_form_metrics["test_mse"]),
                        "train_rmse": float(closed_form_metrics["train_rmse"]),
                        "test_rmse": float(closed_form_metrics["test_rmse"]),
                        "train_r2": float(closed_form_metrics["train_r2"]),
                        "test_r2": float(closed_form_metrics["test_r2"]),
                    },
                    "vqbr": vqbr_metrics,
                },
            }
        )


    aggregate = _aggregate_results(per_seed_results)
    total_runtime = float(time.perf_counter() - experiment_start)

    print("")
    print("=== Aggregate Results (mean +/- std over seeds) ===")
    print(
        "VQBR state cosine:      "
        f"{_format_mean_std(aggregate['vqbr']['cosine_similarity'], scientific=False)}"
    )
    print(
        "VQBR feature cosine:    "
        f"{_format_mean_std(aggregate['vqbr']['feature_cosine_similarity'], scientific=False)}"
    )
    print(f"VQBR train RMSE:        {_format_mean_std(aggregate['vqbr']['train_rmse'], scientific=True)}")
    print(f"VQBR test RMSE:         {_format_mean_std(aggregate['vqbr']['test_rmse'], scientific=True)}")
    print(f"VQBR train R^2:         {_format_mean_std(aggregate['vqbr']['train_r2'], scientific=False)}")
    print(f"VQBR test R^2:          {_format_mean_std(aggregate['vqbr']['test_r2'], scientific=False)}")
    print(f"Closed-form train RMSE: {_format_mean_std(aggregate['closed_form']['train_rmse'], scientific=True)}")
    print(f"Closed-form test RMSE:  {_format_mean_std(aggregate['closed_form']['test_rmse'], scientific=True)}")
    print(f"Closed-form train R^2:  {_format_mean_std(aggregate['closed_form']['train_r2'], scientific=False)}")
    print(f"Closed-form test R^2:   {_format_mean_std(aggregate['closed_form']['test_r2'], scientific=False)}")
    print(
        "VQBR fit runtime (s):   "
        f"{_format_mean_std(aggregate['runtime_seconds']['vqbr_fit'], scientific=True)}"
    )
    print(f"Total wall-clock runtime: {total_runtime:.3f} s")

    output_path = Path(args.output_path).expanduser().resolve()
    training_csv_path = Path(args.training_csv_path).expanduser().resolve()
    per_seed_csv_path = Path(args.per_seed_csv_path).expanduser().resolve()
    summary_csv_path = Path(args.summary_csv_path).expanduser().resolve()
    run_config_path = Path(args.run_config_path).expanduser().resolve()
    seed_artifacts_dir = Path(args.seed_artifacts_dir).expanduser().resolve()
    seed_manifest_path = seed_artifacts_dir / "seed_artifacts_manifest.json"

    _write_training_csv(training_csv_path, training_rows)
    _write_per_seed_metrics_csv(per_seed_csv_path, per_seed_results)
    _write_summary_csv(summary_csv_path, aggregate)
    with seed_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_tag": str(getattr(args, "resolved_run_tag", "")),
                "seed_artifacts_dir": str(seed_artifacts_dir),
                "seed_artifact_files": [str(path) for path in seed_artifact_files],
            },
            f,
            indent=int(args.json_indent),
        )

    payload: Dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": str(getattr(args, "resolved_run_tag", "")),
        "run_dir": str(getattr(args, "resolved_run_dir", "")),
        "dataset": {
            "path": str(dataset_path),
            "sheet_name": str(selected_sheet_name),
            "target_col": str(args.target_col).upper(),
            "feature_cols": [str(col) for col in feature_cols],
            "n_samples_original": int(n_samples_original),
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "subsampled": bool(n_samples < n_samples_original),
            "dropped_missing_or_non_numeric_rows": int(dropped_rows),
        },
        "config": {
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
                "shots": int(args.vqbr_shots),
                "use_shot_noise": effective_use_shot_noise,
                "loss": effective_loss,
                "loss_formula": effective_loss_formula,
            },
            "preprocessing": {
                "feature_standardization": True,
                "target_standardization": True,
                "standardization_scope": "fit on train split only, then apply to prior and test",
            },
            "dataset_subsampling": {
                "max_samples": None if max_samples is None else int(max_samples),
                "sample_seed": int(sample_seed),
            },
        },
        "split_sizes": split_sizes_reference if split_sizes_reference is not None else {},
        "per_seed": per_seed_results,
        "aggregate": aggregate,
        "runtime_seconds_total": float(total_runtime),
        "artifacts": {
            "run_dir": str(getattr(args, "resolved_run_dir", "")),
            "run_config_path": str(run_config_path),
            "training_csv_path": str(training_csv_path),
            "per_seed_csv_path": str(per_seed_csv_path),
            "summary_csv_path": str(summary_csv_path),
            "seed_artifacts_dir": str(seed_artifacts_dir),
            "seed_artifacts_manifest_path": str(seed_manifest_path),
            "seed_artifact_files": [str(path) for path in seed_artifact_files],
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
                "console_log": str(args.console_log_path).strip(),
                "seed_artifacts_dir": str(seed_artifacts_dir),
                "seed_artifacts_manifest": str(seed_manifest_path),
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
    print(f"Saved seed artifacts:     {seed_artifacts_dir}")
    print(f"Saved seed manifest:      {seed_manifest_path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Energy dataset VQBR experiment using prior/train/test split and train-only "
            "standardization applied consistently to prior and test."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(ROOT_DIR / "data" / "real" / "energy" / "ENB2012_data.xlsx"),
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        default=None,
        help="Optional worksheet name. Default auto-selects the first sheet containing the target column.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="Y1",
        choices=TARGET_CHOICES,
        help="Target column in the Energy dataset workbook.",
    )
    parser.add_argument(
        "--feature-cols",
        type=str,
        default=None,
        help="Optional comma-separated feature column names. Default uses X* columns.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional cap on dataset rows for faster runs. "
            "Set <= 0 or omit to use all rows."
        ),
    )

    parser.add_argument("--prior-ratio", type=float, default=0.2)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--test-ratio", type=float, default=0.2)

    parser.add_argument("--num-seeds", type=int, default=5)
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

    parser.add_argument("--vqbr-reps", type=int, default=2)
    parser.add_argument(
        "--vqbr-su2-gates",
        type=str,
        default="ry,x",
        help="Comma-separated SU2 rotation gates (e.g. 'ry' or 'rx,ry').",
    )
    parser.add_argument("--vqbr-entanglement", type=str, default="linear")
    parser.add_argument("--vqbr-maxiter", type=int, default=1000)
    parser.add_argument("--vqbr-cobyla-tol", type=float, default=1e-8)
    parser.add_argument(
        "--vqbr-batch-size",
        type=int,
        default=None,
        help="Batch size used by VQBR. Default uses full training batch.",
    )
    parser.add_argument("--vqbr-shots", type=int, default=2048)
    parser.add_argument("--vqbr-loss", type=str, default="log_ratio")
    parser.add_argument("--vqbr-verbose", action="store_true")

    parser.set_defaults(vqbr_use_shot_noise=False)
    parser.add_argument("--vqbr-use-shot-noise", dest="vqbr_use_shot_noise", action="store_true")
    parser.add_argument("--vqbr-no-shot-noise", dest="vqbr_use_shot_noise", action="store_false")

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
    _resolve_run_paths(
        args=args,
        seeds=seeds,
        optimizer=ENFORCED_VQBR_OPTIMIZER,
        loss=ENFORCED_VQBR_LOSS,
        use_shot_noise=bool(args.vqbr_use_shot_noise),
        su2_gates=ENFORCED_VQBR_SU2_GATES,
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
