from __future__ import annotations

import argparse
import csv
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

try:
    from .vqbr import VQBR
except ImportError:
    from vqbr import VQBR


EPS = 1e-12
ROOT_DIR = Path(__file__).resolve().parents[2]
PRIOR_RATIO = 0.20
TRAIN_RATIO = 0.60
TEST_RATIO = 0.20
METHOD_KEYS = ("closed_form", "vqbr")
METRIC_KEYS = (
    "cosine_similarity",
    "relative_l2_distance",
    "train_mse",
    "test_mse",
)


@dataclass
class SplitIndices:
    prior_indices: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class EvaluationMetrics:
    cosine_similarity: float
    relative_l2_distance: float
    train_mse: float
    test_mse: float


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=float).reshape(-1) - np.asarray(y_true, dtype=float).reshape(-1)
    return float(np.mean(err * err))


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


def _normalize_optimizer(raw: str) -> str:
    token = str(raw).strip().upper()
    valid = ("COBYLA", "SPSA")
    if token not in valid:
        raise ValueError(
            f"Unsupported optimizer '{raw}'. Supported values: {', '.join(valid)}."
        )
    return token


def _compute_split_sizes(n_samples: int) -> tuple[int, int, int]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    n_prior = int(np.floor(PRIOR_RATIO * n_samples))
    n_train = int(np.floor(TRAIN_RATIO * n_samples))
    n_test = int(np.floor(TEST_RATIO * n_samples))
    assigned = n_prior + n_train + n_test
    if assigned != n_samples:
        n_train += int(n_samples - assigned)

    if n_prior < 1 or n_train < 1 or n_test < 1:
        raise ValueError(
            "Invalid split sizes from 20/60/20. Increase total dataset size so prior/train/test "
            "are all non-empty."
        )
    return n_prior, n_train, n_test


def _split_dataset_indices(n_samples: int, seed: int) -> SplitIndices:
    n_prior, n_train, _ = _compute_split_sizes(n_samples)

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


def _resolve_seeds(base_seed: int, num_seeds: int, explicit_seeds: str | None) -> list[int]:
    if explicit_seeds is not None:
        tokens = [x.strip() for x in explicit_seeds.split(",") if x.strip()]
        if not tokens:
            raise ValueError("--seeds was provided but no valid integers were found.")
        return [int(x) for x in tokens]

    if num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    return [int(base_seed) + i for i in range(int(num_seeds))]


def _resolve_train_sizes(
    n_features: int,
    n_train_pool: int,
    min_multiplier: int,
    max_multiplier: int,
    step_multiplier: int,
) -> list[int]:
    if n_features <= 0:
        raise ValueError("--D/--n-features must be positive.")
    if min_multiplier <= 0:
        raise ValueError("--train-min-multiplier must be positive.")
    if max_multiplier < min_multiplier:
        raise ValueError("--train-max-multiplier must be >= --train-min-multiplier.")
    if step_multiplier <= 0:
        raise ValueError("--train-step-multiplier must be positive.")

    train_sizes = [
        int(mult * n_features)
        for mult in range(min_multiplier, max_multiplier + 1, step_multiplier)
    ]
    if not train_sizes:
        raise ValueError("No train sizes were produced from the provided multipliers.")
    if train_sizes[-1] > n_train_pool:
        raise ValueError(
            f"Requested max train size {train_sizes[-1]} exceeds train-pool size {n_train_pool}. "
            "Reduce --train-max-multiplier or increase --dataset-multiplier."
        )
    return train_sizes


def _pad_to_pow2(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=complex).reshape(-1)
    n = int(np.ceil(np.log2(vec.size)))
    target = 2**n
    if target == vec.size:
        return vec.copy()
    return np.pad(vec, (0, target - vec.size), mode="constant")


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
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
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
    V = max(_mse(y_prior, y_prior_pred), float(v_floor))

    prior_info = {
        "bootstrap_samples": float(bootstrap_samples),
        "ridge": float(ridge),
        "sigma_floor": float(sigma_floor),
        "V_floor": float(v_floor),
        "prior_fit_mse": float(_mse(y_prior, y_prior_pred)),
    }
    return m0, Sigma0, float(V), prior_info


def _closed_form_map_solution(
    X: np.ndarray,
    y: np.ndarray,
    m0: np.ndarray,
    Sigma: np.ndarray,
    V: float,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    m0 = np.asarray(m0, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)

    if Sigma.ndim == 1:
        sigma_inv = np.diag(1.0 / Sigma)
    else:
        sigma_inv = np.linalg.inv(Sigma)

    gram = X.T @ X + float(V) * sigma_inv
    rhs = X.T @ y + float(V) * (sigma_inv @ m0)
    return np.linalg.solve(gram, rhs)


def _reconstruct_vector_from_state(
    quantum_state: np.ndarray,
    w_reference: np.ndarray,
    n_features: int,
) -> np.ndarray:
    state = np.asarray(quantum_state, dtype=complex).reshape(-1)
    state_norm = float(np.linalg.norm(state))
    if state_norm <= EPS:
        raise ValueError("Quantum state has zero norm.")
    state = state / state_norm

    w_ref = np.asarray(w_reference, dtype=float).reshape(-1)
    w_ref_norm = float(np.linalg.norm(w_ref))
    if w_ref_norm <= EPS:
        raise ValueError("Reference vector has zero norm; cannot reconstruct direction.")
    w_dir = w_ref / w_ref_norm

    w_dir_pad = _pad_to_pow2(w_dir)
    w_dir_pad = w_dir_pad / max(float(np.linalg.norm(w_dir_pad)), EPS)
    overlap = np.vdot(w_dir_pad, state)
    phase = np.exp(-1j * np.angle(overlap)) if np.abs(overlap) > EPS else 1.0 + 0.0j
    state_aligned = state * phase

    feature_state = np.asarray(state_aligned[:n_features], dtype=complex)
    feature_state_real = np.real_if_close(feature_state, tol=1000)
    if np.iscomplexobj(feature_state_real):
        feature_state_real = np.real(feature_state)
    feature_state_real = np.asarray(feature_state_real, dtype=float).reshape(-1)

    feature_norm = float(np.linalg.norm(feature_state_real))
    if feature_norm <= EPS:
        return np.zeros_like(w_ref, dtype=float)

    direction = feature_state_real / feature_norm
    if float(np.dot(direction, w_dir)) < 0.0:
        direction = -direction
    return float(w_ref_norm) * direction


def _build_training_logger(maxiter: int, log_every_iter: int):
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

    def _callback(iteration: int, snapshot: object) -> None:
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
                f"    iter {iteration:>4}/{maxiter:<4} | "
                f"loss={current_loss:.6e}, {_format_delta(current_loss, prev_loss)}",
                flush=True,
            )
        prev_loss = current_loss

    return _callback


def _make_synthetic_dataset(
    n_samples: int,
    n_features: int,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_w = rng.normal(0.0, 1.0, size=n_features)
    X = rng.normal(5.0, 1.0, size=(n_samples, n_features))
    y = X @ true_w + rng.normal(0.0, noise_std, size=n_samples)
    return X, y, true_w


def _safe_mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def _format_mean_std(mean: float, std: float, scientific: bool = True) -> str:
    if not np.isfinite(mean) or not np.isfinite(std):
        return "nan +/- nan"
    if scientific:
        return f"{mean:.6e} +/- {std:.6e}"
    return f"{mean:.6f} +/- {std:.6f}"


def _aggregate_across_seeds(
    per_seed_results: dict[int, dict[int, dict[str, EvaluationMetrics]]],
    train_sizes: Sequence[int],
) -> dict[int, dict[str, dict[str, dict[str, float]]]]:
    summary: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
    seed_list = sorted(per_seed_results.keys())
    for n_train in train_sizes:
        summary[n_train] = {}
        for method in METHOD_KEYS:
            summary[n_train][method] = {}
            for metric_key in METRIC_KEYS:
                values = [
                    getattr(per_seed_results[seed][n_train][method], metric_key)
                    for seed in seed_list
                ]
                mean, std = _safe_mean_std(values)
                summary[n_train][method][metric_key] = {"mean": mean, "std": std}
    return summary


def _metrics_to_dict(metrics: EvaluationMetrics) -> dict[str, float]:
    return {
        "cosine_similarity": float(metrics.cosine_similarity),
        "relative_l2_distance": float(metrics.relative_l2_distance),
        "train_mse": float(metrics.train_mse),
        "test_mse": float(metrics.test_mse),
    }


def _write_summary_csv(
    summary_csv_path: Path,
    summary: dict[int, dict[str, dict[str, dict[str, float]]]],
    train_sizes: Sequence[int],
) -> None:
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "N",
                "method",
                "train_mse_mean",
                "train_mse_std",
                "test_mse_mean",
                "test_mse_std",
                "cosine_similarity_mean",
                "cosine_similarity_std",
                "relative_l2_distance_mean",
                "relative_l2_distance_std",
            ]
        )
        for n_train in train_sizes:
            for method in METHOD_KEYS:
                row = summary[n_train][method]
                writer.writerow(
                    [
                        int(n_train),
                        str(method),
                        float(row["train_mse"]["mean"]),
                        float(row["train_mse"]["std"]),
                        float(row["test_mse"]["mean"]),
                        float(row["test_mse"]["std"]),
                        float(row["cosine_similarity"]["mean"]),
                        float(row["cosine_similarity"]["std"]),
                        float(row["relative_l2_distance"]["mean"]),
                        float(row["relative_l2_distance"]["std"]),
                    ]
                )


def _save_results(
    *,
    output_path: Path,
    summary_csv_path: Path,
    json_indent: int,
    args: argparse.Namespace,
    seeds: Sequence[int],
    train_sizes: Sequence[int],
    n_features: int,
    n_total: int,
    n_prior: int,
    n_train_pool: int,
    n_test: int,
    per_seed_results: dict[int, dict[int, dict[str, EvaluationMetrics]]],
    per_seed_prior_info: dict[int, dict[str, float]],
    summary: dict[int, dict[str, dict[str, dict[str, float]]]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_seed_payload: dict[str, dict[str, object]] = {}
    for seed in seeds:
        by_n: dict[str, dict[str, dict[str, float]]] = {}
        for n_train in train_sizes:
            by_n[str(int(n_train))] = {
                method: _metrics_to_dict(per_seed_results[int(seed)][int(n_train)][method])
                for method in METHOD_KEYS
            }
        per_seed_payload[str(int(seed))] = {
            "prior_estimate": dict(per_seed_prior_info[int(seed)]),
            "metrics_by_train_size": by_n,
        }

    aggregate_payload: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for n_train in train_sizes:
        aggregate_payload[str(int(n_train))] = {
            method: {
                metric: {
                    "mean": float(summary[int(n_train)][method][metric]["mean"]),
                    "std": float(summary[int(n_train)][method][metric]["std"]),
                }
                for metric in METRIC_KEYS
            }
            for method in METHOD_KEYS
        }

    payload: dict[str, object] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "D": int(n_features),
            "dataset_multiplier": int(args.dataset_multiplier),
            "noise_std": float(args.noise_std),
            "seed_mode": "explicit_list" if args.seeds is not None else "base_plus_offsets",
            "seeds": [int(s) for s in seeds],
            "num_seeds": int(len(seeds)),
            "train_min_multiplier": int(args.train_min_multiplier),
            "train_max_multiplier": int(args.train_max_multiplier),
            "train_step_multiplier": int(args.train_step_multiplier),
            "train_sizes": [int(x) for x in train_sizes],
            "prior_estimation": {
                "bootstrap_samples": int(args.prior_bootstrap_samples),
                "ridge": float(args.prior_ridge),
                "sigma_floor": float(args.prior_sigma_floor),
                "V_floor": float(args.V_floor),
            },
            "vqbr": {
                "optimizer": str(args.optimizer).upper(),
                "reps": int(args.reps),
                "batch_size": None if args.batch_size is None else int(args.batch_size),
                "maxiter": int(args.maxiter),
                "shots": int(args.shots),
                "use_shot_noise": bool(args.use_shot_noise),
            },
        },
        "split": {
            "ratios": {
                "prior": float(PRIOR_RATIO),
                "train": float(TRAIN_RATIO),
                "test": float(TEST_RATIO),
            },
            "sizes": {
                "total": int(n_total),
                "prior": int(n_prior),
                "train_pool": int(n_train_pool),
                "test": int(n_test),
            },
        },
        "per_seed": per_seed_payload,
        "aggregate": aggregate_payload,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=json_indent)
    _write_summary_csv(summary_csv_path, summary, train_sizes)

    print(f"Saved JSON results to: {output_path}")
    print(f"Saved summary CSV to:  {summary_csv_path}")


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


def _resolve_feature_dimensions(n_features: int, n_features_list: str | None) -> list[int]:
    if n_features_list is None:
        if int(n_features) <= 0:
            raise ValueError("--D/--n-features must be positive.")
        return [int(n_features)]

    tokens = [x.strip() for x in str(n_features_list).split(",") if x.strip()]
    if not tokens:
        raise ValueError("--Ds was provided but no valid integers were found.")
    dims = [int(x) for x in tokens]
    if any(d <= 0 for d in dims):
        raise ValueError("All values in --Ds must be positive integers.")

    unique_dims: list[int] = []
    for d in dims:
        if d not in unique_dims:
            unique_dims.append(int(d))
    return unique_dims


def _resolve_output_path_for_dimension(
    raw_path: str,
    n_features: int,
    multi_dimension_mode: bool,
) -> Path:
    raw = str(raw_path).strip()
    if "{D}" in raw:
        return Path(raw.replace("{D}", str(int(n_features)))).expanduser().resolve()

    base = Path(raw).expanduser().resolve()
    if not multi_dimension_mode:
        return base

    if base.suffix:
        suffixed_name = f"{base.stem}_D{int(n_features)}{base.suffix}"
    else:
        suffixed_name = f"{base.name}_D{int(n_features)}"
    return base.with_name(suffixed_name)


def _run_experiment_for_dimension(
    args: argparse.Namespace,
    *,
    n_features: int,
    output_path: Path,
    summary_csv_path: Path,
) -> None:
    optimizer_name = _normalize_optimizer(args.optimizer)
    if args.shots <= 0:
        raise ValueError("--shots must be a positive integer.")
    if args.maxiter <= 0:
        raise ValueError("--maxiter must be a positive integer.")
    if args.dataset_multiplier <= 0:
        raise ValueError("--dataset-multiplier must be a positive integer.")

    seeds = _resolve_seeds(
        base_seed=int(args.seed),
        num_seeds=int(args.num_seeds),
        explicit_seeds=args.seeds,
    )
    n_total = int(args.dataset_multiplier * n_features)
    n_prior, n_train_pool, n_test = _compute_split_sizes(n_total)
    train_sizes = _resolve_train_sizes(
        n_features=n_features,
        n_train_pool=n_train_pool,
        min_multiplier=int(args.train_min_multiplier),
        max_multiplier=int(args.train_max_multiplier),
        step_multiplier=int(args.train_step_multiplier),
    )

    print("=== VQBR Synthetic Sweep (fixed 20/60/20 split) ===")
    print(
        f"D={n_features}, total_samples={n_total} "
        f"({args.dataset_multiplier}D), noise_std={float(args.noise_std):.3e}"
    )
    print(f"Split sizes: prior={n_prior}, train_pool={n_train_pool}, test={n_test}")
    print(f"Seeds ({len(seeds)}): {', '.join(str(s) for s in seeds)}")
    print(f"Train-size sweep (N): {', '.join(str(n) for n in train_sizes)}")
    print(
        f"Prior estimation config: bootstrap={int(args.prior_bootstrap_samples)}, "
        f"ridge={float(args.prior_ridge):.3e}, sigma_floor={float(args.prior_sigma_floor):.3e}, "
        f"V_floor={float(args.V_floor):.3e}"
    )
    print(
        f"VQBR config: optimizer={optimizer_name}, reps={args.reps}, maxiter={args.maxiter}, "
        f"shots={args.shots}, use_shot_noise={bool(args.use_shot_noise)}, "
        f"batch_size={'full-train' if args.batch_size is None else int(args.batch_size)}"
    )

    per_seed_results: dict[int, dict[int, dict[str, EvaluationMetrics]]] = {}
    per_seed_prior_info: dict[int, dict[str, float]] = {}
    for seed in seeds:
        print("")
        print(f"[seed={seed}] generating data and fixed splits...")
        X, y, _ = _make_synthetic_dataset(
            n_samples=n_total,
            n_features=n_features,
            noise_std=float(args.noise_std),
            seed=int(seed),
        )
        split = _split_dataset_indices(n_samples=n_total, seed=int(seed))

        X_prior = X[split.prior_indices]
        y_prior = y[split.prior_indices]
        X_train_pool = X[split.train_indices]
        y_train_pool = y[split.train_indices]
        X_test = X[split.test_indices]
        y_test = y[split.test_indices]

        m0, Sigma0, V, prior_info = _estimate_prior_from_prior_split(
            X_prior,
            y_prior,
            seed=int(seed),
            bootstrap_samples=int(args.prior_bootstrap_samples),
            ridge=float(args.prior_ridge),
            sigma_floor=float(args.prior_sigma_floor),
            v_floor=float(args.V_floor),
        )
        print(
            f"[seed={seed}] prior fixed: V={V:.6e}, "
            f"mean(Sigma0_diag)={float(np.mean(np.diag(Sigma0))):.6e}, "
            f"prior_fit_mse={prior_info['prior_fit_mse']:.6e}"
        )
        per_seed_prior_info[int(seed)] = {
            "V": float(V),
            "mean_sigma0_diag": float(np.mean(np.diag(Sigma0))),
            "prior_fit_mse": float(prior_info["prior_fit_mse"]),
        }

        per_seed_results[seed] = {}
        for n_train in train_sizes:
            print(f"[seed={seed}] N={n_train}: fit methods and evaluate on train/test.")
            X_train = X_train_pool[:n_train]
            y_train = y_train_pool[:n_train]

            w_star = _closed_form_map_solution(X_train, y_train, m0, Sigma0, V)
            closed_form_metrics = EvaluationMetrics(
                cosine_similarity=1.0,
                relative_l2_distance=0.0,
                train_mse=_mse(y_train, X_train @ w_star),
                test_mse=_mse(y_test, X_test @ w_star),
            )

            batch_size = int(args.batch_size) if args.batch_size is not None else int(n_train)
            if batch_size <= 0:
                raise ValueError("--batch-size must be a positive integer when provided.")
            batch_size = min(batch_size, int(n_train))

            model = VQBR(
                shots=int(args.shots),
                batch_size=batch_size,
                reps=int(args.reps),
                optimizer=optimizer_name,
                maxiter=int(args.maxiter),
                random_state=int(seed + n_train * 13),
                use_shot_noise=bool(args.use_shot_noise),
                verbose=bool(args.verbose),
                loss="neg_ratio",
            )

            iteration_callback = None
            if args.log_training:
                iteration_callback = _build_training_logger(
                    maxiter=int(args.maxiter),
                    log_every_iter=int(args.log_every_iter),
                )

            try:
                quantum_state = model.fit(
                    X_train,
                    y_train,
                    m0,
                    Sigma0,
                    V=V,
                    iteration_callback=iteration_callback,
                )
                if float(np.linalg.norm(w_star)) > EPS:
                    reconstructed_w = _reconstruct_vector_from_state(
                        quantum_state=quantum_state,
                        w_reference=w_star,
                        n_features=int(X_train.shape[1]),
                    )
                else:
                    reconstructed_w = np.zeros_like(w_star, dtype=float)
                vqbr_metrics = EvaluationMetrics(
                    cosine_similarity=_safe_cosine(reconstructed_w, w_star),
                    relative_l2_distance=float(
                        np.linalg.norm(reconstructed_w - w_star)
                        / max(float(np.linalg.norm(w_star)), EPS)
                    ),
                    train_mse=_mse(y_train, X_train @ reconstructed_w),
                    test_mse=_mse(y_test, X_test @ reconstructed_w),
                )
            except Exception as exc:
                print(f"  [seed={seed}][N={n_train}] VQBR failed: {exc}")
                vqbr_metrics = EvaluationMetrics(
                    cosine_similarity=float("nan"),
                    relative_l2_distance=float("nan"),
                    train_mse=float("nan"),
                    test_mse=float("nan"),
                )

            per_seed_results[seed][n_train] = {
                "closed_form": closed_form_metrics,
                "vqbr": vqbr_metrics,
            }

    summary = _aggregate_across_seeds(per_seed_results, train_sizes)

    print("")
    print("=== Aggregate Results Across Seeds (mean +/- std) ===")
    print("Columns: N, method, train_mse, test_mse, cosine_similarity, relative_l2_distance")
    for n_train in train_sizes:
        for method in METHOD_KEYS:
            row = summary[n_train][method]
            train_text = _format_mean_std(
                row["train_mse"]["mean"],
                row["train_mse"]["std"],
                scientific=True,
            )
            test_text = _format_mean_std(
                row["test_mse"]["mean"],
                row["test_mse"]["std"],
                scientific=True,
            )
            cos_text = _format_mean_std(
                row["cosine_similarity"]["mean"],
                row["cosine_similarity"]["std"],
                scientific=False,
            )
            rel_text = _format_mean_std(
                row["relative_l2_distance"]["mean"],
                row["relative_l2_distance"]["std"],
                scientific=True,
            )
            print(
                f"N={n_train:>4d} | {method:<11} | "
                f"train_mse={train_text} | test_mse={test_text} | "
                f"cos={cos_text} | rel_l2={rel_text}"
            )

    _save_results(
        output_path=output_path,
        summary_csv_path=summary_csv_path,
        json_indent=int(args.json_indent),
        args=args,
        seeds=seeds,
        train_sizes=train_sizes,
        n_features=n_features,
        n_total=n_total,
        n_prior=n_prior,
        n_train_pool=n_train_pool,
        n_test=n_test,
        per_seed_results=per_seed_results,
        per_seed_prior_info=per_seed_prior_info,
        summary=summary,
    )


def run_experiment(args: argparse.Namespace) -> None:
    dimensions = _resolve_feature_dimensions(
        n_features=int(args.n_features),
        n_features_list=args.n_features_list,
    )
    multi_dimension_mode = len(dimensions) > 1

    if multi_dimension_mode:
        print("=== Multi-D Sweep Mode ===")
        print(f"Requested D values: {', '.join(str(d) for d in dimensions)}")

    saved_outputs: list[tuple[int, Path, Path]] = []
    for d in dimensions:
        output_path = _resolve_output_path_for_dimension(
            raw_path=args.output_path,
            n_features=int(d),
            multi_dimension_mode=multi_dimension_mode,
        )
        summary_csv_path = _resolve_output_path_for_dimension(
            raw_path=args.summary_csv_path,
            n_features=int(d),
            multi_dimension_mode=multi_dimension_mode,
        )

        if multi_dimension_mode:
            print("")
            print(f"=== Begin D={int(d)} ===")
            print(f"JSON output: {output_path}")
            print(f"CSV output:  {summary_csv_path}")

        _run_experiment_for_dimension(
            args,
            n_features=int(d),
            output_path=output_path,
            summary_csv_path=summary_csv_path,
        )
        saved_outputs.append((int(d), output_path, summary_csv_path))

    if multi_dimension_mode:
        print("")
        print("=== Multi-D Sweep Completed ===")
        for d, json_path, csv_path in saved_outputs:
            print(f"D={d}:")
            print(f"  JSON: {json_path}")
            print(f"  CSV:  {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic VQBR sweep with fixed dataset size (20D, D) and fixed 20/60/20 split "
            "per seed. Prior/test subsets are fixed while training-size N is swept from 1D to 12D. "
            "Supports single D (--D) or multiple D values (--Ds)."
        )
    )
    parser.add_argument("--D", "--n-features", dest="n_features", type=int, default=8)
    parser.add_argument(
        "--Ds",
        "--d-list",
        dest="n_features_list",
        type=str,
        default=None,
        help=(
            "Optional comma-separated D values, e.g. '8,16,32'. "
            "If provided, the script runs once per D."
        ),
    )
    parser.add_argument(
        "--dataset-multiplier",
        type=int,
        default=20,
        help="Total samples are dataset_multiplier * D (default: 20D).",
    )
    parser.add_argument(
        "--train-min-multiplier",
        type=int,
        default=1,
        help="Minimum training size multiplier k in N=kD (default: 1).",
    )
    parser.add_argument(
        "--train-max-multiplier",
        type=int,
        default=12,
        help="Maximum training size multiplier k in N=kD (default: 12).",
    )
    parser.add_argument(
        "--train-step-multiplier",
        type=int,
        default=1,
        help="Step for k in N=kD (default: 1).",
    )
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument(
        "--seed",
        type=int,
        default=2505,
        help="Base seed used when --seeds is not provided.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of seeds (base seed + offset) when --seeds is not provided.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated seed list, e.g. '2505,2506,2507'.",
    )

    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size used by VQBR; default uses all train samples.",
    )
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="COBYLA",
        help="VQBR optimizer: COBYLA or SPSA.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=2048,
        help="Number of shots used by VQBR when shot noise is enabled.",
    )
    parser.add_argument("--use-shot-noise", action="store_true")
    parser.add_argument("--no-shot-noise", dest="use_shot_noise", action="store_false")
    parser.set_defaults(use_shot_noise=True)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--prior-bootstrap-samples",
        type=int,
        default=64,
        help="Bootstrap samples on prior split to estimate m0 and Sigma0.",
    )
    parser.add_argument(
        "--prior-ridge",
        type=float,
        default=1e-6,
        help="Ridge term used in bootstrap linear fits on prior split.",
    )
    parser.add_argument(
        "--prior-sigma-floor",
        type=float,
        default=1e-4,
        help="Minimum diagonal variance for Sigma0.",
    )
    parser.add_argument(
        "--V-floor",
        type=float,
        default=1e-8,
        help="Lower bound for estimated V.",
    )

    parser.set_defaults(log_training=True)
    parser.add_argument("--log-training", action="store_true")
    parser.add_argument("--no-log-training", dest="log_training", action="store_false")
    parser.add_argument("--log-every-iter", type=int, default=10)
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(ROOT_DIR / "results" / "synthetic" / "vqbr_synthetic_test_results.json"),
        help=(
            "Path to save JSON results. In multi-D mode, '_D{D}' is appended automatically "
            "unless the path contains '{D}'."
        ),
    )
    parser.add_argument(
        "--summary-csv-path",
        type=str,
        default=str(ROOT_DIR / "results" / "synthetic" / "vqbr_synthetic_test_summary.csv"),
        help=(
            "Path to save aggregate mean/std CSV summary. In multi-D mode, '_D{D}' is appended "
            "automatically unless the path contains '{D}'."
        ),
    )
    parser.add_argument(
        "--console-log-path",
        type=str,
        default=str(ROOT_DIR / "results" / "synthetic" / "vqbr_synthetic_test_run.log"),
        help=(
            "Path to mirror all console output (stdout and stderr) in real time. "
            "Set empty string to disable file logging."
        ),
    )
    parser.add_argument("--json-indent", type=int, default=2)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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
