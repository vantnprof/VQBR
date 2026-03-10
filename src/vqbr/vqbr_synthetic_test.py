from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

try:
    from .vqbr import VQBR
except ImportError:
    from vqbr import VQBR


EPS = 1e-12


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


def _split_dataset_indices(n_samples: int, seed: int) -> SplitIndices:
    if n_samples < 4:
        raise ValueError("n_samples must be >= 4 to form non-empty 30/50/20 splits.")

    n_prior = int(np.floor(0.30 * n_samples))
    n_train = int(np.floor(0.50 * n_samples))
    n_test = int(n_samples - n_prior - n_train)

    if n_prior < 1 or n_train < 2 or n_test < 1:
        raise ValueError(
            "Invalid split sizes from 30/50/20. Increase --N so prior/train/test are non-empty "
            "and train has at least 2 samples."
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


def run_experiment(args: argparse.Namespace) -> None:
    optimizer_name = _normalize_optimizer(args.optimizer)
    if args.shots <= 0:
        raise ValueError("--shots must be a positive integer.")
    if args.maxiter <= 0:
        raise ValueError("--maxiter must be a positive integer.")

    X, y, true_w = _make_synthetic_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    split = _split_dataset_indices(n_samples=int(args.n_samples), seed=int(args.seed))

    X_prior = X[split.prior_indices]
    y_prior = y[split.prior_indices]
    X_train = X[split.train_indices]
    y_train = y[split.train_indices]
    X_test = X[split.test_indices]
    y_test = y[split.test_indices]

    m0, Sigma0, V, prior_info = _estimate_prior_from_prior_split(
        X_prior,
        y_prior,
        seed=int(args.seed),
        bootstrap_samples=int(args.prior_bootstrap_samples),
        ridge=float(args.prior_ridge),
        sigma_floor=float(args.prior_sigma_floor),
        v_floor=float(args.V_floor),
    )

    w_star = _closed_form_map_solution(X_train, y_train, m0, Sigma0, V)
    closed_form_train_mse = _mse(y_train, X_train @ w_star)
    closed_form_test_mse = _mse(y_test, X_test @ w_star)

    batch_size = int(args.batch_size) if args.batch_size is not None else int(X_train.shape[0])
    if batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer when provided.")

    model = VQBR(
        shots=args.shots,
        batch_size=batch_size,
        reps=args.reps,
        optimizer=optimizer_name,
        maxiter=args.maxiter,
        random_state=args.seed,
        use_shot_noise=args.use_shot_noise,
        verbose=args.verbose,
        loss="neg_ratio",
    )

    print("=== VQBR Synthetic Test (30/50/20 split) ===")
    print(f"N={args.n_samples}, D={args.n_features}, seed={args.seed}")
    print(
        f"Split sizes: prior={X_prior.shape[0]}, train={X_train.shape[0]}, test={X_test.shape[0]}"
    )
    print(
        f"Prior estimation config: bootstrap={int(args.prior_bootstrap_samples)}, "
        f"ridge={float(args.prior_ridge):.3e}, sigma_floor={float(args.prior_sigma_floor):.3e}, "
        f"V_floor={float(args.V_floor):.3e}"
    )
    print(
        f"Estimated prior: V={V:.6e}, "
        f"mean(Sigma0_diag)={float(np.mean(np.diag(Sigma0))):.6e}, "
        f"prior_fit_mse={prior_info['prior_fit_mse']:.6e}"
    )
    print(
        f"VQBR config: optimizer={optimizer_name}, reps={args.reps}, "
        f"maxiter={args.maxiter}, batch_size={batch_size}, shots={args.shots}, "
        f"use_shot_noise={bool(args.use_shot_noise)}"
    )
    print("Starting training...")

    iteration_callback = None
    if args.log_training:
        print("Training log:")
        iteration_callback = _build_training_logger(
            maxiter=int(args.maxiter),
            log_every_iter=int(args.log_every_iter),
        )

    quantum_state = model.fit(
        X_train,
        y_train,
        m0,
        Sigma0,
        V=V,
        iteration_callback=iteration_callback,
    )

    reconstructed_w = _reconstruct_vector_from_state(
        quantum_state=quantum_state,
        w_reference=w_star,
        n_features=int(X_train.shape[1]),
    )
    metrics = EvaluationMetrics(
        cosine_similarity=_safe_cosine(reconstructed_w, w_star),
        relative_l2_distance=float(
            np.linalg.norm(reconstructed_w - w_star) / max(float(np.linalg.norm(w_star)), EPS)
        ),
        train_mse=_mse(y_train, X_train @ reconstructed_w),
        test_mse=_mse(y_test, X_test @ reconstructed_w),
    )

    true_error = float(np.linalg.norm(w_star - true_w) / max(float(np.linalg.norm(true_w)), EPS))

    print("")
    print("Requested metrics (from reconstructed VQBR vector):")
    print(f"  cosine similarity:    {metrics.cosine_similarity:.6f}")
    print(f"  relative l2-distance: {metrics.relative_l2_distance:.6e}")
    print(f"  train mse:            {metrics.train_mse:.6f}")
    print(f"  test mse:             {metrics.test_mse:.6f}")
    print("")
    print("Reference (closed-form MAP on train split):")
    print(f"  train mse:            {closed_form_train_mse:.6f}")
    print(f"  test mse:             {closed_form_test_mse:.6f}")
    print(f"  rel error vs true w:  {true_error:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic VQBR test with 30/50/20 split: "
            "30% prior-estimation, 50% train, 20% test."
        )
    )
    parser.add_argument("--N", "--n-samples", dest="n_samples", type=int, default=100)
    parser.add_argument("--D", "--n-features", dest="n_features", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2505)

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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
