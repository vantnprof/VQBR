from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .vqbr import VQBR
except ImportError:
    from vqbr import VQBR


@dataclass
class ComparisonMetrics:
    fidelity: float
    cosine_similarity: float
    relative_l2: float
    padding_probability_mass: float
    feature_space_cosine: float


def _pad_to_pow2(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=complex).reshape(-1)
    n = int(np.ceil(np.log2(vec.size)))
    target = 2**n
    if target == vec.size:
        return vec.copy()
    return np.pad(vec, (0, target - vec.size), mode="constant")


def closed_form_map_solution(
    X: np.ndarray, y: np.ndarray, m0: np.ndarray, Sigma: np.ndarray, V: float
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


def compare_quantum_to_closed_form(
    quantum_state: np.ndarray, w_star: np.ndarray, n_features: int
) -> ComparisonMetrics:
    state = np.asarray(quantum_state, dtype=complex).reshape(-1)
    state = state / np.linalg.norm(state)

    w_dir = np.asarray(w_star, dtype=float).reshape(-1)
    w_norm = np.linalg.norm(w_dir)
    if w_norm == 0.0:
        raise ValueError("Closed-form solution has zero norm; cannot compare directions.")
    w_dir = w_dir / w_norm
    w_dir_pad = _pad_to_pow2(w_dir)
    w_dir_pad = w_dir_pad / np.linalg.norm(w_dir_pad)

    overlap = np.vdot(w_dir_pad, state)
    phase = np.exp(-1j * np.angle(overlap))
    state_aligned = state * phase

    fidelity = float(np.abs(np.vdot(w_dir_pad, state)) ** 2)
    cosine_similarity = float(np.real(np.vdot(w_dir_pad, state_aligned)))
    relative_l2 = float(np.linalg.norm(state_aligned - w_dir_pad) / np.linalg.norm(w_dir_pad))

    if state.size > n_features:
        padding_probability_mass = float(np.sum(np.abs(state[n_features:]) ** 2))
    else:
        padding_probability_mass = 0.0

    state_feature = state_aligned[:n_features]
    if np.linalg.norm(state_feature) == 0:
        feature_space_cosine = 0.0
    else:
        state_feature = state_feature / np.linalg.norm(state_feature)
        feature_space_cosine = float(np.abs(np.vdot(w_dir, state_feature)))

    return ComparisonMetrics(
        fidelity=fidelity,
        cosine_similarity=cosine_similarity,
        relative_l2=relative_l2,
        padding_probability_mass=padding_probability_mass,
        feature_space_cosine=feature_space_cosine,
    )


def make_synthetic_dataset(
    n_samples: int,
    n_features: int,
    noise_std: float,
    prior_var: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_w = rng.normal(0.0, 1.0, size=n_features)
    X = rng.normal(5, 1.0, size=(n_samples, n_features))
    y = X @ true_w + rng.normal(0.0, noise_std, size=n_samples)

    m0 = rng.normal(0.0, 0.3, size=n_features)
    Sigma = np.diag(np.full(n_features, prior_var, dtype=float))
    return X, y, m0, Sigma, true_w


def save_training_convergence_plot(losses: np.ndarray, output_path: str) -> None:
    if losses.size == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4), dpi=1000)
    iterations = np.arange(1, losses.size + 1)
    ax.plot(iterations, losses, linewidth=1.0, color="#1f77b4", label="COBYLA")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training Loss (L_tilde)")
    ax.set_title("VQBR Training Convergence (COBYLA)")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)


def make_realtime_training_logger(maxiter: int, log_every_iter: int):
    if log_every_iter <= 0:
        raise ValueError("log_every_iter must be positive.")

    def _callback(iteration: int, snapshot) -> None:
        if iteration != 1 and iteration != maxiter and (iteration % log_every_iter != 0):
            return
        print(
            f"    iter {iteration:>4}/{maxiter:<4} | loss={float(snapshot.L_tilde):.6f}",
            flush=True,
        )

    return _callback


def run_experiment(args: argparse.Namespace) -> None:
    X, y, m0, Sigma, true_w = make_synthetic_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        noise_std=args.noise_std,
        prior_var=args.prior_var,
        seed=args.seed,
    )
    w_star = closed_form_map_solution(X, y, m0, Sigma, args.V)
    closed_form_error = float(np.linalg.norm(w_star - true_w) / np.linalg.norm(true_w))

    print("=== VQBR vs Closed-Form MAP (COBYLA) ===")
    print(f"N={args.n_samples}, D={args.n_features}, V={args.V}")
    batch_size = int(args.batch_size) if args.batch_size is not None else int(X.shape[0])
    if batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    print(
        f"Optimizer: COBYLA, maxiter={args.maxiter}, reps={args.reps}, "
        f"batch_size={batch_size}"
    )
    print("Mode: deterministic (full-batch, no shot noise)")
    print("")

    model = VQBR(
        batch_size=batch_size,
        reps=args.reps,
        optimizer="COBYLA",
        maxiter=args.maxiter,
        random_state=args.seed,
        use_shot_noise=False,
        verbose=args.verbose,
    )

    iteration_callback = None
    if args.log_training:
        print("  Training log:")
        iteration_callback = make_realtime_training_logger(
            maxiter=args.maxiter,
            log_every_iter=args.log_every_iter,
        )

    quantum_state = model.fit(X, y, m0, Sigma, V=args.V, iteration_callback=iteration_callback)
    metrics = compare_quantum_to_closed_form(quantum_state, w_star, n_features=X.shape[1])
    losses = np.array([snap.L_tilde for snap in model.history_], dtype=float)

    print("Results:")
    print(f"  Fidelity:                 {metrics.fidelity:.6f}")
    print(f"  Cosine similarity:        {metrics.cosine_similarity:.6f}")
    print(f"  Feature-space cosine:     {metrics.feature_space_cosine:.6f}")
    print(f"  Relative L2 distance:     {metrics.relative_l2:.6f}")
    print(f"  Padding probability mass: {metrics.padding_probability_mass:.6e}")
    print(f"  Iterations evaluated:     {losses.size}")
    print("")

    save_training_convergence_plot(losses, args.convergence_plot)

    print(f"Closed-form relative error to true w: {closed_form_error:.6f}")
    print(f"Training convergence plot saved to:   {args.convergence_plot}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VQBR with deterministic COBYLA and compare against closed-form MAP."
    )
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--n-features", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--prior-var", type=float, default=1.0)
    parser.add_argument("--V", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2505)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size used by VQBR; default uses all training samples.",
    )
    parser.add_argument("--maxiter", type=int, default=200)
    parser.set_defaults(log_training=True)
    parser.add_argument("--log-training", action="store_true")
    parser.add_argument("--no-log-training", dest="log_training", action="store_false")
    parser.add_argument("--log-every-iter", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--convergence-plot",
        type=str,
        default="vqbr_training_convergence.png",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
