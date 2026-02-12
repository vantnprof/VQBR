from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

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


@dataclass
class ShotRunResult:
    shots: int
    losses: np.ndarray
    metrics: ComparisonMetrics


def _pad_to_pow2(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=complex).reshape(-1)
    n = int(np.ceil(np.log2(vec.size)))
    target = 2**n
    if target == vec.size:
        return vec.copy()
    return np.pad(vec, (0, target - vec.size), mode="constant")


def _shot_label(shots: int) -> str:
    p = int(np.log2(shots))
    if 2**p == shots:
        return f"2^{p}"
    return str(shots)


def parse_shot_powers(raw: str) -> List[int]:
    powers = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not powers:
        raise ValueError("At least one shot power must be provided.")
    if any(p < 1 for p in powers):
        raise ValueError("Shot powers must be positive integers.")
    return [2**p for p in powers]


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
    w_dir = w_dir / np.linalg.norm(w_dir)
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
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    y = X @ true_w + rng.normal(0.0, noise_std, size=n_samples)

    m0 = rng.normal(0.0, 0.3, size=n_features)
    Sigma = np.diag(np.full(n_features, prior_var, dtype=float))
    return X, y, m0, Sigma, true_w


def save_training_convergence_pdf(results: List[ShotRunResult], output_pdf: str) -> None:
    if not results:
        return

    fig, ax = plt.subplots(figsize=(6, 4), dpi=1000)
    for run in sorted(results, key=lambda x: x.shots):
        if run.losses.size == 0:
            continue
        iterations = np.arange(1, run.losses.size + 1)
        ax.plot(
            iterations,
            run.losses,
            linewidth=1.0,
            label=f"shots={_shot_label(run.shots)}",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training Loss (L_tilde)")
    ax.set_title("Training Convergence over Shot Budgets")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf", dpi=1000, bbox_inches="tight")
    plt.close(fig)


def save_cosine_vs_shots_pdf(results: List[ShotRunResult], output_pdf: str) -> None:
    if not results:
        return

    results_sorted = sorted(results, key=lambda x: x.shots)
    shots = np.array([r.shots for r in results_sorted], dtype=int)
    cosine = np.array([r.metrics.cosine_similarity for r in results_sorted], dtype=float)
    shot_labels = [_shot_label(int(s)) for s in shots]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=1000)
    ax.plot(shots, cosine, marker="o", linewidth=1.2, color="#d62728")
    ax.set_xscale("log", base=2)
    ax.set_xticks(shots)
    ax.set_xticklabels(shot_labels)
    ax.set_xlabel("Shots")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity to Closed-Form vs Shots")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf", dpi=1000, bbox_inches="tight")
    plt.close(fig)


def print_training_logs(
    losses: np.ndarray,
    shots: int,
    log_every_iter: int,
) -> None:
    if losses.size == 0:
        return
    if log_every_iter <= 0:
        raise ValueError("log_every_iter must be positive.")

    print(f"  Training log (shots={_shot_label(shots)}):")
    for i, loss in enumerate(losses, start=1):
        if i != 1 and i != losses.size and (i % log_every_iter != 0):
            continue
        print(f"    iter {i:>4}/{losses.size:<4} | loss={float(loss):.6f}")


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
    shots_list = parse_shot_powers(args.shot_powers)

    runs: List[ShotRunResult] = []

    print("=== VQBR Shot Sweep vs Closed-Form MAP ===")
    print(f"N={args.n_samples}, D={args.n_features}, V={args.V}")
    print(f"Optimizer: {args.optimizer}, maxiter={args.maxiter}, reps={args.reps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Shot noise: {args.use_shot_noise}")
    print(f"Shot sweep: {', '.join(_shot_label(s) for s in shots_list)}")
    print("")

    for shots in shots_list:
        print(f"Training run for shots={_shot_label(shots)} ...", flush=True)
        model = VQBR(
            shots=shots,
            batch_size=args.batch_size,
            reps=args.reps,
            optimizer=args.optimizer,
            maxiter=args.maxiter,
            random_state=args.seed,
            use_shot_noise=args.use_shot_noise,
            verbose=args.verbose,
        )

        quantum_state = model.fit(X, y, m0, Sigma, V=args.V)
        metrics = compare_quantum_to_closed_form(quantum_state, w_star, n_features=X.shape[1])
        losses = np.array([snap.L_tilde for snap in model.history_], dtype=float)
        runs.append(ShotRunResult(shots=shots, losses=losses, metrics=metrics))

        if args.log_training:
            print_training_logs(
                losses=losses,
                shots=shots,
                log_every_iter=args.log_every_iter,
            )

        print(f"[shots={_shot_label(shots)}]")
        print(f"  Fidelity:             {metrics.fidelity:.6f}")
        print(f"  Cosine similarity:    {metrics.cosine_similarity:.6f}")
        print(f"  Relative L2 distance: {metrics.relative_l2:.6f}")
        print(f"  Iterations evaluated: {losses.size}")
        print("")

    save_training_convergence_pdf(runs, args.convergence_pdf)
    save_cosine_vs_shots_pdf(runs, args.cosine_pdf)

    print(f"Closed-form relative error to true w: {closed_form_error:.6f}")
    print(f"Training convergence plot saved to:   {args.convergence_pdf}")
    print(f"Cosine-vs-shots plot saved to:        {args.cosine_pdf}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run VQBR over multiple shot budgets and compare against "
            "the closed-form Bayesian MAP direction."
        )
    )
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--n-features", type=int, default=6)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--prior-var", type=float, default=1.0)
    parser.add_argument("--V", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2505)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="COBYLA")
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--epochs", type=int, dest="maxiter")
    parser.add_argument("--shot-powers", type=str, default="8,10,12,14,16,18")
    parser.set_defaults(use_shot_noise=True)
    parser.add_argument("--use-shot-noise", action="store_true")
    parser.add_argument("--no-shot-noise", dest="use_shot_noise", action="store_false")
    parser.set_defaults(log_training=True)
    parser.add_argument("--log-training", action="store_true")
    parser.add_argument("--no-log-training", dest="log_training", action="store_false")
    parser.add_argument("--log-every-iter", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--convergence-pdf",
        type=str,
        default="vqbr_training_convergence_vs_shots.pdf",
    )
    parser.add_argument(
        "--cosine-pdf",
        type=str,
        default="vqbr_cosine_similarity_vs_shots.pdf",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
