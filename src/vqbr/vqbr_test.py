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


def build_trained_ansatz_circuit(model: VQBR):
    theta_star = model.get_theta()
    return model._ansatz.assign_parameters(model._theta_bind(theta_star), inplace=False)


def build_u_sel_circuit(model: VQBR, Uv_gate):
    try:
        from qiskit import QuantumCircuit
    except Exception as exc:
        raise ImportError("Qiskit is required to build U_sel.") from exc

    theta_star = model.get_theta()
    n_work = int(model._num_qubits)
    anc = 0
    work = list(range(1, 1 + n_work))

    qc = QuantumCircuit(1 + n_work, name="U_sel")
    qc.x(anc)
    qc.append(Uv_gate.control(1), [anc] + work)
    qc.x(anc)

    bound_ansatz = model._ansatz.assign_parameters(model._theta_bind(theta_star), inplace=False)
    controlled_ansatz = bound_ansatz.to_gate(label="Utheta").control(1)
    qc.append(controlled_ansatz, [anc] + work)
    return qc


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

    if state.size > n_features:
        padding_probability_mass = float(np.sum(np.abs(state[n_features:]) ** 2))
    else:
        padding_probability_mass = 0.0

    state_feature = np.asarray(state_aligned[:n_features], dtype=complex)
    state_feature_real = np.real_if_close(state_feature, tol=1000)
    if np.iscomplexobj(state_feature_real):
        state_feature_real = np.real(state_feature)
    state_feature_real = np.asarray(state_feature_real, dtype=float).reshape(-1)

    state_feature_norm = float(np.linalg.norm(state_feature_real))
    if state_feature_norm == 0.0:
        feature_space_cosine = 0.0
        reconstructed_w = np.zeros_like(w_dir, dtype=float)
    else:
        direction = state_feature_real / state_feature_norm
        # Quantum states are phase/sign-invariant. Align reconstructed direction to w*.
        if float(np.dot(direction, w_dir)) < 0.0:
            direction = -direction
        reconstructed_w = float(w_norm) * direction
        feature_space_cosine = float(np.abs(np.vdot(w_dir, direction)))

    relative_l2 = float(
        np.linalg.norm(reconstructed_w - np.asarray(w_star, dtype=float).reshape(-1))
        / float(w_norm)
    )

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
    ax.plot(iterations, losses, linewidth=1.0, color="#1f77b4", label="VQBR")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training Loss (L_tilde)")
    ax.set_title("VQBR Training Convergence")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)


def _normalize_optimizer(raw: str) -> str:
    token = str(raw).strip().upper()
    valid = ("COBYLA", "SPSA")
    if token not in valid:
        raise ValueError(
            f"Unsupported optimizer '{raw}'. Supported values: {', '.join(valid)}."
        )
    return token


def make_realtime_training_logger(maxiter: int, log_every_iter: int):
    if log_every_iter <= 0:
        raise ValueError("log_every_iter must be positive.")
    prev_loss: float | None = None

    def _format_delta(current: float, previous: float | None) -> str:
        if previous is None:
            return "delta=--"
        if not np.isfinite(current) or not np.isfinite(previous):
            return "delta=n/a"
        diff = float(current - previous)
        if diff > 0.0:
            return f"delta=↑ {abs(diff):.6e}"
        if diff < 0.0:
            return f"delta=↓ {abs(diff):.6e}"
        return "delta=→ 0.000000e+00"

    def _callback(iteration: int, snapshot) -> None:
        nonlocal prev_loss
        current_loss = float(snapshot.L_tilde)
        delta_text = _format_delta(current_loss, prev_loss)
        should_log = (
            iteration == 1
            or iteration == maxiter
            or (iteration % log_every_iter == 0)
        )
        if should_log:
            print(
                f"    iter {iteration:>4}/{maxiter:<4} | "
                f"loss={current_loss:.6f}, {delta_text}",
                flush=True,
            )
        prev_loss = current_loss

    return _callback


def run_experiment(args: argparse.Namespace) -> None:
    optimizer_name = _normalize_optimizer(args.optimizer)
    X, y, m0, Sigma, true_w = make_synthetic_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        noise_std=args.noise_std,
        prior_var=args.prior_var,
        seed=args.seed,
    )
    w_star = closed_form_map_solution(X, y, m0, Sigma, args.V)
    closed_form_error = float(np.linalg.norm(w_star - true_w) / np.linalg.norm(true_w))

    print("=== VQBR vs Closed-Form MAP ===")
    print(f"N={args.n_samples}, D={args.n_features}, V={args.V}")
    batch_size = int(args.batch_size) if args.batch_size is not None else int(X.shape[0])
    if batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.shots <= 0:
        raise ValueError("--shots must be a positive integer.")
    print(
        f"Optimizer: {optimizer_name}, maxiter={args.maxiter}, reps={args.reps}, "
        f"batch_size={batch_size}, shots={args.shots}"
    )
    if args.use_shot_noise:
        print("Mode: stochastic (full-batch, shot noise enabled)")
    else:
        print("Mode: deterministic (full-batch, no shot noise)")
    print("")

    model = VQBR(
        shots=args.shots,
        batch_size=batch_size,
        reps=args.reps,
        optimizer=optimizer_name,
        maxiter=args.maxiter,
        random_state=args.seed,
        use_shot_noise=args.use_shot_noise,
        verbose=args.verbose,
        loss="neg_ratio"
    )

    iteration_callback = None
    if args.log_training:
        print("  Training log:")
        iteration_callback = make_realtime_training_logger(
            maxiter=args.maxiter,
            log_every_iter=args.log_every_iter,
        )

    quantum_state = model.fit(X, y, m0, Sigma, V=args.V, iteration_callback=iteration_callback)
    if args.print_circuit:
        u_sel_x0 = build_u_sel_circuit(model, model._Uxi_gates[0])
        print("U_sel(theta, x_0) circuit (text draw):")
        print(u_sel_x0.draw(output="text", fold=args.circuit_fold))
        print("")

        if model._Uc_gate is not None:
            u_sel_c = build_u_sel_circuit(model, model._Uc_gate)
            print("U_sel(theta, c) circuit (text draw):")
            print(u_sel_c.draw(output="text", fold=args.circuit_fold))
        print("")

        trained_circuit = build_trained_ansatz_circuit(model)
        print("Trained ansatz circuit (text draw):")
        print(trained_circuit.draw(output="text", fold=args.circuit_fold))
        print("")

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
        description="Run VQBR and compare against closed-form MAP."
    )
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--n-features", type=int, default=6)
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
    parser.add_argument(
        "--optimizer",
        type=str,
        default="COBYLA",
        help="VQBR optimizer: COBYLA or SPSA.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=20000,
        help="Number of shots used by VQBR when shot noise is enabled.",
    )
    parser.add_argument("--use-shot-noise", action="store_true")
    parser.add_argument("--no-shot-noise", dest="use_shot_noise", action="store_false")
    parser.set_defaults(use_shot_noise=True)
    parser.set_defaults(log_training=True)
    parser.add_argument("--log-training", action="store_true")
    parser.add_argument("--no-log-training", dest="log_training", action="store_false")
    parser.add_argument("--log-every-iter", type=int, default=10)
    parser.set_defaults(print_circuit=True)
    parser.add_argument("--print-circuit", action="store_true")
    parser.add_argument("--no-print-circuit", dest="print_circuit", action="store_false")
    parser.add_argument(
        "--circuit-fold",
        type=int,
        default=120,
        help="Text-drawer line width; use -1 to disable wrapping.",
    )
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
