from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

# Make local method modules importable when running from repo root.
for module_dir in ("closed_form", "vqbr"):
    module_path = SRC_DIR / module_dir
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from closed_form import ClosedFormMAPBayesianRegression  # noqa: E402


EPS = 1e-12
METHOD_KEYS = ("vqbr_cobyla", "vqbr_spsa")
METRIC_KEYS = (
    "cosine_similarity",
    "relative_l2_distance",
    "train_mse",
    "test_mse",
)
PRIOR_CASES = ("heteroscedastic",)
NAVAL_FEATURE_COLS = [
    "lp",
    "v",
    "GTT",
    "GTn",
    "GGn",
    "Ts",
    "Tp",
    "T48",
    "T1",
    "T2",
    "P48",
    "P1",
    "P2",
    "Pexh",
    "TIC",
    "mf",
]
NAVAL_TARGET_COL_TO_INDEX = {
    "kMc": 16,
    "kMt": 17,
}
VQBR_OPTIMIZER_SETTINGS = (
    {
        "method_key": "vqbr_cobyla",
        "optimizer": "COBYLA",
        "use_full_batch": True,
        "batch_mode": "full_data",
    },
    {
        "method_key": "vqbr_spsa",
        "optimizer": "SPSA",
        "use_full_batch": False,
        "batch_mode": "mini_batch",
    },
)


@dataclass
class SplitData:
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_val_indices: np.ndarray
    val_indices: np.ndarray


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


def _fit_and_center_target(y: np.ndarray) -> tuple[float, np.ndarray]:
    y = np.asarray(y, dtype=float).reshape(-1)
    mean = float(np.mean(y))
    return mean, y - mean


def _make_vqbr_iteration_logger(
    *,
    prior_case: str,
    seed: int,
    method_key: str,
    optimizer: str,
    batch_mode: str,
    batch_size: int,
    log_every_iter: int,
) -> Any:
    if log_every_iter <= 0:
        raise ValueError("--log-every-iter must be a positive integer.")
    prev_batch_loss: float | None = None

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

    def _callback(iteration: int, snapshot: Any) -> None:
        nonlocal prev_batch_loss
        batch_loss = float(getattr(snapshot, "batch_L_tilde", np.nan))
        if not np.isfinite(batch_loss):
            batch_loss = float(getattr(snapshot, "L_tilde", np.nan))
        delta_text = _format_delta(batch_loss, prev_batch_loss)
        should_log = (iteration == 1) or (iteration % log_every_iter == 0)
        if should_log:
            print(
                "    "
                f"[{prior_case}][seed={seed}][{method_key}] "
                f"{optimizer} iter {iteration:>4d} | "
                f"mode={batch_mode}, batch_size={batch_size} | "
                f"batch_loss={batch_loss: .6e}, {delta_text}",
                flush=True,
            )
        prev_batch_loss = batch_loss

    return _callback


def _parse_float_grid(raw: str) -> np.ndarray:
    tokens = [x.strip() for x in raw.split(",") if x.strip()]
    if not tokens:
        raise ValueError("Grid specification cannot be empty.")
    values = np.array([float(x) for x in tokens], dtype=float)
    if np.any(values <= 0.0):
        raise ValueError("All grid values must be strictly positive.")
    return values


def _resolve_seeds(num_seeds: int, seed_offset: int, explicit_seeds: str | None) -> List[int]:
    if explicit_seeds is not None:
        seeds = [int(x.strip()) for x in explicit_seeds.split(",") if x.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but no valid integers were found.")
        return seeds
    if num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    return [seed_offset + i for i in range(num_seeds)]


def _parse_vqbr_optimizer_names(raw: str) -> List[str]:
    tokens = [x.strip().upper() for x in raw.split(",") if x.strip()]
    if not tokens:
        raise ValueError("--vqbr-optimizer must include at least one optimizer name.")

    valid = {str(setting["optimizer"]).upper() for setting in VQBR_OPTIMIZER_SETTINGS}
    invalid = sorted({token for token in tokens if token not in valid})
    if invalid:
        raise ValueError(
            "Unsupported --vqbr-optimizer values: "
            f"{', '.join(invalid)}. Supported values: {', '.join(sorted(valid))}."
        )

    unique_tokens: List[str] = []
    for token in tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)
    return unique_tokens


def _select_vqbr_optimizer_settings(raw: str) -> List[Dict[str, Any]]:
    selected_optimizer_names = _parse_vqbr_optimizer_names(raw)
    settings_by_optimizer = {
        str(setting["optimizer"]).upper(): setting for setting in VQBR_OPTIMIZER_SETTINGS
    }
    return [settings_by_optimizer[name] for name in selected_optimizer_names]


def _normalize_vqbr_loss(raw: str) -> str:
    mode = str(raw).strip().lower()
    aliases = {
        "log_ratio": "log_ratio",
        "log": "log_ratio",
        "neg_ratio": "neg_ratio",
        "ratio": "neg_ratio",
        "raw_ratio": "neg_ratio",
    }
    resolved = aliases.get(mode)
    if resolved is None:
        raise ValueError(
            "Unsupported --vqbr-loss value. Supported values: "
            "log_ratio, neg_ratio (aliases: log, ratio, raw_ratio)."
        )
    return resolved


def _load_naval_dataset(dataset_path: Path, target_col: str) -> tuple[np.ndarray, np.ndarray, List[str]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if target_col not in NAVAL_TARGET_COL_TO_INDEX:
        raise ValueError(
            f"Unsupported --target-col '{target_col}'. "
            f"Supported values: {sorted(NAVAL_TARGET_COL_TO_INDEX.keys())}"
        )

    data = np.loadtxt(dataset_path, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D matrix in {dataset_path}, got shape={data.shape}.")
    if data.shape[1] < 18:
        raise ValueError(
            f"Expected at least 18 columns (16 features + 2 targets), got {data.shape[1]}."
        )

    X = np.asarray(data[:, :16], dtype=float)
    target_index = NAVAL_TARGET_COL_TO_INDEX[target_col]
    y = np.asarray(data[:, target_index], dtype=float).reshape(-1)
    feature_cols = list(NAVAL_FEATURE_COLS)
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("Invalid dataset shapes after loading.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Feature matrix and target vector have inconsistent lengths.")
    if X.shape[0] < 3:
        raise ValueError("Dataset must contain at least 3 rows.")
    return X, y, feature_cols


def _make_splits(
    n_samples: int,
    train_ratio: float,
    val_ratio_within_train: float,
    seed: int,
) -> SplitData:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1).")
    if not (0.0 < val_ratio_within_train < 1.0):
        raise ValueError("val_ratio_within_train must be in (0, 1).")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)

    n_train_total = int(np.floor(train_ratio * n_samples))
    n_train_total = min(max(2, n_train_total), n_samples - 1)

    train_indices = order[:n_train_total]
    test_indices = order[n_train_total:]

    val_rng = np.random.default_rng(seed + 1_009)
    train_order = train_indices.copy()
    val_rng.shuffle(train_order)

    n_val = int(np.floor(val_ratio_within_train * n_train_total))
    n_val = min(max(1, n_val), n_train_total - 1)
    val_indices = train_order[:n_val]
    train_val_indices = train_order[n_val:]

    return SplitData(
        train_indices=train_indices,
        test_indices=test_indices,
        train_val_indices=train_val_indices,
        val_indices=val_indices,
    )


def _build_prior(
    prior_case: str,
    X_reference: np.ndarray,
    lambda_strength: float,
    delta: float,
    variance_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_features = X_reference.shape[1]
    m0 = np.zeros(n_features, dtype=float)

    if prior_case != "heteroscedastic":
        raise ValueError(
            f"Unsupported prior case: {prior_case}. "
            "This experiment only supports 'heteroscedastic'."
        )

    ref = X_reference if variance_reference is None else variance_reference
    feature_var = np.var(ref, axis=0, ddof=0)
    feature_var = np.maximum(feature_var, 0.0)
    precision_diag = lambda_strength / (feature_var + delta)
    sigma_diag = 1.0 / precision_diag
    return m0, sigma_diag


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


def _build_trained_circuit(vqbr_model: Any) -> Any:
    ansatz = getattr(vqbr_model, "_ansatz", None)
    if ansatz is None:
        raise RuntimeError("VQBR ansatz is unavailable after training.")

    theta = np.asarray(vqbr_model.get_theta(), dtype=float).reshape(-1)
    params = list(ansatz.parameters)
    if theta.size != len(params):
        raise RuntimeError(
            "Trained parameter vector length does not match ansatz parameter count: "
            f"{theta.size} vs {len(params)}."
        )
    bindings = {param: float(val) for param, val in zip(params, theta)}
    return ansatz.assign_parameters(bindings, inplace=False)


def _simulate_statevector_from_circuit(circuit: Any) -> np.ndarray:
    qiskit_quantum_info = importlib.import_module("qiskit.quantum_info")
    Statevector = getattr(qiskit_quantum_info, "Statevector")
    sv = Statevector.from_instruction(circuit)
    state = np.asarray(sv.data, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(state))
    if norm <= EPS:
        raise RuntimeError("Statevector simulation returned a zero-norm state.")
    return state / norm


def _serialize_trained_circuit(circuit: Any) -> Dict[str, Any]:
    qasm_repr = None
    if hasattr(circuit, "qasm"):
        try:
            qasm_repr = str(circuit.qasm())
        except Exception:
            qasm_repr = None
    if qasm_repr is None:
        try:
            qiskit_qasm2 = importlib.import_module("qiskit.qasm2")
            qasm_repr = str(qiskit_qasm2.dumps(circuit))
        except Exception:
            qasm_repr = None

    return {
        "num_qubits": int(circuit.num_qubits),
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "text_diagram": str(circuit.draw(output="text")),
        "qasm": qasm_repr,
    }


def _collect_batch_loss_history(vqbr_history: List[Any]) -> List[float]:
    losses: List[float] = []
    for snapshot in vqbr_history:
        batch_loss = float(getattr(snapshot, "batch_L_tilde", np.nan))
        if not np.isfinite(batch_loss):
            batch_loss = float(getattr(snapshot, "L_tilde", np.nan))
        losses.append(batch_loss)
    return losses


def _reconstruct_vqbr_vector_from_state(
    statevector: np.ndarray,
    w_star: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    w_star = np.asarray(w_star, dtype=float).reshape(-1)
    encoding_vector = _state_to_feature_direction(statevector, n_features=w_star.size)
    if encoding_vector.shape[0] != w_star.shape[0]:
        raise RuntimeError(
            "Encoded feature direction length does not match closed-form solution length."
        )
    # Quantum states are phase-invariant; align sign for fair vector comparison.
    if float(np.dot(encoding_vector, w_star)) < 0.0:
        encoding_vector = -encoding_vector
    reconstructed_vector = float(np.linalg.norm(w_star)) * encoding_vector
    return reconstructed_vector, encoding_vector


def _evaluate_vqbr_metrics(
    *,
    reconstructed_vector: np.ndarray,
    encoding_vector: np.ndarray,
    w_star: np.ndarray,
    X_train: np.ndarray,
    y_train_raw: np.ndarray,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    y_train_mean: float,
    final_batch_L_hat: float,
    batch_loss_history: List[float],
) -> Dict[str, Any]:
    y_train_pred = np.asarray(X_train, dtype=float) @ np.asarray(
        reconstructed_vector, dtype=float
    ) + float(y_train_mean)
    y_test_pred = np.asarray(X_test, dtype=float) @ np.asarray(
        reconstructed_vector, dtype=float
    ) + float(y_train_mean)
    w_star_arr = np.asarray(w_star, dtype=float)
    reconstruction_arr = np.asarray(reconstructed_vector, dtype=float)
    w_star_norm = float(np.linalg.norm(w_star_arr))
    relative_l2_distance = float(
        np.linalg.norm(reconstruction_arr - w_star_arr) / max(w_star_norm, EPS)
    )

    return {
        "cosine_similarity": _safe_cosine(reconstructed_vector, w_star),
        "relative_l2_distance": relative_l2_distance,
        "train_mse": _mse(y_train_raw, y_train_pred),
        "test_mse": _mse(y_test_raw, y_test_pred),
        "batch_L_hat": float(final_batch_L_hat),
        "iterations": int(len(batch_loss_history)),
        "batch_loss_history": [float(x) for x in batch_loss_history],
        # Kept for backward compatibility with older plotting helpers.
        "loss_history": [float(x) for x in batch_loss_history],
        "closed_form_norm": w_star_norm,
        "encoding_vector": [float(x) for x in np.asarray(encoding_vector, dtype=float)],
        "reconstructed_vector": [
            float(x) for x in np.asarray(reconstructed_vector, dtype=float)
        ],
    }


def _select_hyperparameters(
    prior_case: str,
    seed: int,
    X_train_for_tuning: np.ndarray,
    y_train_for_tuning_centered: np.ndarray,
    y_train_for_tuning_mean: float,
    X_train_for_prior_variance: np.ndarray,
    X_val: np.ndarray,
    y_val_raw: np.ndarray,
    V_grid: np.ndarray,
    lambda_grid: np.ndarray,
    prior_delta: float,
    log_realtime: bool,
    log_hyperparam_sweep: bool,
) -> Dict[str, float]:
    best: Dict[str, float] = {
        "validation_mse": float("inf"),
        "V": float(V_grid[0]),
        "lambda_strength": float(lambda_grid[0]),
    }

    model = ClosedFormMAPBayesianRegression()
    for V in V_grid:
        for lambda_strength in lambda_grid:
            m0, sigma_diag = _build_prior(
                prior_case=prior_case,
                X_reference=X_train_for_tuning,
                lambda_strength=float(lambda_strength),
                delta=prior_delta,
                variance_reference=X_train_for_prior_variance,
            )
            model.fit(
                X_train_for_tuning,
                y_train_for_tuning_centered,
                m0,
                sigma_diag,
                float(V),
            )
            y_val_pred_centered = model.predict(X_val)
            y_val_pred = y_val_pred_centered + float(y_train_for_tuning_mean)
            val_mse = _mse(y_val_raw, y_val_pred)
            if log_hyperparam_sweep:
                print(
                    "    "
                    f"[{prior_case}][seed={seed}] sweep: "
                    f"V={float(V):.3e}, lambda={float(lambda_strength):.3e}, "
                    f"val_mse={val_mse:.6f}",
                    flush=True,
                )
            if val_mse < best["validation_mse"]:
                best = {
                    "validation_mse": float(val_mse),
                    "V": float(V),
                    "lambda_strength": float(lambda_strength),
                }
    if log_realtime:
        print(
            "    "
            f"[{prior_case}][seed={seed}] selected hyperparams: "
            f"V={best['V']:.3e}, lambda={best['lambda_strength']:.3e}, "
            f"val_mse={best['validation_mse']:.6f}",
            flush=True,
        )
    return best


def _aggregate_case_results(case_seed_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    aggregate: Dict[str, Dict[str, Dict[str, float]]] = {}

    methods_present = set()
    for seed_result in case_seed_results:
        methods_present.update(seed_result["methods"].keys())

    for method in sorted(methods_present):
        aggregate[method] = {}
        for metric in METRIC_KEYS:
            values: List[float] = []
            for seed_result in case_seed_results:
                method_metrics = seed_result["methods"].get(method, {})
                if metric in method_metrics:
                    values.append(float(method_metrics[metric]))
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            aggregate[method][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=0)),
            }
    return aggregate


def _resolve_vqbr_optimizer_configs(
    args: argparse.Namespace,
    n_train_samples: int,
) -> List[Dict[str, Any]]:
    if n_train_samples <= 0:
        raise ValueError("n_train_samples must be positive.")
    if args.vqbr_batch_size <= 0:
        raise ValueError("--vqbr-batch-size must be positive.")

    selected_settings = _select_vqbr_optimizer_settings(args.vqbr_optimizer)

    configs: List[Dict[str, Any]] = []
    for setting in selected_settings:
        use_full_batch = bool(setting["use_full_batch"])
        if use_full_batch:
            batch_size_used = int(n_train_samples)
            shuffle_batches = False
        else:
            batch_size_used = int(min(int(args.vqbr_batch_size), n_train_samples))
            shuffle_batches = bool(args.vqbr_shuffle_batches)
        configs.append(
            {
                "method_key": str(setting["method_key"]),
                "optimizer": str(setting["optimizer"]),
                "batch_mode": str(setting["batch_mode"]),
                "batch_size_used": int(batch_size_used),
                "shuffle_batches": bool(shuffle_batches),
            }
        )
    return configs


def _run_one_seed_one_prior(
    *,
    seed: int,
    prior_case: str,
    X: np.ndarray,
    y: np.ndarray,
    splits: SplitData,
    V_grid: np.ndarray,
    lambda_grid: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    X_train_raw = X[splits.train_indices]
    y_train_raw = y[splits.train_indices]
    X_test_raw = X[splits.test_indices]
    y_test_raw = y[splits.test_indices]

    X_tune_train_raw = X[splits.train_val_indices]
    y_tune_train_raw = y[splits.train_val_indices]
    X_val_raw = X[splits.val_indices]
    y_val_raw = y[splits.val_indices]

    # Inner split preprocessing for hyperparameter selection (training data only).
    tune_mean, tune_std = _fit_feature_standardizer(X_tune_train_raw)
    X_tune_train = _apply_feature_standardizer(X_tune_train_raw, tune_mean, tune_std)
    X_val = _apply_feature_standardizer(X_val_raw, tune_mean, tune_std)
    y_tune_mean, y_tune_train_centered = _fit_and_center_target(y_tune_train_raw)

    selected = _select_hyperparameters(
        prior_case=prior_case,
        seed=seed,
        X_train_for_tuning=X_tune_train,
        y_train_for_tuning_centered=y_tune_train_centered,
        y_train_for_tuning_mean=y_tune_mean,
        X_train_for_prior_variance=X_tune_train_raw,
        X_val=X_val,
        y_val_raw=y_val_raw,
        V_grid=V_grid,
        lambda_grid=lambda_grid,
        prior_delta=args.prior_delta,
        log_realtime=args.log_realtime,
        log_hyperparam_sweep=args.log_hyperparam_sweep,
    )

    # Final train/test preprocessing with scaler fitted only on the outer train split.
    train_mean, train_std = _fit_feature_standardizer(X_train_raw)
    X_train = _apply_feature_standardizer(X_train_raw, train_mean, train_std)
    X_test = _apply_feature_standardizer(X_test_raw, train_mean, train_std)
    y_train_mean, y_train_centered = _fit_and_center_target(y_train_raw)

    m0_train, sigma_diag_train = _build_prior(
        prior_case=prior_case,
        X_reference=X_train,
        lambda_strength=selected["lambda_strength"],
        delta=args.prior_delta,
        variance_reference=X_train_raw,
    )
    V_selected = float(selected["V"])

    # Closed-form MAP reference solution.
    closed_form = ClosedFormMAPBayesianRegression()
    closed_form.fit(X_train, y_train_centered, m0_train, sigma_diag_train, V_selected)
    w_star = closed_form.get_weights()

    methods: Dict[str, Dict[str, Any]] = {}
    y_train_pred_closed_form = np.asarray(closed_form.predict(X_train), dtype=float) + float(y_train_mean)
    y_test_pred_closed_form = np.asarray(closed_form.predict(X_test), dtype=float) + float(y_train_mean)
    methods["closed_form"] = {
        "train_mse": _mse(y_train_raw, y_train_pred_closed_form),
        "test_mse": _mse(y_test_raw, y_test_pred_closed_form),
    }
    if args.log_realtime:
        print(
            "    "
            f"[{prior_case}][seed={seed}][closed_form] "
            f"train_mse={methods['closed_form']['train_mse']:.6f}, "
            f"test_mse={methods['closed_form']['test_mse']:.6f}",
            flush=True,
        )

    vqbr_module = importlib.import_module("vqbr")
    VariationalQuantumBayesianRegression = getattr(
        vqbr_module, "VariationalQuantumBayesianRegression"
    )
    optimizer_configs = _resolve_vqbr_optimizer_configs(
        args=args,
        n_train_samples=int(X_train.shape[0]),
    )
    vqbr_loss = _normalize_vqbr_loss(args.vqbr_loss)
    for opt_cfg in optimizer_configs:
        method_key = str(opt_cfg["method_key"])
        optimizer_name = str(opt_cfg["optimizer"])
        batch_mode = str(opt_cfg["batch_mode"])
        batch_size_used = int(opt_cfg["batch_size_used"])
        shuffle_batches = bool(opt_cfg["shuffle_batches"])

        if args.log_realtime:
            print(
                "    "
                f"[{prior_case}][seed={seed}][{method_key}] "
                f"training start: optimizer={optimizer_name}, "
                f"loss={vqbr_loss}, "
                f"mode={batch_mode}, batch_size={batch_size_used}, "
                f"shuffle_batches={shuffle_batches}",
                flush=True,
            )

        vqbr = VariationalQuantumBayesianRegression(
            shots=args.vqbr_shots,
            batch_size=batch_size_used,
            reps=args.vqbr_reps,
            optimizer=optimizer_name,
            maxiter=args.vqbr_maxiter,
            shuffle_batches=shuffle_batches,
            learning_rate=args.vqbr_learning_rate,
            loss=vqbr_loss,
            random_state=seed,
            use_shot_noise=args.vqbr_use_shot_noise,
            verbose=args.vqbr_verbose,
        )
        iteration_callback = None
        if args.log_realtime:
            iteration_callback = _make_vqbr_iteration_logger(
                prior_case=prior_case,
                seed=seed,
                method_key=method_key,
                optimizer=optimizer_name,
                batch_mode=batch_mode,
                batch_size=batch_size_used,
                log_every_iter=args.log_every_iter,
            )
        fitted_state = vqbr.fit(
            X_train,
            y_train_centered,
            m0_train,
            sigma_diag_train,
            V_selected,
            iteration_callback=iteration_callback,
        )
        trained_circuit = _build_trained_circuit(vqbr)
        statevector_from_circuit = _simulate_statevector_from_circuit(trained_circuit)

        fit_state = np.asarray(fitted_state, dtype=complex).reshape(-1)
        overlap = np.vdot(statevector_from_circuit, fit_state)
        phase = np.exp(-1j * np.angle(overlap)) if np.abs(overlap) > EPS else 1.0 + 0.0j
        fit_state_aligned = fit_state * phase
        fit_vs_sim_l2 = float(np.linalg.norm(fit_state_aligned - statevector_from_circuit))

        reconstructed_vector, encoding_vector = _reconstruct_vqbr_vector_from_state(
            statevector_from_circuit,
            w_star=w_star,
        )
        batch_loss_history = _collect_batch_loss_history(vqbr.history_)
        finite_batch_losses = [x for x in batch_loss_history if np.isfinite(x)]
        final_batch_L_hat = finite_batch_losses[-1] if finite_batch_losses else float("nan")
        vqbr_metrics = _evaluate_vqbr_metrics(
            reconstructed_vector=reconstructed_vector,
            encoding_vector=encoding_vector,
            w_star=w_star,
            X_train=X_train,
            y_train_raw=y_train_raw,
            X_test=X_test,
            y_test_raw=y_test_raw,
            y_train_mean=y_train_mean,
            final_batch_L_hat=final_batch_L_hat,
            batch_loss_history=batch_loss_history,
        )
        vqbr_metrics["optimizer"] = optimizer_name
        vqbr_metrics["batch_mode"] = batch_mode
        vqbr_metrics["batch_size_used"] = int(batch_size_used)
        vqbr_metrics["shuffle_batches"] = bool(shuffle_batches)
        vqbr_metrics["loss"] = str(vqbr_loss)
        vqbr_metrics["trained_circuit"] = _serialize_trained_circuit(trained_circuit)
        vqbr_metrics["statevector_source"] = "simulated_from_trained_circuit"
        vqbr_metrics["fit_vs_sim_state_l2"] = fit_vs_sim_l2
        methods[method_key] = vqbr_metrics

        if args.log_realtime:
            print(
                "    "
                f"[{prior_case}][seed={seed}][{method_key}] "
                f"cos={vqbr_metrics['cosine_similarity']:.6f}, "
                f"rel_l2={vqbr_metrics['relative_l2_distance']:.6e}, "
                f"train_mse={vqbr_metrics['train_mse']:.6f}, "
                f"test_mse={vqbr_metrics['test_mse']:.6f}, "
                f"batch_L_hat={vqbr_metrics['batch_L_hat']:.6e}",
                flush=True,
            )

    return {
        "seed": int(seed),
        "split": {
            "n_train_total": int(X_train.shape[0]),
            "n_train_for_hyperparam": int(X_tune_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
            "train_ratio": float(args.train_ratio),
            "test_ratio": float(1.0 - args.train_ratio),
            "val_ratio_within_train": float(args.val_ratio_within_train),
        },
        "preprocessing": {
            "feature_standardization": {
                "fitted_on": "outer_train_split",
                "mean": [float(x) for x in train_mean],
                "std": [float(x) for x in train_std],
            },
            "target_centering": {
                "fitted_on": "outer_train_split",
                "mean": float(y_train_mean),
            },
        },
        "selected_hyperparameters": {
            "V": float(selected["V"]),
            "lambda_strength": float(selected["lambda_strength"]),
            "validation_mse": float(selected["validation_mse"]),
            "prior_delta": float(args.prior_delta),
        },
        "methods": methods,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = Path(args.dataset_path).resolve()
    output_path = Path(args.output_path).resolve()
    if args.log_every_iter <= 0:
        raise ValueError("--log-every-iter must be a positive integer.")
    if args.vqbr_batch_size <= 0:
        raise ValueError("--vqbr-batch-size must be a positive integer.")
    vqbr_loss = _normalize_vqbr_loss(args.vqbr_loss)
    selected_optimizer_settings = _select_vqbr_optimizer_settings(args.vqbr_optimizer)
    selected_method_keys = [str(setting["method_key"]) for setting in selected_optimizer_settings]
    reported_method_keys = ["closed_form"] + selected_method_keys

    seeds = _resolve_seeds(
        num_seeds=args.num_seeds,
        seed_offset=args.seed_offset,
        explicit_seeds=args.seeds,
    )
    V_grid = _parse_float_grid(args.V_grid)
    lambda_grid = _parse_float_grid(args.lambda_grid)

    X, y, feature_cols = _load_naval_dataset(dataset_path=dataset_path, target_col=args.target_col)
    n_samples, n_features = X.shape

    result: Dict[str, Any] = {
        "dataset": {
            "path": str(dataset_path),
            "target_col": args.target_col,
            "feature_cols": feature_cols,
            "n_samples": int(n_samples),
            "n_features": int(n_features),
        },
        "config": {
            "num_seeds": int(len(seeds)),
            "seeds": [int(s) for s in seeds],
            "prior_cases": list(PRIOR_CASES),
            "train_ratio": float(args.train_ratio),
            "val_ratio_within_train": float(args.val_ratio_within_train),
            "V_grid": [float(x) for x in V_grid],
            "lambda_grid": [float(x) for x in lambda_grid],
            "prior_delta": float(args.prior_delta),
            "methods": reported_method_keys,
            "vqbr": {
                "shots": int(args.vqbr_shots),
                "batch_size": int(args.vqbr_batch_size),
                "reps": int(args.vqbr_reps),
                "maxiter": int(args.vqbr_maxiter),
                "shuffle_batches": bool(args.vqbr_shuffle_batches),
                "learning_rate": float(args.vqbr_learning_rate),
                "loss": str(vqbr_loss),
                "use_shot_noise": bool(args.vqbr_use_shot_noise),
                "optimizers": [
                    {
                        "method_key": str(setting["method_key"]),
                        "optimizer": str(setting["optimizer"]).upper(),
                        "batch_mode": str(setting["batch_mode"]),
                        "batch_size": (
                            "n_train_total"
                            if bool(setting["use_full_batch"])
                            else int(args.vqbr_batch_size)
                        ),
                        "shuffle_batches": (
                            False
                            if bool(setting["use_full_batch"])
                            else bool(args.vqbr_shuffle_batches)
                        ),
                    }
                    for setting in selected_optimizer_settings
                ],
            },
            "logging": {
                "realtime": bool(args.log_realtime),
                "log_every_iter": int(args.log_every_iter),
                "log_hyperparam_sweep": bool(args.log_hyperparam_sweep),
            },
            "preprocessing": {
                "feature_standardization": True,
                "target_centering": True,
                "standardization_scope": "fit on training split only, then applied to validation/test",
            },
        },
        "prior_cases": {},
    }

    print("=== Naval Experiment ===")
    print(f"Dataset: {dataset_path}")
    print(f"Target: {args.target_col}")
    print(f"Shape: N={n_samples}, D={n_features}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Train/Test split: {args.train_ratio:.2f}/{1.0 - args.train_ratio:.2f}")
    print(f"Inner validation ratio (within train): {args.val_ratio_within_train:.2f}")
    print("VQBR config:")
    print(
        "  "
        f"shots={int(args.vqbr_shots)}, "
        f"batch_size={int(args.vqbr_batch_size)}, "
        f"reps={int(args.vqbr_reps)}, "
        f"maxiter={int(args.vqbr_maxiter)}"
    )
    print("  optimizer runs:")
    for setting in selected_optimizer_settings:
        method_key = str(setting["method_key"])
        optimizer = str(setting["optimizer"]).upper()
        batch_mode = str(setting["batch_mode"])
        if bool(setting["use_full_batch"]):
            batch_size_desc = "n_train_total"
            shuffle_batches = False
        else:
            batch_size_desc = f"min({int(args.vqbr_batch_size)}, n_train_total)"
            shuffle_batches = bool(args.vqbr_shuffle_batches)
        print(
            "    "
            f"{method_key}: optimizer={optimizer}, mode={batch_mode}, "
            f"batch_size={batch_size_desc}, shuffle_batches={shuffle_batches}"
        )
    print(
        "  "
        f"shuffle_batches={bool(args.vqbr_shuffle_batches)}, "
        f"learning_rate={float(args.vqbr_learning_rate):.6g}, "
        f"loss={vqbr_loss}"
    )
    print(
        "  "
        f"use_shot_noise={bool(args.vqbr_use_shot_noise)}, "
        f"verbose={bool(args.vqbr_verbose)}, "
        f"log_every_iter={int(args.log_every_iter)}"
    )
    print("")

    for prior_case in PRIOR_CASES:
        print(f"[Prior case: {prior_case}]")
        case_seed_results: List[Dict[str, Any]] = []

        for i, seed in enumerate(seeds, start=1):
            print(f"  Seed {seed} ({i}/{len(seeds)})", flush=True)
            splits = _make_splits(
                n_samples=n_samples,
                train_ratio=args.train_ratio,
                val_ratio_within_train=args.val_ratio_within_train,
                seed=seed,
            )
            seed_result = _run_one_seed_one_prior(
                seed=seed,
                prior_case=prior_case,
                X=X,
                y=y,
                splits=splits,
                V_grid=V_grid,
                lambda_grid=lambda_grid,
                args=args,
            )
            case_seed_results.append(seed_result)

        result["prior_cases"][prior_case] = {
            "per_seed": case_seed_results,
            "aggregate": _aggregate_case_results(case_seed_results),
        }
        print("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=args.json_indent)

    print(f"Saved results to: {output_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run closed-form and VQBR on "
            "the Naval propulsion dataset with 80/20 train/test splits over multiple seeds."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(ROOT_DIR / "data" / "real" / "naval" / "data.txt"),
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="kMc",
        choices=tuple(sorted(NAVAL_TARGET_COL_TO_INDEX.keys())),
        help="Target to predict: kMc (compressor decay) or kMt (turbine decay).",
    )

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio-within-train", type=float, default=0.1)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated explicit seed list. If provided, overrides --num-seeds/--seed-offset.",
    )

    parser.add_argument(
        "--V-grid",
        type=str,
        default="1e-4,1e-3,1e-2,1e-1,1,10",
    )
    parser.add_argument(
        "--lambda-grid",
        type=str,
        default="1e-4,1e-3,1e-2,1e-1,1,10,100",
    )
    parser.add_argument("--prior-delta", type=float, default=1e-8)

    parser.add_argument("--vqbr-shots", type=int, default=4096)
    parser.add_argument("--vqbr-batch-size", type=int, default=100)
    parser.add_argument("--vqbr-reps", type=int, default=2)
    parser.add_argument(
        "--vqbr-optimizer",
        "--vqbr-optimizers",
        dest="vqbr_optimizer",
        type=str,
        default="COBYLA,SPSA",
        help=(
            "Comma-separated VQBR optimizers to run. "
            "Supported values: COBYLA,SPSA (case-insensitive)."
        ),
    )
    parser.add_argument("--vqbr-maxiter", type=int, default=200)
    parser.add_argument("--vqbr-shuffle-batches", action="store_true")
    parser.add_argument("--no-vqbr-shuffle-batches", dest="vqbr_shuffle_batches", action="store_false")
    parser.set_defaults(vqbr_shuffle_batches=True)
    parser.add_argument("--vqbr-learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--vqbr-loss",
        type=str,
        default="log_ratio",
        help=(
            "VQBR objective form. "
            "Supported values: log_ratio, neg_ratio "
            "(aliases: log, ratio, raw_ratio)."
        ),
    )
    parser.add_argument("--vqbr-use-shot-noise", action="store_true")
    parser.add_argument("--no-vqbr-shot-noise", dest="vqbr_use_shot_noise", action="store_false")
    parser.set_defaults(vqbr_use_shot_noise=True)
    parser.add_argument("--vqbr-verbose", action="store_true")

    parser.add_argument("--log-realtime", action="store_true")
    parser.add_argument("--no-log-realtime", dest="log_realtime", action="store_false")
    parser.set_defaults(log_realtime=True)
    parser.add_argument(
        "--log-every-iter",
        type=int,
        default=5,
        help="Log VQBR progress every K iterations when realtime logging is enabled.",
    )
    parser.add_argument(
        "--log-hyperparam-sweep",
        action="store_true",
        help="Print every (V, lambda) validation MSE during inner hyperparameter search.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=str(ROOT_DIR / "results" / "naval" / "naval_experiment_results.json"),
    )
    parser.add_argument("--json-indent", type=int, default=2)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
