from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, TextIO, Tuple

import numpy as np
import pandas as pd

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


@dataclass(frozen=True)
class VQBRGridConfig:
    config_id: str
    shots: int
    maxiter: int
    reps: int
    su2_gates: Tuple[str, ...]
    entanglement: str
    loss: str


@dataclass
class PreparedSeedRun:
    seed: int
    split: Dict[str, Any]
    preprocessing: Dict[str, Any]
    selected_hyperparameters: Dict[str, float]
    X_train: np.ndarray
    y_train_raw: np.ndarray
    X_test: np.ndarray
    y_test_raw: np.ndarray
    y_train_centered: np.ndarray
    y_train_mean: float
    m0_train: np.ndarray
    sigma_diag_train: np.ndarray
    V_selected: float
    w_star: np.ndarray
    closed_form_metrics: Dict[str, float]


class _TeeStream:
    def __init__(self, streams: List[TextIO]) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    @property
    def encoding(self) -> str:
        for stream in self._streams:
            enc = getattr(stream, "encoding", None)
            if isinstance(enc, str):
                return enc
        return "utf-8"


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
    process_pid = int(os.getpid())
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
            cuda_memory = _query_cuda_memory_usage_mb(process_pid)
            print(
                "    "
                f"[{prior_case}][seed={seed}][{method_key}] "
                f"{optimizer} iter {iteration:>4d} | "
                f"mode={batch_mode}, batch_size={batch_size} | "
                f"batch_loss={batch_loss: .6e}, {delta_text}, cuda_mem={cuda_memory}",
                flush=True,
            )
        prev_batch_loss = batch_loss

    return _callback


def _query_cuda_memory_usage_mb(pid: int) -> str:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except Exception:
        return "n/a"

    if result.returncode != 0:
        return "n/a"

    total_used_mb = 0.0
    found = False
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            line_pid = int(parts[0])
            used_mb = float(parts[1])
        except ValueError:
            continue
        if line_pid != pid:
            continue
        total_used_mb += used_mb
        found = True

    if not found:
        return "0 MiB"
    if float(total_used_mb).is_integer():
        return f"{int(total_used_mb)} MiB"
    return f"{total_used_mb:.1f} MiB"


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


def _parse_vqbr_su2_gates(raw: str) -> List[str]:
    gates = [token.strip().lower() for token in str(raw).split(",") if token.strip()]
    if not gates:
        raise ValueError("--vqbr-su2-gates must include at least one gate name.")
    return gates


def _parse_int_grid(raw: str, *, arg_name: str) -> List[int]:
    tokens = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not tokens:
        raise ValueError(f"{arg_name} must include at least one integer value.")
    values: List[int] = []
    for token in tokens:
        value = int(token)
        if value <= 0:
            raise ValueError(f"{arg_name} values must be strictly positive; got {value}.")
        values.append(value)

    unique: List[int] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _parse_str_grid(raw: str, *, arg_name: str) -> List[str]:
    tokens = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not tokens:
        raise ValueError(f"{arg_name} must include at least one value.")
    unique: List[str] = []
    for token in tokens:
        if token not in unique:
            unique.append(token)
    return unique


def _parse_vqbr_su2_gates_grid(raw: str) -> List[Tuple[str, ...]]:
    text = str(raw).strip()
    if text == "":
        raise ValueError("--vqbr-su2-gates-grid must include at least one gate set.")
    group_tokens = [tok.strip() for tok in re.split(r"[;|]", text) if tok.strip()]
    if not group_tokens:
        raise ValueError("--vqbr-su2-gates-grid must include at least one gate set.")

    groups: List[Tuple[str, ...]] = []
    for token in group_tokens:
        gates = tuple(_parse_vqbr_su2_gates(token))
        if gates not in groups:
            groups.append(gates)
    return groups


def _parse_vqbr_loss_grid(raw: str) -> List[str]:
    tokens = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not tokens:
        raise ValueError("--vqbr-loss-grid must include at least one loss value.")

    values: List[str] = []
    for token in tokens:
        normalized = _normalize_vqbr_loss(token)
        if normalized not in values:
            values.append(normalized)
    return values


def _format_su2_gates(gates: Tuple[str, ...]) -> str:
    return ",".join(str(g) for g in gates)


def _grid_config_to_dict(config: VQBRGridConfig) -> Dict[str, Any]:
    return {
        "config_id": str(config.config_id),
        "optimizer": "COBYLA",
        "shots": int(config.shots),
        "maxiter": int(config.maxiter),
        "reps": int(config.reps),
        "su2_gates": [str(g) for g in config.su2_gates],
        "su2_gates_text": _format_su2_gates(config.su2_gates),
        "entanglement": str(config.entanglement),
        "loss": str(config.loss),
    }


def _build_vqbr_grid_configs(args: argparse.Namespace) -> List[VQBRGridConfig]:
    shots_grid = _parse_int_grid(args.vqbr_shots_grid, arg_name="--vqbr-shots-grid")
    maxiter_grid = _parse_int_grid(args.vqbr_maxiter_grid, arg_name="--vqbr-maxiter-grid")
    reps_grid = _parse_int_grid(args.vqbr_reps_grid, arg_name="--vqbr-reps-grid")
    su2_grid = _parse_vqbr_su2_gates_grid(args.vqbr_su2_gates_grid)
    entanglement_grid = _parse_str_grid(
        args.vqbr_entanglement_grid,
        arg_name="--vqbr-entanglement-grid",
    )
    loss_grid = _parse_vqbr_loss_grid(args.vqbr_loss_grid)

    grid_configs: List[VQBRGridConfig] = []
    for idx, (shots, maxiter, reps, su2_gates, entanglement, loss) in enumerate(
        itertools.product(
            shots_grid,
            maxiter_grid,
            reps_grid,
            su2_grid,
            entanglement_grid,
            loss_grid,
        ),
        start=1,
    ):
        grid_configs.append(
            VQBRGridConfig(
                config_id=f"cfg_{idx:04d}",
                shots=int(shots),
                maxiter=int(maxiter),
                reps=int(reps),
                su2_gates=tuple(str(g).lower() for g in su2_gates),
                entanglement=str(entanglement),
                loss=str(loss),
            )
        )
    return grid_configs


def _load_energy_dataset(dataset_path: Path, target_col: str) -> tuple[np.ndarray, np.ndarray, List[str]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_excel(dataset_path)
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {list(df.columns)}"
        )

    feature_cols = [c for c in df.columns if str(c).startswith("X")]
    if not feature_cols:
        feature_cols = [c for c in df.columns if c != target_col]
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    if not feature_cols:
        raise ValueError("No feature columns found in the dataset.")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float).reshape(-1)
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
    vqbr_su2_gates = _parse_vqbr_su2_gates(args.vqbr_su2_gates)
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
                f"su2_gates={vqbr_su2_gates}, "
                f"mode={batch_mode}, batch_size={batch_size_used}, "
                f"shuffle_batches={shuffle_batches}",
                flush=True,
            )

        vqbr = VariationalQuantumBayesianRegression(
            shots=int(getattr(args, "vqbr_shots", 4096)),
            batch_size=batch_size_used,
            reps=args.vqbr_reps,
            optimizer=optimizer_name,
            maxiter=args.vqbr_maxiter,
            shuffle_batches=shuffle_batches,
            learning_rate=args.vqbr_learning_rate,
            loss=vqbr_loss,
            su2_gates=vqbr_su2_gates,
            random_state=seed,
            use_shot_noise=args.vqbr_use_shot_noise,
            verbose=args.vqbr_verbose,
            prefer_gpu=args.vqbr_prefer_gpu,
            require_gpu=args.vqbr_require_gpu,
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
        vqbr_metrics["simulator_device"] = str(getattr(vqbr, "simulator_device_", "CPU"))
        vqbr_metrics["simulator_backend"] = str(
            getattr(vqbr, "simulator_backend_", "statevector_cpu")
        )
        vqbr_metrics["simulator_reason"] = str(
            getattr(vqbr, "simulator_reason_", "Using CPU statevector simulator.")
        )
        vqbr_metrics["trained_circuit"] = _serialize_trained_circuit(trained_circuit)
        vqbr_metrics["statevector_source"] = "simulated_from_trained_circuit"
        vqbr_metrics["fit_vs_sim_state_l2"] = fit_vs_sim_l2
        methods[method_key] = vqbr_metrics

        if args.log_realtime:
            print(
                "    "
                f"[{prior_case}][seed={seed}][{method_key}] "
                f"simulator={vqbr_metrics['simulator_device']} "
                f"({vqbr_metrics['simulator_backend']}), "
                f"reason={vqbr_metrics['simulator_reason']}, "
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


def _prepare_seed_run(
    *,
    seed: int,
    prior_case: str,
    X: np.ndarray,
    y: np.ndarray,
    splits: SplitData,
    V_grid: np.ndarray,
    lambda_grid: np.ndarray,
    args: argparse.Namespace,
) -> PreparedSeedRun:
    X_train_raw = X[splits.train_indices]
    y_train_raw = y[splits.train_indices]
    X_test_raw = X[splits.test_indices]
    y_test_raw = y[splits.test_indices]

    X_tune_train_raw = X[splits.train_val_indices]
    y_tune_train_raw = y[splits.train_val_indices]
    X_val_raw = X[splits.val_indices]
    y_val_raw = y[splits.val_indices]

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

    closed_form = ClosedFormMAPBayesianRegression()
    closed_form.fit(X_train, y_train_centered, m0_train, sigma_diag_train, V_selected)
    w_star = closed_form.get_weights()

    y_train_pred_closed_form = np.asarray(closed_form.predict(X_train), dtype=float) + float(y_train_mean)
    y_test_pred_closed_form = np.asarray(closed_form.predict(X_test), dtype=float) + float(y_train_mean)
    closed_form_metrics = {
        "train_mse": _mse(y_train_raw, y_train_pred_closed_form),
        "test_mse": _mse(y_test_raw, y_test_pred_closed_form),
    }
    if args.log_realtime:
        print(
            "    "
            f"[{prior_case}][seed={seed}][closed_form] "
            f"train_mse={closed_form_metrics['train_mse']:.6f}, "
            f"test_mse={closed_form_metrics['test_mse']:.6f}",
            flush=True,
        )

    return PreparedSeedRun(
        seed=int(seed),
        split={
            "n_train_total": int(X_train.shape[0]),
            "n_train_for_hyperparam": int(X_tune_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
            "train_ratio": float(args.train_ratio),
            "test_ratio": float(1.0 - args.train_ratio),
            "val_ratio_within_train": float(args.val_ratio_within_train),
        },
        preprocessing={
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
        selected_hyperparameters={
            "V": float(selected["V"]),
            "lambda_strength": float(selected["lambda_strength"]),
            "validation_mse": float(selected["validation_mse"]),
            "prior_delta": float(args.prior_delta),
        },
        X_train=X_train,
        y_train_raw=y_train_raw,
        X_test=X_test,
        y_test_raw=y_test_raw,
        y_train_centered=y_train_centered,
        y_train_mean=float(y_train_mean),
        m0_train=m0_train,
        sigma_diag_train=sigma_diag_train,
        V_selected=float(V_selected),
        w_star=np.asarray(w_star, dtype=float),
        closed_form_metrics=closed_form_metrics,
    )


def _run_vqbr_for_grid_config(
    *,
    prior_case: str,
    prepared: PreparedSeedRun,
    config: VQBRGridConfig,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    vqbr_module = importlib.import_module("vqbr")
    VariationalQuantumBayesianRegression = getattr(
        vqbr_module, "VariationalQuantumBayesianRegression"
    )

    method_key = "vqbr_cobyla"
    optimizer_name = "COBYLA"
    batch_mode = "full_data"
    batch_size_used = int(prepared.X_train.shape[0])
    shuffle_batches = False
    su2_gates = list(config.su2_gates)

    if args.log_realtime:
        print(
            "    "
            f"[{prior_case}][seed={prepared.seed}][{config.config_id}] "
            f"training start: optimizer={optimizer_name}, "
            f"shots={config.shots}, "
            f"loss={config.loss}, "
            f"su2_gates={su2_gates}, "
            f"entanglement={config.entanglement}, "
            f"reps={config.reps}, "
            f"maxiter={config.maxiter}",
            flush=True,
        )

    vqbr = VariationalQuantumBayesianRegression(
        shots=config.shots,
        batch_size=batch_size_used,
        reps=config.reps,
        optimizer=optimizer_name,
        maxiter=config.maxiter,
        shuffle_batches=shuffle_batches,
        learning_rate=args.vqbr_learning_rate,
        loss=config.loss,
        su2_gates=su2_gates,
        entanglement=config.entanglement,
        random_state=prepared.seed,
        use_shot_noise=args.vqbr_use_shot_noise,
        verbose=args.vqbr_verbose,
        prefer_gpu=args.vqbr_prefer_gpu,
        require_gpu=args.vqbr_require_gpu,
    )
    iteration_callback = None
    if args.log_realtime:
        iteration_callback = _make_vqbr_iteration_logger(
            prior_case=prior_case,
            seed=prepared.seed,
            method_key=f"{method_key}:{config.config_id}",
            optimizer=optimizer_name,
            batch_mode=batch_mode,
            batch_size=batch_size_used,
            log_every_iter=args.log_every_iter,
        )
    fitted_state = vqbr.fit(
        prepared.X_train,
        prepared.y_train_centered,
        prepared.m0_train,
        prepared.sigma_diag_train,
        prepared.V_selected,
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
        w_star=prepared.w_star,
    )
    batch_loss_history = _collect_batch_loss_history(vqbr.history_)
    finite_batch_losses = [x for x in batch_loss_history if np.isfinite(x)]
    final_batch_L_hat = finite_batch_losses[-1] if finite_batch_losses else float("nan")
    vqbr_metrics = _evaluate_vqbr_metrics(
        reconstructed_vector=reconstructed_vector,
        encoding_vector=encoding_vector,
        w_star=prepared.w_star,
        X_train=prepared.X_train,
        y_train_raw=prepared.y_train_raw,
        X_test=prepared.X_test,
        y_test_raw=prepared.y_test_raw,
        y_train_mean=prepared.y_train_mean,
        final_batch_L_hat=final_batch_L_hat,
        batch_loss_history=batch_loss_history,
    )
    vqbr_metrics["optimizer"] = optimizer_name
    vqbr_metrics["batch_mode"] = batch_mode
    vqbr_metrics["batch_size_used"] = int(batch_size_used)
    vqbr_metrics["shuffle_batches"] = bool(shuffle_batches)
    vqbr_metrics["shots"] = int(config.shots)
    vqbr_metrics["loss"] = str(config.loss)
    vqbr_metrics["su2_gates"] = [str(g) for g in su2_gates]
    vqbr_metrics["entanglement"] = str(config.entanglement)
    vqbr_metrics["reps"] = int(config.reps)
    vqbr_metrics["maxiter"] = int(config.maxiter)
    vqbr_metrics["config_id"] = str(config.config_id)
    vqbr_metrics["simulator_device"] = str(getattr(vqbr, "simulator_device_", "CPU"))
    vqbr_metrics["simulator_backend"] = str(
        getattr(vqbr, "simulator_backend_", "statevector_cpu")
    )
    vqbr_metrics["simulator_reason"] = str(
        getattr(vqbr, "simulator_reason_", "Using CPU statevector simulator.")
    )
    vqbr_metrics["fit_vs_sim_state_l2"] = fit_vs_sim_l2
    # Keep JSON artifacts compact when sweeping many configurations.
    vqbr_metrics.pop("encoding_vector", None)
    vqbr_metrics.pop("reconstructed_vector", None)

    if args.log_realtime:
        print(
            "    "
            f"[{prior_case}][seed={prepared.seed}][{config.config_id}] "
            f"simulator={vqbr_metrics['simulator_device']} "
            f"({vqbr_metrics['simulator_backend']}), "
            f"cos={vqbr_metrics['cosine_similarity']:.6f}, "
            f"rel_l2={vqbr_metrics['relative_l2_distance']:.6e}, "
            f"train_mse={vqbr_metrics['train_mse']:.6f}, "
            f"test_mse={vqbr_metrics['test_mse']:.6f}, "
            f"batch_L_hat={vqbr_metrics['batch_L_hat']:.6e}",
            flush=True,
        )
    return vqbr_metrics


def _compact_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _aggregate_metric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
    }


def _aggregate_closed_form_results(seed_runs: List[PreparedSeedRun]) -> Dict[str, Dict[str, float]]:
    return {
        "train_mse": _aggregate_metric(
            [float(seed_run.closed_form_metrics["train_mse"]) for seed_run in seed_runs]
        ),
        "test_mse": _aggregate_metric(
            [float(seed_run.closed_form_metrics["test_mse"]) for seed_run in seed_runs]
        ),
    }


def _build_training_row(
    *,
    prior_case: str,
    prepared: PreparedSeedRun,
    config: VQBRGridConfig,
    vqbr_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "prior_case": str(prior_case),
        "seed": int(prepared.seed),
        "config_id": str(config.config_id),
        "optimizer": "COBYLA",
        "shots": int(config.shots),
        "maxiter": int(config.maxiter),
        "reps": int(config.reps),
        "su2_gates": _format_su2_gates(config.su2_gates),
        "entanglement": str(config.entanglement),
        "loss": str(config.loss),
        "n_train_total": int(prepared.split["n_train_total"]),
        "n_train_for_hyperparam": int(prepared.split["n_train_for_hyperparam"]),
        "n_val": int(prepared.split["n_val"]),
        "n_test": int(prepared.split["n_test"]),
        "train_ratio": float(prepared.split["train_ratio"]),
        "val_ratio_within_train": float(prepared.split["val_ratio_within_train"]),
        "selected_V": float(prepared.selected_hyperparameters["V"]),
        "selected_lambda_strength": float(prepared.selected_hyperparameters["lambda_strength"]),
        "selected_validation_mse": float(prepared.selected_hyperparameters["validation_mse"]),
        "closed_form_train_mse": float(prepared.closed_form_metrics["train_mse"]),
        "closed_form_test_mse": float(prepared.closed_form_metrics["test_mse"]),
        "cosine_similarity": float(vqbr_metrics["cosine_similarity"]),
        "relative_l2_distance": float(vqbr_metrics["relative_l2_distance"]),
        "train_mse": float(vqbr_metrics["train_mse"]),
        "test_mse": float(vqbr_metrics["test_mse"]),
        "batch_L_hat": float(vqbr_metrics["batch_L_hat"]),
        "iterations": int(vqbr_metrics["iterations"]),
        "fit_vs_sim_state_l2": float(vqbr_metrics.get("fit_vs_sim_state_l2", np.nan)),
        "simulator_device": str(vqbr_metrics.get("simulator_device", "CPU")),
        "simulator_backend": str(vqbr_metrics.get("simulator_backend", "statevector_cpu")),
        "simulator_reason": str(
            vqbr_metrics.get("simulator_reason", "Using CPU statevector simulator.")
        ),
        "batch_loss_history_json": _compact_json_dumps(vqbr_metrics.get("batch_loss_history", [])),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = Path(args.dataset_path).resolve()
    output_path = Path(args.output_path).resolve()
    training_csv_path = Path(args.training_csv_path).resolve()
    summary_csv_path = Path(args.summary_csv_path).resolve()
    raw_log_file_path = str(getattr(args, "log_file_path", "")).strip()
    resolved_log_file_path = (
        str(Path(raw_log_file_path).resolve()) if raw_log_file_path != "" else ""
    )

    if args.log_every_iter <= 0:
        raise ValueError("--log-every-iter must be a positive integer.")
    if args.vqbr_batch_size <= 0:
        raise ValueError("--vqbr-batch-size must be a positive integer.")

    optimizer = str(args.vqbr_optimizer).strip().upper()
    if optimizer != "COBYLA":
        raise ValueError(
            "This grid-search experiment currently fixes the optimizer to COBYLA. "
            f"Received --vqbr-optimizer={args.vqbr_optimizer!r}."
        )

    seeds = _resolve_seeds(
        num_seeds=args.num_seeds,
        seed_offset=args.seed_offset,
        explicit_seeds=args.seeds,
    )
    V_grid = _parse_float_grid(args.V_grid)
    lambda_grid = _parse_float_grid(args.lambda_grid)
    grid_configs = _build_vqbr_grid_configs(args)
    grid_config_dicts = [_grid_config_to_dict(cfg) for cfg in grid_configs]

    X, y, feature_cols = _load_energy_dataset(dataset_path=dataset_path, target_col=args.target_col)
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
            "methods": ["closed_form", "vqbr_cobyla"],
            "vqbr": {
                "optimizer": "COBYLA",
                "batch_size": int(args.vqbr_batch_size),
                "learning_rate": float(args.vqbr_learning_rate),
                "use_shot_noise": bool(args.vqbr_use_shot_noise),
                "prefer_gpu": bool(args.vqbr_prefer_gpu),
                "require_gpu": bool(args.vqbr_require_gpu),
                "grid": {
                    "num_configurations": int(len(grid_configs)),
                    "shots_grid": sorted({int(cfg.shots) for cfg in grid_configs}),
                    "maxiter_grid": sorted({int(cfg.maxiter) for cfg in grid_configs}),
                    "reps_grid": sorted({int(cfg.reps) for cfg in grid_configs}),
                    "su2_gates_grid": [
                        token.split(",")
                        for token in sorted({_format_su2_gates(cfg.su2_gates) for cfg in grid_configs})
                    ],
                    "entanglement_grid": sorted({str(cfg.entanglement) for cfg in grid_configs}),
                    "loss_grid": sorted({str(cfg.loss) for cfg in grid_configs}),
                    "configs": grid_config_dicts,
                },
            },
            "logging": {
                "realtime": bool(args.log_realtime),
                "log_every_iter": int(args.log_every_iter),
                "log_hyperparam_sweep": bool(args.log_hyperparam_sweep),
                "log_file_path": resolved_log_file_path,
            },
            "preprocessing": {
                "feature_standardization": True,
                "target_centering": True,
                "standardization_scope": "fit on training split only, then applied to validation/test",
            },
        },
        "prior_cases": {},
        "csv_outputs": {
            "training_info_path": str(training_csv_path),
            "summary_path": str(summary_csv_path),
        },
    }

    print("=== Energy Experiment (Grid Search) ===")
    print(f"Dataset: {dataset_path}")
    print(f"Target: {args.target_col}")
    print(f"Shape: N={n_samples}, D={n_features}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Train/Test split: {args.train_ratio:.2f}/{1.0 - args.train_ratio:.2f}")
    print(f"Inner validation ratio (within train): {args.val_ratio_within_train:.2f}")
    print("VQBR fixed settings:")
    print(
        "  "
        f"optimizer=COBYLA, "
        f"learning_rate={float(args.vqbr_learning_rate):.6g}, "
        f"use_shot_noise={bool(args.vqbr_use_shot_noise)}, "
        f"prefer_gpu={bool(args.vqbr_prefer_gpu)}, "
        f"require_gpu={bool(args.vqbr_require_gpu)}"
    )
    print(f"VQBR grid size: {len(grid_configs)} configurations")
    print(
        "  "
        f"shots={sorted({int(cfg.shots) for cfg in grid_configs})}, "
        f"maxiter={sorted({int(cfg.maxiter) for cfg in grid_configs})}, "
        f"reps={sorted({int(cfg.reps) for cfg in grid_configs})}, "
        f"entanglement={sorted({str(cfg.entanglement) for cfg in grid_configs})}, "
        f"loss={sorted({str(cfg.loss) for cfg in grid_configs})}"
    )
    print("")

    training_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for prior_case in PRIOR_CASES:
        print(f"[Prior case: {prior_case}]")
        case_seed_results: List[Dict[str, Any]] = []
        prepared_seed_runs: List[PreparedSeedRun] = []
        grid_metric_store: Dict[str, List[Dict[str, Any]]] = {cfg.config_id: [] for cfg in grid_configs}

        for seed_index, seed in enumerate(seeds, start=1):
            print(f"  Seed {seed} ({seed_index}/{len(seeds)})", flush=True)
            splits = _make_splits(
                n_samples=n_samples,
                train_ratio=args.train_ratio,
                val_ratio_within_train=args.val_ratio_within_train,
                seed=seed,
            )
            prepared = _prepare_seed_run(
                seed=seed,
                prior_case=prior_case,
                X=X,
                y=y,
                splits=splits,
                V_grid=V_grid,
                lambda_grid=lambda_grid,
                args=args,
            )
            prepared_seed_runs.append(prepared)

            seed_grid_results: List[Dict[str, Any]] = []
            for cfg_index, cfg in enumerate(grid_configs, start=1):
                print(
                    "    "
                    f"Config {cfg_index:>4d}/{len(grid_configs)}: {cfg.config_id} "
                    f"(shots={cfg.shots}, maxiter={cfg.maxiter}, reps={cfg.reps}, "
                    f"su2_gates={_format_su2_gates(cfg.su2_gates)}, "
                    f"entanglement={cfg.entanglement}, loss={cfg.loss})",
                    flush=True,
                )
                vqbr_metrics = _run_vqbr_for_grid_config(
                    prior_case=prior_case,
                    prepared=prepared,
                    config=cfg,
                    args=args,
                )
                grid_metric_store[cfg.config_id].append(vqbr_metrics)
                training_rows.append(
                    _build_training_row(
                        prior_case=prior_case,
                        prepared=prepared,
                        config=cfg,
                        vqbr_metrics=vqbr_metrics,
                    )
                )

                metrics_for_json = dict(vqbr_metrics)
                metrics_for_json.pop("batch_loss_history", None)
                metrics_for_json.pop("loss_history", None)
                seed_grid_results.append(
                    {
                        "config_id": cfg.config_id,
                        "config": _grid_config_to_dict(cfg),
                        "methods": {"vqbr_cobyla": metrics_for_json},
                    }
                )

            case_seed_results.append(
                {
                    "seed": int(seed),
                    "split": prepared.split,
                    "preprocessing": prepared.preprocessing,
                    "selected_hyperparameters": prepared.selected_hyperparameters,
                    "methods": {"closed_form": prepared.closed_form_metrics},
                    "grid_results": seed_grid_results,
                }
            )

        closed_form_aggregate = _aggregate_closed_form_results(prepared_seed_runs)
        grid_aggregate: List[Dict[str, Any]] = []
        metric_keys = (
            "cosine_similarity",
            "relative_l2_distance",
            "train_mse",
            "test_mse",
            "batch_L_hat",
            "iterations",
        )
        for cfg in grid_configs:
            runs = grid_metric_store[cfg.config_id]
            metric_aggregate = {
                metric: _aggregate_metric([float(run[metric]) for run in runs if metric in run])
                for metric in metric_keys
            }
            grid_aggregate.append(
                {
                    "config_id": cfg.config_id,
                    "config": _grid_config_to_dict(cfg),
                    "methods": {"vqbr_cobyla": metric_aggregate},
                }
            )
            summary_row: Dict[str, Any] = {
                "prior_case": str(prior_case),
                "config_id": str(cfg.config_id),
                "optimizer": "COBYLA",
                "shots": int(cfg.shots),
                "maxiter": int(cfg.maxiter),
                "reps": int(cfg.reps),
                "su2_gates": _format_su2_gates(cfg.su2_gates),
                "entanglement": str(cfg.entanglement),
                "loss": str(cfg.loss),
                "num_seeds": int(len(runs)),
                "closed_form_train_mse_mean": float(closed_form_aggregate["train_mse"]["mean"]),
                "closed_form_train_mse_std": float(closed_form_aggregate["train_mse"]["std"]),
                "closed_form_test_mse_mean": float(closed_form_aggregate["test_mse"]["mean"]),
                "closed_form_test_mse_std": float(closed_form_aggregate["test_mse"]["std"]),
            }
            for metric in metric_keys:
                summary_row[f"{metric}_mean"] = float(metric_aggregate[metric]["mean"])
                summary_row[f"{metric}_std"] = float(metric_aggregate[metric]["std"])
            summary_rows.append(summary_row)

        grid_aggregate.sort(
            key=lambda x: float(
                x["methods"]["vqbr_cobyla"].get("test_mse", {}).get("mean", float("inf"))
            )
        )
        result["prior_cases"][prior_case] = {
            "per_seed": case_seed_results,
            "closed_form_aggregate": closed_form_aggregate,
            "grid_aggregate": grid_aggregate,
        }
        print("")

    training_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    training_df = pd.DataFrame(training_rows)
    if not training_df.empty:
        training_df = training_df.sort_values(
            by=["prior_case", "config_id", "seed"],
            kind="stable",
        ).reset_index(drop=True)
    training_df.to_csv(training_csv_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["prior_case", "test_mse_mean", "config_id"],
            kind="stable",
        ).reset_index(drop=True)
        summary_df["rank_by_test_mse"] = (
            summary_df.groupby("prior_case")["test_mse_mean"]
            .rank(method="first", ascending=True)
            .astype(int)
        )
        first_cols = [
            "prior_case",
            "rank_by_test_mse",
            "config_id",
            "optimizer",
            "shots",
            "maxiter",
            "reps",
            "su2_gates",
            "entanglement",
            "loss",
            "num_seeds",
        ]
        remaining_cols = [c for c in summary_df.columns if c not in first_cols]
        summary_df = summary_df[first_cols + remaining_cols]
    summary_df.to_csv(summary_csv_path, index=False)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=args.json_indent)

    print(f"Saved JSON results to: {output_path}")
    print(f"Saved training CSV to: {training_csv_path}")
    print(f"Saved summary CSV to: {summary_csv_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run energy-dataset VQBR grid search (fixed optimizer: COBYLA) "
            "with train/test splits over multiple random seeds."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(ROOT_DIR / "data" / "real" / "energy" / "ENB2012_data.xlsx"),
    )
    parser.add_argument("--target-col", type=str, default="Y1")

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

    parser.add_argument(
        "--vqbr-shots-grid",
        type=str,
        default="256,512,1024,2048,4096,8192,16384,32768",
        help="Comma-separated measurement-shot grid.",
    )
    parser.add_argument(
        "--vqbr-batch-size",
        type=int,
        default=100,
        help="Retained for compatibility; COBYLA runs in full-data mode and ignores this value.",
    )
    parser.add_argument(
        "--vqbr-optimizer",
        "--vqbr-optimizers",
        dest="vqbr_optimizer",
        type=str,
        default="COBYLA",
        help="Optimizer is fixed to COBYLA for this grid-search script.",
    )
    parser.add_argument("--vqbr-learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--vqbr-maxiter-grid",
        type=str,
        default="50,100,200,300,400,800,1000,2000",
        help="Comma-separated maxiter grid.",
    )
    parser.add_argument(
        "--vqbr-reps-grid",
        type=str,
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated EfficientSU2 reps grid.",
    )
    parser.add_argument(
        "--vqbr-su2-gates-grid",
        type=str,
        default="ry;rx,y",
        help=(
            "Semicolon-separated list of SU(2) gate sets. "
            "Each set uses comma-separated gate names (example: 'ry;rx,y')."
        ),
    )
    parser.add_argument(
        "--vqbr-entanglement-grid",
        type=str,
        default="full,linear,reverse_linear,pairwise,circular,sca",
        help="Comma-separated entanglement grid for EfficientSU2.",
    )
    parser.add_argument(
        "--vqbr-loss-grid",
        type=str,
        default="log_ratio,neg_ratio",
        help="Comma-separated VQBR loss grid (log_ratio, neg_ratio).",
    )
    parser.add_argument("--vqbr-use-shot-noise", action="store_true")
    parser.add_argument("--no-vqbr-shot-noise", dest="vqbr_use_shot_noise", action="store_false")
    parser.set_defaults(vqbr_use_shot_noise=True)
    parser.add_argument("--vqbr-prefer-gpu", action="store_true")
    parser.add_argument("--no-vqbr-prefer-gpu", dest="vqbr_prefer_gpu", action="store_false")
    parser.set_defaults(vqbr_prefer_gpu=True)
    parser.add_argument(
        "--vqbr-require-gpu",
        action="store_true",
        help="Fail fast if GPU statevector simulation cannot be enabled.",
    )
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
        default=str(ROOT_DIR / "results" / "energy" / "energy_experiment_results.json"),
    )
    parser.add_argument(
        "--training-csv-path",
        type=str,
        default=str(ROOT_DIR / "results" / "energy" / "energy_grid_training_info.csv"),
    )
    parser.add_argument(
        "--summary-csv-path",
        type=str,
        default=str(ROOT_DIR / "results" / "energy" / "energy_grid_summary.csv"),
    )
    parser.add_argument(
        "--log-file-path",
        type=str,
        default=str(ROOT_DIR / "results" / "energy" / "energy_grid_run.log"),
        help="Mirror all console output (stdout and stderr) into this log file.",
    )
    parser.add_argument("--json-indent", type=int, default=2)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raw_log_path = str(args.log_file_path).strip()
    if raw_log_path == "":
        run_experiment(args)
        return

    log_path = Path(raw_log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = _TeeStream([original_stdout, log_file])
        sys.stderr = _TeeStream([original_stderr, log_file])
        try:
            print(f"Logging to: {log_path}", flush=True)
            run_experiment(args)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
