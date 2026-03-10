from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
METHOD_KEYS = ("vqbr_cobyla",)
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
)

PAIR_RE = re.compile(r"^N(?P<N>\d+)_D(?P<D>\d+)$")


@dataclass
class InnerSplit:
    train_tune_indices: np.ndarray
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


def _make_inner_split(n_train_total: int, val_ratio_within_train: float, seed: int) -> InnerSplit:
    if n_train_total <= 1:
        raise ValueError("Need at least 2 training samples for inner split.")
    if not (0.0 < val_ratio_within_train < 1.0):
        raise ValueError("val_ratio_within_train must be in (0, 1).")

    rng = np.random.default_rng(seed + 1_009)
    order = rng.permutation(n_train_total)
    n_val = int(np.floor(val_ratio_within_train * n_train_total))
    n_val = min(max(1, n_val), n_train_total - 1)
    val_indices = order[:n_val]
    train_tune_indices = order[n_val:]
    return InnerSplit(train_tune_indices=train_tune_indices, val_indices=val_indices)


def _make_vqbr_iteration_logger(
    *,
    prior_case: str,
    pair_name: str,
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
                f"[{pair_name}][{prior_case}][seed={seed}][{method_key}] "
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


def _parse_int_list(raw: str) -> List[int]:
    tokens = [x.strip() for x in raw.split(",") if x.strip()]
    if not tokens:
        return []
    return [int(x) for x in tokens]


def _resolve_seeds(
    explicit_seeds: str | None,
    num_seeds: int | None,
    seed_offset: int,
    manifest_seeds: Sequence[int],
) -> List[int]:
    if explicit_seeds is not None:
        seeds = [int(x.strip()) for x in explicit_seeds.split(",") if x.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but no valid integers were found.")
        return seeds

    if num_seeds is not None:
        if num_seeds <= 0:
            raise ValueError("--num-seeds must be positive when provided.")
        return [seed_offset + i for i in range(num_seeds)]

    if manifest_seeds:
        return [int(x) for x in manifest_seeds]

    raise ValueError(
        "Unable to resolve seed list. Provide --seeds, --num-seeds, or a manifest with seeds."
    )


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


class _TeeStream:
    """Mirror writes to multiple stream-like objects."""

    def __init__(self, *streams: Any):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


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


def _load_manifest(dataset_root: Path) -> Dict[str, Any]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Synthetic manifest not found at {manifest_path}. "
            "Generate data first with data/synthetic/data_synthetic.py"
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object.")
    return manifest


def _parse_pair_name(pair_name: str) -> tuple[int, int]:
    m = PAIR_RE.match(pair_name)
    if m is None:
        raise ValueError(
            f"Invalid pair name '{pair_name}'. Expected format N{{N}}_D{{D}} (e.g., N80_D16)."
        )
    return int(m.group("N")), int(m.group("D"))


def _resolve_pair_names(
    manifest: Dict[str, Any],
    explicit_pairs: str | None,
    dataset_root: Path,
) -> List[str]:
    if explicit_pairs is not None:
        pair_names = [x.strip() for x in explicit_pairs.split(",") if x.strip()]
        if not pair_names:
            raise ValueError("--pairs was provided but no valid pair names were found.")
        for pair_name in pair_names:
            _parse_pair_name(pair_name)
        return pair_names

    manifest_pairs = manifest.get("pairs", [])
    if isinstance(manifest_pairs, list) and manifest_pairs:
        names: List[str] = []
        for pair_info in manifest_pairs:
            if not isinstance(pair_info, dict):
                continue
            pair_name = pair_info.get("pair_name")
            if isinstance(pair_name, str):
                names.append(pair_name)
        if names:
            return sorted(
                set(names),
                key=lambda x: (_parse_pair_name(x)[1], _parse_pair_name(x)[0]),
            )

    # Fallback to folder discovery.
    pair_names = []
    for p in sorted(dataset_root.glob("N*_D*")):
        if p.is_dir():
            _parse_pair_name(p.name)
            pair_names.append(p.name)
    if not pair_names:
        raise ValueError("No synthetic pair folders found under dataset root.")
    return pair_names


def _manifest_pair_metadata(manifest: Dict[str, Any], pair_name: str) -> Dict[str, Any]:
    manifest_pairs = manifest.get("pairs", [])
    if isinstance(manifest_pairs, list):
        for pair_info in manifest_pairs:
            if isinstance(pair_info, dict) and pair_info.get("pair_name") == pair_name:
                return dict(pair_info)

    N, D = _parse_pair_name(pair_name)
    padded_dim = 2 ** int(np.ceil(np.log2(D)))
    d = int(np.log2(padded_dim))
    return {
        "pair_name": pair_name,
        "N": int(N),
        "D": int(D),
        "d": int(d),
        "padded_dim": int(padded_dim),
    }


def _load_pair_seed_split(
    dataset_root: Path,
    pair_name: str,
    seed: int,
    use_padded_features: bool,
) -> Dict[str, Any]:
    seed_dir = dataset_root / pair_name / f"seed_{int(seed):04d}"
    if not seed_dir.exists():
        raise FileNotFoundError(
            f"Missing split directory for pair={pair_name}, seed={seed}: {seed_dir}"
        )

    train_path = seed_dir / "train.npz"
    test_path = seed_dir / "test.npz"
    meta_path = seed_dir / "split_meta.json"
    params_path = seed_dir / "params.npz"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing train/test files for pair={pair_name}, seed={seed}."
        )

    with np.load(train_path) as train_data, np.load(test_path) as test_data:
        x_key = "X_padded" if use_padded_features and "X_padded" in train_data.files else "X"
        if x_key not in train_data.files or x_key not in test_data.files:
            raise KeyError(
                f"Expected key '{x_key}' in both train/test npz files for pair={pair_name}, seed={seed}."
            )
        if "y" not in train_data.files or "y" not in test_data.files:
            raise KeyError("Expected key 'y' in both train/test npz files.")

        X_train = np.asarray(train_data[x_key], dtype=float)
        y_train = np.asarray(train_data["y"], dtype=float).reshape(-1)
        X_test = np.asarray(test_data[x_key], dtype=float)
        y_test = np.asarray(test_data["y"], dtype=float).reshape(-1)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Loaded train/test feature matrices must be 2D.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train/test feature dimensions do not match.")
    if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
        raise ValueError("Train/test feature-target sizes do not match.")

    split_meta: Dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            raw_meta = json.load(f)
        if isinstance(raw_meta, dict):
            split_meta = raw_meta

    params_meta: Dict[str, Any] = {}
    if params_path.exists():
        with np.load(params_path) as params_data:
            for key in ("N", "D", "d", "padded_dim", "noise_std", "seed"):
                if key in params_data.files:
                    value = params_data[key]
                    if np.asarray(value).shape == ():
                        params_meta[key] = float(value) if key == "noise_std" else int(value)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "split_meta": split_meta,
        "params_meta": params_meta,
        "seed_dir": str(seed_dir),
    }


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
        "loss_history": [float(x) for x in batch_loss_history],
        "closed_form_norm": w_star_norm,
        "encoding_vector": [float(x) for x in np.asarray(encoding_vector, dtype=float)],
        "reconstructed_vector": [float(x) for x in np.asarray(reconstructed_vector, dtype=float)],
    }


def _aggregate_case_results(
    case_seed_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
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
            aggregate[method][metric] = {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0))}
    return aggregate


def _aggregate_overall(pair_results: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    collected: Dict[str, Dict[str, List[float]]] = {}
    for pair_block in pair_results.values():
        prior_cases = pair_block.get("prior_cases", {})
        for prior_case in PRIOR_CASES:
            prior_block = prior_cases.get(prior_case, {})
            for seed_result in prior_block.get("per_seed", []):
                methods = seed_result.get("methods", {})
                for method, method_metrics in methods.items():
                    if method not in collected:
                        collected[method] = {metric: [] for metric in METRIC_KEYS}
                    for metric in METRIC_KEYS:
                        if metric in method_metrics:
                            collected[method][metric].append(float(method_metrics[metric]))

    overall: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method, metric_values in collected.items():
        overall[method] = {}
        for metric, values in metric_values.items():
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            overall[method][metric] = {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0))}
    return overall


def _resolve_vqbr_optimizer_configs(args: argparse.Namespace, n_train_samples: int) -> List[Dict[str, Any]]:
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


# ----------------------------
# Fix A: global (V, lambda) per (pair, prior), estimated on pilot seeds
# ----------------------------
def _select_global_hyperparameters_for_pair(
    *,
    dataset_root: Path,
    pair_name: str,
    prior_case: str,
    pilot_seeds: Sequence[int],
    use_padded_features: bool,
    V_grid: np.ndarray,
    lambda_grid: np.ndarray,
    prior_delta: float,
    val_ratio_within_train: float,
    log_realtime: bool,
    log_hyperparam_sweep: bool,
) -> Dict[str, float]:
    # Accumulate validation MSEs across pilot seeds for each (V, lambda)
    # key: (iV, iL) -> list of mses
    store: Dict[Tuple[int, int], List[float]] = {}

    model = ClosedFormMAPBayesianRegression()
    for seed in pilot_seeds:
        split = _load_pair_seed_split(
            dataset_root=dataset_root,
            pair_name=pair_name,
            seed=int(seed),
            use_padded_features=bool(use_padded_features),
        )
        X_train_raw = np.asarray(split["X_train"], dtype=float)
        y_train_raw = np.asarray(split["y_train"], dtype=float).reshape(-1)

        inner = _make_inner_split(
            n_train_total=int(X_train_raw.shape[0]),
            val_ratio_within_train=float(val_ratio_within_train),
            seed=int(seed),
        )
        X_tune_train_raw = X_train_raw[inner.train_tune_indices]
        y_tune_train_raw = y_train_raw[inner.train_tune_indices]
        X_val_raw = X_train_raw[inner.val_indices]
        y_val_raw = y_train_raw[inner.val_indices]

        tune_mean, tune_std = _fit_feature_standardizer(X_tune_train_raw)
        X_tune_train = _apply_feature_standardizer(X_tune_train_raw, tune_mean, tune_std)
        X_val = _apply_feature_standardizer(X_val_raw, tune_mean, tune_std)
        y_tune_mean, y_tune_train_centered = _fit_and_center_target(y_tune_train_raw)

        for iV, V in enumerate(V_grid):
            for iL, lambda_strength in enumerate(lambda_grid):
                m0, sigma_diag = _build_prior(
                    prior_case=prior_case,
                    X_reference=X_tune_train,
                    lambda_strength=float(lambda_strength),
                    delta=float(prior_delta),
                    variance_reference=X_tune_train_raw,
                )
                model.fit(
                    X_tune_train,
                    y_tune_train_centered,
                    m0,
                    sigma_diag,
                    float(V),
                )
                y_val_pred_centered = model.predict(X_val)
                y_val_pred = y_val_pred_centered + float(y_tune_mean)
                val_mse = _mse(y_val_raw, y_val_pred)

                store.setdefault((iV, iL), []).append(float(val_mse))

                if log_hyperparam_sweep:
                    print(
                        "    "
                        f"[{pair_name}][{prior_case}][pilot_seed={seed}] sweep: "
                        f"V={float(V):.3e}, lambda={float(lambda_strength):.3e}, "
                        f"val_mse={val_mse:.6f}",
                        flush=True,
                    )

    best = {
        "validation_mse_mean": float("inf"),
        "validation_mse_std": float("nan"),
        "V": float(V_grid[0]),
        "lambda_strength": float(lambda_grid[0]),
        "num_pilot_seeds": int(len(pilot_seeds)),
        "pilot_seeds": [int(s) for s in pilot_seeds],
    }

    for (iV, iL), vals in store.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        if mean < best["validation_mse_mean"]:
            best = {
                "validation_mse_mean": mean,
                "validation_mse_std": std,
                "V": float(V_grid[iV]),
                "lambda_strength": float(lambda_grid[iL]),
                "num_pilot_seeds": int(len(pilot_seeds)),
                "pilot_seeds": [int(s) for s in pilot_seeds],
            }

    if log_realtime:
        print(
            "  "
            f"[{pair_name}][{prior_case}] GLOBAL hyperparams selected (Fix A): "
            f"V={best['V']:.3e}, lambda={best['lambda_strength']:.3e}, "
            f"pilot_val_mse_mean={best['validation_mse_mean']:.6f}, "
            f"pilot_val_mse_std={best['validation_mse_std']:.6f}, "
            f"pilot_seeds={best['pilot_seeds']}",
            flush=True,
        )
    return best


def _run_one_seed_one_prior(
    *,
    pair_name: str,
    seed: int,
    prior_case: str,
    X_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
    fixed_V: float,
    fixed_lambda_strength: float,
    args: argparse.Namespace,
    split_meta: Dict[str, Any],
    global_hyperparam_summary: Dict[str, Any],
) -> Dict[str, Any]:
    # Final train/test preprocessing with scaler fitted only on the outer train split.
    train_mean, train_std = _fit_feature_standardizer(X_train_raw)
    X_train = _apply_feature_standardizer(X_train_raw, train_mean, train_std)
    X_test = _apply_feature_standardizer(X_test_raw, train_mean, train_std)
    y_train_mean, y_train_centered = _fit_and_center_target(y_train_raw)

    m0_train, sigma_diag_train = _build_prior(
        prior_case=prior_case,
        X_reference=X_train,
        lambda_strength=float(fixed_lambda_strength),
        delta=args.prior_delta,
        variance_reference=X_train_raw,
    )
    V_selected = float(fixed_V)

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
            f"[{pair_name}][{prior_case}][seed={seed}][closed_form] "
            f"train_mse={methods['closed_form']['train_mse']:.6f}, "
            f"test_mse={methods['closed_form']['test_mse']:.6f}",
            flush=True,
        )

    vqbr_module = importlib.import_module("vqbr")
    VariationalQuantumBayesianRegression = getattr(vqbr_module, "VariationalQuantumBayesianRegression")
    optimizer_configs = _resolve_vqbr_optimizer_configs(args=args, n_train_samples=int(X_train.shape[0]))
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
                f"[{pair_name}][{prior_case}][seed={seed}][{method_key}] "
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
                pair_name=pair_name,
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

        statevector = np.asarray(fitted_state, dtype=complex).reshape(-1)
        state_norm = float(np.linalg.norm(statevector))
        if state_norm <= EPS:
            raise RuntimeError("VQBR fit returned a zero-norm state vector.")
        statevector = statevector / state_norm

        reconstructed_vector, encoding_vector = _reconstruct_vqbr_vector_from_state(statevector, w_star=w_star)
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
        methods[method_key] = vqbr_metrics

        if args.log_realtime:
            print(
                "    "
                f"[{pair_name}][{prior_case}][seed={seed}][{method_key}] "
                f"cos={vqbr_metrics['cosine_similarity']:.6f}, "
                f"rel_l2={vqbr_metrics['relative_l2_distance']:.6e}, "
                f"train_mse={vqbr_metrics['train_mse']:.6f}, "
                f"test_mse={vqbr_metrics['test_mse']:.6f}, "
                f"batch_L_hat={vqbr_metrics['batch_L_hat']:.6e}",
                flush=True,
            )

    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])
    train_ratio = split_meta.get("train_ratio")
    test_ratio = split_meta.get("test_ratio")
    if train_ratio is None or test_ratio is None:
        total = float(n_train + n_test)
        train_ratio = n_train / total
        test_ratio = n_test / total

    return {
        "seed": int(seed),
        "split": {
            "n_train_total": n_train,
            "n_test": n_test,
            "train_ratio": float(train_ratio),
            "test_ratio": float(test_ratio),
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
            # Fix A: report the frozen hyperparams and how they were chosen.
            "selection_mode": "global_frozen_per_pair",
            "V": float(V_selected),
            "lambda_strength": float(fixed_lambda_strength),
            "prior_delta": float(args.prior_delta),
            "pilot_selection": dict(global_hyperparam_summary),
        },
        "methods": methods,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_root = Path(args.dataset_root).resolve()
    output_path = Path(args.output_path).resolve()
    if args.log_every_iter <= 0:
        raise ValueError("--log-every-iter must be a positive integer.")
    if args.vqbr_batch_size <= 0:
        raise ValueError("--vqbr-batch-size must be a positive integer.")

    manifest = _load_manifest(dataset_root)
    pair_names = _resolve_pair_names(manifest, args.pairs, dataset_root)
    manifest_seeds = manifest.get("seeds", [])
    if not isinstance(manifest_seeds, list):
        manifest_seeds = []

    seeds = _resolve_seeds(
        explicit_seeds=args.seeds,
        num_seeds=args.num_seeds,
        seed_offset=args.seed_offset,
        manifest_seeds=[int(x) for x in manifest_seeds],
    )

    V_grid = _parse_float_grid(args.V_grid)
    lambda_grid = _parse_float_grid(args.lambda_grid)
    vqbr_loss = _normalize_vqbr_loss(args.vqbr_loss)
    selected_optimizer_settings = _select_vqbr_optimizer_settings(args.vqbr_optimizer)
    selected_method_keys = [str(setting["method_key"]) for setting in selected_optimizer_settings]
    reported_method_keys = ["closed_form"] + selected_method_keys

    pilot_k = int(args.hyperparam_pilot_seeds)
    if pilot_k <= 0:
        raise ValueError("--hyperparam-pilot-seeds must be positive.")
    pilot_seeds = list(seeds[: min(pilot_k, len(seeds))])

    result: Dict[str, Any] = {
        "dataset": {
            "root": str(dataset_root),
            "source": "synthetic",
            "pair_names": pair_names,
            "num_pairs": int(len(pair_names)),
        },
        "config": {
            "num_seeds": int(len(seeds)),
            "seeds": [int(s) for s in seeds],
            "prior_cases": list(PRIOR_CASES),
            "val_ratio_within_train": float(args.val_ratio_within_train),
            "V_grid": [float(x) for x in V_grid],
            "lambda_grid": [float(x) for x in lambda_grid],
            "prior_delta": float(args.prior_delta),
            "methods": reported_method_keys,
            "use_padded_features": bool(args.use_padded_features),
            "vqbr": {
                "shots": int(args.vqbr_shots),
                "batch_size": int(args.vqbr_batch_size),
                "reps": int(args.vqbr_reps),
                "maxiter": int(args.vqbr_maxiter),
                "shuffle_batches": bool(args.vqbr_shuffle_batches),
                "learning_rate": float(args.vqbr_learning_rate),
                "loss": str(vqbr_loss),
                "use_shot_noise": bool(args.vqbr_use_shot_noise),
            },
            "hyperparam_selection": {
                "mode": "FixA_global_frozen_per_pair",
                "pilot_seeds": [int(s) for s in pilot_seeds],
                "pilot_seeds_count": int(len(pilot_seeds)),
            },
            "logging": {
                "realtime": bool(args.log_realtime),
                "log_every_iter": int(args.log_every_iter),
                "log_hyperparam_sweep": bool(args.log_hyperparam_sweep),
            },
        },
        "pair_results": {},
        "overall_aggregate": {},
    }

    print("=== Synthetic Experiment (Fix A: global frozen hyperparams per pair) ===")
    print(f"Dataset root: {dataset_root}")
    print(f"Pairs ({len(pair_names)}): {pair_names}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Pilot seeds for global (V,lambda): {pilot_seeds}")
    print(f"Use padded features: {bool(args.use_padded_features)}")
    print(f"Inner validation ratio (within train): {args.val_ratio_within_train:.2f}")
    print("VQBR config:")
    print(
        "  "
        f"shots={int(args.vqbr_shots)}, "
        f"batch_size={int(args.vqbr_batch_size)}, "
        f"reps={int(args.vqbr_reps)}, "
        f"maxiter={int(args.vqbr_maxiter)}"
    )
    print("")

    for pair_name in pair_names:
        pair_meta = _manifest_pair_metadata(manifest, pair_name)
        print(
            f"[Pair: {pair_name}] "
            f"N={pair_meta.get('N')}, D={pair_meta.get('D')}, "
            f"d={pair_meta.get('d')}, padded_dim={pair_meta.get('padded_dim')}"
        )

        pair_block: Dict[str, Any] = {"pair": pair_meta, "prior_cases": {}}

        for prior_case in PRIOR_CASES:
            print(f"  [Prior case: {prior_case}]")

            # Fix A: select once, reuse for all seeds for this (pair, prior)
            global_hp = _select_global_hyperparameters_for_pair(
                dataset_root=dataset_root,
                pair_name=pair_name,
                prior_case=prior_case,
                pilot_seeds=pilot_seeds,
                use_padded_features=bool(args.use_padded_features),
                V_grid=V_grid,
                lambda_grid=lambda_grid,
                prior_delta=float(args.prior_delta),
                val_ratio_within_train=float(args.val_ratio_within_train),
                log_realtime=bool(args.log_realtime),
                log_hyperparam_sweep=bool(args.log_hyperparam_sweep),
            )
            fixed_V = float(global_hp["V"])
            fixed_lambda = float(global_hp["lambda_strength"])

            case_seed_results: List[Dict[str, Any]] = []
            for i, seed in enumerate(seeds, start=1):
                print(f"    Seed {seed} ({i}/{len(seeds)})", flush=True)
                split = _load_pair_seed_split(
                    dataset_root=dataset_root,
                    pair_name=pair_name,
                    seed=seed,
                    use_padded_features=bool(args.use_padded_features),
                )

                seed_result = _run_one_seed_one_prior(
                    pair_name=pair_name,
                    seed=seed,
                    prior_case=prior_case,
                    X_train_raw=np.asarray(split["X_train"], dtype=float),
                    y_train_raw=np.asarray(split["y_train"], dtype=float).reshape(-1),
                    X_test_raw=np.asarray(split["X_test"], dtype=float),
                    y_test_raw=np.asarray(split["y_test"], dtype=float).reshape(-1),
                    fixed_V=fixed_V,
                    fixed_lambda_strength=fixed_lambda,
                    args=args,
                    split_meta=dict(split.get("split_meta", {})),
                    global_hyperparam_summary=global_hp,
                )
                seed_result["data_files"] = {"seed_dir": str(split["seed_dir"])}
                case_seed_results.append(seed_result)

            pair_block["prior_cases"][prior_case] = {
                "per_seed": case_seed_results,
                "aggregate": _aggregate_case_results(case_seed_results),
                "global_hyperparameters": dict(global_hp),
            }
            print("")

        result["pair_results"][pair_name] = pair_block

    result["overall_aggregate"] = _aggregate_overall(result["pair_results"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=args.json_indent)

    print(f"Saved results to: {output_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run closed-form and VQBR on generated synthetic datasets over (N, D) pairs "
            "with per-seed train/test splits. Fix A: globally freeze (V,lambda) per pair."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(ROOT_DIR / "data" / "synthetic" / "generated"),
        help="Root directory produced by data/synthetic/data_synthetic.py",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Optional comma-separated subset of pair names (e.g., N16_D8,N80_D16).",
    )
    parser.add_argument("--use-padded-features", action="store_true")
    parser.add_argument("--use-raw-features", dest="use_padded_features", action="store_false")
    parser.set_defaults(use_padded_features=True)

    parser.add_argument("--val-ratio-within-train", type=float, default=0.1)
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--seeds", type=str, default=None)

    parser.add_argument("--V-grid", type=str, default="1e-4,1e-3,1e-2,1e-1,1,10")
    parser.add_argument("--lambda-grid", type=str, default="1e-4,1e-3,1e-2,1e-1,1,10,100")
    parser.add_argument("--prior-delta", type=float, default=1e-8)

    # Fix A: how many seeds to use for pilot hyperparam selection
    parser.add_argument(
        "--hyperparam-pilot-seeds",
        type=int,
        default=3,
        help="Number of earliest seeds used to pick a global (V,lambda) per pair (Fix A).",
    )

    parser.add_argument("--vqbr-shots", type=int, default=1024)
    parser.add_argument("--vqbr-batch-size", type=int, default=100)
    parser.add_argument("--vqbr-reps", type=int, default=4)
    parser.add_argument(
        "--vqbr-optimizer",
        "--vqbr-optimizers",
        dest="vqbr_optimizer",
        type=str,
        default="COBYLA",
        help="Comma-separated VQBR optimizers to run. Supported: COBYLA.",
    )
    parser.add_argument("--vqbr-maxiter", type=int, default=400)
    parser.add_argument("--vqbr-shuffle-batches", action="store_true")
    parser.add_argument("--no-vqbr-shuffle-batches", dest="vqbr_shuffle_batches", action="store_false")
    parser.set_defaults(vqbr_shuffle_batches=True)
    parser.add_argument("--vqbr-learning-rate", type=float, default=0.05)
    parser.add_argument("--vqbr-loss", type=str, default="neg_ratio")
    parser.add_argument("--vqbr-use-shot-noise", action="store_true")
    parser.add_argument("--no-vqbr-shot-noise", dest="vqbr_use_shot_noise", action="store_false")
    parser.set_defaults(vqbr_use_shot_noise=True)
    parser.add_argument("--vqbr-verbose", action="store_true")

    parser.add_argument("--log-realtime", action="store_true")
    parser.add_argument("--no-log-realtime", dest="log_realtime", action="store_false")
    parser.set_defaults(log_realtime=True)
    parser.add_argument("--log-every-iter", type=int, default=5)
    parser.add_argument("--log-hyperparam-sweep", action="store_true")

    parser.add_argument(
        "--output-path",
        type=str,
        default=str(ROOT_DIR / "results" / "synthetic" / "synthetic_experiment_results.json"),
    )
    parser.add_argument(
        "--console-log-path",
        type=str,
        default=str(ROOT_DIR / "results" / "synthetic" / "synthetic_experiment_run.log"),
        help="Path to mirror all console output. Set empty string to disable file logging.",
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
