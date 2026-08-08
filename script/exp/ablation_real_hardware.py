from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize

ROOT_DIR = Path(__file__).resolve().parents[2]
SYNTHETIC_EXPERIMENT_PATH = ROOT_DIR / "script" / "exp" / "synthetic_experiemnt.py"
REAL_HARDWARE_PLOT_PATH = ROOT_DIR / "script" / "plot" / "ablation_real_hardware_plot.py"
DEFAULT_RESULTS_ROOT = ROOT_DIR / "results" / "synthetic"
DEFAULT_IBM_CONFIG_PATH = ROOT_DIR / "config" / "ibm_config.json"


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


def _load_real_hardware_plot_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_real_hardware_plot_helpers",
        REAL_HARDWARE_PLOT_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load plotting helper from: {REAL_HARDWARE_PLOT_PATH}")
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

_estimate_prior_from_prior_split = _SYN._estimate_prior_from_prior_split
_evaluate_regression_metrics = _SYN._evaluate_regression_metrics
_extract_vqbr_history = _SYN._extract_vqbr_history
_generate_synthetic_data = _SYN._generate_synthetic_data
_normalize_state = _SYN._normalize_state
_normalized_closed_form_state = _SYN._normalized_closed_form_state
_reconstruct_weights_from_formula = _SYN._reconstruct_weights_from_formula
_resolve_seeds = _SYN._resolve_seeds
_safe_cosine = _SYN._safe_cosine
_select_snapshot_for_solution = _SYN._select_snapshot_for_solution
_slugify_token = _SYN._slugify_token
_split_dataset_indices = _SYN._split_dataset_indices
_state_to_feature_direction = _SYN._state_to_feature_direction

try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import Operator, SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
except Exception as exc:  # pragma: no cover - optional dependency guard
    QuantumCircuit = None  # type: ignore[assignment]
    StatevectorEstimator = None  # type: ignore[assignment]
    Operator = None  # type: ignore[assignment]
    SparsePauliOp = None  # type: ignore[assignment]
    generate_preset_pass_manager = None  # type: ignore[assignment]
    _QISKIT_IMPORT_ERROR = exc
else:  # pragma: no cover - import bookkeeping
    _QISKIT_IMPORT_ERROR = None

try:
    from qiskit_aer import AerSimulator as _AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    from qiskit_aer.primitives import EstimatorV2 as _AerEstimatorV2
except Exception as exc:  # pragma: no cover - optional dependency guard
    _AerSimulator = None  # type: ignore[assignment]
    NoiseModel = None  # type: ignore[assignment]
    ReadoutError = None  # type: ignore[assignment]
    depolarizing_error = None  # type: ignore[assignment]
    _AerEstimatorV2 = None  # type: ignore[assignment]
    _AER_IMPORT_ERROR = exc
else:  # pragma: no cover - import bookkeeping
    _AER_IMPORT_ERROR = None


@dataclass
class _Snapshot:
    L_tilde: float
    a_hat: float
    c_hat: float
    d_hat: float
    e_hat: float
    h_hat: float
    batch_indices: np.ndarray
    batch_L_tilde: float = np.nan
    full_L_tilde: float = np.nan
    epoch: int = 0
    batch_position: int = 0


@dataclass
class _PreparedVQBR:
    model: VariationalQuantumBayesianRegression
    overlap_circuits: List[Any]
    overlap_observable: Any
    uc_circuit: Any | None
    uc_observable: Any | None
    d_circuit: Any
    d_observable: Any


class _PrimitiveObjectiveEvaluator:
    def __init__(
        self,
        *,
        prepared: _PreparedVQBR,
        estimator: Any,
        method_key: str,
        method_label: str,
        backend_name: str,
        transpile_backend: Any | None,
        transpile_optimization_level: int,
    ) -> None:
        self.prepared = prepared
        self.model = prepared.model
        self.estimator = estimator
        self.method_key = str(method_key)
        self.method_label = str(method_label)
        self.backend_name = str(backend_name)

        self._overlap_circuits: List[Any] = []
        self._overlap_observables: List[Any] = []
        self._uc_circuit: Any | None = None
        self._uc_observable: Any | None = None
        self._d_circuit: Any | None = None
        self._d_observable: Any | None = None

        if transpile_backend is None:
            self._overlap_circuits = list(prepared.overlap_circuits)
            self._overlap_observables = [
                prepared.overlap_observable for _ in prepared.overlap_circuits
            ]
            self._uc_circuit = prepared.uc_circuit
            self._uc_observable = prepared.uc_observable
            self._d_circuit = prepared.d_circuit
            self._d_observable = prepared.d_observable
        else:
            self._prepare_isa_artifacts(
                backend=transpile_backend,
                optimization_level=int(transpile_optimization_level),
            )

    def _prepare_isa_artifacts(self, *, backend: Any, optimization_level: int) -> None:
        if generate_preset_pass_manager is None:
            raise ImportError(
                "generate_preset_pass_manager is unavailable. "
                "Install a compatible Qiskit version to transpile for IBM hardware."
            )
        pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=int(optimization_level),
        )

        self._overlap_circuits = []
        self._overlap_observables = []
        for circuit in self.prepared.overlap_circuits:
            isa_circuit = pass_manager.run(circuit)
            self._overlap_circuits.append(isa_circuit)
            self._overlap_observables.append(
                _apply_layout_if_possible(self.prepared.overlap_observable, isa_circuit)
            )

        if self.prepared.uc_circuit is not None and self.prepared.uc_observable is not None:
            self._uc_circuit = pass_manager.run(self.prepared.uc_circuit)
            self._uc_observable = _apply_layout_if_possible(
                self.prepared.uc_observable,
                self._uc_circuit,
            )

        self._d_circuit = pass_manager.run(self.prepared.d_circuit)
        self._d_observable = _apply_layout_if_possible(
            self.prepared.d_observable,
            self._d_circuit,
        )

    def _run_estimator_with(self, estimator: Any, pubs: Sequence[Any]) -> np.ndarray:
        job = estimator.run(list(pubs))
        result = job.result()
        evs: List[float] = []
        for pub_result in result:
            data = getattr(pub_result, "data", None)
            raw = getattr(data, "evs", None)
            if raw is None:
                evs.append(float("nan"))
                continue
            arr = np.asarray(raw, dtype=float).reshape(-1)
            evs.append(float(arr[0]) if arr.size > 0 else float("nan"))
        return np.asarray(evs, dtype=float)

    def _run_estimator(self, pubs: Sequence[Any]) -> np.ndarray:
        return self._run_estimator_with(self.estimator, pubs)

    def evaluate_snapshot(
        self,
        theta_vec: np.ndarray,
        batch_indices: np.ndarray,
        *,
        scale_batch_to_full: bool,
    ) -> _Snapshot:
        snapshot, _ = self.evaluate_snapshot_with_overlaps(
            theta_vec,
            batch_indices,
            scale_batch_to_full=scale_batch_to_full,
        )
        return snapshot

    def evaluate_snapshot_with_overlaps(
        self,
        theta_vec: np.ndarray,
        batch_indices: np.ndarray,
        *,
        scale_batch_to_full: bool,
    ) -> tuple[_Snapshot, np.ndarray]:
        batch = np.asarray(batch_indices, dtype=int).reshape(-1)
        if batch.size == 0:
            raise ValueError("batch_indices must not be empty.")

        theta_list = [np.asarray(theta_vec, dtype=float).reshape(-1).tolist()]
        pubs: List[Any] = []
        for sample_idx in batch.tolist():
            pubs.append(
                (
                    self._overlap_circuits[int(sample_idx)],
                    self._overlap_observables[int(sample_idx)],
                    theta_list,
                )
            )

        has_uc = self._uc_circuit is not None and self._uc_observable is not None
        if has_uc:
            pubs.append((self._uc_circuit, self._uc_observable, theta_list))
        pubs.append((self._d_circuit, self._d_observable, theta_list))

        evs = self._run_estimator(pubs)
        s = np.asarray(evs[: batch.size], dtype=float)
        cursor = int(batch.size)
        h_hat = 0.0
        if has_uc:
            h_hat = float(evs[cursor])
            cursor += 1
        d_hat = float(evs[cursor])

        r = np.asarray(self.model._row_norms, dtype=float)[batch]
        yb = np.asarray(self.model._y, dtype=float)[batch]
        scale = float(self.model._n_samples / batch.size) if scale_batch_to_full else 1.0
        a_hat = float(scale * np.sum((r * s) ** 2))
        c_hat = float(scale * np.sum(yb * (r * s)))
        e_hat = float(self.model._c_norm * h_hat) if self.model._Uc_gate is not None else 0.0
        denom = float(a_hat + d_hat + self.model.eps)
        numerator = float(c_hat + e_hat)
        L_tilde = float(
            self.model._compute_objective_value(
                numerator=numerator,
                denom=denom,
            )
        )
        snapshot = _Snapshot(
            L_tilde=float(L_tilde),
            a_hat=float(a_hat),
            c_hat=float(c_hat),
            d_hat=float(d_hat),
            e_hat=float(e_hat),
            h_hat=float(h_hat),
            batch_indices=batch.copy(),
            batch_L_tilde=float(L_tilde),
            full_L_tilde=float(L_tilde),
            epoch=0,
            batch_position=0,
        )
        return snapshot, s


class _ZNEObjectiveEvaluator(_PrimitiveObjectiveEvaluator):
    def __init__(
        self,
        *,
        prepared: _PreparedVQBR,
        estimators: Sequence[Any],
        baseline_estimator: Any | None,
        noise_factors: Sequence[float],
        zne_extrapolator: str,
        method_key: str,
        method_label: str,
        backend_name: str,
        transpile_backend: Any | None,
        transpile_optimization_level: int,
    ) -> None:
        if not estimators:
            raise ValueError("At least one estimator is required for ZNE evaluation.")
        resolved_baseline_estimator = baseline_estimator if baseline_estimator is not None else estimators[0]
        super().__init__(
            prepared=prepared,
            estimator=resolved_baseline_estimator,
            method_key=method_key,
            method_label=method_label,
            backend_name=backend_name,
            transpile_backend=transpile_backend,
            transpile_optimization_level=transpile_optimization_level,
        )
        self._baseline_estimator = resolved_baseline_estimator
        self._zne_estimators = list(estimators)
        self._zne_noise_factors = np.asarray(noise_factors, dtype=float).reshape(-1)
        self._zne_extrapolator = str(zne_extrapolator).strip().lower()
        self._manual_extrapolation = len(self._zne_estimators) > 1

    def _run_zne_hadamard_test_estimator(self, pubs: Sequence[Any]) -> np.ndarray:
        if not pubs:
            return np.empty((0,), dtype=float)
        if not self._manual_extrapolation:
            return self._run_estimator_with(self._zne_estimators[0], pubs)
        evs_by_factor = []
        for estimator in self._zne_estimators:
            evs_by_factor.append(self._run_estimator_with(estimator, pubs))
        values = np.asarray(evs_by_factor, dtype=float)
        if values.ndim != 2:
            raise RuntimeError("Unexpected ZNE estimator output shape.")
        out = np.full((values.shape[1],), np.nan, dtype=float)
        for pub_idx in range(values.shape[1]):
            out[pub_idx] = _extrapolate_to_zero(
                noise_factors=self._zne_noise_factors,
                values=values[:, pub_idx],
                extrapolator=self._zne_extrapolator,
            )
        return out

    def evaluate_snapshot_with_overlaps(
        self,
        theta_vec: np.ndarray,
        batch_indices: np.ndarray,
        *,
        scale_batch_to_full: bool,
    ) -> tuple[_Snapshot, np.ndarray]:
        batch = np.asarray(batch_indices, dtype=int).reshape(-1)
        if batch.size == 0:
            raise ValueError("batch_indices must not be empty.")

        theta_list = [np.asarray(theta_vec, dtype=float).reshape(-1).tolist()]
        hadamard_test_pubs: List[Any] = []
        for sample_idx in batch.tolist():
            hadamard_test_pubs.append(
                (
                    self._overlap_circuits[int(sample_idx)],
                    self._overlap_observables[int(sample_idx)],
                    theta_list,
                )
            )

        has_uc = self._uc_circuit is not None and self._uc_observable is not None
        if has_uc:
            hadamard_test_pubs.append((self._uc_circuit, self._uc_observable, theta_list))

        hadamard_test_evs = self._run_zne_hadamard_test_estimator(hadamard_test_pubs)
        d_hat_arr = self._run_estimator_with(
            self._baseline_estimator,
            [(self._d_circuit, self._d_observable, theta_list)],
        )
        s = np.asarray(hadamard_test_evs[: batch.size], dtype=float)
        cursor = int(batch.size)
        h_hat = 0.0
        if has_uc:
            h_hat = float(hadamard_test_evs[cursor])
        d_hat = float(d_hat_arr[0]) if d_hat_arr.size > 0 else float("nan")

        r = np.asarray(self.model._row_norms, dtype=float)[batch]
        yb = np.asarray(self.model._y, dtype=float)[batch]
        scale = float(self.model._n_samples / batch.size) if scale_batch_to_full else 1.0
        a_hat = float(scale * np.sum((r * s) ** 2))
        c_hat = float(scale * np.sum(yb * (r * s)))
        e_hat = float(self.model._c_norm * h_hat) if self.model._Uc_gate is not None else 0.0
        denom = float(a_hat + d_hat + self.model.eps)
        numerator = float(c_hat + e_hat)
        L_tilde = float(
            self.model._compute_objective_value(
                numerator=numerator,
                denom=denom,
            )
        )
        snapshot = _Snapshot(
            L_tilde=float(L_tilde),
            a_hat=float(a_hat),
            c_hat=float(c_hat),
            d_hat=float(d_hat),
            e_hat=float(e_hat),
            h_hat=float(h_hat),
            batch_indices=batch.copy(),
            batch_L_tilde=float(L_tilde),
            full_L_tilde=float(L_tilde),
            epoch=0,
            batch_position=0,
        )
        return snapshot, s


def _require_qiskit_components() -> None:
    if (
        QuantumCircuit is None
        or StatevectorEstimator is None
        or SparsePauliOp is None
        or Operator is None
    ):
        detail = "" if _QISKIT_IMPORT_ERROR is None else f" Original error: {_QISKIT_IMPORT_ERROR}"
        raise ImportError(
            "Qiskit primitives and quantum_info are required for the real-hardware ablation."
            + detail
        )


def _require_aer_components() -> None:
    if (
        _AerSimulator is None
        or _AerEstimatorV2 is None
        or NoiseModel is None
        or ReadoutError is None
        or depolarizing_error is None
    ):
        detail = "" if _AER_IMPORT_ERROR is None else f" Original error: {_AER_IMPORT_ERROR}"
        raise ImportError(
            "qiskit-aer primitives and noise components are required for "
            "the aer_noise hardware-runner."
            + detail
        )


def _parse_float_list(raw: str | None, *, default: Sequence[float]) -> List[float]:
    if raw is None:
        return [float(x) for x in default]
    tokens = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not tokens:
        raise ValueError("Expected a non-empty comma-separated float list.")
    return [float(token) for token in tokens]


def _extrapolate_to_zero(
    *,
    noise_factors: np.ndarray,
    values: np.ndarray,
    extrapolator: str,
) -> float:
    xs = np.asarray(noise_factors, dtype=float).reshape(-1)
    ys = np.asarray(values, dtype=float).reshape(-1)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size == 0:
        return float("nan")
    if xs.size == 1:
        return float(ys[0])

    mode = str(extrapolator).strip().lower()
    degree = 1
    if mode in {"linear", "polynomial_degree_1"}:
        degree = 1
    elif mode in {"quadratic", "polynomial_degree_2"}:
        degree = 2
    elif mode in {"cubic", "polynomial_degree_3"}:
        degree = 3
    elif mode in {"fallback", "none"}:
        min_idx = int(np.argmin(xs))
        return float(ys[min_idx])
    else:
        raise ValueError(
            f"Unsupported Aer ZNE extrapolator '{extrapolator}'. "
            "Supported values are linear, quadratic, cubic, fallback, "
            "polynomial_degree_1, polynomial_degree_2, polynomial_degree_3."
        )

    degree = min(degree, max(int(xs.size) - 1, 1))
    coeffs = np.polyfit(xs, ys, deg=degree)
    return float(np.polyval(coeffs, 0.0))


def _sanitize_argv(argv: Sequence[str]) -> List[str]:
    sanitized: List[str] = []
    skip_next = False
    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == "--ibm-token":
            sanitized.extend(["--ibm-token", "<redacted>"])
            skip_next = True
            continue
        if token.startswith("--ibm-token="):
            sanitized.append("--ibm-token=<redacted>")
            continue
        sanitized.append(str(token))
    return sanitized


def _sanitize_args_for_results(args: argparse.Namespace) -> Dict[str, Any]:
    payload = dict(vars(args))
    if "ibm_token" in payload:
        payload["ibm_token"] = "<redacted>" if str(payload["ibm_token"]).strip() else ""
    return payload


def _backend_name(backend: Any, fallback: str = "") -> str:
    name_attr = getattr(backend, "name", None)
    if callable(name_attr):
        try:
            return str(name_attr())
        except Exception:
            pass
    if name_attr is not None:
        return str(name_attr)
    return str(fallback)


def _config_lookup(config: Dict[str, Any], *keys: str) -> Any:
    lower_map = {str(key).lower(): value for key, value in config.items()}
    for key in keys:
        if key in config:
            return config[key]
        value = lower_map.get(str(key).lower())
        if value is not None:
            return value
    return None


def _load_ibm_runtime_config(config_path: str) -> Dict[str, Any]:
    if not str(config_path).strip():
        raise ValueError(
            "IBM hardware was requested but --ibm-config was not provided. "
            "Pass the JSON config path or use --skip-ibm-hardware."
        )
    path = Path(str(config_path).strip()).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"IBM config JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("IBM config JSON must contain a top-level object.")
    payload["_config_path"] = str(path)
    return payload


def _sanitize_ibm_config_for_results(config: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(config)
    for key in list(sanitized.keys()):
        lower = str(key).lower()
        if any(token in lower for token in ("token", "api_key", "apikey", "password", "secret")):
            value = sanitized.pop(key, None)
            sanitized[f"{key}_present"] = bool(str(value).strip()) if value is not None else False
    return sanitized


def _resolve_ibm_runtime_backend(
    *,
    config_path: str,
    backend_override: str,
    channel_override: str,
    instance_override: str,
    url_override: str,
    token_override: str,
) -> tuple[Any, Dict[str, Any]]:
    config = _load_ibm_runtime_config(config_path)
    resolved_config = _sanitize_ibm_config_for_results(config)

    backend_name = (
        str(backend_override).strip()
        or str(_config_lookup(config, "backend_name", "backend", "ibm_backend") or "").strip()
        or "ibm_kingston"
    )
    channel = (
        str(channel_override).strip()
        or str(_config_lookup(config, "channel") or "").strip()
        or "ibm_cloud"
    )
    instance = (
        str(instance_override).strip()
        or str(_config_lookup(config, "instance", "crn") or "").strip()
    )
    token = (
        str(token_override).strip()
        or str(_config_lookup(config, "token", "api_key") or "").strip()
    )
    url = (
        str(url_override).strip()
        or str(_config_lookup(config, "url") or "").strip()
    )

    if not token:
        raise ValueError(
            "IBM runtime token/API key is missing. "
            "Provide it in the config JSON or via --ibm-token."
        )

    try:
        import qiskit_ibm_runtime as qir
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Failed to import qiskit_ibm_runtime. "
            "Install a Qiskit/IBM Runtime version pair that is mutually compatible "
            "before launching IBM hardware runs. "
            f"Original error: {exc}"
        ) from exc

    service_kwargs: Dict[str, Any] = {
        "channel": str(channel),
        "token": str(token),
    }
    if instance:
        service_kwargs["instance"] = str(instance)
    if url:
        service_kwargs["url"] = str(url)

    service = qir.QiskitRuntimeService(**service_kwargs)
    backend = service.backend(str(backend_name))

    resolved = {
        "config_path": str(config["_config_path"]),
        "backend_name": str(_backend_name(backend, fallback=backend_name)),
        "channel": str(channel),
        "instance": str(instance),
        "url": str(url),
        "runtime_import_available": True,
        "service_kwargs": {
            "channel": str(channel),
            "instance": str(instance),
            "url": str(url),
            "token_present": True,
        },
        "raw_config_sanitized": resolved_config,
    }
    return backend, resolved


def _build_ibm_estimator(
    *,
    backend: Any,
    shots: int,
    seed: int,
    use_zne: bool,
    zne_noise_factors: Sequence[float],
    zne_extrapolator: str,
    zne_measure_mitigation: bool,
) -> tuple[Any, Dict[str, Any]]:
    try:
        import qiskit_ibm_runtime as qir
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Failed to import qiskit_ibm_runtime while creating the IBM estimator. "
            "Install a compatible qiskit-ibm-runtime build. "
            f"Original error: {exc}"
        ) from exc

    estimator_cls = getattr(qir, "EstimatorV2", None)
    if estimator_cls is None:
        estimator_cls = getattr(qir, "Estimator", None)
    if estimator_cls is None:
        raise RuntimeError("Unable to locate EstimatorV2/Estimator in qiskit_ibm_runtime.")

    options: Dict[str, Any] = {
        "default_shots": int(shots),
        "seed_estimator": int(seed),
        "resilience_level": 0,
        "resilience": {
            "measure_mitigation": False,
            "zne_mitigation": False,
        },
    }
    mitigation_label = "w/o ZNE"
    if bool(use_zne):
        options["resilience"] = {
            "measure_mitigation": bool(zne_measure_mitigation),
            "zne_mitigation": True,
            "zne": {
                "amplifier": "gate_folding",
                "noise_factors": [float(x) for x in zne_noise_factors],
                "extrapolator": str(zne_extrapolator),
            },
        }
        mitigation_label = "w/ ZNE"

    estimator = estimator_cls(mode=backend, options=options)
    return estimator, {
        "shots": int(shots),
        "use_zne": bool(use_zne),
        "mitigation_label": mitigation_label,
        "options": options,
    }


def _clip_probability(value: float, *, max_value: float = 0.49) -> float:
    return float(min(max(float(value), 0.0), float(max_value)))


def _clip_depolarizing_probability(value: float, num_qubits: int) -> float:
    max_prob = (4**int(num_qubits) - 1) / (4**int(num_qubits))
    return float(min(max(float(value), 0.0), max_prob - 1e-12))


def _build_aer_noise_model(
    *,
    single_qubit_error: float,
    two_qubit_error: float,
    readout_p01: float,
    readout_p10: float,
) -> Any:
    _require_aer_components()

    noise_model = NoiseModel()

    p1 = _clip_depolarizing_probability(single_qubit_error, num_qubits=1)
    p2 = _clip_depolarizing_probability(two_qubit_error, num_qubits=2)
    if p1 > 0.0:
        one_q_error = depolarizing_error(p1, 1)
        for gate_name in ("sx", "x", "rz"):
            noise_model.add_all_qubit_quantum_error(one_q_error, gate_name)
    if p2 > 0.0:
        two_q_error_obj = depolarizing_error(p2, 2)
        noise_model.add_all_qubit_quantum_error(two_q_error_obj, "cx")

    p01 = _clip_probability(readout_p01)
    p10 = _clip_probability(readout_p10)
    if p01 > 0.0 or p10 > 0.0:
        noise_model.add_all_qubit_readout_error(
            ReadoutError(
                [
                    [1.0 - p01, p01],
                    [p10, 1.0 - p10],
                ]
            )
        )

    return noise_model


def _build_aer_estimator(
    *,
    seed: int,
    noise_model: Any,
    method: str,
) -> tuple[Any, Any]:
    _require_aer_components()
    backend = _AerSimulator(
        method=str(method),
        noise_model=noise_model,
    )
    estimator = _AerEstimatorV2(
        options={
            "backend_options": {
                "method": str(method),
                "noise_model": noise_model,
            },
            "run_options": {
                "seed_simulator": int(seed),
            },
        }
    )
    return estimator, backend


def _build_param_overlap_circuit(
    model: VariationalQuantumBayesianRegression,
    Uv_gate: Any,
) -> Any:
    qc = QuantumCircuit(1 + model._num_qubits)
    anc = 0
    work = list(range(1, 1 + model._num_qubits))
    qc.h(anc)
    qc.x(anc)
    qc.append(Uv_gate.control(1), [anc] + work)
    qc.x(anc)
    controlled_ansatz = model._ansatz.to_gate(label="Utheta").control(1)
    qc.append(controlled_ansatz, [anc] + work)
    qc.h(anc)
    return qc


def _apply_layout_if_possible(observable: Any, circuit: Any) -> Any:
    layout = getattr(circuit, "layout", None)
    if layout is None:
        return observable
    if hasattr(observable, "apply_layout"):
        try:
            return observable.apply_layout(layout)
        except Exception:
            return observable
    return observable


def _build_diagonal_observable(diagonal: np.ndarray) -> Any:
    matrix = np.diag(np.asarray(diagonal, dtype=complex).reshape(-1))
    return SparsePauliOp.from_operator(Operator(matrix)).simplify(atol=1e-12)


def _build_aer_noise_runner(
    *,
    seed: int,
    use_zne: bool,
    zne_noise_factors: Sequence[float],
    zne_extrapolator: str,
    aer_noise_method: str,
    aer_single_qubit_error: float,
    aer_two_qubit_error: float,
    aer_readout_p01: float,
    aer_readout_p10: float,
) -> tuple[List[Any], List[Any], Dict[str, Any]]:
    estimators: List[Any] = []
    backends: List[Any] = []
    factors = [1.0] if not bool(use_zne) else [float(x) for x in zne_noise_factors]
    for factor in factors:
        noise_model = _build_aer_noise_model(
            single_qubit_error=float(aer_single_qubit_error) * float(factor),
            two_qubit_error=float(aer_two_qubit_error) * float(factor),
            readout_p01=float(aer_readout_p01),
            readout_p10=float(aer_readout_p10),
        )
        estimator, backend = _build_aer_estimator(
            seed=int(seed),
            noise_model=noise_model,
            method=str(aer_noise_method),
        )
        estimators.append(estimator)
        backends.append(backend)
    return estimators, backends, {
        "runner": "aer_noise",
        "use_zne": bool(use_zne),
        "noise_factors": factors,
        "zne_extrapolator": str(zne_extrapolator),
        "aer_noise_method": str(aer_noise_method),
        "single_qubit_error": float(aer_single_qubit_error),
        "two_qubit_error": float(aer_two_qubit_error),
        "readout_p01": float(aer_readout_p01),
        "readout_p10": float(aer_readout_p10),
    }


def _prepare_vqbr_model(
    *,
    X: np.ndarray,
    y: np.ndarray,
    m0: np.ndarray,
    Sigma0: np.ndarray,
    V: float,
    reps: int,
    entanglement: str,
    seed: int,
) -> _PreparedVQBR:
    _require_qiskit_components()

    model = VariationalQuantumBayesianRegression(
        shots=1,
        batch_size=int(np.asarray(X, dtype=float).shape[0]),
        reps=int(reps),
        optimizer=ENFORCED_VQBR_OPTIMIZER,
        maxiter=1,
        cobyla_tol=1e-8,
        su2_gates=ENFORCED_VQBR_SU2_GATES,
        entanglement=str(entanglement),
        loss=ENFORCED_VQBR_LOSS,
        random_state=int(seed),
        use_shot_noise=False,
        verbose=False,
    )

    Xp, yp, m0p, Sigma0p, Vp = model._validate_and_prepare_inputs(X, y, m0, Sigma0, V)
    n_samples, n_features = Xp.shape
    num_qubits = int(np.ceil(np.log2(n_features)))
    state_dim = 2**num_qubits

    model.num_qubits_ = num_qubits
    model._num_qubits = num_qubits
    model._n_samples = n_samples
    model._n_features = n_features
    model._state_dim = state_dim
    model._X = Xp
    model._y = yp
    model._V = float(Vp)
    model._configure_statevector_backend()

    precision = model._precision_from_covariance(Sigma0p)
    precision = 0.5 * (precision + precision.T)
    model._precision_matrix = precision
    precision_diag = np.diag(precision)
    off_diag = precision - np.diag(precision_diag)
    model._precision_is_diagonal = bool(np.allclose(off_diag, 0.0, atol=1e-12))
    model._precision_diag_padded = model._pad_to_pow2(precision_diag)
    model._dense_operator_padded = None

    if not model._precision_is_diagonal:
        dense_operator = np.zeros((state_dim, state_dim), dtype=complex)
        dense_operator[:n_features, :n_features] = float(Vp) * precision
        model._dense_operator_padded = dense_operator

    model._row_norms = np.linalg.norm(Xp, axis=1)
    if np.any(model._row_norms <= 0.0):
        raise ValueError("Each row in X must have non-zero norm for normalized state encoding.")

    model._Uxi_gates = [
        model._build_state_prep_gate(Xp[i], label=f"Ux_{i}") for i in range(n_samples)
    ]

    c_vec = float(Vp) * (precision @ m0p)
    model._c_norm = float(np.linalg.norm(c_vec))
    model._Uc_gate = None
    if model._c_norm > 0.0:
        model._Uc_gate = model._build_state_prep_gate(c_vec, label="Uc")

    model._ansatz = model._build_ansatz(num_qubits)
    model._theta_params = list(model._ansatz.parameters)
    model._num_params = len(model._theta_params)

    overlap_observable = SparsePauliOp.from_list([(("I" * num_qubits) + "Z", 1.0)])
    overlap_circuits = [
        _build_param_overlap_circuit(model, gate) for gate in model._Uxi_gates
    ]

    uc_circuit = None
    uc_observable = None
    if model._Uc_gate is not None:
        uc_circuit = _build_param_overlap_circuit(model, model._Uc_gate)
        uc_observable = overlap_observable

    d_circuit = model._ansatz.copy()
    if model._precision_is_diagonal:
        d_diagonal = float(Vp) * np.asarray(model._precision_diag_padded, dtype=float)
        d_observable = _build_diagonal_observable(d_diagonal)
    else:
        d_observable = SparsePauliOp.from_operator(Operator(model._dense_operator_padded)).simplify(
            atol=1e-12
        )

    return _PreparedVQBR(
        model=model,
        overlap_circuits=overlap_circuits,
        overlap_observable=overlap_observable,
        uc_circuit=uc_circuit,
        uc_observable=uc_observable,
        d_circuit=d_circuit,
        d_observable=d_observable,
    )


def _feature_direction_from_overlaps(
    *,
    X_train: np.ndarray,
    row_norms: np.ndarray,
    overlaps: np.ndarray,
    eps: float = EPS,
) -> np.ndarray | None:
    try:
        z = np.asarray(row_norms, dtype=float).reshape(-1) * np.asarray(overlaps, dtype=float).reshape(-1)
        phi_ls, *_ = np.linalg.lstsq(np.asarray(X_train, dtype=float), z, rcond=None)
        phi_ls = np.asarray(phi_ls, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(phi_ls))
        if norm <= eps:
            return None
        return phi_ls / norm
    except Exception:
        return None


def _build_training_logger(
    *,
    seed: int,
    method_label: str,
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
                f"    [seed={seed}, method={method_label}] eval {iteration:>4}/{maxiter:<4} | "
                f"objective={current_loss:.6e}, {_format_delta(current_loss, prev_loss)}",
                flush=True,
            )
        prev_loss = current_loss

    return _callback


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_elapsed(seconds: float) -> str:
    value = float(seconds)
    if not np.isfinite(value):
        return "nan"
    if value < 60.0:
        return f"{value:.3f}s"
    minutes, rem = divmod(value, 60.0)
    if value < 3600.0:
        return f"{int(minutes)}m {rem:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {rem:.1f}s"


def _describe_objective_workload(
    evaluator: _PrimitiveObjectiveEvaluator,
    *,
    batch_size: int,
) -> Dict[str, Any]:
    has_uc = evaluator._uc_circuit is not None and evaluator._uc_observable is not None
    hadamard_pub_count = int(batch_size) + (1 if has_uc else 0)
    d_term_pub_count = 1
    if isinstance(evaluator, _ZNEObjectiveEvaluator):
        return {
            "objective_eval_kind": "zne_split_jobs",
            "objective_estimator_jobs_per_eval": 2,
            "hadamard_test_pub_count": int(hadamard_pub_count),
            "d_term_pub_count": int(d_term_pub_count),
            "total_pub_count": int(hadamard_pub_count + d_term_pub_count),
        }
    return {
        "objective_eval_kind": "single_job",
        "objective_estimator_jobs_per_eval": 1,
        "hadamard_test_pub_count": int(hadamard_pub_count),
        "d_term_pub_count": int(d_term_pub_count),
        "total_pub_count": int(hadamard_pub_count + d_term_pub_count),
    }


def _write_training_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "method_key",
        "method_label",
        "backend_name",
        "status",
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


def _write_per_method_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "method_key",
        "method_label",
        "backend_name",
        "status",
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
        "expected_max_objective_evaluations",
        "postfit_snapshot_evaluations",
        "optimizer_nfev",
        "optimizer_iterations",
        "runtime_seconds_fit",
        "runtime_seconds_postfit",
        "runtime_seconds_total",
        "optimizer_success",
        "optimizer_message",
        "started_at_utc",
        "finished_at_utc",
        "error_message",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_cosine_table_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method_key", "method_label", "status", "cosine_similarity_to_classical"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_cosine_table(
    classical_metrics: Dict[str, Any],
    method_results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {
            "method_key": "classical",
            "method_label": "Classical solution",
            "status": "success",
            "cosine_similarity_to_classical": 1.0,
        }
    ]
    for method in method_results:
        metrics = dict(method.get("metrics", {}))
        rows.append(
            {
                "method_key": str(method.get("method_key", "")),
                "method_label": str(method.get("method_label", "")),
                "status": str(method.get("status", "unknown")),
                "cosine_similarity_to_classical": float(metrics.get("cosine_similarity", np.nan)),
            }
        )
    return rows


def _default_run_tag(
    *,
    args: argparse.Namespace,
    seeds: Sequence[int],
    optimizer: str,
    loss: str,
    su2_gates: Sequence[str],
) -> str:
    gates_tag = "-".join(_slugify_token(g) for g in su2_gates)
    ent_tag = _slugify_token(str(args.vqbr_entanglement))
    runner_tag = _slugify_token(str(getattr(args, "hardware_runner", "ibm")))
    backend_tag = (
        _slugify_token(str(args.ibm_backend).strip() or "ibm-kingston")
        if str(getattr(args, "hardware_runner", "ibm")).strip().lower() == "ibm"
        else runner_tag
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"ablation_real_hardware_N{int(args.n_samples)}_D{int(args.n_features)}_"
        f"seeds{int(len(seeds))}_maxiter{int(args.vqbr_maxiter)}_"
        f"{str(optimizer).lower()}_{_slugify_token(loss)}_"
        f"reps{int(args.vqbr_reps)}_gates{gates_tag}_ent{ent_tag}_"
        f"runner{runner_tag}_backend{backend_tag}_{timestamp}"
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
    per_method_name = _to_filename(args.per_method_csv_path, "per_method_metrics.csv")
    cosine_table_name = _to_filename(args.cosine_table_csv_path, "cosine_table.csv")
    run_config_name = _to_filename(args.run_config_path, "run_config.json")
    objective_plot_name = _to_filename(args.objective_plot_path, "objective_convergence.png")
    cosine_bar_plot_name = _to_filename(args.cosine_bar_plot_path, "cosine_similarity_bar.png")
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
        "per_method_csv": str(run_dir / per_method_name),
        "cosine_table_csv": str(run_dir / cosine_table_name),
        "run_config_json": str(run_dir / run_config_name),
        "objective_plot": str(run_dir / objective_plot_name),
        "cosine_bar_plot": str(run_dir / cosine_bar_plot_name),
        "console_log": str(run_dir / console_log_name) if console_log_name else "",
    }

    args.output_path = resolved["results_json"]
    args.training_csv_path = resolved["training_csv"]
    args.per_method_csv_path = resolved["per_method_csv"]
    args.cosine_table_csv_path = resolved["cosine_table_csv"]
    args.run_config_path = resolved["run_config_json"]
    args.objective_plot_path = resolved["objective_plot"]
    args.cosine_bar_plot_path = resolved["cosine_bar_plot"]
    args.console_log_path = resolved["console_log"]
    args.resolved_run_tag = str(run_tag)
    args.resolved_run_dir = str(run_dir)
    args._resolved_run_paths = resolved
    return resolved


def _generate_plot_artifacts(args: argparse.Namespace) -> Dict[str, Any]:
    plot_module = _load_real_hardware_plot_module()
    return plot_module.generate_artifacts(
        results_path=Path(args.output_path).expanduser().resolve(),
        training_csv_path=Path(args.training_csv_path).expanduser().resolve(),
        objective_plot_path=Path(args.objective_plot_path).expanduser().resolve(),
        cosine_bar_plot_path=Path(args.cosine_bar_plot_path).expanduser().resolve(),
        fig_width=float(args.plot_fig_width),
        fig_height=float(args.plot_fig_height),
        dpi=int(args.plot_dpi),
        show_seed_traces=False,
        prefer_training_csv=True,
        prefer_table_csv=True,
        cosine_table_input_csv_path=Path(args.cosine_table_csv_path).expanduser().resolve(),
        cosine_table_output_csv_path=None,
        table_tex_path=None,
    )


def _fit_method_with_evaluator(
    *,
    args: argparse.Namespace,
    seed: int,
    method_key: str,
    method_label: str,
    backend_name: str,
    prepared: _PreparedVQBR,
    evaluator: _PrimitiveObjectiveEvaluator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    w_closed: np.ndarray,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    theta_init = prepared.model._make_initial_point(prepared.model._num_params)
    full_batch = np.arange(int(X_train.shape[0]), dtype=int)
    history: List[_Snapshot] = []
    workload = _describe_objective_workload(evaluator, batch_size=int(full_batch.size))
    started_at_utc = _utc_now_iso()

    print(
        "  starting fit: "
        f"backend={backend_name}, train_size={int(X_train.shape[0])}, "
        f"maxiter={int(args.vqbr_maxiter)}, objective_jobs/eval={int(workload['objective_estimator_jobs_per_eval'])}",
        flush=True,
    )
    if int(workload["objective_estimator_jobs_per_eval"]) == 1:
        print(
            "  objective workload: "
            f"{int(workload['total_pub_count'])} pubs per evaluation "
            f"({int(workload['hadamard_test_pub_count'])} overlaps/prior + "
            f"{int(workload['d_term_pub_count'])} d-term)",
            flush=True,
        )
    else:
        print(
            "  objective workload: "
            f"{int(workload['hadamard_test_pub_count'])} Hadamard-test pubs + "
            f"{int(workload['d_term_pub_count'])} d-term pub per evaluation "
            "(split across two estimator jobs)",
            flush=True,
        )

    callback = None
    if bool(args.log_training):
        callback = _build_training_logger(
            seed=int(seed),
            method_label=str(method_label),
            maxiter=int(args.vqbr_maxiter),
            log_every_iter=int(args.log_every_iter),
        )

    def objective(theta_vec: np.ndarray) -> float:
        snapshot = evaluator.evaluate_snapshot(
            theta_vec,
            full_batch,
            scale_batch_to_full=False,
        )
        history.append(snapshot)
        if callback is not None:
            callback(len(history), snapshot)
        return float(snapshot.L_tilde)

    method_start = time.perf_counter()
    fit_start = time.perf_counter()
    result = minimize(
        fun=objective,
        x0=np.asarray(theta_init, dtype=float),
        method="COBYLA",
        options={
            "maxiter": int(args.vqbr_maxiter),
            "tol": float(args.vqbr_cobyla_tol),
            "disp": bool(args.vqbr_verbose),
        },
    )
    runtime_fit = float(time.perf_counter() - fit_start)
    print(
        "  optimizer finished: "
        f"success={bool(getattr(result, 'success', False))}, "
        f"nfev={getattr(result, 'nfev', 'n/a')}, "
        f"nit={getattr(result, 'nit', 'n/a')}, "
        f"message={str(getattr(result, 'message', '')).strip()}",
        flush=True,
    )

    theta_final = np.asarray(getattr(result, "x", theta_init), dtype=float).reshape(-1)
    trained_state = _normalize_state(prepared.model._statevector_from_theta(theta_final))
    postfit_start = time.perf_counter()
    final_snapshot, final_overlaps = evaluator.evaluate_snapshot_with_overlaps(
        theta_final,
        full_batch,
        scale_batch_to_full=False,
    )
    runtime_postfit = float(time.perf_counter() - postfit_start)

    phi = _feature_direction_from_overlaps(
        X_train=np.asarray(X_train, dtype=float),
        row_norms=np.asarray(prepared.model._row_norms, dtype=float),
        overlaps=np.asarray(final_overlaps, dtype=float),
    )
    if phi is not None:
        phi_source = "hardware_overlaps_lstsq"
    else:
        phi = _state_to_feature_direction(trained_state, n_features=int(args.n_features))
        phi_source = "statevector_phase_gauge"

    snapshot_for_solution = _select_snapshot_for_solution(
        history,
        target_objective=float(getattr(result, "fun", np.nan)),
    )
    if snapshot_for_solution is None:
        snapshot_for_solution = final_snapshot

    t_hat, w_hat, reconstruction_terms = _reconstruct_weights_from_formula(
        np.asarray(phi, dtype=float),
        snapshot_for_solution,
    )
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
    state_overlap_abs = float(np.abs(np.vdot(closed_form_state, trained_state)))

    history_rows_base, history_series = _extract_vqbr_history(int(seed), history)
    history_rows: List[Dict[str, Any]] = []
    for row in history_rows_base:
        row_out = dict(row)
        row_out["method_key"] = str(method_key)
        row_out["method_label"] = str(method_label)
        row_out["backend_name"] = str(backend_name)
        row_out["status"] = "success"
        history_rows.append(row_out)

    final_objective = float(getattr(result, "fun", np.nan))
    if not np.isfinite(final_objective):
        final_objective = float(getattr(final_snapshot, "L_tilde", np.nan))

    optimizer_nfev_raw = getattr(result, "nfev", np.nan)
    try:
        optimizer_nfev = float(optimizer_nfev_raw)
    except Exception:
        optimizer_nfev = float("nan")

    optimizer_iterations_raw = getattr(result, "nit", np.nan)
    try:
        optimizer_iterations = float(optimizer_iterations_raw)
    except Exception:
        optimizer_iterations = float("nan")

    runtime_total = float(time.perf_counter() - method_start)
    finished_at_utc = _utc_now_iso()
    print(
        "  completed: "
        f"objective_evals={len(history):d}, "
        f"postfit_snapshot_evals=1, "
        f"runtime_fit={_format_elapsed(runtime_fit)}, "
        f"runtime_postfit={_format_elapsed(runtime_postfit)}, "
        f"runtime_total={_format_elapsed(runtime_total)}",
        flush=True,
    )

    metrics = {
        "cosine_similarity": float(cosine_similarity),
        "feature_cosine_similarity": float(feature_cosine_similarity),
        "state_overlap_abs": float(state_overlap_abs),
        "train_rmse": float(regression_metrics["train_rmse"]),
        "test_rmse": float(regression_metrics["test_rmse"]),
        "train_r2": float(regression_metrics["train_r2"]),
        "test_r2": float(regression_metrics["test_r2"]),
        "t_hat": float(t_hat),
        "final_objective": float(final_objective),
        "objective_evaluations": int(len(history_series["objective_history"])),
        "optimizer_nfev": float(optimizer_nfev),
        "optimizer_iterations": float(optimizer_iterations),
        "reconstruction_terms": reconstruction_terms,
        "objective_history": [float(x) for x in history_series["objective_history"]],
        "a_hat_history": [float(x) for x in history_series["a_hat_history"]],
        "c_hat_history": [float(x) for x in history_series["c_hat_history"]],
        "d_hat_history": [float(x) for x in history_series["d_hat_history"]],
        "e_hat_history": [float(x) for x in history_series["e_hat_history"]],
        "h_hat_history": [float(x) for x in history_series["h_hat_history"]],
        "optimizer_success": bool(getattr(result, "success", False)),
        "optimizer_message": str(getattr(result, "message", "")),
        "feature_direction_source": str(phi_source),
        "theta_final": [float(x) for x in theta_final.tolist()],
        "backend_name": str(backend_name),
    }
    method_result = {
        "seed": int(seed),
        "method_key": str(method_key),
        "method_label": str(method_label),
        "backend_name": str(backend_name),
        "status": "success",
        "started_at_utc": str(started_at_utc),
        "finished_at_utc": str(finished_at_utc),
        "expected_max_objective_evaluations": int(args.vqbr_maxiter),
        "postfit_snapshot_evaluations": 1,
        "runtime_seconds_fit": float(runtime_fit),
        "runtime_seconds_postfit": float(runtime_postfit),
        "runtime_seconds_total": float(runtime_total),
        "objective_workload": workload,
        "metrics": metrics,
        "error_message": "",
    }
    return method_result, history_rows


def _failed_method_result(
    *,
    seed: int,
    method_key: str,
    method_label: str,
    backend_name: str,
    error: Exception,
) -> Dict[str, Any]:
    return {
        "seed": int(seed),
        "method_key": str(method_key),
        "method_label": str(method_label),
        "backend_name": str(backend_name),
        "status": "failed",
        "started_at_utc": _utc_now_iso(),
        "finished_at_utc": _utc_now_iso(),
        "expected_max_objective_evaluations": float("nan"),
        "postfit_snapshot_evaluations": float("nan"),
        "runtime_seconds_fit": float("nan"),
        "runtime_seconds_postfit": float("nan"),
        "runtime_seconds_total": float("nan"),
        "objective_workload": {},
        "metrics": {
            "cosine_similarity": float("nan"),
            "feature_cosine_similarity": float("nan"),
            "state_overlap_abs": float("nan"),
            "train_rmse": float("nan"),
            "test_rmse": float("nan"),
            "train_r2": float("nan"),
            "test_r2": float("nan"),
            "t_hat": float("nan"),
            "final_objective": float("nan"),
            "objective_evaluations": 0,
            "optimizer_nfev": float("nan"),
            "optimizer_iterations": float("nan"),
            "optimizer_success": False,
            "optimizer_message": "",
            "objective_history": [],
            "a_hat_history": [],
            "c_hat_history": [],
            "d_hat_history": [],
            "e_hat_history": [],
            "h_hat_history": [],
        },
        "error_message": str(error),
    }


def _per_method_csv_rows(
    *,
    seed: int,
    method_results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in method_results:
        metrics = dict(method.get("metrics", {}))
        rows.append(
            {
                "seed": int(seed),
                "method_key": str(method.get("method_key", "")),
                "method_label": str(method.get("method_label", "")),
                "backend_name": str(method.get("backend_name", "")),
                "status": str(method.get("status", "")),
                "cosine_similarity": float(metrics.get("cosine_similarity", np.nan)),
                "feature_cosine_similarity": float(
                    metrics.get("feature_cosine_similarity", np.nan)
                ),
                "state_overlap_abs": float(metrics.get("state_overlap_abs", np.nan)),
                "train_rmse": float(metrics.get("train_rmse", np.nan)),
                "test_rmse": float(metrics.get("test_rmse", np.nan)),
                "train_r2": float(metrics.get("train_r2", np.nan)),
                "test_r2": float(metrics.get("test_r2", np.nan)),
                "t_hat": float(metrics.get("t_hat", np.nan)),
                "final_objective": float(metrics.get("final_objective", np.nan)),
                "objective_evaluations": float(metrics.get("objective_evaluations", np.nan)),
                "expected_max_objective_evaluations": float(
                    method.get("expected_max_objective_evaluations", np.nan)
                ),
                "postfit_snapshot_evaluations": float(
                    method.get("postfit_snapshot_evaluations", np.nan)
                ),
                "optimizer_nfev": float(metrics.get("optimizer_nfev", np.nan)),
                "optimizer_iterations": float(metrics.get("optimizer_iterations", np.nan)),
                "runtime_seconds_fit": float(method.get("runtime_seconds_fit", np.nan)),
                "runtime_seconds_postfit": float(
                    method.get("runtime_seconds_postfit", np.nan)
                ),
                "runtime_seconds_total": float(method.get("runtime_seconds_total", np.nan)),
                "optimizer_success": bool(metrics.get("optimizer_success", False)),
                "optimizer_message": str(metrics.get("optimizer_message", "")),
                "started_at_utc": str(method.get("started_at_utc", "")),
                "finished_at_utc": str(method.get("finished_at_utc", "")),
                "error_message": str(method.get("error_message", "")),
            }
        )
    return rows


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    run_started_at_utc = _utc_now_iso()
    run_start = time.perf_counter()
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
    if args.ibm_shots <= 0:
        raise ValueError("--ibm-shots must be positive.")
    if args.log_every_iter <= 0:
        raise ValueError("--log-every-iter must be positive.")
    hardware_runner = str(args.hardware_runner).strip().lower()
    if hardware_runner not in {"ibm", "aer_noise", "skip"}:
        raise ValueError("--hardware-runner must be one of: ibm, aer_noise, skip.")

    _require_qiskit_components()
    if hardware_runner == "aer_noise":
        _require_aer_components()

    seeds = _resolve_seeds(
        num_seeds=int(args.num_seeds),
        seed_offset=int(args.seed_offset),
        explicit_seeds=args.seeds,
    )
    if len(seeds) != 1:
        raise ValueError(
            "This ablation is designed for a single seed due to IBM hardware budget. "
            "Use --num-seeds 1 or pass a single value via --seeds."
        )
    seed = int(seeds[0])

    run_paths = _resolve_run_paths(
        args=args,
        seeds=seeds,
        optimizer=ENFORCED_VQBR_OPTIMIZER,
        loss=ENFORCED_VQBR_LOSS,
        su2_gates=ENFORCED_VQBR_SU2_GATES,
    )
    zne_noise_factors = _parse_float_list(
        args.zne_noise_factors,
        default=(1.0, 3.0, 5.0),
    )

    ibm_backend = None
    use_hardware_like_runner = hardware_runner != "skip" and not bool(args.skip_ibm_hardware)
    ibm_backend_info: Dict[str, Any] = {
        "enabled": bool(use_hardware_like_runner),
        "resolved": False,
        "backend_name": str(args.ibm_backend).strip() or "ibm_kingston",
        "hardware_runner": str(hardware_runner),
        "error_message": "",
    }
    if hardware_runner == "ibm" and use_hardware_like_runner:
        ibm_backend, resolved_info = _resolve_ibm_runtime_backend(
            config_path=args.ibm_config,
            backend_override=args.ibm_backend,
            channel_override=args.ibm_channel,
            instance_override=args.ibm_instance,
            url_override=args.ibm_url,
            token_override=args.ibm_token,
        )
        ibm_backend_info.update({"resolved": True, **resolved_info})
    elif hardware_runner == "aer_noise" and use_hardware_like_runner:
        ibm_backend_info.update(
            {
                "resolved": True,
                "backend_name": "aer_noise",
                "runner_label": "Aer noise model",
                "aer_noise_method": str(args.aer_noise_method),
                "single_qubit_error": float(args.aer_single_qubit_error),
                "two_qubit_error": float(args.aer_two_qubit_error),
                "readout_p01": float(args.aer_readout_p01),
                "readout_p10": float(args.aer_readout_p10),
            }
        )

    print("=== Real-Hardware Ablation ===")
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
        f"VQBR config: optimizer={ENFORCED_VQBR_OPTIMIZER}, reps={int(args.vqbr_reps)}, "
        f"su2_gates={ENFORCED_VQBR_SU2_GATES}, entanglement={args.vqbr_entanglement}, "
        f"maxiter={int(args.vqbr_maxiter)}, cobyla_tol={float(args.vqbr_cobyla_tol):.3e}, "
        f"loss={ENFORCED_VQBR_LOSS}"
    )
    print(f"VQBR loss formula: {LOG_RATIO_LOSS_FORMULA}")
    print(f"Hardware runner: {hardware_runner}")
    print(f"Hardware-like branch enabled: {bool(use_hardware_like_runner)}")
    if hardware_runner == "ibm" and ibm_backend is not None:
        print(
            f"IBM backend: {ibm_backend_info.get('backend_name', _backend_name(ibm_backend))} | "
            f"shots={int(args.ibm_shots)} | zne_noise_factors={zne_noise_factors} | "
            f"zne_extrapolator={args.zne_extrapolator}"
        )
    elif hardware_runner == "aer_noise" and use_hardware_like_runner:
        print(
            "Aer noise model: "
            f"method={args.aer_noise_method}, "
            f"p1={float(args.aer_single_qubit_error):.6e}, "
            f"p2={float(args.aer_two_qubit_error):.6e}, "
            f"readout_p01={float(args.aer_readout_p01):.6e}, "
            f"readout_p10={float(args.aer_readout_p10):.6e}, "
            f"zne_noise_factors={zne_noise_factors}, "
            f"zne_extrapolator={args.zne_extrapolator}"
        )
    if use_hardware_like_runner:
        print(
            "ZNE scope: two-qubit gate folding on the Hadamard-test overlap circuits only; "
            "the d_hat precision circuit is left unmitigated."
        )
    print(f"Run directory: {run_paths['run_dir']}")
    print("")

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

    X_prior = np.asarray(X[split.prior_indices], dtype=float)
    y_prior = np.asarray(y[split.prior_indices], dtype=float)
    X_train = np.asarray(X[split.train_indices], dtype=float)
    y_train = np.asarray(y[split.train_indices], dtype=float)
    X_test = np.asarray(X[split.test_indices], dtype=float)
    y_test = np.asarray(y[split.test_indices], dtype=float)

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
    runtime_prior = float(time.perf_counter() - prior_start)
    print(
        "prior estimate: "
        f"V={float(prior_info['V']):.6e}, "
        f"prior_fit_mse={float(prior_info['prior_fit_mse']):.6e}, "
        f"mean_sigma0_diag={float(prior_info['mean_sigma0_diag']):.6e}"
    )

    closed_start = time.perf_counter()
    classical = ClosedFormMAPBayesianRegression()
    classical.fit(X_train, y_train, m0, Sigma0, V)
    w_closed = np.asarray(classical.get_weights(), dtype=float).reshape(-1)
    runtime_closed = float(time.perf_counter() - closed_start)
    classical_metrics = _evaluate_regression_metrics(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        w=w_closed,
    )
    print(
        "closed-form: "
        f"train_rmse={float(classical_metrics['train_rmse']):.6e}, "
        f"test_rmse={float(classical_metrics['test_rmse']):.6e}, "
        f"train_R2={float(classical_metrics['train_r2']):.6f}, "
        f"test_R2={float(classical_metrics['test_r2']):.6f}"
    )

    prepared = _prepare_vqbr_model(
        X=X_train,
        y=y_train,
        m0=m0,
        Sigma0=Sigma0,
        V=V,
        reps=int(args.vqbr_reps),
        entanglement=str(args.vqbr_entanglement),
        seed=int(seed),
    )

    method_specs: List[Dict[str, Any]] = [
        {
            "method_key": "vqbr_sim",
            "method_label": "Statevector simulation",
            "backend_name": "statevector_simulator",
            "kind": "sim",
        }
    ]
    if hardware_runner == "ibm" and ibm_backend is not None:
        ibm_name = str(ibm_backend_info.get("backend_name", _backend_name(ibm_backend)))
        method_specs.extend(
            [
                {
                    "method_key": "vqbr_ibm_no_zne",
                    "method_label": f"IBM hardware ({ibm_name}, w/o mitigation)",
                    "backend_name": ibm_name,
                    "kind": "ibm",
                    "use_zne": False,
                },
                {
                    "method_key": "vqbr_ibm_zne",
                    "method_label": f"IBM hardware ({ibm_name}, w/ ZNE)",
                    "backend_name": ibm_name,
                    "kind": "ibm",
                    "use_zne": True,
                },
            ]
        )
    elif hardware_runner == "aer_noise" and use_hardware_like_runner:
        method_specs.extend(
            [
                {
                    "method_key": "vqbr_ibm_no_zne",
                    "method_label": "Aer noise runner (w/o mitigation)",
                    "backend_name": "aer_noise",
                    "kind": "aer_noise",
                    "use_zne": False,
                },
                {
                    "method_key": "vqbr_ibm_zne",
                    "method_label": "Aer noise runner (w/ ZNE)",
                    "backend_name": "aer_noise",
                    "kind": "aer_noise",
                    "use_zne": True,
                },
            ]
        )

    method_results: List[Dict[str, Any]] = []
    training_rows: List[Dict[str, Any]] = []
    for spec in method_specs:
        method_key = str(spec["method_key"])
        method_label = str(spec["method_label"])
        backend_name = str(spec["backend_name"])
        print(method_label)
        try:
            if spec["kind"] == "sim":
                estimator = StatevectorEstimator()
                evaluator = _PrimitiveObjectiveEvaluator(
                    prepared=prepared,
                    estimator=estimator,
                    method_key=method_key,
                    method_label=method_label,
                    backend_name=backend_name,
                    transpile_backend=None,
                    transpile_optimization_level=int(args.ibm_transpile_optimization_level),
                )
            elif spec["kind"] == "aer_noise":
                estimators, aer_backends, estimator_info = _build_aer_noise_runner(
                    seed=int(seed),
                    use_zne=bool(spec.get("use_zne", False)),
                    zne_noise_factors=zne_noise_factors,
                    zne_extrapolator=str(args.zne_extrapolator),
                    aer_noise_method=str(args.aer_noise_method),
                    aer_single_qubit_error=float(args.aer_single_qubit_error),
                    aer_two_qubit_error=float(args.aer_two_qubit_error),
                    aer_readout_p01=float(args.aer_readout_p01),
                    aer_readout_p10=float(args.aer_readout_p10),
                )
                if bool(spec.get("use_zne", False)):
                    evaluator = _ZNEObjectiveEvaluator(
                        prepared=prepared,
                        estimators=estimators,
                        baseline_estimator=estimators[0],
                        noise_factors=estimator_info["noise_factors"],
                        zne_extrapolator=str(args.zne_extrapolator),
                        method_key=method_key,
                        method_label=method_label,
                        backend_name=backend_name,
                        transpile_backend=aer_backends[0],
                        transpile_optimization_level=int(args.ibm_transpile_optimization_level),
                    )
                    estimator_info = {
                        **estimator_info,
                        "zne_scope": "hadamard_test_overlap_and_prior_circuits_only",
                        "d_circuit_uses_zne": False,
                    }
                else:
                    evaluator = _PrimitiveObjectiveEvaluator(
                        prepared=prepared,
                        estimator=estimators[0],
                        method_key=method_key,
                        method_label=method_label,
                        backend_name=backend_name,
                        transpile_backend=aer_backends[0],
                        transpile_optimization_level=int(args.ibm_transpile_optimization_level),
                    )
                spec["estimator_info"] = estimator_info
            else:
                if bool(spec.get("use_zne", False)):
                    baseline_estimator, baseline_info = _build_ibm_estimator(
                        backend=ibm_backend,
                        shots=int(args.ibm_shots),
                        seed=int(seed),
                        use_zne=False,
                        zne_noise_factors=zne_noise_factors,
                        zne_extrapolator=str(args.zne_extrapolator),
                        zne_measure_mitigation=bool(args.zne_measure_mitigation),
                    )
                    zne_estimator, zne_info = _build_ibm_estimator(
                        backend=ibm_backend,
                        shots=int(args.ibm_shots),
                        seed=int(seed),
                        use_zne=True,
                        zne_noise_factors=zne_noise_factors,
                        zne_extrapolator=str(args.zne_extrapolator),
                        zne_measure_mitigation=bool(args.zne_measure_mitigation),
                    )
                    evaluator = _ZNEObjectiveEvaluator(
                        prepared=prepared,
                        estimators=[zne_estimator],
                        baseline_estimator=baseline_estimator,
                        noise_factors=zne_noise_factors,
                        zne_extrapolator=str(args.zne_extrapolator),
                        method_key=method_key,
                        method_label=method_label,
                        backend_name=backend_name,
                        transpile_backend=ibm_backend,
                        transpile_optimization_level=int(args.ibm_transpile_optimization_level),
                    )
                    spec["estimator_info"] = {
                        "baseline_estimator": baseline_info,
                        "zne_estimator": zne_info,
                        "zne_scope": "hadamard_test_overlap_and_prior_circuits_only",
                        "d_circuit_uses_zne": False,
                    }
                else:
                    estimator, estimator_info = _build_ibm_estimator(
                        backend=ibm_backend,
                        shots=int(args.ibm_shots),
                        seed=int(seed),
                        use_zne=False,
                        zne_noise_factors=zne_noise_factors,
                        zne_extrapolator=str(args.zne_extrapolator),
                        zne_measure_mitigation=bool(args.zne_measure_mitigation),
                    )
                    evaluator = _PrimitiveObjectiveEvaluator(
                        prepared=prepared,
                        estimator=estimator,
                        method_key=method_key,
                        method_label=method_label,
                        backend_name=backend_name,
                        transpile_backend=ibm_backend,
                        transpile_optimization_level=int(args.ibm_transpile_optimization_level),
                    )
                    spec["estimator_info"] = estimator_info

            method_result, method_history_rows = _fit_method_with_evaluator(
                args=args,
                seed=int(seed),
                method_key=method_key,
                method_label=method_label,
                backend_name=backend_name,
                prepared=prepared,
                evaluator=evaluator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                w_closed=w_closed,
            )
            if "estimator_info" in spec:
                method_result["estimator_info"] = spec["estimator_info"]
            method_results.append(method_result)
            training_rows.extend(method_history_rows)
            metrics = dict(method_result["metrics"])
            print(
                f"  cos={float(metrics['cosine_similarity']):.6f}, "
                f"feature_cos={float(metrics['feature_cosine_similarity']):.6f}, "
                f"train_rmse={float(metrics['train_rmse']):.6e}, "
                f"test_rmse={float(metrics['test_rmse']):.6e}, "
                f"t_hat={float(metrics['t_hat']):.6e}, "
                f"runtime={float(method_result['runtime_seconds_fit']):.3f}s"
            )
        except Exception as exc:
            failed = _failed_method_result(
                seed=int(seed),
                method_key=method_key,
                method_label=method_label,
                backend_name=backend_name,
                error=exc,
            )
            method_results.append(failed)
            print(f"  failed: {exc}")
            print("  traceback:")
            traceback.print_exc()

    cosine_table = _build_cosine_table(
        classical_metrics=classical_metrics,
        method_results=method_results,
    )
    per_method_rows = _per_method_csv_rows(
        seed=int(seed),
        method_results=method_results,
    )

    results = {
        "script": str(Path(__file__).resolve()),
        "timestamp_utc": _utc_now_iso(),
        "config": {
            "argv": _sanitize_argv(sys.argv),
            "n_samples": int(args.n_samples),
            "n_features": int(args.n_features),
            "noise_std": float(args.noise_std),
            "w_true_std": float(args.w_true_std),
            "prior_ratio": float(args.prior_ratio),
            "train_ratio": float(args.train_ratio),
            "test_ratio": float(args.test_ratio),
            "seed": int(seed),
            "vqbr": {
                "optimizer": str(ENFORCED_VQBR_OPTIMIZER),
                "loss": str(ENFORCED_VQBR_LOSS),
                "loss_formula": str(LOG_RATIO_LOSS_FORMULA),
                "reps": int(args.vqbr_reps),
                "entanglement": str(args.vqbr_entanglement),
                "su2_gates": list(ENFORCED_VQBR_SU2_GATES),
                "maxiter": int(args.vqbr_maxiter),
                "cobyla_tol": float(args.vqbr_cobyla_tol),
            },
            "ibm": {
                "hardware_runner": str(hardware_runner),
                "skip_hardware": bool(args.skip_ibm_hardware),
                "backend_name": str(ibm_backend_info.get("backend_name", "")),
                "shots": int(args.ibm_shots),
                "transpile_optimization_level": int(args.ibm_transpile_optimization_level),
                "zne_noise_factors": [float(x) for x in zne_noise_factors],
                "zne_extrapolator": str(args.zne_extrapolator),
                "zne_measure_mitigation": bool(args.zne_measure_mitigation),
                "zne_scope": "hadamard_test_overlap_and_prior_circuits_only",
                "resolved_info": ibm_backend_info,
            },
            "aer_noise": {
                "method": str(args.aer_noise_method),
                "single_qubit_error": float(args.aer_single_qubit_error),
                "two_qubit_error": float(args.aer_two_qubit_error),
                "readout_p01": float(args.aer_readout_p01),
                "readout_p10": float(args.aer_readout_p10),
            },
            "run_dir": str(run_paths["run_dir"]),
        },
        "prior": {key: float(value) for key, value in prior_info.items()},
        "classical": {
            "train_rmse": float(classical_metrics["train_rmse"]),
            "test_rmse": float(classical_metrics["test_rmse"]),
            "train_r2": float(classical_metrics["train_r2"]),
            "test_r2": float(classical_metrics["test_r2"]),
            "runtime_seconds_fit": float(runtime_closed),
            "runtime_seconds_prior": float(runtime_prior),
        },
        "methods": method_results,
        "cosine_table": cosine_table,
        "summary": {},
        "artifacts": {
            "run_dir": str(Path(run_paths["run_dir"]).expanduser().resolve()),
            "run_config_json": str(Path(args.run_config_path).expanduser().resolve()),
            "console_log": str(Path(args.console_log_path).expanduser().resolve())
            if str(args.console_log_path).strip()
            else "",
            "objective_convergence_plot": str(
                Path(args.objective_plot_path).expanduser().resolve()
            ),
            "cosine_similarity_bar_plot": str(
                Path(args.cosine_bar_plot_path).expanduser().resolve()
            ),
        },
    }

    run_config_payload = {
        "script": str(Path(__file__).resolve()),
        "timestamp_utc": _utc_now_iso(),
        "argv": _sanitize_argv(sys.argv),
        "args": _sanitize_args_for_results(args),
        "resolved": {
            "run_dir": str(run_paths["run_dir"]),
            "seed": int(seed),
            "ibm_backend_info": ibm_backend_info,
        },
    }

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_training_csv(Path(args.training_csv_path).expanduser().resolve(), training_rows)
    _write_per_method_csv(Path(args.per_method_csv_path).expanduser().resolve(), per_method_rows)
    _write_cosine_table_csv(
        Path(args.cosine_table_csv_path).expanduser().resolve(),
        cosine_table,
    )
    with Path(args.run_config_path).expanduser().resolve().open("w", encoding="utf-8") as f:
        json.dump(run_config_payload, f, indent=int(args.json_indent))
        f.write("\n")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=int(args.json_indent))
        f.write("\n")

    try:
        plot_outputs = _generate_plot_artifacts(args)
        if isinstance(plot_outputs, dict):
            artifacts = plot_outputs.get("artifacts", {})
            if isinstance(artifacts, dict):
                results["artifacts"].update(artifacts)
            cosine_gap_summary = plot_outputs.get("cosine_gap_summary", {})
            if isinstance(cosine_gap_summary, dict):
                results["summary"]["cosine_gap"] = cosine_gap_summary
                summary_text = str(cosine_gap_summary.get("summary_text", "")).strip()
                if summary_text:
                    print(summary_text)
    except Exception as exc:
        results["artifacts"]["plot_generation_error"] = str(exc)
        print(f"Plot generation warning: {exc}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=int(args.json_indent))
        f.write("\n")

    run_finished_at_utc = _utc_now_iso()
    run_total_seconds = float(time.perf_counter() - run_start)
    successful_methods = [m for m in method_results if str(m.get("status", "")) == "success"]
    failed_methods = [m for m in method_results if str(m.get("status", "")) != "success"]
    results["summary"]["run"] = {
        "started_at_utc": str(run_started_at_utc),
        "finished_at_utc": str(run_finished_at_utc),
        "total_runtime_seconds": float(run_total_seconds),
        "successful_methods": int(len(successful_methods)),
        "failed_methods": int(len(failed_methods)),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=int(args.json_indent))
        f.write("\n")

    print("")
    print(f"Run started at (UTC): {run_started_at_utc}")
    print(f"Run finished at (UTC): {run_finished_at_utc}")
    print(f"Total runtime: {_format_elapsed(run_total_seconds)}")
    print(f"Saved results JSON to: {output_path}")
    print(f"Saved run config JSON to: {Path(args.run_config_path).expanduser().resolve()}")
    if str(args.console_log_path).strip():
        print(f"Saved console log to: {Path(args.console_log_path).expanduser().resolve()}")
    print(f"Saved training CSV to: {Path(args.training_csv_path).expanduser().resolve()}")
    print(f"Saved per-method CSV to: {Path(args.per_method_csv_path).expanduser().resolve()}")
    print(f"Saved cosine table CSV to: {Path(args.cosine_table_csv_path).expanduser().resolve()}")
    print(f"Saved objective plot to: {Path(args.objective_plot_path).expanduser().resolve()}")
    print(f"Saved cosine bar plot to: {Path(args.cosine_bar_plot_path).expanduser().resolve()}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single-seed real-IBM ablation on ibm_kingston for a synthetic VQBR dataset. "
            "The run compares the statevector simulator, IBM hardware without mitigation, "
            "and IBM hardware with ZNE on the Hadamard-test measurement circuits, and "
            "exports the convergence and cosine-similarity figures."
        )
    )
    parser.add_argument("--N", "--n-samples", dest="n_samples", type=int, default=50)
    parser.add_argument("--D", "--n-features", dest="n_features", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--w-true-std", type=float, default=1.0)

    parser.add_argument("--prior-ratio", type=float, default=0.2)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--test-ratio", type=float, default=0.2)

    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Single explicit seed. Defaults to 0 for the hardware ablation.",
    )

    parser.add_argument("--prior-bootstrap-samples", type=int, default=64)
    parser.add_argument("--prior-ridge", type=float, default=1e-6)
    parser.add_argument("--prior-sigma-floor", type=float, default=1e-4)
    parser.add_argument("--V-floor", type=float, default=1e-8)

    parser.add_argument("--vqbr-reps", type=int, default=2)
    parser.add_argument("--vqbr-entanglement", type=str, default="linear")
    parser.add_argument("--vqbr-maxiter", type=int, default=100)
    parser.add_argument("--vqbr-cobyla-tol", type=float, default=1e-8)
    parser.add_argument("--vqbr-verbose", action="store_true")

    parser.set_defaults(log_training=True)
    parser.add_argument("--log-training", action="store_true")
    parser.add_argument("--no-log-training", dest="log_training", action="store_false")
    parser.add_argument("--log-every-iter", type=int, default=1)

    parser.add_argument(
        "--hardware-runner",
        type=str,
        default="ibm",
        choices=("ibm", "aer_noise", "skip"),
        help=(
            "Select the hardware-like execution path: real IBM Runtime, "
            "local Aer noise-model simulation, or skip the hardware-like branch."
        ),
    )

    parser.add_argument(
        "--ibm-config",
        type=str,
        default=str(DEFAULT_IBM_CONFIG_PATH),
        help="Path to the IBM Runtime config JSON containing token/API key and instance details.",
    )
    parser.add_argument("--ibm-backend", type=str, default="ibm_kingston")
    parser.add_argument("--ibm-channel", type=str, default="")
    parser.add_argument("--ibm-instance", type=str, default="")
    parser.add_argument("--ibm-url", type=str, default="")
    parser.add_argument("--ibm-token", type=str, default="")
    parser.add_argument("--ibm-shots", type=int, default=1024)
    parser.add_argument("--ibm-transpile-optimization-level", type=int, default=1)
    parser.add_argument(
        "--zne-noise-factors",
        type=str,
        default="1,3,5",
        help="Comma-separated IBM Runtime ZNE noise factors.",
    )
    parser.add_argument(
        "--zne-extrapolator",
        type=str,
        default="linear",
        help="IBM Runtime ZNE extrapolator.",
    )
    parser.set_defaults(zne_measure_mitigation=False)
    parser.add_argument("--zne-measure-mitigation", action="store_true")
    parser.add_argument(
        "--skip-ibm-hardware",
        action="store_true",
        help=(
            "Skip the hardware-like branch and only execute the classical + simulator paths. "
            "This is equivalent to --hardware-runner skip."
        ),
    )
    parser.add_argument("--aer-noise-method", type=str, default="density_matrix")
    parser.add_argument("--aer-single-qubit-error", type=float, default=1e-3)
    parser.add_argument("--aer-two-qubit-error", type=float, default=1e-2)
    parser.add_argument("--aer-readout-p01", type=float, default=2e-2)
    parser.add_argument("--aer-readout-p10", type=float, default=2e-2)

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
        "--per-method-csv-path",
        type=str,
        default="per_method_metrics.csv",
        help="Per-method CSV filename (saved inside the run directory).",
    )
    parser.add_argument(
        "--cosine-table-csv-path",
        type=str,
        default="cosine_table.csv",
        help="Cosine-table CSV filename (saved inside the run directory).",
    )
    parser.add_argument(
        "--run-config-path",
        type=str,
        default="run_config.json",
        help="Run-config JSON filename (saved inside the run directory).",
    )
    parser.add_argument(
        "--objective-plot-path",
        type=str,
        default="objective_convergence.png",
        help="Objective convergence figure filename (saved inside the run directory).",
    )
    parser.add_argument(
        "--cosine-bar-plot-path",
        type=str,
        default="cosine_similarity_bar.png",
        help="Final cosine-similarity bar-chart filename (saved inside the run directory).",
    )
    parser.add_argument("--plot-fig-width", type=float, default=3.0)
    parser.add_argument("--plot-fig-height", type=float, default=2.5)
    parser.add_argument("--plot-dpi", type=int, default=1000)
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
