from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation
    from qiskit.quantum_info import Statevector
    try:
        from qiskit_aer import AerSimulator
    except Exception:
        AerSimulator = None  # type: ignore[assignment]

    try:
        from qiskit.circuit.library import efficient_su2

        _HAS_EFFICIENT_SU2 = True
        _EFFICIENT_SU2_ERROR: Optional[Exception] = None
    except Exception as exc:
        _HAS_EFFICIENT_SU2 = False
        _EFFICIENT_SU2_ERROR = exc
        from qiskit.circuit.library import EfficientSU2

except Exception as exc:  # pragma: no cover - import guard for optional dependency
    QuantumCircuit = None  # type: ignore[assignment]
    StatePreparation = None  # type: ignore[assignment]
    Statevector = None  # type: ignore[assignment]
    AerSimulator = None  # type: ignore[assignment]
    EfficientSU2 = None  # type: ignore[assignment]
    _HAS_EFFICIENT_SU2 = False
    _EFFICIENT_SU2_ERROR = exc


@dataclass
class ObjectiveSnapshot:
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


class VariationalQuantumBayesianRegression:
    """
    Variational Quantum Bayesian Regression (VQBR).

    Notes:
    - The current implementation follows the diagonal-prior estimator in the paper.
      Therefore, `Sigma` is required to be diagonal.
    - `fit` returns the trained statevector amplitudes prepared by the optimized ansatz.
    - If `qiskit-aer` is installed and a GPU device is available, training uses
      `AerSimulator(method="statevector", device="GPU")` automatically.
    - `loss` controls the scalar objective form:
      - `log_ratio` (default): log(a_hat + d_hat) - 2 * log(|c_hat + e_hat|).
      - `neg_ratio`: -((c_hat + e_hat)^2 / (a_hat + d_hat)).
    - `simulator_device_`, `simulator_backend_`, and `simulator_reason_` report
      which statevector backend was selected (and why).
    """

    def __init__(
        self,
        shots: int = 2**10,
        batch_size: int = 2,
        reps: int = 1,
        optimizer: str = "COBYLA",
        maxiter: int = 100,
        epochs: Optional[int] = None,
        shuffle_batches: bool = True,
        learning_rate: Optional[float] = None,
        spsa_perturbation: float = 0.1,
        su2_gates: Sequence[Any] | str = ("ry",),
        entanglement: str = "linear",
        eps: float = 1e-12,
        loss: str = "log_ratio",
        random_state: Optional[int] = None,
        initial_point: Optional[np.ndarray] = None,
        use_shot_noise: bool = True,
        verbose: bool = False,
        ansatz: Optional[Any] = None,
        prefer_gpu: bool = True,
        require_gpu: bool = False,
    ) -> None:
        self.shots = int(shots)
        self.batch_size = int(batch_size)
        self.reps = int(reps)
        self.optimizer = self._normalize_optimizer(optimizer)
        self.maxiter = int(maxiter)
        self.epochs = epochs
        self.shuffle_batches = bool(shuffle_batches)
        self.learning_rate = None if learning_rate is None else float(learning_rate)
        self.spsa_perturbation = float(spsa_perturbation)
        self.su2_gates = self._normalize_su2_gates(su2_gates)
        self.entanglement = entanglement
        self.eps = float(eps)
        self.loss = self._normalize_loss(loss)
        self.initial_point = initial_point
        self.use_shot_noise = bool(use_shot_noise)
        self.verbose = bool(verbose)
        self.ansatz = ansatz
        self.prefer_gpu = bool(prefer_gpu)
        self.require_gpu = bool(require_gpu)

        self._rng = np.random.default_rng(random_state)

        self.history_: List[ObjectiveSnapshot] = []
        self.result_: Optional[OptimizeResult] = None
        self.theta_: Optional[np.ndarray] = None
        self.trained_state_: Optional[np.ndarray] = None
        self.num_qubits_: Optional[int] = None
        self._batch_schedule: List[np.ndarray] = []
        self._batch_cursor: int = 0
        self._batch_epoch: int = 0
        self.simulator_device_: str = "CPU"
        self.simulator_backend_: str = "statevector_cpu"
        self.simulator_reason_: str = "Using CPU statevector simulator."
        self._statevector_backend: Optional[Any] = None

        if self.learning_rate is not None and self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be strictly positive when provided.")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m0: np.ndarray,
        Sigma: np.ndarray,
        V: float,
        iteration_callback: Optional[Callable[[int, ObjectiveSnapshot], None]] = None,
    ) -> np.ndarray:
        """
        Train VQBR and return the trained quantum statevector.

        Parameters
        ----------
        X : ndarray, shape (N, D)
            Training matrix.
        y : ndarray, shape (N,) or (N, 1)
            Targets.
        m0 : ndarray, shape (D,) or (D, 1)
            Prior mean.
        Sigma : ndarray, shape (D,) or (D, D)
            Diagonal prior covariance.
        V : float
            Prior-strength scaling in the MAP objective.
        """
        self._require_qiskit()
        X, y, m0, sigma_diag, V = self._validate_and_prepare_inputs(X, y, m0, Sigma, V)

        n_samples, n_features = X.shape
        num_qubits = int(np.ceil(np.log2(n_features)))
        state_dim = 2**num_qubits

        self.num_qubits_ = num_qubits
        self._num_qubits = num_qubits
        self._n_samples = n_samples
        self._n_features = n_features
        self._state_dim = state_dim
        self._X = X
        self._y = y
        self._V = V
        self._configure_statevector_backend()
        self._log_backend_selection()

        precision_diag = 1.0 / sigma_diag
        self._precision_diag_padded = self._pad_to_pow2(precision_diag)

        self._row_norms = np.linalg.norm(X, axis=1)
        if np.any(self._row_norms <= 0.0):
            raise ValueError("Each row in X must have non-zero norm for normalized state encoding.")

        self._Uxi_gates = [
            self._build_state_prep_gate(X[i], label=f"Ux_{i}") for i in range(n_samples)
        ]

        c = V * precision_diag * m0
        self._c_norm = float(np.linalg.norm(c))
        self._Uc_gate = None
        if self._c_norm > 0.0:
            self._Uc_gate = self._build_state_prep_gate(c, label="Uc")

        self._ansatz = self._build_ansatz(num_qubits)
        self._theta_params = list(self._ansatz.parameters)
        self._num_params = len(self._theta_params)

        theta_init = self._make_initial_point(self._num_params)

        self.history_.clear()
        self._reset_batch_schedule()
        self.result_ = self._fit_with_scipy(
            theta_init,
            iteration_callback=iteration_callback,
        )
        self.theta_ = np.asarray(self.result_.x, dtype=float)
        self.trained_state_ = self._statevector_from_theta(self.theta_)
        return self.trained_state_.copy()

    def get_state(self) -> np.ndarray:
        if self.trained_state_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.trained_state_.copy()

    def get_theta(self) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.theta_.copy()

    def _require_qiskit(self) -> None:
        if QuantumCircuit is None or StatePreparation is None or Statevector is None:
            detail = "" if _EFFICIENT_SU2_ERROR is None else f" Original error: {_EFFICIENT_SU2_ERROR}"
            raise ImportError(
                "Qiskit is required. Install with `pip install qiskit`."
                + detail
            )

    def _validate_and_prepare_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m0: np.ndarray,
        Sigma: np.ndarray,
        V: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        m0 = np.asarray(m0, dtype=float).reshape(-1)
        Sigma = np.asarray(Sigma, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X.shape}.")
        n_samples, n_features = X.shape

        if y.shape[0] != n_samples:
            raise ValueError(f"y must have length {n_samples}, got {y.shape[0]}.")
        if m0.shape[0] != n_features:
            raise ValueError(f"m0 must have length {n_features}, got {m0.shape[0]}.")

        sigma_diag: np.ndarray
        if Sigma.ndim == 1:
            if Sigma.shape[0] != n_features:
                raise ValueError(
                    f"Sigma must have length {n_features}, got {Sigma.shape[0]}."
                )
            sigma_diag = Sigma.copy()
        elif Sigma.ndim == 2:
            if Sigma.shape != (n_features, n_features):
                raise ValueError(
                    f"Sigma must be ({n_features}, {n_features}), got {Sigma.shape}."
                )
            if not np.allclose(Sigma, Sigma.T, atol=1e-12):
                raise ValueError("Sigma must be symmetric.")
            off_diag = Sigma - np.diag(np.diag(Sigma))
            if not np.allclose(off_diag, 0.0, atol=1e-12):
                raise ValueError(
                    "This implementation currently supports only diagonal Sigma."
                )
            sigma_diag = np.diag(Sigma)
        else:
            raise ValueError("Sigma must be either a 1D diagonal vector or a 2D matrix.")

        if np.any(sigma_diag <= 0.0):
            raise ValueError("All diagonal entries of Sigma must be positive.")

        V = float(V)
        if V < 0.0:
            raise ValueError("V must be non-negative.")

        return X, y, m0, sigma_diag, V

    def _make_initial_point(self, n_params: int) -> np.ndarray:
        if self.initial_point is None:
            return 0.01 * self._rng.standard_normal(n_params)
        theta_init = np.asarray(self.initial_point, dtype=float).reshape(-1)
        if theta_init.shape[0] != n_params:
            raise ValueError(
                f"initial_point must have length {n_params}, got {theta_init.shape[0]}."
            )
        return theta_init

    @staticmethod
    def _normalize_loss(loss: str) -> str:
        mode = str(loss).strip().lower()
        aliases = {
            "log_ratio": "log_ratio",
            "log": "log_ratio",
            "neg_ratio": "neg_ratio",
            "ratio": "neg_ratio",
            "raw_ratio": "neg_ratio",
        }
        resolved = aliases.get(mode)
        if resolved is None:
            supported = ("log_ratio", "neg_ratio")
            raise ValueError(
                f"Unsupported loss='{loss}'. Supported values: {supported}. "
                "Accepted aliases: 'log', 'ratio', 'raw_ratio'."
            )
        return resolved

    @staticmethod
    def _normalize_su2_gates(su2_gates: Sequence[Any] | str) -> tuple[Any, ...]:
        if isinstance(su2_gates, str):
            gates: List[Any] = [su2_gates]
        else:
            try:
                gates = list(su2_gates)
            except TypeError as exc:
                raise TypeError(
                    "su2_gates must be a gate name string or a sequence of gates."
                ) from exc

        if len(gates) == 0:
            raise ValueError("su2_gates must contain at least one gate.")

        normalized: List[Any] = []
        for gate in gates:
            if isinstance(gate, str):
                token = gate.strip().lower()
                if token == "":
                    raise ValueError("su2_gates must not contain empty gate names.")
                normalized.append(token)
                continue
            normalized.append(gate)
        return tuple(normalized)

    @staticmethod
    def _coerce_pauli_su2_gate_strings(su2_gates: Sequence[Any]) -> tuple[List[Any], bool]:
        has_pauli_string = any(
            isinstance(gate, str) and gate in {"x", "y", "z"} for gate in su2_gates
        )
        if not has_pauli_string:
            return list(su2_gates), False

        try:
            from qiskit.circuit.library.standard_gates import XGate, YGate, ZGate
        except Exception:
            return list(su2_gates), False

        lookup = {
            "x": XGate(),
            "y": YGate(),
            "z": ZGate(),
        }
        converted: List[Any] = []
        changed = False
        for gate in su2_gates:
            if isinstance(gate, str):
                mapped = lookup.get(gate)
                if mapped is not None:
                    converted.append(mapped)
                    changed = True
                    continue
            converted.append(gate)
        return converted, changed

    def _compute_objective_value(self, numerator: float, denom: float) -> float:
        if self.loss == "log_ratio":
            # Stability transform that is monotonic-equivalent to maximizing numerator^2 / denom.
            return float(np.log(denom) - 2.0 * np.log(np.abs(numerator) + self.eps))
        # Direct objective: minimize the negative ratio.
        return float(-((numerator * numerator) / denom))

    def _reset_batch_schedule(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self._n_samples <= 0:
            raise ValueError("X must contain at least one sample.")
        self._batch_schedule = []
        self._batch_cursor = 0
        self._batch_epoch = 0
        self._refresh_batch_schedule()

    def _refresh_batch_schedule(self) -> None:
        order = np.arange(self._n_samples, dtype=int)
        if self.shuffle_batches:
            self._rng.shuffle(order)

        batch_size = min(self.batch_size, self._n_samples)
        self._batch_schedule = [
            order[i : i + batch_size] for i in range(0, self._n_samples, batch_size)
        ]
        self._batch_cursor = 0
        self._batch_epoch += 1

    def _next_batch_indices(self) -> np.ndarray:
        if not self._batch_schedule:
            self._refresh_batch_schedule()
        if self._batch_cursor >= len(self._batch_schedule):
            self._refresh_batch_schedule()

        batch = self._batch_schedule[self._batch_cursor]
        self._batch_cursor += 1
        return batch

    @staticmethod
    def _normalize_optimizer(optimizer: str) -> str:
        opt = str(optimizer).strip().upper()
        valid = ("COBYLA", "SPSA")
        if opt not in valid:
            raise ValueError(
                f"Unsupported optimizer='{optimizer}'. Supported values: {valid}."
            )
        return opt

    def _resolve_learning_rate(self, theta_init: np.ndarray) -> float:
        if self.learning_rate is not None:
            return float(self.learning_rate)

        theta_abs = np.abs(np.asarray(theta_init, dtype=float).reshape(-1))
        theta_scale = float(np.median(theta_abs)) if theta_abs.size > 0 else 0.0
        if not np.isfinite(theta_scale) or theta_scale <= self.eps:
            theta_scale = 0.01

        # Auto-tune around initial parameter scale and keep it in a stable range.
        auto_lr = float(np.clip(5.0 * theta_scale, 1e-3, 0.5))
        return auto_lr

    def _fit_with_scipy(
        self,
        theta_init: np.ndarray,
        iteration_callback: Optional[Callable[[int, ObjectiveSnapshot], None]],
    ) -> OptimizeResult:
        full_batch = np.arange(self._n_samples, dtype=int)
        scipy_method = "COBYLA"
        spsa_lr = self._resolve_learning_rate(theta_init)

        if self.optimizer == "SPSA" and self.verbose:
            print(
                "Optimizer 'SPSA' uses SciPy COBYLA on the mini-batch objective "
                f"(batch_size={self.batch_size}, learning_rate={spsa_lr:.6g})."
            )

        def objective(theta_vec: np.ndarray) -> float:
            if self.optimizer == "SPSA":
                # Mini-batch objective so `batch_size` directly affects optimization.
                batch_snap = self._reduced_objective(theta_vec)
                # Track full-data objective as a monitoring metric.
                full_snap = self._objective_on_batch(
                    theta_vec,
                    full_batch,
                    use_shot_noise=False,
                    scale_batch_to_full=False,
                )
                self._append_snapshot(
                    ObjectiveSnapshot(
                        L_tilde=float(full_snap.L_tilde),
                        a_hat=batch_snap.a_hat,
                        c_hat=batch_snap.c_hat,
                        d_hat=batch_snap.d_hat,
                        e_hat=batch_snap.e_hat,
                        h_hat=batch_snap.h_hat,
                        batch_indices=batch_snap.batch_indices.copy(),
                        batch_L_tilde=float(batch_snap.L_tilde),
                        full_L_tilde=float(full_snap.L_tilde),
                        epoch=batch_snap.epoch,
                        batch_position=batch_snap.batch_position,
                    ),
                    iteration_callback=iteration_callback,
                )
                return float(batch_snap.L_tilde)

            full_snap = self._objective_on_batch(
                theta_vec,
                full_batch,
                use_shot_noise=self.use_shot_noise,
                scale_batch_to_full=False,
            )
            self._append_snapshot(
                ObjectiveSnapshot(
                    L_tilde=float(full_snap.L_tilde),
                    a_hat=full_snap.a_hat,
                    c_hat=full_snap.c_hat,
                    d_hat=full_snap.d_hat,
                    e_hat=full_snap.e_hat,
                    h_hat=full_snap.h_hat,
                    batch_indices=full_batch.copy(),
                    batch_L_tilde=float(full_snap.L_tilde),
                    full_L_tilde=float(full_snap.L_tilde),
                    epoch=full_snap.epoch,
                    batch_position=full_snap.batch_position,
                ),
                iteration_callback=iteration_callback,
            )
            return float(full_snap.L_tilde)

        options: Dict[str, Any] = {"maxiter": self.maxiter}
        if self.optimizer == "SPSA":
            # In SPSA mode we still solve with SciPy COBYLA; use learning_rate as
            # the initial trust-region radius so it controls update aggressiveness.
            options["rhobeg"] = float(max(spsa_lr, self.eps))
        if self.verbose:
            options["disp"] = True

        return minimize(
            fun=objective,
            x0=theta_init,
            method=scipy_method,
            options=options,
            tol=1e-12,
        )

    def _append_snapshot(
        self,
        snapshot: ObjectiveSnapshot,
        iteration_callback: Optional[Callable[[int, ObjectiveSnapshot], None]],
    ) -> None:
        self.history_.append(snapshot)
        if iteration_callback is not None:
            iteration_callback(len(self.history_), snapshot)

    @staticmethod
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

    def _build_ansatz(self, num_qubits: int) -> QuantumCircuit:
        if self.ansatz is not None:
            if callable(self.ansatz):
                ansatz = self.ansatz(num_qubits)
            else:
                ansatz = self.ansatz
            if not isinstance(ansatz, QuantumCircuit):
                raise TypeError("Custom ansatz must be a qiskit QuantumCircuit.")
            if ansatz.num_qubits != num_qubits:
                raise ValueError(
                    f"Custom ansatz num_qubits={ansatz.num_qubits}, expected {num_qubits}."
                )
            return ansatz

        su2_gates = list(self.su2_gates)
        fallback_gates, has_fallback = self._coerce_pauli_su2_gate_strings(su2_gates)

        if _HAS_EFFICIENT_SU2:
            try:
                return efficient_su2(
                    num_qubits,
                    su2_gates=su2_gates,
                    entanglement=self.entanglement,
                    reps=self.reps,
                )
            except Exception:
                if has_fallback:
                    return efficient_su2(
                        num_qubits,
                        su2_gates=fallback_gates,
                        entanglement=self.entanglement,
                        reps=self.reps,
                    )
                raise

        try:
            return EfficientSU2(
                num_qubits,
                su2_gates=su2_gates,
                entanglement=self.entanglement,
                reps=self.reps,
            )
        except Exception:
            if has_fallback:
                return EfficientSU2(
                    num_qubits,
                    su2_gates=fallback_gates,
                    entanglement=self.entanglement,
                    reps=self.reps,
                )
            raise

    def _build_state_prep_gate(self, vec: np.ndarray, label: str) -> Any:
        vec = np.asarray(vec, dtype=complex).reshape(-1)
        vec = self._pad_to_pow2(vec)
        norm = np.linalg.norm(vec)
        if norm <= 0.0:
            raise ValueError("State preparation vector norm must be non-zero.")
        vec = vec / norm

        prep = StatePreparation(vec)
        if hasattr(prep, "to_gate"):
            return prep.to_gate(label=label)
        return prep

    def _configure_statevector_backend(self) -> None:
        self._statevector_backend = None
        self.simulator_device_ = "CPU"
        self.simulator_backend_ = "statevector_cpu"
        self.simulator_reason_ = "Using CPU statevector simulator."

        def _cpu_fallback(reason: str) -> None:
            self._statevector_backend = None
            self.simulator_device_ = "CPU"
            self.simulator_backend_ = "statevector_cpu"
            self.simulator_reason_ = reason
            if self.verbose:
                print(reason)

        if not self.prefer_gpu:
            reason = "Using CPU statevector simulator (prefer_gpu=False)."
            _cpu_fallback(reason)
            if self.require_gpu:
                raise RuntimeError(
                    "GPU execution was required but disabled with prefer_gpu=False."
                )
            return
        if AerSimulator is None:
            reason = (
                "Using CPU statevector simulator "
                "(qiskit-aer not installed; install qiskit-aer-gpu for GPU support)."
            )
            _cpu_fallback(reason)
            if self.require_gpu:
                raise RuntimeError(
                    "GPU execution was required but qiskit-aer is unavailable. "
                    "Install `qiskit-aer-gpu` in the active environment."
                )
            return

        available_devices: set[str] = set()
        if hasattr(AerSimulator, "available_devices"):
            try:
                available_devices = {
                    str(device).strip().upper() for device in AerSimulator.available_devices()
                }
            except Exception:
                available_devices = set()
        if not available_devices:
            try:
                available_devices = {
                    str(device).strip().upper()
                    for device in AerSimulator().available_devices()
                }
            except Exception:
                available_devices = set()

        if available_devices and "GPU" not in available_devices:
            reason = (
                "Using CPU statevector simulator "
                "(GPU device unavailable in Aer available_devices())."
            )
            _cpu_fallback(reason)
            if self.require_gpu:
                raise RuntimeError(
                    "GPU execution was required but Aer reports no GPU device. "
                    f"Available devices: {sorted(available_devices)}"
                )
            return

        try:
            self._statevector_backend = AerSimulator(method="statevector", device="GPU")
            self.simulator_device_ = "GPU"
            self.simulator_backend_ = "AerSimulator(statevector,GPU)"
            self.simulator_reason_ = "Using AerSimulator(method='statevector', device='GPU')."
            if self.verbose:
                print("Using AerSimulator(method='statevector', device='GPU').")
        except Exception as exc:
            reason = "Using CPU statevector simulator (failed to initialize Aer GPU backend)."
            _cpu_fallback(reason)
            if self.require_gpu:
                raise RuntimeError(
                    "GPU execution was required but Aer GPU backend initialization failed."
                ) from exc

    def _log_backend_selection(self) -> None:
        print(
            "[VQBR] backend="
            f"{self.simulator_backend_}, "
            f"device={self.simulator_device_}, "
            f"reason={self.simulator_reason_}",
            flush=True,
        )

    def _simulate_statevector(self, qc: QuantumCircuit) -> np.ndarray:
        if self._statevector_backend is None:
            sv = Statevector.from_instruction(qc)
            return np.asarray(sv.data, dtype=complex)

        try:
            sim_qc = qc.copy()
            if hasattr(sim_qc, "save_statevector"):
                sim_qc.save_statevector()
            result = self._statevector_backend.run(sim_qc).result()
            state = None
            try:
                data = result.data(0)
                state = data.get("statevector")
                if state is None:
                    state = data.get("final_statevector")
            except Exception:
                state = None
            if state is None:
                state = result.get_statevector(sim_qc)
            return np.asarray(state, dtype=complex)
        except Exception as exc:
            # Fall back if backend execution fails (e.g., GPU transient/unavailable at runtime).
            if self.require_gpu:
                raise RuntimeError(
                    "GPU execution was required but Aer GPU statevector simulation failed."
                ) from exc
            self._statevector_backend = None
            self.simulator_device_ = "CPU"
            self.simulator_backend_ = "statevector_cpu"
            self.simulator_reason_ = (
                "Falling back to CPU statevector simulator (Aer GPU execution failed)."
            )
            if self.verbose:
                print(self.simulator_reason_)
            sv = Statevector.from_instruction(qc)
            return np.asarray(sv.data, dtype=complex)

    def _theta_bind(self, theta_vec: np.ndarray) -> Dict[Any, float]:
        theta_vec = np.asarray(theta_vec, dtype=float).reshape(-1)
        if theta_vec.shape[0] != self._num_params:
            raise ValueError(
                f"Expected theta of length {self._num_params}, got {theta_vec.shape[0]}."
            )
        return {p: float(v) for p, v in zip(self._theta_params, theta_vec)}

    def _build_overlap_circuit(self, Uv_gate: Any, theta_vec: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(1 + self._num_qubits)
        anc = 0
        work = list(range(1, 1 + self._num_qubits))

        qc.h(anc)
        qc.x(anc)
        qc.append(Uv_gate.control(1), [anc] + work)
        qc.x(anc)

        bound_ansatz = self._ansatz.assign_parameters(self._theta_bind(theta_vec), inplace=False)
        controlled_ansatz = bound_ansatz.to_gate(label="Utheta").control(1)
        qc.append(controlled_ansatz, [anc] + work)

        qc.h(anc)
        return qc

    def _estimate_overlap(
        self,
        Uv_gate: Any,
        theta_vec: np.ndarray,
        use_shot_noise: Optional[bool] = None,
    ) -> float:
        qc = self._build_overlap_circuit(Uv_gate, theta_vec)
        state = self._simulate_statevector(qc)
        p0 = float(np.sum(np.abs(state[0::2]) ** 2))
        p0 = float(np.clip(p0, 0.0, 1.0))

        apply_shot_noise = self.use_shot_noise if use_shot_noise is None else bool(use_shot_noise)
        if apply_shot_noise:
            n0 = self._rng.binomial(self.shots, p0)
            p0 = n0 / float(self.shots)

        s_hat = 2.0 * p0 - 1.0
        return float(np.clip(s_hat, -1.0, 1.0))

    def _estimate_d_hat(
        self, theta_vec: np.ndarray, use_shot_noise: Optional[bool] = None
    ) -> float:
        qc = QuantumCircuit(self._num_qubits)
        bound_ansatz = self._ansatz.assign_parameters(self._theta_bind(theta_vec), inplace=False)
        qc.compose(bound_ansatz, inplace=True)

        state = self._simulate_statevector(qc)
        probs = np.abs(state) ** 2
        probs = probs / np.sum(probs)

        apply_shot_noise = self.use_shot_noise if use_shot_noise is None else bool(use_shot_noise)
        if apply_shot_noise:
            counts = self._rng.multinomial(self.shots, probs)
            # now includes V inside
            return float(self._V * np.dot(self._precision_diag_padded, counts) / float(self.shots))

        # now includes V inside
        return float(self._V * np.dot(self._precision_diag_padded, probs))

    def _objective_on_batch(
        self,
        theta_vec: np.ndarray,
        batch_indices: np.ndarray,
        use_shot_noise: Optional[bool],
        scale_batch_to_full: bool,
    ) -> ObjectiveSnapshot:
        batch = np.asarray(batch_indices, dtype=int).reshape(-1)
        if batch.size == 0:
            raise ValueError("batch_indices must not be empty.")

        s = np.array(
            [
                self._estimate_overlap(
                    self._Uxi_gates[i], theta_vec, use_shot_noise=use_shot_noise
                )
                for i in batch
            ],
            dtype=float,
        )
        r = self._row_norms[batch]
        yb = self._y[batch]

        scale = float(self._n_samples / batch.size) if scale_batch_to_full else 1.0
        a_hat = float(scale * np.sum((r * s) ** 2))
        c_hat = float(scale * np.sum(yb * (r * s)))

        h_hat = 0.0
        e_hat = 0.0
        if self._Uc_gate is not None and self._c_norm > 0.0:
            h_hat = self._estimate_overlap(
                self._Uc_gate, theta_vec, use_shot_noise=use_shot_noise
            )
            e_hat = float(self._c_norm * h_hat)

        d_hat = self._estimate_d_hat(theta_vec, use_shot_noise=use_shot_noise)

        denom = a_hat + d_hat + self.eps
        numerator = c_hat + e_hat
        L_tilde = self._compute_objective_value(numerator=numerator, denom=denom)

        return ObjectiveSnapshot(
            L_tilde=float(L_tilde),
            a_hat=a_hat,
            c_hat=c_hat,
            d_hat=d_hat,
            e_hat=e_hat,
            h_hat=float(h_hat),
            batch_indices=batch.copy(),
            batch_L_tilde=float(L_tilde),
            full_L_tilde=float(L_tilde),
            epoch=self._batch_epoch,
            batch_position=self._batch_cursor,
        )

    def _reduced_objective(self, theta_vec: np.ndarray) -> ObjectiveSnapshot:
        batch = self._next_batch_indices()
        return self._objective_on_batch(
            theta_vec,
            batch,
            use_shot_noise=self.use_shot_noise,
            scale_batch_to_full=True,
        )

    def _statevector_from_theta(self, theta_vec: np.ndarray) -> np.ndarray:
        qc = QuantumCircuit(self._num_qubits)
        bound_ansatz = self._ansatz.assign_parameters(self._theta_bind(theta_vec), inplace=False)
        qc.compose(bound_ansatz, inplace=True)
        state = self._simulate_statevector(qc)
        norm = np.linalg.norm(state)
        if norm <= 0.0:
            raise RuntimeError("Invalid trained state with zero norm.")
        return state / norm


class VQBR(VariationalQuantumBayesianRegression):
    pass


__all__ = ["ObjectiveSnapshot", "VariationalQuantumBayesianRegression", "VQBR"]
