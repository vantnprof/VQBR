from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation
    from qiskit.quantum_info import Statevector

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


class VariationalQuantumBayesianRegression:
    """
    Variational Quantum Bayesian Regression (VQBR).

    Notes:
    - The current implementation follows the diagonal-prior estimator in the paper.
      Therefore, `Sigma` is required to be diagonal.
    - `fit` returns the trained statevector amplitudes prepared by the optimized ansatz.
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
        batch_optimizer_maxiter: int = 2,
        learning_rate: float = 0.05,
        spsa_perturbation: float = 0.1,
        su2_gates: Sequence[str] = ("ry",),
        entanglement: str = "linear",
        eps: float = 1e-12,
        random_state: Optional[int] = None,
        initial_point: Optional[np.ndarray] = None,
        use_shot_noise: bool = True,
        verbose: bool = False,
        ansatz: Optional[Any] = None,
    ) -> None:
        self.shots = int(shots)
        self.batch_size = int(batch_size)
        self.reps = int(reps)
        self.optimizer = optimizer
        self.maxiter = int(maxiter)
        self.epochs = epochs
        self.shuffle_batches = bool(shuffle_batches)
        self.batch_optimizer_maxiter = int(batch_optimizer_maxiter)
        self.learning_rate = float(learning_rate)
        self.spsa_perturbation = float(spsa_perturbation)
        self.su2_gates = tuple(su2_gates)
        self.entanglement = entanglement
        self.eps = float(eps)
        self.initial_point = initial_point
        self.use_shot_noise = bool(use_shot_noise)
        self.verbose = bool(verbose)
        self.ansatz = ansatz

        self._rng = np.random.default_rng(random_state)

        self.history_: List[ObjectiveSnapshot] = []
        self.result_: Optional[OptimizeResult] = None
        self.theta_: Optional[np.ndarray] = None
        self.trained_state_: Optional[np.ndarray] = None
        self.num_qubits_: Optional[int] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m0: np.ndarray,
        Sigma: np.ndarray,
        V: float,
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

        precision_diag = 1.0 / sigma_diag
        self._precision_diag_padded = self._pad_to_pow2(precision_diag)

        self._row_norms = np.linalg.norm(X, axis=1)
        if np.any(self._row_norms <= 0.0):
            raise ValueError("Each row in X must have non-zero norm for normalized state encoding.")

        self._Uxi_gates = [
            self._build_state_prep_gate(X[i], label=f"Ux_{i}") for i in range(n_samples)
        ]

        c = precision_diag * m0
        self._c_norm = float(np.linalg.norm(c))
        self._Uc_gate = None
        if self._c_norm > 0.0:
            self._Uc_gate = self._build_state_prep_gate(c, label="Uc")

        self._ansatz = self._build_ansatz(num_qubits)
        self._theta_params = list(self._ansatz.parameters)
        self._num_params = len(self._theta_params)

        theta_init = self._make_initial_point(self._num_params)

        self.history_.clear()

        def objective(theta_vec: np.ndarray) -> float:
            snap = self._reduced_objective(theta_vec)
            self.history_.append(snap)
            return snap.L_tilde

        options: Dict[str, Any] = {"maxiter": self.maxiter}
        if self.verbose:
            options["disp"] = True

        self.result_ = minimize(
            fun=objective,
            x0=theta_init,
            method=self.optimizer,
            options=options,
            tol=1e-12,
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

        if _HAS_EFFICIENT_SU2:
            return efficient_su2(
                num_qubits,
                su2_gates=list(self.su2_gates),
                entanglement=self.entanglement,
                reps=self.reps,
            )
        return EfficientSU2(
            num_qubits,
            su2_gates=list(self.su2_gates),
            entanglement=self.entanglement,
            reps=self.reps,
        )

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

    def _estimate_overlap(self, Uv_gate: Any, theta_vec: np.ndarray) -> float:
        qc = self._build_overlap_circuit(Uv_gate, theta_vec)
        sv = Statevector.from_instruction(qc)
        p0 = float(sv.probabilities(qargs=[0])[0])
        p0 = float(np.clip(p0, 0.0, 1.0))

        if self.use_shot_noise:
            n0 = self._rng.binomial(self.shots, p0)
            p0 = n0 / float(self.shots)

        s_hat = 2.0 * p0 - 1.0
        return float(np.clip(s_hat, -1.0, 1.0))

    def _estimate_d_hat(self, theta_vec: np.ndarray) -> float:
        qc = QuantumCircuit(self._num_qubits)
        bound_ansatz = self._ansatz.assign_parameters(self._theta_bind(theta_vec), inplace=False)
        qc.compose(bound_ansatz, inplace=True)

        sv = Statevector.from_instruction(qc)
        probs = np.asarray(sv.probabilities(), dtype=float)
        probs = probs / np.sum(probs)

        if self.use_shot_noise:
            counts = self._rng.multinomial(self.shots, probs)
            return float(np.dot(self._precision_diag_padded, counts) / float(self.shots))
        return float(np.dot(self._precision_diag_padded, probs))

    def _reduced_objective(self, theta_vec: np.ndarray) -> ObjectiveSnapshot:
        batch_size = min(self.batch_size, self._n_samples)
        batch = self._rng.choice(self._n_samples, size=batch_size, replace=False)

        s = np.array([self._estimate_overlap(self._Uxi_gates[i], theta_vec) for i in batch], dtype=float)
        r = self._row_norms[batch]
        yb = self._y[batch]

        a_hat = float(np.sum((r * s) ** 2))
        c_hat = float(np.sum(yb * (r * s)))

        h_hat = 0.0
        e_hat = 0.0
        if self._Uc_gate is not None and self._c_norm > 0.0:
            h_hat = self._estimate_overlap(self._Uc_gate, theta_vec)
            e_hat = float(self._c_norm * h_hat)

        d_hat = self._estimate_d_hat(theta_vec)

        denom = a_hat + self._V * d_hat + self.eps
        numerator = c_hat + self._V * e_hat
        L_tilde = -((numerator**2) / denom)

        return ObjectiveSnapshot(
            L_tilde=float(L_tilde),
            a_hat=a_hat,
            c_hat=c_hat,
            d_hat=d_hat,
            e_hat=e_hat,
            h_hat=float(h_hat),
            batch_indices=batch,
        )

    def _statevector_from_theta(self, theta_vec: np.ndarray) -> np.ndarray:
        qc = QuantumCircuit(self._num_qubits)
        bound_ansatz = self._ansatz.assign_parameters(self._theta_bind(theta_vec), inplace=False)
        qc.compose(bound_ansatz, inplace=True)
        sv = Statevector.from_instruction(qc)
        state = np.asarray(sv.data, dtype=complex)
        norm = np.linalg.norm(state)
        if norm <= 0.0:
            raise RuntimeError("Invalid trained state with zero norm.")
        return state / norm


class VQBR(VariationalQuantumBayesianRegression):
    pass


__all__ = ["ObjectiveSnapshot", "VariationalQuantumBayesianRegression", "VQBR"]
