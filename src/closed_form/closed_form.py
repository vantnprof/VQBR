from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ObjectiveSnapshot:
    objective: float
    data_term: float
    prior_term: float


class ClosedFormMAPBayesianRegression:
    """
    Classical closed-form MAP Bayesian linear regression.

    This class intentionally mirrors the public shape of
    `VariationalQuantumBayesianRegression`:
    - `fit(X, y, m0, Sigma, V)` returns a normalized, power-of-two padded state vector.
    - `get_state()` returns the same trained state.
    - The learned classical MAP weights are available via `get_weights()`.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = float(eps)

        self.history_: list[ObjectiveSnapshot] = []
        self.result_: Optional[np.ndarray] = None
        self.map_weights_: Optional[np.ndarray] = None
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
        X, y, m0, Sigma, V = self._validate_and_prepare_inputs(X, y, m0, Sigma, V)

        precision = self._precision_from_covariance(Sigma)
        gram = X.T @ X + V * precision
        rhs = X.T @ y + V * (precision @ m0)
        w_map = self._solve_map_system(gram, rhs)

        trained_state = self._weights_to_state(w_map)
        snapshot = self._objective_snapshot(X, y, m0, precision, V, w_map)

        self.history_.clear()
        self.history_.append(snapshot)

        self.result_ = w_map.copy()
        self.map_weights_ = w_map.copy()
        self.trained_state_ = trained_state
        self.num_qubits_ = int(np.log2(trained_state.size))

        return trained_state.copy()

    def get_state(self) -> np.ndarray:
        if self.trained_state_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.trained_state_.copy()

    def get_weights(self) -> np.ndarray:
        if self.map_weights_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.map_weights_.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        w = self.get_weights()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X.shape}.")
        if X.shape[1] != w.shape[0]:
            raise ValueError(
                f"X must have {w.shape[0]} features to match fitted weights, got {X.shape[1]}."
            )
        return X @ w

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
        if n_samples == 0 or n_features == 0:
            raise ValueError("X must be non-empty.")

        if y.shape[0] != n_samples:
            raise ValueError(f"y must have length {n_samples}, got {y.shape[0]}.")
        if m0.shape[0] != n_features:
            raise ValueError(f"m0 must have length {n_features}, got {m0.shape[0]}.")

        if Sigma.ndim == 1:
            if Sigma.shape[0] != n_features:
                raise ValueError(
                    f"Sigma must have length {n_features}, got {Sigma.shape[0]}."
                )
            if np.any(Sigma <= 0.0):
                raise ValueError("All entries of 1D Sigma must be positive.")
        elif Sigma.ndim == 2:
            if Sigma.shape != (n_features, n_features):
                raise ValueError(
                    f"Sigma must be ({n_features}, {n_features}), got {Sigma.shape}."
                )
            if not np.allclose(Sigma, Sigma.T, atol=1e-12):
                raise ValueError("Sigma must be symmetric.")
            try:
                np.linalg.cholesky(Sigma)
            except np.linalg.LinAlgError as exc:
                raise ValueError("Sigma must be positive definite.") from exc
        else:
            raise ValueError("Sigma must be either a 1D diagonal vector or a 2D matrix.")

        V = float(V)
        if V < 0.0:
            raise ValueError("V must be non-negative.")

        return X, y, m0, Sigma, V

    def _precision_from_covariance(self, Sigma: np.ndarray) -> np.ndarray:
        if Sigma.ndim == 1:
            return np.diag(1.0 / Sigma)
        eye = np.eye(Sigma.shape[0], dtype=float)
        return np.linalg.solve(Sigma, eye)

    def _solve_map_system(self, gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            # Handles singular/ill-conditioned cases (e.g. V=0 and rank-deficient X).
            return np.linalg.lstsq(gram, rhs, rcond=self.eps)[0]

    def _objective_snapshot(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m0: np.ndarray,
        precision: np.ndarray,
        V: float,
        w_map: np.ndarray,
    ) -> ObjectiveSnapshot:
        residual = X @ w_map - y
        delta = w_map - m0
        data_term = float(np.dot(residual, residual))
        prior_term = float(delta.T @ precision @ delta)
        objective = data_term + V * prior_term
        return ObjectiveSnapshot(
            objective=float(objective),
            data_term=data_term,
            prior_term=prior_term,
        )

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

    def _weights_to_state(self, w_map: np.ndarray) -> np.ndarray:
        state = self._pad_to_pow2(np.asarray(w_map, dtype=complex).reshape(-1))
        norm = float(np.linalg.norm(state))
        if norm <= self.eps:
            # Zero MAP vector has no direction; keep a valid normalized state for API parity.
            state = np.zeros_like(state, dtype=complex)
            state[0] = 1.0 + 0.0j
            return state
        return state / norm


class ClosedFormBR(ClosedFormMAPBayesianRegression):
    pass


def closed_form_map_solution(
    X: np.ndarray,
    y: np.ndarray,
    m0: np.ndarray,
    Sigma: np.ndarray,
    V: float,
) -> np.ndarray:
    model = ClosedFormMAPBayesianRegression()
    model.fit(X, y, m0, Sigma, V)
    return model.get_weights()


__all__ = [
    "ObjectiveSnapshot",
    "ClosedFormMAPBayesianRegression",
    "ClosedFormBR",
    "closed_form_map_solution",
]
