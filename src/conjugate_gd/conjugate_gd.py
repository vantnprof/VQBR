from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CGSnapshot:
    iteration: int
    relative_residual: float
    residual_norm: float
    alpha: float
    beta: float


class ConjugateGradientMAPBayesianRegression:
    """
    Conjugate-gradient baseline for MAP Bayesian linear regression.

    Solves, for a fixed iteration budget T (`maxiter`):

        (X^T X + V Sigma^{-1}) w = X^T y + V Sigma^{-1} m0

    Notes
    -----
    - `fit` always runs a fixed-length history of `maxiter` entries for fair
      budget-matched comparison.
    - Optional Jacobi preconditioning can be enabled with
      `preconditioner="jacobi"`.
    - `fit` returns a normalized, power-of-two padded state vector so it can be
      compared against VQBR state outputs.
    """

    def __init__(
        self,
        maxiter: int = 100,
        preconditioner: str = "none",
        eps: float = 1e-12,
        initial_point: Optional[np.ndarray] = None,
    ) -> None:
        self.maxiter = int(maxiter)
        self.preconditioner = str(preconditioner).lower()
        self.eps = float(eps)
        self.initial_point = initial_point

        if self.maxiter <= 0:
            raise ValueError("maxiter must be a positive integer.")
        if self.preconditioner not in {"none", "jacobi"}:
            raise ValueError("preconditioner must be either 'none' or 'jacobi'.")

        self.history_: list[CGSnapshot] = []
        self.residual_history_: Optional[np.ndarray] = None
        self.result_: Optional[np.ndarray] = None
        self.map_weights_: Optional[np.ndarray] = None
        self.trained_state_: Optional[np.ndarray] = None
        self.relative_residual_: Optional[float] = None
        self.num_qubits_: Optional[int] = None
        self.system_matrix_: Optional[np.ndarray] = None
        self.rhs_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m0: np.ndarray,
        Sigma: np.ndarray,
        V: float,
    ) -> np.ndarray:
        """
        Fit CG/PCG baseline on the MAP linear system and return a normalized state vector.

        Parameters
        ----------
        X : ndarray, shape (N, D)
            Training matrix.
        y : ndarray, shape (N,) or (N, 1)
            Targets.
        m0 : ndarray, shape (D,) or (D, 1)
            Prior mean.
        Sigma : ndarray, shape (D,) or (D, D)
            Prior covariance.
        V : float
            Prior-strength scaling in the MAP objective.
        """
        X, y, m0, Sigma, V = self._validate_and_prepare_inputs(X, y, m0, Sigma, V)
        n_features = X.shape[1]

        precision = self._precision_from_covariance(Sigma)
        A = X.T @ X + V * precision
        b = X.T @ y + V * (precision @ m0)

        w0 = self._make_initial_point(n_features)
        w_map, history = self._run_cg(A, b, w0)

        state = self._weights_to_state(w_map)

        self.history_ = history
        self.residual_history_ = np.array([snap.relative_residual for snap in history], dtype=float)
        self.result_ = w_map.copy()
        self.map_weights_ = w_map.copy()
        self.trained_state_ = state
        self.relative_residual_ = float(self.residual_history_[-1]) if self.residual_history_.size else None
        self.num_qubits_ = int(np.log2(state.size))
        self.system_matrix_ = A.copy()
        self.rhs_ = b.copy()

        return state.copy()

    def get_state(self) -> np.ndarray:
        if self.trained_state_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.trained_state_.copy()

    def get_weights(self) -> np.ndarray:
        if self.map_weights_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.map_weights_.copy()

    def get_relative_residual(self) -> float:
        if self.relative_residual_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return float(self.relative_residual_)

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

    def _make_initial_point(self, n_features: int) -> np.ndarray:
        if self.initial_point is None:
            return np.zeros(n_features, dtype=float)
        w0 = np.asarray(self.initial_point, dtype=float).reshape(-1)
        if w0.shape[0] != n_features:
            raise ValueError(
                f"initial_point must have length {n_features}, got {w0.shape[0]}."
            )
        return w0.copy()

    def _precision_from_covariance(self, Sigma: np.ndarray) -> np.ndarray:
        if Sigma.ndim == 1:
            return np.diag(1.0 / Sigma)
        eye = np.eye(Sigma.shape[0], dtype=float)
        return np.linalg.solve(Sigma, eye)

    def _build_preconditioner_diag(self, A: np.ndarray) -> Optional[np.ndarray]:
        if self.preconditioner == "none":
            return None
        diag = np.diag(A)
        if np.any(np.abs(diag) <= self.eps):
            raise ValueError("Jacobi preconditioner is undefined because A has near-zero diagonal.")
        return 1.0 / diag

    def _relative_residual(self, r: np.ndarray, b_norm: float) -> tuple[float, float]:
        r_norm = float(np.linalg.norm(r))
        if b_norm > self.eps:
            return r_norm / b_norm, r_norm
        return r_norm, r_norm

    def _append_idle_history(
        self,
        history: list[CGSnapshot],
        start_iteration: int,
        residual_norm: float,
        relative_residual: float,
    ) -> None:
        for it in range(start_iteration, self.maxiter + 1):
            history.append(
                CGSnapshot(
                    iteration=it,
                    relative_residual=float(relative_residual),
                    residual_norm=float(residual_norm),
                    alpha=0.0,
                    beta=0.0,
                )
            )

    def _run_cg(
        self,
        A: np.ndarray,
        b: np.ndarray,
        w0: np.ndarray,
    ) -> tuple[np.ndarray, list[CGSnapshot]]:
        w = w0.copy()
        r = b - A @ w
        b_norm = float(np.linalg.norm(b))

        m_inv_diag = self._build_preconditioner_diag(A)
        if m_inv_diag is None:
            z = r.copy()
        else:
            z = m_inv_diag * r

        p = z.copy()
        rz_old = float(np.dot(r, z))
        history: list[CGSnapshot] = []

        rel_res, r_norm = self._relative_residual(r, b_norm)
        if r_norm <= self.eps:
            self._append_idle_history(history, 1, r_norm, rel_res)
            return w, history

        for it in range(1, self.maxiter + 1):
            Ap = A @ p
            pAp = float(np.dot(p, Ap))

            if abs(pAp) <= self.eps:
                self._append_idle_history(history, it, r_norm, rel_res)
                return w, history

            alpha = rz_old / pAp
            w = w + alpha * p
            r = r - alpha * Ap
            rel_res, r_norm = self._relative_residual(r, b_norm)

            if m_inv_diag is None:
                z = r.copy()
            else:
                z = m_inv_diag * r
            rz_new = float(np.dot(r, z))

            beta = 0.0
            if abs(rz_old) > self.eps:
                beta = rz_new / rz_old

            history.append(
                CGSnapshot(
                    iteration=it,
                    relative_residual=float(rel_res),
                    residual_norm=float(r_norm),
                    alpha=float(alpha),
                    beta=float(beta),
                )
            )

            if r_norm <= self.eps or abs(rz_new) <= self.eps:
                self._append_idle_history(history, it + 1, r_norm, rel_res)
                return w, history

            p = z + beta * p
            rz_old = rz_new

        return w, history

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
            state = np.zeros_like(state, dtype=complex)
            state[0] = 1.0 + 0.0j
            return state
        return state / norm


class CGDBR(ConjugateGradientMAPBayesianRegression):
    pass


def conjugate_gradient_map_solution(
    X: np.ndarray,
    y: np.ndarray,
    m0: np.ndarray,
    Sigma: np.ndarray,
    V: float,
    maxiter: int = 100,
    preconditioner: str = "none",
    eps: float = 1e-12,
    initial_point: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    model = ConjugateGradientMAPBayesianRegression(
        maxiter=maxiter,
        preconditioner=preconditioner,
        eps=eps,
        initial_point=initial_point,
    )
    model.fit(X, y, m0, Sigma, V=V)
    return model.get_weights(), model.get_relative_residual()


__all__ = [
    "CGSnapshot",
    "ConjugateGradientMAPBayesianRegression",
    "CGDBR",
    "conjugate_gradient_map_solution",
]
