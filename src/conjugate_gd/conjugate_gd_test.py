from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the sibling module importable when running pytest from repo root.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from conjugate_gd import (  # noqa: E402
    CGDBR,
    ConjugateGradientMAPBayesianRegression,
    conjugate_gradient_map_solution,
)


def _make_problem(
    *,
    seed: int = 19,
    n_samples: int = 30,
    n_features: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.normal(size=n_features)
    y = X @ true_w + 0.01 * rng.normal(size=n_samples)
    m0 = rng.normal(size=n_features)

    A = rng.normal(size=(n_features, n_features))
    Sigma = A.T @ A + 0.4 * np.eye(n_features)
    V = 0.2
    return X, y, m0, Sigma, V


def _map_direct_solution(
    X: np.ndarray,
    y: np.ndarray,
    m0: np.ndarray,
    Sigma: np.ndarray,
    V: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if Sigma.ndim == 1:
        precision = np.diag(1.0 / Sigma)
    else:
        precision = np.linalg.solve(Sigma, np.eye(Sigma.shape[0]))
    A = X.T @ X + V * precision
    b = X.T @ y + V * (precision @ m0)
    w = np.linalg.solve(A, b)
    return w, A, b


def test_fit_matches_direct_map_solution_full_covariance() -> None:
    X, y, m0, Sigma, V = _make_problem(n_features=6)
    maxiter = 20
    model = ConjugateGradientMAPBayesianRegression(maxiter=maxiter)

    state = model.fit(X, y, m0, Sigma, V)
    expected_w, A, b = _map_direct_solution(X, y, m0, Sigma, V)

    np.testing.assert_allclose(model.get_weights(), expected_w, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(state, model.get_state())
    assert state.shape == (8,)
    assert np.isclose(np.linalg.norm(state), 1.0)
    assert model.num_qubits_ == 3
    assert len(model.history_) == maxiter
    assert model.residual_history_ is not None
    assert model.residual_history_.shape == (maxiter,)

    manual_rel = np.linalg.norm(A @ model.get_weights() - b) / np.linalg.norm(b)
    assert np.isclose(model.get_relative_residual(), manual_rel, rtol=1e-10, atol=1e-12)
    assert model.get_relative_residual() < 1e-8


def test_fit_matches_direct_map_solution_diagonal_covariance() -> None:
    X, y, m0, _, V = _make_problem(seed=23, n_features=5)
    sigma_diag = np.full(5, 1.3)
    model = ConjugateGradientMAPBayesianRegression(maxiter=12)
    model.fit(X, y, m0, sigma_diag, V)

    expected_w, _, _ = _map_direct_solution(X, y, m0, sigma_diag, V)
    np.testing.assert_allclose(model.get_weights(), expected_w, rtol=1e-9, atol=1e-9)


def test_helper_solution_matches_class_api() -> None:
    X, y, m0, Sigma, V = _make_problem(seed=29, n_features=6)
    model = ConjugateGradientMAPBayesianRegression(maxiter=15)
    model.fit(X, y, m0, Sigma, V)

    w_helper, rr_helper = conjugate_gradient_map_solution(
        X, y, m0, Sigma, V=V, maxiter=15, preconditioner="none"
    )
    np.testing.assert_allclose(w_helper, model.get_weights(), rtol=1e-12, atol=1e-12)
    assert np.isclose(rr_helper, model.get_relative_residual(), rtol=1e-12, atol=1e-12)


def test_predict_matches_matrix_vector_product() -> None:
    X, y, m0, Sigma, V = _make_problem(seed=31, n_features=4)
    model = ConjugateGradientMAPBayesianRegression(maxiter=10)
    model.fit(X, y, m0, Sigma, V)

    pred = model.predict(X)
    np.testing.assert_allclose(pred, X @ model.get_weights(), rtol=1e-12, atol=1e-12)
    assert pred.shape == (X.shape[0],)


def test_fixed_budget_history_is_padded_after_early_convergence() -> None:
    X = np.eye(3, dtype=float)
    y = np.array([1.0, -2.0, 0.5], dtype=float)
    m0 = np.zeros(3, dtype=float)
    Sigma = np.ones(3, dtype=float)
    V = 1.0
    maxiter = 7

    model = ConjugateGradientMAPBayesianRegression(maxiter=maxiter)
    model.fit(X, y, m0, Sigma, V)

    assert len(model.history_) == maxiter
    # With A = 2I, CG converges in one iteration from zero init.
    assert model.history_[0].alpha != 0.0
    for snap in model.history_[1:]:
        assert snap.alpha == 0.0
        assert snap.beta == 0.0
    assert model.get_relative_residual() <= 1e-12


def test_jacobi_preconditioner_path_and_alias() -> None:
    X, y, m0, Sigma, V = _make_problem(seed=37, n_features=6)

    model = ConjugateGradientMAPBayesianRegression(maxiter=14, preconditioner="jacobi")
    state = model.fit(X, y, m0, Sigma, V)
    assert np.isfinite(model.get_relative_residual())
    assert model.get_relative_residual() >= 0.0
    assert state.shape == (8,)
    assert len(model.history_) == 14

    alias_model = CGDBR(maxiter=14, preconditioner="jacobi")
    alias_model.fit(X, y, m0, Sigma, V)
    assert alias_model.get_weights().shape == (X.shape[1],)


def test_zero_rhs_has_zero_relative_residual_and_basis_state() -> None:
    X = np.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])
    y = np.zeros(3)
    m0 = np.zeros(2)
    Sigma = np.ones(2)
    V = 1.0

    model = ConjugateGradientMAPBayesianRegression(maxiter=5)
    state = model.fit(X, y, m0, Sigma, V)

    np.testing.assert_allclose(model.get_weights(), np.zeros(2))
    assert model.get_relative_residual() == 0.0
    np.testing.assert_allclose(state, np.array([1.0 + 0.0j, 0.0 + 0.0j]))
    assert len(model.history_) == 5


def test_unfitted_methods_raise() -> None:
    model = ConjugateGradientMAPBayesianRegression(maxiter=4)

    with pytest.raises(RuntimeError, match="Model is not fitted yet"):
        model.get_state()
    with pytest.raises(RuntimeError, match="Model is not fitted yet"):
        model.get_weights()
    with pytest.raises(RuntimeError, match="Model is not fitted yet"):
        model.get_relative_residual()
    with pytest.raises(RuntimeError, match="Model is not fitted yet"):
        model.predict(np.zeros((2, 2)))


def test_predict_input_validation() -> None:
    X, y, m0, Sigma, V = _make_problem(seed=41, n_features=5)
    model = ConjugateGradientMAPBayesianRegression(maxiter=8)
    model.fit(X, y, m0, Sigma, V)

    with pytest.raises(ValueError, match="X must be 2D"):
        model.predict(np.zeros(5))
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((2, 7)))


def test_constructor_and_fit_input_validation_errors() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ConjugateGradientMAPBayesianRegression(maxiter=0)
    with pytest.raises(ValueError, match="preconditioner"):
        ConjugateGradientMAPBayesianRegression(preconditioner="bad")

    model = ConjugateGradientMAPBayesianRegression(maxiter=6)
    X = np.eye(3)
    y = np.array([1.0, 2.0, 3.0])
    m0 = np.zeros(3)
    sigma_diag = np.ones(3)
    V = 0.1

    with pytest.raises(ValueError, match="X must be 2D"):
        model.fit(np.array([1.0, 2.0, 3.0]), y, m0, sigma_diag, V)
    with pytest.raises(ValueError, match="non-empty"):
        model.fit(np.empty((0, 3)), np.array([]), m0, sigma_diag, V)
    with pytest.raises(ValueError, match="y must have length"):
        model.fit(X, y[:2], m0, sigma_diag, V)
    with pytest.raises(ValueError, match="m0 must have length"):
        model.fit(X, y, np.zeros(2), sigma_diag, V)
    with pytest.raises(ValueError, match="Sigma must have length"):
        model.fit(X, y, m0, np.ones(2), V)
    with pytest.raises(ValueError, match="must be positive"):
        model.fit(X, y, m0, np.array([1.0, -1.0, 2.0]), V)
    with pytest.raises(ValueError, match="must be symmetric"):
        model.fit(
            X, y, m0, np.array([[1.0, 2.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), V
        )
    with pytest.raises(ValueError, match="positive definite"):
        model.fit(
            X, y, m0, np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), V
        )
    with pytest.raises(ValueError, match="either a 1D diagonal vector or a 2D matrix"):
        model.fit(X, y, m0, np.ones((3, 3, 1)), V)
    with pytest.raises(ValueError, match="non-negative"):
        model.fit(X, y, m0, sigma_diag, -1.0)

    model_with_init = ConjugateGradientMAPBayesianRegression(maxiter=6, initial_point=np.ones(2))
    with pytest.raises(ValueError, match="initial_point must have length 3"):
        model_with_init.fit(X, y, m0, sigma_diag, V)
