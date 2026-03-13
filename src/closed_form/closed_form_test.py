from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the sibling module importable when running pytest from repo root.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from closed_form import (
    ClosedFormBR,
    ClosedFormMAPBayesianRegression,
    closed_form_map_solution,
)


def _make_problem(
    *,
    seed: int = 7,
    n_samples: int = 24,
    n_features: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.normal(size=n_features)
    y = X @ true_w + 0.01 * rng.normal(size=n_samples)
    m0 = rng.normal(size=n_features)
    A = rng.normal(size=(n_features, n_features))
    Sigma = A.T @ A + 0.5 * np.eye(n_features)
    V = 0.2
    return X, y, m0, Sigma, V


def test_fit_matches_closed_form_with_full_covariance() -> None:
    X, y, m0, Sigma, V = _make_problem()
    model = ClosedFormMAPBayesianRegression()

    state = model.fit(X, y, m0, Sigma, V)

    precision = np.linalg.solve(Sigma, np.eye(Sigma.shape[0]))
    expected_w = np.linalg.solve(
        X.T @ X + V * precision,
        X.T @ y + V * (precision @ m0),
    )

    np.testing.assert_allclose(model.get_weights(), expected_w, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(state, model.get_state())
    assert state.shape == (8,)
    assert np.isclose(np.linalg.norm(state), 1.0)
    assert model.num_qubits_ == 3
    assert len(model.history_) == 1


def test_fit_matches_closed_form_with_diagonal_covariance_vector() -> None:
    X, y, m0, _, V = _make_problem(n_features=5)
    sigma_diag = np.full(5, 1.7)
    model = ClosedFormMAPBayesianRegression()

    model.fit(X, y, m0, sigma_diag, V)

    precision = np.diag(1.0 / sigma_diag)
    expected_w = np.linalg.solve(
        X.T @ X + V * precision,
        X.T @ y + V * (precision @ m0),
    )
    np.testing.assert_allclose(model.get_weights(), expected_w, rtol=1e-10, atol=1e-10)


def test_helper_solution_matches_model_weights() -> None:
    X, y, m0, Sigma, V = _make_problem(seed=11)
    model = ClosedFormMAPBayesianRegression()
    model.fit(X, y, m0, Sigma, V)

    helper_w = closed_form_map_solution(X, y, m0, Sigma, V)
    np.testing.assert_allclose(helper_w, model.get_weights(), rtol=1e-12, atol=1e-12)


def test_predict_matches_matrix_vector_product() -> None:
    X, y, m0, Sigma, V = _make_problem(seed=12)
    model = ClosedFormMAPBayesianRegression()
    model.fit(X, y, m0, Sigma, V)

    pred = model.predict(X)
    np.testing.assert_allclose(pred, X @ model.get_weights(), rtol=1e-12, atol=1e-12)
    assert pred.shape == (X.shape[0],)


def test_zero_map_vector_returns_basis_state() -> None:
    X = np.eye(3, dtype=float)
    y = np.zeros(3, dtype=float)
    m0 = np.zeros(3, dtype=float)
    Sigma = np.ones(3, dtype=float)
    V = 1.0

    model = ClosedFormMAPBayesianRegression()
    state = model.fit(X, y, m0, Sigma, V)

    np.testing.assert_allclose(model.get_weights(), np.zeros(3))
    assert state.shape == (4,)
    np.testing.assert_allclose(state, np.array([1.0 + 0.0j, 0.0j, 0.0j, 0.0j]))


def test_lstsq_fallback_for_singular_system() -> None:
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])
    m0 = np.zeros(2)
    Sigma = np.ones(2)
    V = 0.0

    model = ClosedFormMAPBayesianRegression(eps=1e-12)
    model.fit(X, y, m0, Sigma, V)

    gram = X.T @ X
    rhs = X.T @ y
    expected_w = np.linalg.lstsq(gram, rhs, rcond=1e-12)[0]
    np.testing.assert_allclose(model.get_weights(), expected_w, rtol=1e-10, atol=1e-10)


def test_alias_closed_form_br_works() -> None:
    X, y, m0, Sigma, V = _make_problem(seed=17)
    model = ClosedFormBR()
    state = model.fit(X, y, m0, Sigma, V)
    assert np.isclose(np.linalg.norm(state), 1.0)
    assert model.get_weights().shape == (X.shape[1],)


def test_unfitted_getters_and_predict_raise() -> None:
    model = ClosedFormMAPBayesianRegression()

    with pytest.raises(RuntimeError, match="Model is not fitted yet"):
        model.get_state()
    with pytest.raises(RuntimeError, match="Model is not fitted yet"):
        model.get_weights()
    with pytest.raises(RuntimeError, match="Model is not fitted yet"):
        model.predict(np.zeros((2, 2)))


def test_predict_input_validation() -> None:
    X, y, m0, Sigma, V = _make_problem(seed=13)
    model = ClosedFormMAPBayesianRegression()
    model.fit(X, y, m0, Sigma, V)

    with pytest.raises(ValueError, match="X must be 2D"):
        model.predict(np.zeros(3))
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((4, X.shape[1] + 1)))


def test_fit_input_validation_errors() -> None:
    model = ClosedFormMAPBayesianRegression()

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
        model.fit(X, y, m0, np.array([[1.0, 2.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), V)
    with pytest.raises(ValueError, match="positive definite"):
        model.fit(X, y, m0, np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), V)
    with pytest.raises(ValueError, match="either a 1D diagonal vector or a 2D matrix"):
        model.fit(X, y, m0, np.ones((3, 3, 1)), V)
    with pytest.raises(ValueError, match="non-negative"):
        model.fit(X, y, m0, sigma_diag, -1.0)
