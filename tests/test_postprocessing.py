r"""Tests for the regularized fixed points in clds/postprocessing.py.

Run with JAX_ENABLE_X64=1 as
```
JAX_ENABLE_X64=1 pytest tests/test_postprocessing.py
```
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from clds.postprocessing import (
    identifiability_transform,
    get_fixed_points,
    expected_latent_given_u,
    get_regularized_fixed_points,
)


@pytest.fixture(scope="module")
def postprocessing_case(clds_case):
    """Add random batched data and a condition grid to the clds_case fixture."""
    model = clds_case["model"]
    params = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, jnp.float64), clds_case["params"]
    )
    B, T, M = 3, 40, clds_case["condition_dim"]
    u_key, y_key = jax.random.split(jax.random.key(1))
    conditions = jax.random.uniform(u_key, (B, T, M), maxval=2 * jnp.pi)
    emissions = jax.random.normal(y_key, (B, T, model.emission_dim))
    conditions_range = jnp.linspace(0.0, 2 * jnp.pi, 25)[:, None]
    return model, params, emissions, conditions, conditions_range


def test_expected_latent_given_u_solves_ridge_regression(postprocessing_case):
    """Weights are the ridge (unit prior) regression of smoothed means on the basis."""
    model, params, emissions, conditions, _ = postprocessing_case
    E_x_given_u, weights = expected_latent_given_u(model, params, emissions, conditions)

    Z, X = [], []
    for Y, U in zip(emissions, conditions):
        _, _, (m, _, _) = model.smoother(params, emissions=Y, inputs=U)
        Z.append(E_x_given_u.evaluate_basis(U))
        X.append(m)
    Z, X = np.concatenate(Z), np.concatenate(X)
    L = Z.shape[1]
    # ridge as least squares on the augmented system [Z; I] w = [X; 0]
    expected, *_ = np.linalg.lstsq(
        np.vstack([Z, np.eye(L)]), np.vstack([X, np.zeros((L, X.shape[1]))]), rcond=None
    )

    assert weights.shape == (L, model.state_dim, 1)
    np.testing.assert_allclose(weights[..., 0], expected, rtol=1e-8, atol=1e-10)


def test_expected_latent_given_u_ignores_padding(postprocessing_case):
    """Masked fit on garbage-padded trials (leading and trailing) matches the unpadded fit."""
    model, params, emissions, conditions, _ = postprocessing_case
    _, weights = expected_latent_given_u(model, params, emissions, conditions)

    B, T, _ = emissions.shape
    leads, trails = [0, 3, 5], [7, 4, 2]  # same padded length T + 7
    pad = lambda x, i, key: jnp.concatenate([
        1e3 * jax.random.normal(key, (leads[i],) + x.shape[1:]),
        x,
        1e3 * jax.random.normal(jax.random.fold_in(key, 1), (trails[i],) + x.shape[1:]),
    ])
    keys = jax.random.split(jax.random.key(2), B)
    padded_emissions = jnp.stack([pad(emissions[i], i, keys[i]) for i in range(B)])
    padded_conditions = jnp.stack([pad(conditions[i], i, keys[i]) for i in range(B)])
    mask = jnp.stack([
        jnp.arange(T + 7) >= leads[i] for i in range(B)
    ]) & jnp.stack([jnp.arange(T + 7) < leads[i] + T for i in range(B)])

    _, padded_weights = expected_latent_given_u(
        model, params, padded_emissions, padded_conditions, mask=mask
    )
    np.testing.assert_allclose(padded_weights, weights, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("ROTATE", [False, True])
def test_regularized_fixed_points_limits(postprocessing_case, ROTATE):
    """tau -> 0 recovers the exact fixed points; tau -> inf recovers the anchors."""
    model, params, emissions, conditions, conditions_range = postprocessing_case
    args = (model, params, emissions, conditions, conditions_range)

    points, _ = get_regularized_fixed_points(*args, ROTATE=ROTATE, tau=1e-6)
    exact = get_fixed_points(model, params, conditions_range, ROTATE=ROTATE)
    np.testing.assert_allclose(points, exact, rtol=1e-6, atol=1e-8)

    points, anchors = get_regularized_fixed_points(*args, ROTATE=ROTATE, tau=1e6)
    np.testing.assert_allclose(points, anchors, rtol=1e-6, atol=1e-8)


def test_regularized_fixed_points_anchors_follow_rotation(postprocessing_case):
    """With ROTATE, anchors must be expressed in the same (rotated) coordinates as A, b."""
    model, params, emissions, conditions, conditions_range = postprocessing_case
    args = (model, params, emissions, conditions, conditions_range)
    H, _ = identifiability_transform(params)

    _, anchors = get_regularized_fixed_points(*args, ROTATE=False)
    _, anchors_rotated = get_regularized_fixed_points(*args, ROTATE=True)
    np.testing.assert_allclose(anchors_rotated, anchors @ H.T, rtol=1e-8, atol=1e-10)
