r"""Tests for the initialization utilities `subspace_id` and `ppca` on toy LDS / factor
analysis models. Both only identify the latent space up to a change of basis, so the tests
compare basis-invariant quantities (traces of powers of A, W W^T, C C^T) or first align the bases.
"""

from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jxr
import pytest

from clds.utils import initialize_basis, ppca, subspace_id, Tm_basis


def _assert_close(actual, expected, atol=0.0, rtol=0.0):
    """jnp version of np.testing.assert_allclose, reporting the largest violation."""
    actual, expected = jnp.asarray(actual), jnp.asarray(expected)
    assert actual.shape == expected.shape, f"shapes {actual.shape} != {expected.shape}"
    excess = jnp.abs(actual - expected) - (atol + rtol * jnp.abs(expected))
    assert jnp.all(excess <= 0), f"tolerance exceeded by up to {float(excess.max()):.3g}"


def _similarity_invariants(A):
    """tr(A^k) for k = 1..D: invariant to A -> T^{-1} A T, they determine the eigenvalues of A
    (Newton's identities). Used instead of jnp.linalg.eigvals, which only runs on CPU."""
    return jnp.stack(
        [jnp.trace(jnp.linalg.matrix_power(A, k)) for k in range(1, A.shape[0] + 1)]
    )


def _pad(ys, n_valid, fill=1e3):
    """Pad trials after their first `n_valid` steps with garbage, and return the mask."""
    n_batches, n_steps, _ = ys.shape
    mask = jnp.arange(n_steps)[None, :] < n_valid
    return jnp.where(mask[..., None], ys, fill), jnp.broadcast_to(mask, (n_batches, n_steps))


@pytest.fixture(scope="module")
def toy_lds():
    """Zero-mean, stable LDS  x[t+1] = A x[t] + N(0, Q),  y[t] = C x[t] + N(0, R),
    sampled from its stationary regime."""
    key_C, key_x, key_y = jxr.split(jxr.PRNGKey(0), 3)
    D, N, B, T, burn_in = 2, 5, 20, 1000, 200
    theta, rho = 0.3, 0.9
    A = rho * jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
    )
    C = jxr.normal(key_C, (N, D))
    Q, R = 0.1 * jnp.eye(D), 0.1 * jnp.eye(N)

    def step(x, key):
        x_next = x @ A.T + jxr.multivariate_normal(key, jnp.zeros(D), Q, shape=(B,))
        return x_next, x

    _, xs = jax.lax.scan(step, jnp.zeros((B, D)), jxr.split(key_x, T + burn_in))
    xs = jnp.swapaxes(xs, 0, 1)[:, burn_in:]  # (B, T, D)
    ys = xs @ C.T + jxr.multivariate_normal(key_y, jnp.zeros(N), R, shape=(B, T))
    return SimpleNamespace(A=A, C=C, ys=ys, state_dim=D)


@pytest.fixture(scope="module")
def toy_fa():
    """Factor analysis model with isotropic noise (the PPCA model)
        y = W x + mu + N(0, sigma^2 I),    x ~ N(0, I)."""
    key_W, key_mu, key_x, key_y = jxr.split(jxr.PRNGKey(1), 4)
    D, N, B, T = 2, 6, 10, 2000
    W = jxr.normal(key_W, (N, D))
    mu = jxr.normal(key_mu, (N,))
    sigma2 = 0.2
    x = jxr.normal(key_x, (B, T, D))
    ys = x @ W.T + mu + jnp.sqrt(sigma2) * jxr.normal(key_y, (B, T, N))
    return SimpleNamespace(W=W, mu=mu, sigma2=sigma2, ys=ys, latent_dim=D)


# %% subspace identification
def test_subspace_id_recovers_lds_up_to_similarity(toy_lds):
    A_hat, C_hat = subspace_id(toy_lds.ys, toy_lds.state_dim, num_lags=3)
    assert A_hat.shape == toy_lds.A.shape and C_hat.shape == toy_lds.C.shape

    # A is only identified up to similarity: compare similarity invariants
    _assert_close(
        _similarity_invariants(A_hat), _similarity_invariants(toy_lds.A), atol=0.05
    )

    # align the bases with C_hat = C T, then A_hat = T^{-1} A T
    T = jnp.linalg.lstsq(toy_lds.C, C_hat)[0]
    scale = jnp.linalg.norm(C_hat)
    _assert_close(toy_lds.C @ T, C_hat, atol=0.05 * scale)
    _assert_close(jnp.linalg.solve(T, toy_lds.A @ T), A_hat, atol=0.05)


def test_subspace_id_ignores_masked_steps(toy_lds):
    n_valid = 600
    padded, mask = _pad(toy_lds.ys, n_valid)
    A_mask, C_mask = subspace_id(padded, toy_lds.state_dim, 3, mask)
    A_ref, C_ref = subspace_id(toy_lds.ys[:, :n_valid], toy_lds.state_dim, 3)

    # invariant to the sign of each singular vector
    _assert_close(_similarity_invariants(A_mask), _similarity_invariants(A_ref), atol=1e-8)
    _assert_close(C_mask @ C_mask.T, C_ref @ C_ref.T, atol=1e-8)


# %% probabilistic PCA
def test_ppca_is_closed_form_mle(toy_fa):
    """W W^T + sigma^2 I keeps the top-D eigenpairs of the sample covariance
    and replaces the remaining eigenvalues by their mean sigma^2."""
    D = toy_fa.latent_dim
    W, mu, sigma2 = ppca(toy_fa.ys, D)
    N = W.shape[0]
    y = toy_fa.ys.reshape(-1, N)

    _assert_close(mu, y.mean(0), atol=1e-12)
    evals, evecs = jnp.linalg.eigh(jnp.cov(y.T, bias=True))
    evals, evecs = evals[::-1], evecs[:, ::-1]
    _assert_close(sigma2, evals[D:].mean(), rtol=1e-10)

    U = evecs[:, :D]
    expected = U @ jnp.diag(evals[:D]) @ U.T + sigma2 * (jnp.eye(N) - U @ U.T)
    _assert_close(W @ W.T + sigma2 * jnp.eye(N), expected, atol=1e-10)


def test_ppca_recovers_fa_model(toy_fa):
    W, mu, sigma2 = ppca(toy_fa.ys, toy_fa.latent_dim)
    assert W.shape == toy_fa.W.shape

    _assert_close(mu, toy_fa.mu, atol=0.05)
    _assert_close(sigma2, toy_fa.sigma2, rtol=0.05)
    # W is only identified up to a rotation of the latent space
    _assert_close(W @ W.T, toy_fa.W @ toy_fa.W.T, atol=0.1)


def test_ppca_ignores_masked_steps(toy_fa):
    n_valid = 1500
    padded, mask = _pad(toy_fa.ys, n_valid)
    W_mask, mu_mask, s_mask = ppca(padded, toy_fa.latent_dim, mask)
    W_ref, mu_ref, s_ref = ppca(toy_fa.ys[:, :n_valid], toy_fa.latent_dim)

    _assert_close(mu_mask, mu_ref, atol=1e-10)
    _assert_close(s_mask, s_ref, rtol=1e-10)
    _assert_close(W_mask @ W_mask.T, W_ref @ W_ref.T, atol=1e-10)


def test_ppca_full_rank_reproduces_sample_covariance(toy_fa):
    N = toy_fa.ys.shape[-1]
    W, _, sigma2 = ppca(toy_fa.ys, N)
    y = toy_fa.ys.reshape(-1, N)
    assert sigma2 == 0.0
    _assert_close(W @ W.T, jnp.cov(y.T, bias=True), atol=1e-10)


def _implied_kernel(basis, u, v):
    """k(u, v) = sum_l phi_l(u) phi_l(v) of a weight-space GP with N(0, 1) weights."""
    return sum(f(u) * f(v) for f in basis)


def _wrapped_se_kernel(u, v, sigma, lengthscale, period):
    """Squared exponential kernel on the torus, summing over the windings."""
    shifts = jnp.arange(-3, 4)[:, None] * period
    d = (u - v)[None, :] + shifts
    return sigma**2 * jnp.prod(jnp.exp(-(d**2) / (2 * lengthscale**2)).sum(0))


@pytest.mark.parametrize(
    "M_conditions, kappa, period",
    [(1, 0.1, 1.0), (2, 0.1, 1.0), (2, jnp.array([0.1, 0.15]), jnp.array([2 * jnp.pi, 3.0]))],
)
def test_Tm_basis_reproduces_se_kernel(M_conditions, kappa, period):
    sigma = 0.7
    basis = Tm_basis(12, M_conditions=M_conditions, sigma=sigma, kappa=kappa, period=period)
    assert len(basis) == 23**M_conditions
    lengthscale = jnp.broadcast_to(kappa * period, (M_conditions,))
    period = jnp.broadcast_to(period, (M_conditions,))
    u = 0.3 * jnp.ones(M_conditions)
    # include differences of opposite signs, which the kernel must not distinguish
    for d in [0.0, 0.1, -0.1, 0.25]:
        v = u + d * jnp.where(jnp.arange(M_conditions) % 2, -1.0, 1.0)
        _assert_close(
            _implied_kernel(basis, u, v),
            _wrapped_se_kernel(u, v, sigma, lengthscale, period),
            atol=1e-8,
        )


def test_Tm_basis_constant_is_first():
    basis = Tm_basis(4, M_conditions=2, kappa=0.2)
    u = jnp.array([[0.1, 0.2], [0.5, 0.9]])
    phi0 = jax.vmap(basis[0])(u)
    assert phi0.shape == (2,)
    assert phi0[0] == phi0[1]


def test_initialize_basis_periods():
    """A periodic dimension keeps its period, a non-periodic one is uncorrelated across its range."""
    key_hd, key_speed = jxr.split(jxr.PRNGKey(0))
    hd = jxr.uniform(key_hd, (4, 100), maxval=2 * jnp.pi)
    speed = jxr.uniform(key_speed, (4, 100), minval=1.0, maxval=1.5)
    conditions, mask = _pad(jnp.stack([hd, speed], -1), n_valid=80)
    lengthscale = jnp.array([0.5, 0.1])
    basis = initialize_basis(conditions, lengthscale, period=[2 * jnp.pi, None], mask=mask)

    speed_range = speed[:, :80].max() - speed[:, :80].min()
    period = jnp.array([2 * jnp.pi, speed_range + 3 * 0.1])
    lo = jnp.array([0.1, speed[:, :80].min()])
    for v in [lo + jnp.array([-0.2, 0.0]), lo + jnp.array([0.0, 0.1]), lo + jnp.array([0.0, speed_range])]:
        _assert_close(
            _implied_kernel(basis, lo, v),
            _wrapped_se_kernel(lo, v, 1.0, lengthscale, period),
            atol=1e-4,  # tolerance of the default N
        )
    # ends of the speed range are only correlated through the padded wrap
    assert _implied_kernel(basis, lo, lo + jnp.array([0.0, speed_range])) < 0.02


def test_initialize_basis_rejects_bad_shapes():
    with pytest.raises(ValueError):
        initialize_basis(jnp.zeros((4, 100)), 0.1)
    with pytest.raises(ValueError):
        initialize_basis(jnp.zeros((4, 100, 2)), 0.1, period=[1.0, None, None])
