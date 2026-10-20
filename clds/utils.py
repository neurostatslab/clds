# %%
# -*- coding: utf-8 -*-
"""
@author: Amin, Victor
"""

import jax.numpy as jnp

import nemos as nmo
import jax
import numpy as np
from functools import reduce
from jaxtyping import Float, Array
import mpmath
import itertools


# %%
def random_rotation(key, n, theta):
    rot = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
    )
    out = jnp.eye(n)
    out = out.at[:2, :2].set(rot)
    q = jnp.linalg.qr(jax.random.uniform(key, shape=(n, n)))[0]

    return q.dot(out).dot(q.T)


# %%
def split_data_cv(data, props, seeds):
    # props: train, validation, test
    # seeds: test, validation
    # data: y (possibly mu, sigma, F, mu_g, sigma_g)

    assert (
        "train" in props.keys()
        and "test" in props.keys()
        and "validation" in props.keys()
    )
    assert props["train"] + props["test"] + props["validation"] == 1
    assert "test" in seeds.keys() and "validation" in seeds.keys()
    assert "y" in data.keys()

    N, M, D = data["y"].shape

    trial_indices = jax.random.permutation(
        jax.random.PRNGKey(seeds["test"]), np.arange(N)
    )

    test_trials = trial_indices[-int(props["test"] * N) :]

    train_trials = jax.random.choice(
        jax.random.PRNGKey(seeds["validation"]),
        shape=(int(N * props["train"]),),
        a=trial_indices[: -int(props["test"] * N)],
        replace=False,
    ).sort()

    validation_trials = jnp.setdiff1d(
        trial_indices[: -int(props["test"] * N)], train_trials
    ).tolist()

    out = {}
    for k in data.keys():
        out[k + "_train"] = data[k][train_trials, ...]
        out[k + "_test"] = data[k][test_trials, ...]
        out[k + "_validation"] = data[k][validation_trials, ...]

    return out


# %%
def get_kernel(params, diag):
    """Returns the full kernel of multi-dimensional condition spaces"""
    if len(params) > 1:
        kernel = lambda x, y: diag * jnp.all(x == y) + reduce(
            lambda a, b: a * b,
            [
                _get_kernel(params[i]["type"], params[i])(x[i], y[i])
                for i in range(len(params))
            ],
        )
    else:
        kernel = lambda x, y: diag * (x == y) + _get_kernel(
            params[0]["type"], params[0]
        )(x, y)

    return kernel


# %%
def _get_kernel(kernel, params):
    """Private function, returns the kernel corresponding to a single dimension"""
    if kernel == "periodic":
        return lambda x, y: params["scale"] * jnp.exp(
            -2
            * (jnp.sin(jnp.pi * jnp.abs(x - y) / params["normalizer"]) ** 2)
            / (params["sigma"] ** 2)
        )
    if kernel == "RBF":
        return lambda x, y: params["scale"] * jnp.exp(
            -(jnp.linalg.norm((x - y) / params["normalizer"]) ** 2)
            / (2 * params["sigma"] ** 2)
        )


# %%
def lgssm_filter(
    m0: Float[Array, "latent_dim"],  # initial mean
    S0: Float[Array, "latent_dim latent_dim"],  # initial covariance
    As: Float[Array, "ntime-1 latent_dim latent_dim"],  # dynamics matrices
    Q: Float[Array, "latent_dim latent_dim"],  # dynamics covariance
    bs: Float[Array, "ntime-1 latent_dim"],  # dynamics biases
    Cs: Float[Array, "ntime emission_dim latent_dim"],  # emission matrices
    R: Float[Array, "emission_dim emission_dim"],  # emission covariance
    ys: Float[Array, "ntime emission_dim"],
    ds: Float[Array, "ntime emission_dim"] = None,  # emissions biases
):
    r"""Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
    Copied and adapted from dynamax.
    """
    num_timesteps = len(ys)

    def _log_likelihood(pred_mean, pred_cov, H, d, R, y):
        m = H @ pred_mean + d  # ADD D HERE?
        S = R + H @ pred_cov @ H.T
        # return MVN(m, S).log_prob(y)
        return logprob_analytic(y, m, S)

    def _condition_on(m, P, H, d, R, y):  # update step
        S = R + H @ P @ H.T
        K = psd_solve(S, H @ P).T  # eq. (56)
        Sigma_cond = P - K @ S @ K.T  # eq. (58)
        mu_cond = m + K @ (y - H @ m - d)  # eq. (57) ADD D HERE?
        return mu_cond, symmetrize(Sigma_cond)

    def _predict(m, S, F, b, Q):
        mu_pred = F @ m + b  # eq. (54), returns mu_{t*}
        Sigma_pred = F @ S @ F.T + Q  # eq. (55), returns V_{t*}
        return mu_pred, Sigma_pred

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Shorthand: get parameters and inputs for time index t
        y = ys[t]

        # Update the log likelihood
        if ds is None:
            ll += _log_likelihood(pred_mean, pred_cov, Cs[t], 0, R, y)
        else:
            ll += _log_likelihood(pred_mean, pred_cov, Cs[t], ds[t], R, y)

        # Condition on this emission
        if ds is None:
            filtered_mean, filtered_cov = _condition_on(
                pred_mean, pred_cov, Cs[t], 0, R, y
            )
        else:
            filtered_mean, filtered_cov = _condition_on(
                pred_mean, pred_cov, Cs[t], ds[t], R, y
            )

        # Predict the next state
        # pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, As[t], bs[t], Q)
        pred_mean, pred_cov = jax.lax.cond(
            t < num_timesteps - 1,
            lambda _: _predict(
                filtered_mean, filtered_cov, As[t], bs[t], Q
            ),
            lambda _: (filtered_mean, filtered_cov), # handles last step
            operand=None,
        )
        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the Kalman filter
    carry = (0.0, m0, S0)  # mu_{0*}, V_{0*}
    (ll, _, _), (filtered_means, filtered_covs) = jax.lax.scan(
        _step, carry, jnp.arange(num_timesteps)
    )
    return ll, filtered_means, filtered_covs


# %%
def lgssm_smoother(
    m0: Float[Array, "latent_dim"],  # initial mean
    S0: Float[Array, "latent_dim latent_dim"],  # initial covariance
    As: Float[Array, "ntime latent_dim latent_dim"],  # dynamics matrices
    Q: Float[Array, "latent_dim latent_dim"],  # dynamics covariance
    bs: Float[Array, "ntime latent_dim"],  # dynamics biases
    Cs: Float[Array, "ntime emission_dim latent_dim"],  # emission matrices
    R: Float[Array, "emission_dim emission_dim"],  # emission covariance
    ys: Float[Array, "ntime emission_dim"],  # emissions
    ds: Float[Array, "ntime emission_dim"] = None,  # emissions biases
):
    num_timesteps = len(ys)

    # Run the Kalman filter
    ll, filtered_means, filtered_covs = lgssm_filter(
        m0=m0, S0=S0, As=As, Q=Q, bs=bs, Cs=Cs, R=R, ds=ds, ys=ys
    )

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # This is like the Kalman gain but in reverse
        # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
        G = psd_solve(Q + As[t] @ filtered_cov @ As[t].T, As[t] @ filtered_cov).T

        # Compute the smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (
            smoothed_mean_next - As[t] @ filtered_mean - bs[t]
        )
        smoothed_cov = (
            filtered_cov
            + G @ (smoothed_cov_next - As[t] @ filtered_cov @ As[t].T - Q) @ G.T
        )

        # Compute the smoothed expectation of z_t z_{t+1}^T
        smoothed_cross = G @ smoothed_cov_next + jnp.outer(
            smoothed_mean, smoothed_mean_next
        )

        return (smoothed_mean, smoothed_cov), (
            smoothed_mean,
            smoothed_cov,
            smoothed_cross,
        )

    # Run the Kalman smoother
    init_carry = (filtered_means[-1], filtered_covs[-1])
    args = (
        jnp.arange(num_timesteps - 2, -1, -1),
        filtered_means[:-1][::-1],
        filtered_covs[:-1][::-1],
    )
    _, (smoothed_means, smoothed_covs, smoothed_cross) = jax.lax.scan(
        _step, init_carry, args
    )

    # Reverse the arrays and return
    smoothed_means = jnp.vstack((smoothed_means[::-1], filtered_means[-1][None, ...]))
    smoothed_covs = jnp.vstack((smoothed_covs[::-1], filtered_covs[-1][None, ...]))
    smoothed_cross = smoothed_cross[::-1]

    filter_results = (filtered_means, filtered_covs)
    smoother_results = (smoothed_means, smoothed_covs, smoothed_cross)
    return ll, filter_results, smoother_results


# %%
def squared_exponential_spectral_measure(m, sigma, kappa):
    C_inf = float(mpmath.jtheta(3, 0.0, mpmath.exp(-2 * mpmath.pi**2 * kappa**2)))
    return (sigma**2 / C_inf) * jnp.exp(-2 * jnp.pi**2 * kappa**2 * m**2)


def T1_basis(
    N: int, sigma: float = 1.0, kappa: float = 1.0, period: float = 1.0
) -> list:
    """
    Regular Fourier Features sample approximation to GP over T1.
    https://arxiv.org/pdf/2006.10160, equation (13)
    https://jmlr.org/papers/volume18/16-579/16-579.pdf, eq 17
    """

    def coef(m):
        return jnp.sqrt(squared_exponential_spectral_measure(m, sigma, kappa))

    constant_func = lambda x: coef(0) * jnp.ones_like(x)
    basis_funcs = [constant_func]
    for n in jnp.arange(1, N):

        def _f_sin(x, n=n):  # make sure to add n=n to avoid late binding
            return coef(n) * jnp.sin(n * 2 * jnp.pi * x / period)

        def _f_cos(x, n=n):
            return coef(n) * jnp.cos(n * 2 * jnp.pi * x / period)

        basis_funcs.append(_f_sin)
        basis_funcs.append(_f_cos)

    assert len(basis_funcs) == 2 * (N - 1) + 1, len(basis_funcs)
    return basis_funcs


# def T2_basis(N: int, sigma: float=1.0, kappa: float=1.0, period1: float=1.0, period2=None) -> list:
#     '''Returns basis functions for T2 manifold'''
#     def weight_space_coefficients(m,n):
#         return sigma * jnp.sqrt(jnp.exp(- 2* jnp.pi**2 * kappa**2 * (m**2+n**2)))
#     if period2 is None:
#         period2 = period1

#     basis_funcs = []
#     for n in jnp.arange(-N, N):
#         for m in jnp.arange(-N, N):
#             def _f_sin(x, m=m, n=n): # defaults to avoid late binding
#                 return weight_space_coefficients(m,n) * jnp.sin(2*jnp.pi * (m * x[0] / period1 + n * x[1] / period2))
#             def _f_cos(x, m=m, n=n):
#                 return weight_space_coefficients(m,n) * jnp.cos(2*jnp.pi * (m * x[0] / period1 + n * x[1] / period2))
#             basis_funcs.append(_f_sin)
#             basis_funcs.append(_f_cos)
#     return basis_funcs


class TorusBasis:
    def __init__(
        self,
        frequencies: int,
        ndim: int,
        bounds: tuple = None,
        sigma: float = 1.0,
        kappa: float = 1.0,
    ):
        self.basis = nmo.basis.FourierEval(
            frequencies=frequencies, ndim=ndim, bounds=bounds, frequency_mask="all"
        )
        self.sigma = sigma
        self.kappa = kappa
        coef = jnp.sqrt(
            TorusBasis.squared_exponential_spectral_measure(
                jnp.linalg.norm(self.basis._freq_combinations, axis=0), sigma, kappa
            )
        )
        self.coef = jnp.hstack((coef, coef[1:]))

    @staticmethod
    def squared_exponential_spectral_measure(m, sigma, kappa):
        C_inf = float(mpmath.jtheta(3, 0.0, mpmath.exp(-2 * mpmath.pi**2 * kappa**2)))
        return (sigma**2 / C_inf) * jnp.exp(-2 * jnp.pi**2 * kappa**2 * m**2)

    def evaluate(self, u):
        return self.basis.evaluate(*u.T) * self.coef


def Tm_basis(
    N: int,
    M_conditions: int = 1,
    sigma: float = 1.0,
    kappa: float = 1.0,
    period: jnp.ndarray | float = None,
) -> list:
    """
    Regular Fourier Features sample approximation to GP over the M-dimensional torus.
    For M=1, this is equivalent to T1_basis.
    Args:
        N: number of basis functions for each dimension (so total number of basis functions is 2*(N**M_conditions - 1) + 1)
        M_conditions: number of conditions
        sigma: kernel parameter
        kappa: kernel parameter
        period: period of the torus. Provide `period >= data_interval + 6 * kappa` for non-periodic data.
    """

    def coef(index_array):
        return jnp.sqrt(
            squared_exponential_spectral_measure(
                jnp.linalg.norm(index_array), sigma, kappa
            )
        )

    if period is None:
        period = jnp.ones(M_conditions)
    if isinstance(period, float):
        period = period * jnp.ones(M_conditions)

    basis_funcs = []
    for index in itertools.product(jnp.arange(N), repeat=M_conditions):
        if index == (0,) * M_conditions:
            constant_func = lambda x: coef(jnp.zeros(M_conditions))
            basis_funcs.append(constant_func)  # only one constant function
        else:

            def _f_sin(x, index=index):  # use defaults to avoid late binding
                return coef(jnp.array(index)) * jnp.sin(
                    2 * jnp.pi * jnp.dot(jnp.array(index), jnp.divide(x, period))
                )

            def _f_cos(x, index=index):
                return coef(jnp.array(index)) * jnp.cos(
                    2 * jnp.pi * jnp.dot(jnp.array(index), jnp.divide(x, period))
                )

            basis_funcs.append(_f_sin)
            basis_funcs.append(_f_cos)

    assert len(basis_funcs) == 2 * (N**M_conditions - 1) + 1
    return basis_funcs


# # %%
# def logprob_analytic(
#     x: Float[Array, "N"], mu: Float[Array, "N"], cov: Float[Array, "N N"]
# ) -> float:
#     """
#     Analytic log probability of a multivariate normal distribution. Handles singular covariance matrices.
#     """
#     N = x.shape[0]
#     z = jnp.linalg.solve(cov, x - mu)  # solve the linear system cov * z = x - mu
#     _, logdetC = jnp.linalg.slogdet(cov + 1e-06 * jnp.eye(N))
#     logprob = -(jnp.dot(x - mu, z) + logdetC + N * jnp.log(2.0 * jnp.pi)) / 2.0
#     return logprob

def logprob_analytic(x, mu, cov):
    """Gaussian log density for a positive-definite covariance."""
    cov = symmetrize(cov)
    L = jnp.linalg.cholesky(cov)
    z = jax.scipy.linalg.solve_triangular(
        L, x - mu, lower=True
    )
    logdet = 2.0 * jnp.log(jnp.diag(L)).sum()
    return -0.5 * (
        jnp.dot(z, z)
        + logdet
        + x.shape[0] * jnp.log(2.0 * jnp.pi)
    )


# %%


def symmetrize(A):
    """Symmetrize one or more matrices."""
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def psd_solve(A, b, diagonal_boost=0.0):
    """A wrapper for coordinating the linalg solvers used in the library for psd matrices."""
    A = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1])
    L, lower = jax.scipy.linalg.cho_factor(A, lower=True)
    x = jax.scipy.linalg.cho_solve((L, lower), b)
    return x


def safe_wrap(X):
    return jnp.where(jnp.isclose(X, 0.0), 0.0, X)


# %%
def jax_solve_sylvester_BS(A, B, C, **kwargs):
    """
    Solve the Sylvester equation AX + XB = C for X using the Bartels-Stewart algorithm.
    # WARNING: only implemented for A, B with real eigenvalues.
    """

    def solve_triangular_system(R, S, F):
        """
        Solve the triangular system RY + YS = C for Y using forward substitution on the blocks.
        """
        n = R.shape[0]
        m = S.shape[0]
        Y = jnp.zeros((n, m), dtype=C.dtype)
        for k in range(m):
            Y_k = jax.scipy.linalg.solve(
                R + S[k, k] * jnp.eye(n, dtype=R.dtype),
                F[:, k] - Y[:, :k] @ S[:k, k], #F[:, k] - Y[:, k + 1 :] @ S[k + 1 :, k]
            )
            Y = Y.at[:, k].set(Y_k)
        return Y

    # Compute the Schur decompositions of A and B
    R, U = jax.scipy.linalg.schur(A)
    S, V = jax.scipy.linalg.schur(B)

    # Transform C into the Schur basis
    F = U.T @ C @ V

    # Solve the triangular system
    Y = solve_triangular_system(R, S, F)

    # Transform the solution back to the original basis
    X = U @ Y @ V.T
    return X

def solve_sylvester_symmetric(A, B, C):
    # Appropriate when A and B are theoretically symmetric.
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)

    a, U = jnp.linalg.eigh(A)
    b, V = jnp.linalg.eigh(B)

    F = U.T @ C @ V
    Y = F / (a[:, None] + b[None, :])
    return U @ Y @ V.T

def test_sylvester():
    key = jax.random.PRNGKey(0)
    dim_1, dim_2 = 3, 2
    subkeys = jax.random.split(key, 3)
    A = jax.random.normal(subkeys[0], (dim_1, dim_1))
    B = jax.random.normal(subkeys[1], (dim_2, dim_2))

    # Make A and B have real eigenvalues only
    A = A + A.T
    B = B + B.T

    X = jax.random.normal(subkeys[2], (dim_1, dim_2))
    C = A @ X + X @ B

    X_hat = jax_solve_sylvester_BS(A, B, C)
    assert jnp.allclose(X, X_hat, atol=1e-5), X - X_hat

import math
from functools import partial

import jax
import jax.numpy as jnp


def test_dynamics_update(
    model, params, emissions, conditions, mask,
    atol=1e-3, rtol=1e-6,
):
    # One E-step; freeze these statistics for the comparison.
    batch_stats, lls_old = jax.vmap(partial(model.e_step, params))(
        emissions, conditions, mask
    )
    G, _, H, YTY, N = jax.tree_util.tree_map(
        lambda x: jnp.sum(x, axis=0),
        batch_stats[3],  # dynamics_gp_stats
    )

    def pack(p):
        weights = p.dynamics_weights
        W = weights.transpose(0, 2, 1).reshape(-1, weights.shape[1])
        if model.use_dynamics_bias:
            W = jnp.concatenate([W, p.dynamics_bias[..., 0]], axis=0)
        return W

    def dynamics_objective(p):
        W, Q = pack(p), p.dynamics_cov
        residual_sum = YTY - W.T @ H - H.T @ W + W.T @ G @ W
        L = jnp.linalg.cholesky(Q)
        logdet = 2 * jnp.log(jnp.diag(L)).sum()
        return -0.5 * (
            N * logdet
            + jnp.trace(jnp.linalg.solve(Q, residual_sum))
            + jnp.sum(W**2)
        )

    # Compute the full proposal, but retain only its dynamics updates.
    proposal = model.m_step(params, batch_stats)
    params_dyn = type(params)(
        initial_mean=params.initial_mean,
        initial_cov=params.initial_cov,
        dynamics_weights=proposal.dynamics_weights,
        dynamics_bias=proposal.dynamics_bias,
        dynamics_cov=params.dynamics_cov,  # Keep Q fixed for this test.
        emissions_weights=params.emissions_weights,
        emissions_bias=params.emissions_bias,
        emissions_cov=params.emissions_cov,
    )

    # Recompute the actual likelihood at the dynamics-only proposal.
    _, lls_dyn = jax.vmap(partial(model.e_step, params_dyn))(
        emissions, conditions, mask
    )

    lp_old = float(lls_old.sum() + model.log_prior(params))
    lp_dyn = float(lls_dyn.sum() + model.log_prior(params_dyn))
    f_old = float(dynamics_objective(params))
    f_dyn = float(dynamics_objective(params_dyn))

    assert bool(jnp.array_equal(
        params_dyn.dynamics_cov, params.dynamics_cov
    ))

    W0 = pack(params)
    dW = pack(params_dyn) - W0
    Q0 = params.dynamics_cov

    # Derivative predicted by the frozen posterior statistics.
    grad_likelihood = jnp.linalg.solve(
        Q0, (H - G @ W0).T
    ).T

    expected_likelihood_slope = jnp.sum(grad_likelihood * dW)
    expected_prior_slope = -jnp.sum(W0 * dW)

    # Differentiate the actual likelihood and prior independently.
    def actual_terms(p):
        _, trial_lls = jax.vmap(partial(model.e_step, p))(
            emissions, conditions, mask
        )
        return jnp.stack([
            trial_lls.sum(),
            model.log_prior(p),
        ])

    # Only dynamics weights/bias should differ between these parameters.
    direction = jax.tree_util.tree_map(
        lambda old, new: new - old,
        params,
        params_dyn,
    )

    _, actual_slopes = jax.jvp(
        actual_terms,
        (params,),
        (direction,),
    )

    print(
        "Likelihood slope: "
        f"actual={float(actual_slopes[0]):.12f}, "
        f"from moments={float(expected_likelihood_slope):.12f}, "
        f"gap={float(actual_slopes[0] - expected_likelihood_slope):.3e}"
    )
    print(
        "Prior slope:      "
        f"actual={float(actual_slopes[1]):.12f}, "
        f"from weights={float(expected_prior_slope):.12f}, "
        f"gap={float(actual_slopes[1] - expected_prior_slope):.3e}"
    )

    assert all(math.isfinite(v) for v in [lp_old, lp_dyn, f_old, f_dyn])

    delta_lp = lp_dyn - lp_old
    delta_f = f_dyn - f_old
    gap = delta_lp - delta_f

    # Separate tolerances for the two comparisons.
    tol_f = atol + rtol * max(abs(f_old), abs(f_dyn))
    tol_bound = tol_f + atol + rtol * max(abs(lp_old), abs(lp_dyn))

    print(f"Log posterior: {lp_old:.6f} -> {lp_dyn:.6f}")
    print(f"Dynamics surrogate improvement: {delta_f:.6f}")
    print(f"Actual MAP improvement:         {delta_lp:.6f}")
    print(f"EM bound gap (should be >= 0):  {gap:.6f}")
    print(f"Bound tolerance:               {tol_bound:.6f}")

    assert delta_f >= -tol_f, "Dynamics surrogate decreased."
    assert gap >= -tol_bound, (
        "EM bound failed: actual MAP improvement is smaller "
        "than the dynamics surrogate improvement."
    )

    return params_dyn, {
        "delta_log_posterior": delta_lp,
        "delta_surrogate": delta_f,
        "bound_gap": gap,
    }

def diagnose_em_blocks(model, params, emissions, conditions, mask=None):
    axes = (0, 0, None if mask is None else 0)

    batch_stats, lls_old = jax.vmap(
        lambda y, u, m: model.e_step(params, y, u, m),
        in_axes=axes,
    )(emissions, conditions, mask)

    proposal = model.m_step(params, batch_stats)

    ll_old = lls_old.sum()
    prior_old = model.log_prior(params)
    n_trials = emissions.shape[0]

    groups = {
        "dynamics": (
            "dynamics_weights", "dynamics_bias", "dynamics_cov",
        ),
        "emissions": (
            "emissions_weights", "emissions_bias", "emissions_cov",
        ),
        "initial": (
            "initial_mean", "initial_cov",
        ),
    }
    all_fields = tuple(k for fields in groups.values() for k in fields)
    groups["all"] = all_fields

    for name, changed_fields in groups.items():
        candidate = type(params)(**{
            k: getattr(proposal if k in changed_fields else params, k)
            for k in all_fields
        })

        lls_new = jax.vmap(
            lambda y, u, m: model.smoother(candidate, y, u, m)[0],
            in_axes=axes,
        )(emissions, conditions, mask)

        delta_ll = lls_new.sum() - ll_old
        delta_prior = model.log_prior(candidate) - prior_old

        print(
            f"{name:>9}: "
            f"Δlik={float(delta_ll):+.9f}, "
            f"Δprior={float(delta_prior):+.9f}, "
            f"ΔMAP={float(delta_ll + delta_prior):+.9f}, "
            f"Δold_logging={float(delta_ll + n_trials * delta_prior):+.9f}"
        )

    return proposal

if __name__ == "__main__":
    M_conditions = 1
    print(Tm_basis(5, 1, 1.0, 1.0))
