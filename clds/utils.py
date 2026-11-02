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
from typing import Optional
import mpmath
import itertools
import warnings


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
    kappa: jnp.ndarray | float = 1.0,
    period: jnp.ndarray | float = None,
) -> list:
    """
    Regular Fourier Features sample approximation to the squared exponential GP over the
    M-dimensional torus,
        k(u, u') = sigma^2 prod_d exp(-(u_d - u'_d)^2 / (2 (kappa_d period_d)^2)),
    with u_d - u'_d wrapped around period_d. The frequencies are n in {-(N-1), ..., N-1}^M,
    keeping one of each pair +-n, each with a sin and a cos weighted by sqrt(2 S(n)), where S is
    the spectral measure. The constant function is the first one.
    For M=1, this is T1_basis with the sqrt(2) weighting.
    Args:
        N: number of frequencies 0, ..., N-1 for each dimension (so total number of basis functions is (2N-1)**M_conditions)
        M_conditions: number of conditions
        sigma: kernel parameter, the prior std of the GP
        kappa: kernel parameter, the lengthscale in units of the period, scalar or one per dimension
        period: period of the torus, scalar or one per dimension, by default 1. See `initialize_basis` for non-periodic data.
    """
    kappa = np.broadcast_to(np.asarray(kappa, dtype=float), (M_conditions,))
    period = jnp.broadcast_to(
        jnp.asarray(1.0 if period is None else period, dtype=float), (M_conditions,)
    )

    def coef(index):
        # the spectral measure factorizes over dimensions, and sums to sigma^2 over Z^M
        return sigma * np.sqrt(
            np.prod(
                [
                    float(squared_exponential_spectral_measure(n, 1.0, k))
                    for n, k in zip(index, kappa)
                ]
            )
        )

    constant = coef((0,) * M_conditions)
    basis_funcs = [lambda x: constant]  # only one constant function
    for index in itertools.product(range(-(N - 1), N), repeat=M_conditions):
        negative_index = tuple(-n for n in index)
        if index <= negative_index:  # keep half the indicies
            continue
        c, n = np.sqrt(2.0) * coef(index), jnp.array(index, dtype=float)

        def _f_sin(x, c=c, n=n):  # use defaults to avoid late binding
            return c * jnp.sin(2 * jnp.pi * jnp.dot(n, jnp.divide(x, period)))

        def _f_cos(x, c=c, n=n):
            return c * jnp.cos(2 * jnp.pi * jnp.dot(n, jnp.divide(x, period)))

        basis_funcs.append(_f_sin)
        basis_funcs.append(_f_cos)

    assert len(basis_funcs) == (2 * N - 1) ** M_conditions
    return basis_funcs


def initialize_basis(
    conditions: Float[Array, "num_batches num_timesteps M_conditions"],
    lengthscale: jnp.ndarray | float,
    N: int = None,
    sigma: float = 1.0,
    period: list | float = None,
    mask: Float[Array, "num_batches num_timesteps"] = None,
    pad: float = 3.0,
    tol: float = 1e-2,
) -> list:
    """
    Fourier basis of `Tm_basis` for the conditions, with lengthscales in the units of the conditions.
    Args:
        conditions: conditions u, used for the range of the non-periodic dimensions
        lengthscale: lengthscale of the squared exponential kernel, scalar or one per dimension
        N: number of frequencies per dimension, by default the smallest such that the spectral
            measure of the highest frequency is below `tol` relative to the constant one
        sigma: prior std of the GP
        period: period of each dimension, None for a non-periodic one (scalar for all). A
            non-periodic dimension gets the period range(u_d) + pad * lengthscale_d, so that the
            ends of the data are pad lengthscales apart around the torus (correlation exp(-pad^2/2)).
        mask: timesteps of the conditions to use for their range
        pad: padding of the non-periodic dimensions, in lengthscales
        tol: tolerance for the default N
    """
    if conditions.ndim != 3:
        raise ValueError(
            "Unrecognized conditions shape. Pass (num_trials, num_timesteps, M_conditions)."
        )
    M_conditions = conditions.shape[-1]
    u = conditions.reshape(-1, M_conditions)
    if mask is not None:
        u = u[mask.reshape(-1).astype(bool)]
    u_range = u.max(axis=0) - u.min(axis=0)

    lengthscale = jnp.broadcast_to(
        jnp.asarray(lengthscale, dtype=float), (M_conditions,)
    )
    if np.ndim(period) == 0:
        period = [period] * M_conditions
    if len(period) != M_conditions:
        raise ValueError(f"Pass one period per condition dimension ({M_conditions})")
    period = jnp.array(
        [
            u_range[d] + pad * lengthscale[d] if p is None else p
            for d, p in enumerate(period)
        ]
    )

    kappa = lengthscale / period # make kappa in domain units
    if N is None:
        # exp(-2 pi^2 kappa^2 (N-1)^2) < tol for the smallest kappa
        N = int(np.ceil(np.sqrt(np.log(1 / tol) / 2) / (np.pi * float(kappa.min())))) + 1

    return Tm_basis(
        N=N, M_conditions=M_conditions, sigma=sigma, kappa=kappa, period=period
    )

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

def regression_stats_from_moments(
        Phi_x: Optional[Float[Array, "n_steps n_basis_x"]],
        Phi_1: Optional[Float[Array, "n_steps n_basis_1"]],
        ExxT: Optional[Float[Array, "n_steps input_dim input_dim"]] = None,
        Ex: Optional[Float[Array, "n_steps input_dim"]] = None,
        ExyT: Optional[Float[Array, "n_steps input_dim output_dim"]] = None,
        Ey: Optional[Float[Array, "n_steps output_dim"]] = None,
    ) -> tuple:
    """
    Compute the expected sufficient statistics Z^T Z and Z^T Y, summed over the time
    steps t, of the regression
        y_t = W^T z_t + noise,    z_t = [phi_1 x_t ; ... ; phi_L x_t ; Phi_1(u_t)],
    where phi_l = Phi_x(u_t)[l], i.e. the Kronecker block Phi_x(u_t) kron x_t is ordered
    by basis function first, then by state dimension.

    x_t is the regressor (the latent state) and y_t the target (x_{t+1} for the dynamics,
    the observation for the emissions, x_0 for the initial state). Since x_t is latent,
    the statistics are built from its per-step posterior moments: ExxT = E[x_t x_t^T],
    Ex = E[x_t], ExyT = E[x_t y_t^T] and Ey = E[y_t].

    Phi_x and Phi_1 are the bases of the coefficient on x_t and of the bias; a static
    coefficient has the constant basis [1], and a term is dropped when its basis is None
    (in which case its moments are not needed).
    """
    ZTZ, ZTY = [], []
    if Phi_x is not None:
        n_x = Phi_x.shape[1] * ExxT.shape[-1]
        ZTZ.append(
            [jnp.einsum("tk,tl,tij->kilj", Phi_x, Phi_x, ExxT).reshape(n_x, n_x)]
        )
        ZTY.append(jnp.einsum("tk,tim->kim", Phi_x, ExyT).reshape(n_x, -1))
    if Phi_1 is not None:
        ZTZ_11 = Phi_1.T @ Phi_1
        if Phi_x is not None:
            ZTZ_x1 = jnp.einsum("tk,tl,ti->kil", Phi_x, Phi_1, Ex).reshape(n_x, -1)
            ZTZ[0].append(ZTZ_x1)
            ZTZ.append([ZTZ_x1.T, ZTZ_11])
        else:
            ZTZ.append([ZTZ_11])
        ZTY.append(Phi_1.T @ Ey)
    return jnp.block(ZTZ), jnp.concatenate(ZTY, axis=0)

def solve_map_regression(ZTZ, ZTY, has_prior, Sigma):
    r'''
    MAP estimate of W in the regression y = W^T z + N(0, Sigma) given sufficient
    statistics. Takes a N(0, 1) prior on the rows of W where `has_prior` is True
    (GP weights) and a flat prior elsewhere (static coefficients). 
    
    W solves
        ZTZ W + diag(has_prior) W Sigma = ZTY,
    column-wise in the eigenbasis of Sigma = U diag(s) U^T.
        (ZTZ + s_i Lambda)(WU)[:,i] = (ZTY @ U)[:,i]
    '''
    if any(has_prior):          # if there's a GP prior
        Lambda = jnp.diag(jnp.asarray(has_prior, dtype=ZTZ.dtype))
        s, U = jnp.linalg.eigh(Sigma)
        W = jax.vmap(
            lambda s_j, ZTY_j: jax.scipy.linalg.solve(
                ZTZ + s_j * Lambda, 
                ZTY_j, 
                assume_a="pos"
            ),
            in_axes=(0, 1),     # s_j, j-th column of Z^T Y U
            out_axes=1,         # output as columns
        )(s, ZTY @ U) @ U.T
    else:
        # flat prior: ordinary least squares, independent of Sigma
        W = psd_solve(ZTZ, ZTY)
    return W

def residual_covariance(ZTZ, ZTY, YTY, N, W):
    """Expected residual second moment."""
    Sigma = (YTY - W.T @ ZTY - ZTY.T @ W + W.T @ ZTZ @ W) / N
    return Sigma

# %%
def block_hankel(
    y: Float[Array, "n_steps dim"], num_rows: int
) -> Float[Array, "num_rows*dim n_cols"]:
    """
    Block Hankel matrix whose column t stacks y[t], y[t+1], ..., y[t+num_rows-1],
    for t = 0, ..., n_steps - num_rows.
    """
    n_cols = y.shape[0] - num_rows + 1
    return jnp.concatenate([y[i : i + n_cols].T for i in range(num_rows)], axis=0)


def subspace_id(
    ys: Float[Array, "n_batches n_steps emission_dim"],
    state_dim: int,
    num_lags: int,
    mask: Optional[Float[Array, "n_batches n_steps"]] = None,
) -> tuple:
    r"""
    Covariance-based stochastic subspace identification of the autonomous, static LDS
        x_{t+1} = A x_t + noise,    y_t = C x_t + noise,
    from batches of zero-mean emissions.

    Stacking k = `num_lags` past emissions y_p = [y_t; ...; y_{t+k-1}] and future emissions
    y_f = [y_{t+k}; ...; y_{t+2k-1}], their cross-covariance factorizes as
        E[y_f y_p^T] = O K,    O = [C; C A; ...; C A^{k-1}] in \R^{kN x state_dim}
    so its top `state_dim` left singular vectors span the extended observability matrix O.
    C is the first block row of O, and A solves the shift invariance O[:-1] A = O[1:] (in blocks).
    Only the windows of 2k fully observed steps (under `mask`) are used.

    Returns (A, C), in an arbitrary basis of the latent space.
    """
    B, T, N = ys.shape
    k = num_lags
    if mask is None:
        mask = jnp.ones((B, T))

    def hankel(y, m):
        valid = block_hankel(m[:, None], 2 * k).min(axis=0)  # window fully observed
        return block_hankel(jnp.where(m[:, None] > 0, y, 0.0), 2 * k) * valid, valid.sum()

    H, counts = jax.vmap(hankel)(ys, mask)
    past, future = H[:, : k * N], H[:, k * N :] # slice rows per batch

    # For each k, y_f = O_k x_{t+k} + noise[t+k:], so E[y_f y_p^T] = O_k E[x_{t+k} y_p^T]
    cov = jnp.einsum("bic,bjc->ij", future, past) / counts.sum() # empirical E[y_f y_p^T]

    # E[y_f y_p^T] = O K, and we know O has rank state_dim. Form O from SVD.
    U, s, _ = jnp.linalg.svd(cov)
    O = U[:, :state_dim] * jnp.sqrt(s[:state_dim])

    # Read of from O = [C; C A; ...; C A^{k-1}]
    C = O[:N]
    A = jnp.linalg.lstsq(O[:-N], O[N:])[0]
    return A, C


def ppca(
    ys: Float[Array, "n_batches n_steps emission_dim"],
    latent_dim: int,
    mask: Optional[Float[Array, "n_batches n_steps"]] = None,
) -> tuple:
    r"""
    Closed-form probabilistic PCA (Tipping & Bishop, 1999) of the emissions
        y = W x + mu + N(0, sigma^2 I),    x ~ N(0, I),
    with loadings W = U_D (Lambda_D - sigma^2 I)^{1/2} from the top D = `latent_dim`
    eigenpairs of the sample covariance, and sigma^2 the mean of the remaining eigenvalues.
    Only the observed steps (under `mask`) are used.

    Returns (W, mu, sigma^2).
    """
    N = ys.shape[-1]
    y = ys.reshape(-1, N)
    w = jnp.ones(y.shape[0]) if mask is None else mask.reshape(-1).astype(y.dtype)
    y = jnp.where(w[:, None] > 0, y, 0.0)

    mu = w @ y / w.sum()
    yc = (y - mu) * w[:, None]
    evals, evecs = jnp.linalg.eigh(yc.T @ yc / w.sum())
    evals, evecs = evals[::-1], evecs[:, ::-1]  # descending order

    sigma2 = evals[latent_dim:].mean() if latent_dim < N else 0.0
    W = evecs[:, :latent_dim] * jnp.sqrt(jnp.maximum(evals[:latent_dim] - sigma2, 0.0))
    return W, mu, sigma2

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

if __name__ == "__main__":
    M_conditions = 1
    print(Tm_basis(5, 1, 1.0, 1.0))
