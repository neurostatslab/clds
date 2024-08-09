# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import jax.numpy as jnp
import jax
import numpy as np
from functools import reduce
from jaxtyping import Float, Array

# %%
def random_rotation(key, n, theta):
    rot = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], 
         [jnp.sin(theta),  jnp.cos(theta)]]
    )
    out = jnp.eye(n)
    out = out.at[:2, :2].set(rot)
    q = jnp.linalg.qr(
        jax.random.uniform(key, shape=(n, n))
    )[0]
    
    return q.dot(out).dot(q.T)


# %%
def split_data_cv(data,props,seeds):
    # props: train, validation, test
    # seeds: test, validation
    # data: y (possibly mu, sigma, F, mu_g, sigma_g)

    assert 'train' in props.keys() and 'test' in props.keys() and 'validation' in props.keys()
    assert props['train'] + props['test'] + props['validation'] == 1
    assert 'test' in seeds.keys() and 'validation' in seeds.keys()
    assert 'y' in data.keys()
     
    N,M,D = data['y'].shape
    
    trial_indices = jax.random.permutation(
        jax.random.PRNGKey(seeds['test']),
        np.arange(N)
    )

    test_trials = trial_indices[-int(props['test']*N):]

    train_trials = jax.random.choice(
        jax.random.PRNGKey(seeds['validation']),
        shape=(int(N*props['train']),),
        a=trial_indices[:-int(props['test']*N)],
        replace=False
    ).sort()

    validation_trials = jnp.setdiff1d(trial_indices[:-int(props['test']*N)],train_trials).tolist()

    out = {}
    for k in data.keys():
        out[k+'_train'] = data[k][train_trials,...]
        out[k+'_test'] = data[k][test_trials,...]
        out[k+'_validation'] = data[k][validation_trials,...]
    
    return out



# %%
def get_kernel(params,diag):
    '''Returns the full kernel of multi-dimensional condition spaces
    '''
    if len(params) > 1: 
        kernel = lambda x,y: diag*jnp.all(x==y)+reduce(
            lambda a,b: a*b, [
                _get_kernel(
                    params[i]['type'],
                    params[i])(x[i],y[i]
                ) for i in range(len(params))
            ])
    else: 
        kernel = lambda x,y: diag*(x==y)+_get_kernel(params[0]['type'],params[0])(x,y)
    
    return kernel
        
# %%
def _get_kernel(kernel,params):
    '''Private function, returns the kernel corresponding to a single dimension
    '''
    if kernel == 'periodic': 
        return lambda x,y: params['scale']*jnp.exp(-2*(jnp.sin(jnp.pi*jnp.abs(x-y)/params['normalizer'])**2)/(params['sigma']**2))
    if kernel == 'RBF': 
        return lambda x,y: params['scale']*jnp.exp(-(jnp.linalg.norm((x-y)/params['normalizer'])**2)/(2*params['sigma']**2))

# %%
def lgssm_filter(
        m0: Float[Array, "latent_dim"],                     # initial mean
        S0: Float[Array, "latent_dim latent_dim"],          # initial covariance
        As: Float[Array, "ntime-1 latent_dim latent_dim"],    # dynamics matrices
        Q : Float[Array, "latent_dim latent_dim"],          # dynamics covariance
        bs: Float[Array, "ntime-1 latent_dim"],               # dynamics biases
        Cs: Float[Array, "ntime emission_dim latent_dim"],  # emission matrices
        R : Float[Array, "emission_dim emission_dim"],      # emission covariance
        ys: Float[Array, "ntime emission_dim"],
    ):
    r"""Run a Kalman filter to produce the marginal likelihood and filtered state estimates.
    Copied and adapted from dynamax.
    """
    num_timesteps = len(ys)

    def _log_likelihood(pred_mean, pred_cov, H, R, y):
        m = H @ pred_mean
        S = R + H @ pred_cov @ H.T
        # return MVN(m, S).log_prob(y)
        return logprob_analytic(y, m, S)
    
    def _condition_on(m, P, H, R, y): # update step
        S = R + H @ P @ H.T
        K = psd_solve(S, H @ P).T           # eq. (56)
        Sigma_cond = P - K @ S @ K.T        # eq. (58)
        mu_cond = m + K @ (y - H @ m)       # eq. (57)
        return mu_cond, symmetrize(Sigma_cond)

    def _predict(m, S, F, b, Q):
        mu_pred = F @ m + b                 # eq. (54), returns mu_{t*}
        Sigma_pred = F @ S @ F.T + Q        # eq. (55), returns V_{t*}
        return mu_pred, Sigma_pred

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Shorthand: get parameters and inputs for time index t
        y = ys[t]

        # Update the log likelihood
        ll += _log_likelihood(pred_mean, pred_cov, Cs[t], R, y)

        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(pred_mean, pred_cov, Cs[t], R, y)

        # Predict the next state
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, As[t], bs[t], Q)
        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the Kalman filter
    carry = (0.0, m0, S0) # mu_{0*}, V_{0*}
    (ll, _, _), (filtered_means, filtered_covs) = jax.lax.scan(_step, carry, jnp.arange(num_timesteps))
    return ll, filtered_means, filtered_covs

# %%
def lgssm_smoother(
        m0: Float[Array, "latent_dim"],                     # initial mean
        S0: Float[Array, "latent_dim latent_dim"],          # initial covariance
        As: Float[Array, "ntime latent_dim latent_dim"],    # dynamics matrices
        Q : Float[Array, "latent_dim latent_dim"],          # dynamics covariance
        bs: Float[Array, "ntime latent_dim"],               # dynamics biases
        Cs: Float[Array, "ntime emission_dim latent_dim"],  # emission matrices
        R : Float[Array, "emission_dim emission_dim"],      # emission covariance
        ys: Float[Array, "ntime emission_dim"],             # emissions
    ):
    num_timesteps = len(ys)

    # Run the Kalman filter
    ll, filtered_means, filtered_covs = lgssm_filter(
        m0=m0, S0=S0, As=As, Q=Q, bs=bs, Cs=Cs, R=R, ys=ys
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
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - As[t] @ filtered_mean - bs[t])
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - As[t] @ filtered_cov @ As[t].T - Q) @ G.T

        # Compute the smoothed expectation of z_t z_{t+1}^T
        smoothed_cross = G @ smoothed_cov_next + jnp.outer(smoothed_mean, smoothed_mean_next)

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov, smoothed_cross)

    # Run the Kalman smoother
    init_carry = (filtered_means[-1], filtered_covs[-1])
    args = (jnp.arange(num_timesteps - 2, -1, -1), filtered_means[:-1][::-1], filtered_covs[:-1][::-1])
    _, (smoothed_means, smoothed_covs, smoothed_cross) = jax.lax.scan(_step, init_carry, args)

    # Reverse the arrays and return
    smoothed_means = jnp.vstack((smoothed_means[::-1], filtered_means[-1][None, ...]))
    smoothed_covs = jnp.vstack((smoothed_covs[::-1], filtered_covs[-1][None, ...]))
    smoothed_cross = smoothed_cross[::-1]
    
    filter_results = (filtered_means, filtered_covs)
    smoother_results = (smoothed_means, smoothed_covs, smoothed_cross)
    return ll, filter_results, smoother_results

# %%
def T1_basis(N: int, sigma: float=1.0, kappa: float=1.0, period: float=1.0) -> list:
    '''Returns 4*N basis functions for a torus T1 manifold'''
    def weight_space_coefficients(m):
        return sigma * jnp.sqrt(jnp.exp(- 2* jnp.pi**2 * kappa**2 * m**2))
    
    basis_funcs = []
    for n in jnp.arange(-N, N):
        def _f_sin(x, n=n): # make sure to add n=n to avoid late binding
            return weight_space_coefficients(n) * jnp.sin(n * 2*jnp.pi * x / period)
        def _f_cos(x, n=n):
            return weight_space_coefficients(n) * jnp.cos(n * 2*jnp.pi * x / period)
        basis_funcs.append(_f_sin)
        basis_funcs.append(_f_cos)

    assert len(basis_funcs) == 4*N, len(basis_funcs)
    return basis_funcs

def T2_basis(N: int, sigma: float=1.0, kappa: float=1.0, period1: float=1.0, period2=None) -> list:
    '''Returns basis functions for T2 manifold'''
    def weight_space_coefficients(m,n):
        return sigma * jnp.sqrt(jnp.exp(- 2* jnp.pi**2 * kappa**2 * (m**2+n**2)))
    if period2 is None:
        period2 = period1

    basis_funcs = []
    for n in jnp.arange(-N, N):
        for m in jnp.arange(-N, N):
            def _f_sin(x, m=m, n=n): # defaults to avoid late binding
                return weight_space_coefficients(m,n) * jnp.sin(2*jnp.pi * (m * x[0] / period1 + n * x[1] / period2))
            def _f_cos(x, m=m, n=n):
                return weight_space_coefficients(m,n) * jnp.cos(2*jnp.pi * (m * x[0] / period1 + n * x[1] / period2))
            basis_funcs.append(_f_sin)
            basis_funcs.append(_f_cos)
    return basis_funcs

# %%
def logprob_analytic(
        x: Float[Array, "N"], mu: Float[Array, "N"], cov: Float[Array, "N N"]
        ) -> float:
    '''
    Analytic log probability of a multivariate normal distribution. Handles singular covariance matrices.
    '''
    N = x.shape[0]
    z = jnp.linalg.solve(cov, x - mu) # solve the linear system cov * z = x - mu
    _, logdetC = jnp.linalg.slogdet(cov + 1e-06 * jnp.eye(N))
    logprob = -(jnp.dot(x - mu, z) + logdetC + N * jnp.log(2.0 * jnp.pi)) / 2.0
    return logprob
# %%

def symmetrize(A):
    """Symmetrize one or more matrices."""
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))

def psd_solve(A, b, diagonal_boost=1e-9):
    """A wrapper for coordinating the linalg solvers used in the library for psd matrices."""
    A = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1])
    L, lower = jax.scipy.linalg.cho_factor(A, lower=True)
    x = jax.scipy.linalg.cho_solve((L, lower), b)
    return x