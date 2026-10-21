r"""Regression tests for the CLDS E/M-step. This file tests the E/M math.

Run with JAX_ENABLE_X64=1 as 
```
JAX_ENABLE_X64=1 pytest tests/test_clds_EM.py
```

ASSUMPTIONS
-----------
* x[t+1] = A(u[t]) x[t] + b(u[t]) + noise: SOURCE-time indexing.
* smoothed_cross_covariances means E[x[t] x[t+1].T | y], not centered covariance.
* Shared, independent N(0,1) priors on enabled GP weights.
* No Q/R/S0 prior.
"""

from dataclasses import is_dataclass, replace
from importlib import import_module
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest


LOCAL_UTILS_MODULE = "clds.utils" 
ATOL = 5e-8
RTOL = 5e-10
BLOCKS = {
    "dynamics": ("dynamics_weights", "dynamics_bias", "dynamics_cov"),
    "emissions": ("emissions_weights", "emissions_bias", "emissions_cov"),
    "initial": ("initial_mean", "initial_cov"),
}
FLAGS = {
    "dynamics": "use_dynamics_prior",
    "emissions": "use_emissions_prior",
    "initial": "use_initial_prior",
}


def _close(actual, expected, message=""):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.all(np.isfinite(actual)), f"{message}: nonfinite actual value"
    assert np.all(np.isfinite(expected)), f"{message}: nonfinite reference value"
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL,
                               err_msg=message)


def _nonnegative(value, *scales, message):
    value = float(value)
    tol = ATOL + RTOL * max(1.0, *(abs(float(x)) for x in scales))
    assert np.isfinite(value) and value >= -tol, (
        f"{message}: difference={value:+.12g}, tolerance={tol:.3g}"
    )


def _replace(params, **updates):
    if is_dataclass(params):
        return replace(params, **updates)
    if hasattr(params, "_replace"):
        return params._replace(**updates)
    fields = {k: getattr(params, k) for names in BLOCKS.values() for k in names}
    return type(params)(**(fields | updates))


def _candidate(old, new, block, weights_only=False):
    '''update parameter set for one block with new values, potentially omitting the covariance [-1]'''
    fields = BLOCKS[block][:-1] if weights_only else BLOCKS[block]
    return _replace(old, **{k: getattr(new, k) for k in fields})


def _time_array(array, steps, shape):
    if array is None:
        return np.zeros((steps,) + shape)
    array = np.asarray(array)
    if array.shape == shape:
        return np.broadcast_to(array, (steps,) + shape)
    assert array.shape[1:] == shape and len(array) >= steps, array.shape
    return array[:steps]


def _logpdf_numpy(y, mean, cov):
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, y - mean)
    return -0.5 * (z @ z + 2 * np.log(np.diag(L)).sum()
                   + len(y) * np.log(2 * np.pi))


def _dense_posterior(*, m0, S0, As, Q, bs, Cs, R, ys, ds=None):
    r"""
    Compute posterior moments and MLL for a linear gaussian ssm, 
    without usinghte Kalman filter/smoother.
    """
    # Ensure formatting
    ys, m0, S0, Q, R = map(np.asarray, (ys, m0, S0, Q, R))
    T, E = ys.shape
    D = m0.size
    As = _time_array(As, T - 1, (D, D))
    bs = _time_array(bs, T - 1, (D,))
    Cs = _time_array(Cs, T, (E, D))
    ds = _time_array(ds, T, (E,))

    # Forward loop
    mean = np.zeros((T, D))
    mean[0] = m0
    transform = np.zeros((T * D, T * D))
    noise = np.zeros_like(transform)
    observation = np.zeros((T * E, T * D))
    for t in range(T):
        ix = slice(t * D, (t + 1) * D)
        if t:
            previous = slice((t - 1) * D, t * D)
            mean[t] = As[t - 1] @ mean[t - 1] + bs[t - 1]
            transform[ix] = As[t - 1] @ transform[previous]
        transform[ix, ix] = np.eye(D)
        noise[ix, ix] = S0 if t == 0 else Q
        observation[t * E:(t + 1) * E, ix] = Cs[t]

    # Marginals
    latent_cov = transform @ noise @ transform.T
    obs_mean = observation @ mean.ravel() + ds.ravel()
    obs_cov = observation @ latent_cov @ observation.T + np.kron(np.eye(T), R)

    # Posterior
    gain = np.linalg.solve(obs_cov, observation @ latent_cov).T
    posterior_mean = (mean.ravel() + gain @ (ys.ravel() - obs_mean)).reshape(T, D) # E[x_t|Y]
    posterior_cov = latent_cov - gain @ observation @ latent_cov
    covs = np.stack([posterior_cov[t * D:(t + 1) * D, t * D:(t + 1) * D]        # Cov(x_t|Y)
                     for t in range(T)])
    cross = np.stack([                                                          # E[x_t x_t^T|Y], not centered
        posterior_cov[t * D:(t + 1) * D, (t + 1) * D:(t + 2) * D]
        + np.outer(posterior_mean[t], posterior_mean[t + 1])
        for t in range(T - 1)
    ])
    return _logpdf_numpy(ys.ravel(), obs_mean, obs_cov), (posterior_mean, covs, cross)


@pytest.fixture(scope="module")
def local_utils():
    return import_module(LOCAL_UTILS_MODULE)


@pytest.fixture(scope="module")
def tiny_lgssm():
    rng = np.random.default_rng(41)
    T, D, E = 7, 2, 3
    t = np.arange(T - 1)
    As = np.stack([np.array([[0.55 + 0.2 * np.sin(k), 0.12],
                            [-0.16, 0.35 + 0.17 * np.cos(k)]]) for k in t])
    C = np.array([[1.0, 0.3], [-0.2, 0.8], [0.7, -0.6]])
    return {
        k: jnp.asarray(v, dtype=jnp.float64) 
        for k, v in dict(
                m0=np.array([0.5, -0.4]), S0=np.array([[1.8, 0.25], [0.25, 0.6]]),
                As=As, Q=np.array([[0.45, 0.12], [0.12, 1.1]]),
                bs=np.column_stack([0.3 * np.cos(t), 0.2 * np.sin(2 * t)]),
                Cs=np.stack([C + 0.08 * rng.normal(size=(E, D)) for _ in range(T)]),
                R=np.array([[0.6, 0.08, 0.02], [0.08, 0.8, -0.05], [0.02, -0.05, 0.5]]),
                ds=0.2 * rng.normal(size=(T, E)), ys=rng.normal(size=(T, E)),
            ).items()
            }


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_local_filter_and_smoother_match_dense_gaussian(local_utils, tiny_lgssm, compiled):
    filter_fn = jax.jit(local_utils.lgssm_filter) if compiled else local_utils.lgssm_filter
    smooth_fn = jax.jit(local_utils.lgssm_smoother) if compiled else local_utils.lgssm_smoother
    ref_ll, ref_moments = _dense_posterior(**tiny_lgssm)
    filter_ll, _, _ = filter_fn(**tiny_lgssm)
    ll, _, moments = smooth_fn(**tiny_lgssm)
    _close(filter_ll, ref_ll, "Filter likelihood; check source-time indexing and logpdf")
    _close(ll, ref_ll, "Smoother likelihood")
    for name, actual, expected in zip(("mean", "covariance", "lag moment"), moments, ref_moments):
        _close(actual, expected, f"Smoother {name}")


@pytest.mark.parametrize("scale", [1.0, 1e-4])
def test_logprob_uses_one_covariance_for_quad_and_logdet(local_utils, scale):
    cov = scale * np.array([[0.6, 0.2], [0.2, 1.4]])
    mean = np.array([0.3, -0.5])
    y = mean + np.sqrt(scale) * np.array([0.4, -0.1])
    actual = local_utils.logprob_analytic(jnp.asarray(y), jnp.asarray(mean), jnp.asarray(cov))
    _close(actual, _logpdf_numpy(y, mean, cov), "Gaussian density; check hidden jitter")


def test_sylvester_residual_for_psd_gram_and_full_covariance(local_utils):
    rng = np.random.default_rng(7)
    features = rng.normal(size=(12, 6))
    G = features.T @ features
    Q = np.array([[0.7, 0.25], [0.25, 2.4]])
    truth = rng.normal(size=(6, 2))
    H = G @ truth + truth @ Q
    W = local_utils.jax_solve_sylvester_BS(*map(jnp.asarray, (G, Q, H)))
    _close(W, truth, "Sylvester solution")
    _close(G @ np.asarray(W) + np.asarray(W) @ Q, H, "Sylvester residual")

def test_sylvester_symmetric_residual_for_psd_gram_and_full_covariance(local_utils):
    rng = np.random.default_rng(8)
    features = rng.normal(size=(12, 6))
    G = features.T @ features
    Q = np.array([[0.7, 0.25], [0.25, 2.4]]) + np.square(np.diag(rng.normal(size=(2))))
    truth = rng.normal(size=(6, 2))
    H = G @ truth + truth @ Q
    W = local_utils.solve_sylvester_symmetric(*map(jnp.asarray, (G, Q, H)))
    _close(W, truth, "Sylvester solution")
    _close(G @ np.asarray(W) + np.asarray(W) @ Q, H, "Sylvester residual")


@pytest.fixture(scope="module")
def fitting_case(clds_case):
    """The only model-construction adapter is the external clds_case fixture."""
    model = clds_case["model"]

    def as64(x):
        a = jnp.asarray(x)
        return a.astype(jnp.float64) if jnp.issubdtype(a.dtype, jnp.floating) else a

    params = jax.tree_util.tree_map(as64, clds_case["params"])
    D, E = params.dynamics_cov.shape[0], params.emissions_cov.shape[0]
    fixed = frozenset(clds_case.get("fixed_covariances", ()))
    assert fixed <= BLOCKS.keys(), "fixed_covariances must name parameter blocks"
    assert D >= 2 and E >= 2, "Use at least two dimensions to exercise matrix/bias layout"
    assert model.use_dynamics_prior and model.use_initial_prior, (
        "Enable dynamics and initial GP priors to cover the regressions we diagnosed"
    )
    assert model.use_dynamics_bias, "Enable dynamics bias to test the joint A,b blocks"
    assert not np.allclose(params.initial_cov, np.eye(D)), (
        "Use nonidentity initial_cov: identity hides the old initial-mean ridge bug"
    )
    if "dynamics" not in fixed:
        assert not np.allclose(params.dynamics_cov, np.eye(D)), (
            "Use nonidentity dynamics_cov to exercise covariance-weighted regression"
        )
    for block, fields in BLOCKS.items():
        cov = np.asarray(getattr(params, fields[-1]))
        _close(cov, cov.T, f"Fixture {block} covariance symmetry")
        assert np.linalg.eigvalsh(cov).min() > 0, f"Fixture {block} covariance must be PD"
    rng = np.random.default_rng(2026)
    B, T, M = 3, 8, clds_case.get("condition_dim", 1)
    u = rng.uniform(-1.1, 1.1, size=(B, T, M))
    y = rng.normal(size=(B, T, E)) + 0.25 * np.sin(np.arange(T))[None, :, None]
    for block, prior_name in (("dynamics", "dynamics"), ("initial", "init")):
        phi = np.asarray(model.priors[prior_name].evaluate_basis(jnp.asarray(u[0])))
        assert phi.shape[1] >= 2 and not np.allclose(phi[1:], phi[:-1]), (
            f"Use a varying {block} basis; constant conditions hide indexing bugs"
        )
    return SimpleNamespace(model=model, params=params, y=jnp.asarray(y),
                           u=jnp.asarray(u), fixed=fixed)


def _batch_e(fitting_case, params):
    return jax.vmap(lambda y, u: fitting_case.model.e_step(params, y, u, None))(
        fitting_case.y, fitting_case.u
    )


def _loglik(fitting_case, params):
    return fitting_case.model.marginal_log_lik(
        params,
        fitting_case.y,
        fitting_case.u,
    )

def _logposterior(fitting_case, params):
    return _loglik(fitting_case, params) + fitting_case.model.log_prior(params)


@pytest.fixture(
    scope="module",
    params=[0, 3],
    ids=["initial", "after_three_steps"],
)
def step(fitting_case, request):
    """Prepare one E/M update for the diagnostics.

    Test both at initialization and after three warm-up EM iterations.
    Return the old parameters, their E-step statistics, and the proposed
    M-step parameters. All candidate updates are evaluated against this
    same fixed E-step posterior.
    """
    n_warmup_steps = request.param
    params = fitting_case.params

    # Move to the parameter setting at which we want to test an update.
    for _ in range(n_warmup_steps):
        batch_stats, _ = _batch_e(fitting_case, params)
        params = fitting_case.model.m_step(params, batch_stats)

    # E-step: compute each trial's statistics under the OLD parameters.
    batch_stats, trial_logliks = _batch_e(fitting_case, params)

    # Aggregate across trials for the diagnostic calculations.
    stats = jax.tree_util.tree_map(
        lambda x: x.sum(axis=0),
        batch_stats,
    )

    # M-step: propose new parameters using this fixed posterior.
    # Pass unaggregated stats because m_step performs its own batch sum.
    proposal = fitting_case.model.m_step(params, batch_stats)

    return SimpleNamespace(
        old=params,
        new=proposal,
        batch_stats=batch_stats,
        stats=stats,
        old_ll=trial_logliks.sum(),
    )


def _regression(model, stats, block):
    """Return ZTZ, ZTY, YTY, N from batch-summed E-step statistics. 

    W is the regression coefficient matrix, including bias rows when enabled.
    All statistics are expectations under the fixed E-step posterior.
    
    To make clear the notation, these stats are used to update [W, Σ] with:
        (1) ZTZ W + W Σ = ZTY (solving gives W_new,  +WΣ is due to the N(0,1) prior on W)
        (2) Σ_new = (YTY - W_new.T @ ZTY - ZTY.T @ W_new + W_new.T @ ZTZ @ W_new) / N
    """
    if block == "initial":
        if model.use_initial_prior:
            # (ZTZ_m0, ZTY_m0, Ex0x0T, N0)
            return stats[1]

        sum_x0, YTY, N = stats[0]
        ZTZ = jnp.asarray(N).reshape(1, 1)
        ZTY = sum_x0[None, :]
    else:
        static_index, gp_index = {
            "dynamics": (2, 3),
            "emissions": (4, 5),
        }[block]

        index = gp_index if getattr(model, FLAGS[block]) else static_index
        ZTZ, ZTY, YTY, N = stats[index]
    return ZTZ, ZTY, YTY, N


def _pack(model, params, block):
    '''Concatenate [W_A, W_b] if needed'''
    if block == "initial":
        m = params.initial_mean
        return m[..., 0] if model.use_initial_prior else m.reshape(1, -1)
    weight_name, bias_name, _ = BLOCKS[block]
    w, bias = getattr(params, weight_name), getattr(params, bias_name)
    if not getattr(model, f"use_{block}_bias"):
        bias = None
    if getattr(model, FLAGS[block]):
        W = w.transpose(0, 2, 1).reshape(-1, w.shape[1])
        b = None if bias is None else bias[..., 0]
    else:
        W = w.T
        b = None if bias is None else bias.reshape(1, -1)
    return W if b is None else jnp.concatenate((W, b), axis=0)


def _regression_terms(fitting_case, params, stats, block):
    """Return the block's expected complete-data log likelihood and log prior.

    Evaluate candidate parameters using fixed E-step sufficient statistics.
    Parameter-independent constants are omitted (other terms in the ECLL). 
    The sum is the block's MAP-EM surrogate objective.
    """
    ZTZ, ZTY, YTY, N = _regression(fitting_case.model, stats, block)
    
    W = _pack(fitting_case.model, params, block)
    cov = getattr(params, BLOCKS[block][-1])
    L = jnp.linalg.cholesky(cov)

    # Objectives
    residual_second_moment = YTY - W.T @ ZTY - ZTY.T @ W + W.T @ ZTZ @ W
    expected_loglik = -0.5 * (2 * N * jnp.log(jnp.diag(L)).sum()
                 + jnp.trace(jnp.linalg.solve(cov, residual_second_moment)))
    log_prior = -0.5 * jnp.sum(W ** 2) if getattr(fitting_case.model, FLAGS[block]) else jnp.asarray(0.0)
    return jnp.stack((expected_loglik, log_prior))


def _evaluated(fitting_case, params, i):
    A, C, b, d, m0 = fitting_case.model.weights_to_params(params, fitting_case.u[i])
    return dict(m0=m0, S0=params.initial_cov, As=A, Q=params.dynamics_cov,
                bs=b, Cs=C, R=params.emissions_cov, ds=d, ys=fitting_case.y[i])


def test_model_smoother_matches_source_time_dense_reference(fitting_case, step):
    for i in range(len(fitting_case.y)):
        ref_ll, reference = _dense_posterior(**_evaluated(fitting_case, step.old, i))
        ll, _, moments = fitting_case.model.smoother(step.old, fitting_case.y[i], fitting_case.u[i], None)
        _close(ll, ref_ll, "Model wrapper likelihood; A(u[t]) must govern t -> t+1")
        for actual, expected in zip(moments, reference):
            _close(actual, expected, "Model wrapper posterior moments")


def _design(phi, mean, with_bias):
    """
    Important for the [weight, bias] combinations.
    Explicit z=[phi kron x; phi]; ordering is (basis, state)."""
    J = np.kron(np.asarray(phi)[:, None], np.eye(len(mean)))
    z = J @ mean
    if with_bias:
        z = np.concatenate((z, phi))
        J = np.vstack((J, np.zeros((len(phi), len(mean)))))
    return z, J


def _reference_stats(fitting_case, params, block):
    """Build ZTZ, ZTY, YTY, N with explicit per-time features and dense posteriors."""
    model = fitting_case.model
    result = None
    for trial in range(len(fitting_case.y)):
        _, (mu, cov, cross) = _dense_posterior(**_evaluated(fitting_case, params, trial))
        y, u = np.asarray(fitting_case.y[trial]), fitting_case.u[trial]
        T, D = mu.shape
        if block == "initial":
            phi = (np.asarray(model.priors["init"].evaluate_basis(u[:1]))[0]
                   if model.use_initial_prior else np.ones(1))
            terms = [(np.outer(phi, phi), np.outer(phi, mu[0]),
                      cov[0] + np.outer(mu[0], mu[0]), 1)]
        else:
            is_dyn = block == "dynamics"
            n = T - 1 if is_dyn else T
            phi = (np.asarray(model.priors[block].evaluate_basis(u[:n]))
                   if getattr(model, FLAGS[block]) else np.ones((n, 1)))
            with_bias = getattr(model, f"use_{block}_bias")
            terms = []
            for t in range(n):
                z, J = _design(phi[t], mu[t], with_bias)
                ZTZ = np.outer(z, z) + J @ cov[t] @ J.T
                if is_dyn:
                    lag_cov = cross[t] - np.outer(mu[t], mu[t + 1])
                    ZTY = np.outer(z, mu[t + 1]) + J @ lag_cov
                    YTY = cov[t + 1] + np.outer(mu[t + 1], mu[t + 1])
                else:
                    ZTY, YTY = np.outer(z, y[t]), np.outer(y[t], y[t])
                terms.append((ZTZ, ZTY, YTY, 1))
        for term in terms:
            result = tuple(np.array(x, copy=True) for x in term) if result is None else tuple(
                a + b for a, b in zip(result, term)
            )
    return result


@pytest.mark.parametrize("block", BLOCKS)
def test_e_step_statistics_match_explicit_features(fitting_case, step, block):
    actual = _regression(fitting_case.model, step.stats, block)
    expected = _reference_stats(fitting_case, step.old, block)
    for name, a, b in zip(("G", "H", "YTY", "N"), actual, expected):
        _close(a, b, f"{block} {name}: check basis ordering, bias blocks, counts, lag moments")


@pytest.mark.parametrize("block", BLOCKS)
def test_likelihood_and_prior_scores_match_fixed_moments(fitting_case, step, block):
    rng = np.random.default_rng(29)
    directions = {
        k: jnp.asarray(rng.normal(size=getattr(step.old, k).shape))
        for k in BLOCKS[block][:-1] if getattr(step.old, k) is not None
    }

    def at(alpha):
        return _replace(step.old, **{k: getattr(step.old, k) + alpha * v
                                     for k, v in directions.items()})

    def actual(alpha):
        p = at(alpha)
        return jnp.stack((_loglik(fitting_case, p), fitting_case.model.log_prior(p)))

    def from_moments(alpha):
        # step.stats is held fixed; do NOT differentiate through another E-step.
        return _regression_terms(fitting_case, at(alpha), step.stats, block)

    zero, one = jnp.asarray(0.0), jnp.asarray(1.0)
    _, actual_score = jax.jvp(actual, (zero,), (one,))
    _, expected_score = jax.jvp(from_moments, (zero,), (one,))
    _close(actual_score[0], expected_score[0], f"{block} likelihood score/Fisher identity")
    _close(actual_score[1], expected_score[1], f"{block} prior score; shared prior counted once")


@pytest.mark.parametrize("block", BLOCKS)
def test_m_step_normal_equation_and_covariance(fitting_case, step, block):
    ZTZ, ZTY, YTY, N = _regression(fitting_case.model, step.stats, block)
    W = _pack(fitting_case.model, step.new, block)
    old_cov = getattr(step.old, BLOCKS[block][-1])
    new_cov = getattr(step.new, BLOCKS[block][-1])
    lhs = ZTZ @ W
    if getattr(fitting_case.model, FLAGS[block]):
        lhs = lhs + W @ old_cov
    _close(lhs, ZTY, f"{block} normal equation: use OLD covariance, not batch-summed covariance")
    _close(new_cov, new_cov.T, f"{block} covariance symmetry")
    assert np.linalg.eigvalsh(np.asarray(new_cov)).min() > 0, f"{block} covariance is not PD"
    if block in fitting_case.fixed:
        _close(new_cov, old_cov, f"{block} covariance was meant to stay fixed")
    else:
        residual = YTY - W.T @ ZTY - ZTY.T @ W + W.T @ ZTZ @ W
        _close(new_cov, residual / N, f"{block} covariance: include mean/bias and all cross terms")


@pytest.mark.parametrize("block", BLOCKS)
def test_m_step_increases_fixed_posterior_surrogate(fitting_case, step, block):
    weight_candidate = _candidate(step.old, step.new, block, weights_only=True)
    f_old = _regression_terms(fitting_case, step.old, step.stats, block).sum()
    f_weights = _regression_terms(fitting_case, weight_candidate, step.stats, block).sum()
    f_cov = _regression_terms(fitting_case, step.new, step.stats, block).sum()
    _nonnegative(f_weights - f_old, f_weights, f_old, message=f"{block}: F(weights)-F(old)")
    _nonnegative(f_cov - f_weights, f_cov, f_weights, message=f"{block}: F(cov)-F(weights)")


@pytest.mark.parametrize("block", [*BLOCKS, "all"])
def test_single_step_map_improvement_and_em_bound(fitting_case, step, block):
    candidate = step.new if block == "all" else _candidate(step.old, step.new, block)
    selected = tuple(BLOCKS) if block == "all" else (block,)
    f_old = sum(_regression_terms(fitting_case, step.old, step.stats, b).sum() for b in selected)
    f_new = sum(_regression_terms(fitting_case, candidate, step.stats, b).sum() for b in selected)
    lp_old = step.old_ll + fitting_case.model.log_prior(step.old)
    lp_new = _logposterior(fitting_case, candidate)
    delta_f, delta_lp = f_new - f_old, lp_new - lp_old
    gap = delta_lp - delta_f
    print(f"{block}: surrogate={float(delta_f):+.12g}, MAP={float(delta_lp):+.12g}, "
          f"bound_gap={float(gap):+.12g}")
    _nonnegative(delta_f, f_new, f_old, message=f"{block}: surrogate decreased")
    _nonnegative(delta_lp, lp_new, lp_old, message=f"{block}: MAP decreased")
    _nonnegative(gap, lp_old, lp_new, f_old, f_new, message=f"{block}: EM bound violated")


def test_padding_does_not_change_valid_likelihood_moments_or_stats(fitting_case):
    # Only a valid prefix followed by padding; internal missingness is a different model.
    model, params = fitting_case.model, fitting_case.params
    y, u = fitting_case.y[0, :6], fitting_case.u[0, :6]
    n, pad = len(y), 3
    padded_y = jnp.concatenate((y, jnp.full((pad, y.shape[1]), 7.0)))
    padded_u = jnp.concatenate((u, jnp.full((pad, u.shape[1]), -0.7)))
    mask = jnp.arange(n + pad) < n
    ll_ref, _, moments_ref = model.smoother(params, y, u, None)
    ll_pad, _, moments_pad = model.smoother(params, padded_y, padded_u, mask)
    _close(ll_pad, ll_ref, "Padding contributes zero likelihood, not log N(0;0,R)")
    for i, (a, b) in enumerate(zip(moments_pad, moments_ref)):
        _close(a[:n - 1 if i == 2 else n], b, "Padding changed valid posterior moments")
    stats_ref, _ = model.e_step(params, y, u, None)
    stats_pad, _ = model.e_step(params, padded_y, padded_u, mask)
    assert jax.tree_util.tree_structure(stats_pad) == jax.tree_util.tree_structure(stats_ref)
    for i, (a, b) in enumerate(zip(jax.tree_util.tree_leaves(stats_pad),
                                  jax.tree_util.tree_leaves(stats_ref))):
        _close(a, b, f"Padding leaked into sufficient-statistic leaf {i}")
