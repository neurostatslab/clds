# -*- coding: utf-8 -*-
"""
@author: Amin
"""
import jax.numpy as jnp
import jax.random as jxr

from jax.example_libraries import optimizers

from jax import jit, value_and_grad, vmap
from tqdm.auto import trange

from .params import ParamsCLDS, ParamsGP, ParamsBasis

from .models import CLDS
from functools import partial
from jaxtyping import Array, Float
from typing import Optional, NamedTuple


# %%
class Recognition:
    def __init__(self, params: NamedTuple, **args):
        self.params = params

    def __call__(self, params: ParamsBasis, ts: Float[Array, "T M"]):
        raise NotImplementedError


class Delta(Recognition):
    def __call__(self, params: ParamsGP, ts: Float[Array, "T M"]):
        return params


class Basis(Recognition):
    def __init__(self, params: ParamsBasis, wgps: dict):
        self.wgps = wgps
        self.params = params

    def __call__(self, params: ParamsBasis, ts: Float[Array, "T M"]):
        latents = ParamsGP(
            As=self.wgps["A"](params.A_weights, ts[1:]),
            bs=self.wgps["b"](params.b_weights, ts),
            Ls=self.wgps["L"](params.L_weights, ts),
        )

        return latents


# %%
def fit_em(
    model: CLDS,
    # params: ParamsCLDS,
    emissions: Float[Array, "num_batches num_timesteps emission_dim"],
    conditions: Optional[Float[Array, "num_batches num_timesteps input_dim"]] = None,
    num_iters: int = 50,
):
    """
    Requires the model to have the e_step and m_step functions implemented
    """
    assert (
        emissions.ndim == 3
    ), "emissions should be 3D, of shape (num_batches, num_timesteps, emission_dim)"

    if model.initial_params is None:
        model.initial_params = model.initialize_params(emissions.shape[1])

    params = model.initial_params

    @jit
    def em_step(params):
        # Obtain current E-step stats and model log prob
        batch_stats, lls = vmap(partial(model.e_step, params))(emissions, conditions)
        log_priors = vmap(partial(model.log_prior, params))(conditions)
        mll = lls.sum()
        lp = log_priors.sum() + mll

        # Update with M-step
        params = model.m_step(params, batch_stats)

        return params, (lp, mll)

    log_probs, marginal_log_liks = [], []

    pbar = trange(num_iters)
    pbar.set_description("jit compiling ...")

    for i in pbar:
        next_params, (log_prob, marginal_log_lik) = em_step(params)
        log_probs.append(log_prob)
        marginal_log_liks.append(marginal_log_lik)

        if i > 2 and marginal_log_lik < marginal_log_liks[-2]:
            pbar.set_description(
                f"EM stopped at iteration {i+1} due to decreasing marginal_log_lik"
            )
            break

        if jnp.isnan(log_prob):
            pbar.set_description(f"EM stopped at iteration {i+1} due to NaN values")
            break

        params = next_params
        pbar.set_description(
            f"Iter {i+1}/{num_iters}, log-prob = {log_prob:.2f}, marginal log-lik = {marginal_log_lik:.2f}"
        )

    model.params = params
    return params, log_probs
