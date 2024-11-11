# -*- coding: utf-8 -*-
"""
@author: Amin
"""
import jax.numpy as jnp
import jax.random as jxr

from jax.example_libraries import optimizers

from jax import jit, value_and_grad, vmap
from tqdm.auto import trange

from params import ParamsGPLDS, \
        ParamswGPLDS, \
        ParamsGP, \
        ParamsBasis
        
from models import wGPLDS, GPLDS
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
            As = self.wgps['A'](params.A_weights, ts[1:]),
            bs = self.wgps['b'](params.b_weights, ts),
            Ls = self.wgps['L'](params.L_weights, ts)
        )

        return latents
    
# %%
class NoisyRecognition:
    def __init__(self, params: NamedTuple, **args):
        self.params = params

    def __call__(self, params: NamedTuple, key: jxr.PRNGKey, ys: Float[Array, "T N"], ts: Float[Array, "T M"]):
        '''
        Returns a sample and its log probability
        '''
        raise NotImplementedError
# %%
def variational_inference(
        key: jxr.PRNGKey,
        model: GPLDS,
        recognition: NoisyRecognition,
        ys: Float[Array, "num_batches num_timesteps emission_dim"],
        ts: Float[Array, "num_batches num_timesteps input_dim"],
        n_iter: int = 10000,
        step_size: float = .1,
        gamma: float = 1
    ):
    '''
    Variational inference is useful if the observations and dynamics models are not conjuage (e.g. Poisson observations)
    In that case computing the log_marginal is not tractable anymore and a recognition model for both local latents `x` and 
    global latents `y` is used.
    '''

    opt_init, opt_update, get_params = optimizers.adam(
        step_size,b1=0.8,b2=0.9,
    )
    opt_state = opt_init((model.params, recognition.params))
    params = get_params(opt_state)
    
    params = get_params(opt_state)


    def batch_elbo(params,y,ts,keys):
        B = len(ys)
        def _elbo(params,y,t,key):
            (model_params, recognition_params) = params
            # variational sample and log prob  
            k1, key = jxr.split(key,2)
            (x,latents),(log_px,log_platents) = recognition(recognition_params,k1,y,t)
            log_prior = log_px.sum() + log_platents.sum()/B

            # Compute model log probability over many samples of
            log_model = model.log_prob(model_params,latents,y,x,ts)

            # ELBO is the expected model log probability plus entropy
            return  -log_model+gamma*log_prior, log_model, log_prior


        _batch_elbo = vmap(
            lambda y,key: _elbo(params, y, ts, key),
            in_axes=(0,0),
            out_axes=0
        )

        val = _batch_elbo(y,keys)
        elbo_loss, log_model, log_prior = jnp.array(val).mean(1)
        return elbo_loss, (log_model, log_prior)
        

    @jit
    def update(params,y,ts,i,key,opt_state):
        ''' Perform a forward pass, calculate the MSE & perform a SGD step. '''
        (loss, (log_model, log_prior)), grads = value_and_grad(
                batch_elbo,has_aux=True
            )(params,y,ts,key)
        
        opt_state = opt_update(i,grads,opt_state)
        params = get_params(opt_state)
        
        return loss, log_model, log_prior, opt_state, params 


    pbar = trange(n_iter)
    pbar.set_description('jit compiling ...')
    
    
    losses = []

    for i in pbar:
        k1, key = jxr.split(key,2)
        keys = jxr.split(k1,ys.shape[0])
        loss, log_model, log_prior, opt_state, params = \
            update(params,ys,ts,i,keys,opt_state)
        
        losses.append(loss)
        if i % 10 == 0:
            pbar.set_description('ELBO: {:.2f}, Log Joint: {:.2f}, Log Prior: {:.2f}'.format(loss, log_model, log_prior))

    (model_params, recognition_params) = params
    model.set_params(model_params)
    recognition.params = recognition_params

    return losses


# %%
def fit_map(
        model: GPLDS,
        recognition: Recognition,
        ys: Float[Array, "num_batches num_timesteps emission_dim"],
        ts: Float[Array, "num_batches num_timesteps input_dim"],
        n_iter: int = 10000,
        step_size: float = .1,
        gamma: float = 1
    ):
    '''
    MAP inference is useful if the EM updates are not tractable for the model (e.g. function space GP prior)
    '''
    opt_init, opt_update, get_params = optimizers.adam(
        step_size,b1=0.8,b2=0.9,
    )
    opt_state = opt_init((model.params, recognition.params))
    params = get_params(opt_state)
    

    def batch_loss(params,ys,ts):
        B = len(ys)
        def loss_(params,y,t):
            (model_params, recognition_params) = params
            latents = recognition(recognition_params, t)
            log_marginal = model.log_marginal(model_params,latents,y)
            log_prior = model.log_prior(latents,t)/B
            loss = - log_marginal + gamma*log_prior
            return loss, log_marginal, log_prior
        
        batch_loss_ = vmap(
            lambda y,t: loss_(params,y,t),
            in_axes=(0,0),
            out_axes=0
        )

        val = batch_loss_(ys,ts)
        loss, log_marginal, log_prior = jnp.array(val).mean(1)
        
        return loss, (log_marginal, log_prior)

    @jit
    def update(params,ys,ts,i,opt_state):
        ''' Perform a forward pass, calculate the MSE & perform a SGD step. '''
        (loss, (log_model, log_prior)), grads = value_and_grad(
                batch_loss,has_aux=True
            )(params,ys,ts)
        
        opt_state = opt_update(i, grads, opt_state)
        params = get_params(opt_state)
        
        return loss, log_model, log_prior, opt_state, params 


    pbar = trange(n_iter)
    pbar.set_description('jit compiling ...')
    
    losses = []

    for i in pbar:
        loss, log_marginal, log_prior, opt_state, params = \
            update(params,ys,ts,i,opt_state)
        
        losses.append(loss)
        if i % 10 == 0:
            pbar.set_description('Loss: {:.2f}, Log Marginal: {:.2f}, Log Prior: {:.2f}'.format(loss, log_marginal, log_prior))
    
    (model_params, recognition_params) = params
    model.set_params(model_params)
    recognition.params = recognition_params
    
    return losses



# %%
def fit_em(
        model: wGPLDS,
        params: ParamswGPLDS,
        emissions: Float[Array, "num_batches num_timesteps emission_dim"],
        conditions: Optional[Float[Array, "num_batches num_timesteps input_dim"]]=None,
        num_iters: int=50,
    ):
    '''
    Requires the model to have the e_step and m_step functions implemented
    '''
    assert emissions.ndim == 3, 'emissions should be 3D, of shape (num_batches, num_timesteps, emission_dim)'

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
    pbar.set_description('jit compiling ...')
    
    for i in pbar:
        next_params, (log_prob, marginal_log_lik) = em_step(params)
        log_probs.append(log_prob)
        marginal_log_liks.append(marginal_log_lik)

        if i > 2 and marginal_log_lik < marginal_log_liks[-2]:
            pbar.set_description(f'EM stopped at iteration {i+1} due to decreasing marginal_log_lik')
            break

        if jnp.isnan(log_prob):
            pbar.set_description(f'EM stopped at iteration {i+1} due to NaN values')
            break

        params = next_params
        pbar.set_description(f'Iter {i+1}/{num_iters}, log-prob = {log_prob:.2f}, marginal log-lik = {marginal_log_lik:.2f}')

    return params, log_probs