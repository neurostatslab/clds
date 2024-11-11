# -*- coding: utf-8 -*-
"""
@author: Amin, Victor
"""

# %%
import jax
import jax.numpy as jnp
import jax.random as jxr

from jaxtyping import Array, Float
from typing import Optional, Tuple, Dict

from utils import logprob_analytic
from functools import partial
from jax import jit, lax, vmap

import logging
logging.basicConfig(level=logging.INFO, format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from scipy.linalg import solve_sylvester

from params import \
        ParamsEmission, \
        ParamsNormalLikelihood, \
        ParamsGPLDS, \
        ParamswGPLDS

import utils

import numpyro.distributions as dist

# %%
class GaussianProcess:
    '''
    Gaussian Process prior
    '''
    def __init__(self, kernel, D1: int = 1, D2: int = 1):
        self.kernel = kernel
        self.D1 = D1
        self.D2 = D2

    def evaluate_kernel(self, xs: Float[Array, "T M"], ys: Float[Array, "T M"]):
        return vmap(lambda x: vmap(lambda y: self.kernel(x,y))(xs))(ys)

    def sample(self, key: jxr.PRNGKey, ts: Float[Array, "T M"]) -> Float[Array, "T D1 D2"]:
        
        T = ts.shape[0]
        covariance = self.evaluate_kernel(ts,ts)
        distribution = dist.MultivariateNormal(
            jnp.zeros(T),
            covariance_matrix=covariance
        )
        fs = distribution.sample(key,sample_shape=(self.D1,self.D2))
        return fs.transpose(2,0,1)
    
    def log_prob(self,ts,fs):
        T = ts.shape[0]
        c_g = self.evaluate_kernel(ts,ts)
        distribution = dist.MultivariateNormal(
            jnp.zeros(T),
            covariance_matrix=c_g
        )
        lp = distribution.log_prob(fs.reshape(T,-1).T)
        return lp
    

#  %%
class InitialCondition:
    '''
    Distribution over the first time point of a state space model
    '''
    def __init__(self, D: int = 1):
        self.D = D
    
    def sample(self, stats, key: jxr.PRNGKey) -> Float[Array, "D"]:
        b0, L0 = stats
        x = dist.MultivariateNormal(
            b0,scale_tril=L0
            ).sample(key)
    
        return x

    def log_prob(self, stats, x: Float[Array, "D"]) -> Float:
        b0, L0 = stats
        lp = dist.MultivariateNormal(
            b0,scale_tril=L0
            ).log_prob(x)
        
        return lp

# %%
class TimeVaryingLDS:
    '''
    Differentiable representation of time varying linear dynamical system
    '''
    def __init__(self, D: int, initial: InitialCondition):
        # dynamics dimension
        self.D = D
        
        # initial condition distribution
        self.initial = initial

    def sample(self, stats: Tuple, key: jxr.PRNGKey, x0: Optional[Float[Array, "D"]]=None) -> Float[Array, "T D"]:
        (As, bs, Ls) = stats
        T, D = len(As)+1, self.D

        @jit
        def transition(carry, args):
            xs, k = carry
            A_new, b_new, L_new, _ = args
            k1, k = jax.random.split(k,2)
            mu = A_new@xs[-1]+b_new[:,0]
            x_new = dist.MultivariateNormal(
                mu,scale_tril=L_new
                ).sample(k1)
            xs = jnp.vstack((xs[1:],x_new))
            return (xs,k), None
        
        
        k1, key = jax.random.split(key,2)

        if x0 is None:
            x0 = self.initial.sample(stats=(bs[0],Ls[0]),key=k1)
        
        history = jnp.vstack((jnp.ones((T-1,D)),x0[None]))
        (xs,_),_ = lax.scan(
            transition, 
            (history,key), 
            (As,bs[1:],Ls[1:],jnp.arange(1,T))
        )
        
        return xs


    def log_prob(self, stats: Tuple, xs: Float[Array, "T D"]):
        (As, bs, Ls) = stats
        T, D = len(As)+1, self.D

        @jit
        def transition(carry, args):
            xs,lp = carry
            A_new, b_new, L_new, x_new, _ = args
            mu = A_new@xs[-1]+b_new[:,0]

            lp += dist.MultivariateNormal(
                mu,
                scale_tril=L_new
                ).log_prob(x_new)
            
            xs = jnp.vstack((xs[1:],x_new))
            return (xs,lp), None
        
        
        history = jnp.vstack((jnp.ones((T-1,D)),xs[0][None]))
        (_,lp) , _ = lax.scan(
            transition, 
            (history,self.initial.log_prob(x=xs[0],stats=(bs[0],Ls[0]))), 
            (As,bs[1:],Ls[1:],xs[1:],jnp.arange(1,T))
        )
        
        return lp

    def filter(self, stats):
        As, bs, Ls = stats
        D = self.D

        def predict(carry,inputs):
            m_last, P_last = carry
            A,b,L = inputs
            Q = L @ L.T
            m_pred = A @ m_last[:,0] + b[:,0]
            P_pred = A @ P_last @ A.T + Q
            return (m_pred[:,None], P_pred), (m_pred, P_pred)
        
        A0 = jnp.eye(D)[None]
        b0 = jnp.zeros((D,1))[None]
        L0 = jnp.zeros((D,D))[None]

        A_ = jnp.concatenate((A0,As))
        b_ = jnp.concatenate((b0,bs[1:]))
        L_ = jnp.concatenate((L0,Ls[1:]))

        _, (mus, sigmas) = jax.lax.scan(
                predict,
                (bs[0],Ls[0]@Ls[0].T),
                (A_,b_,L_)
            )
        return mus.squeeze(), sigmas
    

# %%
class ConditionalLikelihood:
    def sample(self, params, stats, key):
        self.params = None
        raise NotImplementedError
    
    def log_prob(self, params, stats, y):
        raise NotImplementedError

# %%
class PoissonConditionalLikelihood(ConditionalLikelihood):
    def __init__(self, D: int):
        self.D = D
    
    def sample(self, params: None, stats: Float[Array, "D"], key: jxr.PRNGKey) -> Float[Array, "D"]:
        rate = stats
        y = dist.Poisson(jax.nn.softplus(rate)).to_event(1).sample(key)
        return y
    
    def log_prob(self, params: None, stats: Float[Array, "D"], y: Float[Array, "D"]) -> Float:
        rate = stats
        lp = dist.Poisson(jax.nn.softplus(rate)).to_event(1).log_prob(y)
        return lp
    
# %%
class NormalConditionalLikelihood(ConditionalLikelihood):
    def __init__(self, params: ParamsNormalLikelihood, D: int = 1):
        self.D = D
        self.params = params

    def sample(self, params: ParamsNormalLikelihood, stats: Float[Array, "N"], key: jxr.PRNGKey) -> Float[Array, "N"]:
        y = dist.MultivariateNormal(
            stats,scale_tril=params.scale_tril
            ).sample(key)
        return y
    
    def log_prob(self,params: ParamsNormalLikelihood, y: Float[Array, "N"], stats: Float[Array, "N"]) -> Float:
        lp = dist.MultivariateNormal(
            stats,scale_tril=params.scale_tril
            ).log_prob(y)
        return lp
    

# %%
class LinearEmission:
    def __init__(self, params: ParamsEmission, D: int, N: int):
        self.D = D # dynamics dimension
        self.N = N # observation dimension
        self.params = params

    def __call__(self, params: ParamsEmission, x: Float[Array, "D"]) -> Float[Array, "N"]:
        y = x@params.Cs.T + params.ds[None]
        return y

# %%
class GPLDS:
    '''
    GPLDS with function space parameterization of the priors for the parameters {A, b, L}.
    '''
    def __init__(self, gps: Dict, dynamics: TimeVaryingLDS, emissions: LinearEmission, likelihood: ConditionalLikelihood):
        self.gps = gps
        self.dynamics = dynamics
        self.emissions = emissions
        self.likelihood = likelihood

        self.params = ParamsGPLDS(
            emissions = emissions.params,
            likelihood = likelihood.params
        )
        
    def sample(self, params: ParamsGPLDS, ts: Float[Array, "T M"], key: jxr.PRNGKey) -> Float[Array, "N"]:
        '''
        Sample from the full GPLDS prior at the points `ts`
        '''
        kA,kb,kL,key = jax.random.split(key,4)
        
        stats = (self.gps['A'].sample(kA,ts), self.gps['b'].sample(kb,ts), self.gps['L'].sample(kL,ts))
        k1, key = jax.random.split(key,2)
        x = self.dynamics.sample(
            stats=stats,
            key=k1,x0=None
        )

        stats = self.emissions(params.emissions,x)

        k1, key = jax.random.split(key,2)
        y = self.likelihood.sample(
            params.likelihood,
            stats=stats,key=k1
        )
        
        return y

    def log_prob(self, params: ParamsGPLDS, stats: Tuple, y: Float[Array, "T N"], x: Float[Array, "D"], ts: Float[Array, "T M"]) -> Float:
        '''
        Evaluate the log probability of the GPLDS draws at the time points `ts`, latents `x`, and observations `y`
        '''
        (As, bs, Ls) = stats

        lpA = self.gps['A'].log_prob(ts,As)
        lpb = self.gps['b'].log_prob(ts,bs)
        lpL = self.gps['L'].log_prob(ts,Ls)

        ld = self.dynamics.log_prob(stats=stats,xs=x)
        stats = self.emissions(params.emissions,x)
        le = self.likelihood.log_prob(params.likelihood,stats,y=y)
        
        return lpA.sum()+lpb.sum()+lpL.sum()+ld.sum()+le.sum()
    
    def log_marginal(self, params: ParamsGPLDS, stats: Tuple, y: Float[Array, "T N"]) -> Float:
        '''
        Function for integrating out local latents (x)
        '''
        (As, bs, Ls) = stats
        def transition(carry,inputs):
            # Unpack carry from last iteration
            m_last, P_last = carry

            scale_tril_y = params.likelihood.scale_tril
            C = params.emissions.Cs
            d = params.emissions.ds
            
            A,b,L,y = inputs
            
            # Compute covariances.
            Q = L @ L.T
            R = scale_tril_y @ scale_tril_y.T

            # Prediction step (propogate dynamics, eq 4.20 in Sarkka)
            m_pred = A @ m_last[:,0] + b[:,0]
            P_pred = A @ P_last @ A.T + Q

            # Compute log p(y[k] | y[1], ..., y[k-1]) using eq 4.19 in Sarkka
            S = C @ P_pred @ C.T + R
            y_pred = C @ m_pred + d
            log_prob_y = jax.scipy.stats.multivariate_normal.logpdf(y, y_pred, S)

            # Update step (condition on y, eq 4.21 in Sarkka)
            v = y - y_pred
            K = jnp.linalg.solve(S,C@P_pred).T
            m = m_pred + K @ v
            P = P_pred - K @ S @ K.T

            # Carry over mean and covariance of x[t], conditioned on y[t].
            # Output log probability of y[t], conditioned on y[1], ..., y[t-1].
            return (m[:,None], P), log_prob_y
        
        D = self.dynamics.D

        A_ = jnp.concatenate((jnp.eye(D)[None],As))
        b_ = jnp.concatenate((jnp.zeros((D,1))[None],bs[1:]))
        L_ = jnp.concatenate((jnp.zeros((D,D))[None],Ls[1:]))
        
        log_marginal = jax.lax.scan(
                transition, 
                (bs[0],Ls[0]@Ls[0].T),
                (A_,b_,L_,y)
            )[1].sum()
        
        return  log_marginal
    
    def log_prior(self, stats: Tuple, ts: Float[Array, "T M"]) -> Float:
        (As, bs, Ls) = stats
        lpA = self.gps['A'].log_prob(ts[1:],As)
        lpb = self.gps['b'].log_prob(ts,bs)
        lpL = self.gps['L'].log_prob(ts,Ls)

        # sum over time points
        log_prior = (lpA.sum()+lpb.sum()+lpL.sum())

        # pior needs to be counted once for the whole batch
        return log_prior

    def filter(self, params: ParamsGPLDS, stats: Tuple):
        '''
        Returns the mean and covariance of latents and observations by running filtering
        '''

        mus, sigmas = self.dynamics.filter(stats)

        sigmas_obs = jnp.array([
            params.emissions.Cs@sigmas[t]@params.emissions.Cs.T + \
            params.likelihood.scale_tril@params.likelihood.scale_tril.T
            for t in range(len(sigmas))])
        mus_obs = (params.emissions.Cs @ mus.T).T + params.emissions.ds

        return (mus.squeeze(), sigmas), (mus_obs.squeeze(), sigmas_obs)
    
    def set_params(self, params:ParamsGPLDS):
        self.likelihood.params = params.likelihood
        self.emissions.params = params.emissions

# %%
class WeightSpaceGaussianProcess():
    '''
    Weight-space Gaussian Process prior
    '''
    def __init__(self, basis_funcs, D1: int=1, D2: int=1):
        self.basis_funcs = basis_funcs
        self.D1 = D1
        self.D2 = D2

    def __call__(self, 
            weights: Float[Array, "len_basis D1 D2"], 
            xs: Float[Array, "M T"]
        ) -> Float[Array, "T D1 D2"]:
        '''
        Evaluate A_ij(x) = \sum_k w^{(ij)} \phi_k(x) at the M-dimensional points `xs`
        with `weights` w^{(ij)} and basis functions \phi_k.
        '''
        PhiX = self.evaluate_basis(xs)
        return jnp.einsum('kij,tk->tij', weights, PhiX)
    
    def sample_weights(self, key: jxr.PRNGKey) -> Float[Array, "len_basis D1 D2"]:
        return jxr.normal(key, shape=(len(self.basis_funcs), self.D1, self.D2))
    
    def evaluate_basis(self, x: Float[Array, "T M"]) -> Float[Array, "T len_basis"]:
        return jnp.array([jax.vmap(f)(x) for f in self.basis_funcs]).T

    def sample(self, key: jxr.PRNGKey, xs: Float[Array, "T M"]) -> Float[Array, "T D1 D2"]:
        '''
        Sample from the GP prior at the points `xs`
        '''
        weights = self.sample_weights(key)
        PhiX = self.evaluate_basis(xs)
        return self.__call__(weights, xs)
    
    def log_prob(self, xs: Float[Array, "T M"], fs: Float[Array, "T D1 D2"]) -> Float[Array, "D1 D2"]:
        '''
        Compute the log probability of the GP draws at the time points `ts`
        '''
        if fs.ndim == 2:
            assert (self.D1 == 1) ^ (self.D2 == 1), 'Incorrect dimensions'
            fs = fs.reshape(-1, self.D1, self.D2)
        assert fs.shape[1] == self.D1 and fs.shape[2] == self.D2, 'Incorrect dimensions'
        T = len(fs)
        Phi = self.evaluate_basis(xs) # T x K
        cov = jnp.dot(Phi, Phi.T)   # T x T
        return dist.MultivariateNormal(jnp.zeros(T), covariance_matrix=cov).log_prob(fs.reshape(T,-1).T).reshape(self.D1, self.D2)
        # return jax.vmap(lambda _f: logprob_analytic(_f, jnp.zeros(T), cov), in_axes=(1))(fs.reshape(T, -1)).reshape(self.D1, self.D2)

    def log_prob_weights(self, weights: Float[Array, "len_basis D1 D2"]) -> float:
        '''
        Standard Gaussian prior on the weights
        '''
        return -0.5 * jnp.sum(weights**2)

# %%
class wGPLDS():
    '''
    GPLDS with weight-space view parametrization of the priors for the parameters {A, b, C}. 
    By default: A has weight GP prior, whereas b and C are optional. 
                If weight GP priors are not provided for b and C, they are learned as C fixed, b time-varying.
    This is a early version, only currently supporting EM. Does not support sampling. Does not support inputs other than GP conditions. 
    '''
    def __init__(self, params: ParamswGPLDS, wgps: dict, state_dim: int, emission_dim: int):
        # TODO: Add other dynamics and emission models if we want to sample from it
        self.wgps = wgps
        assert 'A' in self.wgps, 'Dynamics GP prior is required'
        if 'b' not in self.wgps:
            self.wgps['b'] = None
        if 'C' not in self.wgps:
            self.wgps['C'] = None
        if 'm0' not in self.wgps:
            self.wgps['m0'] = None
        
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.params = params

    def log_prior(self, params: ParamswGPLDS, inputs):
        '''Compute the log prior of the parameters. Conditions are inputs'''
        logprior_A = self.wgps['A'].log_prob_weights(params.dynamics_gp_weights)

        if self.wgps['b'] is None:
            logprior_b = 0.
        else:
            logprior_b = self.wgps['b'].log_prob_weights(params.bias_gp_weights)

        if self.wgps['C'] is None:
            logprior_C = 0.
        else:
            logprior_C = self.wgps['C'].log_prob_weights(params.emissions_gp_weights)

        if self.wgps['m0'] is None:
            logprior_m0 = 0.
        else:
            logprior_m0 = self.wgps['m0'].log_prob_weights(params.m0_gp_weights)

        
        return logprior_A + logprior_b + logprior_C + logprior_m0

    def weights_to_params(self, params, inputs):
        '''Transform weights of weight space into parameters. 
            Implement as needed for all weight-space GP priors.'''
        As = self.wgps['A'](params.dynamics_gp_weights, inputs)
        Cs = self.wgps['C'](params.emissions_gp_weights, inputs) if self.wgps['C'] is not None else params.Cs
        bs = self.wgps['b'](params.bias_gp_weights, inputs) if self.wgps['b'] is not None else params.bs
        m0 = self.wgps['m0'](params.m0_gp_weights, inputs)[0] if self.wgps['m0'] is not None else params.m0 #! Some unnecessary computation, keeping only t=0
        return As, Cs, bs.squeeze(), m0.squeeze()

    def smoother(self, params: ParamswGPLDS, emissions, inputs):
        '''inputs as conditions'''
        # Format params
        As, Cs, bs, m0 = self.weights_to_params(params, inputs)
        if Cs.ndim == 2:
            Cs = jnp.tile(Cs[None], (len(inputs), 1, 1))

        # Run the smoother
        lgssm_params = {
            'm0': m0,
            'S0': params.S0,
            'As': As,
            'bs': bs,
            'Q': params.Q,
            'Cs': Cs,
            'R': params.R,
        }
        return utils.lgssm_smoother(**lgssm_params, ys=emissions)

    def e_step(
        self,
        params: ParamswGPLDS,
        emissions: Float[Array, "num_timesteps emission_dim"],
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None,
    ):
        def weightspace_stats(
                XTX: Float[Array, 'T D2 D2'], 
                XTY: Float[Array, 'T D2 D1'], 
                wgp_prior: WeightSpaceGaussianProcess,
                conditions: Float[Array, "T condition_dim"]
                ) -> tuple:
            '''
            Compute the expected sufficient statistics for the weight-space GP prior. 
            Provide the sufficient stats X^T X and X^T Y for the problem Y = A(C)X + noise.
            This returns the expanded stats Phi @ X^T X @ Phi^T and Phi @ X^T Y for the basis functions Phi(C).
            '''
            _Phi = wgp_prior.evaluate_basis(conditions)

            ZTZ = jnp.einsum('tk,tij,tl->ikjl', _Phi, XTX, _Phi)
            ZTY = jnp.einsum('tk,tim->ikm', _Phi, XTY)

            ZTZ = ZTZ.reshape(len(wgp_prior.basis_funcs) * wgp_prior.D2, len(wgp_prior.basis_funcs) * wgp_prior.D2)
            ZTY = ZTY.reshape(len(wgp_prior.basis_funcs) * wgp_prior.D2, wgp_prior.D1)
            return (ZTZ, ZTY)

        '''take inputs to be theta'''
        num_timesteps = emissions.shape[0]
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 1))

        # Run the smoother to get posterior expectations
        marginal_loglik, filter_results, smoother_results = self.smoother(params, emissions, inputs)
        smoothed_means, smoothed_covariances, smoothed_cross_covariances = smoother_results

        # shorthand
        Ex = smoothed_means
        Exp = smoothed_means[:-1]
        Exn = smoothed_means[1:]
        Vx = smoothed_covariances
        Vxp = smoothed_covariances[:-1]
        Vxn = smoothed_covariances[1:]
        Expxn = smoothed_cross_covariances
        
        # Append bias to the inputs
        # inputs = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
        up = inputs[:-1]
        # u = inputs

        # expected sufficient statistics for the initial distribution
        Ex0 = smoothed_means[0]
        Ex0x0T = smoothed_covariances[0] + jnp.outer(Ex0, Ex0)
        init_stats = (Ex0, Ex0x0T, 1)
        if self.wgps['m0'] is None:
            wgpm0_stats = None
        else:
            m0_targets = Ex0.reshape(1, 1, self.state_dim)
            XTX_m0 = jnp.ones((1, 1, 1))

            _cond = up[0].reshape(1) if up[0].ndim == 0 else up[0].reshape(1, up.shape[1])
            wgpm0_stats = weightspace_stats(XTX_m0, m0_targets, self.wgps['m0'], _cond)

        # expected sufficient statistics for the dynamics
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        sum_xpxnT = Expxn.sum(0)
        sum_xpxpT = Vxp.sum(0) + Exp.T @ Exp
        sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
        dynamics_stats = (sum_xpxpT, sum_xpxnT, sum_xnxnT, num_timesteps - 1)

        # Dynamics wGP sufficient stats
        
        # # Partial E-step: assuming delta posterior around mean, hence a lin reg from the smoothed means only
        # PhiAp = self.wgps['A'].evaluate_basis(inputs)
        # _Z = jnp.einsum('tk,ti->tik', PhiAp, Exp)
        # _Y = Exn - params.bs[:len(inputs)]
        # _ZTZ = jnp.einsum('tik,tjl->ikjl', _Z, _Z).reshape(len(self.wgps['A'].basis_funcs) * self.wgps['A'].D2, len(self.wgps['A'].basis_funcs) * self.wgps['A'].D2)
        # _ZTY = jnp.einsum('tik,tj->ikj', _Z, _Y).reshape(len(self.wgps['A'].basis_funcs) * self.wgps['A'].D2, self.wgps['A'].D1)

        # full E-step sufficient stats
        bs = self.wgps['b'](params.bias_gp_weights, up).squeeze() if self.wgps['b'] is not None else params.bs
        Expxn_b = Expxn - jnp.einsum('ti,tj->tij', Exp, bs)
        ExpxpT = jnp.einsum('ti,tj->tij', Exp, Exp) + Vxp
        wgpA_stats = weightspace_stats(ExpxpT, Expxn_b, self.wgps['A'], up)
        wgpA_sylvester_stats = (wgpA_stats[0], params.Q, wgpA_stats[1], 1)

        # Q sufficient stats # TODO. Currently uses static sufficient stats
        # Vxn, Vxp = Vx[1:], Vx[:-1]
        # sum_AExpxnT = jnp.einsum('tij,tjk->ik', F, Expxn) #.sum(0)
        # sum_AExpxpAT = jax.vmap(lambda _m, _S, _A: _A @ (_m @ _m.T + _S) @ _A.T)(Exp, Vxp, F).sum(0)
        # Q = (sum_xnxnT - _A_ExpxnT - _A_ExpxnT.T + _A_ExpxpT_A) / (dynamics_stats[-1] - 1)
        
        # bias sufficient stats
        F = self.wgps['A'](params.dynamics_gp_weights, up)
        bias_targets = Exn - jnp.einsum('tij,tj->ti', F, Exp)
        if self.wgps['b'] is None:
            bias_stats = (bias_targets, 1)
        else:
            bias_targets = bias_targets.reshape(len(up), 1, self.state_dim)
            _XTXb = jnp.ones((len(up), 1, 1))
            bias_stats = weightspace_stats(_XTXb, bias_targets, self.wgps['b'], up)

        # more expected sufficient statistics for the emissions
        y = emissions
        sum_xxT = Vx.sum(0) + Ex.T @ Ex
        sum_xyT = Ex.T @ y
        sum_yyT = emissions.T @ emissions
        emission_stats = (sum_xxT, sum_xyT, sum_yyT, num_timesteps)

        if self.wgps['C'] is None:
            wgpC_stats = None
            wgpC_sylvester_stats = None
        else:
            _xxT = jnp.einsum('ti,tj->tij', Ex, Ex) + Vx
            _xyT = jnp.einsum('ti,tj->tij', Ex, y)
            wgpC_stats = weightspace_stats(_xxT, _xyT, self.wgps['C'], inputs)

            wgpC_sylvester_stats = (wgpC_stats[0], params.R, wgpC_stats[1], 1)

        return (init_stats, wgpm0_stats, dynamics_stats, wgpA_stats, bias_stats, emission_stats, wgpC_stats, wgpA_sylvester_stats, wgpC_sylvester_stats), marginal_loglik
    
    def m_step(
            self,
            params: ParamswGPLDS,
            batch_stats: Tuple, #inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None,
        ) -> ParamswGPLDS:
        def fit_linear_regression(ExxT, ExyT, EyyT, N):
            # Solve a linear regression given sufficient statistics
            W = utils.psd_solve(ExxT, ExyT).T
            Sigma = (EyyT - W @ ExyT - ExyT.T @ W.T + W @ ExxT @ W.T) / N
            return W, Sigma

        def fit_gplinear_regression(ZTZ, ZTY, wgp_prior):
            # Solve a linear regression in weight-space given sufficient statistics
            weights = jax.scipy.linalg.solve(
                ZTZ + jnp.eye(len(wgp_prior.basis_funcs) * wgp_prior.D2), ZTY, 
                assume_a='pos'
                )
            weights = weights.reshape(wgp_prior.D2, len(wgp_prior.basis_funcs), wgp_prior.D1).transpose(1,2,0)
            return weights
        
        def fit_gplinear_regression_sylvester(ZTZ, Sigma, ZTY, wgp_prior):
            # Solve a linear regression in weight-space given sufficient statistics
            # weights = utils.jax_solve_sylvester(B, ZTZ, ZTY, assume_a='pos')
            weights = utils.jax_solve_sylvester_BS(ZTZ, Sigma, ZTY)
            weights = weights.reshape(wgp_prior.D2, len(wgp_prior.basis_funcs), wgp_prior.D1).transpose(1,2,0)
            return weights

        # Sum the statistics across all batches
        stats = jax.tree_util.tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, wgpm0_stats, dynamics_stats, wgpA_stats, bias_stats, emission_stats, wgpC_stats, wgpA_sylvester_stats, wgpC_sylvester_stats = stats

        # Perform MLE estimation jointly
        sum_x0, sum_x0x0T, N = init_stats
        S = sum_x0x0T / N - jnp.outer(sum_x0, sum_x0) / (N**2)
        if self.wgps['m0'] is None:
            W_m0 = None
            m = sum_x0 / N
        else:
            W_m0 = fit_gplinear_regression(*wgpm0_stats, self.wgps['m0'])
            m = None

        # Dynamics M-step
        # W_A = fit_gplinear_regression(*wgpA_stats, self.wgps['A'])
        W_A = fit_gplinear_regression_sylvester(
            wgpA_sylvester_stats[0], wgpA_sylvester_stats[1] / wgpA_sylvester_stats[3], wgpA_sylvester_stats[2],
            wgp_prior=self.wgps['A']
            )

        # TODO: Q M-step. Currently uses static sufficient stats
        # Vxn, Vxp = Vx[1:], Vx[:-1]
        # _ExnxnT = Exn.T @ Exn + Vxn.sum(0) # Unchanged
        # _A_ExpxnT = jnp.einsum('tij,tjk->ik', F, Expxn)
        # _A_ExpxpT_A = jax.vmap(lambda _m, _S, _A: _A @ (_m @ _m.T + _S) @ _A.T)(Exp, Vxp, F).sum(0)
        # Q = (_ExnxnT - _A_ExpxnT - _A_ExpxnT.T + _A_ExpxpT_A) / (dynamics_stats[-1] - 1)

        F_static, Q = fit_linear_regression(*dynamics_stats)

        # Bias update
        if self.wgps['b'] is None:
            W_b = None
            bs = bias_stats[0]/bias_stats[1]
        else: 
            # In weight space
            W_b = fit_gplinear_regression(*bias_stats, self.wgps['b'])
            bs = None
        # # Use the following for homogeneous bias
        # b = jnp.mean(bs, axis=0)
        # bs = jnp.tile(b, (len(up), 1))

        # Emission M-step
        H_static, R = fit_linear_regression(*emission_stats)
        if self.wgps['C'] is None:
            W_C = None
            Cs = H_static

            # print('Warning: Emission C is fixed')
            # true_C = jxr.normal(jxr.PRNGKey(0), (self.emission_dim, 2)) # Overwrites neuron tuning
            # Cs = jnp.tile(true_C, (100, 1, 1))
        else:
            # In weight space
            # W_C = fit_gplinear_regression(*wgpC_stats, self.wgps['C'])
            W_C = fit_gplinear_regression_sylvester(
                wgpC_sylvester_stats[0], wgpC_sylvester_stats[1] / wgpC_sylvester_stats[3], wgpC_sylvester_stats[2],
                wgp_prior=self.wgps['C']
                )
            Cs = None

        # logger.warning('Warning, fixing R')
        # Q = jnp.eye(self.state_dim) # Can fix Q to be identity (for identifiability)
        # R = jnp.eye(self.emission_dim)
        params = ParamswGPLDS(
            m0=m, S0=S, 
            dynamics_gp_weights=W_A, 
            bias_gp_weights=W_b, 
            emissions_gp_weights=W_C, 
            m0_gp_weights=W_m0,
            Cs=Cs, bs=bs,
            Q=Q, R=R,
        )
        return params

    def log_prob(self, params, emissions, conditions):
        '''Compute the log probability of the emissions given the parameters'''
        def batch_log_prob(_emissions, _conditions):
            log_prior = self.log_prior(params, _conditions)
            marginal_loglik, _, _ = self.smoother(params, _emissions, _conditions)
            return log_prior + marginal_loglik

        return vmap(batch_log_prob)(emissions, conditions).sum()

    def marginal_log_lik(self, params, emissions, conditions):
        '''Compute the marginal log likelihood of the emissions given the parameters'''
        def batch_marginal_log_lik(_emissions, _conditions):
            (marginal_loglik, _, _) = self.smoother(params, _emissions, _conditions)
            return marginal_loglik
        return vmap(batch_marginal_log_lik)(emissions, conditions).sum()