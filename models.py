# -*- coding: utf-8 -*-
"""
@author: Amin
"""

# %%
import jax
import jax.numpy as jnp
import jax.random as jxr
from jaxtyping import Array, Float
from typing import NamedTuple, Optional, Tuple
from utils import logprob_analytic
from functools import partial
from jax import jit, lax, vmap

import utils

import numpyro.distributions as dist

# %%
class GaussianProcess:
    def __init__(self,kernel,D1,D2):
        self.kernel = kernel
        self.D1 = D1
        self.D2 = D2

    def evaluate_kernel(self, xs, ys):
        return vmap(lambda x: vmap(lambda y: self.kernel(x,y))(xs))(ys)


    def sample(self,key,ts):
        T = ts.shape[0]
        c_g = self.evaluate_kernel(ts,ts)
        fs = dist.MultivariateNormal(
            jnp.zeros(T),covariance_matrix=c_g
        ).sample(key,sample_shape=(self.D1,self.D2))
        return fs

    
    def log_prob(self,ts,fs):
        T = ts.shape[0]
        c_g = self.evaluate_kernel(ts,ts)
        lp = dist.MultivariateNormal(
            jnp.zeros(T),covariance_matrix=c_g
        ).log_prob(fs.reshape(T,-1).T)
        return lp
    

#  %%
class InitialCondition:
    def __init__(self,D,mu=None,scale_tril=None):
        pass
        # self.D = D
        # self.mu = jnp.zeros(D) if mu is None else mu
        # self.scale_tril = jnp.eye(D) if scale_tril is None else scale_tril

    def set_params(self,params):
        pass
        # self.mu,self.scale_tril = params
        
    @property
    def params(self):
        return ()
        # return (self.mu,self.scale_tril)
    
    def sample(self,key,params):
        mu,scale_tril = params
        # mu,scale_tril = self.mu, self.scale_tril
        x0 = dist.MultivariateNormal(
            mu,scale_tril=scale_tril
        ).sample(key)
        return x0

    def log_prob(self,x0,params):
        mu,scale_tril = params
        lp = dist.MultivariateNormal(
            mu,scale_tril=scale_tril
        ).log_prob(x0)
        return lp



# %%
class TimeVarLDS:
    '''Differentiable representation of time varying LDS for inference
    '''
    def __init__(self,D,initial):
        # dynamics dimension
        self.D = D
        
        # initial condition distribution
        self.initial = initial

    def sample(self,key,As,bs,Ls,x0=None):
        D = self.D
        T = len(As)+1

        @jit
        def transition(carry, args):
            xs, k = carry
            A_new, b_new, L_new, _ = args
            k1, k = jax.random.split(k,2)
            mu = A_new@xs[-1]+b_new[:,0]
            x_new = dist.MultivariateNormal(mu,scale_tril=L_new).sample(k1)
            xs = jnp.vstack((xs[1:],x_new))
            return (xs,k), None
        
        
        k1, key = jax.random.split(key,2)
        x0 = self.initial.sample(k1,(bs[0].squeeze(),Ls[0])) if x0 is None else x0
        
        history = jnp.vstack((jnp.ones((T-1,D)),x0[None]))
        (xs,_),_ = lax.scan(
            transition, 
            (history,key), 
            (As,bs[1:],Ls[1:],jnp.arange(1,T))
        )
        
        return xs
    

    def set_params(self,params):
        pass
        # self.initial.set_params(params[:2])
        

    @property
    def params(self):
        return ()
        # return self.initial.params
    

    def log_prob(self,As,bs,Ls,xs,params):
        D = self.D
        T = len(As)+1
        # init_params = params[:2]
        init_params = (bs[0],Ls[0])

        @jit
        def transition(carry, args):
            xs,lp = carry
            A_new, b_new, L_new, x_new, _ = args
            mu = A_new@xs[-1]+b_new[:,0]
            lp += dist.MultivariateNormal(mu,scale_tril=L_new).log_prob(x_new)
            xs = jnp.vstack((xs[1:],x_new))
            return (xs,lp), None
        
        
        history = jnp.vstack((jnp.ones((T-1,D)),xs[0][None]))
        (_,lp) , _ = lax.scan(
            transition, 
            (history,self.initial.log_prob(xs[0],init_params)), 
            (As,bs[1:],Ls[1:],xs[1:],jnp.arange(1,T))
        )
        
        return lp

    def evolve_stats(self,As,bs,Ls):
        D = As.shape[1]

        def scanned_func(carry,inputs):
            m_last, P_last = carry
            A,b,L = inputs
            Q = L @ L.T
            m_pred = A @ m_last + b
            P_pred = A @ P_last @ A.T + Q
            return (m_pred, P_pred), (m_pred, P_pred)

        
        A0 = jnp.eye(D)[None]
        b0 = jnp.zeros((D))[None]
        L0 = jnp.zeros((D,D))[None]

        A_ = jnp.concatenate((A0,As))
        b_ = jnp.concatenate((b0,bs[1:]))
        L_ = jnp.concatenate((L0,Ls[1:]))

        _, (mus, sigmas) = jax.lax.scan(
                scanned_func, 
                (bs[0],Ls[0]@Ls[0].T),
                (A_,b_,L_)
            )
        return mus, sigmas
# %%
class PoissonConditionalLikelihood:
    def __init__(self,D):
        self.D = D
    
    def sample(self,key,stats):
        rate = stats
        Y = dist.Poisson(jax.nn.softplus(rate)).to_event(1).sample(key)
        return Y
    
    def log_prob(self,stats,y,params):
        rate = stats
        return dist.Poisson(jax.nn.softplus(rate)).to_event(1).log_prob(y)
    
    @property
    def params(self):
        return ()
    
    def set_params(self,params):
        pass

# %%
class NormalConditionalLikelihood:
    def __init__(self,D,scale_tril=None):
        self.D = D

        if scale_tril is not None:
            self.scale_tril = scale_tril
        else:
            self.scale_tril = jnp.eye(D) 

    def sample(self,key,stats):
        mu,scale_tril=stats,self.scale_tril
        y = dist.MultivariateNormal(mu,scale_tril=scale_tril).sample(key)
        return y
    
    def log_prob(self,stats,y,params):
        mu,scale_tril=stats,params[0]
        return dist.MultivariateNormal(mu,scale_tril=scale_tril).log_prob(y)
    
    @property
    def params(self):
        return (self.scale_tril,)
    
    def set_params(self,params):
        self.scale_tril = params[0]


# %%
class LinearEmission:
    def __init__(self,key,D,N,C=None,d=None):
        self.D = D # dynamics dimension
        self.N = N # observation dimension

        k1, k2 = jax.random.split(key,2)

        # initialize the weight matrix if not given
        if C is None:
            self.C =  dist.Normal(0,1/N).sample(k1,sample_shape=(self.N,self.D))
        else:
            self.C = C
        
        if d is None:
            self.d = dist.Normal(0,1/N).sample(k2,sample_shape=(self.N,))
        else:
            self.d = d

    def set_params(self, params):
        self.C,self.D = params[0:2]
        
    @property
    def params(self):
        return (self.C,self.d)

    def f(self,x,params):
        C,d = params[:2]
        y = x@C.T + d[None]
        return y
# %%
class GPLDS:
    def __init__(self, gps, dynamics, emissions, likelihood):
        self.gps = gps
        self.dynamics = dynamics
        self.emissions = emissions
        self.likelihood = likelihood
        
    def model(self,y,u,key):
        T,N = y.shape
        B = 1
        kA,kb,kL,key = jax.random.split(key,4)
        
        As = self.gps['A'].sample(kA)
        bs = self.gps['b'].sample(kb)
        Ls = self.gps['L'].sample(kL)
        
        k1, k2 = jax.random.split(key,2)
        x = self.dynamics.sample(
            key=k1,As=As,bs=bs,Ls=Ls,x0=None,u=u,T=T
        )
        stats = self.emissions.f(
            x,self.emissions.params
        )
        y = self.likelihood.sample(
            key=k2,stats=stats
        )
        
        return y

    def log_prob(self,y,x,As,bs,Ls,us,params):
        ld,le,ll = len(self.dynamics.params), len(self.emissions.params), len(self.likelihood.params)
        dynamics_params, emissions_params, likelihood_params = params[:ld], params[ld:ld+le], params[ld+le:ld+le+ll]

        lpA = self.gps['A'].log_prob(us,As)
        lpb = self.gps['b'].log_prob(us,bs)
        lpL = self.gps['L'].log_prob(us,Ls)

        ld = self.dynamics.log_prob(As,bs,Ls,x,params=dynamics_params)
        stats = self.emissions.f(x,params=emissions_params)
        le = self.likelihood.log_prob(stats,y=y,params=likelihood_params)
        
        return lpA.sum()+lpb.sum()+lpL.sum()+ld.sum()+le.sum()
    
    
    def set_params(self,params):
        ld,le,ll = len(self.dynamics.params), len(self.emissions.params), len(self.likelihood.params)
        dynamics_params, emissions_params, likelihood_params = params[:ld], params[ld:ld+le], params[ld+le:ld+le+ll]
        
        self.dynamics.set_params(dynamics_params)
        self.emissions.set_params(emissions_params)
        self.likelihood.set_params(likelihood_params)

    @property
    def params(self):
        return self.dynamics.params+self.emissions.params+self.likelihood.params

def safe_wrap(X):
    return jnp.where(jnp.isclose(X, 0.), 0., X)

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
    
    def weights(self, key: jxr.PRNGKey) -> Float[Array, "len_basis D1 D2"]:
        '''
        retrieve weights associated with key
        '''
        return jxr.normal(key, shape=(len(self.basis_funcs), self.D1, self.D2))
    
    def evaluate_basis(self, x: Float[Array, "T M"]) -> Float[Array, "T len_basis"]:
        return jnp.array([jax.vmap(f)(x) for f in self.basis_funcs]).T

    def sample(self, key: jxr.PRNGKey, xs: Float[Array, "T M"]) -> Float[Array, "T D1 D2"]:
        '''
        Sample from the GP prior at the points `xs`
        '''
        weights = self.weights(key)
        PhiX = self.evaluate_basis(xs)
        return self.__call__(weights, xs)
    
    def log_prob(self, xs: Float[Array, "T M"], fs: Float[Array, "T D1 D2"]) -> Float[Array, "D1 D2"]:
        '''
        Compute the log probability of the GP draws at the time points ts
        '''
        if fs.ndim == 2:
            assert (self.D1 == 1) ^ (self.D2 == 1), 'Incorrect dimensions'
            fs = fs.reshape(-1, self.D1, self.D2)
        assert fs.shape[1] == self.D1 and fs.shape[2] == self.D2, 'Incorrect dimensions'
        T = len(fs)
        Phi = self.evaluate_basis(xs) # T x K
        cov = jnp.dot(Phi, Phi.T)   # T x T
        # return dist.MultivariateNormal(jnp.zeros(T), covariance_matrix=cov).log_prob(fs.reshape(T,-1).T).reshape(self.D1, self.D2)
        return jax.vmap(lambda _f: logprob_analytic(_f, jnp.zeros(T), cov), in_axes=(1))(fs.reshape(T, -1)).reshape(self.D1, self.D2)

# %%
class ParamswGPLDS(NamedTuple):
    m0: Float[Array, "state_dim"]
    S0: Float[Array, "state_dim state_dim"]
    dynamics_gp_weights: Float[Array, "state_dim state_dim len_basis"]
    bs: Float[Array, "num_timesteps state_dim"]
    Q: Float[Array, "state_dim state_dim"]
    Cs: Float[Array, "num_timesteps emission_dim state_dim"]
    R: Float[Array, "emission_dim emission_dim"]

# %%
class wGPLDS():
    '''
    GPLDS with weight-space view parametrization of the parameter priors
    '''
    def __init__(self, wgps: dict, state_dim: int, emission_dim: int):
        #! Add other dynamics and emission models if we want to sample from it
        self.wgps = wgps
        assert 'A' in self.wgps, 'Dynamics GP prior is required'
        if 'b' not in self.wgps:
            self.wgps['b'] = None
        
        self.state_dim = state_dim
        self.emission_dim = emission_dim

    def log_prior(self, params: ParamswGPLDS, inputs):
        '''Compute the log prior of the parameters. Conditions are inputs'''
        F = self.wgps['A'](params.dynamics_gp_weights, inputs)
        logprior_A = self.wgps['A'].log_prob(inputs[:len(F)], F).sum()

        if self.wgps['b'] is not None:
            logprior_b = self.wgps['b'].log_prob(inputs[:len(params.bs)], params.bs).sum()
        else:
            logprior_b = 0.
        return logprior_A + logprior_b

    def weights_to_params(self, params, inputs):
        '''Transform weights of weight space into parameters. Implement as needed for all weight-space GP priors.'''
        As = self.wgps['A'](params.dynamics_gp_weights, inputs)
        return As

    def smoother(self, params: ParamswGPLDS, emissions, inputs):
        '''inputs as conditions'''
        As = self.weights_to_params(params, inputs)
        lgssm_params = {
            'm0': params.m0,
            'S0': params.S0,
            'As': As,
            'bs': params.bs,
            'Q': params.Q,
            'Cs': params.Cs,
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
            Provide the sufficient stats X^T X and X^T Y for the problem Y = AX + noise.
            This returns the expanded stats Phi @ X^T X @ Phi^T and Phi @ X^T Y for the basis functions Phi.
            '''
            _Phi = wgp_prior.evaluate_basis(conditions)

            ZTZ = jnp.einsum('tk,tij,tl->ikjl', _Phi, XTX, _Phi)
            ZTY = jnp.einsum('tk,tim->ikm', _Phi, XTY)

            ZTZ = ZTZ.reshape(len(wgp_prior.basis_funcs) * wgp_prior.D2, len(wgp_prior.basis_funcs) * wgp_prior.D2)
            ZTY = ZTY.reshape(len(wgp_prior.basis_funcs) * wgp_prior.D2, wgp_prior.D1)
            return (ZTZ, ZTY)

        '''take inputs to be theta'''
        len_basis = len(self.wgps['A'].basis_funcs)
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

        # expected sufficient statistics for the dynamics
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        sum_xpxnT = Expxn.sum(0)
        sum_xpxpT = Vxp.sum(0) + Exp.T @ Exp
        sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
        dynamics_stats = (sum_xpxpT, sum_xpxnT, sum_xnxnT, num_timesteps - 1)

        # Dynamics wGP sufficient stats
        # PhiAp = self.wgps['A'].evaluate_basis(inputs)
        # partial E-step: assuming delta posterior around mean, hence a lin reg from the smoothed means only
        # _Z = jnp.einsum('tk,ti->tik', PhiAp, Exp)
        # _Y = Exn - params.bs[:len(inputs)]
        # _ZTZ = jnp.einsum('tik,tjl->ikjl', _Z, _Z).reshape(len(self.wgps['A'].basis_funcs) * self.wgps['A'].D2, len(self.wgps['A'].basis_funcs) * self.wgps['A'].D2)
        # _ZTY = jnp.einsum('tik,tj->ikj', _Z, _Y).reshape(len(self.wgps['A'].basis_funcs) * self.wgps['A'].D2, self.wgps['A'].D1)

        # full E-step sufficient stats
        Expxn_b = Expxn - jnp.einsum('ti,tj->tij', Exp, params.bs[:len(up)])
        ExpxpT = jnp.einsum('ti,tj->tij', Exp, Exp) + Vxp
        wgpA_stats = weightspace_stats(ExpxpT, Expxn_b, self.wgps['A'], up)

        # Q sufficient stats
        # Vxn, Vxp = Vx[1:], Vx[:-1]
        # sum_AExpxnT = jnp.einsum('tij,tjk->ik', F, Expxn) #.sum(0)
        # sum_AExpxpAT = jax.vmap(lambda _m, _S, _A: _A @ (_m @ _m.T + _S) @ _A.T)(Exp, Vxp, F).sum(0)
        # Q = (sum_xnxnT - _A_ExpxnT - _A_ExpxnT.T + _A_ExpxpT_A) / (dynamics_stats[-1] - 1) #! Need to check As to match A_t x_{t-1}
        
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

        _xxT = jnp.einsum('ti,tj->tij', Ex, Ex) + Vx
        _xyT = jnp.einsum('ti,tj->tij', Ex, y)
        # wgpC_stats = weightspace_stats(_xxT, _xyT, self.wgps['C'], inputs)

        return (init_stats, dynamics_stats, wgpA_stats, bias_stats, emission_stats), marginal_loglik
    
    def m_step(
            self,
            params: ParamswGPLDS,
            batch_stats: Tuple,
            inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None,
        ) -> ParamswGPLDS:
        def fit_linear_regression(ExxT, ExyT, EyyT, N):
            # Solve a linear regression given sufficient statistics
            W = utils.psd_solve(ExxT, ExyT).T
            Sigma = (EyyT - W @ ExyT - ExyT.T @ W.T + W @ ExxT @ W.T) / N
            return W, Sigma

        def fit_gplinear_regression(ZTZ, ZTY, wgp_prior, conditions):
            # Solve a linear regression in weight-space given sufficient statistics
            weights = jax.scipy.linalg.solve(
                ZTZ + jnp.eye(len(wgp_prior.basis_funcs) * wgp_prior.D2), ZTY, 
                assume_a='pos'
                )
            weights = weights.reshape(wgp_prior.D2, len(wgp_prior.basis_funcs), wgp_prior.D1).transpose(1,2,0)

            # Evaluate the weighted basis functions to obtain the parameters
            vals = wgp_prior(weights, conditions)
            return weights, vals

        up = inputs[:-1]

        # Sum the statistics across all batches
        stats = jax.tree_util.tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, wgpA_stats, bias_stats, emission_stats = stats

        # Perform MLE estimation jointly
        sum_x0, sum_x0x0T, N = init_stats
        S = sum_x0x0T / N - jnp.outer(sum_x0, sum_x0) / (N**2) # changed to N**2 because of outer.
        m = sum_x0 / N

        # Dynamics M-step
        W, F = fit_gplinear_regression(*wgpA_stats, self.wgps['A'], up)

        # Vxn, Vxp = Vx[1:], Vx[:-1]
        # _ExnxnT = Exn.T @ Exn + Vxn.sum(0) # Unchanged
        # _A_ExpxnT = jnp.einsum('tij,tjk->ik', F, Expxn) #.sum(0)
        # _A_ExpxpT_A = jax.vmap(lambda _m, _S, _A: _A @ (_m @ _m.T + _S) @ _A.T)(Exp, Vxp, F).sum(0)
        # Q = (_ExnxnT - _A_ExpxnT - _A_ExpxnT.T + _A_ExpxpT_A) / (dynamics_stats[-1] - 1) #! Need to check As to match A_t x_{t-1}

        F_static, Q = fit_linear_regression(*dynamics_stats)

        # Bias update
        if self.wgps['b'] is None:
            # bs = Exn/bias_stats[-1] - jnp.einsum('tij,tj->ti', F, Exp/bias_stats[-1])
            bs = bias_stats[0]/bias_stats[1]
        else: 
            # In weight space
            _, bs_reconstructed = fit_gplinear_regression(*bias_stats, self.wgps['b'], up)
            bs = bs_reconstructed.squeeze()
        # The following for homogeneous bias
        # b = jnp.mean(bs, axis=0)
        # bs = jnp.tile(b, (len(up), 1))

        # Emission M-step
        H, R = fit_linear_regression(*emission_stats)
        H = jxr.normal(jxr.PRNGKey(0), shape=(self.emission_dim, self.state_dim))
        Cs = jnp.tile(H, (len(inputs), 1, 1)) # Repeat the same emission matrix for all time steps

        params = ParamswGPLDS(
            m0=m, S0=S, dynamics_gp_weights=W, bs=bs, Q=Q, Cs=Cs, R=R,
        )
        return params

    def fit_em(
            self,
            params: ParamswGPLDS,
            emissions: Float[Array, "num_batches num_timesteps emission_dim"],
            inputs: Optional[Float[Array, "num_batches num_timesteps input_dim"]]=None,
            num_iters: int=50,
        ):
        assert emissions.ndim == 3, 'emissions should be 3D'

        @jit
        def em_step(params):
            # Obtain current E-step stats and joint log prob
            batch_stats, lls = vmap(partial(self.e_step, params))(emissions, inputs)
            log_prior = self.log_prior(params, inputs[0])
            mll = lls.sum()
            lp = log_prior + mll

            # Update with M-step
            params = self.m_step(params, batch_stats, inputs[0])
            return params, (lp, mll)

        log_probs = []
        for i in range(num_iters):
            params, (log_prob, marginal_log_lik) = em_step(params)
            log_probs.append(log_prob)
            print(f'Iter {i+1}/{num_iters}, log-prob = {log_prob:.2f}, marginal log-lik = {marginal_log_lik:.2f}')
        return params, jnp.array(log_probs)

# %%