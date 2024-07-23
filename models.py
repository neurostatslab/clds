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
            xs: Float[Array, "T"]
        ) -> Float[Array, "T D1 D2"]:
        '''
        Evaluate A_ij(x) = \sum_k w^{(ij)} \phi_k(x) at the points `xs`
        with `weights` w^{(ij)} and basis functions \phi_k.
        '''
        PhiX = self.evaluate_basis(xs)
        # print(weights, PhiX)
        return jnp.einsum('kij,tk->tij', weights, PhiX)
    
    def weights(self, key: jxr.PRNGKey) -> Float[Array, "len_basis D1 D2"]:
        '''retrieve weights associated with key'''
        return jxr.normal(key, shape=(len(self.basis_funcs), self.D1, self.D2))
    
    def evaluate_basis(self, x: Float[Array, "T"]) -> Float[Array, "T len_basis"]:
        '''Evaluate the basis functions at the array x'''
        return jnp.array([f(x) for f in self.basis_funcs]).T

    def sample(self, key: jxr.PRNGKey, xs: Float[Array, "T"]) -> Float[Array, "T D1 D2"]:
        '''Sample from the GP prior at the points `xs`'''
        weights = self.weights(key)
        PhiX = self.evaluate_basis(xs)
        return self.__call__(weights, xs)
    
    def log_prob(self, xs: Float[Array, "T"], fs: Float[Array, "T D1 D2"]) -> Float[Array, "D1 D2"]:
        '''
        Compute the log probability of the GP draws at the time points ts
        Args:
            ts: Array, (T,)
            fs: Array, (T, D1, D2)
        Returns:
            log_prob: Array, (D1, D2)
        '''
        assert fs.shape[1] == self.D1 and fs.shape[2] == self.D2, 'Dimension mismatch'
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

# params = ParamswGPLDS(
#     m0=jnp.array([0.0, 0.0]),
#     S0=jnp.eye(2),
#     dynamics_gp_weights=jnp.zeros((2, 2, 2)),
#     bs=jnp.zeros((3, 2)),
#     Q=jnp.eye(2),
#     Cs=jnp.zeros((3, 2, 2)),
#     R=jnp.eye(2),
# )
# print(params.dynamics_gp_weights)

# %%
class wGPLDS():
    '''
    GPLDS with weight-space view parametrization of the parameter priors
    '''
    def __init__(self, wgps, state_dim: int, emission_dim: int):
        #! Add other dynamics and emission models if we want to sample from it
        self.wgps = wgps
        self.state_dim = state_dim
        self.emission_dim = emission_dim

    def log_prior(self, params: ParamswGPLDS):
        # return self.wgps['A'].log_prob(params.dynamics_gp_weights).sum()
        raise NotImplementedError

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
        inputs: Optional[Float[Array, "num_timesteps-1 input_dim"]]=None,
    ):
        '''take inputs to be theta'''
        len_basis = len(self.wgps['A'].basis_funcs)
        num_timesteps = emissions.shape[0]
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 1))

        # Run the smoother to get posterior expectations
        # jax.debug.print('params = {}', params)
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
        up = inputs#[:-1]
        # u = inputs

        # expected sufficient statistics for the initial distribution
        Ex0 = smoothed_means[0]
        Ex0x0T = smoothed_covariances[0] + jnp.outer(Ex0, Ex0)
        init_stats = (Ex0, Ex0x0T, 1)

        # expected sufficient statistics for the dynamics
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        # sum_zpzpT = jnp.block([[Exp.T @ Exp, Exp.T @ up], [up.T @ Exp, up.T @ up]])
        # sum_zpzpT = sum_zpzpT.at[:self.state_dim, :self.state_dim].add(Vxp.sum(0))
        # sum_zpxnT = jnp.block([[Expxn.sum(0)], [up.T @ Exn]])
        sum_xpxnT = Expxn.sum(0)
        sum_xpxpT = Vxp.sum(0) + Exp.T @ Exp
        sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
        dynamics_stats = (sum_xpxpT, sum_xpxnT, sum_xnxnT, num_timesteps - 1)

        PhiAp = self.wgps['A'].evaluate_basis(up)
        # print('E-step: PhiAp = {}'.format(PhiAp))
        # print('E-step: Ex = {}'.format(Ex))

        # _t1 = jnp.einsum('ti,tj->tij', Exp, Exp)
        # _t1 = _t1 + Vxp  # shape tij
        # _t2 = jnp.einsum('tk,tij->kij', PhiAp, _t1)

        # def _outer(Phi, m, S):
        #     '''Phi shape (K,), m shape (D,), S shape (D, D)'''
        #     __t1 = m @ m.T + S # shape (D, D)
        #     __t2 = jnp.einsum('k,ij->kij', Phi, __t1)
        #     __t3 = jnp.einsum('k,tij->tijk', Phi, __t2)
        #     return __t3

        PhiAp_Exp = jnp.einsum('tk,tj->tkj', PhiAp, Exp).reshape(-1, len(self.wgps['A'].basis_funcs) * self.state_dim)
        # # print('E-step: PhiAp_Exp = {}'.format(PhiAp_Exp.sum()))
        
        # # _t2 = jax.vmap(_outer)(PhiAp, Exp, Vxp).sum(0).reshape(len(self.wgps['A'].basis_funcs) * self.state_dim, len(self.wgps['A'].basis_funcs) * self.state_dim)        
        # PhiAp_Vxp = jnp.einsum('tk,tij->tkij', PhiAp, Vxp) #.reshape(-1, len(self.wgps['A'].basis_funcs) * self.state_dim, self.state_dim)
        # PhiAp_Vxp_PhiAp = jnp.einsum('tkij,tl->kijl', PhiAp_Vxp, PhiAp).reshape(len(self.wgps['A'].basis_funcs) * self.state_dim, len(self.wgps['A'].basis_funcs) * self.state_dim)
        # _t2 = PhiAp_Exp.T @ PhiAp_Exp + PhiAp_Vxp_PhiAp
        _t3 = jnp.einsum('tk,tij->kij', PhiAp, Expxn).reshape(len(self.wgps['A'].basis_funcs) * self.state_dim, self.state_dim) # X^T Y term

        # # # Parametrize _t2 directly in terms of its U U^T form, with U = PhiAp * (m + L)
        # _t2 = jnp.zeros((len(self.wgps['A'].basis_funcs) * self.state_dim, len(self.wgps['A'].basis_funcs) * self.state_dim))
        # for t in range(len(up)):
        #     Lxp_t = jnp.linalg.cholesky(Vxp[t])
        #     PhiAp_t = PhiAp[t]
        #     PhiAp_Lxp_t = jnp.multiply(PhiAp_t[:, jnp.newaxis, jnp.newaxis], Lxp_t[jnp.newaxis, :, :])
        #     _t2 += jnp.outer(PhiAp_Lxp_t, PhiAp_Lxp_t)

        # _PhiAp_Exp = jnp.einsum('tk,tj->tkj', PhiAp, Exp)
        # _PhiAp_Lxp = jnp.einsum('tk,tij->tkij', PhiAp, Lxp)#.reshape(len(inputs), -1)
        # _t2 = jnp.einsum('tki,tlj->kilj', _PhiAp_Exp, _PhiAp_Exp) + jnp.einsum('tkid,tldj->kilj', _PhiAp_Lxp, _PhiAp_Lxp)


        _Z = jnp.einsum('tk,ti->tik', PhiAp, Exp)
        _ZTZ = jnp.einsum('tki,tlj->kilj', _Z, _Z).reshape(len(self.wgps['A'].basis_funcs) * self.state_dim, len(self.wgps['A'].basis_funcs) * self.state_dim)
        _Y = Exn - params.bs[:len(inputs)]
        _ZTY = jnp.einsum('tki,tj->kij', _Z, _Y).reshape(len(self.wgps['A'].basis_funcs) * self.state_dim, self.state_dim)
        # _PhiApb_Exp = _PhiAp_Exp + params.bs[:len(inputs)][:, None, :]
        # # Lxp = jax.vmap(lambda _V: jnp.linalg.cholesky(_V))(Vxp)
        # # _PhiAp_Lxp = jnp.einsum('tk,tij->tkij', PhiAp, Lxp)
        # _t2 = jnp.einsum('tki,tlj->kilj', _PhiAp_Exp, _PhiAp_Exp) #+ \
        #         # jnp.einsum('tkid,tljd->kilj', _PhiAp_Lxp, _PhiAp_Lxp)
        # _t2 = _t2.reshape(len(self.wgps['A'].basis_funcs) * self.state_dim, len(self.wgps['A'].basis_funcs) * self.state_dim)
        # _t2 = PhiAp_Exp.T @ PhiAp_Exp + PhiAp_Lxp.T @ PhiAp_Lxp
        # _t2 = __t2 @ __t2.T
        
        # bias sufficient stats
        F = self.wgps['A'](params.dynamics_gp_weights, inputs)
        b_targets = Exn - jnp.einsum('tij,tj->ti', F, Exp) # shape (T-1, state_dim)
        b_regressors = self.wgps['b'].evaluate_basis(inputs) # shape (T-1, len_basis)

        # more expected sufficient statistics for the emissions
        y = emissions
        sum_xxT = Vx.sum(0) + Ex.T @ Ex
        sum_xyT = Ex.T @ y
        sum_yyT = emissions.T @ emissions
        emission_stats = (sum_xxT, sum_xyT, sum_yyT, num_timesteps)

        return (init_stats, dynamics_stats, (PhiAp, PhiAp_Exp, Exp, Exn), (b_targets, b_regressors), (_ZTZ, _ZTY), emission_stats), marginal_loglik
    
    def m_step(
            self,
            params: ParamswGPLDS,
            batch_stats: Tuple,
            inputs: Optional[Float[Array, "num_timesteps-1 input_dim"]]=None,
        ) -> ParamswGPLDS:
        def fit_linear_regression(ExxT, ExyT, EyyT, N):
            # Solve a linear regression given sufficient statistics
            W = utils.psd_solve(ExxT, ExyT).T
            Sigma = (EyyT - W @ ExyT - ExyT.T @ W.T + W @ ExxT @ W.T) / N
            return W, Sigma
        
        K_A = len(self.wgps['A'].basis_funcs)

        # Sum the statistics across all batches
        # print(batch_stats[2][0].shape)
        stats = jax.tree_util.tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, (PhiAp, PhiAp_Exp, Exp, Exn),  (b_targets, b_regressors),  (_ZTZ, _ZTY), emission_stats = stats
        # print('M-step: PhiAp_Exp = {}'.format(PhiAp_Exp.sum(0)))

        # Perform MLE estimation jointly
        sum_x0, sum_x0x0T, N = init_stats
        S = sum_x0x0T / N - jnp.outer(sum_x0, sum_x0) / (N**2) # changed to N**2 because of outer.
        m = sum_x0 / N

        # Dynamics M-step
        # PhiAp = self.wgps['A'].evaluate_basis(inputs[:-1])
        # print(PhiAp.shape, inputs.shape)
        # PhiAp_Exp = jnp.einsum('tk,tj->tkj', PhiAp, Exp).reshape(-1, len(self.wgps['A'].basis_funcs) * self.state_dim)
        # W = jax.scipy.linalg.solve(PhiAp_Exp.T @ PhiAp_Exp + jnp.eye(PhiAp_Exp.shape[1]), PhiAp_Exp.T @ Exn).reshape(K_A, self.state_dim, self.state_dim)

        def is_symmetric(X):
            return jnp.allclose(X, X.T)

        # W = jax.scipy.linalg.solve(_t2 + jnp.eye(_t2.shape[0]), _t3, assume_a='pos')
        # # W = W.reshape(K_A, self.state_dim, self.state_dim)
        # W = W.reshape(self.state_dim, K_A, self.state_dim).transpose(1, 0, 2)

        W = jax.scipy.linalg.solve(_ZTZ + jnp.eye(K_A * self.state_dim), _ZTY, assume_a='pos')
        W = W.reshape(self.wgps['A'].D2, len(self.wgps['A'].basis_funcs), self.wgps['A'].D1).transpose(1,2,0)
        # jax.debug.print('W = {}', W)
        # jax.debug.print('safe_wrap(W) = {}', safe_wrap(W))

        # len_basis = len(self.wgps['A'].basis_funcs)
        # _Phi = self.wgps['A'].evaluate_basis(inputs).T
        # Phi_Exp = jnp.multiply(_Phi[:, :, jnp.newaxis], Exp) # shape (len_basis, num_timesteps, latent_dim)
        # _X = Phi_Exp.transpose(1, 0, 2).reshape(-1, K_A * self.state_dim)
        # _Y = Exn
        # W = jax.scipy.linalg.solve(_X.T @ _X + jnp.eye(_X.T.shape[0]), _X.T @ _Y).reshape(K_A, self.state_dim, self.state_dim)

        F = self.wgps['A'](W, inputs)
        # jax.debug.print('F = {}', F)
        # jax.debug.print('F = {}', F)
        # F = jnp.einsum('tk, kji->tij', PhiAp, W)
        # jax.debug.print('F dets: {}', [jnp.linalg.det(_F).item() for _F in F])

        # Vxn, Vxp = Vx[1:], Vx[:-1]
        # Exp = Ex[:-1]
        # Fp = F[:-1]
        # _ExnxnT = Exn.T @ Exn + Vxn.sum(0) # Unchanged
        # _A_ExpxnT = jnp.einsum('tij,tjk->ik', Fp, Expxn) #.sum(0)
        # _A_ExpxpT_A = jax.vmap(lambda _m, _S, _A: _A @ (_m @ _m.T + _S) @ _A.T)(Exp, Vxp, Fp).sum(0)
        # Q = (_ExnxnT - _A_ExpxnT - _A_ExpxnT.T + _A_ExpxpT_A) / (num_timesteps - 1) #! Need to check As to match A_t x_{t-1}

        # F_static = FB[:, :self.state_dim]
        # b = Exn.mean(0) - F.mean(0) @ Exp.mean(0)
        # bs = Exn - jnp.einsum('tij,tj->ti', F, Exp) 
        F_static, Q = fit_linear_regression(*dynamics_stats)
        # F = jnp.tile(F_static, (Exn.shape[0], 1, 1))

        # Bias update in weight space
        # b_targets = Exn - jnp.einsum('tij,tj->ti', F, Exp) # shape (T, state_dim)
        # b_regressors = self.wgps['b'].evaluate_basis(inputs[:-1]) # shape (T, len_basis)
        # b_weights = jax.scipy.linalg.solve(b_regressors.T @ b_regressors + jnp.eye(b_regressors.shape[1]), b_regressors.T @ b_targets)
        # b_weights = b_weights.reshape(self.wgps['b'].D2, len(self.wgps['b'].basis_funcs), self.wgps['b'].D1).transpose(1,2,0)
        # # bs = jnp.einsum('tk,kj->tj', b_regressors, b_weights)
        # bs = self.wgps['b'](b_weights, inputs).squeeze()

        # bs = jnp.tile(b, (Exn.shape[0], 1))
        # jax.debug.print('bs = {}', bs)
        # jax.debug.print('expected bias = {}', Exn.mean(0) - F.mean(0) @ Exp.mean(0))

        # B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self.has_dynamics_bias \
        #     else (FB[:, self.state_dim:], None)

        H, R = fit_linear_regression(*emission_stats)
        # H = HD[:, :self.state_dim]

        # # Fix all others to the true values
        # H = jxr.normal(jxr.PRNGKey(0), shape=(n_neurons, latent_dim))
        Q = 0.1 * jnp.eye(self.state_dim)
        R = 0.1 * jnp.eye(self.emission_dim)
        # m = jnp.zeros(self.state_dim)
        # S = jnp.eye(self.state_dim)
        bs = jnp.ones((len(inputs), self.state_dim))
        # B = jnp.zeros((self.state_dim, self.input_dim))
        # d = jnp.zeros(self.emission_dim)
        # D = jnp.zeros((self.emission_dim, self.input_dim))
        # # R = (noise_scale **2) * jnp.eye(self.emission_dim)

        # D, d = (HD[:, self.state_dimA_:-1], HD[:, -1]) if self.has_emissions_bias \
        #     else (HD[:, self.state_dim:], None)

        Cs = jnp.tile(H, (len(inputs)+1, 1, 1)) # Repeat the same emission matrix for all time steps
        # print(W.shape, Cs.shape, bs.shape)
        params = ParamswGPLDS(
            m0=m, S0=S, dynamics_gp_weights=W, bs=bs, Q=Q, Cs=Cs, R=R,
        )
        # jax.debug.print('params = {}', params)
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
            mll = lls.sum() #! add log_prior term

            # Update with M-step
            params = self.m_step(params, batch_stats, inputs[0])
            return params, mll

        log_probs = []
        for i in range(num_iters):
            params, marginal_log_lik = em_step(params)
            log_probs.append(marginal_log_lik)
            print(f'Iter {i+1}/{num_iters}, marginal log-likelihood = {marginal_log_lik:.2f}')
        return params, jnp.array(log_probs)

# %%