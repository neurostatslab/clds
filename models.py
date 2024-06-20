# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import numpy as onp
import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from jax import vmap

import jax.numpy as jnp
import jax
from jax import jit, lax, value_and_grad, vmap

from jax.example_libraries import optimizers

import numpyro.distributions as dist
import numpyro

from numpyro.infer.autoguide import AutoDelta, AutoNormal

from jax.example_libraries import stax
from jax.tree_util import register_pytree_node_class


    
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
            jnp.zeros(T),
            covariance_matrix=c_g
        ).sample(key,sample_shape=(self.D1,self.D2))
        return fs

    
    def log_prob(self,ts,fs):
        T = ts.shape[0]
        c_g = self.evaluate_kernel(ts,ts)
        lp = dist.MultivariateNormal(
            jnp.zeros(T),
            covariance_matrix=c_g
        ).log_prob(fs.reshape(T,-1).T)
        return lp
    

#  %%
class InitialCondition:
    def __init__(self,D,mu=None,scale_tril=None):
        self.D = D
        self.mu = jnp.zeros(D) if mu is None else mu
        self.scale_tril = jnp.eye(D) if scale_tril is None else scale_tril

    def set_params(self, params):
        self.mu,self.scale_tril = params
        # self.scale_tril
    @property
    def params(self):
        return (self.mu,self.scale_tril) #self.scale_tril
    
    def sample(self,key):
        mu,scale_tril = self.mu, self.scale_tril
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
        '''initialize an instance
        '''       
        # dynamics dimension
        self.D = D
        
        # initial condition distribution
        self.initial = initial

        # if scale_tril is None:
        #     self.scale_tril = self.initial.scale_tril
        # else:
        #     self.scale_tril = scale_tril

    def sample(self,key,As,bs,Ls,x0=None):
        D = self.D
        T = len(As)+1

        @jit
        def transition(carry, args):
            xs, k = carry
            A_new, b_new, L_new, _ = args
            k1, k = jax.random.split(k,2)
            mu = A_new@xs[-1]+b_new[:,0]
            x_new = dist.MultivariateNormal(
                mu,
                scale_tril=L_new #self.scale_tril
            ).sample(k1)
            xs = jnp.vstack((xs[1:],x_new))
            return (xs,k), None
        
        
        k1, key = jax.random.split(key,2)
        x0 = self.initial.sample(k1) if x0 is None else x0
        history = jnp.vstack((jnp.ones((T-1,D)),x0[None]))
        (xs,_),_ = lax.scan(
            transition, 
            (history,key), 
            (As,bs,Ls,jnp.arange(1,T))
        )
        
        return xs
    

    def set_params(self,params):
        self.initial.set_params(params[:2])
        # self.scale_tril = params[2]

    @property
    def params(self):
        return self.initial.params#+(self.scale_tril,)
    

    def log_prob(self,As,bs,Ls,xs,params):
        D = self.D
        T = len(As)+1
        init_params = params[:2]
        # scale_tril = params[2]

        @jit
        def transition(carry, args):
            xs,lp = carry
            A_new, b_new, L_new, x_new, _ = args
            mu = A_new@xs[-1]+b_new[:,0]
            lp += dist.MultivariateNormal(
                mu,
                scale_tril=L_new # scale_tril
            ).log_prob(x_new)
            xs = jnp.vstack((xs[1:],x_new))
            return (xs,lp), None
        

        
        history = jnp.vstack((jnp.ones((T-1,D)),xs[0][None]))
        (_,lp) , _ = lax.scan(
            transition, 
            (history,self.initial.log_prob(xs[0],init_params)), 
            (As,bs,Ls,xs[1:],jnp.arange(1,T))
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
        b_ = jnp.concatenate((b0,bs))
        L_ = jnp.concatenate((L0,Ls))

        _, (mus, sigmas) = jax.lax.scan(
                scanned_func, 
                (self.initial.params[0],self.initial.params[1]@self.initial.params[1].T),
                (A_,b_,L_)
            )
        return mus, sigmas
# %%
class PoissonConditionalLikelihood:
    def __init__(self,D):
        self.D = D
    
    def sample(self,key,stats):
        rate = stats
        Y = dist.Poisson(
            jax.nn.softplus(rate)
            ).to_event(1).sample(key)
        return Y
    
    def log_prob(self,stats,y,params):
        rate = stats
        return dist.Poisson(
            jax.nn.softplus(rate)
        ).to_event(1).log_prob(y)
    
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
        y = dist.MultivariateNormal(
            mu,scale_tril=scale_tril
        ).sample(key)
        return y
    
    def log_prob(self,stats,y,params):
        mu,scale_tril=stats,params[0]
        # mu,scale_tril=stats,self.scale_tril
        return dist.MultivariateNormal(
            mu,scale_tril=scale_tril
        ).log_prob(y)
    
    @property
    def params(self):
        # return ()
        return (self.scale_tril,)
    
    def set_params(self,params):
        # pass
        self.scale_tril = params[0]

# %%
class LinearEmission:
    def __init__(self,key,D,N,C=None,d=None):
        # D: dynamics dimension
        self.D = D
        
        # N: observation dimension
        self.N = N

        k1, k2 = jax.random.split(key,2)

        # initialize the weight matrix if not given
        if C is None:
            self.C =  dist.Normal(0,1/N).sample(
                k1,sample_shape=(self.N,self.D)
            )
        else:
            self.C = C
        
        if d is None:
            self.d = dist.Normal(0,1/N).sample(
                k2,sample_shape=(self.N,)
            )
        else:
            self.d = d

    def set_params(self, params):
        # pass
        self.C,self.D = params[0:2]
        
    @property
    def params(self):
        # return ()
        return (self.C,self.d)

    def f(self,x,params):
        # C,d = self.C, self.d
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

    def log_prob(self,y,x,As,bs,Ls,ts,params):
        ld,le,ll = len(self.dynamics.params), len(self.emissions.params), len(self.likelihood.params)
        dynamics_params, emissions_params, likelihood_params = params[:ld], params[ld:ld+le], params[ld+le:ld+le+ll]

        lpA = self.gps['A'].log_prob(ts,As)
        lpb = self.gps['b'].log_prob(ts,bs)
        lpL = self.gps['L'].log_prob(ts,Ls)

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
