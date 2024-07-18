# -*- coding: utf-8 -*-
"""
@author: Amin
"""

from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoDelta, AutoNormal

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro

from jax.example_libraries import stax
from jax.example_libraries import optimizers
from jax import jit, lax, value_and_grad, vmap
from jax.tree_util import register_pytree_node_class

from tqdm.auto import trange
import flax.linen as nn

from functools import reduce
from functools import partial

# %%
class VariationalFF:
    def __init__(self,key,input_dim,T,hidden_dim,out_shape,scale=1e-2):
        out_dim = reduce((lambda x, y: x * y), out_shape)

        self.shape = (T-1,)+out_shape

        init_mu, fn_mu =  stax.serial(
            stax.Dense(hidden_dim),
            stax.Tanh,
            stax.Dense(out_dim)
        )

        init_sg, fn_sg =  stax.serial(
            stax.Dense(hidden_dim),
            stax.Dense(out_dim),
            stax.Softplus
        )

        self.fn_mu = lambda par,ts: scale*fn_mu(par,ts)
        self.fn_sg = lambda par,ts: scale*fn_sg(par,ts)

        _, self.vars_mu = init_mu(key, (input_dim,))
        _, self.vars_sg = init_sg(key, (input_dim,))
        
    def set_params(self, params):
        self.vars_mu, self.vars_sg = params

    @property
    def params(self):
        return (self.vars_mu, self.vars_sg)

               
    def f(self,key,y,ts,params):
        vars_mu,vars_sg = params

        mu = self.fn_mu(vars_mu,ts[:,None]).reshape(self.shape)
        sg = self.fn_sg(vars_sg,ts[:,None]).reshape(self.shape)

        sample = dist.Normal(mu,sg).sample(key)
        lp = dist.Normal(mu,sg).log_prob(sample)

        return sample,lp.sum()
    
    def posterior_mean(self,y,ts,params):
        vars_mu = params[0]
        mu = self.fn_mu(vars_mu,ts[:,None]).reshape(self.shape)
        return mu
    
    def posterior_scale(self,y,ts,params):
        vars_sg = params[1]
        scale = self.fn_sg(vars_sg,ts[:,None]).reshape(self.shape)
        return scale

# %%
class VariationalLSTM:
    def __init__(self,shape):
        self.shape = shape
        self.fn_mu = [nn.softmax]
        self.fn_sg = [nn.softmax]
    
    def init(self,key,series,shape):
        vars = []
        for i in range(len(series)):
            if not hasattr(series[i],'init'):
                continue
            k1, key = jax.random.split(key,2)
            vars.append(series[i].init(
                k1, jnp.ones(shape[i])
            ))
        return tuple(vars)

    def apply(self,series,params,y):
        for i in range(len(series)):
            if hasattr(series[i],'apply'):
                y = series[i].apply(params[i],y)
            else:
                y = series[i](y) 
        return y

    def set_params(self,params):
        self.vars_mu = params[:2]
        self.vars_sg = params[2:]
    
    @property
    def params(self):
        return self.vars_mu + self.vars_sg
    
        
    def f(self,key,y,ts,params):
        vars_mu = params[:2]
        vars_sg = params[2:]

        mu = self.apply(self.fn_mu,vars_mu,y).reshape(self.shape)
        sg = self.apply(self.fn_sg,vars_sg,y).reshape(self.shape)

        sample = dist.Normal(mu,sg).sample(key)
        lp = dist.Normal(mu,sg).log_prob(sample)

        return sample,lp.sum()
    
    def posterior_mean(self,y,ts,params):
        vars_mu = params[:2]
        mu = self.apply(self.fn_mu,vars_mu,y).reshape(self.shape)
        return mu
    
    def posterior_scale(self,y,ts,params):
        vars_sg = params[2:]
        scale = self.apply(self.fn_sg,vars_sg,y).reshape(self.shape)
        return scale
    
# %%
class LocalLatent(VariationalLSTM):
    def __init__(self,key,input_dim,T,hidden_dim,out_shape,scale=1e-1):
        '''initialize an instance
        '''
        super(LocalLatent, self).__init__(shape=(1,T,)+out_shape)
        
        out_dim = reduce((lambda x, y: x * y), out_shape)

        self.fn_mu = [
            nn.RNN(nn.LSTMCell(hidden_dim)),
            nn.Dense(out_dim)
        ]

        self.fn_sg = [
            nn.RNN(nn.LSTMCell(hidden_dim)),
            nn.Dense(out_dim),
            lambda x: scale*nn.softplus(x)
        ]

        k1, key = jax.random.split(key,2)
        self.vars_mu = self.init(k1,self.fn_mu,[(1,T,input_dim),(1,T,hidden_dim)])

        k1, key = jax.random.split(k1,2)
        self.vars_sg = self.init(key,self.fn_sg,[(1,T,input_dim),(1,T,hidden_dim)])

# %%
class GlobalLatent(VariationalLSTM):
    def __init__(self,key,input_dim,T,hidden_dim,out_shape,scale=1e-2):
        '''initialize an instance
        '''
        super(GlobalLatent, self).__init__(shape=(T-1,)+out_shape)
        
        out_dim = reduce((lambda x, y: x * y), out_shape)

        self.fn_mu = [
            nn.RNN(nn.LSTMCell(hidden_dim)),
            nn.Dense(out_dim),
            lambda x: scale*x
        ]

        self.fn_sg = [
            nn.RNN(nn.LSTMCell(hidden_dim)),
            nn.Dense(out_dim),
            lambda x: scale*nn.softplus(x)
        ]

        k1, key = jax.random.split(key,2)
        self.vars_mu = self.init(k1,self.fn_mu,[(T,input_dim),(T,hidden_dim)])

        k1, key = jax.random.split(k1,2)
        self.vars_sg = self.init(key,self.fn_sg,[(T,input_dim),(T,hidden_dim)])

                
    def f(self,key,y,ts,params):
        vars_mu = params[:2]
        vars_sg = params[2:]

        mu = self.apply(self.fn_mu,vars_mu,ts[:,None]).reshape(self.shape)
        sg = self.apply(self.fn_sg,vars_sg,ts[:,None]).reshape(self.shape)

        sample = dist.Normal(mu,sg).sample(key)
        lp = dist.Normal(mu,sg).log_prob(sample)

        return sample,lp.sum()
    
    def posterior_mean(self,y,ts,params):
        vars_mu = params[:2]
        mu = self.apply(self.fn_mu,vars_mu,ts[:,None]).reshape(self.shape)
        return mu
    
    def posterior_scale(self,y,ts,params):
        vars_sg = params[2:]
        scale = self.apply(self.fn_sg,vars_sg,ts[:,None]).reshape(self.shape)
        return scale
    

# %%
class VariationalBasis:
    def __init__(self,key,basis_dim,T,out_shape,basis_fn,scale=1e-2):
        out_dim = reduce((lambda x, y: x * y), out_shape)
        self.shape = (T-1,)+out_shape

        self.mu_params = jnp.zeros((basis_dim,out_dim))
        self.sg_params = jnp.zeros((basis_dim,out_dim))

        self.fn_mu = lambda par, ts: scale*jnp.einsum('co,btc->bto',par,basis_fn(ts))
        self.fn_sg = lambda par, ts: scale*jax.nn.softplus(jnp.einsum('co,btc->bto',par,basis_fn(ts)))
        

    def set_params(self, params):
        self.mu_params, self.sg_params = params
    @property
    def params(self):
        return (self.mu_params, self.sg_params)
               
    def f(self,key,y,ts,params):
        vars_mu,vars_sg = params

        mu = self.fn_mu(vars_mu,ts[:,None]).reshape(self.shape)
        sg = self.fn_sg(vars_sg,ts[:,None]).reshape(self.shape)

        sample = dist.Normal(mu,sg).sample(key)
        lp = dist.Normal(mu,sg).log_prob(sample)

        return sample,lp.sum()
    
    def posterior_mean(self,y,ts,params):
        vars_mu = params[0]
        mu = self.fn_mu(vars_mu,ts[:,None]).reshape(self.shape)
        return mu
    
    def posterior_scale(self,y,ts,params):
        vars_sg = params[1]
        scale = self.fn_sg(vars_sg,ts[:,None]).reshape(self.shape)
        return scale
# %%
class BasisZ:
    def __init__(
            self,key,D,T,basis_fn,basis_dim=1,
            scale_A=1e0,scale_b=1e0,scale_L=1e-1,
        ):
        '''initialize an instance
        '''        
        self.D = D # dynamics dimension
        self.T = T # time dimension

        keys = jax.random.split(key,3)
        
        self.A = VariationalBasis(keys[0],basis_dim,T,(D,D),basis_fn,scale=scale_A)
        self.b = VariationalBasis(keys[1],basis_dim,T+1,(D,1),basis_fn,scale=scale_b)
        self.L = VariationalBasis(keys[2],basis_dim,T+1,(D,D),basis_fn,scale=scale_L)

    def set_params(self,params):
        self.A.set_params(params[:2])
        self.b.set_params(params[2:4])
        self.L.set_params(params[4:])
    
    @property
    def params(self):
        return self.A.params+self.b.params+self.L.params
    
    def f(self,key,y,ts,params):
        # TODO: Fix this
        keys = jax.random.split(key,3)
        A,lpA =  self.A.f(keys[1],y,ts[1:],params[:2])
        b,lpb =  self.b.f(keys[2],y,ts,params[2:4])
        L,lpL =  self.L.f(keys[3],y,ts,params[4:])
        return (A,b[...,0],L),lpA+lpb+lpL

    def posterior_mean(self,y,ts,params):
        # TODO: Fix this
        EA =  self.A.posterior_mean(y,ts[1:],params[:2])
        Eb =  self.b.posterior_mean(y,ts,params[2:4])
        EL =  self.L.posterior_mean(y,ts,params[4:])
        return (EA,Eb[...,0],EL)
    
    def posterior_scale(self,y,ts,params):
        # TODO: Fix this
        CA =  self.A.posterior_scale(y,ts[1:],params[:2])
        Cb =  self.b.posterior_scale(y,ts,params[2:4])
        CL =  self.L.posterior_scale(y,ts,params[4:])
        return (CA,Cb[...,0],CL)
    

# %%
class AmortizedFFZ:
    def __init__(
            self,key,D,T,C=1,H=10,
            scale_A=1e0,scale_b=1e0,scale_L=1e-1,
        ):
        '''initialize an instance
        '''        
        self.D = D # dynamics dimension
        self.T = T # time dimension

        keys = jax.random.split(key,3)
        
        self.A = VariationalFF(keys[0],C,T,H,(D,D),scale=scale_A)
        self.b = VariationalFF(keys[1],C,T,H,(D,1),scale=scale_b)
        self.L = VariationalFF(keys[2],C,T,H,(D,D),scale=scale_L)

    def set_params(self,params):
        self.A.set_params(params[:2])
        self.b.set_params(params[2:4])
        self.L.set_params(params[4:])
    
    @property
    def params(self):
        return self.A.params+self.b.params+self.L.params
    
    def f(self,key,y,ts,params):
        # TODO: Fix this
        # ts = jnp.stack((jnp.sin(ts),jnp.cos(ts))).T

        keys = jax.random.split(key,3)
        A,lpA =  self.A.f(keys[1],y,ts,params[:2])
        b,lpb =  self.b.f(keys[2],y,ts,params[2:4])
        L,lpL =  self.L.f(keys[3],y,ts,params[4:])
        return (A,b[...,0],L),lpA+lpb+lpL

    def posterior_mean(self,y,ts,params):
        # TODO: Fix this
        # ts = jnp.stack((jnp.sin(ts),jnp.cos(ts))).T

        EA =  self.A.posterior_mean(y,ts,params[:2])
        Eb =  self.b.posterior_mean(y,ts,params[2:4])
        EL =  self.L.posterior_mean(y,ts,params[4:])
        return (EA,Eb[...,0],EL)
    
    def posterior_scale(self,y,ts,params):
        # TODO: Fix this
        # ts = jnp.stack((jnp.sin(ts),jnp.cos(ts))).T

        CA =  self.A.posterior_scale(y,ts,params[:2])
        Cb =  self.b.posterior_scale(y,ts,params[2:4])
        CL =  self.L.posterior_scale(y,ts,params[4:])
        return (CA,Cb[...,0],CL)


# %%
class AmortizedZ:
    def __init__(
            self,key,D,T,C=1,H=10,
            scale_A=1e0,scale_b=1e0,scale_L=1e-1,
        ):
        '''initialize an instance
        '''        
        self.D = D # dynamics dimension
        self.T = T # time dimension

        keys = jax.random.split(key,3)
        
        self.A = GlobalLatent(keys[0],C,T,H,(D,D),scale=scale_A)
        self.b = GlobalLatent(keys[1],C,T,H,(D,1),scale=scale_b)
        self.L = GlobalLatent(keys[2],C,T,H,(D,D),scale=scale_L)

    def set_params(self,params):
        self.A.set_params(params[:4])
        self.b.set_params(params[4:8])
        self.L.set_params(params[8:])
    
    @property
    def params(self):
        return self.A.params+self.b.params+self.L.params
    
    def f(self,key,y,ts,params):
        keys = jax.random.split(key,3)
        A,lpA =  self.A.f(keys[1],y,ts,params[:4])
        b,lpb =  self.b.f(keys[2],y,ts,params[4:8])
        L,lpL =  self.L.f(keys[3],y,ts,params[8:])
        return (A,b[...,0],L),lpA+lpb+lpL

    def posterior_mean(self,y,ts,params):
        EA =  self.A.posterior_mean(y,ts,params[:4])
        Eb =  self.b.posterior_mean(y,ts,params[4:8])
        EL =  self.L.posterior_mean(y,ts,params[8:])
        return (EA,Eb[...,0],EL)
    
    def posterior_scale(self,y,ts,params):
        CA =  self.A.posterior_scale(y,ts,params[:4])
        Cb =  self.b.posterior_scale(y,ts,params[4:8])
        CL =  self.L.posterior_scale(y,ts,params[8:])
        return (CA,Cb[...,0],CL)




# %%
class DeltaZ:
    def __init__(
            self,key,D,T,scale_A=1.,scale_L=1e-2,
            A=None,b=None,L=None
        ):
        '''initialize an instance
        '''        
        self.D = D # dynamics dimension
        self.T = T # time dimension
        
        if A is None:
            self.A = scale_A*jnp.tile(jnp.eye(D), (T-1,1,1))
        else:
            self.A = A
        
        if b is None:
            self.b = jnp.zeros((T-1,D))
        else:
            self.b = b

        if L is None:
            self.L = scale_L*jnp.tile(jnp.eye(D), (T-1,1,1))
        else:
            self.L = L
        
        
    def set_params(self,params):
        self.A,self.b,self.L = params
    
    @property
    def params(self):
        return (self.A,self.b,self.L)
    
    
    def posterior_mean(self,y,t,z_params):
        return z_params


# %%
def infer(
        key,joint,phi_x,phi_z,y,ts,
        n_iter=10000,step_size=.1,gamma=1
    ):

    opt_init, opt_update, get_params = optimizers.adam(step_size)
    
    opt_state = opt_init(
        phi_x.params+phi_z.params+joint.params
    )

    lr = len(phi_x.params)+len(phi_z.params)
    lx = len(phi_x.params)
    
    params = get_params(opt_state)

    def elbo(params,y,ts,key):
        # variational sample and log prob  
        k1, key = jax.random.split(key,2)
        x,logx = phi_x.f(k1,y,ts,params[:lx])

        k1, key = jax.random.split(key,2)
        (A,b),logz = phi_z.f(k1,y,ts,params[lx:lr])
        
        log_prior = logx + logz

        # Compute joint log probability over many samples of
        log_joint = joint.log_prob(
            y,x[0],A,b,
            ts,params[lr:]
        )

        # ELBO is the expected joint log probability plus entropy
        return  -log_joint+gamma*log_prior, log_joint, log_prior

    def batch_elbo(params,y,ts,keys):
        fun = jax.vmap(
            lambda y,key: elbo(params, y, ts, key),
            in_axes=(0,0),
            out_axes=0
        )
        val = fun(y,keys)
        elbo_loss, log_joint, log_prior = jnp.array(val).mean(1)
        return elbo_loss, (log_joint, log_prior)
        

    @jit
    def update(params,y,ts,i,key,opt_state):
        ''' Perform a forward pass, calculate the MSE & perform a SGD step. '''
        (loss, (log_joint, log_prior)), grads = value_and_grad(
                batch_elbo,has_aux=True
            )(params,y,ts,key)
        
        opt_state = opt_update(i,grads,opt_state)
        params = get_params(opt_state)
        
        return loss, log_joint, log_prior, opt_state, params 


    pbar = trange(n_iter)
    pbar.set_description('jit compiling ...')
    
    
    losses = []

    for i in pbar:
        k1, key = jax.random.split(key,2)
        keys = jax.random.split(k1,y.shape[0])
        loss, log_joint, log_prior, opt_state, params = \
            update(params,y,ts,i,keys,opt_state)
        
        losses.append(loss)
        if i % 10 == 0:
            pbar.set_description('ELBO: {:.2f}, Log Joint: {:.2f}, Log Prior: {:.2f}'.format(loss, log_joint, log_prior))

        
    phi_x.set_params(params[:lx])
    phi_z.set_params(params[lx:lr])
    
    return losses


# %%
def infer_global(
        key,joint,phi_x,phi_z,y,ts,
        n_iter=10000,step_size=.1,gamma=1
    ):

    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(phi_z.params+joint.params)

    lr = len(phi_z.params)
    
    params = get_params(opt_state)

    def elbo(params,y,ts,key):
        # variational sample and log prob  
        k1, key = jax.random.split(key,2)
        x,logx = phi_x.f(k1,y,ts,phi_x.params)

        k1, key = jax.random.split(key,2)
        (A,b),logz = phi_z.f(k1,y,ts,params[:lr])
        
        log_prior = logx + logz

        # Compute joint log probability over many samples of
        # log_joint = joint.log_prob(
        #     A,b,x[0],
        #     params[lr:]
        # )

        log_joint = joint.log_prob(
            y,x[0],A,b,ts,
            params[lr:]
        )

        # ELBO is the expected joint log probability plus entropy
        return  -log_joint+gamma*log_prior, log_joint, log_prior

    def batch_elbo(params,y,ts,keys):
        fun = jax.vmap(
            lambda y,ts,key: elbo(params, y, ts, key),
            in_axes=(0,0,0),
            out_axes=0
        )
        val = fun(y,ts,keys)
        elbo_loss, log_joint, log_prior = jnp.array(val).mean(1)
        return elbo_loss, (log_joint, log_prior)
        

    @jit
    def update(params,y,ts,i,key,opt_state):
        ''' Perform a forward pass, calculate the MSE & perform a SGD step. '''
        (loss, (log_joint, log_prior)), grads = value_and_grad( #(loss, (log_joint, log_prior))
                batch_elbo,has_aux=True
            )(params, y, ts, key)
        
        opt_state = opt_update(i, grads, opt_state)
        params = get_params(opt_state)
        
        return loss, log_joint, log_prior, opt_state, params 


    pbar = trange(n_iter)
    pbar.set_description('jit compiling ...')
    
    ts = ts[None].repeat(y.shape[0],axis=0)
    losses = []

    for i in pbar:
        k1, key = jax.random.split(key,2)
        keys = jax.random.split(k1,y.shape[0])
        loss, log_joint, log_prior, opt_state, params = \
            update(params,y,ts,i,keys,opt_state)
        
        losses.append(loss)
        if i % 10 == 0:
            pbar.set_description('ELBO: {:.2f}, Log Joint: {:.2f}, Log Prior: {:.2f}'.format(loss, log_joint, log_prior))

        
    phi_z.set_params(params[:lr])
    joint.set_params(params[lr:])
    
    return losses


# %%
def map(
        joint,delta,ys,ts,
        n_iter=10000,step_size=.1,gamma=1
):
    D = joint.dynamics.D
    B,T,N = ys.shape


    opt_init, opt_update, get_params = optimizers.adam(
        step_size,b1=0.8,b2=0.9,
    )
    opt_state = opt_init(joint.params+delta.params)
    params = get_params(opt_state)

    def marginal_ll(params,y,t):
        
        ld,le,ll = len(joint.dynamics.params), len(joint.emissions.params), len(joint.likelihood.params)
        dynamics_params, emissions_params, likelihood_params = params[:ld], params[ld:ld+le], params[ld+le:ld+le+ll]
        
        z_params = params[ld+le+ll:]

        # Function for integrating out local latents (x)
        def scanned_func(carry,inputs):
            # Unpack carry from last iteration
            m_last, P_last = carry

            
            # scale_tril_y = joint.likelihood.scale_tril
            scale_tril_y = likelihood_params[0]
            C,d = emissions_params
            # C = joint.emissions.C
            
            A,b,L,y = inputs
            
            # Compute covariances.
            Q = L @ L.T
            R = scale_tril_y @ scale_tril_y.T

            # Prediction step (propogate dynamics, eq 4.20 in Sarkka)
            m_pred = A @ m_last + b
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
            return (m, P), log_prob_y


        
        # ld,lj = len(joint.dynamics.params), len(joint.params)
        # lds_params, z_params = params[:ld], params[lj:]
        
        A0 = jnp.eye(D)[None]
        b0 = jnp.zeros((D))[None]
        L0 = jnp.zeros((D,D))[None]
        
        (A,b,L) = delta.posterior_mean(y,t,z_params)

        A_ = jnp.concatenate((A0,A))
        b_ = jnp.concatenate((b0,b[1:]))
        L_ = jnp.concatenate((L0,L[1:]))


        log_marginal = jax.lax.scan(
                scanned_func, 
                (b[0],L[0]@L[0].T),
                (A_,b_,L_,y)
            )[1].sum()
        
        lpA = joint.gps['A'].log_prob(t[1:],A)
        lpb = joint.gps['b'].log_prob(t,b)
        lpL = joint.gps['L'].log_prob(t,L)

        # sum over time points
        log_prior = (lpA.sum()+lpb.sum()+lpL.sum())/B

        # pior needs to be counted once for the whole batch
        return  -(gamma*log_prior+log_marginal), log_marginal, log_prior

    def batch_loss(params,ys,ts):
        fun = jax.vmap(
            lambda y,t: marginal_ll(params,y,t),
            in_axes=(0,0),
            out_axes=0
        )
        val = fun(ys,ts)
        loss, log_marginal, log_prior = jnp.array(val).mean(1)
        
        return loss, (log_marginal, log_prior)
        

    @jit
    def update(params,ys,ts,i,opt_state):
        ''' Perform a forward pass, calculate the MSE & perform a SGD step. '''
        (loss, (log_joint, log_prior)), grads = value_and_grad(
                batch_loss,has_aux=True
            )(params,ys,ts)
        
        opt_state = opt_update(i, grads, opt_state)
        params = get_params(opt_state)
        
        return loss, log_joint, log_prior, opt_state, params 


    pbar = trange(n_iter)
    pbar.set_description('jit compiling ...')
    
    losses = []

    for i in pbar:
        loss, log_marginal, log_prior, opt_state, params = \
            update(params,ys,ts,i,opt_state)
        
        losses.append(loss)
        if i % 10 == 0:
            pbar.set_description('ELBO: {:.2f}, Log Marginal: {:.2f}, Log Prior: {:.2f}'.format(loss, log_marginal, log_prior))

    
    lj,lz = len(joint.params), len(delta.params)
    joint.set_params(params[:lj])
    delta.set_params(params[lj:])
    
    return losses