# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import jax.numpy as jnp
import jax.random as jxr
import numpy as np

import jax
import utils

from scipy.io import loadmat
from functools import partial
from scipy.stats import ortho_group

# %%
class Dataset:
    def load_train_data(self):
        return self.data['y_train'], self.data['ts_train']

    def load_test_data(self):
        return self.data['y_test'], self.data['ts_test']

    def load_validation_data(self):
        return self.data['y_validation'], self.data['ts_test']

    def project(self,x,params):
        D = x.shape[-1]
        self.C = ortho_group.rvs(
            params['N'],
            random_state=np.random.seed(params['seed'])
        )[:,:D]

        y = jnp.einsum('nd,btd->btn',self.C,x)

        key = jax.random.PRNGKey(params['seed'])
        y += params['obs_noise']*jax.random.normal(key,y.shape)
        self.scale_tril = params['obs_noise']*jnp.eye(params['N'])

        return y


# %%
class Saddle(Dataset):
    def __init__(self,params):
        '''simple saddle model
        cohs (list or np.array) list of the constant input drives 
            (using -.1 and 0.1 as default)
        params (dict): dictionary of the parameters in the model (a,b,c) and system evolution
            (dt, euler timesep size), (ntrials, number of sample trials),
            (sigma, noise variance)
        time (int): number of 'units to run', time / dt is the number of steps
        '''

        a,b,c = params.get('a',-.6), params.get('b',2), params.get('c',-1)
        dt, time, cohs, ntrials, sigma = params['dt'], params['time'], params['cohs'], params['ntrials'], params['sigma'] 

        steps = int(time / dt)

        key = jax.random.PRNGKey(params['seed'])
        k1,key = jax.random.split(key,2)

        x = sigma*jax.random.normal(k1,shape=(len(cohs),ntrials,steps,2))
        
        cohs = np.array(cohs)
        cohs = np.repeat(cohs[:,np.newaxis],ntrials,axis=1)
        input_optimized = np.zeros((len(cohs),ntrials,steps,2))
        for i in range(1,steps):
            k1,key = jax.random.split(key,2)

            dx = a*x[:,:,i-1,0]**3 + b*x[:,:,i-1,0] + cohs
            dy = c*x[:,:,i-1,1] + cohs
            dx = np.concatenate([
                dx[:,:,np.newaxis],
                dy[:,:,np.newaxis]
                ],
                axis=2
            )
            input_optimized[:,:,i] = 0
            
            rand = sigma*jax.random.normal(
                k1,shape=(len(cohs),ntrials,2)
            )
            x = x.at[:,:,i].set(
                x[:,:,i-1] + dt*(dx+input_optimized[:,:,i]+rand)
            )
            
                
        cond_avg = np.mean(x,axis=1)

        x=x[0]
        b = x[:,1:]*0
        y = self.project(x,params)

        ts = params['dt']*np.arange(1,steps).astype(float)
        ts = jnp.tile(ts,(ntrials,1))

        self.data = utils.split_data_cv(
            {'y':y,'x':x,'b':b,'ts':ts},
            params['props'],
            params['seeds']
        )
        self.cond_avg = cond_avg

    

# %%
class LineAttractor(Dataset):
    def __init__(self,params):
        '''line attractor model
        '''
        params_saddle = params.copy()
        params_saddle['seed'] += 1
        cond_avgs = Saddle(params_saddle).cond_avg

        l0 = params.get('l0',[1,1])
        sigma = params.get('sigma',0)
        r0 = params.get('r0',[1,0])
        dt = params.get('dt',1)
        ntrials = params.get('ntrials',10)
        eval1 = params.get('eval1',-1)
        time = params['time']

        l0 /= np.linalg.norm(l0)
        evals = np.diag([0,eval1])
        
        r1 = l0
        R = np.array([r0,r1])
        L = np.linalg.inv(R)
        A = R @ evals @ L
        theta = np.radians(45)
        c, s = np.cos(theta), np.sin(theta)
        Mrot = np.array(((c, -s), (s, c)))
        A = Mrot @ A

        steps = int(time / dt)

        key = jax.random.PRNGKey(params['seed'])
        k1,key = jax.random.split(key,2)
        x = sigma*jax.random.normal(k1,shape=(cond_avgs.shape[0],ntrials,steps,2))
        input_optimized = np.zeros((cond_avgs.shape[0],ntrials,steps,2))

        for i in range(1,steps):
            k1,key = jax.random.split(key,2)
            dx = np.einsum('ij,mkj->mki',A,x[:,:,i-1])
            xavg = x[:,:,i-1].mean(axis=1)
            dx_avg = A @ xavg 
            inp = (1/dt) * (cond_avgs[:,i]-xavg-dx_avg*dt)
            input_optimized[:,:,i] = np.repeat(
                inp[:,np.newaxis],
                ntrials,
                axis=1
            )
            rand = sigma*jax.random.normal(
                k1,shape=(cond_avgs.shape[0],ntrials,2)
            )
            x = x.at[:,:,i].set(
                x[:,:,i-1] + dt*(dx+input_optimized[:,:,i]+rand)
            )

        
        self.A = A

        x = x[0]
        b = input_optimized[0][:,1:]*params['dt']
        y = self.project(x,params)

        ts = params['dt']*np.arange(1,steps).astype(float)
        ts = jnp.tile(ts,(ntrials,1))

        self.data = utils.split_data_cv(
            {'y':y,'x':x,'b':b,'ts':ts},
            params['props'],
            params['seeds']
        )

# %%
class PointAttractor(Dataset):
    def __init__(self,params):
        '''Point attractor model
        '''
        params_saddle = params.copy()
        params_saddle['seed'] += 1
        cond_avgs = Saddle(params_saddle).cond_avg
        time = params['time']
        
        a1 = params.get('a1',-0.5) # eigenvalue 1
        a2 = params.get('a2',-1) # eigenvalue 2

        sigma = params.get('sigma',0)
        dt = params.get('dt',1)
        ntrials = params.get('ntrials',10)

        # make sure they're negative
        A = np.diag([-np.abs(a1),-np.abs(a2)]) 
        
        steps = int(time / dt)
        
        key = jax.random.PRNGKey(params['seed'])
        k1,key = jax.random.split(key,2)
        x = sigma*jax.random.normal(k1,shape=(cond_avgs.shape[0],ntrials,steps,2))
        input_optimized = np.zeros((cond_avgs.shape[0],ntrials,steps,2))

        
        for i in range(1,steps):
            k1,key = jax.random.split(key,2)
            dx = np.einsum('ij,mkj->mki',A,x[:,:,i-1])
            xavg = x[:,:,i-1].mean(axis=1)
            dx_avg = A @ xavg
            inp = (1/dt) * (cond_avgs[:,i] - xavg - dx_avg * dt)
            input_optimized[:,:,i] = np.repeat(inp[:,np.newaxis],ntrials,axis=1)      
            rand = sigma*jax.random.normal(k1,shape=(cond_avgs.shape[0],ntrials,2))
            x = x.at[:,:,i].set(x[:,:,i-1] + dt*(dx+input_optimized[:,:,i]+rand))


        self.A = A

        x = x[0]
        b = input_optimized[0][:,1:]*params['dt']
        y = self.project(x,params)

        ts = params['dt']*np.arange(0,steps).astype(float)
        ts = jnp.tile(ts,(ntrials,1))

        self.data = utils.split_data_cv(
            {'y':y,'x':x,'b':b,'ts':ts},
            params['props'],
            params['seeds']
        )
        


# %%
class Affine(Dataset):
    def init(self,params):

        file,num_trials,sigma = params['file'],params.get('num_trials',200),params.get('sigma',.05)
        # Run simulation
        data = loadmat(file, squeeze_me=True)
        w,dt,tau,inp,t = data['w'], data['dt'], data['tau'], data['inp'], params['t']

        num_neurons = w.shape[0]

        # RNN update
        def r_update(w,r,u,sig): 
            return -r+r@w+u+sig*np.random.randn(w.shape[0])

        # parameters
        r = np.zeros((num_trials,2,len(t),num_neurons))
        u = np.zeros((num_trials,2,len(t)))
        for trial in range(num_trials):
            for k in range(2):
                r[trial,k,0] = sigma*np.random.randn(num_neurons)
                for i in range(len(t)-1):
                    u_ = 2*((k==1)-.5)*inp[k]*(t[i]>.2 and t[i]<.3) + 2*((k==1)-.5)*inp[k]*(t[i]>1.3 and t[i]<1.4)
                    r[trial,k,i+1] = r[trial,k,i] + (dt/tau)*r_update(w,r[trial,k,i],u_,sigma)
                    u[trial,k,i+1] = u_[0]
        data['sigma'] = data

        return r, u, data

# %%
class RNNData(Dataset):
    def __init__(self,params):
        '''Point attractor model
        '''
        data = jnp.load(
            params['file'],allow_pickle=True
        ).item()


        y = jnp.array(data['states'].transpose(1,0,2))
        # y = y-y.mean(0).mean(0)[None,None]
        pos = data['p'].transpose(1,0,2)[:,1:]
        vel = data['v'].transpose(1,0,2)[:,1:]
        theta = np.arccos(pos[...,0])

        self.grid = jnp.linspace(0,2*jnp.pi,y.shape[1]-1)[:,None]

        self.data = utils.split_data_cv(
            {'y':y,'ts':theta,'pos':pos,'vel':vel},
            params['props'],
            params['seeds']
        )

# %%
class HeadDirectionSimulation(Dataset):
    def __init__(self,params):
        # Constants
        n_neurons = params['N']
        latent_dim = params['D']
        
        # tuning curve peaks
        peaks = jnp.linspace(
            -params['tuning_peak'], 
            params['tuning_peak'], 
            n_neurons + 1
            )[:-1]
        
        # tuning curve widths
        widths = params['tuning_width'] * jnp.ones(n_neurons)  
        epsilon = params['epsilon']
        noise_scale = params['noise_scale']

        @partial(jax.vmap, in_axes=(0, 0))
        def dynamics(theta, omega):
            theta = (theta % (2 * jnp.pi)) - jnp.pi
            u = jnp.array([jnp.cos(theta), jnp.sin(theta)])
            v = jnp.array([-jnp.sin(theta), jnp.cos(theta)])
            A = (1 - epsilon) * v[:, None] * v[None, :]
            b = u + omega * v
            z = jnp.clip(theta - peaks, -jnp.pi * widths, jnp.pi * widths)
            C = (1 + jnp.cos(z / widths))[:, None] * (
                jnp.tile(u[None, :], (n_neurons, 1))
            )
            return (A, b, C)

        def initial_condition(theta, omega):
            m0 = jnp.array([jnp.cos(theta), jnp.sin(theta)])
            S0 = noise_scale * jnp.eye(2)
            return (m0, S0)

        def run_dynamics(key, As, bs, Cs, m0, S0):
            def f(x, inp):
                A, b, C, (subkey1, subkey2) = inp
                emission_noise = noise_scale * (
                    jxr.multivariate_normal(subkey2, jnp.zeros(n_neurons), jnp.eye(n_neurons))
                )
                y = C @ x + emission_noise

                dynamics_noise = noise_scale * (
                    jxr.multivariate_normal(subkey1, jnp.zeros(2), jnp.eye(2))
                )
                x_next = A @ x + b + dynamics_noise
                return x_next, (x_next, y)

            x_init = jxr.multivariate_normal(key, m0, S0)
            subkeys = jxr.split(key, num=(As.shape[0],2))
            _, (x_nexts, ys) = jax.lax.scan(f, x_init, xs=(As, bs, Cs, subkeys))
            
            xs = jnp.concatenate([x_init[None, :], x_nexts[:-1]], axis=0)
            return xs, ys


        # Sample heading direction with constant velocity
        num_timesteps = params['T']
        speed = params['speed']
        omega = jnp.ones(num_timesteps) * speed
        theta = jnp.cumsum(omega)


        # Generate time-varying linear dynamics and initial condition
        true_As, true_bs, true_Cs = dynamics(theta, omega)
        key = jxr.PRNGKey(params['seed'])
        key, k1 = jxr.split(key, 2)
        true_C = jxr.normal(k1, (n_neurons, 2)) # Overwrites neuron tuning
        true_Cs = jnp.tile(true_C, (num_timesteps, 1, 1))
        true_m0, true_S0 = initial_condition(theta[0], omega[0])

        # Run the dynamics to get a batch of data
        
        B = params['B']
        keys = jxr.split(key, B)
        x, y = jax.vmap(
            lambda i: run_dynamics(keys[i], true_As, true_bs, true_Cs, true_m0, true_S0)
        )(jnp.arange(B))

        ts = jnp.vstack((omega,theta)).T[None].repeat(B,0)
        As = true_As[None].repeat(B,0)
        bs = true_bs[None].repeat(B,0)
        Cs = true_Cs[None].repeat(B,0)
        
        self.data = utils.split_data_cv(
            {'y':y,'x':x,'ts':ts,'As':As,'bs':bs,'Cs':Cs},
            params['props'],
            params['seeds']
        ) 