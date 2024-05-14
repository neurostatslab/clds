# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import jax.numpy as jnp
import numpy as np

import models
import jax
import utils

from scipy.io import loadmat


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

        ts = params['dt']*np.arange(1,steps).astype(float)
        ts = jnp.tile(ts,(ntrials,1))

        self.data = utils.split_data_cv(
            {'y':y,'x':x,'b':b,'ts':ts},
            params['props'],
            params['seeds']
        )


# %%
class GPLDSLoader(Dataset):
    def __init__(self,params):
        seed = params['seed']
        key = jax.random.PRNGKey(seed)

        initial = models.InitialCondition(
            params['D'],
            scale_tril=params['init_noise']*jnp.eye(params['D'])
        )

        lds = models.TimeVarLDS(
            D=params['D'],initial=initial,
        )

        k1,key = jax.random.split(key,2)

        emission = models.LinearEmission(
            key=k1,D=params['D'],N=params['N'],
            C=ortho_group.rvs(
                params['N'],
                random_state=np.random.seed(params['seed'])
            )[:,:params['D']],
            d=jnp.zeros(params['N'])
        )

        kernel_A = utils.get_kernel(params['kernel_A'],params['kernel_A_diag'])
        kernel_b = utils.get_kernel(params['kernel_b'],params['kernel_b_diag'])
        kernel_L = utils.get_kernel(params['kernel_L'],params['kernel_L_diag'])

        gps = {
            'A': models.GaussianProcess(kernel_A,params['D'],params['D']),
            'b': models.GaussianProcess(kernel_b,params['D'],1),
            'L': models.GaussianProcess(kernel_L,params['D'],params['D'])
        }

        likelihood = models.NormalConditionalLikelihood(
            params['N'],
            scale_tril=params['obs_noise']*jnp.eye(params['N'])
        )
        joint = models.GPLDS(
            gps,
            lds,
            emission,
            likelihood
        )

        
        A,b,L = [],[],[]
        x,y = [],[]
        ts = []
        for n in range(params['n_samples']):
            # k1,key = jax.random.split(key,2)
            # init_theta = jax.random.uniform(k1, minval=0, maxval=2*np.pi)
            # k1,key = jax.random.split(key,2)
            # d_thetas = jax.random.normal(k1,shape=(params['T'],)) * params['theta_drift_scale']
            # thetas = init_theta + jnp.cumsum(d_thetas)
            thetas = jnp.linspace(0,2*jnp.pi,params['T']-1)
            ts.append(thetas)

            As = jnp.eye(params['D'])[...,None] - params['theta_drift_scale']*np.array([
                [np.cos(thetas), -np.sin(thetas)],
                [np.sin(thetas), np.cos(thetas)]
            ])
            bs = .1*jnp.ones((params['D'],1,params['T']-1))

            
            # k1,key = jax.random.split(key,2)
            # bs = gps['bs'].sample(k1,thetas)
            k1,key = jax.random.split(key,2)
            Ls = gps['L'].sample(k1,thetas)

            A.append(As)
            b.append(bs)
            L.append(Ls)
            
            x_ = lds.sample(
                k1,
                As.transpose(2,0,1),
                bs.transpose(2,0,1),
                Ls.transpose(2,0,1),
            )
            x.append(x_)

            stats = emission.f(x_,emission.params)
            k1,key = jax.random.split(key,2)
            y_ = likelihood.sample(key=k1,stats=stats)
            y.append(y_)

        y = jnp.stack(y)
        # y = y-y.mean(0).mean(0)[None,None]
        
        x = jnp.stack(x)

        A = jnp.stack(A)
        b = jnp.stack(b)
        L = jnp.stack(L)
        ts = jnp.stack(ts)

        self.joint = joint

        self.data = utils.split_data_cv(
            {'y':y,'x':x,'b':b,'ts':ts,'As':A,'bs':b,'Ls':L},
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


        y = data['states'].transpose(1,0,2)
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
