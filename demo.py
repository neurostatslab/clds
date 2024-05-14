# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import models
import jax.numpy as jnp
import jax

import inference
import visualizations
import utils

%load_ext autoreload
%autoreload 2

# %% Make a fake dataset
N = 4  # neurons
D = 2 # dynamics dimension
T = 100 # time points
seed = 2
num_samples = 10


key = jax.random.PRNGKey(seed)

k1,key = jax.random.split(key,2)
mean_A = utils.random_rotation(k1,D,.1)

true_initial = models.InitialCondition(D)
true_lds = models.TimeVarLDS(D=D,initial=true_initial)

k1,key = jax.random.split(key,2)
true_emission = models.LinearEmission(key=k1,D=D,N=N)


sigma = 10
kernel_A = lambda x, y: 1e-2*(
    1e-6*(x==y)+jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma**2))
)
kernel_b = lambda x, y: 1e-3*(
    1e-6*(x==y)+jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma**2))
)
kernel_L = lambda x, y: 1e-3*(
    1e-6*(x==y)+jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma**2))
)

true_gps = {
    'A': models.GaussianProcess(kernel_A,D,D),
    'b': models.GaussianProcess(kernel_b,D,1),
    'L': models.GaussianProcess(kernel_L,D,D)
}

# true_likelihood = models.PoissonConditionalLikelihood(N)
true_likelihood = models.NormalConditionalLikelihood(
    N,
    scale_tril=1e-3*jnp.eye(N)
)
true_joint = models.GPLDS(
    true_gps,
    true_lds,
    true_emission,
    true_likelihood
)
ts = jnp.arange(1,T).astype(float)
k1,key = jax.random.split(key,2)

As = true_gps['A'].sample(k1,ts) + mean_A[:,:,None]
bs = true_gps['b'].sample(k1,ts)
Ls = true_gps['L'].sample(k1,ts)

# %%
x,y = [],[]
for n in range(num_samples):
    k1,key = jax.random.split(key,2)
    x_ = true_lds.sample(
        k1,
        As.transpose(2,0,1),
        bs.transpose(2,0,1),
        Ls.transpose(2,0,1),
        # x0=jnp.array([1,1])
    )
    x.append(x_)

    stats = true_emission.f(x_,true_emission.params)
    k1,key = jax.random.split(key,2)
    y_ = true_likelihood.sample(key=k1,stats=stats)
    y.append(y_)

# num_samples, T, N
y = jnp.stack(y)
x = jnp.stack(x)


# %%
visualizations.plot_states([x[0],x[1]],['x','x'])
visualizations.plot_states([y[0],y[1]],['y','y'])

# %%
k1,key = jax.random.split(key,2)
recognition = inference.AmortizedRNN(key,D,N,T,H=50)

# %%
k1,key = jax.random.split(key,2)

loss = inference.infer(
    k1,true_joint,recognition,y,ts,
    n_iter=1000,step_size=.1,gamma=1
)

# %%
visualizations.plot_loss(loss)
# %%
z,lp = recognition.f(k1,y[0],ts,recognition.params)
mu = recognition.posterior_mean(y[0],ts,recognition.params)

# %%
visualizations.plot_states([mu['x'][0]],['x'],titlestr='Posterior Mean')
visualizations.plot_states([z['x'][0]],['x'],titlestr='Inferred')
visualizations.plot_states([x[0]],['x'],titlestr='True')

# %%
