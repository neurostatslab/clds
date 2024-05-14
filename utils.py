# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import jax.numpy as jnp
import jax
import numpy as np
from functools import reduce

# %%
def random_rotation(key, n, theta):
    rot = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], 
         [jnp.sin(theta),  jnp.cos(theta)]]
    )
    out = jnp.eye(n)
    out = out.at[:2, :2].set(rot)
    q = jnp.linalg.qr(
        jax.random.uniform(key, shape=(n, n))
    )[0]
    
    return q.dot(out).dot(q.T)


# %%
def split_data_cv(data,props,seeds):
    # props: train, validation, test
    # seeds: test, validation
    # data: y (possibly mu, sigma, F, mu_g, sigma_g)

    assert 'train' in props.keys() and 'test' in props.keys() and 'validation' in props.keys()
    assert props['train'] + props['test'] + props['validation'] == 1
    assert 'test' in seeds.keys() and 'validation' in seeds.keys()
    assert 'y' in data.keys()
     
    N,M,D = data['y'].shape
    
    trial_indices = jax.random.permutation(
        jax.random.PRNGKey(seeds['test']),
        np.arange(N)
    )

    test_trials = trial_indices[-int(props['test']*N):]

    train_trials = jax.random.choice(
        jax.random.PRNGKey(seeds['validation']),
        shape=(int(N*props['train']),),
        a=trial_indices[:-int(props['test']*N)],
        replace=False
    ).sort()

    validation_trials = jnp.setdiff1d(trial_indices[:-int(props['test']*N)],train_trials).tolist()

    out = {}
    for k in data.keys():
        out[k+'_train'] = data[k][train_trials,...]
        out[k+'_test'] = data[k][test_trials,...]
        out[k+'_validation'] = data[k][validation_trials,...]
    
    return out



# %%
def get_kernel(params,diag):
    '''Returns the full kernel of multi-dimensional condition spaces
    '''
    if len(params) > 1: 
        kernel = lambda x,y: diag*jnp.all(x==y)+reduce(
            lambda a,b: a*b, [
                _get_kernel(
                    params[i]['type'],
                    params[i])(x[i],y[i]
                ) for i in range(len(params))
            ])
    else: 
        kernel = lambda x,y: diag*(x==y)+_get_kernel(params[0]['type'],params[0])(x,y)
    
    return kernel
        
# %%
def _get_kernel(kernel,params):
    '''Private function, returns the kernel corresponding to a single dimension
    '''
    if kernel == 'periodic': 
        return lambda x,y: params['scale']*jnp.exp(-2*(jnp.sin(jnp.pi*jnp.abs(x-y)/params['normalizer'])**2)/(params['sigma']**2))
    if kernel == 'RBF': 
        return lambda x,y: params['scale']*jnp.exp(-(jnp.linalg.norm((x-y)/params['normalizer'])**2)/(2*params['sigma']**2))
