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
import loader

from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA

import argparse
import yaml
import os

# %%
def get_args():
    '''Parsing the arguments when this file is called from console
    '''
    parser = argparse.ArgumentParser(description='Runner for CCM',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', '-c', metavar='Configuration',help='Configuration file address',default='/')
    parser.add_argument('--output', '-o', metavar='Output',help='Folder to save the output and results',default='/')
    
    return parser.parse_args()

# %%
if __name__ == '__main__':
    args = get_args()
    with open(args.config, 'r') as stream: pm = yaml.safe_load(stream)

    data_params, model_params, optim_params = pm['data_params'], pm['model_params'], pm['optim_params']

    seed = pm['model_params']['seed']
    file = args.output

    if not os.path.exists(file): os.makedirs(file)

    dataloader = eval('loader.'+data_params['dataset'])(data_params)
    y,ts = dataloader.load_train_data()

    
    visualizations.plot_signals(
        [y[0]],
        titlestr='Trial',
        save=True,
        file=file+'trial'
    )

    print('Trials,Times,Neurons',y.shape)
    
    B,T,N = y.shape
    D = model_params['D']

    empirical_sigmas = jnp.array(
        [EmpiricalCovariance().fit(y[:,t]).covariance_ 
        for t in range(T)]
    )

    
    model = dataloader.__class__.__name__
    if hasattr(dataloader,'A'):
        visualizations.time_var_dynamics(
            dataloader.A[None,:,:][[0]*T],
            titlestr=model+' (True)',
            scale=.1/jnp.abs(dataloader.A).max(),
            xlim=[-1,1],
            ylim=[-.2,.2],
            save=True,
            file=file+'A_true'
        )

    visualizations.plot_states(
        [y[0],y[1]],
        ['y','y'],
        legend=False,
        save=True,
        file=file+'states'
    )

    seed = model_params['seed']
    key = jax.random.PRNGKey(seed)

    initial = models.InitialCondition(
        D,
        scale_tril=jnp.std(y[:,0,0])*jnp.eye(D)
    )

    lds = models.TimeVarLDS(
        D=D,initial=initial,
    )

    pca = PCA(n_components=D)
    pca.fit(y.mean(0))

    k1,key = jax.random.split(key,2)

    emission = models.LinearEmission(
        key=k1,D=D,N=N,
        C=pca.components_.T,
        d=jnp.zeros(N)
    )

    kernel_A = utils.get_kernel(model_params['kernel_A'],model_params['kernel_A_diag'])
    kernel_b = utils.get_kernel(model_params['kernel_b'],model_params['kernel_b_diag'])
    kernel_L = utils.get_kernel(model_params['kernel_L'],model_params['kernel_L_diag'])

    gps = {
        'A': models.GaussianProcess(kernel_A,D,D),
        'b': models.GaussianProcess(kernel_b,D,1),
        'L': models.GaussianProcess(kernel_L,D,D)
    }


    likelihood = models.NormalConditionalLikelihood(
        N,
        scale_tril=model_params['obs_noise']*jnp.eye(N)
    )
    joint = models.GPLDS(
        gps,
        lds,
        emission,
        likelihood
    )

    k1,key = jax.random.split(key,2)
    recognition = eval('inference.'+optim_params['variational'])(
        k1,D,T,
        scale_L=optim_params['scale_L'] if 'scale_L' in optim_params else 1,
    )

    loss = inference.map(
        joint,
        recognition,
        y,
        ts,
        n_iter=optim_params['n_iter'],
        step_size=optim_params['step_size'],
        gamma=optim_params['gamma']
    )

    visualizations.plot_loss(
        loss,
        save=True,
        file=file+'loss'
    )

    out = {}

    EA,Eb,EL = recognition.posterior_mean(
        y[0],ts[0],recognition.params
    )


    out['EA'] = EA
    out['Eb'] = Eb
    out['EL'] = EL

    if 'b_train' in dataloader.data:
        visualizations.plot_states(
            [Eb,dataloader.data['b_train'][0]],
            ['Inferred','True'],
            titlestr='$b$',
            save=True,
            file=file+'b_states'
        )

        visualizations.plot_signals(
            [Eb,dataloader.data['b_train'][0]],
            labels=['Inferred','True'],
            margin=Eb.max(),
            save=True,
            file=file+'b_signals'
        )
        out['b_true'] = dataloader.data['b_train'][0]
    else:
        visualizations.plot_states(
            [Eb],
            ['Inferred'],
            titlestr='$b$',
            save=True,
            file=file+'b_states'
        )
        visualizations.plot_signals(
            [Eb,],
            labels=['Inferred'],
            margin=Eb.max(),
            save=True,
            file=file+'b_signals'
        )

    mus,sigmas = lds.evolve_stats(EA,Eb,EL)

    out['sigmas'] = sigmas
    out['mus'] = mus

    sigmas_obs = jnp.array([
        emission.C@sigmas[i]@emission.C.T + \
        likelihood.scale_tril@likelihood.scale_tril.T
        for i in range(len(sigmas))
    ])
    mus_obs = (emission.C @ mus.T).T

    out['sigmas_obs'] = sigmas_obs
    out['mus_obs'] = mus_obs

    sigmas_empirical = jnp.array(
        [EmpiricalCovariance().fit(y[:,t]).covariance_ 
        for t in range(T)]
    )

    mus_empirical = y.mean(0)
    out['sigmas_empirical'] = sigmas_empirical
    out['mus_empirical'] = mus_empirical

    if 'x_train' in dataloader.data and hasattr(dataloader,'C') and hasattr(dataloader,'scale_tril'):
        sigmas_true = jnp.array(
            [EmpiricalCovariance().fit(dataloader.data['x_train'][:,t]).covariance_ 
            for t in range(T)]
        )

        sigmas_true = jnp.array([
            dataloader.C@sigmas_true[i]@dataloader.C.T + \
            dataloader.scale_tril@dataloader.scale_tril.T
            for i in range(len(sigmas)
            )
        ])

        mus_true = (dataloader.C @ dataloader.data['x_train'].mean(0).T).T

        out['sigmas_true'] = sigmas_true
        out['mus_true'] = mus_true

        visualizations.visualize_pc(
            mus_true[:,None],sigmas_true,
            pc=y,
            linewidth=1,
            titlestr='True',
            save=True,
            file=file+'pc_true'
        )

    
    visualizations.visualize_pc(
        mus_empirical[:,None],sigmas_empirical,
        pc=y,
        linewidth=1,
        titlestr='Empirical',
        save=True,
        file=file+'pc_emp'
    )


    visualizations.visualize_pc(
        mus_obs[:,None],sigmas_obs,
        pc=y,
        linewidth=1,
        titlestr='Inferred',
        save=True,
        file=file+'pc_inferred'
    )

    visualizations.visualize_pc(
        mus[:,None],sigmas,
        linewidth=1,
        save=True,
        file=file+'pc_latent'
    )

    if hasattr(dataloader,'grid'):
        grid_EA,grid_Eb,grid_EL = recognition.posterior_mean(
            y[0],dataloader.grid,recognition.params
        )

        mus_grid,sigmas_grid = lds.evolve_stats(grid_EA,grid_Eb,grid_EL)

        sigmas_obs_grid = jnp.array([
            emission.C@sigmas_grid[i]@emission.C.T + \
            likelihood.scale_tril@likelihood.scale_tril.T
            for i in range(len(sigmas))])
        mus_obs_grid = (emission.C @ mus_grid.T).T


        visualizations.plot_states(
            [grid_Eb],
            ['Inferred'],
            titlestr='$b$',
            save=True,
            file=file+'grid_b_states'
        )
        visualizations.plot_signals(
            [grid_Eb],
            labels=['Inferred'],
            margin=Eb.max(),
            save=True,
            file=file+'grid_b_signals'
        )

        visualizations.time_var_dynamics(
            grid_EA-jnp.eye(D)[None],
            titlestr=model,
            scale=.1/grid_EA.max(),
            xlim=[-1,1],
            ylim=[-.2,.2],
            save=True,
            file=file+'A'
        )

        out['grid_Eb'] = grid_Eb
        out['grid_EA'] = grid_EA
        out['grid_EL'] = grid_EL

        out['mus_grid'] = mus_grid
        out['sigmas_grid'] = sigmas_grid
    else:
        visualizations.time_var_dynamics(
            EA-jnp.eye(D)[None],
            titlestr=model,
            scale=5/EA.max(),
            xlim=[-1,1],
            ylim=[-.2,.2],
            save=True,
            file=file+'A'
        )

    jnp.save(file+'stats',out)

