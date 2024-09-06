# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation

from sklearn.decomposition import PCA
from matplotlib import cm
import matplotlib

from scipy import stats

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib import cm

plt.style.use('bmh')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

colors_ = np.concatenate((
    cm.get_cmap('tab20c', 20).colors,
    cm.get_cmap('tab20b', 20).colors
))

color_dict = {
    'x': 'k',
    'y': 'r',
    'True': 'k',
    'Inferred': 'r',
    'Position': 'g',
    'Velocity': 'b'
}
        
label_dict = {
    'x': '$x$',
    'y': '$y$',
    'True': 'True',
    'Inferred': 'Inferred',
    'Position': 'Position',
    'Velocity': 'Velocity'
}

# %%
def plot_states(
        states,
        labels=None,
        titlestr='',
        fontsize=15,
        legend=True,
        save=False,
        file=None
    ):
    plt.figure(figsize=(4,4))

    
    if states[0].shape[1] == 2:
        states_pca = states
    else:
        pca = PCA(n_components=2)
        pca.fit(states[0])
        states_pca = [[]]*len(states)
        for i in range(len(states)):
            states_pca[i] = pca.transform(states[i])
            states_pca[i] = states_pca[i]-states_pca[i].mean(0)[None]
    
    for i in range(len(states_pca)):
        plt.plot(
            states_pca[i][:,0],states_pca[i][:,1],linewidth=.5,ls='--',
            c=color_dict[labels[i]] if labels is not None else 'k',
            label=label_dict[labels[i]] if labels is not None else labels,
        )
        plt.gca().scatter(
            states_pca[i][:,0],states_pca[i][:,1],lw=2,
            c=color_dict[labels[i]] if labels is not None else 'k',
            alpha=[np.linspace(0,1,len(states_pca[i]))],
            # label=label_dict[labels[i]] if labels is not None else labels
        )

        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(3))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))

    if legend:
        plt.legend(fontsize=fontsize)
    plt.xlabel('PC1',fontsize=fontsize)
    plt.ylabel('PC2',fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(titlestr,fontsize=fontsize)
    plt.tight_layout()
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()


# %%
def plot_loss(loss,ylabel='',fontsize=12,save=False,file=None):
    plt.figure(figsize=(3,3))
    plt.plot(loss,'k')
    plt.xlabel('Iteration',fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)


    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(4))

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()


# %%
def visualize_pc(
        means,
        covs,
        pc=None,
        color=None,
        legend=False,
        titlestr='',
        fontsize=15,
        dotsize=50,
        linewidth=5,
        std=3,
        text=False,
        lim=None,
        save=False,
        file=None
    ):
    '''Visualize point clouds and atlases learned from them
    '''

    ylorbr = cm.get_cmap('viridis', means.shape[0])
    colors = ylorbr(np.linspace(0,1,means.shape[0]))

    if means.shape[2] > 2:
        pca = PCA(n_components=2)
        pca.fit_transform(np.vstack(means))
        
        means = [pca.transform(means[i]) for i in range(len(means))]
        covs = [pca.components_@covs[i]@pca.components_.T for i in range(len(means))]
        
        if pc is not None:
            pc = pca.transform(np.vstack(pc))
    elif pc is not None:
        pc = np.vstack(pc)
            

    plt.figure(figsize=(8,8))
    plt.title(titlestr,fontsize=fontsize)


    plt.plot(
        np.mean(means,1)[:,0],np.mean(means,1)[:,1],
        lw=linewidth,linestyle='dashed',color='k',alpha=.3
    )

    facecolors = colors.copy()
    facecolors[:,-1] /= 10

    if pc is not None:
        plt.scatter(
            pc[:,0],
            pc[:,1],
            s=dotsize/2,
            marker='.',
            c='k'
        )

    for j in range(len(means)):
        plt.scatter(
            means[j][:,0],
            means[j][:,1], 
            s=dotsize,
            marker='o',
            c=colors[j] if color is None else color[j].repeat(3)
        )
        
        draw_ellipse(
            means[j].mean(0)[:2],covs[j][:2,:2],colors[j],ax=plt.gca(),
            std_devs=std,facecolor=facecolors[j],linewidth=linewidth
        )

        if text: 
            plt.text(means[j][:,0],means[j][:,1],str(j))


    if legend: 
        plt.legend(
            fontsize=fontsize,
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )
    plt.grid(False)

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)


    if lim is not None:
        plt.xlim(lim)
        plt.ylim(lim)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()


# %%
def draw_ellipse(
        mu, cov, colors, ax, std_devs=3.0, 
        facecolor='none', linewidth=1, **kwargs
    ):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
        facecolor=facecolor, edgecolor=colors, linewidth=linewidth
    )

    scale_x = np.sqrt(cov[0, 0]) * std_devs
    scale_y = np.sqrt(cov[1, 1]) * std_devs

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mu[0], mu[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# %%
def plot_signals(
        X,
        t=None,
        labels=None,
        titlestr='',
        fontsize=15,
        linewidth=2,
        margin=.1,
        save=False,
        file=None
    ):
    plt.figure(figsize=(3,.5*len(X[0].T)))
    
    colors=['k','r','b']

    if labels is None: labels = [None]*len(X)
    
    offset = np.append(0.0, np.nanmax(X[0][:,0:-1,],0)-np.nanmin(X[0][:,0:-1],0))
    shifts = np.cumsum(offset+margin)
    for i,x in enumerate(X):
        s = (x-np.nanmin(x,0)[None,:]+shifts[None,:])
        
        if t is not None:
            plt.plot(t,s,linewidth=linewidth,color=colors[i],label=labels[i])
        else:
            plt.plot(s,linewidth=linewidth,color=colors[i],label=labels[i])
        plt.grid('off')

        # if i == 0: plt.ylim([s.min()-margin,s.max()+margin])
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    
    if labels is not None:
        plt.legend(handles, labels, loc='center left', fontsize=fontsize, bbox_to_anchor=(1, 0.5))


    plt.title(titlestr,fontsize=fontsize)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

# %%
def time_var_dynamics(
        As,
        bs=None,
        colors=['k','r'],
        xlim=[-1,1],
        ylim=[-1,1],
        n_points=10,
        titlestr='',
        fontsize=15,
        xlabel='$x^{(1)}$',
        ylabel='$x^{(2)}$',
        linewidth=.01,
        scale=1,
        save=False,
        file=None
    ):
    # Function to update the quiver plot for each frame
    def update_quiver(num, Qs, X, Y):
        for i in range(len(As)):
            matrix = As[i][num]@np.stack((X.flatten(),Y.flatten()))
            if bs is not None: matrix = matrix + bs[i][num][:,None]
        
            Qs[i].set_UVC(scale*matrix[0], scale*matrix[1])
        return Qs

    # Create a grid of points
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)

    # Create a quiver plot
    fig, ax = plt.subplots()
    
    
    Qs = [ax.quiver(
        X, 
        Y, 
        np.zeros_like(X), 
        np.zeros_like(Y), 
        pivot='mid', 
        scale=1,
        color=colors[i],
        units='width',
        width=linewidth
    ) for i in range(len(As))]
    
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))

    plt.title(titlestr,fontsize=fontsize)

    plt.xlim(xlim)
    plt.ylim(ylim)
    
    # ax.set_aspect('equal',adjustable='box')
    plt.tight_layout()

    # Create the animation
    ani = animation.FuncAnimation(
        fig, 
        update_quiver, 
        frames=As[0].shape[0], 
        fargs=(Qs, X, Y), 
        interval=100, 
        blit=True
    )

    if save:
        ani.save(file+'.gif', writer='ffmpeg')
        plt.close('all')
    else:
        plt.show()


# %%
def flow_2d(
        A,
        means,
        covs,
        pc=None,
        xlim=[-1,1],
        ylim=[-1,1],
        n_points=10,
        titlestr='',
        fontsize=15,
        xlabel='$x^{(1)}$',
        ylabel='$x^{(2)}$',
        scale=1,
        color=None,
        legend=False,
        dotsize=50,
        linewidth=5,
        std=3,
        text=False,
        lim=None,
        save=False,
        file=None
    ):

    # Create a grid of points
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)

    # Create a quiver plot
    fig, ax = plt.subplots()
    
    Q = ax.quiver(
        X, 
        Y, 
        np.zeros_like(X), 
        np.zeros_like(Y), 
        pivot='mid', 
        scale=1
    )
    
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))

    plt.title(titlestr,fontsize=fontsize)

    plt.xlim(xlim)
    plt.ylim(ylim)
    
    plt.tight_layout()

    matrix = scale*A@np.stack((X.flatten(),Y.flatten()))
    Q.set_UVC(matrix[0], matrix[1])



    ylorbr = cm.get_cmap('viridis', means.shape[0])
    colors = ylorbr(np.linspace(0,1,means.shape[0]))

    
    plt.title(titlestr,fontsize=fontsize)


    plt.plot(
        np.mean(means,1)[:,0],np.mean(means,1)[:,1],
        lw=linewidth,linestyle='dashed',color='k',alpha=.3
    )

    facecolors = colors.copy()
    facecolors[:,-1] /= 10

    for j in range(len(means)):
        plt.scatter(
            means[j][:,0],
            means[j][:,1], 
            s=dotsize,
            marker='o',
            c=colors[j] if color is None else color[j].repeat(3)
        )
        
        draw_ellipse(
            means[j].mean(0)[:2],covs[j][:2,:2],colors[j],ax=plt.gca(),
            std_devs=std,facecolor=facecolors[j],linewidth=linewidth
        )

        if text: 
            plt.text(means[j][:,0],means[j][:,1],str(j))


    
    if pc is not None:
        plt.scatter(
            pc[:,0],
            pc[:,1],
            s=dotsize/2,
            marker='.',
            c='k'
        )


    if legend: 
        plt.legend(
            fontsize=fontsize,
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )
    plt.grid(False)

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)


    if lim is not None:
        plt.xlim(lim)
        plt.ylim(lim)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()


# %%
def plot_hd_params(theta,true_As,true_bs,true_Cs,theta_indices=[3,76,58]):
    # Plot heading direction trajectory
    plt.plot(theta % (2 * np.pi))
    plt.ylabel('heading direction')
    plt.xlabel('time')
    plt.title('heading direction over time')
    plt.show()


    # Plot
    fig, axs = plt.subplots(ncols=3, figsize=[8,3], constrained_layout=True)
    axs[0].plot(true_As[:,1])
    axs[1].plot(true_bs)
    axs[2].plot(theta % (2 * np.pi), true_Cs[:,:,0], '-')
    axs[0].set_title("A")
    axs[1].set_title("b")
    axs[2].set_title("C")
    fig.suptitle("True params")
    plt.show()


    def plot_vector_field(A, ax):
        x = np.linspace(-1, 1, 8)
        y = np.linspace(-1, 1, 8)
        X, Y = np.meshgrid(x, y)
        U = A[0, 0] * X + A[0, 1] * Y
        V = A[1, 0] * X + A[1, 1] * Y

        ax.quiver(X, Y, U, V, units='width', scale=8, width=0.01)

    fig, axs = plt.subplots(ncols=3, figsize=[8,3], constrained_layout=True)
    for i, _theta_ind in enumerate(theta_indices):
        _theta = theta[_theta_ind]
        plot_vector_field(true_As[_theta_ind] - np.eye(2), axs[i])
        axs[i].set_title(f"theta = {_theta % (2*np.pi):.2f} rad")


    for ax in axs:
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("True As")

    plt.show()


    # fig, ax = plt.subplots(figsize=[4,3])
    # ax.plot(A_prior.sample(params['seed'], theta)[:,0,0])
    # ax.plot(A_prior.sample(jxr.PRNGKey(0), theta)[:,0,1])
    # ax.set_title('A prior samples')


    fig, axs = plt.subplots(ncols=latent_dim, nrows=latent_dim, figsize=(5, 4), constrained_layout=True);

    for i in range(2):
        for j in range(2):
            ax = axs[i,j]
            ax.plot(true_As[:,i,j], 'k--', label='True')
            ax.plot(A_prior(initial_params.dynamics_gp_weights, theta[:-1])[:,i,j], label='init')
            ax.plot(A_prior(params.dynamics_gp_weights, theta[:-1])[:,i,j], label='EM')
            # ax.set_title(f'A[{i},{j}]')
    axs[1,0].set_xlabel('Time')
    axs[1,1].legend()


    for ax in axs[0]:
        ax.set_xticklabels([])
        
    fig.suptitle('Dynamics matrix $A$')

    plt.show()

    fig, ax = plt.subplots()
    ax.plot(true_bs,'k--',label='True')
    ax.plot(b_prior(params.bias_gp_weights,theta).squeeze(),c='tab:orange',label='EM')
    ax.set_title("Bias")
    ax.set_xlabel('Time')
    plt.show()


    for neuron_id in range(N)[:4]:
        plt.plot(y[0,:100,neuron_id]  + 4*neuron_id, c='k', ls='--')
        plt.plot(reconstructed_ys[:100,neuron_id] + 4*neuron_id, c='tab:orange')
        plt.fill_between(
            jnp.arange(len(reconstructed_ys))[:100],
            reconstructed_ys[:100,neuron_id] + 4*neuron_id - 2 * jnp.sqrt(reconstructed_ys_covs[:100,neuron_id,neuron_id]),
            reconstructed_ys[:100,neuron_id] + 4*neuron_id + 2 * jnp.sqrt(reconstructed_ys_covs[:100,neuron_id,neuron_id]),
            alpha=0.3, color='tab:orange',
        )

    # plt.legend()
    plt.title('Reconstructed data, half the neurons')
    plt.show()




    fig, ax = plt.subplots()
    for i in range(2):
        ax.plot(smoothed_means[1:, i], color='tab:orange', label='Smoothed')
        ax.fill_between(
            jnp.arange(T-1),
            smoothed_means[1:, i] - 2 * jnp.sqrt(smoothed_covs[1:, i, i]),
            smoothed_means[1:, i] + 2 * jnp.sqrt(smoothed_covs[1:, i, i]),
            color='tab:orange',
            alpha=0.3,
        )
        ax.plot(dataloader.data['x_train'][0, 1:, i], 'k--', label='True')
    ax.set_title('Smoother recovery')
    ax.legend()



    fig, axs = plt.subplots(nrows=4, figsize=(6, 5), height_ratios=[1,2,2,2], constrained_layout=True)

    axs[0].plot(theta % (2 * jnp.pi), '.', c='tab:blue')

    # Reconstruction

    for neuron_id in range(N)[::2]:
        axs[1].plot(y[0][:50,neuron_id], c='tab:blue')
        axs[1].plot(reconstructed_ys[:50,neuron_id], c='tab:orange')
        axs[1].fill_between(
            jnp.arange(len(reconstructed_ys))[:50],
            reconstructed_ys[:50,neuron_id] - 2 * jnp.sqrt(reconstructed_ys_covs[:50,neuron_id,neuron_id]),
            reconstructed_ys[:50,neuron_id] + 2 * jnp.sqrt(reconstructed_ys_covs[:50,neuron_id,neuron_id]),
            alpha=0.3, color='tab:orange',
        )

    lines = [plt.Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in ['tab:blue', 'tab:orange']]
    labels = ['True', 'Reconstructed']
    axs[1].legend(lines, labels)

    # Bias 

    axs[2].plot(theta[:30] % (2*jnp.pi), true_bs[:30],'k--', label='True');
    axs[2].plot(theta[:30] % (2*jnp.pi), params.bs[:30], c='tab:orange', label='EM');
    # axs[1].set_title("Bias")
    # axs[1].set_xlabel('Time')

    # Dynamics

    for i in range(2):
        for j in range(2):
            axs[3].plot(theta % (2*jnp.pi), true_As[:,i,j], 'k--', label='True')
            # axs[3].plot(A_prior(initial_params.dynamics_gp_weights, theta[:-1])[:,i,j], label='init')
            axs[3].plot(theta[:-1] % (2*jnp.pi), A_prior(params.dynamics_gp_weights, theta[:-1])[:,i,j], c='tab:orange', label='EM')
            # ax.set_title(f'A[{i},{j}]')

    # Labels and formatting

    for ax in axs[:-1]:
        ax.set_xticklabels([])
    axs[-1].set_xticks([0, jnp.pi/2, jnp.pi, 3*jnp.pi/2, 2*jnp.pi])
    axs[-1].set_xticklabels(['0', r"$\frac{\pi}{2}$", '$\pi$', r"$\frac{3\pi}{2}$", '$2\pi$'])


    lines = [plt.Line2D([0], [0], color='k', linewidth=2, linestyle='--'), plt.Line2D([0], [0], color='tab:orange', linewidth=2, linestyle='-')]
    labels = ['True', 'EM']
    axs[2].legend(lines, labels)



    # Plot vector field given 2x2 matrix
    def plot_vector_field(A, ax, color='k'):
        x = jnp.linspace(-1, 1, 6)
        y = jnp.linspace(-1, 1, 6)
        X, Y = jnp.meshgrid(x, y)
        U = A[0, 0] * X + A[0, 1] * Y
        V = A[1, 0] * X + A[1, 1] * Y

        # Thicken the arrows
        ax.quiver(X, Y, U, V, units='width', scale=6, width=0.015, color=color)


    fig, axs = plt.subplots(ncols=3, figsize=[8,3], constrained_layout=True)
    plot_vector_field(true_As[3] - jnp.eye(model_params['D']), axs[0])
    plot_vector_field(_As[3] - jnp.eye(model_params['D']), axs[0], color='tab:orange')
    print(theta[3] % (2 * jnp.pi) / jnp.pi)

    plot_vector_field(true_As[76] - jnp.eye(model_params['D']), axs[1])
    plot_vector_field(_As[76] - jnp.eye(model_params['D']), axs[1], color='tab:orange')
    print(theta[76] % (2 * jnp.pi) / jnp.pi)

    plot_vector_field(true_As[58] - jnp.eye(model_params['D']), axs[2])
    plot_vector_field(_As[58] - jnp.eye(model_params['D']), axs[2], color='tab:orange')
    print(theta[58] % (2 * jnp.pi)  / jnp.pi)

    for ax in axs:
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
