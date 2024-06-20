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
    plt.figure(figsize=(5,5))

    
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
def plot_loss(loss,ylabel='',save=False,file=None):
    plt.plot(loss,'k')
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)

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
    plt.figure(figsize=(6*len(X),1*len(X[0].T)))
    
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

    if labels is not None:
        plt.legend(fontsize=fontsize)

    plt.title(titlestr,fontsize=fontsize)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

# %%
def time_var_dynamics(
        A,
        xlim=[-1,1],
        ylim=[-1,1],
        n_points=10,
        titlestr='',
        fontsize=15,
        xlabel='$x^{(1)}$',
        ylabel='$x^{(2)}$',
        scale=1,
        save=False,
        file=None
    ):
    # Function to update the quiver plot for each frame
    def update_quiver(num, Q, X, Y):
        matrix = scale*A[num]@np.stack((X.flatten(),Y.flatten()))
        Q.set_UVC(matrix[0], matrix[1])
        return Q,

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
    
    # ax.set_aspect('equal',adjustable='box')
    plt.tight_layout()

    # Create the animation
    ani = animation.FuncAnimation(
        fig, 
        update_quiver, 
        frames=A.shape[0], 
        fargs=(Q, X, Y), 
        interval=50, 
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
