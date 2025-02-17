'''
Helper functions for postprocessing, given trained models
'''
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from scipy.stats import binned_statistic_2d
from tqdm import tqdm 

from models import ParamsCLDS, CLDS

def identifiability_transform(params):
    '''
    Given parameters of the model, return the linear transformation H
    that makes Q = I, and its inverse Hinv. 
    '''
    U, S, _ = jnp.linalg.svd(params.Q)
    H = U @ jnp.linalg.pinv(jnp.diag(jnp.sqrt(S))) @ U.T
    Hinv = U @ jnp.diag(jnp.sqrt(S)) @ U.T
    return H, Hinv

# %%
def get_fixed_points(
        model, params: ParamsCLDS, conditions_range: Float[Array, "T M"], ROTATE: bool=True,
        ) -> Float[Array, 'T state_dim']:
    '''
    Compute fixed points of the system, given parameters evaluated over condition range.
    Args:
        params: ParamsCLDS, the parameters of the model
        conditions_range: Float[Array, "T M"], the range of conditions to evaluate the fixed points
        ROTATE: bool, whether to rotate the latent space (chosen so that Q becomes identity)
    '''
    # assert model.wgps['A'] and model.wgps['b'], 'A and b priors are required' 
    # As_conditions = model.wgps['A'](params.dynamics_gp_weights, conditions_range)
    # bs_conditions = model.wgps['b'](params.bias_gp_weights, conditions_range).squeeze()
    As_conditions, _, bs_conditions, _ = model.weights_to_params(params, conditions_range)

    # Rotate
    if ROTATE:
        H, Hinv = identifiability_transform(params)
        As_conditions = jnp.einsum('ij,tjk,kl->til', H, As_conditions, Hinv)
        bs_conditions = jnp.einsum('ij,tj->ti', H, bs_conditions)

    # Compute fixed points
    fixed_points = jax.vmap(
        lambda t: jax.scipy.linalg.solve(jnp.eye(model.state_dim) - As_conditions[t], bs_conditions[t])
        )(jnp.arange(len(conditions_range)))
    
    # if model.fixed_point_func is not None:
    #     fixed_points = model.fixed_point_func(conditions_range)
    
    return fixed_points

# %%
def compute_composite_dynamics(
        model: CLDS, params: ParamsCLDS, 
        conditions_range: Float[Array, "N M"], emissions: Float[Array, "B T emission_dim"], conditions: Float[Array, "B T M"],
        ROTATE=True, n_bins=20, pad=1.0
        ):
    # assert model.wgps['A'] and model.wgps['b'], 'A and b priors are required' 
    assert model.state_dim == 2, 'Only 2D latent space is supported'

    H, Hinv = identifiability_transform(params)
    
    # Create grid around fixed points
    fixed_points = get_fixed_points(model, params, conditions_range, ROTATE=ROTATE)

    x_lims = (jnp.amin(fixed_points[:,0])-pad, jnp.amax(fixed_points[:,0])+pad)
    y_lims = (jnp.amin(fixed_points[:,1])-pad, jnp.amax(fixed_points[:,1])+pad)

    x_bin_edges = jnp.linspace(*x_lims, n_bins)
    y_bin_edges = jnp.linspace(*y_lims, n_bins)

    # Helper statistic functions
    def get_statistic_per_bin(values, xs, statistic='sum'):
        statistics, _, _, _ = binned_statistic_2d(
            xs[:,0], xs[:,1], values, bins=[x_bin_edges, y_bin_edges], statistic=statistic
            )
        return statistics

    def get_values_per_bin(parameters, xs):
        D1, D2 = parameters.shape[1], parameters.shape[2]
        parameters_sum_per_bin = get_statistic_per_bin(
            jnp.array([[parameters[:,i,j] for i in range(D1)] for j in range(D2)]).reshape(-1, len(xs)),
            xs,
        )
        parameters_sum_per_bin = parameters_sum_per_bin.reshape(D1, D2, len(x_bin_edges)-1, len(y_bin_edges)-1) 
        return parameters_sum_per_bin
    
    # Compute statistics, per bin, averaged over all batches

    As_bin_sums, bs_bin_sums, thetas_bin_sums, bin_counts = [], [], [], []
    for batch_id in tqdm(range(len(emissions))):
        # Compute posterior x
        x_smooth = model.smoother(params, emissions[batch_id], conditions[batch_id])[2][0]
        x_smooth_rot = jnp.einsum('ij,tj->ti', H, x_smooth) if ROTATE else x_smooth

        # MAP estimates of As and bs
        batch_As, _, batch_bs, _ = model.weights_to_params(params, conditions[batch_id])
        batch_bs = batch_bs[..., None]
        # batch_As = model.wgps['A'](params.dynamics_gp_weights, conditions[batch_id])
        batch_As_rot = jnp.einsum('ij,tjk,kl->til', H, batch_As, Hinv) if ROTATE else batch_As

        # batch_bs = model.wgps['b'](params.bias_gp_weights, conditions[batch_id])
        batch_bs_rot = jnp.einsum('ij,tjk->tik', H, batch_bs) if ROTATE else batch_bs

        # Estimate parameters and statistics per bin
        As_sums = get_values_per_bin(batch_As_rot, x_smooth_rot)
        bs_sums = get_values_per_bin(batch_bs_rot, x_smooth_rot).squeeze()
        thetas_sums = get_statistic_per_bin(conditions[batch_id], x_smooth_rot, statistic='sum')
        counts = get_statistic_per_bin(conditions[batch_id], x_smooth_rot, statistic='count')
        
        # Append
        As_bin_sums.append(As_sums)
        bs_bin_sums.append(bs_sums)
        thetas_bin_sums.append(thetas_sums)
        bin_counts.append(counts)

    # Compute the mean over all batches, per bin
    bin_counts = jnp.nansum(jnp.array(bin_counts), axis=0)
    A_per_bin = jnp.nansum(jnp.array(As_bin_sums), axis=0)/bin_counts[None, None, ...]
    b_per_bin = jnp.nansum(jnp.array(bs_bin_sums), axis=0)/bin_counts[None, ...]
    U_per_bin = jnp.nansum(jnp.array(thetas_bin_sums), axis=0)/bin_counts[None, ...]

    # Finally, compute composite dynamics per bin
    composite_F = jnp.zeros((len(x_bin_edges)-1, len(y_bin_edges)-1, model.state_dim))
    for x_id, _x in enumerate(x_bin_edges[:-1]):
        for y_id, _y in enumerate(y_bin_edges[:-1]):
            bin_x = jnp.array([_x, _y])
            Fx = (A_per_bin[:,:, x_id, y_id] - jnp.eye(model.state_dim)) @ bin_x + b_per_bin[:, x_id, y_id]
            composite_F = composite_F.at[x_id, y_id].set(Fx)

    return x_bin_edges[:-1], y_bin_edges[:-1], composite_F.T, U_per_bin, bin_counts 