# -*- coding: utf-8 -*-
"""
@author: Amin, Victor
"""

from jaxtyping import Array, Float
from typing import NamedTuple


# %%
class ParamsGP(NamedTuple):
    As: Float[Array, "num_timesteps state_dim state_dim"]
    bs: Float[Array, "num_timesteps state_dim"]
    Ls: Float[Array, "num_timesteps state_dim state_dim"]
    

class ParamsEmission(NamedTuple):
    Cs: Float[Array, "num_timesteps emission_dim state_dim"]
    ds: Float[Array, "num_timesteps state_dim"]

class ParamsNormalLikelihood(NamedTuple):
    scale_tril: Float[Array, "emission_dim emission_dim"]

class ParamsGPLDS(NamedTuple):
    emissions:  ParamsEmission
    likelihood: ParamsNormalLikelihood


class ParamswGPLDS(NamedTuple):
    m0: Float[Array, "state_dim"]                               # Accessed if 'm0' wgp prior is None
    m0_gp_weights: Float[Array, "state_dim 1 len_basis"]
    S0: Float[Array, "state_dim state_dim"]
    dynamics_gp_weights: Float[Array, "state_dim state_dim len_basis"]
    emissions_gp_weights: Float[Array, "emission_dim state_dim len_basis"]
    Cs: Float[Array, "num_timesteps emission_dim state_dim"]    # Accessed if 'C' wgp prior is None
    bias_gp_weights: Float[Array, "state_dim 1 len_basis"]
    bs: Float[Array, "num_timesteps state_dim"]                 # Accessed if 'b' wgp prior is None
    Q: Float[Array, "state_dim state_dim"]
    R: Float[Array, "emission_dim emission_dim"]


class ParamsBasis(NamedTuple):
    A_weights: Float[Array, "state_dim state_dim len_basis"]
    b_weights: Float[Array, "state_dim state_dim len_basis"]
    L_weights: Float[Array, "state_dim state_dim len_basis"]
