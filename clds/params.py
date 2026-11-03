# -*- coding: utf-8 -*-
"""
@author: Amin, Victor
"""

from jaxtyping import Array, Float, Union
from typing import NamedTuple
import warnings


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


class ParamsCLDS(NamedTuple):
    initial_mean: Union[
        Float[Array, "state_dim"], Float[Array, "state_dim 1 len_basis"]
    ]
    initial_cov: Float[Array, "state_dim state_dim"]
    dynamics_weights: Union[
        Float[Array, "len_basis state_dim state_dim"],  # W_b
        Float[Array, "num_timesteps state_dim state_dim"],  # A_t
    ]
    dynamics_bias: Union[
        Float[Array, "len_basis 1 state_dim"],  # W_b
        Float[Array, "num_timesteps state_dim"],  # b_t
    ]
    dynamics_cov: Float[Array, "state_dim state_dim"]  # Q
    emissions_weights: Float[Array, "num_timesteps emission_dim state_dim"]
    emissions_bias: Float[Array, "num_timesteps emission_dim"]  # ds
    emissions_cov: Float[Array, "emission_dim emission_dim"]  # R


class ParamsBasis(NamedTuple):
    A_weights: Float[Array, "state_dim state_dim len_basis"]
    b_weights: Float[Array, "state_dim state_dim len_basis"]
    L_weights: Float[Array, "state_dim state_dim len_basis"]

def __getattr__(name):
    if name == "ParamsCLDS2":
        warnings.warn("ParamsCLDS2 has been renamed to ParamsCLDS.", DeprecationWarning, stacklevel=2)
        return ParamsCLDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
