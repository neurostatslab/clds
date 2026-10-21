from clds.models import CLDS2
from clds.utils import Tm_basis
import jax.numpy as jnp
import jax
import pytest

@pytest.fixture(scope="module")
def clds_case():
    latent_dim = 2
    emission_dim = 3
    condition_dim = 1

    # Define bases
    _sigma, _kappa, _period = 0.5, 0.1, 2*jnp.pi
    torus_basis_funcs = Tm_basis(5, M_conditions=1, sigma=_sigma, kappa=_kappa, period=_period)

    # Define model
    model = CLDS2(
        state_dim=latent_dim, 
        emission_dim=emission_dim,
        basis=torus_basis_funcs,
        use_dynamics_prior=True,
        use_dynamics_bias=True,
        use_emissions_prior=False,
        use_emissions_bias=False,
        use_initial_prior=True,
        )

    # Define params
    S0 = jnp.array([[1.0, 0.2], [0.2, 0.8]])
    Q = jnp.array([[1.0, 0.1], [0.1, 10.0]])
    assert jnp.all(jnp.linalg.eigvalsh(S0) >= -1e-7)
    assert jnp.all(jnp.linalg.eigvalsh(Q) >= -1e-7)

    params = model.initialize_params(
        key=jax.random.key(0)
        )._replace( # use nonidentity to ensure no false-wins
            initial_cov=S0,
            dynamics_cov=Q,
            # emissions_cov=noise_scale**2 * jnp.eye(emission_dim),
        )

    return {
        "model": model,
        "params": params,
        "condition_dim": condition_dim,
        "fixed_covariances": (), # or ("dynamics",) when Q is fixed
    }