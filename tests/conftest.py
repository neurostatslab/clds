import jax
import jax.numpy as jnp
import pytest

from clds.models import CLDS
from clds.utils import Tm_basis

@pytest.fixture(scope="module", params=[False, True], ids=["learn_Q", "fix_Q"])
def clds_case(request):
    fix_Q = request.param
    latent_dim = 2
    emission_dim = 3
    condition_dim = 1

    # Define bases
    torus_basis_funcs = Tm_basis(5, M_conditions=1, sigma=0.5, kappa=0.1, period=2*jnp.pi)

    # Define model
    model = CLDS(
        state_dim=latent_dim,
        emission_dim=emission_dim,
        basis=torus_basis_funcs,
        gp_priors=["A", "b", "m"],
        use_dynamics_bias=True,
        use_emissions_bias=False,
        fix_Q=fix_Q,
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
            dynamics_cov=jnp.eye(latent_dim) if fix_Q else Q,
            # emissions_cov=noise_scale**2 * jnp.eye(emission_dim),
        )

    return {
        "model": model,
        "params": params,
        "condition_dim": condition_dim,
        "fixed_covariances": ("dynamics",) if fix_Q else (),
    }
