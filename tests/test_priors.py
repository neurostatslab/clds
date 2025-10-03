import jax
from clds.models import WeightSpaceGaussianProcess
import pytest
import jax.numpy as jnp
import jax.random as jxr
from clds import utils

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def torus_basis_funcs(n_basis_funcs):
    _sigma, _kappa, _period = 0.5, 0.1, 2 * jnp.pi
    return utils.Tm_basis(
        n_basis_funcs, M_conditions=1, sigma=_sigma, kappa=_kappa, period=_period
    )


@pytest.mark.parametrize("n_basis_funcs", [3, 5])
@pytest.mark.parametrize("input_dim", [1, 2])
@pytest.mark.parametrize("output_dim", [1, 2, 10])
@pytest.mark.parametrize("include_bias", [True, False])
class TestWeightSpaceGaussianProcess:

    # def test_init(self):
    @pytest.fixture
    def instantiate_prior(
        self, n_basis_funcs, input_dim, output_dim, include_bias, torus_basis_funcs
    ):
        return WeightSpaceGaussianProcess(
            basis=torus_basis_funcs,
            input_dim=input_dim,
            output_dim=output_dim,
            include_bias=include_bias,
        )

    def test_sample_weights_shape(
        self,
        n_basis_funcs,
        input_dim,
        output_dim,
        include_bias,
        instantiate_prior,
        torus_basis_funcs,
    ):
        prior = instantiate_prior
        key = jxr.PRNGKey(0)
        weights = prior.sample_weights(key)
        expected_shape = (prior.n_basis_funcs, output_dim, input_dim + include_bias)
        assert (
            weights.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {weights.shape}"

    @pytest.mark.parametrize("n_steps", [1, 100])
    def test_call(
        self,
        n_steps,
        n_basis_funcs,
        input_dim,
        output_dim,
        include_bias,
        instantiate_prior,
        torus_basis_funcs,
    ):
        prior = instantiate_prior
        key = jxr.PRNGKey(0)
        weights = prior.sample_weights(key)
        conditions = jnp.cumsum(0.4 * jxr.normal(key, shape=(n_steps,))) % (2 * jnp.pi)
        out = prior(weights, conditions)
        expected_shape = (n_steps, output_dim, input_dim + include_bias)
        assert (
            out.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {out.shape}"

        phi = prior.evaluate_basis(conditions)
        expected = jnp.zeros(expected_shape)
        for i in range(output_dim):
            for j in range(input_dim + include_bias):
                expected = expected.at[:, i, j].set(phi @ weights[:, i, j])

        assert jnp.allclose(out, expected), "Output does not match manual computation"
