import jax
from clds.models import CLDS2, WeightSpaceGaussianProcess
from clds.params import ParamsCLDS2
import pytest
import jax.numpy as jnp
import jax.random as jxr
from clds import utils
import numpy as np
from dynamax.linear_gaussian_ssm.inference import make_lgssm_params, lgssm_smoother
from dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
from functools import partial

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def torus_basis_funcs():
    n_basis_funcs = 5
    _sigma, _kappa, _period = 0.5, 0.1, 2 * jnp.pi
    return utils.Tm_basis(
        n_basis_funcs, M_conditions=1, sigma=_sigma, kappa=_kappa, period=_period
    )


# @pytest.fixture
# def sample_parameters(
#     key, n_steps, state_dim, emission_dim, use_dynamics_bias, use_emissions_bias
# ):
#     As = jxr.normal(key, (n_steps, state_dim, state_dim))
#     if use_dynamics_bias:
#         bs = jxr.normal(key, (n_steps, state_dim))
#     else:
#         bs = jnp.zeros((n_steps, state_dim))
#     Q = jnp.eye(state_dim)
#     Cs = jxr.normal(key, (n_steps, emission_dim, state_dim))
#     if use_emissions_bias:
#         ds = jxr.normal(key, (n_steps, emission_dim))
#     else:
#         ds = jnp.zeros((n_steps, emission_dim))
#     R = jnp.eye(emission_dim)
#     m0 = jnp.zeros((state_dim,))
#     S0 = jnp.eye(state_dim)
#     return As, bs, Q, Cs, ds, R, m0, S0


# def instantiate_static_model_and_params(
#     As,
#     bs,
#     Q,
#     Cs,
#     ds,
#     R,
#     m0,
#     S0,
#     state_dim,
#     emission_dim,
#     use_dynamics_bias,
#     use_emissions_bias,
# ):
#     params = ParamsCLDS2(
#         dynamics_weights=As,
#         dynamics_bias=bs,
#         dynamics_cov=Q,
#         emissions_weights=Cs,
#         emissions_bias=ds,
#         emissions_cov=R,
#         initial_mean=m0,
#         initial_cov=S0,
#     )
#     model = CLDS2(
#         basis=None,
#         state_dim=state_dim,
#         emission_dim=emission_dim,
#         use_dynamics_prior=False,
#         use_emissions_prior=False,
#         use_initial_prior=False,
#         use_dynamics_bias=use_dynamics_bias,
#         use_emissions_bias=use_emissions_bias,
#     )
#     model.params = params
#     return model, params


def initiate_dynamax_model_and_params(
    As,
    bs,
    Q,
    Cs,
    ds,
    R,
    m0,
    S0,
    state_dim,
    emission_dim,
    use_dynamics_bias,
    use_emissions_bias,
):
    dynamax_params = make_lgssm_params(
        initial_mean=m0,
        initial_cov=S0,
        dynamics_weights=As,
        dynamics_cov=Q,
        dynamics_bias=bs,
        emissions_weights=Cs,
        emissions_cov=R,
        emissions_bias=ds,
    )
    dynamax_model = LinearGaussianSSM(
        state_dim=state_dim,
        emission_dim=emission_dim,
        has_dynamics_bias=use_dynamics_bias,
        has_emissions_bias=use_emissions_bias,
    )
    return dynamax_model, dynamax_params


class TestCLDS:

    @pytest.mark.parametrize("state_dim", [1, 2])
    @pytest.mark.parametrize("emission_dim", [1, 10])
    @pytest.mark.parametrize("use_dynamics_prior", [True, False])
    @pytest.mark.parametrize("use_emissions_prior", [True, False])
    @pytest.mark.parametrize("use_initial_prior", [True, False])
    @pytest.mark.parametrize("use_dynamics_bias", [True, False])
    @pytest.mark.parametrize("use_emissions_bias", [True, False])
    class TestCLDSInit:
        """Tests for CLDS initialization and parameter setting."""

        @pytest.fixture
        def initialize_model(
            self,
            state_dim,
            emission_dim,
            use_dynamics_prior,
            use_emissions_prior,
            use_initial_prior,
            use_dynamics_bias,
            use_emissions_bias,
            torus_basis_funcs,
        ):
            model = CLDS2(
                state_dim=state_dim,
                emission_dim=emission_dim,
                basis=torus_basis_funcs,
                use_dynamics_prior=use_dynamics_prior,
                use_emissions_prior=use_emissions_prior,
                use_initial_prior=use_initial_prior,
                use_dynamics_bias=use_dynamics_bias,
                use_emissions_bias=use_emissions_bias,
            )
            return model

        def test_initialize_model(
            self,
            state_dim,
            emission_dim,
            use_dynamics_prior,
            use_emissions_prior,
            use_initial_prior,
            use_dynamics_bias,
            use_emissions_bias,
            torus_basis_funcs,
            initialize_model,
        ):
            """Test parameter setting and prior initialization"""
            model = initialize_model
            assert model.state_dim == state_dim
            assert model.emission_dim == emission_dim
            assert model.use_dynamics_prior == use_dynamics_prior
            assert model.use_emissions_prior == use_emissions_prior
            assert model.use_initial_prior == use_initial_prior
            assert model.use_dynamics_bias == use_dynamics_bias
            assert model.use_emissions_bias == use_emissions_bias

            if use_dynamics_prior:
                assert isinstance(model.priors["dynamics"], WeightSpaceGaussianProcess)
                assert model.priors["dynamics"].output_dim == state_dim
                assert model.priors["dynamics"].input_dim == state_dim
            else:
                assert model.priors["dynamics"] is None

            if use_emissions_prior:
                assert isinstance(model.priors["emissions"], WeightSpaceGaussianProcess)
                assert model.priors["emissions"].output_dim == emission_dim
                assert model.priors["emissions"].input_dim == state_dim
            else:
                assert model.priors["emissions"] is None

            if use_initial_prior:
                assert isinstance(model.priors["init"], WeightSpaceGaussianProcess)
                assert model.priors["init"].output_dim == state_dim
                assert model.priors["init"].input_dim == 1
            else:
                assert model.priors["init"] is None

        @pytest.mark.parametrize("n_steps", [10, 100])
        def test_initialize_params(
            self,
            state_dim,
            emission_dim,
            use_dynamics_prior,
            use_emissions_prior,
            use_initial_prior,
            use_dynamics_bias,
            use_emissions_bias,
            torus_basis_funcs,
            initialize_model,
            n_steps,
        ):
            model = initialize_model
            params = model.initialize_params(jxr.PRNGKey(0), n_steps)
            assert isinstance(params, ParamsCLDS2)
            # check dynamics
            if use_dynamics_prior:
                n_basis_funcs = model.priors["dynamics"].n_basis_funcs
                assert params.dynamics_weights.shape == (
                    n_basis_funcs,
                    state_dim,
                    state_dim,
                )
                if use_dynamics_bias:
                    assert params.dynamics_bias.shape == (n_basis_funcs, state_dim, 1)
                else:
                    assert params.dynamics_bias.shape == (n_steps, state_dim)
                    assert jnp.all(params.dynamics_bias == 0)
            else:
                assert params.dynamics_weights.shape == (n_steps, state_dim, state_dim)
                assert params.dynamics_bias.shape == (n_steps, state_dim)
                if not use_dynamics_bias:
                    assert jnp.all(params.dynamics_bias == 0)
            assert params.dynamics_cov.shape == (state_dim, state_dim)

            # check emissions
            if use_emissions_prior:
                n_basis_funcs = model.priors["emissions"].n_basis_funcs
                assert params.emissions_weights.shape == (
                    n_basis_funcs,
                    emission_dim,
                    state_dim,
                )
                if use_emissions_bias:
                    assert params.emissions_bias.shape == (
                        n_basis_funcs,
                        emission_dim,
                        1,
                    )
                else:
                    assert params.emissions_bias.shape == (n_steps, emission_dim)
                    assert jnp.all(params.emissions_bias == 0)
            else:
                assert params.emissions_weights.shape == (
                    n_steps,
                    emission_dim,
                    state_dim,
                )
                assert params.emissions_bias.shape == (n_steps, emission_dim)
                if not use_emissions_bias:
                    assert jnp.all(params.emissions_bias == 0)
            assert params.emissions_cov.shape == (emission_dim, emission_dim)

            # check initial
            if use_initial_prior:
                n_basis_funcs = model.priors["init"].n_basis_funcs
                assert params.initial_mean.shape == (n_basis_funcs, state_dim, 1)
            else:
                assert params.initial_mean.shape == (state_dim,)
            assert params.initial_cov.shape == (state_dim, state_dim)

    @pytest.mark.parametrize("key", [jxr.PRNGKey(0), jxr.PRNGKey(2)])
    @pytest.mark.parametrize("n_steps", [10, 100])
    @pytest.mark.parametrize("state_dim", [2, 5])
    @pytest.mark.parametrize("emission_dim", [2, 10])
    @pytest.mark.parametrize("use_dynamics_bias", [True, False])
    @pytest.mark.parametrize("use_emissions_bias", [True, False])
    class TestCLDSStatic:
        """Tests for CLDS functions with parameters that do not depend on gaussian process prior."""

        @pytest.fixture
        def instantiate_static_model_and_params(
            self,
            key,
            n_steps,
            state_dim,
            emission_dim,
            use_dynamics_bias,
            use_emissions_bias,
        ):
            # initialize random parameters
            As = jxr.normal(key, (n_steps, state_dim, state_dim))
            if use_dynamics_bias:
                bs = jxr.normal(key, (n_steps, state_dim))
            else:
                bs = jnp.zeros((n_steps, state_dim))
            Q = jnp.eye(state_dim)
            Cs = jxr.normal(key, (n_steps, emission_dim, state_dim))
            if use_emissions_bias:
                ds = jxr.normal(key, (n_steps, emission_dim))
            else:
                ds = jnp.zeros((n_steps, emission_dim))
            R = jnp.eye(emission_dim)
            m0 = jnp.zeros((state_dim,))
            S0 = jnp.eye(state_dim)

            # set up model and params
            params = ParamsCLDS2(
                dynamics_weights=As,
                dynamics_bias=bs,
                dynamics_cov=Q,
                emissions_weights=Cs,
                emissions_bias=ds,
                emissions_cov=R,
                initial_mean=m0,
                initial_cov=S0,
            )
            model = CLDS2(
                basis=None,
                state_dim=state_dim,
                emission_dim=emission_dim,
                use_dynamics_prior=False,
                use_emissions_prior=False,
                use_initial_prior=False,
                use_dynamics_bias=use_dynamics_bias,
                use_emissions_bias=use_emissions_bias,
            )
            model.params = params
            return model, params, (As, bs, Q, Cs, ds, R, m0, S0)

        def test_sample(
            self,
            key,
            n_steps,
            state_dim,
            emission_dim,
            use_dynamics_bias,
            use_emissions_bias,
            instantiate_static_model_and_params,
        ):
            """Test sampling from the model"""
            model, params, (As, bs, Q, Cs, ds, R, m0, S0) = (
                instantiate_static_model_and_params
            )

            subkeys = jxr.split(key, num=(As.shape[0], 2))
            expected_xs = np.zeros((n_steps + 1, state_dim))
            expected_xs[0] = m0 + jxr.multivariate_normal(key, m0, S0)
            expected_ys = np.zeros((n_steps, emission_dim))
            for i in range(n_steps):
                dy_key, em_key = subkeys[i]
                expected_xs[i + 1] = (
                    As[i] @ expected_xs[i]
                    + bs[i]
                    + jxr.multivariate_normal(dy_key, jnp.zeros(Q.shape[0]), Q)
                )
                expected_ys[i] = (
                    Cs[i] @ expected_xs[i]
                    + ds[i]
                    + jxr.multivariate_normal(em_key, jnp.zeros(R.shape[0]), R)
                )

            xs, ys = CLDS2.run_dynamics(key, As, bs, Q, Cs, ds, R, m0, S0)
            assert jnp.allclose(xs, expected_xs[:-1])
            assert jnp.allclose(ys, expected_ys)

            model.params = params
            model_xs, model_ys = model.sample(jnp.zeros(n_steps), key)
            assert jnp.allclose(model_xs, expected_xs[:-1])
            assert jnp.allclose(model_ys, expected_ys)

        def test_e_step_dynamax(
            self,
            key,
            n_steps,
            state_dim,
            emission_dim,
            use_dynamics_bias,
            use_emissions_bias,
            instantiate_static_model_and_params,
        ):
            """Test E-step using dynamax implementation"""

            # CLDS setup
            model, params, args = instantiate_static_model_and_params
            _, ys = CLDS2.run_dynamics(key, *args)

            (init_stats, _, dynamics_stats, _, emission_stats, _), marginal_ll = (
                model.e_step(params, ys, jnp.zeros(n_steps))
            )

            # dynamax setup
            dynamax_model, dynamax_params = initiate_dynamax_model_and_params(
                *args,
                state_dim,
                emission_dim,
                use_dynamics_bias,
                use_emissions_bias,
            )
            (ex_init_stats, ex_dynamics_stats, ex_emission_stats), ex_marginal_ll = (
                dynamax_model.e_step(dynamax_params, ys)
            )

            assert all(
                [jnp.allclose(x, ex) for x, ex in zip(init_stats, ex_init_stats)]
            )
            assert all(
                [
                    jnp.allclose(x, ex)
                    for x, ex in zip(dynamics_stats, ex_dynamics_stats)
                ]
            )
            assert all(
                [
                    jnp.allclose(x, ex)
                    for x, ex in zip(emission_stats, ex_emission_stats)
                ]
            )
            assert jnp.isclose(marginal_ll, ex_marginal_ll)

        def test_m_step_dynamax(
            self,
            key,
            n_steps,
            state_dim,
            emission_dim,
            use_dynamics_bias,
            use_emissions_bias,
            instantiate_static_model_and_params,
        ):
            """Test M-step using dynamax implementation"""
            model, params, args = instantiate_static_model_and_params
            _, ys = CLDS2.run_dynamics(key, *args)

            # reshape for single batch
            batch_stats, _ = jax.vmap(partial(model.e_step, params))(
                ys[None, ...], jnp.zeros((1, n_steps))
            )
            next_params = model.m_step(params, batch_stats)

            dynamax_model, dynamax_params = initiate_dynamax_model_and_params(
                *args,
                state_dim,
                emission_dim,
                use_dynamics_bias,
                use_emissions_bias,
            )
            # reshape for single batch
            batch_stats, ex_marginal_ll = jax.vmap(
                partial(dynamax_model.e_step, dynamax_params)
            )(ys[None, ...])
            next_dynamax_params, _ = dynamax_model.m_step(
                dynamax_params, None, batch_stats, None
            )

            assert jnp.allclose(
                next_params.initial_mean, next_dynamax_params.initial.mean
            )
            assert jnp.allclose(
                next_params.initial_cov, next_dynamax_params.initial.cov
            )
            assert jnp.allclose(
                next_params.dynamics_weights, next_dynamax_params.dynamics.weights
            )
            if use_dynamics_bias:
                assert jnp.allclose(
                    next_params.dynamics_bias.squeeze(),
                    next_dynamax_params.dynamics.bias,
                )
            assert jnp.allclose(
                next_params.dynamics_cov, next_dynamax_params.dynamics.cov
            )
            assert jnp.allclose(
                next_params.emissions_weights, next_dynamax_params.emissions.weights
            )
            if use_emissions_bias:
                assert jnp.allclose(
                    next_params.emissions_bias.squeeze(),
                    next_dynamax_params.emissions.bias,
                )
            assert jnp.allclose(
                next_params.emissions_cov, next_dynamax_params.emissions.cov
            )

    @pytest.mark.parametrize("key", [jxr.PRNGKey(0), jxr.PRNGKey(2)])
    @pytest.mark.parametrize("n_steps", [10, 100])
    @pytest.mark.parametrize("state_dim", [2, 3])
    @pytest.mark.parametrize("emission_dim", [2, 5])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_e_step_identity_basis(
        self, key, n_steps, state_dim, emission_dim, use_bias
    ):
        """Test that GP dynamics with identity basis and intercept input gives same result as non-gp dynamics"""
        emission_dim = 2
        inputs = jnp.ones(n_steps)
        basis_funcs = [lambda x: x]
        model = CLDS2(
            state_dim=state_dim,
            emission_dim=emission_dim,
            basis=basis_funcs,
            use_dynamics_prior=True,
            use_emissions_prior=True,
            use_initial_prior=True,
            use_dynamics_bias=use_bias,
            use_emissions_bias=use_bias,
        )
        model.params = ParamsCLDS2(
            dynamics_weights=jnp.ones((1, state_dim, state_dim)),
            dynamics_cov=jnp.eye(state_dim),
            dynamics_bias=jnp.ones((1, state_dim, 1)),
            emissions_weights=jnp.ones((1, emission_dim, state_dim)),
            emissions_cov=jnp.eye(emission_dim),
            emissions_bias=jnp.ones((1, emission_dim, 1)),
            initial_mean=jnp.ones((1, state_dim, 1)),
            initial_cov=jnp.eye(state_dim),
        )
        _, ys = model.sample(inputs, key)
        (
            init_stats,
            init_gp_stats,
            dynamics_stats,
            dynamics_gp_stats,
            emissions_states,
            emissions_gp_stats,
        ), _ = model.e_step(model.params, ys, inputs)
        # dynamics
        assert jnp.allclose(dynamics_stats[0], dynamics_gp_stats[0])
        assert jnp.allclose(dynamics_stats[1], dynamics_gp_stats[2])
        assert jnp.allclose(dynamics_stats[2], dynamics_gp_stats[3])
        # emissions
        assert jnp.allclose(emissions_states[0], emissions_gp_stats[0])
        assert jnp.allclose(emissions_states[1], emissions_gp_stats[2])
        assert jnp.allclose(emissions_states[2], emissions_gp_stats[3])
        # initial
        assert jnp.allclose(init_stats[0], init_gp_stats[0])
