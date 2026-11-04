import jax
from clds.models import CLDS, WeightSpaceGaussianProcess
from clds.params import ParamsCLDS
import pytest
import jax.numpy as jnp
import jax.random as jxr
from clds import utils
import numpy as np
# from dynamax.linear_gaussian_ssm.inference import make_lgssm_params, lgssm_smoother
# from dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
import copy
import itertools
from functools import partial

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def torus_basis_funcs():
    n_basis_funcs = 5
    _sigma, _kappa, _period = 0.5, 0.1, 2 * jnp.pi
    return utils.Tm_basis(
        n_basis_funcs, M_conditions=1, sigma=_sigma, kappa=_kappa, period=_period
    )


def _disables_prior_bias(gp_priors, use_dynamics_bias, use_emissions_bias):
    """Whether gp_priors puts a GP prior on a disabled bias, which CLDS rejects."""
    return ("b" in gp_priors and not use_dynamics_bias) or ("d" in gp_priors and not use_emissions_bias)


# (gp_priors, use_dynamics_bias, use_emissions_bias)
PRIOR_BIAS_CONFIGS = list(
    itertools.product(
        [["A"], ["A", "b"], ["A", "C"], ["A", "b", "C"], ["A", "m"], ["A", "b", "m"], ["A", "C", "m"], ["A", "b", "C", "m"]],
        [True, False],
        [True, False],
    )
)


class TestCLDS:

    @pytest.mark.parametrize("state_dim", [1, 2])
    @pytest.mark.parametrize("emission_dim", [1, 10])
    @pytest.mark.parametrize(
        "gp_priors,use_dynamics_bias,use_emissions_bias",
        [c for c in PRIOR_BIAS_CONFIGS if not _disables_prior_bias(*c)],
    )
    class TestCLDSInit:
        """Tests for CLDS initialization and parameter setting."""

        @pytest.fixture
        def initialize_model(
            self,
            state_dim,
            emission_dim,
            gp_priors,
            use_dynamics_bias,
            use_emissions_bias,
            torus_basis_funcs,
        ):
            model = CLDS(
                state_dim=state_dim,
                emission_dim=emission_dim,
                basis=torus_basis_funcs,
                gp_priors=gp_priors,
                use_dynamics_bias=use_dynamics_bias,
                use_emissions_bias=use_emissions_bias,
            )
            return model

        def test_initialize_model(
            self,
            state_dim,
            emission_dim,
            gp_priors,
            use_dynamics_bias,
            use_emissions_bias,
            torus_basis_funcs,
            initialize_model,
        ):
            """Test parameter setting and prior initialization"""
            model = initialize_model
            assert model.state_dim == state_dim
            assert model.emission_dim == emission_dim
            assert model.gp_priors == set(gp_priors)
            assert model.use_dynamics_bias == use_dynamics_bias
            assert model.use_emissions_bias == use_emissions_bias

            if "A" in model.gp_priors:
                assert isinstance(model.priors["A"], WeightSpaceGaussianProcess)
                assert model.priors["A"].output_dim == state_dim
                assert model.priors["A"].input_dim == state_dim
            else:
                assert model.priors["A"] is None

            if "C" in model.gp_priors:
                assert isinstance(model.priors["C"], WeightSpaceGaussianProcess)
                assert model.priors["C"].output_dim == emission_dim
                assert model.priors["C"].input_dim == state_dim
            else:
                assert model.priors["C"] is None

            if "m" in model.gp_priors:
                assert isinstance(model.priors["m"], WeightSpaceGaussianProcess)
                assert model.priors["m"].output_dim == state_dim
                assert model.priors["m"].input_dim == 1
            else:
                assert model.priors["m"] is None

        def test_initialize_params(
            self,
            state_dim,
            emission_dim,
            gp_priors,
            use_dynamics_bias,
            use_emissions_bias,
            torus_basis_funcs,
            initialize_model,
        ):
            model = initialize_model
            params = model.initialize_params(jxr.PRNGKey(0))
            assert isinstance(params, ParamsCLDS)
            # check dynamics
            if "A" in model.gp_priors:
                n_basis_funcs = model.priors["A"].n_basis_funcs
                assert params.dynamics_weights.shape == (
                    n_basis_funcs,
                    state_dim,
                    state_dim,
                )
            else:
                assert params.dynamics_weights.shape == (state_dim, state_dim)

            # check bias
            if "b" in model.gp_priors and use_dynamics_bias:
                assert params.dynamics_bias.shape == (n_basis_funcs, state_dim, 1)
            elif use_dynamics_bias:
                assert params.dynamics_bias.shape == (state_dim,)
            else:
                assert params.dynamics_bias.shape == (state_dim,)
                assert jnp.all(params.dynamics_bias == 0)

            assert params.dynamics_cov.shape == (state_dim, state_dim)

            # check emissions
            if "C" in model.gp_priors:
                n_basis_funcs = model.priors["C"].n_basis_funcs
                assert params.emissions_weights.shape == (
                    n_basis_funcs,
                    emission_dim,
                    state_dim,
                )
            else:
                assert params.emissions_weights.shape == (
                    emission_dim,
                    state_dim,
                )
            
            if "d" in model.gp_priors and use_emissions_bias:
                assert params.emissions_bias.shape == (
                    n_basis_funcs,
                    emission_dim,
                    1,
                )
            elif use_emissions_bias:
                assert params.emissions_bias.shape == (emission_dim,)
            else:
                assert params.emissions_bias.shape == (emission_dim,)
                assert jnp.all(params.emissions_bias == 0)
            assert params.emissions_cov.shape == (emission_dim, emission_dim)

            # check initial
            if "m" in model.gp_priors:
                n_basis_funcs = model.priors["m"].n_basis_funcs
                assert params.initial_mean.shape == (n_basis_funcs, state_dim, 1)
            else:
                assert params.initial_mean.shape == (state_dim,)
            assert params.initial_cov.shape == (state_dim, state_dim)

    @pytest.mark.parametrize(
        "gp_priors,use_dynamics_bias,use_emissions_bias",
        [c for c in PRIOR_BIAS_CONFIGS if _disables_prior_bias(*c)],
    )
    def test_init_disabled_bias_raises(
        self, gp_priors, use_dynamics_bias, use_emissions_bias, torus_basis_funcs
    ):
        """Test that a GP prior on a disabled bias raises"""
        with pytest.raises(ValueError, match="which are disabled"):
            CLDS(
                state_dim=2,
                emission_dim=3,
                basis=torus_basis_funcs,
                gp_priors=gp_priors,
                use_dynamics_bias=use_dynamics_bias,
                use_emissions_bias=use_emissions_bias,
            )

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
            params = ParamsCLDS(
                dynamics_weights=As,
                dynamics_bias=bs,
                dynamics_cov=Q,
                emissions_weights=Cs,
                emissions_bias=ds,
                emissions_cov=R,
                initial_mean=m0,
                initial_cov=S0,
            )
            model = CLDS(
                basis=None,
                state_dim=state_dim,
                emission_dim=emission_dim,
                gp_priors=[],
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

            xs, ys = CLDS.run_dynamics(key, As, bs, Q, Cs, ds, R, m0, S0)
            assert jnp.allclose(xs, expected_xs[:-1])
            assert jnp.allclose(ys, expected_ys)

            model.params = params
            model_xs, model_ys = model.sample(jnp.zeros(n_steps), key)
            assert jnp.allclose(model_xs, expected_xs[:-1])
            assert jnp.allclose(model_ys, expected_ys)

    @pytest.mark.parametrize("key", [jxr.PRNGKey(0), jxr.PRNGKey(2)])
    @pytest.mark.parametrize("n_steps", [10, 100])
    @pytest.mark.parametrize("state_dim", [2, 3])
    @pytest.mark.parametrize("emission_dim", [2, 5])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_e_step_identity_basis(
        self, key, n_steps, state_dim, emission_dim, use_bias
    ):
        """Test that GP coefficients with identity basis and constant inputs (phi(u) = 1)
        give the same e-step as static coefficients"""
        inputs = jnp.ones(n_steps)
        model_kwargs = dict(
            state_dim=state_dim,
            emission_dim=emission_dim,
            use_dynamics_bias=use_bias,
            use_emissions_bias=use_bias,
        )
        static_model = CLDS(basis=None, gp_priors=[], **model_kwargs)
        gp_model = CLDS(
            basis=[lambda x: x],
            gp_priors=["A", "C", "m"] + (["b", "d"] if use_bias else []),
            **model_kwargs,
        )

        # static coefficients, and the same coefficients as single-basis GP weights
        k_A, k_b, k_C, k_d, k_m, k_sample = jxr.split(key, 6)
        A = 0.5 * jxr.normal(k_A, (state_dim, state_dim)) / jnp.sqrt(state_dim)
        b = jxr.normal(k_b, (state_dim,)) if use_bias else jnp.zeros(state_dim)
        C = jxr.normal(k_C, (emission_dim, state_dim))
        d = jxr.normal(k_d, (emission_dim,)) if use_bias else jnp.zeros(emission_dim)
        m = jxr.normal(k_m, (state_dim,))
        covs = dict(
            dynamics_cov=jnp.eye(state_dim),
            emissions_cov=jnp.eye(emission_dim),
            initial_cov=jnp.eye(state_dim),
        )
        static_params = ParamsCLDS(
            dynamics_weights=A,
            dynamics_bias=b,
            emissions_weights=C,
            emissions_bias=d,
            initial_mean=m,
            **covs,
        )
        gp_params = ParamsCLDS(
            dynamics_weights=A[None],
            dynamics_bias=b[None, :, None] if use_bias else b,
            emissions_weights=C[None],
            emissions_bias=d[None, :, None] if use_bias else d,
            initial_mean=m[None, :, None],
            **covs,
        )
        static_model.params = static_params
        _, ys = static_model.sample(inputs, k_sample)

        static_stats, static_ll = static_model.e_step(static_params, ys, inputs)
        gp_stats, gp_ll = gp_model.e_step(gp_params, ys, inputs)

        assert jnp.isclose(static_ll, gp_ll)
        for block in ["initial", "dynamics", "emissions"]:
            for s_static, s_gp in zip(static_stats[block], gp_stats[block]):
                assert jnp.allclose(s_static, s_gp), block

    @pytest.mark.parametrize("state_dim", [2, 3])
    @pytest.mark.parametrize("emission_dim", [2, 5])
    @pytest.mark.parametrize("use_prior", [True, False])
    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize(
        "mask",
        [
            jnp.ones(100, dtype=bool),
            jnp.concatenate([jnp.ones(50, dtype=bool), jnp.zeros(50, dtype=bool)]),
            # jnp.concatenate([jnp.zeros(50, dtype=bool), jnp.ones(50, dtype=bool)]),
        ],
    )
    def test_mask_e_step(
        self, mask, state_dim, emission_dim, use_prior, use_bias, torus_basis_funcs
    ):
        """Test that masking works as expected. E-step can only mask the ends of sequences."""
        key = jxr.PRNGKey(0)
        n_steps = 100
        inputs = jnp.cumsum(0.4 * jxr.normal(key, shape=(n_steps,))) % (2 * jnp.pi)
        gp_priors = []
        if use_prior:
            gp_priors += ["A", "C", "m"]
            if use_bias:
                gp_priors += ["b", "d"]
        model = CLDS(
            state_dim=state_dim,
            emission_dim=emission_dim,
            basis=torus_basis_funcs if use_prior else None,
            gp_priors=gp_priors,
            use_dynamics_bias=use_bias,
            use_emissions_bias=use_bias,
        )
        model.params = model.initialize_params(key)
        _, ys = model.sample(inputs, key)

        # with jax.disable_jit():
        stats_masked, ll_masked = model.e_step(model.params, ys, inputs, mask=mask)
        stats_idx, ll_idx = model.e_step(model.params, ys[mask], inputs[mask])
        for block in ["initial", "dynamics", "emissions"]:
            for m, i in zip(stats_masked[block], stats_idx[block]):
                assert jnp.allclose(m, i), block
        assert jnp.isclose(ll_masked, ll_idx)
        # assert jnp.isclose(ll_masked, ll_idx)

    @pytest.mark.parametrize("state_dim", [2, 3])
    @pytest.mark.parametrize("emission_dim", [2, 5])
    @pytest.mark.parametrize("use_prior", [True, False])
    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize(
        "mask",
        [
            jnp.ones(100, dtype=bool),
            jnp.concatenate([jnp.ones(50, dtype=bool), jnp.zeros(50, dtype=bool)]),
            jnp.concatenate([jnp.zeros(50, dtype=bool), jnp.ones(50, dtype=bool)]),
        ],
    )
    def test_mask_fit(
        self, mask, state_dim, emission_dim, use_prior, use_bias, torus_basis_funcs
    ):
        """Test that masking works as expected"""
        key = jxr.PRNGKey(0)
        n_steps = 100
        inputs = jnp.cumsum(0.4 * jxr.normal(key, shape=(n_steps,))) % (2 * jnp.pi)
        gp_priors = ["A"]
        if use_prior:
            gp_priors += ["C", "m"]
            if use_bias:
                gp_priors += ["b", "d"]
        model = CLDS(
            state_dim=state_dim,
            emission_dim=emission_dim,
            basis=torus_basis_funcs,
            gp_priors=gp_priors,
            use_dynamics_bias=use_bias,
            use_emissions_bias=use_bias,
        )
        params = model.initialize_params(key)
        model.params = copy.deepcopy(params)
        _, ys = model.sample(inputs, key)

        # with jax.disable_jit():
        params_masked, ll_masked = model.fit(
            ys[None],
            inputs[None],
            initial_params=params,
            mask=mask[None],
            num_iters=1,
            data_driven_init=False
        )
        params_idx, ll_idx = model.fit(
            ys[None, mask],
            inputs[None, mask],
            initial_params=params,
            num_iters=1,
            data_driven_init=False
        )
        for p_masked, p_idx in zip(params_masked, params_idx):
            if p_masked is None:
                assert p_idx is None
            else:
                assert jnp.allclose(p_masked, p_idx)
        # assert jnp.isclose(ll_masked[0], ll_idx[0])
