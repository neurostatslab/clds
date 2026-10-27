# -*- coding: utf-8 -*-
"""
@author: Amin, Victor
"""

# %%
import jax
import jax.numpy as jnp
import jax.random as jxr

from jaxtyping import Array, Float
from typing import Optional, Tuple, Dict, Iterable
import warnings

from .utils import logprob_analytic
from functools import partial
from jax import jit, lax, vmap
from tqdm.auto import trange
import logging

logging.basicConfig(
    level=logging.INFO, format="[%(filename)s][%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from .params import (
    ParamsEmission,
    ParamsNormalLikelihood,
    ParamsGPLDS,
    ParamsCLDS,
    ParamsCLDS2,
)

import clds.utils as utils

# %%
class WeightSpaceGaussianProcess:
    r"""
    Weight-space Gaussian Process prior for matrix-valued random functions
        A_ij(u) = \sum_l w^{(ij)} \phi_l(u),       w^{(ij)} ~ N(0, 1)
    where w are the weights and \phi_l are the basis functions.

    Constants:
        L: number of basis functions
        D1: output dimension
        D2: input dimension
        M: dimension of u, the "conditions"
    """

    def __init__(
        self,
        basis: list,
        input_dim: int = 1,
        output_dim: int = 1,
        include_bias: bool = False,
    ):
        self.basis = basis
        self.n_basis_funcs = len(basis)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.include_bias = include_bias

    def __call__(
        self,
        weights: Float[Array, "n_basis_funcs output_dim input_dim"],
        conditions: Float[Array, "n_steps n_conditions"],
    ) -> Float[Array, "n_steps output_dim input_dim"]:
        r"""
        Evaluate A_ij(u) = \sum_l w^{(ij)} \phi_l(u) at the M-dimensional points u in `conditions`
        with `weights` w^{(ij)} and basis functions \phi_l.
        """
        PhiX = self.evaluate_basis(conditions)
        return jnp.einsum("lij,tl->tij", weights, PhiX)

    def sample_weights(
        self, key: jxr.PRNGKey
    ) -> Float[Array, "n_basis_funcs output_dim input_dim"]:
        return jxr.normal(
            key,
            shape=(
                self.n_basis_funcs,
                self.output_dim,
                self.input_dim + self.include_bias,
            ),
        )

    def evaluate_basis(
        self, u: Float[Array, "n_steps n_conditions"]
    ) -> Float[Array, "n_steps n_basis_funcs"]:
        return jnp.array([jax.vmap(f)(u) for f in self.basis]).T

    def sample(
        self, key: jxr.PRNGKey, conditions: Float[Array, "n_steps n_conditions"]
    ) -> Float[Array, "n_steps output_dim input_dim"]:
        """
        Sample from the GP prior at the points `conditions`
        """
        weights = self.sample_weights(key)
        return self.__call__(weights, conditions)

    def log_prob_weights(
        self, weights: Float[Array, "n_basis_funcs output_dim input_dim"]
    ) -> float:
        """
        Standard Gaussian prior N(0,1) on the weights
        """
        return -0.5 * jnp.sum(weights**2) - 0.5 * jnp.asarray(
            weights.shape
        ).prod() * jnp.log(2 * jnp.pi)


class WeightSpaceGaussianProcess2:
    r"""
    Weight-space Gaussian Process prior for matrix-valued random functions
        A_ij(u) = \sum_l w^{(ij)} \phi_l(u),       w^{(ij)} ~ N(0, 1)
    where w are the weights and \phi_l are the basis functions.

    Parameters
    ----------
    basis : list
        List of basis functions.
    input_dim : int, optional
        Dimension of the input, by default 1.
    output_dim : int, optional
        Dimension of the output, by default 1.
    include_bias : bool, optional
        Whether to include a bias term in the basis functions, by default False.
    """

    def __init__(
        self,
        basis: list,
        input_dim: int = 1,
        output_dim: int = 1,
        include_bias: bool = False,
    ):
        self.basis = basis
        self.n_basis_funcs = len(basis.coef)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.include_bias = include_bias

    def __call__(
        self,
        weights: Float[Array, "n_basis_funcs output_dim input_dim"],
        conditions: Float[Array, "n_steps n_conditions"],
    ) -> Float[Array, "n_steps output_dim input_dim"]:
        r"""
        Evaluate A_ij(u) = \sum_l w^{(ij)} \phi_l(u) at the M-dimensional points u in `conditions`
        with `weights` w^{(ij)} and basis functions \phi_l.
        """
        PhiX = self.evaluate_basis(conditions)
        return jnp.einsum("lij,tl->tij", weights, PhiX)

    def sample_weights(
        self, key: jxr.PRNGKey
    ) -> Float[Array, "n_basis_funcs output_dim input_dim"]:
        return jxr.normal(
            key,
            shape=(
                self.n_basis_funcs,
                self.output_dim,
                self.input_dim + self.include_bias,
            ),
        )

    def evaluate_basis(
        self, u: Float[Array, "n_steps n_conditions"]
    ) -> Float[Array, "n_steps n_basis_funcs"]:
        return self.basis.evaluate(u)

    def sample(
        self, key: jxr.PRNGKey, conditions: Float[Array, "n_steps n_conditions"]
    ) -> Float[Array, "n_steps output_dim input_dim"]:
        """
        Sample from the GP prior at the points `conditions`
        """
        weights = self.sample_weights(key)
        return self.__call__(weights, conditions)

    def log_prob_weights(
        self, weights: Float[Array, "n_basis_funcs output_dim input_dim"]
    ) -> float:
        """
        Standard Gaussian prior N(0,1) on the weights
        """
        return -0.5 * jnp.sum(weights**2) - 0.5 * jnp.asarray(
            weights.shape
        ).prod() * jnp.log(2 * jnp.pi)


# %%
class CLDS:
    """
    Conditionally Linear Dynamical System (CLDS) model, with LDS dynamics and
    weight-space view parametrization of the GP priors for the parameters {A, b, C, m0}.

    Init args:
        wgps: dict of WeightSpaceGaussianProcess (wGP) objects for the parameters {A, b, C, m0}.
        state_dim: dimension of the latent state space.
        emission_dim: dimension of the observation space.

    By default: A has wGP prior, whereas b and C are optional.
                If wGP priors are not provided for b and C, they are learned as C fixed, b time-varying.
    This is a early version, only currently supporting EM. Does not support sampling. Does not support inputs other than GP conditions.
    """

    def __init__(self, wgps: dict, state_dim: int, emission_dim: int):
        self.wgps = wgps
        assert "A" in self.wgps, "Dynamics GP prior is required"
        if "b" not in self.wgps:
            self.wgps["b"] = None
        if "C" not in self.wgps:
            self.wgps["C"] = None
        if "m0" not in self.wgps:
            self.wgps["m0"] = None

        self.state_dim = state_dim
        self.emission_dim = emission_dim

    def log_prior(self, params: ParamsCLDS, inputs):
        """Compute the log prior of the parameters. Conditions are inputs"""
        logprior_A = self.wgps["A"].log_prob_weights(params.dynamics_gp_weights)

        if self.wgps["b"] is None:
            logprior_b = 0.0
        else:
            logprior_b = self.wgps["b"].log_prob_weights(params.bias_gp_weights)

        if self.wgps["C"] is None:
            logprior_C = 0.0
        else:
            logprior_C = self.wgps["C"].log_prob_weights(params.emissions_gp_weights)

        if self.wgps["m0"] is None:
            logprior_m0 = 0.0
        else:
            logprior_m0 = self.wgps["m0"].log_prob_weights(params.m0_gp_weights)

        return logprior_A + logprior_b + logprior_C + logprior_m0

    def weights_to_params(self, params, inputs):
        """Transform weights of weight space into parameters.
        Implement as needed for all weight-space GP priors."""
        As = self.wgps["A"](params.dynamics_gp_weights, inputs)
        Cs = (
            self.wgps["C"](params.emissions_gp_weights, inputs)
            if self.wgps["C"] is not None
            else params.Cs
        )
        bs = (
            self.wgps["b"](params.bias_gp_weights, inputs)
            if self.wgps["b"] is not None
            else params.bs
        )
        m0 = (
            self.wgps["m0"](params.m0_gp_weights, inputs)[0]
            if self.wgps["m0"] is not None
            else params.m0
        )  #! Some unnecessary computation, keeping only t=0
        return As, Cs, bs.squeeze(), None, m0.squeeze()

    def smoother(self, params: ParamsCLDS, emissions, inputs):
        """inputs as conditions"""
        # Format params
        As, Cs, bs, _, m0 = self.weights_to_params(params, inputs)
        if Cs.ndim == 2:
            Cs = jnp.tile(Cs[None], (len(inputs), 1, 1))

        # Run the smoother
        lgssm_params = {
            "m0": m0,
            "S0": params.S0,
            "As": As,
            "bs": bs,
            "Q": params.Q,
            "Cs": Cs,
            "R": params.R,
        }
        return utils.lgssm_smoother(**lgssm_params, ys=emissions)

    def e_step(
        self,
        params: ParamsCLDS,
        emissions: Float[Array, "num_timesteps emission_dim"],
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    ):
        def weightspace_stats(
            XTX: Float[Array, "T D2 D2"],
            XTY: Float[Array, "T D2 D1"],
            wgp_prior: WeightSpaceGaussianProcess,
            conditions: Float[Array, "T condition_dim"],
        ) -> tuple:
            """
            Compute the expected sufficient statistics for the weight-space GP prior.
            Provide the sufficient stats X^T X and X^T Y for the problem Y = A(C)X + noise.
            This returns the expanded stats Phi @ X^T X @ Phi^T and Phi @ X^T Y for the basis functions Phi(C).
            """
            _Phi = wgp_prior.evaluate_basis(conditions)

            ZTZ = jnp.einsum("tk,tij,tl->ikjl", _Phi, XTX, _Phi)
            ZTY = jnp.einsum("tk,tim->ikm", _Phi, XTY)

            ZTZ = ZTZ.reshape(
                wgp_prior.n_basis_funcs * wgp_prior.input_dim,
                wgp_prior.n_basis_funcs * wgp_prior.input_dim,
            )
            ZTY = ZTY.reshape(
                wgp_prior.n_basis_funcs * wgp_prior.input_dim, wgp_prior.output_dim
            )
            return (ZTZ, ZTY)

        """take inputs to be theta"""
        num_timesteps = emissions.shape[0]
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 1))

        # Run the smoother to get posterior expectations
        marginal_loglik, filter_results, smoother_results = self.smoother(
            params, emissions, inputs
        )
        smoothed_means, smoothed_covariances, smoothed_cross_covariances = (
            smoother_results
        )

        # shorthand
        Ex = smoothed_means
        Exp = smoothed_means[:-1]
        Exn = smoothed_means[1:]
        Vx = smoothed_covariances
        Vxp = smoothed_covariances[:-1]
        Vxn = smoothed_covariances[1:]
        Expxn = smoothed_cross_covariances

        # Append bias to the inputs
        # inputs = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
        up = inputs[:-1]
        # u = inputs

        # expected sufficient statistics for the initial distribution
        Ex0 = smoothed_means[0]
        Ex0x0T = smoothed_covariances[0] + jnp.outer(Ex0, Ex0)
        init_stats = (Ex0, Ex0x0T, 1)
        if self.wgps["m0"] is None:
            wgpm0_stats = None
        else:
            m0_targets = Ex0.reshape(1, 1, self.state_dim)
            XTX_m0 = jnp.ones((1, 1, 1))

            _cond = (
                up[0].reshape(1) if up[0].ndim == 0 else up[0].reshape(1, up.shape[1])
            )
            wgpm0_stats = weightspace_stats(XTX_m0, m0_targets, self.wgps["m0"], _cond)

        # expected sufficient statistics for the dynamics
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        sum_xpxnT = Expxn.sum(0)
        sum_xpxpT = Vxp.sum(0) + Exp.T @ Exp
        sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
        dynamics_stats = (sum_xpxpT, sum_xpxnT, sum_xnxnT, num_timesteps - 1)

        # Dynamics wGP sufficient stats
        # full E-step sufficient stats
        bs = (
            self.wgps["b"](params.bias_gp_weights, up).squeeze()
            if self.wgps["b"] is not None
            else params.bs
        )
        Expxn_b = Expxn - jnp.einsum("ti,tj->tij", Exp, bs)
        ExpxpT = jnp.einsum("ti,tj->tij", Exp, Exp) + Vxp
        wgpA_stats = weightspace_stats(ExpxpT, Expxn_b, self.wgps["A"], up)
        wgpA_sylvester_stats = (wgpA_stats[0], params.Q, wgpA_stats[1], 1)

        # Q sufficient stats # TODO. Currently uses static sufficient stats
        # Vxn, Vxp = Vx[1:], Vx[:-1]
        # sum_AExpxnT = jnp.einsum('tij,tjk->ik', F, Expxn) #.sum(0)
        # sum_AExpxpAT = jax.vmap(lambda _m, _S, _A: _A @ (_m @ _m.T + _S) @ _A.T)(Exp, Vxp, F).sum(0)
        # Q = (sum_xnxnT - _A_ExpxnT - _A_ExpxnT.T + _A_ExpxpT_A) / (dynamics_stats[-1] - 1)

        # bias sufficient stats
        F = self.wgps["A"](params.dynamics_gp_weights, up)
        bias_targets = Exn - jnp.einsum("tij,tj->ti", F, Exp)
        if self.wgps["b"] is None:
            bias_stats = (bias_targets, 1)
        else:
            bias_targets = bias_targets.reshape(len(up), 1, self.state_dim)
            _XTXb = jnp.ones((len(up), 1, 1))
            bias_stats = weightspace_stats(_XTXb, bias_targets, self.wgps["b"], up)

        # more expected sufficient statistics for the emissions
        y = emissions
        sum_xxT = Vx.sum(0) + Ex.T @ Ex
        sum_xyT = Ex.T @ y
        sum_yyT = emissions.T @ emissions
        emission_stats = (sum_xxT, sum_xyT, sum_yyT, num_timesteps)

        if self.wgps["C"] is None:
            wgpC_stats = None
            wgpC_sylvester_stats = None
        else:
            _xxT = jnp.einsum("ti,tj->tij", Ex, Ex) + Vx
            _xyT = jnp.einsum("ti,tj->tij", Ex, y)
            wgpC_stats = weightspace_stats(_xxT, _xyT, self.wgps["C"], inputs)

            wgpC_sylvester_stats = (wgpC_stats[0], params.R, wgpC_stats[1], 1)

        return (
            init_stats,
            wgpm0_stats,
            dynamics_stats,
            wgpA_stats,
            bias_stats,
            emission_stats,
            wgpC_stats,
            wgpA_sylvester_stats,
            wgpC_sylvester_stats,
        ), marginal_loglik

    def m_step(
        self,
        params: ParamsCLDS,
        batch_stats: Tuple,  # inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None,
    ) -> ParamsCLDS:
        def fit_linear_regression(ExxT, ExyT, EyyT, N):
            # Solve a linear regression given sufficient statistics
            W = utils.psd_solve(ExxT, ExyT).T
            Sigma = (EyyT - W @ ExyT - ExyT.T @ W.T + W @ ExxT @ W.T) / N
            return W, Sigma

        def fit_gplinear_regression(ZTZ, ZTY, wgp_prior):
            # Solve a linear regression in weight-space given sufficient statistics
            weights = jax.scipy.linalg.solve(
                ZTZ + jnp.eye(wgp_prior.n_basis_funcs * wgp_prior.input_dim),
                ZTY,
                assume_a="pos",
            )
            weights = weights.reshape(
                wgp_prior.input_dim, wgp_prior.n_basis_funcs, wgp_prior.output_dim
            ).transpose(1, 2, 0)
            return weights

        def fit_gplinear_regression_sylvester(ZTZ, Sigma, ZTY, wgp_prior):
            # Solve a linear regression in weight-space given sufficient statistics
            # weights = utils.jax_solve_sylvester(B, ZTZ, ZTY, assume_a='pos')
            weights = utils.jax_solve_sylvester_BS(ZTZ, Sigma, ZTY)
            weights = weights.reshape(
                wgp_prior.input_dim, wgp_prior.n_basis_funcs, wgp_prior.output_dim
            ).transpose(1, 2, 0)
            return weights

        # Sum the statistics across all batches
        stats = jax.tree_util.tree_map(partial(jnp.sum, axis=0), batch_stats)
        (
            init_stats,
            wgpm0_stats,
            dynamics_stats,
            wgpA_stats,
            bias_stats,
            emission_stats,
            wgpC_stats,
            wgpA_sylvester_stats,
            wgpC_sylvester_stats,
        ) = stats

        # Perform MLE estimation jointly
        sum_x0, sum_x0x0T, N = init_stats
        S = sum_x0x0T / N - jnp.outer(sum_x0, sum_x0) / (N**2)
        if self.wgps["m0"] is None:
            W_m0 = None
            m = sum_x0 / N
        else:
            W_m0 = fit_gplinear_regression(*wgpm0_stats, self.wgps["m0"])
            m = None

        # Dynamics M-step
        # W_A = fit_gplinear_regression(*wgpA_stats, self.wgps['A'])
        W_A = fit_gplinear_regression_sylvester(
            wgpA_sylvester_stats[0],
            wgpA_sylvester_stats[1] / wgpA_sylvester_stats[3],
            wgpA_sylvester_stats[2],
            wgp_prior=self.wgps["A"],
        )

        # TODO: Q M-step. Currently uses static sufficient stats
        # Vxn, Vxp = Vx[1:], Vx[:-1]
        # _ExnxnT = Exn.T @ Exn + Vxn.sum(0) # Unchanged
        # _A_ExpxnT = jnp.einsum('tij,tjk->ik', F, Expxn)
        # _A_ExpxpT_A = jax.vmap(lambda _m, _S, _A: _A @ (_m @ _m.T + _S) @ _A.T)(Exp, Vxp, F).sum(0)
        # Q = (_ExnxnT - _A_ExpxnT - _A_ExpxnT.T + _A_ExpxpT_A) / (dynamics_stats[-1] - 1)

        F_static, Q = fit_linear_regression(*dynamics_stats)

        # Bias update
        if self.wgps["b"] is None:
            W_b = None
            bs = bias_stats[0] / bias_stats[1]
        else:
            # In weight space
            W_b = fit_gplinear_regression(*bias_stats, self.wgps["b"])
            bs = None
        # # Use the following for homogeneous bias
        # b = jnp.mean(bs, axis=0)
        # bs = jnp.tile(b, (len(up), 1))

        # Emission M-step
        H_static, R = fit_linear_regression(*emission_stats)
        if self.wgps["C"] is None:
            W_C = None
            Cs = H_static
        else:
            # In weight space
            # W_C = fit_gplinear_regression(*wgpC_stats, self.wgps['C'])
            W_C = fit_gplinear_regression_sylvester(
                wgpC_sylvester_stats[0],
                wgpC_sylvester_stats[1] / wgpC_sylvester_stats[3],
                wgpC_sylvester_stats[2],
                wgp_prior=self.wgps["C"],
            )
            Cs = None

        # logger.warning('Warning, fixing Q and R')
        # Q = jnp.eye(self.state_dim) # Can fix Q to be identity (for identifiability)
        # R = jnp.eye(self.emission_dim)
        params = ParamsCLDS(
            m0=m,
            S0=S,
            dynamics_gp_weights=W_A,
            bias_gp_weights=W_b,
            emissions_gp_weights=W_C,
            m0_gp_weights=W_m0,
            Cs=Cs,
            bs=bs,
            Q=utils.symmetrize(Q),
            R=utils.symmetrize(R),
        )
        return params

    def log_prob(self, params, emissions, conditions):
        """Compute the log probability of the emissions given the parameters"""

        def batch_log_prob(_emissions, _conditions):
            log_prior = self.log_prior(params, _conditions)
            marginal_loglik, _, _ = self.smoother(params, _emissions, _conditions)
            return log_prior + marginal_loglik

        return vmap(batch_log_prob)(emissions, conditions).sum()

    def marginal_log_lik(self, params, emissions, conditions):
        """Compute the marginal log likelihood of the emissions given the parameters"""

        def batch_marginal_log_lik(_emissions, _conditions):
            (marginal_loglik, _, _) = self.smoother(params, _emissions, _conditions)
            return marginal_loglik

        return vmap(batch_marginal_log_lik)(emissions, conditions).sum()


# %%
class CLDS2:
    """
    Conditionally Linear Dynamical System (CLDS) model, with LDS dynamics
        x_0     ~ N(m(u_0), S0)
        x_{t+1} = A(u_t) x_t + b(u_t) + N(0, Q)
        y_t     = C(u_t) x_t + d(u_t) + N(0, R)
    and weight-space view parametrization of the GP priors for the coefficients {A, b, C, d, m}.

    Each coefficient listed in `gp_priors` has a weight-space GP prior; all others are
    learned as static parameters.

    Parameters
    ----------
    state_dim : int
        dimension of the latent state space.
    emission_dim : int
        dimension of the observation space.
    basis : list
        list of basis functions for the weight-space GP priors.
    gp_priors : Iterable[str], optional
        - coefficients among {"A", "b", "C", "d", "m"} with a weight-space GP prior, 
            e.g. ["b"] for x_{t+1} = A x_t + b(u_t), y_t = C x_t + d. 
        - "b" (resp. "d") requires use_dynamics_bias (resp. use_emissions_bias). 
        - By default (None), every coefficient of the model has a GP prior.
    use_dynamics_bias : bool, optional
        whether to include a bias term b in the dynamics, by default True.
    use_emissions_bias : bool, optional
        whether to include a bias term d in the emissions, by default False.
    fix_Q : bool, optional
        whether to fix the dynamics covariance Q to the identity (for identifiability),
        by default True. If False, Q is learned in the M-step.
    initial_params : Optional[ParamsCLDS2], optional
        initial parameters for the model, by default None.
    use_dynamics_prior, use_emissions_prior, use_initial_prior : bool, optional
        deprecated, use `gp_priors` instead. Each flag puts a GP prior on (A, b), (C, d)
        and m respectively, and defaults to True when any of them is given.
    """

    # coefficient -> field of ParamsCLDS2
    COEFFICIENTS = {
        "A": "dynamics_weights",
        "b": "dynamics_bias",
        "C": "emissions_weights",
        "d": "emissions_bias",
        "m": "initial_mean",
    }
    # (coefficient on x, bias) of each regression block solved in the M-step
    BLOCKS = {
        "initial": (None, "m"),
        "dynamics": ("A", "b"),
        "emissions": ("C", "d"),
    }

    def __init__(
        self,
        state_dim: int,
        emission_dim: int,
        basis: list,  # soon to be class
        gp_priors: Optional[Iterable[str]] = None,
        use_dynamics_bias: bool = True,
        use_emissions_bias: bool = False,
        fix_Q: bool = True,
        initial_params: Optional[ParamsCLDS2] = None,
        use_dynamics_prior: Optional[bool] = None,  #! depracated
        use_emissions_prior: Optional[bool] = None, #! depracated
        use_initial_prior: Optional[bool] = None,   #! depracated
    ):
        # store parameters and flags
        self.state_dim = state_dim
        self.emission_dim = emission_dim

        self.use_dynamics_bias = use_dynamics_bias
        self.use_emissions_bias = use_emissions_bias
        self.fix_Q = fix_Q
        self.gp_priors = self._resolve_gp_priors(
            gp_priors, use_dynamics_prior, use_emissions_prior, use_initial_prior
        )

        self.initial_params = initial_params
        self.params = None

        # initialize priors
        self.priors = self.initialize_priors(basis)

    def _equations(self) -> str:
        def coef(c, t="t"):
            return f"{c}(u_{t})" if c in self.gp_priors else c

        dynamics = f"{coef('A')} x_t" + (f" + {coef('b')}" if self.use_dynamics_bias else "")
        emissions = f"{coef('C')} x_t" + (f" + {coef('d')}" if self.use_emissions_bias else "")
        Q = "I" if self.fix_Q else "Q"
        return (
            f"CLDS2(state_dim={self.state_dim}, emission_dim={self.emission_dim})\n"
            f"  x_0     ~ N({coef('m', 0)}, S0)\n"
            f"  x_(t+1) = {dynamics} + N(0, {Q})\n"
            f"  y_t     = {emissions} + N(0, R)"
        )


    def _resolve_gp_priors(
        self, gp_priors, use_dynamics_prior, use_emissions_prior, use_initial_prior
    ) -> frozenset:
        """Validate `gp_priors`, or translate the deprecated use_*_prior flags into it."""
        legacy = (use_dynamics_prior, use_emissions_prior, use_initial_prior)
        if any(flag is not None for flag in legacy):
            if gp_priors is not None:
                raise ValueError(
                    "Pass either gp_priors or the deprecated use_*_prior flags, not both."
                )
            warnings.warn(
                "use_dynamics_prior, use_emissions_prior and use_initial_prior are "
                "deprecated, pass gp_priors instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            # unspecified flags keep their former default (True)
            flags = dict(zip("ACm", (flag is None or flag for flag in legacy)))
            flags["b"], flags["d"] = flags["A"], flags["C"]
            gp_priors = [c for c, flag in flags.items() if flag and self._has(c)]
        elif gp_priors is None:
            warnings.warn(
                "Neither `gp_priors` provided nor deprecated `use_*_prior` flags: "
                "setting all requested coefficients to have GP prior."
            )
            gp_priors = [c for c in self.COEFFICIENTS if self._has(c)]

        gp_priors = frozenset(gp_priors)
        unknown = gp_priors - self.COEFFICIENTS.keys()
        if unknown:
            raise ValueError(
                f"Unknown coefficients {sorted(unknown)} in gp_priors, "
                f"expected a subset of {list(self.COEFFICIENTS)}."
            )
        disabled = sorted(c for c in gp_priors if not self._has(c))
        if disabled:
            raise ValueError(
                f"gp_priors contains the biases {disabled}, which are disabled; "
                "set use_dynamics_bias / use_emissions_bias."
            )
        return gp_priors

    def _has(self, c) -> bool:
        """
        Whether coefficient c is part of the model: 
        A, C and m always are, the biases b and d are optional.
        """
        if c == "b":
            return self.use_dynamics_bias
        if c == "d":
            return self.use_emissions_bias
        return c in self.COEFFICIENTS

    def _coef_shape(self, c) -> tuple:
        """Shape of coefficient c at a single time step."""
        D, N = self.state_dim, self.emission_dim
        return {"A": (D, D), "b": (D,), "C": (N, D), "d": (N,), "m": (D,)}[c]

    def _n_basis(self, c) -> int:
        """Number of basis functions of coefficient c (a static one has the basis [1])."""
        return self.priors[c].n_basis_funcs if c in self.gp_priors else 1

    def _basis(self, c, u):
        """Basis Phi_c(u) of coefficient c at conditions u, [1] for a static coefficient."""
        if c in self.gp_priors:
            return self.priors[c].evaluate_basis(u)
        return jnp.ones((u.shape[0], 1))

    def initialize_priors(self, basis: list):
        """Initialize a weight-space GP prior for each coefficient in `gp_priors`"""
        priors = dict.fromkeys(self.COEFFICIENTS)
        for c in self.gp_priors:
            output_dim, *input_dim = self._coef_shape(c)
            priors[c] = WeightSpaceGaussianProcess(
                basis=basis,
                input_dim=input_dim[0] if input_dim else 1,
                output_dim=output_dim,
            )
        return priors

    def initialize_params(self, key: jxr.PRNGKey, noise_scale: float = 1.0):
        """
        Initialize the model parameters based on the priors and flags.
        GP weights are sampled from the priors, static coefficients are randomly
        initialized, and disabled biases are set to zero.
        """
        keys = dict(zip(self.COEFFICIENTS, jxr.split(key, len(self.COEFFICIENTS))))

        def init(c):
            if c in self.gp_priors:
                return self.priors[c].sample_weights(keys[c])
            if self._has(c):
                return jxr.normal(keys[c], self._coef_shape(c))
            return jnp.zeros(self._coef_shape(c))

        # pack together
        return ParamsCLDS2(
            **{field: init(c) for c, field in self.COEFFICIENTS.items()},
            initial_cov=noise_scale**2 * jnp.eye(self.state_dim),
            dynamics_cov=(1.0 if self.fix_Q else noise_scale**2) * jnp.eye(self.state_dim),
            emissions_cov=noise_scale**2 * jnp.eye(self.emission_dim),
        )

    @staticmethod
    def run_dynamics(
        key: jxr.PRNGKey,
        As: Float[Array, "num_timesteps state_dim state_dim"],
        bs: Float[Array, "num_timesteps state_dim"],
        Q: Float[Array, "state_dim state_dim"],
        Cs: Float[Array, "num_timesteps emission_dim state_dim"],
        ds: Float[Array, "num_timesteps emission_dim"],
        R: Float[Array, "emission_dim emission_dim"],
        m0: Float[Array, "state_dim"],
        S0: Float[Array, "state_dim state_dim"],
    ):
        """
        Run CLDS dynamics to generate states and emissions, following the system:
            x_0 ~ N(m0, S0)
            x_t = A_t x_{t-1} + b_t + N(0, Q)
            y_t = C_t x_t + d_t + N(0, R)

        Parameters
        ----------
        key : jxr.PRNGKey
            random key for sampling
        As : Float[Array, "num_timesteps state_dim state_dim"]
            dynamics matrices
        bs : Float[Array, "num_timesteps state_dim"]
            dynamics biases
        Q : Float[Array, "state_dim state_dim"]
            dynamics covariance
        Cs : Float[Array, "num_timesteps emission_dim state_dim"]
            emissions matrices
        ds : Float[Array, "num_timesteps emission_dim"]
            emissions biases
        R : Float[Array, "emission_dim emission_dim"]
            emissions covariance
        m0 : Float[Array, "state_dim"]
            initial state mean
        S0 : Float[Array, "state_dim state_dim"]
            initial state covariance

        Returns
        -------
        xs : Float[Array, "num_timesteps state_dim"]
            latent state dynamics
        ys : Float[Array, "num_timesteps emission_dim"]
            emissions
        """

        def f(x, args):
            A, b, C, d, (dy_key, em_key) = args

            dynamics_noise = jxr.multivariate_normal(dy_key, jnp.zeros(Q.shape[0]), Q)
            x_next = A @ x + b + dynamics_noise

            emissions_noise = jxr.multivariate_normal(em_key, jnp.zeros(R.shape[0]), R)
            y = C @ x + d + emissions_noise

            return x_next, (x_next, y)

        x_init = jxr.multivariate_normal(key, m0, S0)
        subkeys = jxr.split(key, num=(As.shape[0], 2))
        _, (x_nexts, ys) = jax.lax.scan(f, x_init, xs=(As, bs, Cs, ds, subkeys))
        xs = jnp.concatenate((x_init[None, :], x_nexts[:-1]), axis=0)
        return xs, ys

    def sample(
        self,
        inputs: Float[Array, "num_timesteps input_dim"],
        key: jxr.PRNGKey = jxr.PRNGKey(0),
    ):
        """
        Sample the states and emissions given the inputs as conditions and the fitted model parameters.

        Parameters
        ----------
        inputs : Float[Array, "num_timesteps input_dim"]
            input conditions for the GP priors
        seed : int, optional
            random seed for sampling, by default 2

        Returns
        -------
        xs : Float[Array, "num_timesteps state_dim"]
            predicted latent states
        ys : Float[Array, "num_timesteps emission_dim"]
            predicted emissions
        """
        if self.params is None:
            raise ValueError("Model parameters have not been fit. Call fit() first.")
        As, Cs, bs, ds, m0 = self.weights_to_params(self.params, inputs)
        return CLDS2.run_dynamics(
            key,
            As,
            bs,
            self.params.dynamics_cov,
            Cs,
            ds,
            self.params.emissions_cov,
            m0,
            self.params.initial_cov,
        )

    def log_prior(self, params: ParamsCLDS2):
        """Compute the log prior of the parameters (static coefficients have a flat prior)."""
        return sum(
            (
                self.priors[c].log_prob_weights(getattr(params, field))
                for c, field in self.COEFFICIENTS.items()
                if c in self.gp_priors
            ),
            start=0.0,
        )

    def weights_to_params(self, params, inputs):
        """Evaluate the coefficients at the conditions `inputs`: GP coefficients through
        their weight-space prior, static ones broadcast over time, disabled biases as zeros."""

        def evaluate(c, u):
            shape = (u.shape[0],) + self._coef_shape(c)
            w = getattr(params, self.COEFFICIENTS[c])
            if not self._has(c):
                return jnp.zeros(shape)
            if c in self.gp_priors:
                return self.priors[c](w, u).reshape(shape) # replaces squeeze() for {b, d, m}
            # Handle remaining static by broadcasting over time
            return jnp.broadcast_to(jnp.reshape(w, (-1,) + shape[1:]), shape)

        As, bs = evaluate("A", inputs), evaluate("b", inputs)
        Cs, ds = evaluate("C", inputs), evaluate("d", inputs)
        m0 = evaluate("m", inputs[:1])[0]
        return As, Cs, bs, ds, m0

    def smoother(self, params: ParamsCLDS, emissions, inputs, mask=None):
        """inputs as conditions"""
        # Format params
        As, Cs, bs, ds, m0 = self.weights_to_params(params, inputs)
        # Q = params.dynamics_cov
        # R = params.emissions_cov

        # force dynamics to retain the last valid state if mask is provided
        if mask is not None:
            As = jnp.where(mask[:, None, None], As, jnp.eye(As.shape[1]))
            bs = jnp.where(mask[:, None], bs, 0.0)
            Cs = jnp.where(mask[:, None, None], Cs, jnp.zeros_like(Cs))
            ds = jnp.where(mask[:, None], ds, 0.0)
            emissions = jnp.where(mask[:, None], emissions, 0.0)
            # Q = jnp.where(
            #     mask[:, None, None], jnp.tile(Q[None], (len(mask), 1, 1)), 0.0
            # )
            # R = jnp.where(
            #     mask[:, None, None], jnp.tile(R[None], (len(mask), 1, 1)), 0.0
            # )

        # Run the smoother
        lgssm_params = {
            "m0": m0,
            "S0": params.initial_cov,
            "As": As,
            "bs": bs,
            "Q": params.dynamics_cov,
            "Cs": Cs,
            "R": params.emissions_cov,
            "ds": ds,
        }

        ll, filter_results, smoother_results = utils.lgssm_smoother(
            **lgssm_params, ys=emissions
        )

        if mask is not None:
            #? Mask also filter/smoother results?
            n_masked = jnp.sum(~mask.astype(bool))
            zero = jnp.zeros(
                (emissions.shape[-1],), dtype=emissions.dtype
            )
            # Use the exact same log-density helper as the filter.
            ll -= n_masked * utils.logprob_analytic(
                zero, zero, params.emissions_cov
            )

        return ll, filter_results, smoother_results

    def e_step(
        self,
        params: ParamsCLDS,
        emissions: Float[Array, "num_timesteps emission_dim"],
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
        mask: Optional[Float[Array, "num_timesteps"]] = None,
    ):
        """
        Compute the expected sufficient statistics (ZTZ, ZTY, YTY, N) of the "initial",
        "dynamics" and "emissions" regression blocks under the smoothing posterior.
        Each block regresses its target on the features z_t = [Phi_x(u_t) kron x_t ; Phi_1(u_t)],
        where Phi_x and Phi_1 are the bases of the coefficient on x and of the bias
        (the constant [1] for a static coefficient), evaluated at the conditions u_t = inputs[t].
        """
        def features(c, u, valid=None):
            """Basis of coefficient c at conditions u, zeroed at invalid steps (None if disabled)."""
            if not self._has(c):
                return None
            Phi = self._basis(c, u)
            return Phi if valid is None else jnp.where(valid[:, None], Phi, 0.0)

        if mask is not None:
            num_timesteps = mask.sum().astype(int)
        else:
            num_timesteps = emissions.shape[0]

        if inputs is None:
            inputs = jnp.zeros((emissions.shape[0], 1))

        # Run the smoother to get posterior expectations
        marginal_loglik, _, smoother_results = self.smoother(
            params, emissions, inputs, mask
        )
        Ex, Vx, Expxn = smoother_results
        y = emissions

        # zero out padded steps so they do not contribute to any stats;
        # transition t -> t+1 is valid iff mask[t+1]
        if mask is not None:
            Ex = jnp.where(mask[:, None], Ex, 0.0)
            Vx = jnp.where(mask[:, None, None], Vx, 0.0)
            Expxn = jnp.where(mask[1:, None, None], Expxn, 0.0)
            y = jnp.where(mask[:, None], y, 0.0)
        ExxT = Vx + jnp.einsum("ti,tj->tij", Ex, Ex)

        ## initial state: x[0] = m(u[0]) + noise
        # slice with [:1] to keep the leading time axis (a single step) expected by regression_stats
        ZTZ, ZTY = utils.regression_stats_from_moments(
            Phi_x=None,
            Phi_1=features("m", inputs[:1]),
            Ey=Ex[:1],
        )
        init_stats = (ZTZ, ZTY, ExxT[0], 1)

        ## dynamics: x[t+1] = A(u[t]) x[t] + b(u[t]) + noise, for t = 0...T-2
        up = inputs[:-1]
        valid = None if mask is None else mask[1:]
        ZTZ, ZTY = utils.regression_stats_from_moments(
            Phi_x=features("A", up, valid),
            Phi_1=features("b", up, valid),
            ExxT=ExxT[:-1], # ExpxpT
            Ex=Ex[:-1],     # Exp
            ExyT=Expxn,
            Ey=Ex[1:],      # Exn
        )
        dynamics_stats = (ZTZ, ZTY, ExxT[1:].sum(0), num_timesteps - 1)

        ## emissions: y[t] = C(u[t]) x[t] + d(u[t]) + noise
        ZTZ, ZTY = utils.regression_stats_from_moments(
            Phi_x=features("C", inputs, mask),
            Phi_1=features("d", inputs, mask),
            ExxT=ExxT,
            Ex=Ex,
            ExyT=jnp.einsum("ti,tj->tij", Ex, y),
            Ey=y,
        )
        emissions_stats = (ZTZ, ZTY, y.T @ y, num_timesteps)

        return {
            "initial": init_stats,
            "dynamics": dynamics_stats,
            "emissions": emissions_stats,
        }, marginal_loglik

    def m_step(
        self,
        params: ParamsCLDS2,
        batch_stats: Tuple,  # inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None,
    ) -> ParamsCLDS2:
        def fit_block(block, stats, Sigma, update_Sigma=True):
            '''Fit the regression of one block and unpack W into its (weights, bias, Sigma)'''
            lin_coef, bias_coef = self.BLOCKS[block]
            n_lin  = self._n_basis(lin_coef) * self.state_dim if lin_coef else 0
            n_bias = self._n_basis(bias_coef) if self._has(bias_coef) else 0

            # For possible mixed cases of GP priors on [linear, bias] blocks
            has_prior = [lin_coef in self.gp_priors] * n_lin + \
                        [bias_coef in self.gp_priors] * n_bias

            W = utils.solve_map_regression(*stats[:2], has_prior, Sigma)
            if update_Sigma:
                Sigma = utils.residual_covariance(*stats, W)

            # (n_basis * input_dim, output_dim) -> (n_basis, output_dim, input_dim)
            output_dim = W.shape[1]

            # Static coefficients drop the basis (0-th) axis
            ## Linear component
            weights = W[:n_lin].reshape(-1, self.state_dim, output_dim).transpose(0, 2, 1)
            if lin_coef is not None and lin_coef not in self.gp_priors:
                weights = weights[0]

            ## Bias component
            bias = W[n_lin:].reshape(-1, output_dim, 1)
            if not self._has(bias_coef): # disabled biases are zero
                bias = jnp.zeros(output_dim)
            elif bias_coef not in self.gp_priors:
                bias = bias[0, :, 0]
            
            return weights, bias, utils.symmetrize(Sigma)

        # Sum the statistics across all batches
        stats = jax.tree_util.tree_map(partial(jnp.sum, axis=0), batch_stats)

        _, m, S = fit_block(
            block="initial", 
            stats=stats["initial"], 
            Sigma=params.initial_cov
            )

        # with fix_Q, Q stays the identity (for identifiability)
        As, bs, Q = fit_block(
            block="dynamics",
            stats=stats["dynamics"],
            Sigma=jnp.eye(self.state_dim) if self.fix_Q else params.dynamics_cov,
            update_Sigma=not self.fix_Q,
        )

        Cs, ds, R = fit_block(
            block="emissions", 
            stats=stats["emissions"], 
            Sigma=params.emissions_cov
            )

        params = ParamsCLDS2(
            initial_mean=m,
            initial_cov=S,
            dynamics_weights=As,
            dynamics_bias=bs,
            dynamics_cov=Q,
            emissions_weights=Cs,
            emissions_bias=ds,
            emissions_cov=R,
        )
        return params

    def log_prob(self, params, emissions, conditions):
        """Compute the log probability of the emissions given the parameters"""

        def _mll(_emissions, _conditions):
            marginal_loglik, _, _ = self.smoother(params, _emissions, _conditions)
            return marginal_loglik

        # the prior on the parameters is counted once, not per batch
        return vmap(_mll)(emissions, conditions).sum() + self.log_prior(params)

    def marginal_log_lik(self, params, emissions, conditions):
        """Compute the marginal log likelihood of the emissions given the parameters"""

        def _mll(_emissions, _conditions):
            (marginal_loglik, _, _) = self.smoother(params, _emissions, _conditions)
            return marginal_loglik

        return vmap(_mll)(emissions, conditions).sum()

    def fit(
        self,
        emissions: Float[Array, "num_batches num_timesteps emission_dim"],
        conditions: Float[Array, "num_batches num_timesteps input_dim"],
        initial_params: ParamsCLDS2 = None,
        mask: Float[Array, "num_batches num_timesteps"] = None,
        num_iters: int = 50,
        seed: int = 2,
    ):
        """
        Fit the CLDS model to the emissions and conditions using the EM algorithm.

        Parameters
        ----------
        emissions : Float[Array, "num_batches num_timesteps emission_dim"]
            observed emissions data.
        conditions : Float[Array, "num_batches num_timesteps input_dim"]
            input conditions for the GP priors.
        initial_params : ParamsCLDS2, optional
            initial parameters for the model, by default None.
        mask : Float[Array, "num_batches num_timesteps"], optional
            mask for missing data, by default None.
        num_iters : int, optional
            number of EM iterations, by default 50.
        seed : int, optional
            random seed for initialization, by default 2.

        Returns
        -------
        params : ParamsCLDS2
            fitted model parameters.
        log_probs : list
            log probabilities at each iteration.
        """
        if emissions.ndim != 3:
            raise ValueError(
                "emissions should be 3D, of shape (num_batches, num_timesteps, emission_dim)"
            )

        # apply mask to front load valid data
        if mask is not None:
            shift = jnp.argmax(mask, axis=1)
            emissions = jax.vmap(partial(jnp.roll, axis=0))(emissions, -shift)
            conditions = jax.vmap(partial(jnp.roll, axis=0))(conditions, -shift)
            mask = jax.vmap(partial(jnp.roll, axis=0))(mask, -shift)

        if (self.initial_params is None) and (initial_params is None):
            # use default initialization if no initial params provided
            initial_params = self.initialize_params(jxr.PRNGKey(seed))
        elif (self.initial_params is not None) and (initial_params is None):
            # use previously stored initial params if no initial params provided
            initial_params = self.initial_params

        if self.fix_Q:
            initial_params = initial_params._replace(
                dynamics_cov=jnp.eye(self.state_dim)
            )

        # overwrite initial params
        self.initial_params = initial_params

        @jit
        def em_step(params, emissions, conditions, mask):
            # Obtain current E-step stats and model log prob
            batch_stats, lls = vmap(partial(self.e_step, params))(
                emissions, conditions, mask
            )
            mll = lls.sum()
            log_prior = self.log_prior(params)
            lp = mll + log_prior

            # Update with M-step
            next_params = self.m_step(params, batch_stats)

            return next_params, (lp, mll)

        log_probs, marginal_log_liks = [], []

        pbar = trange(num_iters)
        pbar.set_description("jit compiling ...")

        params = initial_params
        for i in pbar:
            next_params, (log_prob, marginal_log_lik) = em_step(
                params, emissions, conditions, mask
            )
            log_probs.append(log_prob)
            marginal_log_liks.append(marginal_log_lik)

            if i > 2 and marginal_log_lik < marginal_log_liks[-2]:
                pbar.set_description(
                    f"EM stopped at iteration {i+1} due to decreasing marginal_log_lik"
                )
                break

            if jnp.isnan(log_prob):
                pbar.set_description(f"EM stopped at iteration {i+1} due to NaN values")
                break

            params = next_params
            pbar.set_description(
                f"Iter {i+1}/{num_iters}, log-prob = {log_prob:.2f}, marginal log-lik = {marginal_log_lik:.2f}"
            )

        self.params = params
        return params, (marginal_log_liks, log_probs)

# %%
