# -*- coding: utf-8 -*-
"""
@author: Amin, Victor
"""

# %%
import jax
import jax.numpy as jnp
import jax.random as jxr

from jaxtyping import Array, Float
from typing import Optional, Tuple, Dict

from .utils import logprob_analytic
from functools import partial
from jax import jit, lax, vmap
from tqdm.auto import trange

from dynamax.linear_gaussian_ssm.inference import make_lgssm_params, lgssm_smoother

import logging

logging.basicConfig(
    level=logging.INFO, format="[%(filename)s][%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
from scipy.linalg import solve_sylvester

from .params import (
    ParamsEmission,
    ParamsNormalLikelihood,
    ParamsGPLDS,
    ParamsCLDS,
    ParamsCLDS2,
)

import clds.utils as utils

import numpyro.distributions as dist


# %%
class WeightSpaceGaussianProcess:
    """
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
        """
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

    # def log_prob(
    #     self,
    #     conditions: Float[Array, "n_steps n_conditions"],
    #     fs: Float[Array, "n_steps input_dim output_dim"],
    # ) -> Float[Array, "input_dim output_dim"]:
    #     """
    #     Compute the log probability of the GP draws at the points `conditions`
    #     """
    #     # Check dimensions
    #     if fs.ndim == 2:
    #         assert (self.output_dim == 1) ^ (self.input_dim == 1), "Incorrect dimensions"
    #         fs = fs.reshape(-1, self.output_dim, self.input_dim)
    #     assert fs.shape[1] == self.output_dim and fs.shape[2] == self.input_dim, "Incorrect dimensions"

    #     # Compute log prob
    #     T = len(fs)
    #     Phi = self.evaluate_basis(conditions)  # T x L
    #     cov = jnp.dot(Phi, Phi.T)  # T x T
    #     # return jax.vmap(lambda _f: logprob_analytic(_f, jnp.zeros(T), cov), in_axes=(1))(fs.reshape(T, -1)).reshape(self.output_dim, self.input_dim)

    #     model_dist = dist.MultivariateNormal(jnp.zeros(T), covariance_matrix=cov)
    #     return model_dist.log_prob(fs.reshape(T, -1).T).reshape(self.output_dim, self.input_dim)

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
    """
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
        """
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
        return As, Cs, bs.squeeze(), m0.squeeze()

    def smoother(self, params: ParamsCLDS, emissions, inputs):
        """inputs as conditions"""
        # Format params
        As, Cs, bs, m0 = self.weights_to_params(params, inputs)
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
            Q=Q,
            R=R,
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
    Conditionally Linear Dynamical System (CLDS) model, with LDS dynamics and
    weight-space view parametrization of the GP priors for the parameters {A, b, C, d, m0}.

    If weight-space GP priors are not specified for any parameter, it is learned as a static parameter.

    Parameters
    ----------
    state_dim : int
        dimension of the latent state space.
    emission_dim : int
        dimension of the observation space.
    basis : list
        list of basis functions for the weight-space GP priors.
    use_dynamics_prior : bool, optional
        whether to use a GP prior for the dynamics matrix A, by default True.
    use_emissions_prior : bool, optional
        whether to use a GP prior for the emissions matrix C, by default True.
    use_initial_prior : bool, optional
        whether to use a GP prior for the initial state mean m0, by default True.
    use_dynamics_bias : bool, optional
        whether to include a bias term in the dynamics, by default True.
    use_emissions_bias : bool, optional
        whether to include a bias term in the emissions, by default False.
    initial_params : Optional[ParamsCLDS2], optional
        initial parameters for the model, by default None.
    """

    def __init__(
        self,
        state_dim: int,
        emission_dim: int,
        basis: list,  # soon to be class
        use_dynamics_prior: bool = True,
        use_emissions_prior: bool = True,
        use_initial_prior: bool = True,
        use_dynamics_bias: bool = True,
        use_emissions_bias: bool = False,
        initial_params: Optional[ParamsCLDS2] = None,
    ):
        # store parameters and flags
        self.state_dim = state_dim
        self.emission_dim = emission_dim

        self.use_dynamics_prior = use_dynamics_prior
        self.use_dynamics_bias = use_dynamics_bias
        self.use_emissions_prior = use_emissions_prior
        self.use_emissions_bias = use_emissions_bias
        self.use_initial_prior = use_initial_prior

        self.initial_params = initial_params
        self.params = None

        # initialize priors
        self.priors = self.initialize_priors(basis)

    def initialize_priors(self, basis: list):
        """Initialize the priors based on the flags"""
        priors = {
            "dynamics": None,
            "emissions": None,
            "init": None,
        }
        if self.use_dynamics_prior:
            priors["dynamics"] = WeightSpaceGaussianProcess(
                basis=basis,
                input_dim=self.state_dim,
                output_dim=self.state_dim,
                include_bias=self.use_dynamics_bias,
            )
        if self.use_emissions_prior:
            priors["emissions"] = WeightSpaceGaussianProcess(
                basis=basis,
                input_dim=self.state_dim,
                output_dim=self.emission_dim,
                include_bias=self.use_emissions_bias,
            )
        if self.use_initial_prior:
            priors["init"] = WeightSpaceGaussianProcess(
                basis=basis,
                input_dim=1,
                output_dim=self.state_dim,
                include_bias=False,
            )
        return priors

    def initialize_params(self, key: jxr.PRNGKey, noise_scale: float = 0.2):
        """
        Initialize the model parameters based on the priors and flags.
        If priors are used, GP weights are sampled from the priors.
        If priors are not used, static parameters are randomly initialized.
        """
        Ab_key, Cd_key, m0_key = jxr.split(key, 3)

        def get_params(key, use_prior, prior, input_dim, output_dim, use_bias):
            if use_prior:
                weights = prior.sample_weights(key)
                if use_bias:
                    bias = weights[:, :, -1:]
                    weights = weights[:, :, :-1]
                else:
                    bias = jnp.zeros(output_dim)
            else:
                weights = jxr.normal(key, (output_dim, input_dim))
                bias = (
                    jxr.normal(key, (output_dim)) if use_bias else jnp.zeros(output_dim)
                )
            return weights, bias

        # initialize dynamics
        dynamics_weights, dynamics_bias = get_params(
            Ab_key,
            self.use_dynamics_prior,
            self.priors["dynamics"],
            self.state_dim,
            self.state_dim,
            self.use_dynamics_bias,
        )

        # initialize emissions
        emissions_weights, emissions_bias = get_params(
            Cd_key,
            self.use_emissions_prior,
            self.priors["emissions"],
            self.state_dim,
            self.emission_dim,
            self.use_emissions_bias,
        )

        # initialize initial state
        if self.use_initial_prior:
            initial_mean = self.priors["init"].sample_weights(m0_key)
        else:
            initial_mean = jxr.normal(m0_key, self.state_dim)

        # pack together
        return ParamsCLDS2(
            dynamics_weights=dynamics_weights,
            dynamics_bias=dynamics_bias,
            emissions_weights=emissions_weights,
            emissions_bias=emissions_bias,
            initial_mean=initial_mean,
            initial_cov=noise_scale**2 * jnp.eye(self.state_dim),
            dynamics_cov=noise_scale**2 * jnp.eye(self.state_dim),
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

    def log_prior(self, params: ParamsCLDS2, inputs):
        """Compute the log prior of the parameters."""

        def get_log_prior(use_prior, prior, weights, bias, use_bias):
            if use_prior:
                return prior.log_prob_weights(weights) + (
                    prior.log_prob_weights(bias) if use_bias else 0.0
                )
            else:
                return 0.0

        logprior_Ab = get_log_prior(
            self.use_dynamics_prior,
            self.priors["dynamics"],
            params.dynamics_weights,
            params.dynamics_bias,
            self.use_dynamics_bias,
        )
        logprior_Cd = get_log_prior(
            self.use_emissions_prior,
            self.priors["emissions"],
            params.emissions_weights,
            params.emissions_bias,
            self.use_emissions_bias,
        )

        if self.use_initial_prior:
            logprior_m0 = self.priors["init"].log_prob_weights(params.initial_mean)
        else:
            logprior_m0 = 0.0

        return logprior_Ab + logprior_Cd + logprior_m0

    def weights_to_params(self, params, inputs):
        """Transform weights of weight space into parameters.
        Implement as needed for all weight-space GP priors."""

        def get_params(use_prior, prior, weights, bias, inputs, use_bias):
            if use_prior:
                A = prior(weights, inputs)
                b = (
                    prior(bias, inputs)
                    if use_bias
                    else jnp.zeros((inputs.shape[0], A.shape[1]))
                )
            else:
                A = weights
                if A.ndim == 2:
                    A = jnp.tile(A[None], (len(inputs), 1, 1))
                b = bias if use_bias else jnp.zeros((inputs.shape[0], A.shape[1]))
                if b.shape[0] != len(inputs):
                    b = jnp.tile(b, (len(inputs), 1, 1))
            return A, b

        As, bs = get_params(
            self.use_dynamics_prior,
            self.priors["dynamics"],
            params.dynamics_weights,
            params.dynamics_bias,
            inputs,
            self.use_dynamics_bias,
        )

        Cs, ds = get_params(
            self.use_emissions_prior,
            self.priors["emissions"],
            params.emissions_weights,
            params.emissions_bias,
            inputs,
            self.use_emissions_bias,
        )

        m0 = (
            self.priors["init"](params.initial_mean, inputs)[0]
            if self.use_initial_prior
            else params.initial_mean
        )  #! Some unnecessary computation, keeping only t=0
        return As, Cs, bs.squeeze(), ds.squeeze(), m0.squeeze()

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
        lgssm_params = make_lgssm_params(
            initial_mean=m0,
            initial_cov=params.initial_cov,
            dynamics_weights=As,
            dynamics_cov=params.dynamics_cov,
            dynamics_bias=bs,
            emissions_weights=Cs,
            emissions_cov=params.emissions_cov,
            emissions_bias=ds,
        )
        smooth_params = lgssm_smoother(lgssm_params, emissions=emissions)

        # mask results
        if mask is not None:
            filter_results = (
                jnp.where(mask[:, None], smooth_params.filtered_means, 0.0),
                jnp.where(mask[:, None, None], smooth_params.filtered_covariances, 0.0),
            )
            smoother_results = (
                jnp.where(mask[:, None], smooth_params.smoothed_means, 0.0),
                jnp.where(mask[:, None, None], smooth_params.smoothed_covariances, 0.0),
                jnp.where(
                    mask[1:, None, None],
                    smooth_params.smoothed_cross_covariances,
                    0.0,
                ),
            )
        else:
            filter_results = (
                smooth_params.filtered_means,
                smooth_params.filtered_covariances,
            )
            smoother_results = (
                smooth_params.smoothed_means,
                smooth_params.smoothed_covariances,
                smooth_params.smoothed_cross_covariances,
            )
        return smooth_params.marginal_loglik, filter_results, smoother_results

        # lgssm_params = {
        #     "m0": m0,
        #     "S0": params.initial_cov,
        #     "As": As,
        #     "bs": bs,
        #     "Q": params.dynamics_cov,
        #     "Cs": Cs,
        #     "R": params.emissions_cov,
        # }
        # return utils.lgssm_smoother(**lgssm_params, ys=emissions)

    def e_step(
        self,
        params: ParamsCLDS,
        emissions: Float[Array, "num_timesteps emission_dim"],
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
        mask: Optional[Float[Array, "num_timesteps"]] = None,
    ):
        def weightspace_stats(
            Phi: Float[Array, "n_steps n_basis_funcs"],
            XTX: Float[Array, "n_steps input_dim input_dim"] = None,
            XTY: Float[Array, "n_steps input_dim output_dim"] = None,
        ) -> tuple:
            """
            Compute the expected sufficient statistics for the weight-space GP prior.
            Provide the sufficient stats X^T X and X^T Y for the problem Y = A(C)X + noise.
            This returns the expanded stats Phi @ X^T X @ Phi^T and Phi @ X^T Y for the basis functions Phi(C).
            """
            n_basis_funcs = Phi.shape[-1]

            if XTX is not None:
                input_dim = XTX.shape[-1]
                ZTZ = jnp.einsum("tk,tl,tij->kilj", Phi, Phi, XTX).reshape(
                    -1,
                    n_basis_funcs * input_dim,
                )
            else:
                ZTZ = None

            if XTY is not None:
                output_dim = XTY.shape[-1]
                ZTY = jnp.einsum("tk,tim->kim", Phi, XTY).reshape(-1, output_dim)
            else:
                ZTY = None

            return ZTZ, ZTY

        """take inputs to be theta"""
        if mask is not None:
            num_timesteps = mask.sum().astype(int)
        else:
            num_timesteps = emissions.shape[0]

        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 1))

        # Run the smoother to get posterior expectations
        marginal_loglik, filter_results, smoother_results = self.smoother(
            params, emissions, inputs, mask
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
        b = jnp.ones((inputs.shape[0], 1))
        y = emissions
        up = inputs[:-1]

        # mask shorthand if needed
        if mask is not None:
            Exp = jnp.where(mask[1:, None], Exp, 0.0)
            Vxp = jnp.where(mask[1:, None, None], Vxp, 0.0)
            b = jnp.where(mask[:, None], b, 0.0)
            y = jnp.where(mask[:, None], y, 0.0)

        ## expected sufficient statistics for the initial distribution
        Ex0 = smoothed_means[0]
        Ex0x0T = smoothed_covariances[0] + jnp.outer(Ex0, Ex0)
        init_stats = (Ex0, Ex0x0T, 1)

        # full E-step sufficient stats for initial wGP
        if self.use_initial_prior:
            m0_targets = Ex0.reshape(1, 1, self.state_dim)
            XTX_m0 = jnp.ones((1, 1, 1))

            _cond = (
                up[0].reshape(1) if up[0].ndim == 0 else up[0].reshape(1, up.shape[1])
            )
            init_gp_stats = weightspace_stats(
                self.priors["init"].evaluate_basis(_cond),
                XTX_m0,
                m0_targets,
            )
        else:
            init_gp_stats = None

        ## expected sufficient statistics for the dynamics
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        sum_xpxpT = Vxp.sum(0) + Exp.T @ Exp
        sum_xpxnT = Expxn.sum(0)
        sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
        if self.use_dynamics_bias:
            bp = b[1:]
            sum_xpT = Exp.T @ bp
            sum_xpxpT = jnp.block([[sum_xpxpT, sum_xpT], [sum_xpT.T, bp.T @ bp]])
            sum_xpxnT = jnp.block([[Expxn.sum(0)], [bp.T @ Exn]])
        dynamics_stats = (sum_xpxpT, sum_xpxnT, sum_xnxnT, num_timesteps - 1)

        # full E-step sufficient stats for dynamics wGP
        if self.use_dynamics_prior:
            _Phi = self.priors["dynamics"].evaluate_basis(up)
            if mask is not None:
                _Phi = jnp.where(mask[1:, None], _Phi, 0.0)
            ExpxpT = jnp.einsum("ti,tj->tij", Exp, Exp) + Vxp
            sum_zpzpT, sum_zpxnT = weightspace_stats(_Phi, XTX=ExpxpT, XTY=Expxn)
            if self.use_dynamics_bias:
                sum_zpT, sum_znT = weightspace_stats(
                    _Phi, XTX=Exp[:, None, :], XTY=Exn[:, None, :]
                )
                sum_zpzpT = jnp.block(
                    [
                        [sum_zpzpT, sum_zpT.T],
                        [sum_zpT, jnp.einsum("tk,tl->kl", _Phi, _Phi)],
                    ]
                )
                sum_zpxnT = jnp.concatenate([sum_zpxnT, sum_znT], axis=0)
            dynamics_gp_stats = (
                sum_zpzpT,
                params.dynamics_cov,
                sum_zpxnT,
                sum_xnxnT,
                num_timesteps - 1,
            )
        else:
            dynamics_gp_stats = None

        ## expected sufficient statistics for the emissions
        sum_xxT = Vx.sum(0) + Ex.T @ Ex
        sum_xyT = Ex.T @ y
        sum_yyT = y.T @ y
        if self.use_emissions_bias:
            sum_xT = Ex.T @ b
            sum_xxT = jnp.block([[sum_xxT, sum_xT], [sum_xT.T, b.T @ b]])
            sum_xyT = jnp.block([[sum_xyT], [b.T @ y]])
        emission_stats = (sum_xxT, sum_xyT, sum_yyT, num_timesteps)

        # full E-step sufficient stats for emissions wGP
        if self.use_emissions_prior:
            _Phi = self.priors["emissions"].evaluate_basis(inputs)
            if mask is not None:
                _Phi = jnp.where(mask[:, None], _Phi, 0.0)
            ExxT = jnp.einsum("ti,tj->tij", Ex, Ex) + Vx
            ExyT = jnp.einsum("ti,tj->tij", Ex, y)
            sum_zzT, sum_zyT = weightspace_stats(_Phi, ExxT, ExyT)

            if self.use_emissions_bias:
                sum_zT, sum_yT = weightspace_stats(
                    _Phi, XTX=Ex[:, None, :], XTY=y[:, None, :]
                )
                sum_zzT = jnp.block(
                    [
                        [sum_zzT, sum_zT.T],
                        [sum_zT, jnp.einsum("tk,tl->kl", _Phi, _Phi)],
                    ]
                )
                sum_zyT = jnp.concatenate([sum_zyT, sum_yT], axis=0)

            emissions_gp_stats = (
                sum_zzT,
                params.emissions_cov,
                sum_zyT,
                sum_yyT,
                num_timesteps,
            )
        else:
            emissions_gp_stats = None

        return (
            init_stats,
            init_gp_stats,
            dynamics_stats,
            dynamics_gp_stats,
            emission_stats,
            emissions_gp_stats,
        ), marginal_loglik

    def m_step(
        self,
        params: ParamsCLDS,
        batch_stats: Tuple,  # inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None,
    ) -> ParamsCLDS:
        def fit_linear_regression(ExxT, ExyT, EyyT, N, use_bias):
            # Solve a linear regression given sufficient statistics
            W = utils.psd_solve(ExxT, ExyT).T
            Sigma = (EyyT - W @ ExyT - ExyT.T @ W.T + W @ ExxT @ W.T) / N
            if use_bias:
                bias = W[:, -1:]
                W = W[:, :-1]
            else:
                bias = None
            return W, bias, Sigma

        def fit_gplinear_regression(ZTZ, ZTY, wgp_prior):
            # Solve a linear regression in weight-space given sufficient statistics
            weights = jax.scipy.linalg.solve(
                ZTZ + jnp.eye(wgp_prior.n_basis_funcs * wgp_prior.input_dim),
                ZTY,
                assume_a="pos",
            )
            weights = weights.reshape(
                wgp_prior.n_basis_funcs, wgp_prior.input_dim, wgp_prior.output_dim
            ).transpose(0, 2, 1)
            return weights

        def fit_gplinear_regression_sylvester(
            ZTZ, Sigma, ZTY, YTY, N, wgp_prior, use_bias
        ):
            # Solve a linear regression in weight-space given sufficient statistics
            # weights = utils.jax_solve_sylvester(B, ZTZ, ZTY, assume_a='pos')
            weights = utils.jax_solve_sylvester_BS(ZTZ, Sigma, ZTY)
            Sigma = (
                YTY - weights.T @ ZTY - ZTY.T @ weights + weights.T @ ZTZ @ weights
            ) / N
            if use_bias:
                bias = weights[wgp_prior.n_basis_funcs * wgp_prior.input_dim :].reshape(
                    wgp_prior.n_basis_funcs, wgp_prior.output_dim, 1
                )
                weights = (
                    weights[: wgp_prior.n_basis_funcs * wgp_prior.input_dim]
                    .reshape(
                        wgp_prior.n_basis_funcs,
                        wgp_prior.input_dim,
                        wgp_prior.output_dim,
                    )
                    .transpose(0, 2, 1)
                )
            else:
                bias = None
                weights = weights.reshape(
                    wgp_prior.n_basis_funcs, wgp_prior.input_dim, wgp_prior.output_dim
                ).transpose(0, 2, 1)

            return weights, bias, Sigma

        # Sum the statistics across all batches
        stats = jax.tree_util.tree_map(partial(jnp.sum, axis=0), batch_stats)
        (
            init_stats,
            init_gp_stats,
            dynamics_stats,
            dynamics_gp_stats,
            emission_stats,
            emissions_gp_stats,
        ) = stats

        # Perform MLE estimation jointly
        sum_x0, sum_x0x0T, N = init_stats
        S = sum_x0x0T / N - jnp.outer(sum_x0, sum_x0) / (N**2)
        # S = (sum_x0x0T - jnp.outer(sum_x0, sum_x0)) / N

        if self.use_initial_prior:
            m = fit_gplinear_regression(*init_gp_stats, self.priors["init"])
        else:
            m = sum_x0 / N

        # Dynamics M-step
        As, bs, Q = fit_linear_regression(*dynamics_stats, self.use_dynamics_bias)
        if self.use_dynamics_prior:
            As, bs, Q = fit_gplinear_regression_sylvester(
                *dynamics_gp_stats,
                self.priors["dynamics"],
                self.use_dynamics_bias,
            )

        # # Use the following for homogeneous bias
        # b = jnp.mean(bs, axis=0)
        # bs = jnp.tile(b, (len(up), 1))

        # Emission M-step
        Cs, ds, R = fit_linear_regression(*emission_stats, self.use_emissions_bias)
        if self.use_emissions_prior:
            # In weight space
            Cs, ds, _ = fit_gplinear_regression_sylvester(
                *emissions_gp_stats,
                self.priors["emissions"],
                self.use_emissions_bias,
            )

        # logger.warning('Warning, fixing Q and R')
        # Q = jnp.eye(self.state_dim)  # Can fix Q to be identity (for identifiability)
        # R = jnp.eye(self.emission_dim)
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
            initial_params = self.initialize_params(
                jxr.PRNGKey(seed), emissions.shape[1]
            )
        elif (self.initial_params is not None) and (initial_params is None):
            # use previously stored initial params if no initial params provided
            initial_params = self.initial_params

        # overwrite initial params
        self.initial_params = initial_params

        @jit
        def em_step(params, emissions, conditions, mask):
            # Obtain current E-step stats and model log prob
            batch_stats, lls = vmap(partial(self.e_step, params))(
                emissions, conditions, mask
            )
            log_priors = vmap(partial(self.log_prior, params))(conditions)
            mll = lls.sum()
            lp = log_priors.sum() + mll

            # Update with M-step
            params = self.m_step(params, batch_stats)

            return params, (lp, mll)

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
        return params, log_probs
