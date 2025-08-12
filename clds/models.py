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
        basis_funcs: list,
        input_dim: int = 1,
        output_dim: int = 1,
        include_bias: bool = False,
    ):
        self.basis_funcs = basis_funcs
        self.n_basis_funcs = len(basis_funcs)
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
        return jnp.einsum("lji,tl->tji", weights, PhiX)

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
        return jnp.array([jax.vmap(f)(u) for f in self.basis_funcs]).T

    def sample(
        self, key: jxr.PRNGKey, conditions: Float[Array, "n_steps n_conditions"]
    ) -> Float[Array, "n_steps output_dim input_dim"]:
        """
        Sample from the GP prior at the points `conditions`
        """
        weights = self.sample_weights(key)
        PhiX = self.evaluate_basis(conditions)
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

    Parameters
    ----------
    priors :
        dict of WeightSpaceGaussianProcess (wGP) objects for the parameters {A, b, C, d, m0}.
    state_dim : int
        dimension of the latent state space.
    emission_dim : int
        dimension of the observation space.


    By default: A has wGP prior, whereas b and C are optional.
                If wGP priors are not provided for b and C, they are learned as C fixed, b time-varying.
    This is a early version, only currently supporting EM. Does not support sampling. Does not support inputs other than GP conditions.
    """

    def __init__(
        self,
        priors: dict,
        state_dim: int,
        emission_dim: int,
        initial_params: Optional[ParamsCLDS2] = None,
    ):
        self.priors = {
            "dynamics": None,
            "emissions": None,
            "init": None,
        }
        self.priors.update(priors)
        self.use_dynamics_prior = self.priors["dynamics"] is not None
        self.include_dynamics_bias = (
            self.priors["dynamics"].include_bias if self.use_dynamics_prior else False
        )
        self.use_emissions_prior = self.priors["emissions"] is not None
        self.include_emissions_bias = False
        self.use_initial_prior = self.priors["init"] is not None
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.initial_params = initial_params
        self.params = None

    def initialize_params(self, num_samples, seed: int = 2):
        Ab_key, Cd_key, m0_key = jxr.split(jxr.PRNGKey(seed), 3)
        if self.use_dynamics_prior:
            dynamics_weights = self.priors["dynamics"].sample_weights(Ab_key)
            if self.include_dynamics_bias:
                dynamics_bias = dynamics_weights[:, :, -1:]
                dynamics_weights = dynamics_weights[:, :, :-1]
            else:
                dynamics_bias = jnp.zeros((num_samples, self.state_dim))
        else:
            dynamics_weights = jxr.normal(
                Ab_key, (num_samples, self.state_dim, self.state_dim)
            )
            dynamics_bias = (
                jxr.normal(Ab_key, (num_samples, self.state_dim, self.state_dim))
                if self.include_dynamics_bias
                else jnp.zeros((num_samples, self.state_dim))
            )

        if self.use_emissions_prior:
            emissions_weights = self.priors["emissions"].sample_weights(Cd_key)
            if self.include_emissions_bias:
                emissions_bias = emissions_weights[:, :, -1:]
                emissions_weights = emissions_weights[:, :, :-1]
            else:
                emissions_bias = jnp.zeros((num_samples, self.emission_dim))
        else:
            emissions_weights = jnp.tile(
                jxr.normal(Cd_key, (self.emission_dim, self.state_dim)),
                (num_samples, 1, 1),
            )
            emissions_bias = (
                jxr.normal(Cd_key, (num_samples, self.emission_dim))
                if self.include_emissions_bias
                else jnp.zeros((num_samples, self.emission_dim))
            )

        if self.use_initial_prior:
            initial_mean = self.priors["init"].sample_weights(m0_key)
        else:
            initial_mean = jxr.normal(m0_key, (num_samples, self.state_dim))

        return ParamsCLDS2(
            dynamics_weights=dynamics_weights,
            dynamics_bias=dynamics_bias,
            emissions_weights=emissions_weights,
            emissions_bias=emissions_bias,
            initial_mean=initial_mean,
            initial_cov=jnp.eye(self.state_dim),
            dynamics_cov=jnp.eye(self.state_dim),
            emissions_cov=jnp.eye(self.emission_dim),
        )

    def sample_dynamics(
        self, inputs: Float[Array, "num_timesteps input_dim"], seed: int = 2
    ):
        As, Cs, bs, ds, m0 = self.weights_to_params(self.params, inputs)
        key = jxr.PRNGKey(seed)

        def f(x, args):
            A, b, C, d, (em_key, dy_key) = args

            emissions_noise = jxr.multivariate_normal(
                em_key, jnp.zeros(self.emission_dim), self.params.emissions_cov
            )
            y = C @ x + d + emissions_noise

            dynamics_noise = jxr.multivariate_normal(
                dy_key, jnp.zeros(self.state_dim), self.params.dynamics_cov
            )
            x_next = A @ x + b + dynamics_noise
            return x_next, (x_next, y)

        x_init = jxr.multivariate_normal(key, m0, self.params.initial_cov)
        subkeys = jxr.split(key, num=(As.shape[0], 2))
        _, (x_nexts, ys) = jax.lax.scan(f, x_init, xs=(As, bs, Cs, ds, subkeys))
        xs = jnp.concatenate((x_init[None, :], x_nexts[:-1]), axis=0)
        return xs, ys

    def log_prior(self, params: ParamsCLDS2, inputs):
        """Compute the log prior of the parameters. Conditions are inputs"""

        if self.use_dynamics_prior:
            logprior_Ab = self.priors["dynamics"].log_prob_weights(
                jnp.concatenate(
                    (params.dynamics_weights, params.dynamics_bias), axis=-1
                )
            )
        else:
            logprior_Ab = 0.0

        if self.use_emissions_prior:
            if self.include_emissions_bias:
                logprior_Cd = self.priors["emissions"].log_prob_weights(
                    jnp.concatenate(
                        (params.emissions_weights, params.emissions_bias), axis=-1
                    )
                )
            else:
                logprior_Cd = self.priors["emissions"].log_prob_weights(
                    params.emissions_weights
                )
        else:
            logprior_Cd = 0.0

        if self.use_initial_prior:
            logprior_m0 = self.priors["init"].log_prob_weights(params.initial_mean)
        else:
            logprior_m0 = 0.0

        return logprior_Ab + logprior_Cd + logprior_m0

    def weights_to_params(self, params, inputs):
        """Transform weights of weight space into parameters.
        Implement as needed for all weight-space GP priors."""
        if self.use_dynamics_prior:
            As = self.priors["dynamics"](params.dynamics_weights, inputs)
            bs = (
                self.priors["dynamics"](params.dynamics_bias, inputs)
                if self.priors["dynamics"].include_bias
                else jnp.zeros((inputs.shape[0], self.priors["dynamics"].output_dim))
            )
        else:
            As = params.dynamics_weights
            bs = params.dynamics_bias

        if self.use_emissions_prior:
            Cs = self.priors["emissions"](params.emissions_weights, inputs)
            ds = (
                self.priors["emissions"](params.emissions_bias, inputs)
                if self.priors["emissions"].include_bias
                else jnp.zeros((inputs.shape[0], self.priors["emissions"].output_dim))
            )
        else:
            Cs = jnp.tile(params.emissions_weights, (inputs.shape[0], 1, 1))
            ds = params.emissions_bias

        m0 = (
            self.priors["init"](params.initial_mean, inputs)[0]
            if self.use_initial_prior
            else params.initial_mean
        )  #! Some unnecessary computation, keeping only t=0
        return As, Cs, bs.squeeze(), ds.squeeze(), m0.squeeze()

    def smoother(self, params: ParamsCLDS, emissions, inputs):
        """inputs as conditions"""
        # Format params
        As, Cs, bs, ds, m0 = self.weights_to_params(params, inputs)
        if Cs.ndim == 2:
            Cs = jnp.tile(Cs[None], (len(inputs), 1, 1))

        # Run the smoother
        lgssm_params = {
            "m0": m0,
            "S0": params.initial_cov,
            "As": As,
            "bs": bs,
            "Q": params.dynamics_cov,
            "Cs": Cs,
            "ds": ds,
            "R": params.emissions_cov,
        }
        return utils.lgssm_smoother(**lgssm_params, ys=emissions)

    def e_step(
        self,
        params: ParamsCLDS,
        emissions: Float[Array, "num_timesteps emission_dim"],
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    ):
        def weightspace_stats(
            XTX: Float[Array, "n_steps input_dim input_dim"],
            XTY: Float[Array, "n_steps input_dim output_dim"],
            wgp_prior: WeightSpaceGaussianProcess,
            conditions: Float[Array, "n_steps condition_dim"],
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

        def weightspace_stats2(
            basis: Float[Array, "n_steps n_basis_funcs"],
            XTX: Float[Array, "n_steps input_dim input_dim"] = None,
            XTY: Float[Array, "n_steps input_dim output_dim"] = None,
        ) -> tuple:
            """
            Compute the expected sufficient statistics for the weight-space GP prior.
            Provide the sufficient stats X^T X and X^T Y for the problem Y = A(C)X + noise.
            This returns the expanded stats Phi @ X^T X @ Phi^T and Phi @ X^T Y for the basis functions Phi(C).
            """
            n_basis_funcs = basis.shape[-1]

            if XTX is not None:
                input_dim = XTX.shape[-1]
                ZTZ = jnp.einsum("tk,tl,tij->kilj", _Phi, _Phi, XTX).reshape(
                    -1,
                    n_basis_funcs * input_dim,
                )
            else:
                ZTZ = None

            if XTY is not None:
                output_dim = XTY.shape[-1]
                ZTY = jnp.einsum("tk,tim->kim", _Phi, XTY).reshape(-1, output_dim)
            else:
                ZTY = None

            return ZTZ, ZTY

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
        # upb = jnp.concatenate((up, jnp.ones((num_timesteps - 1, 1))), axis=1)
        upb = jnp.vstack((up, jnp.ones((num_timesteps - 1)))).T
        # u = inputs

        # expected sufficient statistics for the initial distribution
        Ex0 = smoothed_means[0]
        Ex0x0T = smoothed_covariances[0] + jnp.outer(Ex0, Ex0)
        init_stats = (Ex0, Ex0x0T, 1)
        if self.use_initial_prior:
            m0_targets = Ex0.reshape(1, 1, self.state_dim)
            XTX_m0 = jnp.ones((1, 1, 1))

            _cond = (
                up[0].reshape(1) if up[0].ndim == 0 else up[0].reshape(1, up.shape[1])
            )
            wgpm0_stats = weightspace_stats(
                XTX_m0, m0_targets, self.priors["init"], _cond
            )
        else:
            wgpm0_stats = None

        # expected sufficient statistics for the dynamics
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        sum_xpxpT = Vxp.sum(0) + Exp.T @ Exp
        sum_xpT = Exp.T @ upb
        sum_xpxpT = jnp.block([[sum_xpxpT, sum_xpT], [sum_xpT.T, upb.T @ upb]])
        sum_xpxnT = jnp.block([[Expxn.sum(0)], [upb.T @ Exn]])
        # sum_xpxnT = Expxn.sum(0)
        sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
        dynamics_stats = (sum_xpxpT, sum_xpxnT, sum_xnxnT, num_timesteps - 1)

        # full E-step sufficient stats
        if self.use_dynamics_prior:
            ExpxpT = jnp.einsum("ti,tj->tij", Exp, Exp) + Vxp
            _Phi = self.priors["dynamics"].evaluate_basis(up)

            # (LD2 x LD2), (LD2 x D1)
            sum_zpzpT, sum_zpxnT = weightspace_stats2(_Phi, XTX=ExpxpT, XTY=Expxn)
            if self.include_dynamics_bias:
                # (L x LD2), (L x D1)
                sum_zpT, sum_znT = weightspace_stats2(
                    _Phi, XTX=Exp[:, None, :], XTY=Exn[:, None, :]
                )
                # ((LD2+L) x (LD2+L)))
                sum_zpzpT = jnp.block(
                    [
                        [sum_zpzpT, sum_zpT.T],
                        [sum_zpT, jnp.einsum("tk,tl->kl", _Phi, _Phi)],
                    ]
                )
                # ((LD2+L) x D1)
                sum_zpxnT = jnp.concatenate([sum_zpxnT, sum_znT], axis=0)
            dynamics_sylvester_stats = (
                sum_zpzpT,
                params.dynamics_cov,
                sum_zpxnT,
                sum_xnxnT,
                num_timesteps - 1,
            )

        # more expected sufficient statistics for the emissions
        y = emissions
        sum_xxT = Vx.sum(0) + Ex.T @ Ex
        sum_xyT = Ex.T @ y
        sum_yyT = emissions.T @ emissions

        if self.include_emissions_bias:
            ub = jnp.vstack((inputs, jnp.ones((num_timesteps)))).T
            sum_xT = Ex.T @ ub
            sum_xxT = jnp.block([[sum_xxT, sum_xT], [sum_xT.T, ub.T @ ub]])
            sum_xyT = jnp.block([[sum_xyT], [ub.T @ y]])

        emission_stats = (sum_xxT, sum_xyT, sum_yyT, num_timesteps)

        if self.use_emissions_prior:
            _Phi = self.priors["emissions"].evaluate_basis(inputs)
            ExxT = jnp.einsum("ti,tj->tij", Ex, Ex) + Vx
            ExyT = jnp.einsum("ti,tj->tij", Ex, y)
            sum_zzT, sum_zyT = weightspace_stats2(_Phi, ExxT, ExyT)

            if self.include_emissions_bias:
                sum_zT, sum_yT = weightspace_stats2(
                    _Phi, XTX=Ex[:, None, :], XTY=y[:, None, :]
                )
                sum_zzT = jnp.block(
                    [
                        [sum_zzT, sum_zT.T],
                        [sum_zT, jnp.einsum("tk,tl->kl", _Phi, _Phi)],
                    ]
                )
                sum_zyT = jnp.concatenate([sum_zyT, sum_yT], axis=0)

            emissions_sylvester_stats = (
                sum_zzT,
                params.emissions_cov,
                sum_zyT,
                sum_yyT,
                num_timesteps,
            )

        else:
            emissions_sylvester_stats = None

        return (
            init_stats,
            wgpm0_stats,
            dynamics_stats,
            emission_stats,
            emissions_sylvester_stats,
            dynamics_sylvester_stats,
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

        def fit_gplinear_regression_sylvester2(
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
            wgpm0_stats,
            dynamics_stats,
            emission_stats,
            Cd_sylvester_stats,
            Ab_sylvester_stats,
        ) = stats

        # Perform MLE estimation jointly
        sum_x0, sum_x0x0T, N = init_stats
        S = sum_x0x0T / N - jnp.outer(sum_x0, sum_x0) / (N**2)
        if self.use_initial_prior:
            m = fit_gplinear_regression(*wgpm0_stats, self.priors["init"])
        else:
            m = sum_x0 / N

        # Dynamics M-step
        # F_static, Q = fit_linear_regression(*dynamics_stats)

        # flip input and output dimensions
        As, bs, Q = fit_gplinear_regression_sylvester2(
            *Ab_sylvester_stats,
            self.priors["dynamics"],
            self.include_dynamics_bias,
        )

        # # Use the following for homogeneous bias
        # b = jnp.mean(bs, axis=0)
        # bs = jnp.tile(b, (len(up), 1))

        # Emission M-step
        H_static, R = fit_linear_regression(*emission_stats)
        if self.use_emissions_prior:
            # In weight space
            # W_C = fit_gplinear_regression(*wgpC_stats, self.wgps['C'])
            Cs, ds, R = fit_gplinear_regression_sylvester2(
                *Cd_sylvester_stats,
                self.priors["emissions"],
                self.include_emissions_bias,
            )
            # fit_gplinear_regression_sylvester(
            #     wgpC_sylvester_stats[0],
            #     wgpC_sylvester_stats[1] / wgpC_sylvester_stats[3],
            #     wgpC_sylvester_stats[2],
            #     wgp_prior=self.priors["emissions"],
            # )
        else:
            Cs = H_static

        # logger.warning('Warning, fixing Q and R')
        # Q = jnp.eye(self.state_dim) # Can fix Q to be identity (for identifiability)
        # R = jnp.eye(self.emission_dim)
        params = ParamsCLDS2(
            initial_mean=m,
            initial_cov=S,
            dynamics_weights=As,
            dynamics_bias=bs,
            dynamics_cov=Q,
            emissions_weights=Cs,
            emissions_bias=params.emissions_bias,
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
