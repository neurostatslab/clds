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
        return jnp.einsum("lji,tl->tij", weights, PhiX)

    def sample_weights(
        self, key: jxr.PRNGKey
    ) -> Float[Array, "n_basis_funcs output_dim input_dim"]:
        return jxr.normal(
            key,
            shape=(
                self.n_basis_funcs,
                self.output_dim + self.include_bias,
                self.input_dim,
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
        latent_dim: dimension of the latent state space.
        emission_dim: dimension of the observation space.

    By default: A has wGP prior, whereas b and C are optional.
                If wGP priors are not provided for b and C, they are learned as C fixed, b time-varying.
    This is a early version, only currently supporting EM. Does not support sampling. Does not support inputs other than GP conditions.
    """

    def __init__(self, wgps: dict, latent_dim: int, emission_dim: int):
        self.wgps = wgps
        assert "A" in self.wgps, "Dynamics GP prior is required"
        if "b" not in self.wgps:
            self.wgps["b"] = None
        if "C" not in self.wgps:
            self.wgps["C"] = None
        if "m0" not in self.wgps:
            self.wgps["m0"] = None

        self.latent_dim = latent_dim
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
            m0_targets = Ex0.reshape(1, 1, self.latent_dim)
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
            bias_targets = bias_targets.reshape(len(up), 1, self.latent_dim)
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
            weights = jax.scipy.n_basis_funcsinalg.solve(
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
        # Q = jnp.eye(self.latent_dim) # Can fix Q to be identity (for identifiability)
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
    latent_dim : int
        dimension of the latent state space.
    emission_dim : int
        dimension of the observation space.


    By default: A has wGP prior, whereas b and C are optional.
                If wGP priors are not provided for b and C, they are learned as C fixed, b time-varying.
    This is a early version, only currently supporting EM. Does not support sampling. Does not support inputs other than GP conditions.
    """

    def __init__(self, priors: dict, latent_dim: int, emission_dim: int):
        self.priors = {
            "dynamics": None,
            "emissions": None,
            "init": None,
        }
        self.priors.update(priors)
        self.latent_dim = latent_dim
        self.emission_dim = emission_dim
        self.params = None

    def log_prior(self, params: ParamsCLDS2, inputs):
        """Compute the log prior of the parameters. Conditions are inputs"""

        if self.priors["dynamics"] is None:
            logprior_Ab = 0.0
        else:
            logprior_Ab = self.priors["dynamics"].log_prob_weights(
                jnp.concatenate((params.dynamics_matrix, params.dynamics_bias), axis=1)
            )

        if self.priors["emissions"] is None:
            logprior_Cd = 0.0
        else:
            logprior_Cd = self.priors["emissions"].log_prob_weights(
                jnp.concatenate(
                    (params.emissions_matrix, params.emissions_bias), axis=1
                )
            )

        if self.priors["init"] is None:
            logprior_m0 = 0.0
        else:
            logprior_m0 = self.priors["init"].log_prob_weights(params.init_mean)

        return logprior_Ab + logprior_Cd + logprior_m0

    def weights_to_params(self, params, inputs):
        """Transform weights of weight space into parameters.
        Implement as needed for all weight-space GP priors."""
        if self.priors["dynamics"] is None:
            As = params.dynamics_matrix
            bs = params.dynamics_bias
        else:
            As = self.priors["dynamics"](params.dynamics_matrix, inputs)
            bs = (
                self.priors["dynamics"](params.dynamics_bias, inputs)
                if self.priors["dynamics"].include_bias
                else jnp.zeros((As.shape[0], As.shape[-1]))
            )

        if self.priors["emissions"] is None:
            Cs = params.emissions_matrix
            ds = params.emissions_bias
        else:
            Cs = self.priors["emissions"](params.emissions_matrix, inputs)
            ds = (
                self.priors["emissions"](params.emissions_bias, inputs)
                if self.priors["emissions"].include_bias
                else jnp.zeros(Cs.shape[0], Cs.shape[-1])
            )

        m0 = (
            self.priors["init"](params.init_mean, inputs)[0]
            if self.priors["init"] is not None
            else params.init_mean
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
            "S0": params.init_cov,
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
        if self.priors["init"] is None:
            wgpm0_stats = None
        else:
            m0_targets = Ex0.reshape(1, 1, self.latent_dim)
            XTX_m0 = jnp.ones((1, 1, 1))

            _cond = (
                up[0].reshape(1) if up[0].ndim == 0 else up[0].reshape(1, up.shape[1])
            )
            wgpm0_stats = weightspace_stats(
                XTX_m0, m0_targets, self.priors["init"], _cond
            )

        # expected sufficient statistics for the dynamics
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        sum_xpxnT = Expxn.sum(0)
        sum_xpxpT = Vxp.sum(0) + Exp.T @ Exp
        sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
        dynamics_stats = (sum_xpxpT, sum_xpxnT, sum_xnxnT, num_timesteps - 1)

        # full E-step sufficient stats
        ExpxpT = jnp.einsum("ti,tj->tij", Exp, Exp) + Vxp
        _Phi = self.priors["dynamics"].evaluate_basis(up)

        # W_a terms
        # (T x L) (T x L) (T x D2 X D2) -> (LD2 x LD2)
        # (T x L) (T x D2 x D2) (T x L) -> (L x D2 x D2 x L)
        N1T = jnp.einsum("tk,tl,tij->kilj", _Phi, _Phi, ExpxpT).reshape(
            -1,
            self.priors["dynamics"].n_basis_funcs * self.priors["dynamics"].input_dim,
        )
        # (T x L) (T x D2 x D1) -> (LD2 x D1)
        # (L x D2 x D1)
        NDel = jnp.einsum("tk,tim->kim", _Phi, Expxn).reshape(
            -1, self.priors["dynamics"].output_dim
        )

        # b / W_b terms
        if self.priors["dynamics"] is None:
            # (T x L) (T x D2) -> (1 x LD2)
            PhiTZ = jnp.einsum("tl,ti->li", _Phi, Exp).reshape(
                1,
                self.priors["dynamics"].output_dim * self.priors["dynamics"].input_dim,
            )
            # (1 x 1)
            _PhiTPhi = jnp.ones((1, 1))
            # (1 X D1)
            PhiTX = Exn.sum(axis=0).reshape(1, -1)
        else:
            # (T x L) (T x L) (T x D2) -> (L x LD2)
            # (T x L) (T x 1 x D2) (T x L) -> (L x 1 X D2 x L)
            PhiTZ = jnp.einsum("tk,tl,ti->kli", _Phi, _Phi, Exp).reshape(
                -1,
                self.priors["dynamics"].n_basis_funcs
                * self.priors["dynamics"].input_dim,
            )
            # (T x L) (T x L) -> (L x L)
            _PhiTPhi = jnp.einsum("tk,tl->kl", _Phi, _Phi)
            # (T x L) (T x D1) -> (L x D1)
            # (T x L) (T x 1 x D1) -> (L x 1 x D1)
            PhiTX = jnp.einsum("tk,tm->km", _Phi, Exn)
        # (LD2 x L)
        # or (LD2 x 1)
        ZTPhi = PhiTZ.T

        ## sylvester equation stats
        ##-- A --##
        # left --
        # ((LD2+L) x LD2) for W_b
        # or ((LD2+1) x LD2) for b
        AL = jnp.concatenate((N1T, PhiTZ), axis=0)
        # right --
        # ((LD2+L) x L) for W_b
        # or ((LD2+1) x 1) for b
        AR = jnp.concatenate((ZTPhi, _PhiTPhi), axis=0)
        # combine --
        # ((LD2+L) x (LD2+L)))) for W_b
        # or ((LD2+1) x (LD2+1)) for b
        A = jnp.concatenate((AL, AR), axis=-1)

        ##-- B --##
        B = params.dynamics_cov

        ##-- C --##
        # ((LD2+L) x D1) for W_b
        # or ((LD2+1) x D1) for b
        C = jnp.concatenate((NDel, PhiTX), axis=0)
        Ab_sylvester_stats = (A, B, C)

        # more expected sufficient statistics for the emissions
        y = emissions
        sum_xxT = Vx.sum(0) + Ex.T @ Ex
        sum_xyT = Ex.T @ y
        sum_yyT = emissions.T @ emissions
        emission_stats = (sum_xxT, sum_xyT, sum_yyT, num_timesteps)

        if self.priors["emissions"] is None:
            wgpC_stats = None
            wgpC_sylvester_stats = None
        else:
            _xxT = jnp.einsum("ti,tj->tij", Ex, Ex) + Vx
            _xyT = jnp.einsum("ti,tj->tij", Ex, y)
            wgpC_stats = weightspace_stats(_xxT, _xyT, self.priors["emissions"], inputs)

            wgpC_sylvester_stats = (wgpC_stats[0], params.R, wgpC_stats[1], 1)

        return (
            init_stats,
            wgpm0_stats,
            dynamics_stats,
            emission_stats,
            wgpC_stats,
            wgpC_sylvester_stats,
            Ab_sylvester_stats,
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
        stats = jax.tree_util.tree_map(partial(jnp.mean, axis=0), batch_stats)
        (
            init_stats,
            wgpm0_stats,
            dynamics_stats,
            emission_stats,
            wgpC_stats,
            wgpC_sylvester_stats,
            Ab_sylvester_stats,
        ) = stats

        # Perform MLE estimation jointly
        sum_x0, sum_x0x0T, N = init_stats
        S = sum_x0x0T / N - jnp.outer(sum_x0, sum_x0) / (N**2)
        if self.priors["init"] is None:
            m = sum_x0 / N
        else:
            m = fit_gplinear_regression(*wgpm0_stats, self.priors["init"])

        # Dynamics M-step
        F_static, Q = fit_linear_regression(*dynamics_stats)

        # flip input and output dimensions
        weights = utils.jax_solve_sylvester_BS(*Ab_sylvester_stats)
        As = (
            weights[
                : (
                    self.priors["dynamics"].n_basis_funcs
                    * self.priors["dynamics"].input_dim
                )
            ]
            .reshape(
                self.priors["dynamics"].n_basis_funcs,
                self.priors["dynamics"].input_dim,
                self.priors["dynamics"].output_dim,
            )
            .transpose(0, 2, 1)
        )
        if self.priors["dynamics"] is None:
            bs = weights[
                (
                    self.priors["dynamics"].n_basis_funcs
                    * self.priors["dynamics"].input_dim
                ) :
            ]
        else:
            bs = weights[
                (
                    self.priors["dynamics"].n_basis_funcs
                    * self.priors["dynamics"].input_dim
                ) :
            ].reshape(
                self.priors["dynamics"].n_basis_funcs,
                1,
                self.priors["dynamics"].output_dim,
            )

        # # Use the following for homogeneous bias
        # b = jnp.mean(bs, axis=0)
        # bs = jnp.tile(b, (len(up), 1))

        # Emission M-step
        H_static, R = fit_linear_regression(*emission_stats)
        if self.priors["emissions"] is None:
            Cs = H_static
        else:
            # In weight space
            # W_C = fit_gplinear_regression(*wgpC_stats, self.wgps['C'])
            Cs = fit_gplinear_regression_sylvester(
                wgpC_sylvester_stats[0],
                wgpC_sylvester_stats[1] / wgpC_sylvester_stats[3],
                wgpC_sylvester_stats[2],
                wgp_prior=self.priors["emissions"],
            )

        # logger.warning('Warning, fixing Q and R')
        # Q = jnp.eye(self.latent_dim) # Can fix Q to be identity (for identifiability)
        # R = jnp.eye(self.emission_dim)
        params = ParamsCLDS2(
            init_mean=m,
            init_cov=S,
            dynamics_matrix=As,
            dynamics_bias=bs,
            dynamics_cov=Q,
            emissions_matrix=Cs,
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
