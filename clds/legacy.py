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
from typing import NamedTuple

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
from clds.models import WeightSpaceGaussianProcess

class ParamsGPLDS(NamedTuple):
    emissions: ParamsEmission
    likelihood: ParamsNormalLikelihood
    
class ParamsCLDS(NamedTuple):
    r'''!Deprecated: use `clds.params.ParamsCLDS` (formerly CLDS2).'''
    m0: Float[Array, "state_dim"]  # Accessed if 'm0' wgp prior is None
    m0_gp_weights: Float[Array, "state_dim 1 len_basis"]
    S0: Float[Array, "state_dim state_dim"]
    dynamics_gp_weights: Float[Array, "state_dim state_dim len_basis"]
    emissions_gp_weights: Float[Array, "emission_dim state_dim len_basis"]
    Cs: Float[
        Array, "num_timesteps emission_dim state_dim"
    ]  # Accessed if 'C' wgp prior is None
    bias_gp_weights: Float[Array, "state_dim 1 len_basis"]
    bs: Float[Array, "num_timesteps state_dim"]  # Accessed if 'b' wgp prior is None
    Q: Float[Array, "state_dim state_dim"]
    R: Float[Array, "emission_dim emission_dim"]

class CLDS:
    """
    !Deprecated: use `clds.models.CLDS` (formerly CLDS2).

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
        warnings.warn(
            "clds.legacy.CLDS is deprecated and will be removed; use clds.models.CLDS.",
            DeprecationWarning, stacklevel=2,
        )
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
