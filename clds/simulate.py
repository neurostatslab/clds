import jax
import jax.numpy as jnp
import jax.random as jxr
from .models import CLDS2
from functools import partial


# @partial(jax.vmap, in_axes=0)
def simulate_hd_dynamics(theta, epsilon=0.1, noise_scale=0.2):
    # initial conditions
    m0 = jnp.array(
        [jnp.cos(theta[0]), jnp.sin(theta[0])]
    )  # Change to 0 if needed for composite dynamics
    S0 = noise_scale * jnp.eye(2)

    # dynamics
    u = jnp.array([jnp.cos(theta), jnp.sin(theta)])
    v = jnp.array([-jnp.sin(theta), jnp.cos(theta)])
    A = ((1 - epsilon) * v[:, None] * v[None, :]).transpose(-1, 0, 1)
    b = u.T  # + omega * v

    Q = noise_scale**2 * jnp.eye(2)
    return A, b, Q, m0, S0


def simulate_hd_data(
    n_steps=100,
    n_batches=100,
    n_neurons=10,
    epsilon=0.1,
    noise_scale=0.2,
    seed=1,
):
    def form_batch(i):
        key = jxr.PRNGKey(seed + i)
        theta = jnp.cumsum(0.4 * jxr.normal(key, shape=(n_steps,))) % (2 * jnp.pi)
        true_As, true_bs, true_Q, true_m0, true_S0 = simulate_hd_dynamics(
            theta, epsilon, noise_scale
        )
        x, y = CLDS2.run_dynamics(
            key,
            true_As,
            true_bs,
            true_Q,
            true_Cs,
            true_ds,
            true_R,
            true_m0,
            true_S0,
        )
        return theta, x, y

    # neuron tuning parameters
    true_C = jxr.normal(jxr.PRNGKey(0), (n_neurons, 2))
    true_Cs = jnp.tile(true_C, (n_steps, 1, 1))
    true_ds = jnp.zeros((n_steps, n_neurons))
    true_R = noise_scale**2 * jnp.eye(n_neurons)

    thetas, X, Y = jax.vmap(form_batch)(jnp.arange(n_batches))
    thetas = jnp.stack(thetas, axis=0)
    X = jnp.stack(X, axis=0)
    Y = jnp.stack(Y, axis=0)

    return thetas, X, Y, true_Cs
