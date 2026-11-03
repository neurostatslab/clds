# Conditionally Linear Dynamical Systems

Neural population activity exhibits complex, nonlinear dynamics, varying in time, over trials, and across experimental conditions. 
Here, we develop Conditionally Linear Dynamical System (CLDS) models as a general-purpose method to characterize these dynamics.
These models use Gaussian Process (GP) priors to capture the nonlinear dependence of circuit dynamics on task and behavioral variables. 
Conditioned on these covariates, the data is modeled with linear dynamics. 
This allows for transparent interpretation and tractable Bayesian inference. 

This code package introduces the CLDS generative model, as well as example use cases. For more details, see our NeurIPS 2025 paper at https://openreview.net/forum?id=xgmMdTTzjj.

**Note:** This research code remains a work-in-progress to some extent. Please use at your own risk and reach out to us (vgeadah@flatironinstitute.org, awilliams@flatironinstitute.org) if you have questions.

## Repo structure and preliminary guide

A succinct structure of the repo is as follows:
```
.
├── clds/                           The CLDS package
│   ├── models.py                       CLDS and WeightSpaceGaussianProcess model classes
│   ├── params.py                       Parameter datastructures
│   ├── inference.py                    Inference methods -- see primarly fit_em()
│   ├── utils.py                        Basis construction, initialization (subspace ID, PPCA), helpers
│   ├── postprocessing.py               Analysis of fitted models: fixed points, E[x|u], composite dynamics
│   ├── simulate.py                     Simulated head-direction (HD) data
│   └── legacy.py                       Deprecated original CLDS model
├── notebooks/                      Demo notebooks fitting CLDS models
│   ├── start_here.ipynb                Minimal example from the Usage section below
│   ├── HD_simulated.ipynb              Fit CLDS to synthetic mouse HD experiment
│   ├── HD_mouse.ipynb                  Fit CLDS to mouse thalamic HD data
│   └── conditional_linreg.ipynb        Closed-form MAP inference in conditional linear regression
├── tests/                          pytest test suite
└── pyproject.toml                  Package metadata and dependencies
```

We recommend checking out `notebooks/HD_simulated.ipynb` and `notebooks/HD_mouse.ipynb` for example uses of CLDS models on 
respectively synthetic and real neural datasets.

## Installation

Requires Python >= 3.10. Dependencies are listed in [`pyproject.toml`](pyproject.toml).

1. Clone the repository
```
git clone https://github.com/neurostatslab/clds.git
cd clds
```

2. Create a **virtual environment** and activate it
```
python -m venv .venv
source .venv/bin/activate
```

3. Install the package in editable mode, along with its requirements. To also add the dependencies for running the tests (pytest), run `pip install -e ".[dev]"` instead.
```
pip install -e .
```

4. (Optional) For GPU support (still in progress), install [**JAX**](https://docs.jax.dev/en/latest/installation.html) with CUDA, picking the version that matches your driver, e.g.
```
pip install -U "jax[cuda13]"
```

To use the environment in the notebooks, select the `.venv` interpreter as the Jupyter kernel (`ipykernel` is included in the requirements).

Since the code is preliminary, you will be able to use `git pull` to get updates as we release them.

## Usage

CLDS fits latent dynamics whose parameters vary with observed **conditions**, such as time, stimulus, or behavior. 
Specifically, CLDS models responses $y_t \in \mathbb{R}^N$ as emissions from a latent time-varying linear dynamical system in $x_t \in \mathbb{R}^D$, with dynamics governed by the conditions $u_t \in \mathbb{R}^M$,
$$
x_{t+1} = A(u_t) x_t + b(u_t) + \epsilon_t
\qquad
y_t = C(u_t)x_t + d(u_t) + \omega_t
$$
over time steps $t \in \{1, \dots, T\}$ from initial condition $x_1 \sim \mathcal{N}(m(u_1), Q_1)$, and where $\epsilon_t \sim \mathcal{N}(0, Q)$ and $\omega_t \sim N (0, R)$ are sources of noise, i.i.d. in time. 

To use and fit CLDS to your data, you need two arrays:

- `emissions`: observations of shape `(B, T, N)`—`B` trials, `T` timesteps, and `N` observed variables.
- `conditions`: covariates of shape `(B, T, M)`, where `M` is the number of condition variables.

### Example

For our example, let us time as the single condition $u_t = t$ and fit
$$
x_{t+1} = A(t)x_t + b(t) + \epsilon_t,
\qquad
y_t = Cx_t + \omega_t.
$$
We place GP priors on $A$ and $b$, allowing the dynamics to vary smoothly over time. The emission matrix $C$ is constant, and there is no emission bias.

```python
import jax.numpy as jnp
from jax import random
from clds import utils
from clds.models import CLDS

# Placeholder observations: replace with your data, shaped (B, T, N).
B, T, N = 8, 100, 96
emissions = random.normal(random.key(0), shape=(B, T, N))

# Use the same normalized time condition for every trial.
time = jnp.linspace(0.0, 1.0, T)
conditions = jnp.broadcast_to(time[None, :, None], (B, T, 1))

# Choose the latent dimension and initialize the model.
D = 2
basis = utils.initialize_basis(conditions, lengthscale=0.1)
model = CLDS(
    state_dim=D,
    emission_dim=N,
    basis=basis,
    gp_priors=["A", "b"],
    use_dynamics_bias=True,
    use_emissions_bias=False,
)

# Fit the model. An optional mask has shape (B, T).
params, (log_liks, _) = model.fit(emissions, conditions, num_iters=100)
```

For your own application, replace `emissions` with your observations and choose the latent dimension `D`. To model dependence on variables other than time, replace `conditions` with your covariates of shape `(B, T, M)` and choose a GP lengthscale appropriate to their scale.