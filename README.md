# Conditionally Linear Dynamical Systems

Neural population activity exhibits complex, nonlinear dynamics, varying in time, over trials, and across experimental conditions. 
Here, we develop Conditionally Linear Dynamical System (CLDS) models as a general-purpose method to characterize these dynamics.
These models use Gaussian Process (GP) priors to capture the nonlinear dependence of circuit dynamics on task and behavioral variables. 
Conditioned on these covariates, the data is modeled with linear dynamics. 
This allows for transparent interpretation and tractable Bayesian inference. 

This code package introduces the CLDS generative model, as well as example use cases.

**Note:** This research code remains a work-in-progress to some extent. It could use more documentation and examples. Please use at your own risk and reach out to us (victor.geadah@princeton.edu, anejatbakhsh@flatironinstitute.org) if you have questions.

## Repo structure and preliminary guide

A succinct structure of the repo is as follows:
```
.
├── models.py                       CLDS and WeightSpaceGaussianProcess model classes
├── params.py                       Parameter datastructures
├── inference.py                    Inference methods -- see primarly fit_em()
├── loader.py                   
├── notebooks                       Demo notebooks fitting CLDS models
│   ├── HD_simulated.ipynb              Fit CLDS to synthetic mouse HD experiment
│   ├── HD_simulated_simple.ipynb
│   ├── HD_mouse.ipynb                  Fit CLDS to mouse thalamic HD data
│   ├── conditional_linreg.ipynb        Closed-form MAP inference in conditional linear regression
├── postprocessing.py
└── utils.py
```

We recommend checking out `notebooks/HD_simulated.ipynb` and `notebooks/HD_mouse.ipynb` for example uses of CLDS models on 
respectively synthetic and real neural datasets. 

## Installation

1. Download and install [**anaconda**](https://docs.anaconda.com/anaconda/install/index.html)
2. Create a **virtual environment** using anaconda and activate it
```
conda create -n clds
conda activate clds
```

1. Install [**JAX**](https://github.com/google/jax) package on your machine, with CUDA if possible.

2. Install other requirements (matplotlib, scipy, sklearn, flax)

Since the code is preliminary, you will be able to use `git pull` to get updates as we release them.