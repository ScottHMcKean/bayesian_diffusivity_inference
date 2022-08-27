# Bayesian Diffusivity Inference
This repository contains data and source code the following manuscript:
"Segregating Hydraulic Fracturing Created Microseismicity from Induced Seismicity through Bayesian Inference of Non-Linear Pressure Diffusivity"

# Quick Start
The code is developed using Python 3.9., with the environment defined in `environment.yaml`. The main dependencies include pymc, arviz, pygam, pandas, and seaborn. Because we are using a Jupyter notebook we also required the ipykernel.

Install `via conda env create -f environment.yaml`

A run notebook is provided in the root of the repository: `run_bayesian_diffusivity_inference.ipynb`. The notebook is meant to run a single stage at a time - the results from the entire hydraulic fracturing program was generated using a compute cluster with multithreaded execution. Those results are summarized in `compiled_results.csv`.

The individual results from the three case studies are reviewed in `analyze_bayesian_diffusivity.ipynb`. This notebook reloads the PyMC model and thread traces from each scenario, producing trace plots, marginal energy plots, posterior distribution, and showing the filtered results.

A comparison notebook is also provided in `compile_bayesian_diffusivities.ipynb`. This notebook loads the precompiled results, runs a maximum likelihood quantile of linear diffusivity for comparison and generates comparison plots used in the paper.

# Figures
The `plots.py` module has functions to reproduce plots from the paper. These include:
Figure X:
- make_quantile_contour_plot(fig_size=(8,6))
- make_basic_diffusivity_plot(fig_size=(6,6))
- make_stage_plot(distances, params, well, stage, figsize=(10,3))




