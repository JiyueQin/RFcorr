# rfcorr
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

## Overview
**rfcorr** estimates the correlation between two random fields (maps) observed on the same grid, and gives standard errors, confidence intervals and p-values that account for spatial dependence. Neighbouring points of smooth maps are not independent, so treating them as i.i.d. understates the uncertainty of the sample correlation and makes tests far too liberal. rfcorr instead estimates the variance of the correlation with a heteroskedasticity and autocorrelation consistent (HAC) estimator over spatial lags. It works on grids of any dimension (1D, 2D images, 3D volumes) and on irregular regions defined by a mask.

## Features
- Estimate the correlation of two fields with four standard error methods:
  | `se_method` | Method | Assumes |
  |---|---|---|
  | `'general'` | RFCorr | stationarity only |
  | `'gauss'` | RFCorr-Gauss | jointly Gaussian fields |
  | `'indep'` | RFCorr-Indep | independent fields (Clifford-type) |
  | `'iid'` | i.i.d. | no spatial dependence |

  Each method returns one standard error at the estimated correlation (for confidence intervals) and one under rho = 0 (for tests).
- Choose the truncation lag of the Bartlett weights by the rule from the paper (`M='optimal_rate'`, the default: ceil(constant · n^(1/(d+2))) for n valid grid points in d dimensions, the rate that minimizes the mean squared error of the variance estimate), or fix it by hand
- Restrict the analysis to a mask of valid grid points
- Simulate correlated smooth fields (Gaussian, t, chi-squared noise; fields with different smoothness; fields that are uncorrelated but dependent) and run simulation studies comparing the methods

## Installation

1. Create a Python 3.11 or newer environment, for example with conda:

    ```
    conda create --name rfcorr python=3.12
    conda activate rfcorr
    ```

2. Install the package:

    ```
    pip install git+https://github.com/JiyueQin/RFcorr.git
    ```

    For development, clone the repository and install it in editable mode with the test and job-script dependencies:

    ```
    git clone https://github.com/JiyueQin/RFcorr.git
    cd RFcorr
    pip install -e ".[jobs,dev]"
    ```

## Usage

Every function is available from the top-level package:

```python
import rfcorr as rc
```

- Simulate two smooth 2D maps with correlation 0.4, and an elliptical region to analyse

```python
X, Y = rc.gen_corr_field(rho=0.4, dim=(100, 100), fwhm=4, seed=0)
mask = rc.gen_mask(X.shape, axes=(1, 0.8))
```

- Estimate the correlation and its standard errors

```python
rho_est, se, se_under_null, runtime, M = rc.corr(X, Y, se_method="general", mask=mask)
# rho_est = 0.436, se = 0.034, se_under_null = 0.073, lag truncation M = (9, 9)
```

- Build a confidence interval and test H0: rho = 0

```python
lower, upper = rc.confidence_interval(rho_est, se)  # (0.367, 0.501), on the Fisher z scale
p = rc.pvalue(rho_est, se_under_null)
```

- Run a small simulation study comparing the methods

```python
setting_df = rc.make_setting_df(nx=[25, 50], rho=[0, 0.4], fwhm=[2, 4])
method_df = rc.make_method_df(method=["general", "gauss", "indep", "iid"])
summary = rc.sim_multiple(setting_df, method_df, m_sim=100, return_summary=True)
summary[["nx", "rho", "fwhm", "method", "bias_mean", "est_sd", "se_mean", "cover_rate", "power"]]
```

## Package Structure

| Folder | Contents |
|---|---|
| `rfcorr/fields/` | `gen_field`, `gen_corr_field`, `gen_mask`, `gaussian_filter_acf` |
| `rfcorr/estimation/` | `corr`, `confidence_interval`, `pvalue` (plus development functions, see below) |
| `rfcorr/simulation/` | `make_setting_df`, `make_method_df`, `sim_single`, `sim_multiple`, `results_summary` |
| `rfcorr/defaults/` | package-wide settings in `defaults.toml`, read with `get_defaults` |
| `examples/` | short examples for every function (`ex_<function>.py`) |
| `tutorials/` | `demo.ipynb`: simulation, estimation, inference and a small simulation study in 1D, 2D and 3D |
| `tests/` | tests for every function; run with `python -m pytest tests` |

Each function's docstring describes its arguments and output, e.g. `help(rc.corr)`.

### Development functions
`rc.corr` offers only the methods of the paper. `rc.corr_dev` is the development version behind it, with every option: other kernels, experimental and legacy lag rules (including a data-driven plug-in rule that is still under development and not verified), a known true correlation, and diagnostics. The simulation functions use it. The building blocks `get_se`, `get_est_exp_ls`, `est_exp_X0Yh`, `kernel`, `kernel_weights`, `plugin_bandwidth` and `canonical_se_method` are also development functions. Their interfaces may change.

### Simulation scripts
The repository also holds the code used for the simulation studies, which uses the package but is not part of it:
- `server_jobs/job_inputs/gen_inputs.py`: builds the settings and methods tables for server jobs
- `server_jobs/array_job/`: a Slurm job array that splits the settings across tasks with checkpoints, and `combine_and_summarize.py` to collect the results. Install the pinned environment on the server with `pip install -r requirements.txt` from the repository root
- `plot_sim_results.Rmd`: plots of the simulation results

## Related Publication and Citation
This package implements the methods developed in:

Jiyue Qin, Armin Schwartzman, Samuel Davenport. Asymptotic properties of the Pearson correlation between weakly dependent random fields. To appear.

If you use this package, please cite the paper above.

## Contact

Maintained by **Jiyue Qin**,
PhD Candidate in Biostatistics, University of California San Diego
📧 j5qin@ucsd.edu
