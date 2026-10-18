# CLAUDE.md - rfcorr Repository Guidelines

## Package Overview
rfcorr estimates the correlation between two random fields (maps) on a grid of any dimension, with standard errors that account for spatial dependence, and runs the simulation studies that evaluate them. It provides:
- Field generation (`rfcorr/fields/`): smooth Gaussian and non-Gaussian fields, correlated pairs (including uncorrelated-but-dependent designs), the population autocorrelation of Gaussian-filtered fields, and elliptical masks
- Estimation (`rfcorr/estimation/`): `rc.corr` returns the correlation with its HAC standard errors at the plugin rho and under rho = 0 (methods 'general', 'gauss', 'indep', 'iid' = RFCorr, RFCorr-Gauss, RFCorr-Indep, iid in the paper), plus confidence intervals and p-values. `rc.corr_dev` and the building blocks (kernels, lag rules) are development functions, see below
- Simulation (`rfcorr/simulation/`): settings and method tables, `rc.sim_single` / `rc.sim_multiple`, and `rc.results_summary`
- Package-wide settings (`rfcorr/defaults/defaults.toml`, read with `rc.get_defaults()`): the SE method names, their earlier aliases, and the dependence designs kept only with rho = 0. Put new settings there, not as module-level constants

## Repository Layout
- `rfcorr/<subpackage>/` — functions; each subpackage `__init__.py` imports its functions, and `rfcorr/__init__.py` star-imports the subpackages to give the flat `rc.<function>` API
- `tests/<subpackage>/test_<function>.py` — tests; shared fixtures (`fields_2d`, `method_df`) are in `tests/conftest.py`. Run all with `python -m pytest tests`
- `examples/<subpackage>/ex_<function>.py` — about three short examples per function, separated by `#%%`
- `tutorials/` — notebooks walking through the package (`demo.ipynb`)
- Research code that uses the package (not part of it): `job_inputs/gen_inputs.py` (settings for server jobs), `array_job/` and `test_job/` (Slurm jobs), `plot_sim_results.Rmd`
- `notes/` — personal working notes, ignored by git (not published). Put scratch notebooks here, not at the top level. It includes `main.ipynb`, the simulation notebook (runs and summaries saved to `../results/`; paths are relative to `notes/`)

## Public and Development Functions
- `rc.corr` is the public function: only the paper's methods (Bartlett kernel, `M='optimal_rate'` or a fixed int/tuple lag, `constant`, `mask`). It validates its arguments and calls `rc.corr_dev`
- `rc.corr_dev` is the development version with every option (other kernels, `true_rho`, `check`, `b_qs`, `standardize`, and all lag rules). The simulation functions call it. Put new or experimental options here, and promote them to `rc.corr` only once they are verified and in the paper
- Lag rules accepted by `rc.corr_dev` / `rc.get_se` but not user facing: `'auto'` (earlier name of `'optimal_rate'`, still used in `job_inputs/` and saved results, so keep it), the legacy `'cubic_root'` and `'linear'`, and `'plugin'` (still under development and not verified). Document them only in development docstrings and here, not in the README or public examples
- `get_se`, `get_est_exp_ls`, `est_exp_X0Yh`, `kernel`, `kernel_weights`, `plugin_bandwidth` and `canonical_se_method` are development building blocks; their docstrings say so

## Code Structure Rules

### One Function Per File
- Each `.py` file contains **exactly one function**; the filename matches the function name (e.g. `gen_mask.py` contains `def gen_mask(...)`)
- Group functions into subpackages by topic. Avoid classes unless unavoidable
- When asked to create a new function, suggest which subpackage it belongs in

### Function Structure Template

All functions follow this docstring structure:

```python
def function_name(arg1: type1, arg2: type2, optional_arg: type3 = default_value) -> return_type:
    """
    Brief one-line description of what the function does.

    Extended description if needed, explaining the function's purpose
    and behavior in more detail.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    arg1 : type1
        Description of arg1.
    arg2 : type2
        Description of arg2.
    optional_arg : type3, optional (default=default_value)
        Description of optional_arg.

    ---------------------------------------------------------------------------
    OUTPUT:
    return_var : return_type
        Description of what is returned.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    # Function implementation
    return return_var
```

### Key Conventions
- **Use Python type hints** in function signatures (arguments and return type)
- Docstring section separators are exactly 75 dashes
- Section headers: `ARGUMENTS:`, `OUTPUT:`, `AUTHORS:`
- Types repeated in the docstring as `arg_name : type`; optional arguments as `optional (default=value)`; every returned value listed separately under OUTPUT
- Refer to other package functions in docstrings as `rc.<function>`

### Import Convention
- Import the package as `import rfcorr as rc`
- Inside the package, call other functions as `rc.<function>` (with `import rfcorr as rc` at the top of the file), never by relative import of the function

## Workflow

### When creating or modifying a function
1. Search the package for existing functions that already do the job, and reuse them
2. Write or modify the function in `rfcorr/<subpackage>/<function>.py`, following the template above
3. Add the import to `rfcorr/<subpackage>/__init__.py` (a new subpackage also needs a line in `rfcorr/__init__.py`)
4. Write tests in `tests/<subpackage>/test_<function>.py` and run them: `python -m pytest tests/<subpackage>/test_<function>.py` (each test file also runs directly with `python <file>`)
5. Write examples in `examples/<subpackage>/ex_<function>.py` (about three simple examples separated by `#%%`) and run the file. Do not skip running it
6. Run the full suite: `python -m pytest tests`
- When modifying a function, keep existing tests and examples working unless told otherwise, and update them if the behavior changes on purpose
- Results in `results/` and `job_inputs/` were produced with this code. Changes to estimation or simulation functions must not change their numerical output unless that is the intent; say so explicitly when it is

### Setup
- Install in editable mode from the repository root: `uv pip install -e ".[jobs,dev]"` (or `pip install -r requirements.txt` for the pinned server environment)
- `jobs` adds joblib and openpyxl for the server scripts; `dev` adds pytest
- Do not git commit unless asked
