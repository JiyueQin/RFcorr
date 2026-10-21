"""Create the simulation settings and methods for server jobs.

Saves `setting_df` and `method_df` (joblib) next to this script. Jobs load them and run
`rc.sim_multiple` on a subset of the rows (see test_job/sim.py, or array_job/ to split the rows across a job array). The tables are built with
`rc.make_setting_df` and `rc.make_method_df`, the same functions used in the simulation notebook
(notes/main.ipynb), so a server run and a notebook run with the same values give the same settings.
"""
import os

import joblib

import rfcorr as rc

# outputs are saved next to this script, wherever it is run from
here = os.path.dirname(os.path.abspath(__file__))

# ---- simulation settings: one row per combination ----
# side lengths roughly evenly spaced on the log scale: each step multiplies nx by about sqrt(2),
# so N (about nx^2) roughly doubles. Errors shrink with ratios of N (Monte Carlo error like
# N^(-1/2), Bartlett bias more slowly), so equal ratios are equally informative, and the points
# spread evenly on the log-N axis of the plots. Large sizes are included because the bias
# decays slowly.
# Add new sizes only at the END of the list: setting_df is ordered by nx, so appending keeps the
# row numbers of the existing settings, and their checkpoints in array_job/checkpoints/ are reused
# (the first run used [25, ..., 300]; 420 and 600 were appended later). Inserting a size in the
# middle renumbers the rows after it, and those settings would be rerun.
nx_ls = [25, 35, 50, 70, 100, 140, 200, 300, 420, 600]
rho_ls = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, -0.1, -0.2, -0.4, -0.6, -0.8, -0.9]
fwhm_ls = [1, 2, 3, 4]
n_dim_ls = [2]
len_scale_ls = [None]
gen_method_ls = ['gaussian_filter']
distribution_ls = ['gaussian']
noise_df_ls = [1]
# None = Gaussian fields, 'square' = chi^2_1 fields
trans_ls = [None, 'square']
# (mask, mask_axes): no mask, or a fixed ellipse with semi-axes (1, 0.8) as fractions of half the
# side lengths
mask_ls = [('ellipse', (1, 0.8))]
# the field entering only Y has fwhm / fwhm_ratio
fwhm_ratio_ls = [2]
# (dep_model, shift_lag): (None, None) for X and Y dependent only through rho; ('shift', 1) or
# ('symmetric_shift', 1) for uncorrelated but dependent fields, kept only with rho = 0
#dep_ls = [('shift', 1), ('symmetric_shift', 1)]

dep_ls = [(None, None), ('symmetric_shift', 1)]

setting_df = rc.make_setting_df(nx=nx_ls, rho=rho_ls, fwhm=fwhm_ls, n_dim=n_dim_ls,
                                len_scale=len_scale_ls, gen_method=gen_method_ls,
                                distribution=distribution_ls, noise_df=noise_df_ls,
                                trans=trans_ls, mask=mask_ls, fwhm_ratio=fwhm_ratio_ls,
                                dep=dep_ls)
print(setting_df)

# ---- standard error methods: one row per combination ----
method_ls = ['iid', 'indep', 'general', 'gauss']
#M_ls = ['auto', 'plugin']
M_ls = ['auto']
kernel_ls = ['bartlett']
use_true_rho_ls = [False]
b_qs_ls = [None]
constant_ls = [1]

method_df = rc.make_method_df(method=method_ls, M=M_ls, kernel=kernel_ls,
                              use_true_rho=use_true_rho_ls, b_qs=b_qs_ls, constant=constant_ls)
print(method_df)

joblib.dump(setting_df, os.path.join(here, 'setting_df'))
joblib.dump(method_df, os.path.join(here, 'method_df'))
