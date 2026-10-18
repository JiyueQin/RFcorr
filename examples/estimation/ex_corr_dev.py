# rc.corr_dev is the development version of rc.corr: it has every option, including
# experimental and legacy ones. Users should call rc.corr.
import rfcorr as rc

X, Y = rc.gen_corr_field(rho=0.4, dim=(60, 60), fwhm=3, seed=0)

#%% Example 1: the same result as rc.corr with the paper's rule
print(rc.corr_dev(X, Y, M="optimal_rate")[:3])
print(rc.corr(X, Y)[:3])

#%% Example 2: other kernels and the legacy lag rules
for kernel in ["bartlett", "bartlett_radial", "qs"]:
    print(kernel, round(rc.corr_dev(X, Y, M="optimal_rate", kernel=kernel)[1], 4))
for M in ["cubic_root", "linear"]:
    print(M, rc.corr_dev(X, Y, M=M)[4])

#%% Example 3: plug in the true rho, with the data-driven lag
# (the 'plugin' rule is still under development and not verified)
rho_est, se, se_under_null, _, M = rc.corr_dev(X, Y, M="plugin", true_rho=0.4)
print(round(se, 4), M)
