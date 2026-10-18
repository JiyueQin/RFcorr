import numpy as np
import rfcorr as rc

#%% Example 1: p-value for H0: rho = 0 on the Fisher z scale
print(rc.pvalue(0.2, 0.1))

#%% Example 2: the plain Wald test
print(rc.pvalue(0.2, 0.1, fisher_trans=False))

#%% Example 3: test with the null standard error from rc.corr, for several fields
X, Y = rc.gen_corr_field(rho=0.0, dim=(50, 50), fwhm=3, seed=0)
rho_est, _, se_under_null, _, _ = rc.corr(X, Y)
print(round(rho_est, 3), round(rc.pvalue(rho_est, se_under_null), 3))
print(rc.pvalue(np.array([0.05, 0.1, 0.3]), np.full(3, 0.1)).round(4))
