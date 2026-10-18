import numpy as np
import rfcorr as rc

#%% Example 1: 95% interval on the Fisher z scale
print(rc.confidence_interval(0.6, 0.1))

#%% Example 2: the Wald interval, clipped to [-1, 1], and a 90% interval
print(rc.confidence_interval(0.9, 0.1, fisher_trans=False))
print(rc.confidence_interval(0.6, 0.1, alpha=0.1))

#%% Example 3: vectorized, from estimates of rc.corr
X, Y = rc.gen_corr_field(rho=0.3, dim=(50, 50), fwhm=3, seed=0)
rho_est, se, *_ = rc.corr(X, Y)
lower, upper = rc.confidence_interval(np.array([rho_est, -rho_est]), np.array([se, se]))
print(lower.round(3), upper.round(3))
