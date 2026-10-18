import numpy as np
import rfcorr as rc

rng = np.random.default_rng(0)
X, Y = rng.normal(size=(6, 7)), rng.normal(size=(6, 7))

#%% Example 1: the 9 lag-wise arrays for the general method
est = rc.get_est_exp_ls("general", X, Y)
print(len(est), est[0].shape)

#%% Example 2: Gaussian method, with and without imposing rho = 0 at lag 0
full = rc.get_est_exp_ls("gauss", X, Y)
null = rc.get_est_exp_ls("gauss", X, Y, impose_null=True)
print(round(full[0][5, 6], 4), round(null[0][5, 6], 4))

#%% Example 3: with a mask, X and Y set to 0 outside it
mask = rc.gen_mask(X.shape)
est = rc.get_est_exp_ls("indep", np.where(mask, X, 0), np.where(mask, Y, 0), mask=mask)
print(np.isnan(est[0]).sum(), "lags with no valid pair")
