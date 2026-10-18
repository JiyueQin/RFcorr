import numpy as np
import rfcorr as rc

#%% Example 1: lag-wise mean of products in 1D
X = np.array([1.0, 2.0, 3.0])
Y = np.array([1.0, 0.0, -1.0])
print(rc.est_exp_X0Yh(X, Y))

#%% Example 2: return the sums, pair counts, and lags too
S, C, E, lag_ls = rc.est_exp_X0Yh(X, Y, return_counts_lags=True)
print(S, C, lag_ls[0])

#%% Example 3: lag-1 autocorrelation of a smooth 2D field
Z = rc.gen_field(dim=(80, 80), method="gaussian_filter", fwhm=3, seed=0)
Z = (Z - Z.mean()) / Z.std()
E = rc.est_exp_X0Yh(Z, Z)
print(round(E[80, 79], 3), round(rc.gaussian_filter_acf((1, 0), fwhm=3), 3))
