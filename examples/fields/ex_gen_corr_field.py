import numpy as np
import rfcorr as rc

#%% Example 1: two smooth 2D fields with correlation 0.6
X, Y = rc.gen_corr_field(rho=0.6, dim=(100, 100), fwhm=3, seed=0)
print(X.shape, round(np.corrcoef(X.ravel(), Y.ravel())[0, 1], 3))

#%% Example 2: Y rougher than X (fwhm_ratio=2) and chi^2_1 fields (trans='square')
X, Y = rc.gen_corr_field(rho=0.4, dim=(60, 60), fwhm=4, fwhm_ratio=2, trans="square", seed=1)
print(round(X.mean(), 2), round(Y.mean(), 2))

#%% Example 3: uncorrelated but dependent fields (symmetric shift design, rho = 0)
X, Y = rc.gen_corr_field(rho=0, dim=(80, 80), fwhm=3, dep_model="symmetric_shift",
                         shift_lag=1, seed=2)
print("lag 0:", round(np.corrcoef(X.ravel(), Y.ravel())[0, 1], 3),
      "lag 1:", round(np.corrcoef(X[:-1].ravel(), Y[1:].ravel())[0, 1], 3))
