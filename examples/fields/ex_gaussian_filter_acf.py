import rfcorr as rc

#%% Example 1: lag-1 correlation of a 2D field for several smoothness levels
for fwhm in [1, 2, 3, 4]:
    print(fwhm, round(rc.gaussian_filter_acf((1, 0), fwhm=fwhm), 3))

#%% Example 2: the correlation is separable, so a diagonal lag gives the product
print(round(rc.gaussian_filter_acf((1, 1), fwhm=3), 4),
      round(rc.gaussian_filter_acf(1, fwhm=3) ** 2, 4))

#%% Example 3: compare with the sample lag-1 correlation of a simulated field
import numpy as np
Z = rc.gen_field(dim=(200, 200), method="gaussian_filter", fwhm=3, seed=0)
print(round(np.corrcoef(Z[:-1].ravel(), Z[1:].ravel())[0, 1], 3),
      round(rc.gaussian_filter_acf((1, 0), fwhm=3), 3))
