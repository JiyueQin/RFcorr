# NOTE: the plug-in rule is still under development and has not been verified.
import numpy as np
import rfcorr as rc

#%% Example 1: an AR(1)-like 1D series with r = 0.6 (Andrews 1991)
n, r = 1000, 0.6
h = np.arange(-(n - 1), n)
cov_ls = [r ** np.abs(h)] + [0] * 8
print(rc.plugin_bandwidth(cov_ls, np.array([1, 0, 0]), n, (n,)))

#%% Example 2: smoother 2D fields get a larger lag (through rc.corr_dev with M='plugin')
for fwhm in [1, 2, 4]:
    X, Y = rc.gen_corr_field(0, (60, 60), fwhm=fwhm, seed=0)
    print(fwhm, rc.corr_dev(X, Y, M="plugin")[4])

#%% Example 3: a larger scaling constant gives a larger lag
X, Y = rc.gen_corr_field(0, (60, 60), fwhm=3, seed=0)
print(rc.corr_dev(X, Y, M="plugin", constant=1)[4],
      rc.corr_dev(X, Y, M="plugin", constant=2)[4])
