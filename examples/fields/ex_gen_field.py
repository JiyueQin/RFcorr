import rfcorr as rc

#%% Example 1: a smooth 2D Gaussian field from i.i.d. noise and a Gaussian kernel
X = rc.gen_field(dim=(50, 50), method="gaussian_filter", fwhm=4, seed=0)
print(X.shape, round(X.mean(), 3), round(X.std(), 3))

#%% Example 2: a 1D field with heavy-tailed noise (t with 5 df), and its coordinates
X, coords = rc.gen_field(dim=100, method="gaussian_filter", fwhm=3, distribution="t",
                         noise_df=5, seed=1, return_coordinate=True)
print(X.shape, coords[0][:5])

#%% Example 3: a chi^2_1 field (squared Gaussian field) and an i.i.d. field
X_sq = rc.gen_field(dim=(30, 30), method="gaussian_filter", fwhm=2, trans="square", seed=2)
X_iid = rc.gen_field(dim=(30, 30), method="iid", var=2.0, seed=2)
print(round(X_sq.min(), 3) >= 0, round(X_iid.var(), 2))
