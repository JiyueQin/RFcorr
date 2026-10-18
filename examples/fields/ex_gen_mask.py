import rfcorr as rc

#%% Example 1: the ellipse inscribed in a 20 x 30 grid
mask = rc.gen_mask((20, 30))
print(mask.shape, mask.sum())

#%% Example 2: a narrower ellipse, semi-axes (1, 0.5) of half the sides
mask = rc.gen_mask((11, 11), axes=(1, 0.5))
print("\n".join("".join("#" if v else "." for v in row) for row in mask))

#%% Example 3: a 3D ellipsoid, used as a mask when estimating a correlation
X, Y = rc.gen_corr_field(rho=0.5, dim=(15, 15, 15), fwhm=2, seed=0)
mask = rc.gen_mask(X.shape, axes=0.8)
rho_est, se, *_ = rc.corr(X, Y, mask=mask)
print(mask.sum(), round(rho_est, 3), round(se, 3))
