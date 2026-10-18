import rfcorr as rc

#%% Example 1: correlation of two 2D fields and its standard errors (RFCorr and RFCorr-0)
X, Y = rc.gen_corr_field(rho=0.4, dim=(60, 60), fwhm=3, seed=0)
rho_est, se, se_under_null, runtime, M = rc.corr(X, Y)
print(round(rho_est, 3), round(se, 4), round(se_under_null, 4), M)

#%% Example 2: compare the standard error methods on the same fields
for se_method in ["general", "gauss", "indep", "iid"]:
    _, se, se0, _, _ = rc.corr(X, Y, se_method=se_method)
    print(se_method, round(se, 4), round(se0, 4))

#%% Example 3: only the points inside an elliptical mask, with a fixed truncation lag
mask = rc.gen_mask(X.shape, axes=(1, 0.7))
rho_est, se, se_under_null, _, M = rc.corr(X, Y, M=5, mask=mask)
print(round(rho_est, 3), round(se, 4), M)
