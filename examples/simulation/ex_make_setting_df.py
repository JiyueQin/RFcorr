import rfcorr as rc

#%% Example 1: cross side lengths, correlations, and smoothness
setting_df = rc.make_setting_df(nx=[25, 50], rho=[0, 0.4], fwhm=[2, 3])
print(setting_df[["nx", "rho", "fwhm"]])

#%% Example 2: Gaussian and chi^2_1 fields inside an elliptical mask
setting_df = rc.make_setting_df(nx=[30], rho=[0.4], fwhm=[2], trans=[None, "square"],
                                mask=[("ellipse", (1, 0.8))])
print(setting_df[["trans", "mask", "mask_axes"]])

#%% Example 3: a dependence design, kept only in the rho = 0 rows
setting_df = rc.make_setting_df(nx=[30], rho=[0, 0.4], fwhm=[2],
                                dep=[(None, None), ("symmetric_shift", 1)])
print(setting_df[["rho", "dep_model", "shift_lag"]])
