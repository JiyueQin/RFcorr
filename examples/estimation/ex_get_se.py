import numpy as np
import rfcorr as rc

X, Y = rc.gen_corr_field(rho=0.4, dim=(50, 50), fwhm=3, seed=0)
X_s = (X - X.mean()) / X.std()
Y_s = (Y - Y.mean()) / Y.std()
rho_est = np.corrcoef(X.ravel(), Y.ravel())[0, 1]
dim = np.array(X.shape)

#%% Example 1: the general (RFCorr) standard error from the lag-wise estimates
est_exp_ls = rc.get_est_exp_ls("general", X_s, Y_s)
print(rc.get_se(dim, est_exp_ls, rho_est, M="auto", se_method="general"))

#%% Example 2: the same at rho = 0 (RFCorr-0), and with a fixed lag M = 3
print(rc.get_se(dim, est_exp_ls, 0, M="auto", se_method="general"))
print(rc.get_se(dim, est_exp_ls, rho_est, M=3, se_method="general"))

#%% Example 3: Gaussian method with the radial Bartlett kernel
est_exp_ls = rc.get_est_exp_ls("gauss", X_s, Y_s)
print(rc.get_se(dim, est_exp_ls, rho_est, M="auto", se_method="gauss", kernel="bartlett_radial"))
