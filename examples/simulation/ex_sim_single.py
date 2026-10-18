import rfcorr as rc

method_df = rc.make_method_df(method=["general", "iid"])

#%% Example 1: summary per method over 20 replications of a 2D setting
summary = rc.sim_single(method_df, rho=0.4, m_sim=20, nx=30, gen_method="gaussian_filter",
                        fwhm=2, seed=0)
print(summary[["method", "bias_mean", "est_sd", "se_mean", "cover_rate", "power"]])

#%% Example 2: raw results, one row per replication and method
raw = rc.sim_single(method_df, rho=0.0, m_sim=3, nx=30, gen_method="gaussian_filter",
                    fwhm=2, seed=0, return_summary=False)
print(raw[["sim_id", "method", "est", "se", "p", "reject"]])

#%% Example 3: chi^2_1 fields in an elliptical mask
summary = rc.sim_single(method_df, rho=0.4, m_sim=10, nx=30, gen_method="gaussian_filter",
                        fwhm=2, trans="square", mask="ellipse", mask_axes=(1, 0.8), seed=0)
print(summary[["method", "n_obs", "cover_rate"]])
