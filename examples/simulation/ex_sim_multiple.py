import rfcorr as rc

setting_df = rc.make_setting_df(nx=[20, 30], rho=[0, 0.4], fwhm=[2])
method_df = rc.make_method_df(method=["general", "iid"])

#%% Example 1: summary for every setting and method
summary = rc.sim_multiple(setting_df, method_df, m_sim=10, return_summary=True, verbose=False)
print(summary[["setting_id", "nx", "rho", "method", "cover_rate", "power"]])

#%% Example 2: raw results, with progress messages
raw = rc.sim_multiple(setting_df.head(2), method_df, m_sim=2)
print(raw.shape)

#%% Example 3: summarize raw results afterwards with rc.results_summary
print(rc.results_summary(raw, group=["setting_id", "method_id"])[
    ["setting_id", "method", "se_mean", "cover_rate"]])
