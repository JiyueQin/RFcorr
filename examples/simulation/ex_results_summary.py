import pandas as pd
import rfcorr as rc

method_df = rc.make_method_df(method=["general", "iid"])
raw = rc.sim_single(method_df, rho=0.4, m_sim=10, nx=25, gen_method="gaussian_filter",
                    fwhm=2, seed=0, return_summary=False)

#%% Example 1: one summary row per method
print(rc.results_summary(raw, group=["method_id"])[["method", "se_mean", "cover_rate"]])

#%% Example 2: a single summary row over all rows
print(rc.results_summary(raw))

#%% Example 3: a hand-made results table
results = pd.DataFrame({
    "bias": [0.1, -0.1], "est": [0.5, 0.3], "se": [0.1, 0.3], "cover": [1, 0],
    "cover_raw": [1, 1], "reject": [1, 1], "reject_raw": [0, 1], "runtime": [1.0, 3.0],
})
print(rc.results_summary(results))
