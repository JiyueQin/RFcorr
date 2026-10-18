import rfcorr as rc

#%% Example 1: all four standard error methods with the 'auto' lag
print(rc.make_method_df())

#%% Example 2: two lag rules for two methods
print(rc.make_method_df(method=["general", "gauss"], M=["auto", "plugin"]))

#%% Example 3: plug in the true rho, with the QS kernel
print(rc.make_method_df(method=["general"], kernel=["qs"], b_qs=[5], use_true_rho=[True]))
