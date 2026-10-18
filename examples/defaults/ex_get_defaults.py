import rfcorr as rc

#%% Example 1: the standard error methods accepted by rc.corr
print(rc.get_defaults()["se_methods"])

#%% Example 2: earlier method names and what they map to
print(dict(rc.get_defaults()["se_method_aliases"]))

#%% Example 3: dependence designs that rc.make_setting_df keeps only with rho = 0
print(rc.get_defaults()["rho_zero_dep_models"])
