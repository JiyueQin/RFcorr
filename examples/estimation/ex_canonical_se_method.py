import rfcorr as rc

#%% Example 1: names used in the paper are returned as they are
print(rc.canonical_se_method("general"), rc.canonical_se_method("iid"))

#%% Example 2: earlier names and upper case are mapped to the paper names
print(rc.canonical_se_method("hac"), rc.canonical_se_method("HAC_GAUSSIAN"))

#%% Example 3: an unknown name raises an error listing the valid ones
try:
    rc.canonical_se_method("newey")
except ValueError as e:
    print(e)
