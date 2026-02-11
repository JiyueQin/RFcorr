import numpy as np
import pandas as pd
import os
import joblib

#print('current directory in python:', os.getcwd())

# create various simulation settings
from itertools import product

n_ls = [10, 20, 40, 60, 80, 100, 150, 200]
fwhm_ls = [0, 2, 4, 6, 8]
shape_ls = ['ellipse','ramp']
noise_type_ls = ['gaussian','t']
noise_df_ls = [None]

# put n last to better estimate time to request
# for 50*50, the actual fwhm_noise is half its input value
setting_df = pd.DataFrame(product([50],[50],fwhm_ls, shape_ls, [1, 10], noise_type_ls, noise_df_ls, n_ls), 
columns=['w', 'h', 'fwhm_noise', 'shape', 'std', 'noise_type', 'noise_df', 'n'])
# for 100*100, the actual fwhm_noise is its input value
setting_df2 = pd.DataFrame(product([100],[100],fwhm_ls, shape_ls, [1, 10], noise_type_ls, noise_df_ls, n_ls), 
columns=['w', 'h', 'fwhm_noise', 'shape', 'std', 'noise_type', 'noise_df', 'n'])
setting_df = pd.concat([setting_df, setting_df2], ignore_index=True)
#print(setting_df)

# decide to remove obs-nonparametric since it is the same as res-nonparametric 
#method_df1 = pd.DataFrame(product(['obs', 'res'], ['nonparametric'], ['t', 'z'], ['NA']), columns=['boot_data_type', 'boot_type', 'standardize', 'multiplier'])
method_df1 = pd.DataFrame(product(['res'], ['nonparametric'], ['t', 'z'], ['NA']), columns=['boot_data_type', 'boot_type', 'standardize', 'multiplier'])
method_df2 = pd.DataFrame(product(['res'], ['multiplier'], ['t', 'z'], ['r','g']), columns=['boot_data_type', 'boot_type', 'standardize', 'multiplier'])
method_df = pd.concat([method_df1, method_df2], ignore_index=True)

#print(method_df)

joblib.dump(setting_df, 'setting_df')
#joblib.dump(method_df, 'method_df')