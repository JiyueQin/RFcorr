import numpy as np
import pandas as pd
import os
import joblib

#print('current directory in python:', os.getcwd())

# create various simulation settings
from itertools import product

#nx_ls = [100,200,500,800,1000]
#nx_ls = [50, 100,150, 200, 300, 400, 500]
#nx_ls = [600, 700, 800, 900, 1000]
#nx_ls = [50, 100,200, 300, 400, 500, 800, 1000, 1200, 1500,2000, 2500, 3000]
#nx_ls = [100]
nx_ls = [10,20,30,50,80, 100, 150, 200, 300, 400, 500]
#nx_ls = [10,20,30,50]
#nx_ls = [100, 150, 200, 300, 400, 500]
# negative estimate for newey-west may occur in practive when M is extremely too large 
#M_ls = [(3,3),(4,4), (6,6), (8,8), (10,10), (30,30), (40,40), (50,50), (60,60), (70,70)]
#M_ls = [(10,10), (30,30)]
#M_ls = ['cubic_root', 'linear']
#M_ls = ['cubic_root']
M_ls = ['auto', 'cubic_root', 'linear']
#M_ls = ['linear']
#b_qs_ls =[3,5,6,8]
b_qs_ls = [None]
#rho_ls = [0, -0.4, 0.9]
#rho_ls = [-0.4]
rho_ls = [0,-0.4, 0.4]
#len_scale_ls = [0.1, 1, 3, 5]
#len_scale_ls = [1]
len_scale_ls = [None]
#fwhm_ls = [1,2,3,4]
#fwhm_ls = [2,3,4]
fwhm_ls = [1]
gen_method_ls = ['gaussian_filter']
#gen_method_ls = ['gstools']
#n_dim_ls = [1]
n_dim_ls = [2]
#n_dim_ls = [3]
# put n last to better estimate time to request
setting_df = pd.DataFrame(product(n_dim_ls, rho_ls, len_scale_ls, M_ls,b_qs_ls, fwhm_ls, gen_method_ls,nx_ls), 
                          columns=['n_dim', 'rho', 'len_scale', 'M', 'b_qs', 'fwhm', 'gen_method', 'nx'])
print(setting_df)


# negative estimate for newey-west may occur in practive when M is extremely too large 
#M_ls = [(3,3),(4,4), (6,6), (8,8), (10,10), (30,30), (40,40), (50,50), (60,60), (70,70)]
#M_ls = [(10,10), (30,30)]
#M_ls = ['cubic_root', 'linear']
#M_ls = ['cubic_root']
M_ls = ['auto', 'cubic_root', 'linear']
#M_ls = ['linear']
#b_qs_ls =[3,5,6,8]
b_qs_ls = [None]
method_df = pd.DataFrame(product([False, True], ['bartlett'], M_ls, b_qs_ls), 
                          columns=['use_true_rho', 'method', 'M', 'b_qs'])
print(method_df)


joblib.dump(setting_df, 'setting_df')
joblib.dump(method_df, 'method_df')