import numpy as np 
import pandas as pd 
import joblib
import sys
from itertools import product

nx_ls = [100,200,500,1000]
rho_ls = [0]
len_scale_ls = [0.1]
setting_df = pd.DataFrame(product(nx_ls, rho_ls, len_scale_ls), columns=['nx', 'rho', 'len_scale'])
print(setting_df)

sys.path.append("..")
import src

result = src.sim_multiple(setting_df, m_sim=2, return_summary = False)
joblib.dump(result, 'sim.result') 
