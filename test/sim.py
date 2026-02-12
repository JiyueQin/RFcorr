import numpy as np 
import pandas as pd 
import joblib
import sys

setting_df = joblib.load('../job_inputs/setting_df')
method_df = joblib.load('../job_inputs/method_df')

# add parent folder to seatch path so we can import src
sys.path.append("..")
import src

result = src.sim_multiple(setting_df.iloc[0:20], method_df, m_sim=2, return_summary = False)
joblib.dump(result, 'sim.result') 
