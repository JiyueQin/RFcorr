"""Collect the checkpoints written by sim.py into sim.result.

Run after all tasks have finished: python combine.py. Every row of setting_df must have a
checkpoint that matches the current setting_df and method_df, and all checkpoints must use the
same m_sim. Otherwise the problem rows are listed and nothing is saved, so the tasks that ran
them can be resubmitted.
"""
import os
import sys

import joblib
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
setting_df = joblib.load(os.path.join(here, '../job_inputs/setting_df'))
method_df = joblib.load(os.path.join(here, '../job_inputs/method_df'))
ckpt_dir = os.path.join(here, 'checkpoints')

results, missing, out_of_date, m_sims = [], [], [], set()
for i in setting_df.index:
    path = os.path.join(ckpt_dir, f'setting_{i}')
    if not os.path.exists(path):
        missing.append(i)
        continue
    ckpt = joblib.load(path)
    if not (ckpt['setting'].equals(setting_df.loc[[i]]) and ckpt['method_df'].equals(method_df)):
        out_of_date.append(i)
        continue
    m_sims.add(ckpt['m_sim'])
    results.append(ckpt['result'])

if missing or out_of_date or len(m_sims) > 1:
    print(f'missing settings: {missing}')
    print(f'out-of-date settings: {out_of_date}')
    print(f'm_sim values found: {sorted(m_sims)}')
    sys.exit('not all settings are done; nothing saved')

result = pd.concat(results, ignore_index=True)
joblib.dump(result, os.path.join(here, 'sim.result'))
print(f'saved sim.result: {len(setting_df)} settings, m_sim = {m_sims.pop()}, {len(result)} rows')
