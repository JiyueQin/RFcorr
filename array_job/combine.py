"""Collect the checkpoints written by sim.py into one results file.

Run any time: python combine.py. A checkpoint is used if it matches the current setting_df and
method_df. If every row of setting_df has one, the results are saved to sim.result. If some rows
are missing or out of date (e.g. tasks still running, or killed at the time limit), the finished
rows are saved to sim_partial.result instead, and the problem rows are listed with the tasks that
run them, so those tasks can be resubmitted. Nothing is saved if the checkpoints mix different
m_sim values, since their results would not be comparable.
"""
import os
import sys

import joblib
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
setting_df = joblib.load(os.path.join(here, '../job_inputs/setting_df'))
method_df = joblib.load(os.path.join(here, '../job_inputs/method_df'))
ckpt_dir = os.path.join(here, 'checkpoints')
# must match N_TASKS in job.submit; used only to list which tasks to resubmit
n_tasks = 50

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

if len(m_sims) > 1:
    sys.exit(f'checkpoints use different m_sim values {sorted(m_sims)}; nothing saved')
if not results:
    sys.exit('no finished settings; nothing saved')

result = pd.concat(results, ignore_index=True)
summary = f'{len(results)} of {len(setting_df)} settings, m_sim = {m_sims.pop()}, {len(result)} rows'
if missing or out_of_date:
    # setting position r in setting_df is run by task r mod n_tasks (see sim.py)
    pos = setting_df.index.get_indexer(missing + out_of_date)
    tasks = sorted(set(int(p) % n_tasks for p in pos))
    print(f'missing settings: {missing}')
    print(f'out-of-date settings: {out_of_date}')
    print(f'tasks to resubmit: {",".join(map(str, tasks))}')
    joblib.dump(result, os.path.join(here, 'sim_partial.result'))
    print(f'saved sim_partial.result: {summary}')
else:
    joblib.dump(result, os.path.join(here, 'sim.result'))
    print(f'saved sim.result: {summary}')
