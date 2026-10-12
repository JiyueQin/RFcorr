import os
import sys
import time

import joblib
import pandas as pd

# paths are relative to this script, so it can be run from any folder
here = os.path.dirname(os.path.abspath(__file__))

# add parent folder to search path so we can import src
sys.path.append(os.path.join(here, ".."))
import src

setting_df = joblib.load(os.path.join(here, '../job_inputs/setting_df'))
method_df = joblib.load(os.path.join(here, '../job_inputs/method_df'))

# last 20 rows: the largest nx (300), to time the slowest settings
rows = setting_df.iloc[-20:]
m_sim = 2

# Checkpoints: each setting's result is saved to its own file as soon as it finishes. If the job
# is killed at the time limit, submit it again and the finished settings are loaded instead of
# rerun. A checkpoint also stores its setting row, method_df, and m_sim, and is rerun if any of
# them has changed (e.g. after regenerating the inputs). Delete the folder to start from scratch.
ckpt_dir = os.path.join(here, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)

results = []
for i in rows.index:
    path = os.path.join(ckpt_dir, f'setting_{i}')
    row = rows.loc[[i]]
    if os.path.exists(path):
        ckpt = joblib.load(path)
        if (ckpt['setting'].equals(row) and ckpt['method_df'].equals(method_df)
                and ckpt['m_sim'] == m_sim):
            print(f'----setting {i}: loaded from checkpoint----')
            results.append(ckpt['result'])
            continue
        print(f'----setting {i}: checkpoint is out of date, rerunning----')
    print(f'----setting {i}: running----')
    start = time.time()
    # one-row table, so setting_id in the result is the row index i
    result = src.sim_multiple(row, method_df, m_sim=m_sim, return_summary=False, verbose=False)
    # save the setting, method_df and m_sim with the result, so a rerun can check that the
    # checkpoint matches the current inputs. Write to a temporary file first, then rename it:
    # reruns ignore .tmp files, and a rename cannot be interrupted halfway, so a job killed
    # mid-write leaves either no checkpoint (the setting is rerun) or a complete one, never a
    # truncated one
    joblib.dump({'setting': row, 'method_df': method_df, 'm_sim': m_sim, 'result': result},
                path + '.tmp')
    os.replace(path + '.tmp', path)
    print(f'----setting {i}: done in {time.time() - start:.0f} s----')
    results.append(result)

result = pd.concat(results, ignore_index=True)
joblib.dump(result, os.path.join(here, 'sim.result'))
