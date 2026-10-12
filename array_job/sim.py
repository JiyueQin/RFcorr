"""Run one task's share of the settings, with a checkpoint per setting.

Usage: python sim.py TASK_ID N_TASKS (both default to 0 and 1, i.e. all settings in one job).
The settings are split across N_TASKS tasks (see job.submit for a Slurm job array). When all
tasks are done, run combine.py to collect the checkpoints into sim.result.
"""
import os
import sys
import time

import joblib

# paths are relative to this script, so it can be run from any folder
here = os.path.dirname(os.path.abspath(__file__))

# add parent folder to search path so we can import src
sys.path.append(os.path.join(here, ".."))
import src

setting_df = joblib.load(os.path.join(here, '../job_inputs/setting_df'))
method_df = joblib.load(os.path.join(here, '../job_inputs/method_df'))

m_sim = 1000

# This task runs every N_TASKS-th row starting at TASK_ID: iloc[start::step] takes rows start,
# start + step, start + 2*step, ... to the end. With 50 tasks, task 0 runs rows 0, 50, 100, ...,
# task 1 runs rows 1, 51, 101, ..., so row r goes to task r mod 50 and every row is run exactly
# once. Why not consecutive blocks: setting_df is ordered by nx (all nx = 25 rows first, all
# nx = 300 rows last), so blocks would give the last tasks only the largest, slowest grids.
# Taking every N_TASKS-th row gives each task a similar mix of small and large grids, and similar
# run times. The rows keep their labels from setting_df, so checkpoint names and setting_id do
# not depend on which task ran a setting.
# N_TASKS must be the same for every task, so it is passed in rather than read from Slurm
# (which would count only the tasks in the current submission).
task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
n_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rows = setting_df.iloc[task_id::n_tasks]
print(f'----task {task_id} of {n_tasks}: {len(rows)} settings----')

# Checkpoints: each setting's result is saved to its own file as soon as it finishes, named by
# its row in setting_df, so tasks never write to the same file. If a task is killed at the time
# limit, submit it again and the finished settings are loaded instead of rerun. A checkpoint also
# stores its setting row, method_df, and m_sim, and is rerun if any of them has changed (e.g.
# after regenerating the inputs). Delete the folder to start from scratch.
ckpt_dir = os.path.join(here, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)

for i in rows.index:
    path = os.path.join(ckpt_dir, f'setting_{i}')
    row = rows.loc[[i]]
    if os.path.exists(path):
        ckpt = joblib.load(path)
        if (ckpt['setting'].equals(row) and ckpt['method_df'].equals(method_df)
                and ckpt['m_sim'] == m_sim):
            print(f'----setting {i}: already done----')
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

print(f'----task {task_id} finished----')
