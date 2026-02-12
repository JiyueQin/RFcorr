import numpy as np
import os
home = "/home/jqin1"
#batch_ids = np.arange(32)
# rerun some batches
batch_ids = np.arange(19,31)
folder_ls = [home+'/batch'+ str(batch_id) for batch_id in batch_ids]
#folder_ls = ['sim_test','process_result']
batch_size = 20
for batch_id in batch_ids:
    folder = home + '/batch'+ str(batch_id)
    if not os.path.exists(folder):
        os.mkdir(folder)
    start_id = str(batch_size*batch_id)
    end_id = str(batch_size*(batch_id+1))
    with open(folder+'/sim.py', 'w') as f:
        f.write(
            "import numpy as np \n"
            "import pandas as pd \n"
            "import joblib \n"
            "setting_df = joblib.load('../inputs/setting_df') \n"
            "method_df = joblib.load('../inputs/method_df') \n"
            "from SimuInf.simulation import scb_cover_rate_multiple \n"
            "sim = scb_cover_rate_multiple(setting_df.iloc"+"["+start_id +":"+end_id+"], method_df, m_sim=1000, alpha=0.05,m_boots=1000) \n"
            "joblib.dump(sim, 'sim.result') \n")
    with open(folder+'/job.submit', 'w') as f:
        f.write(
            "#!/bin/bash\n"
            "#SBATCH --job-name='batch" + str(batch_id) + "'\n"
            "#SBATCH --output='out.%j.%N'\n"
            "#SBATCH --partition=shared\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --ntasks-per-node=1\n"
            "#SBATCH --mem=2G\n"
            "#SBATCH --account=csd864\n"
            "#SBATCH --export=ALL\n"
            "#SBATCH -t 48:00:00\n"
            "source ~/.bashrc\n"
            "conda activate nilearn\n"
            "python -u sim.py")

with open(home+"/submit_all.sh", "w") as f:
    for folder in folder_ls:
        f.write(
            "cd " + folder + "\n"
            "sbatch job.submit\n")

# get file content: with open('/home/jqin1/batch1/sim.submit') as f: content = f.read()