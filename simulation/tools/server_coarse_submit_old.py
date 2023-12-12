from pathlib import Path
import shlex
import subprocess
import glob
import re
import numpy as np

address = '../data/width_200/viscosity_10/density_0.70/length_50'

tobe_list = glob.glob( address + '/*/*/seed_*/dump/*.mpiio.data' )
par_list = []
for item in tobe_list:
    item = re.findall(r"\d*\.*\d+", item)[4:7]
    if item not in par_list:
        par_list.append(item)

print(par_list)



def submit(k,a,s):

    Path(f'coarse_log/{k}_{a}_{s}').mkdir(exist_ok=True, parents=True)

    with open(f"coarse_log/{k}_{a}_{s}/submit.sh", "w") as f:

        def fw(x):
            f.write(x+'\n')

        fw('#!/bin/bash')
        fw('#SBATCH -J coarse')
        fw(f'#SBATCH -o coarse_log/{k}_{a}_{s}/output.txt')
        fw(f'#SBATCH -e coarse_log/{k}_{a}_{s}/error.txt')
        fw('#SBATCH -N 1')
        fw('#SBATCH --ntasks-per-node 1')
        fw('#SBATCH -t 48:00:00')
        fw('#SBATCH --mem=40G')
        fw('')
        fw('module load anaconda3/2021.05')
        fw('')

        fw(f'python3 server_coarse.py --k={k} --a={a} --s={s}')

    subprocess.run(shlex.split(f'sbatch coarse_log/{k}_{a}_{s}/submit.sh'))


for (k,a,s) in par_list:
    submit(k,a,s)

