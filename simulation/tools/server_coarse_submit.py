from pathlib import Path
import shlex
import subprocess
import glob
import re
import numpy as np

address = '../data/density_0.70/'

tobe_list = glob.glob( address + '/*/*/*/dump/*.mpiio.data' )
par_list = []
for item in tobe_list:
    item = re.findall(r"\d*\.*\d+", item)[1:4]
    if item not in par_list:
        par_list.append(item)

print(par_list)



def submit(k,a,n):

    Path(f'coarse_log/{k}/{a}/{n}').mkdir(exist_ok=True, parents=True)

    with open(f"coarse_log/{k}/{a}/{n}/submit.sh", "w") as f:

        def fw(x):
            f.write(x+'\n')

        fw('#!/bin/bash')
        fw('#SBATCH -J coarse')
        fw(f'#SBATCH -o coarse_log/{k}/{a}/{n}/output.txt')
        fw(f'#SBATCH -e coarse_log/{k}/{a}/{n}/error.txt')
        fw('#SBATCH -N 1')
        fw('#SBATCH --ntasks-per-node 1')
        fw('#SBATCH -t 48:00:00')
        fw('#SBATCH --mem=40G')
        fw('')
        fw('module load anaconda3/2021.05')
        fw('')

        fw(f'python3 server_coarse.py --k={k} --a={a} --name={n}')

    subprocess.run(shlex.split(f'sbatch coarse_log/{k}/{a}/{n}/submit.sh'))


for (k,a,n) in par_list:
    submit(k,a,n)

