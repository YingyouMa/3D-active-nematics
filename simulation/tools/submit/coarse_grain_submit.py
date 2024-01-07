from pathlib import Path
import shlex
import subprocess
import glob
import re
import numpy as np

DENSITY = 0.7
if_IFFT = True
if_diag = True

address = f'../../data/density_{DENSITY:0.2f}/'

tobe_list = glob.glob( address + '/*/*/*/dump/*.data' )
par_list = []
for item in tobe_list:
    item = re.findall(r"\d*\.*\d+", item)[1:4]
    if item not in par_list:
        par_list.append(item)

print(par_list)

def submit(k,a,n):

    Path(f'log_coarse/{k}/{a}/{n}').mkdir(exist_ok=True, parents=True)

    with open(f"log_coarse/{k}/{a}/{n}/submit.sh", "w") as f:

        def fw(x):
            f.write(x+'\n')

        fw('#!/bin/bash')
        fw('#SBATCH -A TG-MCB090163')
        fw('#SBATCH -p shared')
        fw(f'#SBATCH -J coarse_k{k}_a{a}_n{n}')
        fw(f'#SBATCH -o log_coarse/{k}/{a}/{n}/output.txt')
        fw(f'#SBATCH -e log_coarse/{k}/{a}/{n}/error.txt')
        fw('#SBATCH -N 1')
        fw('#SBATCH --ntasks-per-node 1')
        fw('#SBATCH -t 48:00:00')
        fw('#SBATCH --mem=40G')
        fw('')
        fw('module load anaconda3/2021.05')
        fw('')

        fw(f'python3 ../coarse_grain.py --k={k} --a={a} --name={n} --if_IFFT={if_IFFT} --if_diag={if_diag}')

    subprocess.run(shlex.split(f'sbatch log_coarse/{k}/{a}/{n}/submit.sh'))


for (k,a,n) in par_list:
    submit(k,a,n)