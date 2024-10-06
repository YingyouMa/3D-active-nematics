from pathlib import Path
import shlex
import subprocess
import glob
import re
from pathlib import Path
import numpy as np

from pathname import get_mainpath

DENSITY = 0.7
if_IFFT = True
if_diag = True

here = str(Path(__file__).resolve().parent)
code = str(Path(__file__).resolve().parent.parent) + '/coarse_grain.py'
address = get_mainpath(DENSITY, '*', '*', '*')

tobe_list = glob.glob( str(address) + '/dump/*.data' )
par_list = []
for item in tobe_list:
    item = re.findall(r"\d*\.*\d+", item)[1:4] #! format
    if item not in par_list:
        par_list.append(item)
        print(int(item[0]), float(item[1]), int(item[2]))

def submit(k,a,n):

    Path(f'{here}/log_coarse/{k}/{a}/{n}').mkdir(exist_ok=True, parents=True)

    with open(f"{here}/log_coarse/{k}/{a}/{n}/submit.sh", "w") as f:

        def fw(x):
            f.write(x+'\n')

        fw('#!/bin/bash')
        # fw('#SBATCH -A TG-MCB090163')
        fw('#SBATCH -p shared')
        fw(f'#SBATCH -J coarse_k{k}_a{a}_n{n}')
        fw(f'#SBATCH -o {here}/log_coarse/{k}/{a}/{n}/output.txt')
        fw(f'#SBATCH -e {here}/log_coarse/{k}/{a}/{n}/error.txt')
        fw('#SBATCH -N 1')
        fw('#SBATCH --ntasks-per-node 1')
        fw('#SBATCH -t 48:00:00')
        fw('#SBATCH --mem=40G')
        fw('')
        # fw('module load anaconda3/2021.05')
        fw('module load python/3.9.5')
        fw('')

        fw(f'python3 {code} --k={k} --a={a} --name={n} --if_IFFT={if_IFFT} --if_diag={if_diag}')

    subprocess.run(shlex.split(f'sbatch {here}/log_coarse/{k}/{a}/{n}/submit.sh'))


for (k,a,n) in par_list:
    submit(k,a,n)