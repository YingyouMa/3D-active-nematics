import os
import glob
from pathlib import Path
import subprocess
import shlex
import re

import pandas as pd
import numpy as np

ROOT = str(Path(__file__).resolve().parent.parent)
import sys
sys.path.append(ROOT)
from pathname import *

script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
os.chdir('..')

DENSITY = 0.7
IF_COVER = False

SIZE = 128

parameters = pd.read_csv("../parameters.csv")
par_list = []
tobe_list = glob.glob( f'../data/density_{DENSITY:0.2f}/*/*/*/end.txt' )

print('Finished trials: ')
for item in tobe_list:

    stiffness, activity, name = re.findall(r"\d*\.*\d+", item)[1:4]
    stiffness = int(stiffness)
    activity  = float(activity)
    name      = int(name)

    with open(item, 'r') as f:
        end = float(f.readline().strip())

    find_parameter = (parameters['stiffness'] == stiffness) & (parameters['activity'] == activity)
    max_time = parameters.loc[ find_parameter, 'max_time'].values[0]

    if end >= max_time:
        print(stiffness, activity, name)
        if_do = False

        if IF_COVER:
            if_do = True
        else:

            address = f"../diag/density_{DENSITY:0.2f}/stiffness_{stiffness}/activity_{activity}/{name}/"
            out_path = str(get_analysispath(DENSITY, stiffness, activity, name)) + '/defect/'

            files = glob.glob(f'{address}/{SIZE}/*')

            if len(glob.glob(out_path+'/beta.npy')) == 0:
                if_do = True
            else:
                num_old = np.load(out_path +'/beta.npy')
                if len(num_old) < len(files):
                    if_do = True

        if if_do:
            par_list.append([stiffness, activity, name])
            print('to be analyzed')



def submit(k,a,n):

    Path(f'submit/log_defect_geometry/{k}/{a}/{n}').mkdir(exist_ok=True, parents=True)

    with open(f"submit/log_defect_geometry/{k}/{a}/{n}/submit.sh", "w") as f:

        def fw(x):
            f.write(x+'\n')

        fw('#!/bin/bash')
        fw('#SBATCH -A TG-MCB090163')
        fw('#SBATCH -p shared')
        fw(f'#SBATCH -J defect_geometry_k{k}_a{a}_n{n}')
        fw(f'#SBATCH -o submit/log_defect_geometry/{k}/{a}/{n}/output.txt')
        fw(f'#SBATCH -e submit/log_defect_geometry/{k}/{a}/{n}/error.txt')
        fw('#SBATCH -N 1')
        fw('#SBATCH --ntasks-per-node 1')
        fw('#SBATCH -t 48:00:00')
        fw('#SBATCH --mem=60G')
        fw('')
        fw('module load anaconda3/2021.05')
        fw('')

        fw(f'python3 defect_geometry.py --k={k} --a={a} --name={n} --if_cover={IF_COVER}')

    
    subprocess.run(shlex.split(f'sbatch submit/log_defect_geometry/{k}/{a}/{n}/submit.sh'))


for par in par_list:
    submit(*par)