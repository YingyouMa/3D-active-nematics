import glob
import re
from pathlib import Path
import os
import subprocess
import shlex

import pandas as pd

script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
os.chdir('..')

DENSITY     = 0.7
TIME_STEP   = 0.001

N_OUT       = 400
IF_COVER    = True

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

        end_directory = glob.glob(
        f'../data/density_{DENSITY:0.2f}/stiffness_{stiffness}/activity_{activity}/{name}/analysis/final/endframe*_Nout*'
                                    )
        
        if len(end_directory)>0:
            analyzed = [re.findall(r"\d*\.*\d+", item)[-2:] for item in end_directory]
            if [str(int(end/TIME_STEP)), str(N_OUT)] in analyzed:
                if IF_COVER == False:
                    print(stiffness, activity, name)
                else:
                    par_list.append([stiffness, activity, name])
                    print(stiffness, activity, name, 'to be analyzed')
            else:
                par_list.append([stiffness, activity, name])
                print(stiffness, activity, name, 'to be analyzed')
        else:
            par_list.append([stiffness, activity, name])
            print(stiffness, activity, name, 'to be analyzed')

def submit(k,a,n):

    Path(f'submit/log_analysis_final/{k}/{a}/{n}').mkdir(exist_ok=True, parents=True)

    with open(f"submit/log_analysis_final/{k}/{a}/{n}/submit.sh", "w") as f:

        def fw(x):
            f.write(x+'\n')

        fw('#!/bin/bash')
        fw('#SBATCH -A TG-MCB090163')
        fw('#SBATCH -p shared')
        fw(f'#SBATCH -J analysis_final_k{k}_a{a}_n{n}')
        fw(f'#SBATCH -o submit/log_analysis_final/{k}/{a}/{n}/output.txt')
        fw(f'#SBATCH -e submit/log_analysis_final/{k}/{a}/{n}/error.txt')
        fw('#SBATCH -N 1')
        fw('#SBATCH --ntasks-per-node 1')
        fw('#SBATCH -t 48:00:00')
        fw('#SBATCH --mem=60G')
        fw('')
        fw('module load anaconda3/2021.05')
        fw('')

        fw(f'python3 analysis_final_frame.py --k={k} --a={a} --name={n}')

    
    subprocess.run(shlex.split(f'sbatch submit/log_analysis_final/{k}/{a}/{n}/submit.sh'))


for par in par_list:
    submit(*par)




