import glob
import re
from pathlib import Path
import json
import argparse

import numpy as np

from Nematics3D.disclination import defect_detect

DENSITY = 0.7

def main(address, length_min=200, time_step=0.001, if_cover=False):

    save_path = address + f'/analysis/time_scale/'
    num_path = address + f'/analysis/time_scale/{length_min}/' 
    Path( num_path ).mkdir(exist_ok=True, parents=True)

    files = glob.glob(f'{address}/diagonal/128/n_*.npy')
    frames = np.array([int(re.findall(r'\d+', file)[-1]) for file in files])
    frames = np.sort(frames)
    defect_num = np.zeros(len(frames))

    if if_cover == False and len(glob.glob(num_path + '/num.npy')) != 0:
        print('Found previous data of # defects')
        defect_num_old = np.load(num_path+'/num.npy')
        defect_num[:len(defect_num_old)] = defect_num_old
    else:
        defect_num_old = []

    for i in range(len(defect_num_old), len(frames)):
        frame = frames[i]
        n = np.load(f'{address}/diagonal/128/n_{frame}.npy')
        defect_indices = defect_detect(n, boundary=True)
        print(frame, len(defect_indices))
        defect_num[i] = len(defect_indices)

    np.save(num_path + '/num.npy', defect_num)

    frame_loop = frames[ np.argmax(defect_num>length_min) ]

    time_loop = int( frame_loop * time_step )
    print(f'time_loop = {time_loop}')

    out_path = address + '/analysis/k_instability/'
    frames = np.load(out_path+'/frames.npy')
    result = np.load(out_path+'/result.npy')
    result = result[:, :, 0, 1]  # Biggest Fourier coefficient
    result = result[:, 1] + result[:, 2] # Only Q_xy + Q_xz
    time_nonlinear = int( frames[np.argmax(result)] * time_step )
    print(f'time_nonlinear = {time_nonlinear}')

    result = {"time_loop":      time_loop,
              "time_nonlinear": time_nonlinear,
              "length_min":     length_min}
    
    with open(save_path+"/result.json", "w") as f:
        json.dump(result, f, indent=4)


# Input Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int) 						        # stiffness
parser.add_argument("--a", type=float) 								# activity
parser.add_argument("--name", type=int) 							# name
parser.add_argument("--length_min", type=int, default=200)
parser.add_argument("--if_cover", type=bool, default=False) 
args = parser.parse_args()

if args.k == None:
    print('No parameters input. Use the default parameters provided by the program instead.')
    k                   = 100
    a                   = 3.5
    name                = 1
    address             = f"../data/density_{DENSITY:0.2f}/stiffness_{k}/activity_{a}/{name}/"
    length_min          = 200
    if_cover            = False
else:
    k                   = args.k
    a                   = args.a
    name                = args.name 
    address             = f"../data/density_{DENSITY:0.2f}/stiffness_{k}/activity_{a}/{name}/"
    length_min          = args.length_min
    if_cover            = args.if_cover

main(address, length_min=length_min, if_cover=if_cover)