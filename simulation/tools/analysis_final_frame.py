import numpy as np
from pathlib import Path
import glob
import os

import sys
sys.path
sys.path.append('../Nematics3D')
sys.path


def main(
        stiffness, activity, name,
        sig=2, N_out=400, L=200
        )
    
    address 	= f"../data/density_{DENSITY:0.2f}/stiffness_{stiffness}/activity_{activity}/{name}/"
    if len(glob.glob(address)) == 0:
        address = "../" + address 
    coarse_path = address + "/coarse/"
    save_path 	= address +'/final/'

    Path( save_path+'/data' ).mkdir(exist_ok=True, parents=True)
    Path( coarse_path+f'/FFT' ).mkdir(exist_ok=True)
    Path( coarse_path+f'/result_{N_out}' ).mkdir(exist_ok=True)

    with open( address + '/end.txt', 'r') as f:
        end = int(f.readline())

    files 	= glob.glob(coarse_path+'/FFT/*.h5py')
    frames 	= np.array([int(re.findall(r'\d+', file)[-2]) for file in files])
    
    if end not in frames:

        from coarse import read_params, read_pos, count_monomer, truncate_rfft_coefficients, IFFT_nematics, kernal_fft

        


