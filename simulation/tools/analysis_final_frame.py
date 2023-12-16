import numpy as np
from pathlib import Path
import glob
import os
import re
import h5py
import argparse

# ! Remember to reshape qtensor before save in the future version


DENSITY     = 0.7
TIME_STEP   = 0.001


def main(
        address, stiffness, activity, name,
        sig=2, N_out=400, L=200
        ):
    
    coarse_path = address + "/coarse/"
    save_path 	= address +'/final/'

    Path( save_path+'/data' ).mkdir(exist_ok=True, parents=True)
    Path( coarse_path+f'/FFT' ).mkdir(exist_ok=True, parents=True)
    Path( coarse_path+f'/result_{N_out}' ).mkdir(exist_ok=True, parents=True)
    Path( address + f"/diagonal/{N_out}/" ).mkdir(exist_ok = True, parents=True)

    with open( address + '/end.txt', 'r') as f:
        end = float(f.readline().strip())
        end = int( end / TIME_STEP )

    files 	= glob.glob(coarse_path+'/FFT/*.h5py')
    frames 	= np.array([int(re.findall(r'\d+', file)[-2]) for file in files])
    
    if end not in frames:

        from Nematics3D.coarse import coarse_one_frame
        coarse_one_frame(address, stiffness, activity, name, end, N_out=N_out)

    else:

        files 	= glob.glob( coarse_path + f'/result_{N_out}/*.h5py')
        frames 	= np.array([int(re.findall(r'\d+', file)[-2]) for file in files])

        if end not in frames:

            from Nematics3D.coarse import kernal_fft, IFFT_nematics

            with h5py.File(coarse_path+'/FFT/'+str(end)+'.h5py', 'R') as f:
                F_density = f['density'][...]
                F_qtensor = f['qtensor'][...]

            Fd = kernal_fft(F_density, sig, L)
            Fq = kernal_fft(F_qtensor, sig, L)

            den, qtensor = IFFT_nematics(Fd, Fq, N_out=N_out)
            qtensor = qtensor.transpose((1,2,3,0))

            with h5py.File(coarse_path+f'/result_{N_out}/'+str(end)+'.h5py', 'w') as fw:
                fw.create_dataset('density', data=den)
                fw.create_dataset('qtensor', data=qtensor)

            del Fd, Fq, den, qtensor

        files   = glob.glob( address + f'/diagonal/{N_out}/S_*.npy')
        frames 	= np.array([int(re.findall(r'\d+', file)[-1]) for file in files])

        if end not in frames:

            from Nematics3D.coarse import diagonalizeQ

            with h5py.File(coarse_path+f'/result_{N_out}/'+str(end)+'.h5py', 'r') as fw:
                qtensor = f['qtensor'][...]

                S, n = diagonalizeQ(qtensor)

                np.save( address + f"/diagonal/{N_out}/S_{end}.npy", S )
                np.save( address + f"/diagonal/{N_out}/n_{end}.npy", n )

    S = np.load( address + f"/diagonal/{N_out}/S_{end}.npy" )
    n = np.load( address + f"/diagonal/{N_out}/n_{end}.npy" )

# Input Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int) 						        # stiffness
parser.add_argument("--a", type=float) 								# activity
parser.add_argument("--name", type=int) 							# name
parser.add_argument("--sig", default=2, type=int) 					# sigma for Gaussian filter
parser.add_argument("--N_out", default=128, type=int)				# the final grid dimensions in real space
args = parser.parse_args()

k       = args.k
a       = args.a
name    = args.name
address = f"../../data/density_{DENSITY:0.2f}/stiffness_{k}/activity_{a}/{name}/"
sig     = args.sig
N_out   = args.N_out

main(address, k, a, name)