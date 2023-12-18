import numpy as np
from pathlib import Path
import glob
import os
import re
import h5py
import argparse

from scipy.stats import mode
import matplotlib.pyplot as plt

from Nematics3D.field import calc_lp_n, calc_lp_S, exp_decay

# ! Remember to reshape qtensor before save in the future version


DENSITY     = 0.7
TIME_STEP   = 0.001
WIDTH       = 200

plt.rcParams.update({'font.size': 30})


def main(
        address, stiffness, activity, name,
        sig=2, N_out=400, L=200, 
        lp_N_ratio=8, lp_head_ratio=18
        ):
    
    coarse_path = address + "/coarse/"
    save_path 	= address +'/final/'
    final_path  = address + "/analysis/final/"

    Path( save_path+'/data' ).mkdir(exist_ok=True, parents=True)
    Path( coarse_path+f'/FFT' ).mkdir(exist_ok=True, parents=True)
    Path( coarse_path+f'/result_{N_out}' ).mkdir(exist_ok=True, parents=True)
    Path( address + f"/diagonal/{N_out}/" ).mkdir(exist_ok = True, parents=True)
    Path( final_path ).mkdir(exist_ok = True, parents=True)

    with open( address + '/end.txt', 'r') as f:
        end = float(f.readline().strip())
        end_frame = int( end / TIME_STEP )

    files 	= glob.glob(coarse_path+'/FFT/*.h5py')
    frames 	= np.array([int(re.findall(r'\d+', file)[-2]) for file in files])
    
    if end_frame not in frames:

        from Nematics3D.coarse import coarse_one_frame
        coarse_one_frame(address, stiffness, activity, name, end_frame, N_out=N_out)

    else:

        files 	= glob.glob( coarse_path + f'/result_{N_out}/*.h5py')
        frames 	= np.array([int(re.findall(r'\d+', file)[-2]) for file in files])

        if end_frame not in frames:

            from Nematics3D.coarse import kernal_fft, IFFT_nematics

            with h5py.File(coarse_path+'/FFT/'+str(end_frame)+'.h5py', 'R') as f:
                F_density = f['density'][...]
                F_qtensor = f['qtensor'][...]

            Fd = kernal_fft(F_density, sig, L)
            Fq = kernal_fft(F_qtensor, sig, L)

            den, qtensor = IFFT_nematics(Fd, Fq, N_out=N_out)
            qtensor = qtensor.transpose((1,2,3,0))

            with h5py.File(coarse_path+f'/result_{N_out}/'+str(end_frame)+'.h5py', 'w') as fw:
                fw.create_dataset('density', data=den)
                fw.create_dataset('qtensor', data=qtensor)

            del Fd, Fq, den, qtensor

        files   = glob.glob( address + f'/diagonal/{N_out}/S_*.npy')
        frames 	= np.array([int(re.findall(r'\d+', file)[-1]) for file in files])

        if end_frame not in frames:

            from Nematics3D.field import diagonalizeQ

            with h5py.File(coarse_path+f'/result_{N_out}/'+str(end_frame)+'.h5py', 'r') as f:
                qtensor = f['qtensor'][...]

                S, n = diagonalizeQ(qtensor)

                np.save( address + f"/diagonal/{N_out}/S_{end_frame}.npy", S )
                np.save( address + f"/diagonal/{N_out}/n_{end_frame}.npy", n )


    S = np.load( address + f"/diagonal/{N_out}/S_{end_frame}.npy" )
    n = np.load( address + f"/diagonal/{N_out}/n_{end_frame}.npy" )

    S_mean  = S.mean()
    S_mode  = mode(np.round(S,2).reshape(-1))
    plt.figure((1920, 1080))
    plt.hist(S.reshape(-1), bins=np.sqrt(np.size(S)))
    plt.title('Distribution of S')
    plt.savefig( save_path + f'/end_frame_{end_frame}/Nout_{N_out}/distS.jpg' )
    plt.close()

    lp_max_N = int( N_out / lp_N_ratio )
    lp_head  = int( N_out / lp_head_ratio )

    lp_S    = calc_lp_S(S, max_N=lp_max_N, width=WIDTH, head_skip=lp_head)
    lp_n    = calc_lp_n(n, max_N=lp_max_N, width=WIDTH, head_skip=lp_head)






    

# Input Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int) 						        # stiffness
parser.add_argument("--a", type=float) 								# activity
parser.add_argument("--name", type=int) 							# name
parser.add_argument("--sig", default=2, type=int) 					# sigma for Gaussian filter
parser.add_argument("--N_out", default=400, type=int)               # the final grid dimensions in real space
parser.add_argument("--lp_N_ratio", default=5, type=int)			
parser.add_argument("--lp_head_ratio", default=18, type=int)
args = parser.parse_args()

if args.k == None:
    print('No parameters input. Use the default parameters provided by the program instead.')

else:
    k               = args.k
    a               = args.a
    name            = args.name
    address         = f"../../data/density_{DENSITY:0.2f}/stiffness_{k}/activity_{a}/{name}/"
    sig             = args.sig
    N_out           = args.N_out
    lp_N_ratio      = args.lp_N_ratio
    lp_head_ratio   = args.lp_head_ratio

main(address, k, a, name)