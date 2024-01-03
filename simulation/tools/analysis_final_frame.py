import numpy as np
from pathlib import Path
import glob
import os
import re
import h5py
import argparse
import json

from scipy.stats import mode
import matplotlib.pyplot as plt
'''
import sys
sys.path
sys.path.append(r'E:\Program\GitHub\3D-active-nematics\simulation')
sys.path
'''
from Nematics3D.field import calc_lp_S, exp_decay
from Nematics3D.elastic import get_deform_Q


DENSITY     = 0.7
TIME_STEP   = 0.001
WIDTH       = 200

plt.rcParams.update({'font.size': 30})


def main(
        address, stiffness, activity, name,
        sig=2, N_out=400, L=200, 
        lp_max_init_ratio=8, lp_skip_init_ratio=40
        ):
    
    global lp_popt_S, S_cor_local
    
    coarse_path = address + "/coarse/"

    Path( coarse_path+'/FFT' ).mkdir(exist_ok=True, parents=True)
    Path( coarse_path+f'/result_{N_out}' ).mkdir(exist_ok=True, parents=True)
    Path( address + f"/diagonal/{N_out}/" ).mkdir(exist_ok = True, parents=True)

    with open( address + '/end.txt', 'r') as f:
        end = float(f.readline().strip())
        end_frame = int( end / TIME_STEP )
        
    final_path  = address + f"/analysis/final/endframe{end_frame}_Nout{N_out}/"
    Path( final_path ).mkdir(exist_ok = True, parents=True)

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

            with h5py.File(coarse_path+'/FFT/'+str(end_frame)+'.h5py', 'r') as f:
                F_density = f['density'][...]
                F_qtensor = f['qtensor'][...]

            Fd = kernal_fft(F_density, sig, L)
            Fq = kernal_fft(F_qtensor, sig, L)

            den, qtensor = IFFT_nematics(Fd, Fq, N_out=N_out)
            qtensor = qtensor.transpose((1,2,3,0))

            with h5py.File(coarse_path+f'/result_{N_out}/'+str(end_frame)+'.h5py', 'w') as fw:
                fw.create_dataset('density', data=den)
                fw.create_dataset('qtensor', data=qtensor)
                fw.create_dataset('sigma', data=sig)

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
    S_mode  = mode(np.round(S,2).reshape(-1))[0][0]
    plt.figure(figsize=(16, 9))
    plt.hist(S.reshape(-1), bins=int(np.sqrt(np.size(S)/2)))
    plt.title('Distribution of S')
    plt.savefig( final_path +'/distS.jpg' )
    plt.close()
    
    lp_max = int( N_out / lp_max_init_ratio )
    lp_skip  = int( N_out / lp_skip_init_ratio )
    
    output_S = calc_lp_S(S, max_init=lp_max, width=WIDTH, skip_init=lp_skip) 
    lp_popt_S, S_cor_local, skip_S = output_S
    
    plt.figure(figsize=(16,9))
    plt.plot(S_cor_local[:skip_S,0], S_cor_local[:skip_S,1], 
             'o', color='red', label='experiment (skipped)')
    plt.plot(S_cor_local[skip_S:,0], S_cor_local[skip_S:,1], 
             'o', color='blue', label='experiment (fitted)')
    plt.plot(S_cor_local[skip_S:,0], exp_decay(S_cor_local[skip_S:,0], *lp_popt_S), 
             color='green', label=rf'$l_p$={round(lp_popt_S[1],2)}')
    plt.xlabel(r'$\Delta$r')
    plt.ylabel(r'$\langle \Delta S(r) \Delta S(r+\Delta r)\rangle_{space}$',)
    plt.legend()
    plt.title(f'frame={end_frame}')
    plt.savefig( final_path +'/corrS.jpg' )
    plt.close()
    
    print('analyzing Q')
    deform = get_deform_Q(n, L, 2)
    deform = np.einsum('inml, nml -> inml', deform, S[1:-1,1:-1,1:-1]**2)
    splay, twist, bend = np.sum(deform, axis=(1,2,3)) * (L/N_out)**3
    
    result = {
        "S_mean":   S_mean,
        "S_mode":   S_mode,
        "lp_S":     lp_popt_S[1],
        "splay":    splay,
        "twist":    twist,
        "bend":     bend
        }
    
    with open(final_path+"/result.json", "w") as f:
        json.dump(result, f, indent=4)
        
    with h5py.File(final_path+'/data.h5py', 'w') as fw:
        fw.create_dataset('splay', data=deform[0])
        fw.create_dataset('twist', data=deform[1])
        fw.create_dataset('bend', data=deform[2])
        fw.create_dataset('S_correlation', data=S_cor_local)
        
        
# Input Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int) 						        # stiffness
parser.add_argument("--a", type=float) 								# activity
parser.add_argument("--name", type=int) 							# name
parser.add_argument("--sig", default=2, type=int) 					# sigma for Gaussian filter
parser.add_argument("--N_out", default=400, type=int)               # the final grid dimensions in real space
parser.add_argument("--lp_max_init_ratio", default=5, type=int)			
parser.add_argument("--lp_skip_init_ratio", default=18, type=int)
args = parser.parse_args()

if args.k == None:
    print('No parameters input. Use the default parameters provided by the program instead.')
    k                   = 100
    a                   = 3.5
    name                = 1000
    address             = f"../data/density_{DENSITY:0.2f}/stiffness_{k}/activity_{a}/{name}/"
    sig                 = 2
    N_out               = 128
    lp_max_init_ratio   = 5
    lp_skip_init_ratio  = 18
else:
    k                   = args.k
    a                   = args.a
    name                = args.name 
    address             = f"../../data/density_{DENSITY:0.2f}/stiffness_{k}/activity_{a}/{name}/"
    sig                 = args.sig
    N_out               = args.N_out
    lp_max_init_ratio   = args.lp_max_init_ratio
    lp_skip_init_ratio  = args.lp_skip_init_ratio

main(address, k, a, name, 
     sig=sig, N_out=N_out, L=200, 
     lp_max_init_ratio=lp_max_init_ratio, lp_skip_init_ratio=lp_skip_init_ratio)
