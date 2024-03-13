from pathlib import Path
import glob
import re
import argparse
import json

import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt

import h5py

'''
import sys
sys.path
sys.path.append(r'E:\Program\GitHub\3D-active-nematics\simulation')
sys.path
'''
from Nematics3D.field import calc_lp_S, calc_lp_n, exp_decay
from Nematics3D.elastic import get_deform_Q
from Nematics3D.disclination import defect_detect, ordered_bulk_size

Path('../figures/S_lp').mkdir(exist_ok=True, parents=True)
Path('../figures/n_lp').mkdir(exist_ok=True, parents=True)

DENSITY     = 0.7
TIME_STEP   = 0.001
WIDTH       = 200

plt.rcParams.update({'font.size': 30})


def main(
        address, stiffness, activity, name,
        sig=2, N_out=400, L=WIDTH,
        lp_max_init_ratio=8, lp_skip_init_ratio=40,
        if_sep=False, if_save_out=True
        ):
    
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

    print('Start to analyze diagonalized data')
    
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
    lp_popt_S, r_S, corr_S, skip_S, perr_S = output_S
    
    plt.figure(figsize=(16,9))
    plt.plot(r_S[:skip_S], corr_S[:skip_S], 
             'o', color='red', label='experiment (skipped)')
    plt.plot(r_S[skip_S:], corr_S[skip_S:], 
             'o', color='blue', label='experiment (fitted)')
    plt.plot(r_S[skip_S:], exp_decay(r_S[skip_S:], *lp_popt_S), 
             color='green', label=rf'$l_p$={round(lp_popt_S[1],2)}')
    plt.xlabel(r'$\Delta r$')
    plt.ylabel(r'$\langle \delta S(r) \delta S(r+\Delta r)\rangle_r$',)
    plt.legend()
    plt.title(f'frame={end_frame}')
    plt.savefig( final_path +'/corrS.jpg' )
    if if_save_out:
        plt.savefig(f'../figures/S_lp/n{name}_k{stiffness}_a{activity}.jpg')
    plt.close()

    output_n = calc_lp_n(n, max_init=lp_max, width=WIDTH, skip_init=lp_skip) 
    lp_popt_n, r_n, corr_n, skip_n, perr_n = output_n
    
    plt.figure(figsize=(16,9))
    plt.plot(r_n[:skip_n], corr_n[:skip_n], 
             'o', color='red', label='experiment (skipped)')
    plt.plot(r_n[skip_n:], corr_n[skip_n:], 
             'o', color='blue', label='experiment (fitted)')
    plt.plot(r_n[skip_n:], exp_decay(r_n[skip_n:], *lp_popt_n), 
             color='green', label=rf'$l_p$={round(lp_popt_n[1],2)}')
    plt.xlabel(r'$\Delta r$')
    plt.ylabel(r'$\langle \delta \hat{Q}(r) : \delta \hat{Q}(r+\Delta r)\rangle_r$',)
    plt.legend()
    plt.title(f'frame={end_frame}')
    plt.savefig( final_path +'/corrn.jpg' )
    if if_save_out:
        plt.savefig(f'../figures/n_lp/n{name}_k{stiffness}_a{activity}.jpg')
    plt.close()
    
    print('analyzing elastic energy')
    deform = get_deform_Q(n, L, 2)
    deform = np.einsum('inml, nml -> inml', deform, S[1:-1,1:-1,1:-1]**2)
    splay, twist, bend = np.sum(deform, axis=(1,2,3)) * (L/N_out)**3

    print('detecting defects')
    defect_indices = defect_detect(n, boundary=True, print_time=True)
    print(f'find {len(defect_indices)} defects')

    if if_sep:
        print('analyzing ordered bulk size')
        defect_sep = ordered_bulk_size(defect_indices, N_out, WIDTH).mean()
    else:
        defect_sep = None

    
    result = {
        "S_mean":       S_mean,
        "S_mode":       S_mode,
        "lp_S":         lp_popt_S[1],
        "lp_n":         lp_popt_n[1],
        "err_lp_S":     perr_S,
        "err_lp_n":     perr_n,  
        "splay":        splay,
        "twist":        twist,
        "bend":         bend,
        "defect_num":   len(defect_indices),
        'defect_sep':   defect_sep
        }
    
    with open(final_path+"/result.json", "w") as f:
        json.dump(result, f, indent=4)
        
    with h5py.File(final_path+'/data.h5py', 'w') as fw:
        fw.create_dataset('splay', data=deform[0])
        fw.create_dataset('twist', data=deform[1])
        fw.create_dataset('bend', data=deform[2])
        fw.create_dataset('r_S', data=r_S)
        fw.create_dataset('corr_S', data=corr_S)
        fw.create_dataset('r_n', data=r_n)
        fw.create_dataset('corr_n', data=corr_n)
        fw.create_dataset('defect_indices', data=defect_indices)
        
        
# Input Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int) 						        # stiffness
parser.add_argument("--a", type=float) 								# activity
parser.add_argument("--name", type=int) 							# name
parser.add_argument("--sig", default=2, type=int) 					# sigma for Gaussian filter
parser.add_argument("--N_out", default=400, type=int)               # the final grid dimensions in real space
parser.add_argument("--N_out_sep", default=128, type=int)           # the final grid dimensions in real space, to calculate ordered bulk size
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
    N_out_sep           = None
    lp_max_init_ratio   = 5
    lp_skip_init_ratio  = 18
else:
    k                   = args.k
    a                   = args.a
    name                = args.name 
    address             = f"../data/density_{DENSITY:0.2f}/stiffness_{k}/activity_{a}/{name}/"
    sig                 = args.sig
    N_out               = args.N_out
    N_out_sep           = args.N_out_sep
    lp_max_init_ratio   = args.lp_max_init_ratio
    lp_skip_init_ratio  = args.lp_skip_init_ratio

main(address, k, a, name, 
     sig=sig, N_out=N_out, L=WIDTH, 
     lp_max_init_ratio=lp_max_init_ratio, lp_skip_init_ratio=lp_skip_init_ratio)

main(address, k, a, name, 
     sig=sig, N_out=N_out_sep, L=WIDTH, 
     lp_max_init_ratio=lp_max_init_ratio, lp_skip_init_ratio=lp_skip_init_ratio,
     if_sep=True, if_save_out=False)
