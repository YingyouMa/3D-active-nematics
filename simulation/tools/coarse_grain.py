import numpy as np
from pathlib import Path
import h5py
import glob
import re
import os
import gzip
import argparse

import sys
sys.path
sys.path.append('../Nematics3D')
sys.path
from coarse import read_params, read_pos, count_monomer, truncate_rfft_coefficients, IFFT_nematics, kernal_fft

import time

# Constants of filaments
DENSITY 	= 0.7
NATOMS 		= 50

SDTN		= 0.9 	# threshold of space difference, normalized in box width


def main(
        stiffness, activity, name, 
        suffix='.mpiio.data', N_raw=300, N_trunc=128, 
        if_IFFT=True, N_out=128, sig=2,
        if_diag=True
        ):

    print(f'... Start coarse graining k={stiffness} a={activity} name={name}')
    
    address = f"../data/density_{DENSITY:0.2f}/stiffness_{stiffness}/activity_{activity}/{name}/"
    path = address + 'dump/'
    save_path = address+'coarse/'
    
    Path(save_path+'FFT').mkdir(exist_ok=True, parents=True)
    
    # Find all the dump files
    files = glob.glob(path+'*.mpiio.data')
    frames = np.array([int(re.findall(r'\d+', file)[-1]) for file in files])
    frames = np.sort(frames)[::-1]

    LX, LY, LZ, length, num_polys, NUM_ATOMS = read_params(frames[-1], path, suffix)
    
    NX = N_raw
    NY = N_raw
    NZ = N_raw
    
    VOXEL = np.array([LX, LY, LZ]) / [NX, NY, NZ]

    sdt = SDTN * LX

    for t in frames:

        print(f'frame={t}')
        start = time.time()

        # Read the coordinates
        r = read_pos(t, path, suffix)

        # Derive the raw density and Q tensor field
        den, qtensor = count_monomer(r, length, sdt, LX, VOXEL, [NX, NY, NZ])

        # Fourier transform the density field and truncate it at maximum wave number
        F_density = np.zeros(shape=(NX,NY,NZ//2+1), dtype='complex128')
        F_density = np.fft.rfftn(den)
        F_density = truncate_rfft_coefficients(F_density, N_trunc, N_trunc, N_trunc)
        del den
        
        # Fourier transform the Q tensor field and truncate it at maximum wave number
        F_qtensor = np.zeros(shape=(5,NX,NY,NZ//2+1), dtype='complex128')
        F_qtensor[0] = np.fft.rfftn(qtensor[0,0])
        F_qtensor[1] = np.fft.rfftn(qtensor[0,1])
        F_qtensor[2] = np.fft.rfftn(qtensor[0,2])
        F_qtensor[3] = np.fft.rfftn(qtensor[1,1])
        F_qtensor[4] = np.fft.rfftn(qtensor[1,2])
        F_qtensor = truncate_rfft_coefficients(F_qtensor, N_trunc, N_trunc, N_trunc)
        del qtensor

        # Store the FFT results
        with h5py.File(save_path+'/FFT/'+str(t)+'.h5py', 'w') as f:
        
            f.create_dataset('qtensor',  dtype='complex128', data=F_qtensor)
            f.create_dataset('density',  dtype='complex128', data=F_density)
        
            params = {"grid_N": (NX, NY, NZ), "FFT_truncate": (N_trunc, N_trunc, N_trunc), \
                      "LX": LX, "LY": LY, "LZ": LZ, "num_polys": num_polys, \
                      "num_atoms": NUM_ATOMS,  "data_path": path, "stiffness": stiffness}
            f.create_dataset('params', data=str(params))

        # Zip the analyzed file
        unzip_file = path + f'/{t}.mpiio.data'
        zip_file = path+'nov.'+str(t)+'.mpiio.data.gz'
        with open(unzip_file, 'rb') as f_in:
            content = f_in.read()
        f = gzip.open( zip_file, 'wb')
        f.write(content)
        f.close()
        if os.path.isfile(zip_file):
            os.remove(unzip_file)

        if if_IFFT == True:
    
            Path(save_path+f'result_{N_out}').mkdir(exist_ok=True)
    
            Fd = kernal_fft(F_density, sig, LX)
            Fq = kernal_fft(F_qtensor, sig, LX)
    
            den, qtensor = IFFT_nematics(Fd, Fq, N_out=N_out)
    
            with h5py.File(save_path+f'/result_{N_out}/'+str(t)+'.h5py', 'w') as fw:
                fw.create_dataset('density', data=den)
                fw.create_dataset('qtensor', data=qtensor)

            if if_diag == True:

                Path( address + f"/diagonal/{N_out}/" ).mkdir(exist_ok = True, parents=True)

                from field import diagonalizeQ

                qtensor = qtensor.transpose((1,2,3,0))
                S, n = diagonalizeQ(qtensor, arccos_digit=5)

                np.save( address + f"/diagonal/{N_out}/S_{t}.npy", S )
                np.save( address + f"/diagonal/{N_out}/n_{t}.npy", n )


    
        print(t, round(time.time()-start,2), 's')



# Input Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int) 						        # stiffness
parser.add_argument("--a", type=float) 								# activity
parser.add_argument("--name", type=int) 							# name
parser.add_argument("--sig", default=2) 							# sigma for Gaussian filter
parser.add_argument("--N_raw", default=300, type=int)				# initial coarse-grained grid dimensions
parser.add_argument("--N_trunc", default=128, type=int) 			# truncated wave number
parser.add_argument("--if_IFFT", default=True, type=bool)			# if inverse Fourier transform the data
parser.add_argument("--N_out", default=128, type=int)				# the final grid dimensions in real space
parser.add_argument("--suffix", default=".mpiio.data", type=str)	# the suffix of dump file names
parser.add_argument("--if_diag", default=True, type=bool)           # if diagonalize the Q tensor
args = parser.parse_args()

if args.k == None:
    print('No parameters input. Use the default parameters provided by the program instead.')
    k       = 100
    a       = 1.0
    name    = 100
    sig     = 2
    N_raw   = 300
    N_trunc = 128
    if_IFFT = True
    N_out   = 400
    suffix  = ".mpiio.data"
    if_diag = True
else:
    k       = args.k
    a       = args.a
    name    = args.name
    sig     = parser.sig
    N_raw   = parser.N_raw
    N_trunc = parser.N_trunc
    if_IFFT = parser.if_IFFT
    N_out   = parser.N_out
    suffix  = parser.suffix
    if_diag = parser.if_diag

main(
    k, a, name, 
    N_raw=N_raw, N_trunc=N_trunc, 
    if_IFFT=if_IFFT, N_out=N_out, sig=sig,
    if_diag=if_diag
    )


