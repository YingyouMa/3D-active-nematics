import numpy as np
from pathlib import Path
import h5py
import glob
import re
import os
import gzip
import argparse

import lammps

import time

density = 0.7
length = 50

sig = 2

parser = argparse.ArgumentParser()
parser.add_argument("--k")
parser.add_argument("--a")
parser.add_argument("--s")
parser.add_argument("--sig", default=2)
parser.add_argument("--N", default=300, type=int)
parser.add_argument("--N_trunc", default=128, type=int)
parser.add_argument("--if_IFFT", default=None, type=bool)
parser.add_argument("--N_out", default=128, type=int)
args = parser.parse_args()

N = args.N
NX_trunc = args.N_trunc
NY_trunc = NX_trunc
NZ_trunc = NX_trunc

N_out = args.N_out
xpad = int((N_out-NX_trunc)/2)

if (type(N) != int or N%2):
    raise ValueError("Grid size must be an even number")
    
def read_params(file_num, path):
    d=path
    data, bounds = lammps.read_lammps(d+str(file_num)+'.mpiio.data')
    NUM_ATOMS = len(data)
    num_polys = np.max(data['mol'])
    length = NUM_ATOMS // num_polys  # polymer length
    xlo, xhi = bounds['x']
    ylo, yhi = bounds['y']
    zlo, zhi = bounds['z']
    LX = xhi - xlo
    LY = yhi - ylo
    LZ = zhi - zlo
    return LX, LY, LZ, length, num_polys, NUM_ATOMS

def read_pos(file_num, path):
    d=path
    data, bounds = lammps.read_lammps(d+str(file_num)+'.mpiio.data')
    data.sort_values(by='id', inplace=True)
    xlo, xhi = bounds['x']
    ylo, yhi = bounds['y']
    zlo, zhi = bounds['z']
    LX = xhi - xlo
    LY = yhi - ylo
    LZ = zhi - zlo
    r = data[['xu', 'yu', 'zu']].values.copy()
    r -= np.array([xlo,ylo,zlo])
    r %= [LX, LY, LZ]
    return r

def truncate_rfft_coefficients(F, new_NX, new_NY, new_NZ):
    if (new_NX%2 or new_NY%2 or new_NZ%2):
        raise ValueError("NX, NY and NZ must be even.")
    NX = F.shape[-3]
    NY = F.shape[-2]
    NZ = (F.shape[-1] - 1) *2
    if (new_NX > NX or new_NY > NY or new_NZ > NZ):
        raise ValueError("New array dimensions larger or equal to input dimensions.")
    if (new_NX == NX and new_NX == NY and new_NZ == NZ):
        return F
    mid_x = NX //2
    mid_y = NY //2
    s = (...,
         slice(mid_x-new_NX//2, mid_x+new_NX//2),
         slice(mid_y-new_NY//2, mid_y+new_NY//2),
         slice(0, new_NZ//2+1), 
         )
    tmp = np.fft.fftshift(F, axes=(-2,-3))
    tmp = tmp[s]
    # tmp = np.fft.ifftshift(tmp, axes=0) /NX /NY
    return tmp /NX /NY /NZ

def kernal_fft(fp, sig, L):
    N = fp.shape[-3]
    F = fp[...] *N**3
    
    kx = np.fft.fftfreq(N, L /N)
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftfreq(N, L /N)
    ky = np.fft.fftshift(ky)
    kz = np.fft.rfftfreq(N, L /N)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    kernel = np.exp(-(Kx**2+Ky**2+Kz**2) *sig**2 *(2 *np.pi)**2 /2)
    
    F = np.einsum('...jkl, jkl->...jkl', F, kernel)
    
    F = F[..., :int(N/2)+1]
    return F
    
def main(stiffness, activity, seed):
    print(f'... Start coarse graining k={stiffness} a={activity}')
    
    address = f"../data/width_200/viscosity_10/density_0.70/length_50/stiffness_{stiffness}/activity_{activity}/seed_{seed}/"
    path = address + 'dump/'
    save_path = address+'coarse/'
    
    Path(save_path+'FFT').mkdir(exist_ok=True, parents=True)
    Path(save_path+f'result_{N_out}').mkdir(exist_ok=True)
    
    files = glob.glob(path+'*.mpiio.data')
    frames = np.array([int(re.findall(r'\d+', file)[-1]) for file in files])
    frames = np.sort(frames)

    LX, LY, LZ, length, num_polys, NUM_ATOMS = read_params(frames[0], path)
    
    NX = N
    NY = N
    NZ = N
    
    VOXEL = np.array([LX, LY, LZ]) / [NX, NY, NZ]
    
    Path(address+'coarse').mkdir(exist_ok=True)
    
    sdtn = 0.9
    # sdtn = space difference threshold normalized in box width
    sdt = sdtn * LX
    
    for t in frames:
        
        print(f'frame={t}')
        start = time.time()
        
        r = read_pos(t, path)
        p = np.gradient(r.reshape([length, -1], order='F'), axis=0).reshape([-1, 3], order='F')
        I = p > sdt; p[I] -= LX
        I = p < -sdt; p[I] += LX
        I = (p < sdt) * (p > sdt/2); p[I] -= LX/2
        I = (p > -sdt) * (p < -sdt/2); p[I] += LX/2
        p = p /np.linalg.norm(p, axis=1)[:,None]
        del I
    
        loc = np.round(r / VOXEL).astype(int)
        loc[:,0] %= NX; loc[:,1] %= NY;  loc[:,2] %= NZ;
        loc = tuple(loc.T) 
        cnt = np.zeros((NX,NY,NZ), dtype=int)
        np.add.at(cnt, loc, 1)
        den_raw = cnt / np.product(VOXEL)
        
        
        M = np.einsum('ij,ik->ijk', p, p)
        qtensor = np.zeros((3,3,NX,NY,NZ))
        np.add.at(qtensor.transpose([2,3,4,0,1]), loc, M)
        qtensor /= np.product(VOXEL)
        
        F_density = np.zeros(shape=(NX,NY,NZ//2+1), dtype='complex128')
        F_density = np.fft.rfftn(den_raw)
        F_density = truncate_rfft_coefficients(F_density, NX_trunc, NY_trunc, NZ_trunc)
        del den_raw
        
        
        F_qtensor = np.zeros(shape=(5,NX,NY,NZ//2+1), dtype='complex128')
        F_qtensor[0] = np.fft.rfftn(qtensor[0,0])
        F_qtensor[1] = np.fft.rfftn(qtensor[0,1])
        F_qtensor[2] = np.fft.rfftn(qtensor[0,2])
        F_qtensor[3] = np.fft.rfftn(qtensor[1,1])
        F_qtensor[4] = np.fft.rfftn(qtensor[1,2])
        F_qtensor = truncate_rfft_coefficients(F_qtensor, NX_trunc, NY_trunc, NZ_trunc)
        del qtensor
        
        
        with h5py.File(save_path+'/FFT/'+str(t)+'.h5py', 'w') as f:
        
            f.create_dataset('qtensor',  dtype='complex128', data=F_qtensor)
            f.create_dataset('density',  dtype='complex128', data=F_density)
        
            params = {"grid_N": (NX, NY, NZ), "FFT_truncate": (NX_trunc, NY_trunc, NZ_trunc), \
                      "LX": LX, "LY": LY, "LZ": LZ, "num_polys": num_polys, \
                      "num_atoms": NUM_ATOMS,  "data_path": path, "stiffness": stiffness}
            f.create_dataset('params', data=str(params))
        
        unzip_file = path + f'/{t}.mpiio.data'
        zip_file = path+'nov.'+str(t)+'.mpiio.data.gz'
        with open(unzip_file, 'rb') as f_in:
            content = f_in.read()
        f = gzip.open( zip_file, 'wb')
        f.write(content)
        f.close()
        if os.path.isfile(zip_file):
            os.remove(unzip_file)

        if args.if_IFFT == True:
            Fd = kernal_fft(F_density, sig, LX)
            Fq = kernal_fft(F_qtensor, sig, LX)
            # print('Gaussian filter is finished')
            
            ratio = (NX_trunc + 2*xpad) / NX_trunc
             
            Fd = np.pad(Fd, ((xpad, xpad), (xpad, xpad), (0, xpad)))
            Fd = np.fft.fftshift(Fd, axes=(-3,-2))
            den = np.fft.irfftn(Fd) * ratio**3
            del Fd
            # print('density derived')
            
            Fq = np.pad(Fq, ((0,0), (xpad, xpad), (xpad, xpad), (0, xpad)))
            Fq = np.fft.fftshift(Fq, axes=(-3,-2))
            qtensor = np.fft.irfftn(Fq, axes=(-3,-2,-1)) * ratio**3
            qtensor /= den[None,...]
            qtensor[0] -= 1/3
            qtensor[3] -= 1/3
            
            with h5py.File(save_path+f'/result_{N_out}/'+str(t)+'.h5py', 'w') as fw:
                fw.create_dataset('density', data=den)
                fw.create_dataset('qtensor', data=qtensor)
               
        print(t, round(time.time()-start,2), 's')

main(args.k, args.a, args.s)



 
    
    
    
