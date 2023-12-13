import numpy as np
import glob
from pathlib import Path
import h5py

from Nematics3D.coarse import kernal_fft, IFFT
from Nematics3D.field import diagonalizeQ

DENSITY = 0.7
WIDTH 	= 200

def main(N_out=400, sig=2, L=200):

	end_list = glog.glob( f'../data/density_{DENSITY:0.2f}/*/*/*/end.txt' )





    address 	= f"../data/density_{DENSITY:0.2f}/stiffness_{stiffness}/activity_{activity}/{name}/"
    fft_path 	= address + "/coarse/"
    save_path 	= address +'/final/'
    
    Path(save_path+'/data').mkdir(exist_ok=True, parents=True)
    Path(fft_path+f'/result_{N_out}').mkdir(exist_ok=True)
    
    files 	= glob.glob(fft_path+'/FFT/*.h5py')
    frames 	= np.array([int(re.findall(r'\d+', file)[-2]) for file in files])
    final 	= np.max(frames)

    coarsed = glob.glob(fft_path+f'/result_{N_out}/*.h5py')
    frames 	= np.array([int(re.findall(r'\d+', file)[-2]) for file in coarsed])

    if final not in frames:

	    with h5py.File(fft_path+'/FFT/'+str(final)+'.h5py', 'r') as f:
	        F_density = f['density'][...]
	        F_qtensor = f['qtensor'][...]

	    Fd = kernal_fft(F_density, sig, LX)
	    Fq = kernal_fft(F_qtensor, sig, LX)

	    den, qtensor = IFFT(Fd, Fq, N_out=N_out)  
	    with h5py.File(fft_path+f'/result_{N_out}/'+str(final)+'.h5py', 'w') as fw:
	        fw.create_dataset('density', data=den)
	        fw.create_dataset('qtensor', data=qtensor)

