import numpy as np
import glob
from pathlib import Path
import h5py
import re

from Nematics3D.coarse import kernal_fft, IFFT
from Nematics3D.field import diagonalizeQ

DENSITY 	= 0.7
WIDTH 		= 200
THRESHOLD 	= 0.9
N_OUT		= 400


end_list = glog.glob( f'../data/density_{DENSITY:0.2f}/*/*/*/end.txt' )
for end in end_list:
	stiffness, activity, name = re.findall(r"\d*\.*\d+", end)[-3:]
	stiffness 	= int(stiffness)
	activity  	= float(activity)
	name 		= int(name)

	with open(end, 'r') as f:
		time_ran = f.readline()

	find_parameter = (parameters['stiffness'] == stiffness) & (parameters['activity'] == activity)
	max_time = parameters.loc[ find_parameter, 'max_step'].values[0]

	if ( max_time <= time_ran ) or ( ( max_time - time_ran ) / max_time > THRESHOLD ):

		address 	= f"../data/density_{DENSITY:0.2f}/stiffness_{stiffness}/activity_{activity}/{name}/"
		fft_path 	= address + "/coarse/"

		Path(fft_path+'/FFT').mkdir(exist_ok=True, parents=True)
    	Path(fft_path+f'/result_{N_out}').mkdir(exist_ok=True)

	    FFTed 		= glob.glob(fft_path+'/FFT/*.h5py')
	    frames 		= np.array([int(re.findall(r'\d+', file)[-2]) for file in FFTed])

	    if max_time not in frames:
	    	



		coarsed = glob.glob(fft_path+f'/result_{N_OUT}/*.h5py')
		frames 	= np.array([int(re.findall(r'\d+', file)[-2]) for file in coarsed])

		if max_time not in frames:

		    with h5py.File(fft_path+'/FFT/'+str(max_time)+'.h5py', 'r') as f:
		        F_density = f['density'][...]
		        F_qtensor = f['qtensor'][...]

		    Fd = kernal_fft(F_density, sig, LX)
		    Fq = kernal_fft(F_qtensor, sig, LX)

		    den, qtensor = IFFT(Fd, Fq, N_out=N_out)  
		    with h5py.File(fft_path+f'/result_{N_out}/'+str(final)+'.h5py', 'w') as fw:
		        fw.create_dataset('density', data=den)
		        fw.create_dataset('qtensor', data=qtensor)



						
	 







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

