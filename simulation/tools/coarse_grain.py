import numpy as np
import glob
import re
import argparse

from Nematics3D.coarse import coarse_one_frame


# Constants of filaments
DENSITY 	= 0.7
NATOMS 		= 50

SDTN		= 0.9 	# threshold of space difference, normalized in box width


def main(
        address, stiffness, activity, name, suffix='.data', 
        N_raw=300, N_trunc=128, sdtn=0.9,
        if_IFFT=True, sig=2, N_out=128,
        if_diag=True
        ):

    print(f'... Start coarse graining k={stiffness} a={activity} name={name}')
    
    # Find all the dump files
    dump_path = address + 'dump/'
    files = glob.glob(dump_path + '*' + suffix)
    frames = np.array([int(re.findall(r'\d+', file)[-1]) for file in files])
    frames = np.sort(frames)[::-1]

    for frame in frames:
        coarse_one_frame(
                    address, stiffness, activity, name, frame, suffix=suffix, 
                    N_raw=N_raw, N_trunc=N_trunc, sdtn=sdtn,
                    if_IFFT=if_IFFT, sig=sig, N_out=N_out,
                    if_diag=if_diag
                    )


# Input Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int) 						        # stiffness
parser.add_argument("--a", type=float) 								# activity
parser.add_argument("--name", type=int) 							# name
parser.add_argument("--suffix", default=".data", type=str)	        # the suffix of dump file names
parser.add_argument("--N_raw", default=300, type=int)				# initial coarse-grained grid dimensions
parser.add_argument("--N_trunc", default=128, type=int) 			# truncated wave number
parser.add_argument("--sdtn", default=0.9, type=float) 	            # the threshold of bond length of wrapped monomers, normalized by box width
parser.add_argument("--if_IFFT", default=True, type=bool)			# if inverse Fourier transform the data
parser.add_argument("--sig", default=2, type=int) 					# sigma for Gaussian filter
parser.add_argument("--N_out", default=128, type=int)				# the final grid dimensions in real space
parser.add_argument("--if_diag", default=True, type=bool)           # if diagonalize the Q tensor
args = parser.parse_args()

# Input parameters
# address may be different for different users
if args.k == None:
    print('No parameters input. Use the default parameters provided by the program instead.')
    k       = 100
    a       = 1.0
    name    = 100
    address = f"../data/density_{DENSITY:0.2f}/stiffness_{k}/activity_{a}/{name}/"
    suffix  = ".data"
    N_raw   = 300
    N_trunc = 128
    sdtn    = 0.9
    if_IFFT = True
    sig     = 2
    N_out   = 400
    if_diag = True
else:
    k       = args.k
    a       = args.a
    name    = args.name
    address = f"../../data/density_{DENSITY:0.2f}/stiffness_{k}/activity_{a}/{name}/"
    suffix  = args.suffix
    N_raw   = args.N_raw
    N_trunc = args.N_trunc
    sdtn    = args.sdtn
    if_IFFT = args.if_IFFT
    sig     = args.sig
    N_out   = args.N_out
    if_diag = args.if_diag

main(
    address, k, a, name, suffix=suffix,
    N_raw=N_raw, N_trunc=N_trunc, sdtn=sdtn,
    if_IFFT=if_IFFT, N_out=N_out, sig=sig,
    if_diag=if_diag
    )


