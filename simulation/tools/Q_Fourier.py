import glob
import re
from pathlib import Path
import argparse

import time

import numpy as np
import h5py
import matplotlib.pyplot as plt

loc1='upper right'
loc2='lower left'

DENSITY = 0.7

Path('../figures/k_instability').mkdir(exist_ok=True, parents=True)

def analyze_fft(data, N=3):
    
    # N: select the top N predominant wave vectors
    
    data = data - np.average(data)
    data_fft = np.abs( np.fft.fftshift( np.fft.fftn( data ) ) )**2
    half = int( np.shape(data)[0]/2 )
    data_fft = data_fft[ half: ]
    data_fft_argsort = np.argsort(data_fft, axis=None)
    
    result = np.zeros((N, 3)) # angle, coefficient, normalized coefficient
    for i in range(1, N+1):
        result[i-1,1] = data_fft.reshape(-1)[data_fft_argsort[-i]]
        k_vector = np.unravel_index( data_fft_argsort[-i], np.shape(data_fft) )
        k_vector = np.array(k_vector) - [0, half, half]
        k_vector = k_vector / np.linalg.norm(k_vector)
        result[i-1,0] = np.arccos( k_vector[0] ) * 180 / np.pi
    result[:, 2] = result[:, 1] / result[0, 1]
    
    return result

def analyze(address, frame, size, N=3):
    start = time.time()
    
    with h5py.File(f'{address}/coarse/result_{size}/{frame}.h5py', 'r') as f:
        qtensor = f['qtensor'][...]

    S = np.load( address + f'/diagonal/{size}/S_{frame}.npy')
    
    # 5: Qxx, Qxy, Qxz, Qyy, Qyz
    # N: Nth coefficients
    # 3: angle, coefficient, normalized coefficient
    result = np.zeros(( 5, N, 3 ))
    for i in range(5):
        result[i] = analyze_fft(qtensor[i], N=N)
    
    print(frame, round(time.time()-start, 2), 's')
    return result, np.average(S)

def main(address, stiffness, activity, name, N=3):

    files = glob.glob(f'{address}/diagonal/128/S_*.npy')
    frames = np.array([int(re.findall(r'\d+', file)[-1]) for file in files])
    frames = np.sort(frames)

    result = np.zeros((len(frames), 5, N, 3))
    S_mean = np.zeros(len(frames))

    out_path = address + '/analysis/k_instability/'

    if len(glob.glob(out_path+'/frames.npy')) != 0:
        print('Found previous data')
        frame_old = np.load(out_path+'/frames.npy')
        result[:len(frame_old)] = np.load(out_path+'/result.npy')
        S_mean[:len(frame_old)] = np.load(out_path+'/S_mean.npy')
    else:
        frame_old = []
    for i in range(len(frame_old), len(frames)):
        frame = frames[i]
        result[i], S_mean[i] = analyze(address, frame, 128, N=N)
        print(f'{i+1-len(frame_old)}/{len(frames)-len(frame_old)}')

    Path(out_path).mkdir(exist_ok=True)

    np.save(out_path+'/result', result)
    np.save(out_path+'/S_mean', S_mean)
    np.save(out_path+'/frame', frames)

    angle_mean = np.sum( result[..., 0] * result[..., 2] / np.sum(result[..., 2], axis=-1, keepdims=True), axis=-1)
    temp = result[:,1:3] # only Q_xy and Q_xz
    angle_mean_both = np.sum( temp[..., 0] * temp[..., 1] / np.sum(temp[..., 1], axis=(-1,-2), keepdims=True), axis=(-1,-2))

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    frame_listx = frames/1000
    lns1 = ax.plot(frame_listx, S_mean, '--', label=r'$\langle S\rangle$', color='black')
    lns2 = ax2.plot(frame_listx, result[:,0,0,1], label='Qxx')
    lns3 = ax2.plot(frame_listx, result[:,1,0,1], label='Qxy', linewidth=3)
    lns4 = ax2.plot(frame_listx, result[:,2,0,1], label='Qxz', linewidth=3)
    lns5 = ax2.plot(frame_listx, result[:,3,0,1], label='Qyy')
    lns6 = ax2.plot(frame_listx, result[:,4,0,1], label='Qyz')
    lns = lns1 + lns2 + lns3 + lns4 + lns5 + lns6
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=loc1)
    ax.set_title('value of largest Fourier coefficient')
    ax.set_xlabel('time')
    fig.savefig(f'../figures/k_instability/k{stiffness}_a{activity}/Fourier.jpg')
    fig.savefig(out_path+'/Fourier.jpg')
    fig.close()
    
    
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    lns1 = ax.plot(frame_listx, S_mean, '--', label=r'$\langle S\rangle$', color='black')
    lns2 = ax2.plot(frame_listx, angle_mean[:,1], label='Qxy')
    lns3 = ax2.plot(frame_listx, angle_mean[:,2], label='Qxz')
    lns4 = ax2.plot(frame_listx, angle_mean_both, label='average', linewidth=3)
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=loc2)
    ax.set_title(r'$\bar{\theta}$')
    ax.set_xlabel('time')
    fig.savefig(f'../figures/k_instability/k{stiffness}_a{activity}_n{name}/theta.jpg')
    fig.savefig(out_path+'/theta.jpg')
    fig.close()

# Input Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int) 						        # stiffness
parser.add_argument("--a", type=float) 								# activity
parser.add_argument("--name", type=int) 							# name
parser.add_argument("--N", type=int)                                # select the top N predominant wave vectors
args = parser.parse_args()

if args.k == None:
    print('No parameters input. Use the default parameters provided by the program instead.')
    k                   = 100
    a                   = 1.0
    name                = 1
    N                   = 3
    address             = f"../data/density_{DENSITY:0.2f}/stiffness_{k}/activity_{a}/{name}/"
else:
    k                   = args.k
    a                   = args.a
    name                = args.name 
    N                   = args.N
    address             = f"../data/density_{DENSITY:0.2f}/stiffness_{k}/activity_{a}/{name}/"

main(address, k, a, name, N=N)

