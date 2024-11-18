import glob
import re
from pathlib import Path
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

ROOT = str(Path(__file__).resolve().parent.parent)
import sys
sys.path.append(ROOT)
from pathname import *
from Nematics3D.disclination import defect_detect
from Nematics3D.elastic import get_deform_Q

DENSITY = 0.7
SIZE = 128
WIDTH = 200

font_scale = 2.0
plt.rcParams["font.family"]      = "serif"
plt.rcParams["font.serif"]       = "Times New Roman"
plt.rcParams["mathtext.rm"]      = "serif"
plt.rcParams["mathtext.it"]      = "serif:italic"
plt.rcParams["mathtext.bf"]      = "serif:bold"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["xtick.direction"]  = "in"
plt.rcParams["ytick.direction"]  = "in"
plt.rcParams["xtick.labelsize"]  = 6.0*font_scale
plt.rcParams["ytick.labelsize"]  = 6.0*font_scale
plt.rcParams["axes.linewidth"]   = 0.5*font_scale
plt.rcParams["axes.labelsize"]   = 9.0*font_scale
plt.rcParams["axes.titlesize"]   = 5.0*font_scale
plt.rcParams["font.size"]        = 9.0*font_scale
plt.rcParams["legend.fontsize"]  = 5.0*font_scale
plt.rcParams["xtick.top"]    = True
plt.rcParams["xtick.bottom"] = True 
plt.rcParams["ytick.left"]   = True 
plt.rcParams["ytick.right"]  = True
plt.rcParams["xtick.minor.top"]    = True
plt.rcParams["xtick.minor.bottom"] = True 
plt.rcParams["ytick.minor.left"]   = True 
plt.rcParams["ytick.minor.right"]  = True
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.major.size"] = 3.0*font_scale
plt.rcParams["ytick.major.size"] = 3.0*font_scale
plt.rcParams["xtick.minor.size"] = 1.5*font_scale
plt.rcParams["ytick.minor.size"] = 1.5*font_scale
plt.rcParams["xtick.major.width"] = 0.5*font_scale
plt.rcParams["ytick.major.width"] = 0.5*font_scale
plt.rcParams["xtick.minor.width"] = 0.35*font_scale
plt.rcParams["ytick.minor.width"] = 0.35*font_scale

def main(stiffness, activity, name, time_step=0.001, if_cover=False):

    path_length = str(get_analysispath(DENSITY, stiffness, activity, name)) + '/defect/'
    path_ela = str(get_analysispath(DENSITY, stiffness, activity, name)) + '/elastic/'
    path_length_fig = str(get_figpath(DENSITY, stiffness, activity, name)) + '/defect/'
    path_ela_fig = str(get_figpath(DENSITY, stiffness, activity, name)) + '/elastic/'
    path_diag = str(get_diagpath(DENSITY, stiffness, activity, name)) + f'/{SIZE}/'

    Path(path_length).mkdir(exist_ok=True, parents=True)
    Path(path_ela).mkdir(exist_ok=True, parents=True)
    Path(path_length_fig).mkdir(exist_ok=True, parents=True)
    Path(path_ela_fig).mkdir(exist_ok=True, parents=True)

    files = glob.glob(path_diag + '/n_*.npy')
    frames = np.array([int(re.findall(r'\d+', file)[-1]) for file in files])
    frames = np.sort(frames)
    defect_num = np.zeros(len(frames))
    splay = np.zeros(len(frames))
    twist = np.zeros(len(frames))
    bend = np.zeros(len(frames))

    frame_last = 0
    if if_cover == False and len(glob.glob(path_length + '/num.npy')) != 0:
        print('Found previous data')
        defect_num_old = np.load(path_length+'/num.npy')
        frame_last = len(defect_num_old)
        defect_num[:frame_last] = defect_num_old
        splay[:frame_last] = np.load(path_ela+'/splay.npy')
        twist[:frame_last] = np.load(path_ela+'/twist.npy')
        bend[:frame_last] = np.load(path_ela+'/bend.npy')
    else:
        defect_num_old = []

    for i in range(frame_last, len(frames)):
        frame = frames[i]
        start = time.time()
        n = np.load(path_diag + f'/n_{frame}.npy')
        S = np.load(path_diag + f'/S_{frame}.npy')
        deform = get_deform_Q(n, WIDTH, 2)
        deform = np.einsum('inml, nml -> inml', deform, S[1:-1,1:-1,1:-1]**2)
        splay[i], twist[i], bend[i] = np.sum(deform, axis=(1,2,3)) * (WIDTH/SIZE)**3
        defect_indices = defect_detect(n, is_boundary_periodic=1)
        defect_num[i] = len(defect_indices)
        print(f'{i+1-frame_last}/{len(frames)-frame_last}', str(round(time.time()-start, 2)) + 's')

    np.save(path_length + '/num.npy', defect_num)
    np.save(path_ela + '/splay.npy', splay)
    np.save(path_ela + '/twist.npy', twist)
    np.save(path_ela + '/bend.npy', bend)

    frame_listx = frames * time_step

    plt.figure(figsize=(15,10))
    plt.plot(frame_listx, defect_num)
    plt.title('# defects')
    plt.xlabel('time')
    plt.savefig(path_length_fig + f'/n{name}_k{stiffness}_a{activity}_defectnum.jpg' )
    plt.close()

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    lns1 = ax.plot(frame_listx, splay, label='splay')
    lns2 = ax.plot(frame_listx, twist, label='twist')
    lns3 = ax.plot(frame_listx, bend, label='bend')
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left')
    ax.set_title('elastic deformation')
    ax.set_xlabel('time')
    fig.savefig( path_ela_fig + f'/n{name}_k{stiffness}_a{activity}_deform.jpg')
    plt.close(fig)


# Input Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int) 						        # stiffness
parser.add_argument("--a", type=float) 								# activity
parser.add_argument("--name", type=int) 							# name
parser.add_argument("--if_cover", type=bool, default=False) 
args = parser.parse_args()

if args.k == None:
    print('No parameters input. Use the default parameters provided by the program instead.')
    k                   = 100
    a                   = 3.5
    name                = 1
    if_cover            = True
else:
    k                   = args.k
    a                   = args.a
    name                = args.name 
    if_cover            = args.if_cover

main(k, a, name, if_cover=if_cover)