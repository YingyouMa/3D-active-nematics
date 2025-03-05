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
from Nematics3D.disclination import defect_detect, defect_classify_into_lines

DENSITY = 0.7
SIZE = 128
WIDTH = 200

BOUND = 100

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

    path_defect = str(get_analysispath(DENSITY, stiffness, activity, name)) + '/defect/'
    path_defect_fig = str(get_figpath(DENSITY, stiffness, activity, name)) + '/defect/'
    path_diag = str(get_diagpath(DENSITY, stiffness, activity, name)) + f'/{SIZE}/'

    Path(path_defect).mkdir(exist_ok=True, parents=True)
    Path(path_defect_fig).mkdir(exist_ok=True, parents=True)

    files = glob.glob(path_diag + '/n_*.npy')
    frames = np.array([int(re.findall(r'\d+', file)[-1]) for file in files])
    frames = np.sort(frames)
    curvature = np.empty(len(frames))
    beta = np.empty(len(frames))


    frame_last = 0
    if if_cover == False and len(glob.glob(path_defect + '/beta.npy')) != 0:
        print('Found previous data')
        beta_old = np.load(path_defect+'/beta.npy')
        frame_last = len(beta_old)
        beta[:frame_last] = beta_old
        curvature[:frame_last] = np.load(path_defect+'/curvature.npy')
    else:
        beta_old = []

    for i in range(frame_last, len(frames)):
        frame = frames[i]
        start = time.time()
        n = np.load(path_diag + f'/n_{frame}.npy')
        defect_indices = defect_detect(n, is_boundary_periodic=1)
        if np.size(defect_indices) >= BOUND:
            lines = defect_classify_into_lines(defect_indices, SIZE, space_index_ratio=WIDTH/SIZE)
            beta_all = np.empty(0)
            curvature_all = np.empty(0)
            for line in lines:
                if line._defect_num >= BOUND:
                    trash = line.update_smoothen()
                    line.update_geometry()
                    beta_here = line.update_beta(n=n)
                    curvature_here = line._curvature
                    beta_all = np.concatenate([beta_here, beta_all])
                    curvature_all = np.concatenate([curvature_here, curvature_all])
            beta[i] = np.average(beta_all)
            curvature[i] = np.average(curvature_all)

        print(f'{i+1-frame_last}/{len(frames)-frame_last}', str(round(time.time()-start, 2)) + 's')

    np.save(path_defect + '/beta.npy', beta)
    np.save(path_defect + '/curvature.npy', curvature)

    frame_listx = frames * time_step

    plt.figure(figsize=(15,10))
    plt.plot(frame_listx, beta)
    plt.title(r'twist angle')
    plt.xlabel('time')
    plt.savefig(path_defect_fig + f'/n{name}_k{stiffness}_a{activity}_beta.jpg' )
    plt.close()

    plt.figure(figsize=(15,10))
    plt.plot(frame_listx, curvature)
    plt.title(r'curvature')
    plt.xlabel('time')
    plt.savefig(path_defect_fig + f'/n{name}_k{stiffness}_a{activity}_curvature.jpg' )
    plt.close()


# Input Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int) 						        # stiffness
parser.add_argument("--a", type=float) 								# activity
parser.add_argument("--name", type=int) 							# name
parser.add_argument("--cover", action="store_true") 
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