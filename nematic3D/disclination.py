# ---------------------------------------------
# Basic detection and processing of disclinations
# Yingyou Ma, Physics @ Brandeis, 2023
# ---------------------------------------------

import numpy as np
import time
from itertools import product

import field

def find_defect(n, S, threshold=0, print_time=False):

  now = time.time()

  N, M, L = np.shape(S)

  here = n[:, :, 1:, :-1]
  here = np.sign(np.einsum('ilmn, ilmn -> lmn', n[:, :, :-1, :-1], here)) * n[:, :, 1:, :-1]
  here = np.sign(np.einsum('ilmn, ilmn -> lmn', n[:, :, 1:, 1:], here)) * n[:, :, 1:, 1:]
  here = np.sign(np.einsum('ilmn, ilmn -> lmn', n[:, :, :-1, 1:], here)) * n[:, :, :-1, 1:]
  test = np.einsum('ilmn, ilmn -> lmn', n[:, :, :-1, :-1], here)
  defect = np.array(np.where(test<threshold)).transpose().astype(float)
  defect[:,1:] = defect[:,1:]+0.5
  if print_time == True:
    print('finish x-direction, with', str(round(time.time()-now,2))+'s')
  now = time.time()

  here = n[:, 1:, :, :-1]
  here = np.sign(np.einsum('ilmn, ilmn -> lmn', n[:, :-1, :, :-1], here)) * n[:, 1:, :, :-1]
  here = np.sign(np.einsum('ilmn, ilmn -> lmn', n[:, 1:, :, 1:], here)) * n[:, 1:, :, 1:]
  here = np.sign(np.einsum('ilmn, ilmn -> lmn', n[:, :-1, :, 1:], here)) * n[:, :-1, :, 1:]
  test = np.einsum('ilmn, ilmn -> lmn', n[:, :-1, :, :-1], here)
  temp = np.array(np.where(test<threshold)).transpose().astype(float)
  temp[:, [0,2]] = temp[:, [0,2]]+0.5
  defect = np.concatenate([ defect, temp ])
  if print_time == True:
    print('finish y-direction, with', str(round(time.time()-now,2))+'s')
  now = time.time()

  here = n[:, 1:, :-1]
  here = np.sign(np.einsum('ilmn, ilmn -> lmn', n[:, :-1, :-1], here)) * n[:, 1:, :-1]
  here = np.sign(np.einsum('ilmn, ilmn -> lmn', n[:, 1:, 1:], here)) * n[:, 1:, 1:]
  here = np.sign(np.einsum('ilmn, ilmn -> lmn', n[:, :-1, 1:], here)) * n[:, :-1, 1:]
  test = np.einsum('ilmn, ilmn -> lmn', n[:, :-1, :-1], here)
  temp = np.array(np.where(test<threshold)).transpose().astype(float)
  temp[:, :-1] = temp[:, :-1]+0.5
  defect = np.concatenate([ defect, temp ])
  if print_time == True:
    print('finish z-direction, with', str(round(time.time()-now,2))+'s')
  now = time.time()

  return defect


def interpolate_box(origin, axes, num, ratio, loop_box, n, S, margin_ratio=2):

  numx, numy, numz = np.array(num) * np.array(ratio)

  e1 = np.array(axes[0]) / ratio[0]
  e2 = np.array(axes[1]) / ratio[1]
  e3 = np.array(axes[2]) / ratio[2]

  box = np.zeros(((numx+1)*(numy+1)*(numz+1), 3))
  box = np.array(list(product(np.arange(numx+1), np.arange(numy+1), np.arange(numz+1))))
  box = np.einsum('ai, ij -> aj', box[:,:3], np.array([e1,e2,e3])) + origin



  n_box = n[:, sl0,sl1,sl2]
  S_box = S[sl0,sl1,sl2]

    
  
  
  
