# --------------------------------------------------------
# Diagonalisation of Q tensor of 3D nematics.
# Currently it onply provides the uniaxial information.
# Will be updated to derive biaxial analysis in the future
# --------------------------------------------------------
# Yingyou Ma, Physics @ Brandeis, 2023
# Algorithm provided by Matthew Peterson
# --------------------------------------------------------

# -------------------------------
# The function of diagonalisation
# -------------------------------

import numpy as np
import time

def func_diag(qtensor, if_time=False):
  """
  Function returns the eigenvalues and eigenvectors of Q as the input array.

  Parameters
  ----------
  qtensor:  numpy array, tensor order parameter Q of each grid
            shape: (N, M, L, 5), where N, M and L are the number of grids in each dimension.
            qtensor[..., 0] = Q_xx, qtensor[..., 1] = Q_xy, and so on. 
  if_time:  bool value, print the time cost or not

  Returns
  -------
  S:  numpy array, the biggest eigenvalue as the scalar order parameter of each grid
      shape: (N, M, L)
  n:  numpy array, the eigenvector corresponding to the biggest eigenvalue, as the director, of each grid.
      shape: (N, M, L)
  """

  start = time.time()
  
  # derive Q field and calculate it with np.einsum() and np.linalg.det()
  N, M, L = np.shape(Q)[:3]
  Q = np.zeros( (N, M, L, 3, 3)  )
  Q[..., 0,0] = qtensor[0]
  Q[..., 0,1] = qtensor[1]
  Q[..., 0,2] = qtensor[2]
  Q[..., 1,0] = qtensor[1]
  Q[..., 1,1] = qtensor[3]
  Q[..., 1,2] = qtensor[4]
  Q[..., 2,0] = qtensor[2]
  Q[..., 2,1] = qtensor[4]
  Q[..., 2,2] = - Q[..., 0,0] - Q[..., 1,1]

  p = 0.5 * np.einsum('ijkab, ijkba -> ijk', Q, Q)
  q = np.linalg.det(Q)
  r = 2 * np.sqrt( p / 3 )
  del Q

  # derive S and n
  S = r * np.cos( 1/3 * np.arccos( 4 * q / r**3 ) )
  temp = np.array( [
      qtensor[2] * ( qtensor[3] - S ) - qtensor[1] * qtensor[4] ,
      qtensor[4] * ( qtensor[0] - S ) - qtensor[1] * qtensor[2] ,
      qtensor[1]**2 - ( qtensor[0] - S ) * ( qtensor[3] - S  )
      ] )
  n = temp / np.linalg.norm(temp, axis = 0)

  if if_time == True:
    print(str(round(time.time()-start, 2)) + 's')

  return S, n

# -------------------------------------------------------------------
# The code to find all the undiagnolised files and to diagnolise them
# -------------------------------------------------------------------

def run(density, length, stiffness, activity, seed, size=128, cover=False):
  
  address = f"data/density_{density}0/length_{length}/stiffness_{stiffness}/activity_{activity}/seed_{seed}/"
  files = glob.glob( address + f'/coarse/result_{size}/*.h5py')
  frames = np.array([int(re.findall(r'\d+', file)[-2]) for file in files])
  if len(frames) == 0:
      raise NameError('no files')
  frames = np.sort(frames)
  
  solved_files = glob.glob( address + f'/diagonal/{size}/S_*.npy')
  solved_frames = np.array([int(re.findall(r'\d+', file)[-1]) for file in solved_files])
  
  for frame in frames:
      if frame in solved_frames and cover == False:
          print(f'{frame} already analyzed')
      else:
          analyze(frame, address, size)

density_list = [0.7]
length_list = [50]
stiffness_list = [100, 175, 250, 375, 500, 625, 750, 875, 1000, 1125, 1250, 1375, 1500]
activity_list = [1,2,5,7.5,10]
seed_list = [1000]
width_list = [200]

for (density, length, stiffness, activity, seed, width) in product(density_list, length_list, stiffness_list, activity_list, seed_list, width_list):
    print(density, length, stiffness, activity, seed)
    run(density, length, stiffness, activity, seed, size=400)

    
  
  
  
