# ------------------------------------
# Analysis of Q field in 3D
# Yingyou Ma, Physics @ Brandeis, 2023
# ------------------------------------

import numpy as np

def diagonalize(qtensor):
  """
  Diagonalization of Q tensor in 3D nematics.
  Currently it onply provides the uniaxial information.
  Will be updated to derive biaxial analysis in the future.
  Algorythm provided by Matthew Peterson:
  https://github.com/YingyouMa/3D-active-nematics/blob/405c8d54d797cc39c1f14c82112cb43d304ef16c/reference/order_parameter_calculation.pdf

  Parameters
  ----------
  qtensor:  numpy array, tensor order parameter Q of each grid
            shape: (N, M, L, 5), where N, M and L are the number of grids in each dimension.
            qtensor[..., 0] = Q_xx, qtensor[..., 1] = Q_xy, and so on. 

  Returns
  -------
  S:  numpy array, the biggest eigenvalue as the scalar order parameter of each grid
      shape: (N, M, L)
  n:  numpy array, the eigenvector corresponding to the biggest eigenvalue, as the director, of each grid.
      shape: (N, M, L)
  """
  
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

  return S, n


    
  
  
  
