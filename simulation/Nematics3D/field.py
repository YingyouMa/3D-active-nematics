# ------------------------------------
# Analysis of Q field in 3D
# Yingyou Ma, Physics @ Brandeis, 2023
# ------------------------------------

from itertools import product
import time

import numpy as np


# --------------------------------------------------------
# Diagonalization of Q tensor in 3D nematics.
# --------------------------------------------------------

def diagonalizeQ(qtensor):
    """
    Diagonalization of Q tensor in 3D nematics.
    Currently it onply provides the uniaxial information.
    Will be updated to derive biaxial analysis in the future.
    Algorythm provided by Matthew Peterson:
    https://github.com/YingyouMa/3D-active-nematics/blob/405c8d54d797cc39c1f14c82112cb43d304ef16c/reference/order_parameter_calculation.pdf

    Parameters
    ----------
    qtensor : numpy array, N x M x L x 5  or  N x M x L x 3x 3
              tensor order parameter Q of each grid
              N, M and L are the number of grids in each dimension.
              The Q tensor for each grid could be represented by 5 numbers or 3 x 3 = 9 numbers
              If 5, then qtensor[..., 0] = Q_xx, qtensor[..., 1] = Q_xy, and so on. 
              If 3 x 3, then qtensor[..., 0,0] = Q_xx, qtensor[..., 0,1] = Q_xy, and so on.
              

    Returns
    -------
    S : numpy array, N x M x L
        the biggest eigenvalue as the scalar order parameter of each grid

    n : numpy array, N x M x L x 3
        the eigenvector corresponding to the biggest eigenvalue, as the director, of each grid.


    Dependencies
    ------------
    - NumPy: 1.22.0

    """

    N, M, L = np.shape(qtensor)[:3]

    if np.shape(qtensor) == (N, M, L, 3, 3):
        Q = qtensor
    elif np.shape(qtensor) == (N, M, L, 5):
        Q = np.zeros( (N, M, L, 3, 3)  )
        Q[..., 0,0] = qtensor[..., 0]
        Q[..., 0,1] = qtensor[..., 1]
        Q[..., 0,2] = qtensor[..., 2]
        Q[..., 1,0] = qtensor[..., 1]
        Q[..., 1,1] = qtensor[..., 3]
        Q[..., 1,2] = qtensor[..., 4]
        Q[..., 2,0] = qtensor[..., 2]
        Q[..., 2,1] = qtensor[..., 4]
        Q[..., 2,2] = - Q[..., 0,0] - Q[..., 1,1]
    else:
        raise NameError(
            "The dimension of qtensor would be (N, M, L, 3, 3) or (N, M, L, 5)"
            )

    p = 0.5 * np.einsum('ijkab, ijkba -> ijk', Q, Q)
    q = np.linalg.det(Q)
    r = 2 * np.sqrt( p / 3 )

    # derive S and n
    temp = 4 * q / r**3
    temp[temp>1]  =  1
    temp[temp<-1] = -1
    S = r * np.cos( 1/3 * np.arccos( temp ) )
    temp = np.array( [
        Q[..., 0,2] * ( Q[..., 1,1] - S ) - Q[..., 0,1] * Q[..., 1,2] ,
        Q[..., 1,2] * ( Q[..., 0,0] - S ) - Q[..., 0,1] * Q[..., 0,2] ,
        Q[..., 0,1]**2 - ( Q[..., 0,0] - S ) * ( Q[..., 1,1] - S  )
        ] )
    n = temp / np.linalg.norm(temp, axis = 0)
    n = n.transpose((1,2,3,0))
    S = S * 1.5

    return S, n

# ----------------------------------------------------------------------
# With the indices of a pair of opposite vertices of sub-orthogonal-box,
# select n and S within that subbox
# ----------------------------------------------------------------------

def select_subbox(subbox_indices, box_grid_size, 
                  margin_ratio=0):

    subbox  = subbox_indices
    N, M, L = box_grid_size

    if margin_ratio != 0:
        xrange, yrange, zrange = subbox_indices[:,1] - subbox_indices[:,0]
        margin = ( np.array([xrange, yrange, zrange]) * margin_ratio/2 ).astype(int)
        subbox[:,0] -= margin
        subbox[:,1] += margin

    xmin, ymin, zmin = subbox[:,0]
    xmax, ymax, zmax = subbox[:,1]

    sl0 = np.array(range(xmin, xmax+1)).reshape(-1,1, 1)%N
    sl1 = np.array(range(ymin, ymax+1)).reshape(1,-1, 1)%M
    sl2 = np.array(range(zmin, zmax+1)).reshape(1,1,-1)%L

    return sl0, sl1, sl2, subbox


# ----------------------------------------------------
# The biaxial analysis of directors within a local box
# ----------------------------------------------------

def local_box_diagonalize(n_box):

    # Derive and take the average of the local Q tensor with the director field around the loop
    Q = np.einsum('abci, abcj -> abcij', n_box, n_box)
    Q = np.average(Q, axis=(0,1,2))
    Q = Q - np.diag((1,1,1))/3

    # Diagonalisation and sort the eigenvalues.
    eigval, eigvec = np.linalg.eig(Q)
    eigvec = np.transpose(eigvec)
    eigidx = np.argsort(eigval)
    eigval = eigval[eigidx]
    eigvec = eigvec[eigidx]

    return eigvec, eigval

# ---------------------------------------------------------------------------
# Interpolate the directors within a local box containing a disclination loop
# The box is given by its unit vector and vertex coordinates
# ---------------------------------------------------------------------------

def interpolate_subbox(vertex_indices, axes_unit, loop_box, n, S, whole_box_grid_size,
                        margin_ratio=2, num_min=20, ratio=[1,1,1]):
    
    #! Rewrite to be easily used in simple interpolation 

    diagnal = vertex_indices[1] - vertex_indices[0]
    num_origin = np.einsum('i, ji -> j', diagnal, axes_unit)
    axes = np.einsum('i, ij -> ij', np.abs(num_origin) / num_origin, axes_unit)
    num_origin = np.abs(num_origin)
    num_scale = num_min / np.min(num_origin) * np.array(ratio)
    numx, numy, numz = np.round(num_scale * num_origin).astype(int)

    box = list(product(np.arange(numx+1), np.arange(numy+1), np.arange(numz+1)))
    box = np.array(box)
    box = np.einsum('ai, ij -> aj', box[:,:3], (axes.T/num_scale).T)
    box = box + vertex_indices[0]

    sl0, sl1, sl2, ortho_box = select_subbox(loop_box, whole_box_grid_size,
                                              margin_ratio=margin_ratio)
    n_box = n[sl0,sl1,sl2]
    S_box = S[sl0,sl1,sl2]

    Q_box = np.einsum('abci, abcj -> abcij', n_box, n_box)
    Q_box = Q_box - np.eye(3)/3
    Q_box = np.einsum('abc, abcij -> abcij', S_box, Q_box)

    xmin, ymin, zmin = ortho_box[:,0]
    xmax, ymax, zmax = ortho_box[:,1]
    points = (
        np.arange(xmin, xmax+1),
        np.arange(ymin, ymax+1),
        np.arange(zmin, zmax+1)
        )

    from scipy.interpolate import interpn
    def interp_Q(index1, index2):
      result = interpn(points, Q_box[..., index1, index2], box)
      result = np.reshape( result, (numx+1, numy+1, numz+1))
      return result

    Q_out = np.zeros( (numx+1, numy+1, numz+1, 3, 3)  )
    Q_out[..., 0,0] = interp_Q(0, 0)
    Q_out[..., 0,1] = interp_Q(0, 1)
    Q_out[..., 0,2] = interp_Q(0, 2)
    Q_out[..., 1,1] = interp_Q(1, 1)
    Q_out[..., 1,2] = interp_Q(1, 2)
    Q_out[..., 1,0] = Q_out[..., 0,1]
    Q_out[..., 2,0] = Q_out[..., 0,2]
    Q_out[..., 2,1] = Q_out[..., 1,2]
    Q_out[..., 2,2] = - Q_out[..., 0,0] - Q_out[..., 1,1]

    Q_out = np.einsum('ab, ijkbc, dc -> ijkad', axes_unit, Q_out, axes_unit)

    return Q_out


# ---------------------------------------
# Exponential decay function, for fitting
# ---------------------------------------

def exp_decay(x, A, t):
    return A * np.exp(-x/t)

# -------------------------------------------------------------------------------------
# Change the correlation function into radial coordinate and fit with exponential decay
# -------------------------------------------------------------------------------------

def corr_sphere_fit(corr, max_init, width=200, skip_init=25, lp0=0.5, iterate=2, skip_ratio=1, max_ratio=10):

    from scipy.optimize import curve_fit

    N = np.shape(corr)[0]
    corr = corr[:max_init, :max_init, :max_init].reshape(-1)

    box = list(product(np.arange(max_init), np.arange(max_init), np.arange(max_init)))
    box = np.array(box).reshape((max_init,max_init,max_init,3))
    r = np.sum(box**2, axis=-1).reshape(-1)
    r = np.sqrt(r) / N * width
    index = r.argsort()
    r = r[index]
    corr = corr[index]

    popt, pcov = curve_fit(exp_decay, 
                           r, corr, 
                           p0=[corr[0],lp0])
    skip = skip_init

    for i in range(iterate):
        skip_length = popt[1] * skip_ratio
        max_length  = popt[1] * max_ratio
        select = ( r > skip_length ) * ( r < max_length )
        popt, pcov = curve_fit(exp_decay, 
                                r[select], corr[select], 
                                p0=[corr[0], popt[1]])
        skip = np.sum(r <= skip_length)

    corr = corr[ r<max_length ]
    r = r[r<max_length]
    perr = np.sqrt(np.diag(pcov))[1]

    return popt, r, corr, skip, perr


# ------------------------------------------------------
# Derive the persistent length of S by Fourier transfrom
# ------------------------------------------------------

def calc_lp_S(S, max_init, width=200, skip_init=25, iterate=2, skip_ratio=1, max_ratio=10, lp0=0.5):

    from scipy.optimize import curve_fit

    N = np.shape(S)[0]

    S_fourier = np.fft.fftn(S - np.average(S))
    S_spectrum = np.absolute(S_fourier)**2
    S_corr = np.real(np.fft.ifftn(S_spectrum)) / N**3

    popt, r, corr, skip, perr = corr_sphere_fit(S_corr, max_init,
                                          width=width, skip_init=skip_init, lp0=lp0, iterate=iterate,
                                          skip_ratio=skip_ratio, max_ratio=max_ratio
                                          )

    return popt, r, corr, skip, perr


# -------------------------------------------------------------
# Calculate the persistent length of n with Legendre polynomial
# -------------------------------------------------------------

def calc_lp_n(n, max_init, width=200, skip_init=25, iterate=2, skip_ratio=1, max_ratio=10, lp0=0.5):

    from scipy.optimize import curve_fit

    N = np.shape(n)[0]

    Q = np.einsum('nmli, nmlj -> nmlij', n, n)

    Q_fourier = np.fft.fftn(Q - np.average(Q, axis=(0,1,2)), axes=(0,1,2))
    Q_spectrum = np.absolute(Q_fourier)**2
    Q_corr = np.real(np.fft.ifftn(Q_spectrum, axes=(0,1,2))) / N**3
    Q_corr = np.sum(Q_corr, axis=(-1,-2))

    popt, r, corr, skip, perr = corr_sphere_fit(Q_corr, max_init,
                                          width = width, skip_init=skip_init, lp0=lp0, iterate=iterate,
                                          skip_ratio=skip_ratio, max_ratio=max_ratio
                                          )

    return popt, r, corr, skip, perr

# ---------------------------------------------------------------------------------
# Visualize the diretors and defects (or the low S region) with given n and S field
# ---------------------------------------------------------------------------------
def visualize_nematics_field(n, S,
                             plotn=True, plotS=True, plotdefects=False,
                             space_index_ratio=1, sub_space=1, interval_n=(1,1,1), origin=(0,0,0),
                             distance=(1.2,1.2,1.2),
                             S_threshold=0.45, S_opacity=1, 
                             S_plot_params={}, S_colorbar_params={}, S_colorbar_range=(0,1),
                             new_figure=True, bgcolor=(0,0,0), if_axes=False):
    
    #! check distance

    from mayavi import mlab

    interval_n  = np.array(interval_n)
    N           = np.array(np.shape(n)[:3])
    if len(np.shape([space_index_ratio])) == 1:
        space_index_ratio = (space_index_ratio, space_index_ratio, space_index_ratio)
    

    # the basic grid for the plotting
    Nx, Ny, Nz = N    # the dimension of grid 
    x = np.linspace(0, Nx*space_index_ratio[0], Nx)
    y = np.linspace(0, Nx*space_index_ratio[1], Ny)
    z = np.linspace(0, Nx*space_index_ratio[2], Nz)

    # create a new figure with the given background if needed
    if new_figure == True:
        mlab.figure(bgcolor=bgcolor)

    if len(np.shape([sub_space])) == 1:
        indexx = np.arange( int(Nx/2 - Nx/2/sub_space), int(Nx/2 + Nx/2/sub_space) )
        indexy = np.arange( int(Ny/2 - Ny/2/sub_space), int(Ny/2 + Ny/2/sub_space) )
        indexz = np.arange( int(Nz/2 - Nz/2/sub_space), int(Nz/2 + Nz/2/sub_space) )
    else:
        indexx = np.arange( sub_space[0][0], sub_space[0][-1]+1 )
        indexy = np.arange( sub_space[1][0], sub_space[1][-1]+1 )
        indexz = np.arange( sub_space[2][0], sub_space[2][-1]+1 )

    if plotS == True:

        inx, iny, inz = np.meshgrid(indexx, indexy, indexz, indexing='ij')
        ind = (inx, iny, inz)

        if np.min(S[ind])>=S_threshold or np.max(S[ind])<=S_threshold:

            print(f'the range of S is ({np.min(S[ind])}, {np.max(S[ind])})')
            print('the threshold of contour plot of S is out of range')

        else:

            X, Y, Z = np.meshgrid(x,y,z, indexing='ij')
            X = X[ind]*distance[0] + origin[0]
            Y = Y[ind]*distance[1] + origin[1]
            Z = Z[ind]*distance[2] + origin[2]

            S_plot_color = S_plot_params.get('color')
            mlab.contour3d(X, Y, Z, S[ind], contours=[S_threshold], opacity=S_opacity, **S_plot_params)
            if S_plot_color == None:
                S_colorbar_label_fmt = S_colorbar_params.get('label_fmt', '%.2f')
                S_colorbar_nb_labels = S_colorbar_params.get('nb_labels', 5)
                S_colorbar_orientation = S_colorbar_params.get('orientation', 'vertical')
                lut_manager = mlab.colorbar(label_fmt=S_colorbar_label_fmt, 
                                            nb_labels=S_colorbar_nb_labels, 
                                            orientation=S_colorbar_orientation,
                                            **S_colorbar_params)
                lut_manager.data_range=(S_colorbar_range)
            else:
                print('The color of S is set. So there is no colorbar for S.')

    if if_axes == True:
        mlab.axes()



    if len(np.shape([sub_space])) == 1:
        indexx = np.arange( int(Nx/2 - Nx/2/sub_space), int(Nx/2 + Nx/2/sub_space), interval_n[0] )
        indexy = np.arange( int(Ny/2 - Ny/2/sub_space), int(Ny/2 + Ny/2/sub_space), interval_n[1] )
        indexz = np.arange( int(Nz/2 - Nz/2/sub_space), int(Nz/2 + Nz/2/sub_space), interval_n[2] )
    else:
        indexx = np.arange( sub_space[0][0], sub_space[0][-1]+1, interval_n[0] )
        indexy = np.arange( sub_space[1][0], sub_space[1][-1]+1, interval_n[1] )
        indexz = np.arange( sub_space[2][0], sub_space[2][-1]+1, interval_n[2] )
    inx, iny, inz = np.meshgrid(indexx, indexy, indexz, indexing='ij')
    ind = (inx, iny, inz)







