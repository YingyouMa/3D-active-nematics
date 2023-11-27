import numpy as np
from mayavi import mlab

from nematics3D.field import select_subbox
from nematics3D.disclination import find_defect, sort_loop_indices

def main(
        loop_box_indices, n_whole, width, 
        margin_ratio=0.6, upper=0, down=0, norm_index=0, 
        tube_radius=0.25, tube_opacity=0.5
        ):

    dmean, d_box, grid, norm_vec, n_box, N = setup(loop_box_indices, 
                                                    n_whole, 
                                                    width, 
                                                    margin_ratio=margin_ratio
                                                    )
    
    down, upper = np.sort([down, upper])
    if upper==down:
        upper = dmean + 0.5
        down  = dmean - 0.5
    else:
        upper = dmean + upper
        down  = dmean + down

    mlab.figure(bgcolor=(0,0,0))
    plot_plane(upper, down, d_box, grid, norm_vec, n_box)
    plot_loop(
            n_box, loop_box_indices[:,0], 
            N=N, width=width, tube_radius=tube_radius, tube_opacity=tube_opacity
            )

def setup(loop_box_indices, n_whole, width, margin_ratio=0.6, norm_index=0):

    # Find the region enclosing the loop. The size of the region is controlled by margin_ratio
    N = np.shape(n_whole)[0]
    sl0, sl1, sl2 = select_subbox(loop_box_indices, [N, N, N], margin_ratio=margin_ratio)

    # Select the local n around the loop
    n_box = n_whole[sl0,sl1,sl2]

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

    # The directors within one cross section of the loop will be shown
    # Select the cross section by its norm vector
    # The norm of the principle plane is the eigenvector corresponding to the smallest eigenvalue
    norm_vec = eigvec[norm_index]

    # Build the grid for visualization
    x = np.arange( loop_box_indices[0][0], loop_box_indices[0][-1]+1 )/N*width
    y = np.arange( loop_box_indices[1][0], loop_box_indices[1][-1]+1 )/N*width
    z = np.arange( loop_box_indices[2][0], loop_box_indices[2][-1]+1 )/N*width
    grid = np.meshgrid(x,y,z, indexing='ij')

    # Find the height of the middle cross section: dmean
    d_box = np.einsum('iabc, i -> abc', grid, norm_vec)
    dmean = np.average(d_box)

    return dmean, d_box, grid, norm_vec, n_box, N

def plot_plane(upper, down, d_box, grid, norm_vec, n_box):

    index = (d_box<upper) * (d_box>down)
    index = np.where(index == True)
    n_plane = n_box[index]
    scalars = np.abs(np.einsum('ij, j -> i', n_plane, norm_vec))

    X, Y, Z = grid
    cord1 = X[index] - n_plane[:,0]/2
    cord2 = Y[index] - n_plane[:,1]/2
    cord3 = Z[index] - n_plane[:,2]/2

    vector = mlab.quiver3d(
            cord1, cord2, cord3,
            n_plane[:,0], n_plane[:,1], n_plane[:,2],
            mode = '2ddash',
            scalars = scalars,
            scale_factor=0.5,
            opacity = 1
            )
    vector.glyph.color_mode = 'color_by_scalar'
    lut_manager = mlab.colorbar(object=vector)
    lut_manager.data_range=(0,1)

def plot_loop(n_box, origin, N=1, width=1, tube_radius=0.75, tube_opacity=0.5):
    loop_indices = find_defect(n_box)
    if len(loop_indices) > 0:
        loop_indices = loop_indices + np.tile(origin, (np.shape(loop_indices)[0],1) )
        loop_coord = sort_loop_indices(loop_indices)/N*width
        loop_coord = np.concatenate([loop_coord, [loop_coord[0]]])
        mlab.plot3d(*(loop_coord.T), tube_radius=tube_radius, opacity=tube_opacity)
