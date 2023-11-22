import numpy as np
from mayavi import mlab

def main(loop_box_coord, n_whole, width, margin_ratio=0.6, upper=0, down=0, norm_index=0):

    dmean, d_box, grid, norm_vec, n_box = setup(loop_box_coord, n_whole, width, margin_ratio=margin_ratio)
    
    down, upper = np.sort([down, upper])
    if upper==down:
        upper = dmean + 0.5
        down  = dmean - 0.5

    make_plot(upper, down, d_box, grid, norm_vec, n_box)

def setup(loop_box_coord, n_whole, width, margin_ratio=0.6, norm_index=0):

    # Find the region enclosing the loop. The size of the region is controlled by margin_ratio
    xrange, yrange, zrange = loop_box_coord[:,1] - loop_box_coord[:,0]
    margin = ( np.array([xrange, yrange, zrange]) * margin_ratio/2 ).astype(int)
    loop_box_coord[:,0] -= margin
    loop_box_coord[:,1] += margin

    # Consider the periodical boundary condition (index out of the simulation box)
    xmin, ymin, zmin = loop_box_coord[:,0]
    xmax, ymax, zmax = loop_box_coord[:,1]
    N = np.shape(S)[0]
    sl0 = np.array(range(xmin, xmax+1)).reshape(-1,1, 1)%N
    sl1 = np.array(range(ymin, ymax+1)).reshape(1,-1, 1)%N
    sl2 = np.array(range(zmin, zmax+1)).reshape(1,1,-1)%N

    # Select the local n and S around the loop
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
    x = np.arange( loop_box_coord[0][0], loop_box_coord[0][-1]+1)/N*width
    y = np.arange( loop_box_coord[1][0], loop_box_coord[1][-1]+1)/N*width
    z = np.arange( loop_box_coord[2][0], loop_box_coord[2][-1]+1)/N*width
    grid = np.meshgrid(x,y,z, indexing='ij')

    # Find the height of the middle cross section: dmean
    d_box = np.einsum('iabc, i -> abc', grid, norm_vec)
    dmean = np.average(d_box)

    return dmean, d_box, grid, norm_vec, n_box

def make_plot(upper, down, d_box, grid, norm_vec, n_box):

    index = (d_box<upper) * (d_box>down)
    index = np.where(index == True)
    n_select = n_box[index]
    scalars = np.abs(np.einsum('ij, j -> i', n_select, norm_vec))

    X, Y, Z = grid
    cord1 = X[index] - n_select[:,0]/2
    cord2 = Y[index] - n_select[:,1]/2
    cord3 = Z[index] - n_select[:,2]/2

    vector = mlab.quiver3d(
            cord1, cord2, cord3,
            n_select[:,0], n_select[:,1], n_select[:,2],
            mode = '2ddash',
            scalars = scalars,
            scale_factor=0.5,
            opacity = 1
            )
    vector.glyph.color_mode = 'color_by_scalar'
    lut_manager = mlab.colorbar(object=vector)
    lut_manager.data_range=(0,1)