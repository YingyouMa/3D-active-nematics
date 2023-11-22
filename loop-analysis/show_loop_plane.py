import numpy as np

def main(loop_box_coord, S_whole, n_whole, width, margin_ratio=0.6):

	N = np.shape(S)[0] 	# grid number on each dimension
	dis = 1/N*width 	# grid size

	dmean = setup(loop_box_coord, S_whole, n_whole, width, margin_ratio=margin_ratio)


def setup(loop_box_coord, S_whole, n_whole, width, margin_ratio=0.6):

	# Find the region enclosing the loop. The size of the region is controlled by margin_ratio
	xrange, yrange, zrange = loop_box[:,1] - loop_box[:,0]
	margin = ( np.array([xrange, yrange, zrange]) * margin_ratio/2 ).astype(int)
	loop_box[:,0] -= margin
	loop_box[:,1] += margin

	# Consider the periodical boundary condition (index out of the simulation box)
	xmin, ymin, zmin = loop_box[:,0]
	xmax, ymax, zmax = loop_box[:,1]
	N = np.shape(S)[0]
	sl0 = np.array(range(xmin, xmax+1)).reshape(-1,1, 1)%N
	sl1 = np.array(range(ymin, ymax+1)).reshape(1,-1, 1)%N
	sl2 = np.array(range(zmin, zmax+1)).reshape(1,1,-1)%N

	# Select the local n and S around the loop
	n_box = n_whole[:, sl0,sl1,sl2]
	S_box = S_whole[sl0,sl1,sl2]

	# Derive and take the average of the local Q tensor with the director field around the loop
	Q = np.einsum('iabc, jabc -> ijabc', n_box, n_box)
	Q = np.average(Q, axis=(-1,-2, -3))
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
	norm_vec = eigvec[0]

	# Build the grid for visualization
	x = np.arange( loop_box[0][0], loop_box[0][-1]+1)/N*width
	y = np.arange( loop_box[1][0], loop_box[1][-1]+1)/N*width
	z = np.arange( loop_box[2][0], loop_box[2][-1]+1)/N*width
	X, Y, Z = np.meshgrid(x,y,z, indexing='ij')

	# Find the height of the middle cross section: dmean
	r_box = np.array((X, Y, Z))
	d_box = np.einsum('iabc, i -> abc', r_box, norm_vec)
	dmean = np.average(d_box)

	return dmean

def make_plot(upper, down, ):
