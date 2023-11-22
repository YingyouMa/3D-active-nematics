import numpy as np


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