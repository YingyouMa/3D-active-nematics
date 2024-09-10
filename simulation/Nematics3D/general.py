import numpy as np
import time

def array_from_single_or_list(input_data):       
    if isinstance(input_data, (int, float)):  
        return np.array([input_data] * 3)    
    elif isinstance(input_data, (list, tuple, np.ndarray)) and len(input_data) == 3:
        return np.array(input_data)         
    else:
        raise ValueError("Input must be either a single number or a list, tuple, or NumPy array of three numbers")
    

def time_record(func):
    def wrapper(*args, print_time=False, **kwargs):  
        start_time = time.time()  
        result = func(*args, **kwargs)  
        end_time = time.time()  
        if print_time:  
            print(f"Function {func.__name__} took {end_time - start_time:.2f}s to run.")
        return result
    return wrapper

def find_neighbor_coord(x, reservoir, dist):
    from scipy.spatial.distance import cdist
    
    if np.array(x).ndim == 1:
        x = [x]
    return np.where( cdist( x, reservoir)  <= dist )

def sort_line_indices(coords):
    '''
    Sort the indices of defects within a line based on their nearest neighbor order.

    Parameters
    ----------
    coords : array_like, (N, M)
             Array representing a line. It contains the indices of all the defects composing the loop. 
             N is the number of points, and M is the dimension (usually 2 or 3).

    Returns
    -------
    output : numpy.ndarray, (N, M)
             Array representing the sorted indices of the line based on nearest neighbor order.
             N is the number of points, and M is the dimension (usually 2 or 3).

    Dependencies
    ------------
    - NumPy: 1.22.0
    - nearest_neighbor_order(): Function for determining nearest neighbor order in the same module.
    '''

    output = coords[nearest_neighbor_order(coords)]
    return output


def nearest_neighbor_order(points):

    '''
    Determine the nearest neighbor order of points.

    Parameters
    ----------
    points : array_like, (N, M)
             Array containing the coordinates of the points.
             N is the number of points, and M is the dimension (usually 2 or 3).

    Returns
    -------
    order : list
            List representing the nearest neighbor order of the points.

    Dependencies
    ------------
    - NumPy: 1.22.0
    - SciPy: 1.7.3
    '''
    from scipy.spatial.distance import cdist

    num_points = len(points)

    # Calculate the pairwise distance matrix
    dist = cdist(points, points) 

    # Initialize variables for tracking visited points and the order
    visited = np.zeros(num_points, dtype=bool)
    visited[0] = True
    order = [0]  

    # Determine nearest neighbors iteratively
    for i in range(num_points - 1):
        current_point = order[-1]
        nearest_neighbor = np.argmin(dist[current_point, :] + visited * np.max(dist))
        order.append(nearest_neighbor)
        visited[nearest_neighbor] = True

    return order


def get_square(size, num, dim=2):

    edge1 = [   np.linspace(0, size, num),      np.zeros(num)               ]
    edge2 = [   np.zeros(num) + size,           np.linspace(0, size, num)   ]
    edge3 = [   np.linspace(size, 0, num),      np.zeros(num) + size        ]
    edge4 = [   np.zeros(num),                  np.linspace(size, 0, num)   ]

    line1 = np.concatenate( [edge1[0][:-1],edge2[0][:-1],edge3[0][:-1],edge4[0][:-1]] )
    line2 = np.concatenate( [edge1[1][:-1],edge2[1][:-1],edge3[1][:-1],edge4[1][:-1]] )
    result = np.vstack([line1, line2]).T

    if dim == 3:
        result = np.hstack([np.array([np.zeros(len(line1))]).T, result])

    return result


def get_plane(points):
    #! how good are points lying in a plane
    #! average rotation vector
    '''
    Calculate the normal vector of the best-fit plane to a set of 3D points 
    using Singular Value Decomposition (SVD).

    Parameters
    ----------
    points : numpy.ndarray, (..., N, 3)
             Array containing the 3D coordinates of the points.
             The last dimension represents the coordinates (x, y, z).
             It will find the averaged normal vector for each group of N points.

    Returns
    -------
    normal_vector : numpy.ndarray, (..., 3)
                    Array representing the normal vector of the best-fit plane.

    Dependencies
    ------------
    - numpy: 1.22.0   
    '''
    # Calculate the center of the points
    center    = points.mean(axis=-2)

    # Translate the points to be relative to the center
    N = np.shape(points)[-2]
    relative  = points - np.tile(center[:, np.newaxis, :], (*(np.ones(points.ndim-2).astype(int)), N, 1))

    # Perform Singular Value Decomposition (SVD) on the transposed relative points
    svd  = np.linalg.svd(np.swapaxes(relative, -1, -2), full_matrices=False)[0]

    # Extract the left singular vector corresponding to the smallest singular value
    normal_vector = svd[:, :, -1]

    return normal_vector

