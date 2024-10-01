import numpy as np 
import time

# ----------------------------------------------------------
# Functions which are being used and general.
# General means the code is for general nematics analysis.
# Not general means the code is specifically for my project.
# ----------------------------------------------------------

from .general import *


DEFECT_NEIGHBOR = np.zeros((10,3))
DEFECT_NEIGHBOR[0] = (1,       0,      0)
DEFECT_NEIGHBOR[1] = (-1,      0,      0)
DEFECT_NEIGHBOR[2] = (0.5,     0.5,    0)
DEFECT_NEIGHBOR[3] = (0.5,     -0.5,   0)
DEFECT_NEIGHBOR[4] = (0.5,     0,      0.5)
DEFECT_NEIGHBOR[5] = (0.5,     0,      -0.5)
DEFECT_NEIGHBOR[6] = (-0.5,     0.5,    0)
DEFECT_NEIGHBOR[7] = (-0.5,     -0.5,   0)
DEFECT_NEIGHBOR[8] = (-0.5,     0,      0.5)
DEFECT_NEIGHBOR[9] = (-0.5,     0,      -0.5)

def defect_detect(n_origin, threshold=0, 
                  boundary_periodic=0, planes=[1,1,1], print_time=False, return_test=False):
    #! defect_indices half integer
    '''
    Detect defects in a 3D director field.
    For each small loop formed by four neighoring grid points,
    calculate the inner product between the beginning and end director.

    Parameters
    ----------
    n_origin : numpy.ndarray, (N, M, L, 3)
               Array containing the 3D director field.
               N, M, L is the number of grids in each dimension

    threshold : float, optional
                Threshold value for defect detection. 
                While calculating the winding number, a defect is identified if
                the inner product between the beginning and end director is smaller than the threshold.
                Default is 0.

    boundary_periodic : bool, or array of three bools, optional
                        Flag to indicate whether to consider periodic boundaries in each dimension. 
                        If only one bool x is given, it is interprepted as (x,x,x)
                        Default is 0, no consideration of periodic boundaries in any dimension

    planes : array, optional
             Indicate the direction of planes whose defects are about to be found.
             Each index stands for x-plane, y-plane, z-plane, seperately.
             For example, if planes=[1,0,0], it will only find defects on seperate x-planes,
             or in other words, it will NOT calculate the winding number along x-direction.
             Default is [1,1,1], to analyze all directions

    print_time : bool, optional
                 Flag to print the time taken for each direction. 
                 Default is False.

    return_test : bool, optional
                  Flag to return the test result of each grid point.
                  Test result is the inner product between the beginning and end director of small loop.
                  Default is False.

    Returns
    -------
    defect_indices : numpy.ndarray, defect_num x 3
                     Array containing the indices of detected defects.
                     In our current algorithm, for each defect's location, there must be one integer and two half-integers.
                     The integer stands for the plane that the defect sits on.
                     #! defect_indices half integer

    Dependencies
    ------------
    - NumPy: 1.26.4

    Called by
    ---------
    '''

    boundary_periodic = array_from_single_or_list(boundary_periodic)

    from .field import add_periodic_boundary

    # Consider the periodic boundary condition
    n = add_periodic_boundary(n_origin, boundary_periodic=boundary_periodic)

    now = time.time()

    defect_indices = np.empty((0,3), float)
    test_result = np.empty((0,), float)
    test_result_all = np.empty((0,), float)

    # X-direction
    if planes[0]:
        here = n[:, 1:, :-1]
        if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:, :-1, :-1], here))
        here = np.einsum('lmn, lmni -> lmni',if_parallel, n[:, 1:, :-1])
        if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:, 1:, 1:], here))
        here = np.einsum('lmn, lmni -> lmni',if_parallel, n[:, 1:, 1:])
        if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:, :-1, 1:], here))
        here = np.einsum('lmn, lmni -> lmni',if_parallel, n[:, :-1, 1:])
        testx = np.einsum('lmni, lmni -> lmn', n[:, :-1, :-1], here)
        temp = np.array(np.where(testx<threshold)).transpose().astype(float)
        temp[:,1:] = temp[:,1:]+0.5
        defect_indices = np.concatenate([ defect_indices, temp ])
        test_result = np.concatenate([test_result, testx[testx<threshold]])
        test_result_all = np.concatenate([test_result_all, testx.reshape(-1)])
        if print_time:
            print('finish x-direction, with', str(round(time.time()-now,2))+'s')
        now = time.time()

    # Y-direction
    if planes[1]:
        here = n[1:, :, :-1]
        if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:-1,:, :-1], here))
        here = np.einsum('lmn, lmni -> lmni',if_parallel, n[1:, :, :-1])
        if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[1:, :, 1:], here))
        here = np.einsum('lmn, lmni -> lmni',if_parallel, n[1:, :, 1:])
        if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:-1, :, 1:], here))
        here = np.einsum('lmn, lmni -> lmni',if_parallel, n[:-1, :, 1:])
        testy = np.einsum('lmni, lmni -> lmn', n[:-1, :, :-1], here)
        temp = np.array(np.where(testy<threshold)).transpose().astype(float)
        temp[:, [0,2]] = temp[:, [0,2]]+0.5
        defect_indices = np.concatenate([ defect_indices, temp ])
        test_result = np.concatenate([test_result, testy[testy<threshold]])
        test_result_all = np.concatenate([test_result_all, testy.reshape(-1)])
        if print_time:
            print('finish y-direction, with', str(round(time.time()-now,2))+'s')
        now = time.time()

    # Z-direction
    if planes[2]:
        here = n[1:, :-1]
        if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:-1, :-1], here))
        here = np.einsum('lmn, lmni -> lmni',if_parallel, n[1:, :-1])
        if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[1:, 1:], here))
        here = np.einsum('lmn, lmni -> lmni',if_parallel, n[1:, 1:])
        if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:-1, 1:], here))
        here = np.einsum('lmn, lmni -> lmni',if_parallel, n[:-1, 1:])
        testz = np.einsum('lmni, lmni -> lmn', n[:-1, :-1], here)
        temp = np.array(np.where(testz<threshold)).transpose().astype(float)
        temp[:, :-1] = temp[:, :-1]+0.5
        defect_indices = np.concatenate([ defect_indices, temp ])
        test_result = np.concatenate([test_result, testz[testz<threshold]])
        test_result_all = np.concatenate([test_result_all, testz.reshape(-1)])
        if print_time:
            print('finish z-direction, with', str(round(time.time()-now,2))+'s')
        now = time.time()

    # Wrap with the periodic boundary condition
    for i, if_periodic in enumerate(boundary_periodic):
        if if_periodic == True:
            defect_indices[:,i] = defect_indices[:,i] % np.shape(n_origin)[i]
    defect_indices, unique = np.unique(defect_indices, axis=0, return_index=True)
    test_result = test_result[unique]

    if return_test:
        return defect_indices, test_result, test_result_all
    else:
        return defect_indices
    

def defect_find_vicinity_grid(defect_indices, num_add=3):

    num = num_add + 2
    length = 4 * num - 4

    result = np.zeros( (np.shape(defect_indices)[0], length, 3) )

    indexx = np.isclose(defect_indices[:, 0], np.round(defect_indices[:, 0]))
    indexy = np.isclose(defect_indices[:, 1], np.round(defect_indices[:, 1]))
    indexz = np.isclose(defect_indices[:, 2], np.round(defect_indices[:, 2]))

    defectx = defect_indices[indexx]
    defecty = defect_indices[indexy]
    defectz = defect_indices[indexz]

    squarex = get_square(1, num, dim=3)
    squarey = squarex.copy()
    squarey[:, [0, 1]] = squarey[:, [1, 0]]
    squarez = squarex.copy()
    squarez[:, [0, 1]] = squarez[:, [1, 0]]
    squarez[:, [1, 2]] = squarez[:, [2, 1]]

    defectx = defectx - np.broadcast_to([0.0, 0.5, 0.5],(np.shape(defectx)[0],3))
    defecty = defecty - np.broadcast_to([0.5, 0.0, 0.5],(np.shape(defecty)[0],3))
    defectz = defectz - np.broadcast_to([0.5, 0.5, 0.0],(np.shape(defectz)[0],3))

    defectx = np.repeat(defectx, length, axis=0).reshape(np.shape(defectx)[0],length,3)
    defecty = np.repeat(defecty, length, axis=0).reshape(np.shape(defecty)[0],length,3)
    defectz = np.repeat(defectz, length, axis=0).reshape(np.shape(defectz)[0],length,3)

    defectx =  defectx + np.broadcast_to(squarex, (np.shape(defectx)[0], length,3))
    defecty =  defecty + np.broadcast_to(squarey, (np.shape(defecty)[0], length,3))
    defectz =  defectz + np.broadcast_to(squarez, (np.shape(defectz)[0], length,3))

    result[indexx] = defectx
    result[indexy] = defecty
    result[indexz] = defectz

    return result


def defect_find_vicinity_Q(defect_indices, n, S=0, num_add=3, boundary_periodic=0):

    from .field import interpolateQ

    defect_vicinity_grid = defect_find_vicinity_grid(defect_indices, num_add=num_add)
    defect_vicinity_Q = interpolateQ(n, defect_vicinity_grid, S=S, boundary_periodic=boundary_periodic)

    return defect_vicinity_Q


def defect_vicinity_analysis(defect_indices, n, S=0, num_add=3, boundary_periodic=0):

    from .field import diagonalizeQ
    from .general import get_plane

    defect_vicinity_Q = defect_find_vicinity_Q(defect_indices, n, 
                                               S=S, num_add=num_add, boundary_periodic=boundary_periodic)
    defect_vicinity_n = diagonalizeQ(defect_vicinity_Q)[1]

    # same_orient = np.sign(np.einsum( 'abi, abi -> ab', defect_vicinity_n[:, 1:], defect_vicinity_n[:, :-1] ))
    # same_orient = np.cumprod( same_orient, axis=-1)[:,-1]
    # final_product = np.einsum( 'a, ai, ai -> a', same_orient, defect_vicinity_n[:,0], defect_vicinity_n[:,-1] )

    norm = get_plane(defect_vicinity_n)

    return norm


def defect_detect_precise(n, S=0, defect_indices=0,
                          threshold1=0.5, threshold2=-0.9, boundary_periodic=0, planes=[1,1,1], num_add=3
                          ):

    if isinstance(defect_indices, int):
        defect_indices = defect_detect(n, threshold=threshold1, boundary_periodic=boundary_periodic, planes=planes)

    final_product = defect_vicinity_analysis(defect_indices, n, S=S, num_add=num_add, boundary_periodic=boundary_periodic)[0]

    return defect_indices[final_product<threshold2]


def calc_coord(defect_indices, origin=(0,0,0), space_index_ratio=1):
    '''
    Derive the coordinates of defects from indices of defects.

    Parameters
    ----------
    defect_indices : array, (M,3)
                     The array that includes all the indices of defects. M is the amount of defects
                     For each defect, one of the indices should be integer and the rest should be half-integer.
                     Usually defect_indices are generated by defect_defect() in this module.

    origin : array of three floats, optional
             Origin of the plot, translating the whole system in real space
             Default is (0, 0, 0), as the system is not translated 

    space_index_ratio : float or array of three floats, optional
                        Ratio between the unit of real space to the unit of grid indices.
                        If the box size is N x M x L and the size of grid of n and S is n x m x l,
                        then space_index_ratio should be (N/n, M/m, L/l).
                        If a single float x is provided, it is interpreted as (x, x, x).
                        Default is 1.

    Returns
    -------
    defect_coord : array, (M,3)
                   The array that includes all the coordinayes of defects.

    '''

    if len(np.shape([space_index_ratio])) == 1:
        space_index_ratio = (space_index_ratio, space_index_ratio, space_index_ratio)
    
    defect_coord = defect_indices + np.broadcast_to(origin, (np.shape(defect_indices)[0],3))
    defect_coord = np.einsum('na, a -> na', defect_coord, space_index_ratio)
    
    return defect_coord


def defect_neighbor_possible_get(defect_index, box_size_periodic=[np.inf, np.inf, np.inf]):
    #! defect_indices half integer
    '''
    Derive all the possible neighboring defects' indices of the given defect index.

    For any defect, one of the index must be integer, representing the layer,
    and the other two indices must be half-integer, representing the center of one pixel in this layer.
    The index is usually provided by defect_detect().
    Supposing defecy1 = (layer, center1, center2), where layer is integer while center1 and center2 are half-integers,
    the set of all the possible neighboring defects is
    (layer+-1,     center1,        center2)
    (layer+-0.5,   center1+-0.5,   center2)
    (layer+-0.5,   center1,        center2+-0.5)
    here +- means plusminus, and the order is unneccessary as (+,+), (-,+), (+,-), (-,-) are all possible
    so there are 2+4+4=10 possible neighboring defects.

    It will also generate all the mirror points of each possible neighboring defect.

    Parameters
    ----------
    defect_index : array of three floats
                   One of them must be integer and the rest two are half integers

    box_size_periodic : array of three floats, or one float, optional
                        The number of indices in each dimension, x, y, z.
                        If box_size is x, it will be interprepted as [x,x,x].
                        If one of the boundary is not periodic, the corresponding value in box_size is np.inf.
                        For example, if the box is periodic in x and y dimension, and the possible maximum index is X and Y,
                        box_size should be [X+1, Y+1, np.inf].
                        Default is [np.inf, np.inf, np.inf], which means the function only return the point itself.

    Returns
    -------
    result : numpy array, (10,3)
             The indices of all possible neighboring defects

    Dependencies
    ------------
    - Numpy : 1.26.4
    - .field.find_mirror_point_boudanry()

    Called by
    ---------
    - .disclination.is_defects_connnected()
    '''
    from .field import find_mirror_point_boundary

    defect_index = np.array(defect_index)
    box_size_periodic = array_from_single_or_list(box_size_periodic)
    neighbor = DEFECT_NEIGHBOR.copy()

    layer_index = np.where( defect_index%1 == 0 )[0][0]
    if layer_index != 0:
        neighbor[:, (0, layer_index)] = neighbor[:, (layer_index, 0)]

    result = np.tile(defect_index,(10,1)) + neighbor
    
    defect_index_in_periodic = defect_index[box_size_periodic!=np.inf]
    box_size_in_periodic = box_size_periodic[box_size_periodic!=np.inf]
    if len(defect_index_in_periodic)>0:
        if np.min(defect_index_in_periodic)<=1 or np.any(defect_index_in_periodic >= box_size_in_periodic-2):
            result = [find_mirror_point_boundary(point, box_size_periodic=box_size_periodic)
                      for point in result]
            result = np.vstack(result)
    
    return result


def is_defects_connected(defect1, defect2, box_size_periodic=[np.inf, np.inf, np.inf]):
    #! defect_indices half integer
    '''
    To examine if two defects are connected.
    For any defect, one of the index must be integer, representing the layer,
    and the other two indices must be half-integer, representing the center of one pixel in this layer.
    The index is usually provided by defect_detect().
    Supposing defecy1 = (layer, center1, center2), where layer is integer while center1 and center2 are half-integers,
    the set of all the possible neighboring defects is
    (layer+-1,     center1,        center2)
    (layer+-0.5,   center1+-0.5,   center2)
    (layer+-0.5,   center1,        center2+-0.5)
    here +- means plusminus, and the order is unneccessary as (+,+), (-,+), (+,-), (-,-) are all possible
    so there are 2+4+4=10 possible neighboring defects.
    This function will examine if defect2 is one of the possible neighboring defects

    Note that, if one of the box_size is np.inf (which means there is no periodic boundary condition),
    then there should NOT be negative value in the correspoinding dimension in point, because it's meaningless.

    Parameters
    ----------
    defect1 : array, (3,)
              The indices of the first defect on the index grid (not coordinate of the real space)

    defect2 : array, (3,)
              The indices of the other defect on the index grid (not coordinate of the real space)

    box_size_periodic : array of three floats, or one float, optional
                        The number of indices in each dimension, x, y, z.
                        If box_size is x, it will be interprepted as [x,x,x].
                        If one of the boundary is not periodic, the corresponding value in box_size is np.inf.
                        For example, if the box is periodic in x and y dimension, and the possible maximum index is X and Y,
                        box_size should be [X+1, Y+1, np.inf].
                        Default is [np.inf, np.inf, np.inf], which means the function only return the point itself.

    Returns
    -------
    result : str
             "same" means these two defects are the same.
             "neighbor" means these two defects are connnected.
             "far" means these two defects are not connnected.

    Dependencies
    ------------
    - NumPy: 1.22.0 
    - .general.array_from_single_or_list()
    - .field.find_mirror_point_boundary()
    
    Called by
    ---------
    - class: DisclinationLine
    '''

    from .field import find_mirror_point_boundary

    box_size_periodic = array_from_single_or_list(box_size_periodic)

    boundary_periodic = box_size_periodic!=np.inf
    defect1 = np.array(defect1)
    defect2 = np.array(defect2)
    defect1[boundary_periodic] = defect1[boundary_periodic] % box_size_periodic[boundary_periodic]
    defect2[boundary_periodic] = defect2[boundary_periodic] % box_size_periodic[boundary_periodic]
    defect_diff = np.abs(defect1 - defect2)
    if np.linalg.norm(defect_diff) == 0:
        return "same"
    
    defect1_neighbor_possible = defect_neighbor_possible_get(defect1, box_size_periodic=box_size_periodic)
    defect2 = find_mirror_point_boundary(defect2, box_size_periodic=box_size_periodic)
    setA = set(map(tuple, defect1_neighbor_possible))
    setB = set(map(tuple, defect2))

    common_points = setA & setB

    if len(common_points) > 0:
        return "neighbor"
    else:
        return "far" 


def add_mid_points_disclination(line, is_loop=False):
    #! defect_indices half integer
    #! add one more point if the line is a loop
    '''
    Add mid-points into the disclination lines.

    Parameters
    ----------
    line : array, (defect_num,3)
           The array that includes all the indices of defects.
           The defects must be sorted, as the neighboring defects have the minimum distance.
           For each defect, one of the indices should be integer and the rest should be half-integer.
           Usually defect_indices are generated by defect_defect() and smoothen_line() in this module.

    is_loop : bool, optional
              If this disclination line is a closed loop.
              If so, this function will add one more point between the start and the end of this loop.
              Default is False.

    Returns
    -------
    line_new : array, ( 2*defect_num-1 , 3 ) or ( 2*defect_num , 3 ), for a crossing line or a loop
               The new array that includes all the indices of defects, with mid-points added

    Dependencies
    ------------
    - NumPy: 1.22.0

    Called by
    ---------
    - Disclination_line                
    '''

    if is_loop == True:
        line = np.vstack([line, line[0]])

    line_new = np.zeros((2*len(line)-1,3))
    line_new[0::2] = line
    defect_diff = line[1:] - line[:-1]
    defect_diff_mid_value = np.sign(defect_diff[np.where( line[:-1]%1 == 0 )]) * 0.5
    defect_diff_mid_orient = (line[:-1]%1 == 0).astype(int)
    line_new[1::2] = line_new[0:-1:2] + np.array([defect_diff_mid_value]).T * defect_diff_mid_orient

    if is_loop == True:
        line = line[:-1]
    
    return line_new 


@time_record
def defect_classify_into_lines(defect_indices, box_size_periodic=[np.inf, np.inf, np.inf],
                               origin=(0,0,0), space_index_ratio=1):
    """
    Short description of the function.
    
    Detailed explanation.
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    
    param2 : type, optional 
        Description of param2.
    
    Returns
    -------
    return_name : return_type
        Description of return values.
    
    Dependencies
    ------------
    - Dependencies
    
    Called by
    ---------
    Called by info
    """
    from .classes.graph import Graph
    from .classes.disclination_line import DisclinationLine
    from .field import unwrap_trajectory

    box_size_periodic = array_from_single_or_list(box_size_periodic)
    defect_indices_hash = make_hash_table(defect_indices)

    graph = Graph()

    for idx1, defect in enumerate(defect_indices):
        neighbor = defect_neighbor_possible_get(defect, box_size_periodic=box_size_periodic)
        search = search_in_reservoir(neighbor, defect_indices_hash, is_reservoir_hash=True)
        search = search[~np.isnan(search)].astype(int)
        for idx2 in search:
            graph.add_edge(idx1, idx2)

    paths = graph.find_path()
    paths = [unwrap_trajectory(defect_indices[path], box_size_periodic=box_size_periodic) 
            for path in paths]

    lines = [DisclinationLine(path, box_size_periodic, 
                              origin=origin, space_index_ratio=space_index_ratio)  
                              for path in paths]

    return lines


def blue_red_in_white_bg():
    '''
    Generate a colormap with a transition from blue to red. 
    The color is normalized to be distinct on white background.
    Mostly used for visualizing disclination lines

    
    Returns
    -------
    colormap : numpy.ndarray, 511 x 3
               Array representing the colormap with RGB values.

               
    Dependencies
    ------------
    - numpy: 1.22.0
    '''

    colormap = np.zeros((511,3))
    colormap[:256,1] = np.arange(256)
    colormap[:256,2] = 255 - np.arange(256)
    colormap[255:,1] = 255 - np.arange(256)
    colormap[255:,0] = np.arange(256)
    colormap = colormap / 255
    colormap = colormap / np.linalg.norm(colormap, axis=-1, keepdims=True)

    return colormap


def sample_far(num):
    '''
    Generate a sequence of length num, 
    where each number is trying to be far away from previous numbers.
    The leading numbers are:
    0, 1, 1/2, 1/4, 3/4, 1/8, 3/8. 5/8, 7/8, 1/16, 3/16 ...

    
    Parameters
    ----------
    num : int
          The length of the generated sequence.

          
    Returns
    -------
    result : numpy.ndarray, shape (num,)
             Array representing the generated sample sequence.

             
    Dependencies
    ------------
    - numpy: 1.22.0
    '''

    result_init = [0,1]
    if num <= 2:
        result = np.array(result_init[:num])
        return result

    n = np.arange(2, num)
    a = 2**np.trunc(np.log2(n-1)+1)
    b = 2*n - a - 1

    result = np.zeros(num)

    result[0] = 0
    result[1] = 1
    result[2:] = b/a
    
    return result


def is_loop_new(lines, loop_indices, 
                threshold=4, box_size_periodic=[np.inf, np.inf, np.inf], min_length=8):

    from scipy.spatial.distance import cdist
    from .field import unwrap_trajectory

    loop_indices = loop_indices[:-1]
    if len(loop_indices) <= min_length:
        return "small", -1

    box_size_periodic = array_from_single_or_list(box_size_periodic)
    loop_indices = np.where(box_size_periodic == np.inf, loop_indices, loop_indices % box_size_periodic)


    for i,line in enumerate(lines): # line: one of the old loops. loop: the new loop to be checked.
        line_indices = line._defect_indices[:-1]
        line_indices = np.where(box_size_periodic == np.inf, line_indices, line_indices % box_size_periodic)
        dist = cdist(loop_indices, line_indices)
        if np.min(dist) <= threshold:
            loop_start_index, line_start_index = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            loop_indices_unwrap = np.concatenate( [ loop_indices[loop_start_index:], loop_indices[:loop_start_index] ] )
            line_indices_unwrap = np.concatenate( [ line_indices[line_start_index:], line_indices[:line_start_index] ] )
            loop_indices_unwrap = unwrap_trajectory(loop_indices_unwrap, box_size_periodic=box_size_periodic)
            line_indices_unwrap = unwrap_trajectory(line_indices_unwrap, box_size_periodic=box_size_periodic)
            dist_unwrap = cdist(loop_indices_unwrap, line_indices_unwrap)
            dist_unwrap = np.min(dist_unwrap, axis=1) # for each defect in loop, find the closest distance between this defect and the line
            if np.max(dist_unwrap) > threshold:
                return "mix", i
            else:
                return "old", i
            
    if len(loop_indices) <= min_length:
        return "small", -1
    else:
        return "new", -1
            
            




    


# -----------------------------------------------------
# Specific functions which are being used in my project
# -----------------------------------------------------


@time_record
def example_visualize_defects(lines, is_wrap=True, min_length=50, window_length=61, 
                              opacity=1, radius=0.5,
                              outline_extent=[0,128,0,128,0,128]):
    
    from mayavi import mlab

    lines = [line for line in lines if line._defect_num>min_length]
    lines = sorted(lines, 
                   key=lambda line: line._defect_num,
                   reverse=True)
    color_map = blue_red_in_white_bg()
    color_map_length = np.shape(color_map)[0] - 1
    lines_color = color_map[ (sample_far(len(lines))*color_map_length).astype(int)  ]

    for i, line in enumerate(lines):
        line.update_smoothen(window_length=window_length)
        line.figure_init(tube_color=tuple(lines_color[i]), is_new=1-bool(i), is_wrap=is_wrap,
                         tube_opacity=opacity, tube_radius=radius)
        
    figure = mlab.gcf()
    mlab.outline(figure=figure, color=(0,0,0), extent=outline_extent, line_width=4)
    mlab.view(distance=450)
        
@time_record
def example_visualize_defects_loops_init(lines, is_wrap=True, min_length=30, window_length=61, 
                                         opacity=1, radius=1,
                                         outline_extent=[0,382,0,382,0,382]):
    
    from mayavi import mlab

    lines = [line for line in lines if line._defect_num>min_length]
    lines = sorted(lines, 
                   key=lambda line: line._defect_num,
                   reverse=True)
    
    color_map = blue_red_in_white_bg()
    color_map_length = np.shape(color_map)[0] - 1

    for i, line in enumerate(lines):
        line.update_smoothen(window_length=window_length)
        line.update_norm()
        tube_color = tuple(color_map[int(np.abs(line._norm[0])*color_map_length)])
        line.figure_init(tube_color=tube_color, 
                         is_new=1-bool(i), is_wrap=is_wrap,
                         tube_opacity=opacity, tube_radius=radius)
        
    figure = mlab.gcf()
    mlab.outline(figure=figure, color=(0,0,0), extent=outline_extent, line_width=4) 
    mlab.view(azimuth=90, elevation=90, distance=950, roll=90)


@time_record
def example_visualize_defects_loop_lack(n, is_wrap=True,
                              min_length=40, boundary_periodic=(1,1,1), 
                              cross_window_length=31, loop_window_length=31):

    defect_indices = defect_detect(n, boundary_periodic=boundary_periodic)
    lines = defect_classify_into_lines(defect_indices, np.shape(n)[:3])
    lines = [line for line in lines if line._defect_num>min_length]
    loops = [line for line in lines if line._end2end_category=='loop']
    crosses = [line for line in lines if line._end2end_category=='cross']
    crosses = sorted(crosses, 
                     key=lambda line: line._defect_num,
                     reverse=True)
    color_map = blue_red_in_white_bg()
    color_map_length = np.shape(color_map)[0] - 1
    crosses_color = color_map[ (sample_far(len(crosses))*color_map_length).astype(int)  ]

    for i, cross in enumerate(crosses):
        cross.update_smoothen(window_length=cross_window_length)
        cross.figure_init(tube_color=tuple(crosses_color[i]), is_new=False, is_wrap=is_wrap)

    for i, loop in enumerate(loops):
        loop.update_smoothen(window_length=loop_window_length) 
        loop.figure_init(tube_color=(0,0,0), is_new=False, is_wrap=is_wrap)





# ---------------------------------------
# Functions which are not currently used.
# Just for back up.
# ---------------------------------------


# def find_defect_n_old(defect_indices, size=0):
#     #! defect_indices half integer
#     '''
#     To find the directors enclosing the defects.
#     For example, the defect locates at (10, 5.5, 14.5) in the unit of indices
#     the directors enclosing the defects would locate at:
#     (10, 5, 14)
#     (10, 5, 15)
#     (10, 6, 14)
#     (10, 6, 15)

#     Parameters
#     ----------
#     defect_indices : array, (defect_num, 3)
#                      The array that includes all the indices of defects.
#                      For each defect, one of the indices should be integer and the rest should be half-integer
#                      Usually defect_indices are generated by defect_defect() in this module

#     size : int, or array of three ints
#            The size of the box, which is used to delete the directors outsize the box.
#            Directors outsize the box may appear when the defect is near the wall and periodic boundary condition is applied.
#            Default is 0, interprepted as do not delete any directors  

#     Returns
#     -------
#     defect_n : array, (4*defect_num, 3) 
#                The indices of directors enclosing the defects

#     Dependencies
#     ------------
#     - NumPy: 1.22.0               
#     '''

#     if len(np.shape([size])) == 1:
#         size = np.array([size]*3)
#     else:
#         size = np.array(size)
#     # delete all the defects which will generate directors out of the box
#     if np.sum(size) != 0:
#         defect_indices = defect_indices[np.where(np.all(defect_indices<=(size-1), axis=1))]

#     defect_num = np.shape(defect_indices)[0]

#     defect_n = np.zeros((4*defect_num,3))

#     defect_n0 = np.zeros((defect_num,3))
#     defect_n1 = np.zeros((defect_num,3))
#     defect_n2 = np.zeros((defect_num,3))
#     defect_n3 = np.zeros((defect_num,3))

#     # find the layer of each defect (where the index is integer)
#     cond_layer = defect_indices%1==0

#     # the layer is unchanged
#     defect_n0[cond_layer] = defect_indices[cond_layer]
#     defect_n1[cond_layer] = defect_indices[cond_layer]
#     defect_n2[cond_layer] = defect_indices[cond_layer]
#     defect_n3[cond_layer] = defect_indices[cond_layer]

#     defect_n0[~cond_layer] = defect_indices[~cond_layer] - 0.5
#     defect_n3[~cond_layer] = defect_indices[~cond_layer] + 0.5

#     temp = defect_indices[~cond_layer] - 0.5
#     temp[1::2] = temp[1::2] + 1
#     defect_n1[~cond_layer] = temp

#     temp = defect_indices[~cond_layer] - 0.5
#     temp[:-1:2] = temp[:-1:2] + 1
#     defect_n2[~cond_layer] = temp

#     index_list = np.arange(0, 4*defect_num, 4)
#     defect_n[index_list] = defect_n0
#     defect_n[index_list+1] = defect_n1
#     defect_n[index_list+2] = defect_n2
#     defect_n[index_list+3] = defect_n3
#     defect_n = np.unique(defect_n, axis=0)
#     defect_n = defect_n.astype(int)

#     return defect_n



# def plot_loop(
#             loop_coord, 
#             tube_radius=0.25, tube_opacity=0.5, tube_color=(0.5,0.5,0.5), if_add_head=True,
#             if_norm=False, 
#             norm_coord=[None,None,None], norm_color=(0,0,1), norm_length=20, 
#             norm_opacity=0.5, norm_width=1.0, norm_orient=1,
#             print_load_mayavi=False
#             ):

#     if print_load_mayavi == True:
#         now = time.time()
#         from mayavi import mlab
#         print(f'loading mayavi cost {round(time.time()-now, 2)}s')
#     else:
#         from mayavi import mlab

#     if if_add_head==True:
#         loop_coord = np.concatenate([loop_coord, [loop_coord[0]]])

#     mlab.plot3d(*(loop_coord.T), tube_radius=tube_radius, opacity=tube_opacity, color=tube_color)

#     if if_norm == True:
#         loop_N = get_plane(loop_coord) * norm_orient
#         loop_center = loop_coord.mean(axis=0)
#         for i, coord in enumerate(norm_coord):
#             if coord != None:
#                 loop_center[i] = coord
#         mlab.quiver3d(
#         *(loop_center), *(loop_N),
#         mode='arrow',
#         color=norm_color,
#         scale_factor=norm_length,
#         opacity=norm_opacity,
#         line_width=norm_width
#         ) 


# def plot_loop_from_n(
#                     n_box, 
#                     origin=[0,0,0], N=1, width=1, 
#                     tube_radius=0.25, tube_opacity=0.5, tube_color=(0.5,0.5,0.5), if_add_head=True,
#                     if_smooth=True, window_ratio=3, order=3, N_out=160,
#                     deform_funcs=[None,None,None],
#                     if_norm=False, 
#                     norm_coord=[None,None,None], norm_color=(0,0,1), norm_length=20, 
#                     norm_opacity=0.5, norm_width=1.0, norm_orient=1

#                     ):

#     loop_indices = defect_detect(n_box)
#     if len(loop_indices) > 0:
#         loop_indices = loop_indices + np.tile(origin, (np.shape(loop_indices)[0],1) )
#         loop_coord = sort_line_indices(loop_indices)/N*width
#         for i, func in enumerate(deform_funcs):
#             if func != None:
#                 loop_coord[:,i] = func(loop_coord[:,i])
#         if if_smooth == True:
#             loop_coord = smoothen_line(
#                                     loop_coord,
#                                     window_ratio=window_ratio, order=order, N_out=N_out,
#                                     mode='wrap'
#                                     )
#         plot_loop(
#                 loop_coord, 
#                 tube_radius=tube_radius, tube_opacity=tube_opacity, tube_color=tube_color,
#                 if_add_head=if_add_head,
#                 if_norm=if_norm,
#                 norm_coord=norm_coord, norm_color=norm_color, norm_length=norm_length, 
#                 norm_opacity=norm_opacity, norm_width=norm_width, norm_orient=norm_orient
#                     ) 


# def show_loop_plane(
#                     loop_box_indices, n_whole, 
#                     width=0, margin_ratio=0.6, upper=0, down=0, norm_index=0, 
#                     tube_radius=0.25, tube_opacity=0.5, scale_n=0.5, 
#                     if_smooth=True,
#                     print_load_mayavi=False
#                     ):
    
#     #! Unify origin (index or sapce unit)

#     # Visualize the disclination loop with directors lying on one cross section
    
#     if print_load_mayavi == True:
#         now = time.time()
#         from mayavi import mlab
#         print(f'loading mayavi cost {round(time.time()-now, 2)}s')
#     else:
#         from mayavi import mlab
#     from Nematics3D.field import select_subbox, local_box_diagonalize

#     def SLP_plot_plane(upper, down, d_box, grid, norm_vec, n_box, scale_n):

#         index = (d_box<upper) * (d_box>down)
#         index = np.where(index == True)
#         n_plane = n_box[index]
#         scalars = np.abs(np.einsum('ij, j -> i', n_plane, norm_vec))

#         X, Y, Z = grid
#         cord1 = X[index] - n_plane[:,0]/2
#         cord2 = Y[index] - n_plane[:,1]/2
#         cord3 = Z[index] - n_plane[:,2]/2

#         vector = mlab.quiver3d(
#                 cord1, cord2, cord3,
#                 n_plane[:,0], n_plane[:,1], n_plane[:,2],
#                 mode = '2ddash',
#                 scalars = scalars,
#                 scale_factor=scale_n,
#                 opacity = 1
#                 )
#         vector.glyph.color_mode = 'color_by_scalar'
#         lut_manager = mlab.colorbar(object=vector)
#         lut_manager.data_range=(0,1)

#     N = np.shape(n_whole)[0]
#     if width == 0:
#         width = N

#     # Find the region enclosing the loop. The size of the region is controlled by margin_ratio
#     sl0, sl1, sl2, _ = select_subbox(loop_box_indices, 
#                                 [N, N, N], 
#                                 margin_ratio=margin_ratio
#                                 )

#     # Select the local n around the loop
#     n_box = n_whole[sl0,sl1,sl2]

#     eigvec, eigval = local_box_diagonalize(n_box)

#     # The directors within one cross section of the loop will be shown
#     # Select the cross section by its norm vector
#     # The norm of the principle plane is the eigenvector corresponding to the smallest eigenvalue
#     norm_vec = eigvec[norm_index]

#     # Build the grid for visualization
#     x = np.arange( loop_box_indices[0][0], loop_box_indices[0][-1]+1 )/N*width
#     y = np.arange( loop_box_indices[1][0], loop_box_indices[1][-1]+1 )/N*width
#     z = np.arange( loop_box_indices[2][0], loop_box_indices[2][-1]+1 )/N*width
#     grid = np.meshgrid(x,y,z, indexing='ij')

#     # Find the height of the middle cross section: dmean
#     d_box = np.einsum('iabc, i -> abc', grid, norm_vec)
#     dmean = np.average(d_box)

#     down, upper = np.sort([down, upper])
#     if upper==down:
#         upper = dmean + 0.5
#         down  = dmean - 0.5
#     else:
#         upper = dmean + upper
#         down  = dmean + down

#     mlab.figure(bgcolor=(0,0,0))
#     SLP_plot_plane(upper, down, d_box, grid, norm_vec, n_box, scale_n)
#     plot_loop_from_n(
#                     n_box, 
#                     origin=loop_box_indices[:,0], N=N, width=width,
#                     tube_radius=tube_radius, tube_opacity=tube_opacity,
#                     if_smooth=if_smooth
#                     )

#     return dmean, eigvec, eigval

# # -----------------------------------------------------
# # Visualize the directors projected on principle planes
# # ----------------------------------------------------- 

# def show_plane_2Ddirector(
#                         n_box, height, 
#                         color_axis=(1,0), height_visual=0,
#                         space=3, line_width=2, line_density=1.5, 
#                         if_omega=True, S_box=0, S_threshold=0.18,
#                         if_cb=True, colormap='blue-red',
#                           ):
#     # Visualize the directors projected on principle planes

#     from mayavi import mlab
    
#     from plotdefect import get_streamlines
    
#     color_axis1 = color_axis / np.linalg.norm(color_axis) 
#     color_axis2 = np.cross( np.array([0,0,1]), np.concatenate( [color_axis1,[0]] ) )
#     color_axis2 = color_axis2[:-1]
    
#     x = np.arange(np.shape(n_box)[0])
#     y = np.arange(np.shape(n_box)[1])
#     z = np.arange(np.shape(n_box)[2])

#     indexy = np.arange(0, np.shape(n_box)[1], space)
#     indexz = np.arange(0, np.shape(n_box)[2], space)
#     iny, inz = np.meshgrid(indexy, indexz, indexing='ij')
#     ind = (iny, inz)

#     n_plot = n_box[height]

#     n_plane = np.array( [n_plot[:,:,1][ind], n_plot[:,:,2][ind] ] )
#     n_plane = n_plane / np.linalg.norm( n_plane, axis=-1, keepdims=True)

#     stl = get_streamlines(
#                 y[indexy], z[indexz], 
#                 n_plane[0].transpose(), n_plane[1].transpose(),
#                 density=line_density)
#     stl = np.array(stl)

#     connect_begin = np.where(np.abs( stl[1:,0] - stl[:-1,1]  ).sum(axis=-1) < 1e-5 )[0]
#     connections = np.zeros((len(connect_begin),2))
#     connections[:,0] = connect_begin
#     connections[:,1] = connect_begin + 1

#     lines_index = np.arange(np.shape(stl)[0])
#     disconnect = lines_index[~np.isin(lines_index, connect_begin)]

#     if height_visual == 0:
#         src_x = stl[:, 0, 0] * 0 + height
#     else:
#         src_x = stl[:, 0, 0] * 0 + height_visual
#     src_y = stl[:, 0, 0]
#     src_z = stl[:, 0, 1]

#     unit = stl[1:, 0] - stl[:-1, 0]
#     unit = unit / np.linalg.norm(unit, axis=-1, keepdims=True)

#     coe1 = np.einsum('ij, j -> i', unit, color_axis1)
#     coe2 = np.einsum('ij, j -> i', unit, color_axis2)
#     coe1 = np.concatenate([coe1, [coe1[-1]]])
#     coe2 = np.concatenate([coe2, [coe2[-1]]])
#     colors = np.arctan2(coe1,coe2)
#     nan_index = np.array(np.where(np.isnan(colors)==1))
#     colors[nan_index] = colors[nan_index-1]
#     colors[disconnect] = colors[disconnect-1]

#     src = mlab.pipeline.scalar_scatter(src_x, src_y, src_z, colors)
#     src.mlab_source.dataset.lines = connections
#     src.update()

#     lines = mlab.pipeline.stripper(src)
#     plot_lines = mlab.pipeline.surface(lines, line_width=line_width, colormap='blue-red')

#     if type(colormap) == np.ndarray:
#         lut = plot_lines.module_manager.scalar_lut_manager.lut.table.to_array()
#         lut[:, :3] = colormap
#         plot_lines.module_manager.scalar_lut_manager.lut.table = lut
    

#     if if_cb == True:
#         cb = mlab.colorbar(object=plot_lines, orientation='vertical', nb_labels=5, label_fmt='%.2f')
#         cb.data_range = (0,1)
#         cb.label_text_property.color = (0,0,0)
    

#     if if_omega == True: 

#         from scipy import ndimage

#         binimg = S_box<S_threshold
#         binimg[:height] = False
#         binimg[(height+1):] = False
#         binimg = ndimage.binary_dilation(binimg, iterations=1)
#         if np.sum(binimg) > 0:
#             labels, num_objects = ndimage.label(binimg, structure=np.zeros((3,3,3))+1)
#             if num_objects > 2:
#                 raise NameError('more than two parts')
#             elif num_objects == 1:
#                 print('only one part. Have to seperate points by hand')
#                 cross_coord = np.transpose(np.where(binimg==True))
#                 cross_center = np.mean(cross_coord, axis=0)
#                 cross_relative = cross_coord - np.tile(cross_center, (np.shape(cross_coord)[0],1))
#                 cross_dis = np.sum(cross_relative**2, axis=1)**(1/2)
#                 cross_to0 = cross_coord[cross_dis < np.percentile(cross_dis, 50), :]
#                 binimg[tuple(np.transpose(cross_to0))] = False
#             labels, num_objects = ndimage.label(binimg, structure=np.zeros((3,3,3))+1)
#             for i in range(1,3):  
#                 index = np.where(labels==i)
#                 X, Y, Z = np.meshgrid(x,y,z, indexing='ij')
#                 cord1 = X[index] - n_box[..., 0][index]/2
#                 cord2 = Y[index] - n_box[..., 1][index]/2
#                 cord3 = Z[index] - n_box[..., 2][index]/2
            
#                 pn = np.vstack((n_box[..., 0][index], n_box[..., 1][index], n_box[..., 2][index]))
#                 Omega = get_plane(pn.T)
#                 Omega= Omega * np.sign(Omega[0])
#                 scale_norm = 20
#                 if height_visual == 0:
#                     xvisual = cord1.mean()
#                 else:
#                     xvisual = height_visual
#                 mlab.quiver3d(
#                         xvisual, cord2.mean(), cord3.mean(),
#                         Omega[0], Omega[1], Omega[2],
#                         mode='arrow',
#                         color=(0,1,0),
#                         scale_factor=scale_norm,
#                         opacity=0.5
#                         )


# def show_loop_plane_2Ddirector(
#                                 n_box, S_box,
#                                 height_list, if_omega_list=[1,1,1], plane_list=[1,1,1],
#                                 height_visual_list=0, if_rescale_loop=True,
#                                 figsize=(1920, 1360), bgcolor=(1,1,1), camera_set=0,
#                                 if_norm=True, norm_length=20, norm_orient=1, norm_color=(0,0,1),
#                                 line_width=2, line_density=1.5,
#                                 tube_radius=0.75, tube_opacity=1,
#                                 print_load_mayavi=False, if_cb=True, n_colormap='blue-red'
#                                 ):
    
#     # Visualize the disclination loop with directors projected on several principle planes

#     if height_visual_list == 0:
#         height_visual_list = height_list
#         if_rescale_loop = False
#         parabola = None
#     if if_rescale_loop == True:
#         x, y, z = height_list
#         coe_matrix = np.array([
#                         [x**2, y**2, z**2],
#                         [x, y, z],
#                         [1,1,1]
#                         ])
#         del x, y, z
#         coe_parabola = np.dot(height_visual_list, np.linalg.inv(coe_matrix))
#         def parabola(x):
#             return coe_parabola[0]*x**2 + coe_parabola[1]*x + coe_parabola[2]
#     else:
#         def parabola(x):
#             return x


#     if print_load_mayavi == True:
#         now = time.time()
#         from mayavi import mlab
#         print(f'loading mayavi cost {round(time.time()-now, 2)}s')
#     else:
#         from mayavi import mlab

#     mlab.figure(size=figsize, bgcolor=bgcolor)

#     plot_loop_from_n(n_box, 
#                      tube_radius=tube_radius, tube_opacity=tube_opacity, 
#                      deform_funcs=[parabola,None,None],
#                      if_norm=if_norm,
#                      norm_coord=[height_visual_list[0],None,None], norm_length=norm_length, norm_orient=norm_orient, norm_color=norm_color,
#                      )

#     for i, if_plane in enumerate(plane_list):
#         if if_plane == True:
#             show_plane_2Ddirector(n_box, height_list[i], 
#                                   height_visual=height_visual_list[i], if_omega=if_omega_list[i], 
#                                   line_width=line_width, line_density=line_density,
#                                   S_box=S_box, if_cb=if_cb, colormap=n_colormap)

#     if camera_set != 0: 
#         mlab.view(*camera_set[:3], roll=camera_set[3])

        
# def plot_defect(n, 
#                 origin=[0,0,0], grid=128, width=200,
#                 if_plot_defect=True, defect_threshold=0, defect_color=(0.2,0.2,0.2), scale_defect=2,
#                 plot_n=True, n_interval=1, ratio_n_dist = 5/6,
#                 print_load_mayavi=False
#                 ):
    
#     # Visualize the disclinations within the simulation box

#     if plot_n == False and plot_defect == False:
#         print('no plot')
#         return 0
#     else:
#         if print_load_mayavi == True:
#             now = time.time()
#             from mayavi import mlab
#             print(f'loading mayavi cost {round(time.time()-now, 2)}s')
#         else:
#             from mayavi import mlab
#         mlab.figure(bgcolor=(1,1,1))

#     if plot_n == True:

#         nx = n[:,:,:,0]
#         ny = n[:,:,:,1]
#         nz = n[:,:,:,2]

#         N, M, L = np.shape(n)[:-1]

#         indexx = np.arange(0, N, n_interval)
#         indexy = np.arange(0, M, n_interval)
#         indexz = np.arange(0, L, n_interval)
#         ind = tuple(np.meshgrid(indexx, indexy, indexz, indexing='ij'))
        
#         x = indexx / grid * width + origin[0]
#         y = indexy / grid * width + origin[1]
#         z = indexz / grid * width + origin[2]
#         X, Y, Z = np.meshgrid(x,y,z, indexing='ij')

#         distance = n_interval / grid * width
#         n_length = distance * ratio_n_dist

#         coordx = X - nx[ind] * n_length / 2
#         coordy = Y - ny[ind] * n_length / 2
#         coordz = Z - nz[ind] * n_length / 2

#         phi = np.arccos(nx[ind])
#         theta = np.arctan2(nz[ind], ny[ind])

#         vector = mlab.quiver3d(
#                                 coordx, coordy, coordz,
#                                 nx[ind],ny[ind],nz[ind],
#                                 mode='cylinder',
#                                 scalars = (1-np.cos(2*phi))*(np.sin(theta%np.pi)+0.3),
#                                 scale_factor=n_length
#                                 )
        
#         vector.glyph.color_mode = 'color_by_scalar'
        
#         lut_manager = mlab.colorbar(object=vector)
#         lut_manager.data_range=(0,1.3)

#     if if_plot_defect == True:

#         defect_indices = defect_detect(n, threshold=defect_threshold)
#         defect_coords  = defect_indices / grid * width
#         defect_coords = defect_coords + origin
#         mlab.points3d(*(defect_coords.T), color=defect_color, scale_factor=scale_defect)









      

# def ordered_bulk_size(defect_indices, N, width, if_print=True):
#     '''
#     Compute the minimum distance from each point in a 3D grid the neirhboring defects.

#     Parameters
#     ----------
#     defect_indices : numpy.ndarray, shape (N,3)
#                      Array containing the coordinates of defect points. 
#                      M is the number of defect points.

#     N : int
#         Size of the cubic 3D grid in each dimension.

#     width : float
#             Width of the simulation box in the unit of real length (not indices).

#     if_print : bool, optional
#               Flag to print the time taken for each octant.
#               Default is True.

#     Returns
#     -------
#     dist_min : numpy.ndarray, shape (N^3,)
#                Array containing the minimum distances from each point in the 3D grid to the nearest defect.

#     Dependencies
#     ------------
#     - numpy: 1.22.0
#     - scipy: 1.7.3
#     '''
#     from itertools import product
#     import time

#     from scipy.spatial.distance import cdist

#     # Ensure defect indices are within the periodic boundary conditions
#     defect_indices = defect_indices % N

#     # Generate the coordinates of each point in the 3D grid.
#     grid = np.array(list(product(np.arange(N), np.arange(N), np.arange(N))))

#     # Divide defect and grid points into octants based on their positions in each dimension
#     defect_check0 = defect_indices[:,0] < int(N/2)
#     defect_check1 = defect_indices[:,1] < int(N/2)
#     defect_check2 = defect_indices[:,2] < int(N/2)

#     defect_box = [defect_indices[ np.where(  defect_check0 *  defect_check1 *  defect_check2 ) ],
#                   defect_indices[ np.where( ~defect_check0 *  defect_check1 *  defect_check2 ) ],
#                   defect_indices[ np.where(  defect_check0 * ~defect_check1 *  defect_check2 ) ],
#                   defect_indices[ np.where(  defect_check0 *  defect_check1 * ~defect_check2 ) ],
#                   defect_indices[ np.where( ~defect_check0 * ~defect_check1 *  defect_check2 ) ],
#                   defect_indices[ np.where( ~defect_check0 *  defect_check1 * ~defect_check2 ) ],
#                   defect_indices[ np.where(  defect_check0 * ~defect_check1 * ~defect_check2 ) ],
#                   defect_indices[ np.where( ~defect_check0 * ~defect_check1 * ~defect_check2 ) ]]

#     grid_check0 = grid[:,0] < int(N/2)
#     grid_check1 = grid[:,1] < int(N/2)
#     grid_check2 = grid[:,2] < int(N/2)

#     grid_box = []
#     grid_box = [grid[ np.where(  grid_check0 *  grid_check1 *  grid_check2 ) ],
#                 grid[ np.where( ~grid_check0 *  grid_check1 *  grid_check2 ) ],
#                 grid[ np.where(  grid_check0 * ~grid_check1 *  grid_check2 ) ],
#                 grid[ np.where(  grid_check0 *  grid_check1 * ~grid_check2 ) ],
#                 grid[ np.where( ~grid_check0 * ~grid_check1 *  grid_check2 ) ],
#                 grid[ np.where( ~grid_check0 *  grid_check1 * ~grid_check2 ) ],
#                 grid[ np.where(  grid_check0 * ~grid_check1 * ~grid_check2 ) ],
#                 grid[ np.where( ~grid_check0 * ~grid_check1 * ~grid_check2 ) ]]

#     # To same memory, consider half of grid points in each octant for each calculation
#     size = len(grid_box[0])
#     half = int(len(grid_box[0])/2)

#     dist_min = np.zeros(N**3)

#     # Calculate!
#     for i, box in enumerate(grid_box):
#         start = time.time()
#         dist_min[ i*size:i*size+half ] = np.min(cdist(box[:half], defect_box[i]), axis=-1)
#         dist_min[ i*size+half:(i+1)*size ] = np.min(cdist(box[half:], defect_box[i]), axis=-1)
#         print(time.time()-start)

#     # Change the unit of distances into real length,    
#     dist_min = dist_min / N * width

#     return dist_min



# def save_xyz(fname, atoms, lattice=None):
#     """
#     Saves data to a file in extended XYZ format. This function will attempt to
#     produce an extended XYZ file even if the input data is incomplete.
#     Usually used to store information of disclination loops.
#     Created by Matthew E. Peterson.

#     Parameters
#     ----------
#     fname : str or pathlib.Path
#             File to write data to.

#     atoms : pandas.DataFrame
#             Atomic data to be saved.

#     lattice : array_like (default None)
#               Lattice vectors of the system, so that lattice[0] is the first lattice vector.

#     Raises
#     ------
#     KeyError
#         If a key in `cols` does not appear in `data`.
#     """

#     import re
#     import os
#     import gzip
#     import shutil

#     gzipped = fname.endswith('.gz')
#     if gzipped:
#         fname = fname[:-3]

#     def map_type(dtype):
#         if np.issubdtype(dtype, np.integer):
#             return 'I'
#         elif np.issubdtype(dtype, np.floating):
#             return 'R'
#         else:
#             return 'S'

#     def collapse_names(names):
#         cols = dict()
#         for name in names:
#             col = re.sub(r'-\w+$', '', name)
#             try:
#                 cols[col].append(name)
#             except:
#                 cols[col] = [name]
#         return cols

#     with open(fname, 'w') as f:
#         # the first line is simply the number of atoms
#         f.write(f"{len(atoms)}\n")

#         # write lattice if given
#         header = ""
#         if lattice is not None:
#             lattice = np.asarray(lattice)
#             lattice = ' '.join(str(x) for x in lattice.ravel())
#             header += f'Lattice="{lattice}" '

#         # now determine property line
#         cols = collapse_names(atoms.columns)
#         props=[]
#         for col, names in cols.items():
#             name = names[0] 
#             tp=map_type(atoms.dtypes[name])
#             dim=len(names)
#             props.append(f"{col}:{tp}:{dim}")
#         props = ':'.join(props)
#         header += f'Properties={props}\n'

#         f.write(header)

#         # writing the atomic data is easy provided we are using a DataFrame
#         atoms.to_csv(f, header=None, sep=" ", float_format="%f", index=False)

#     if gzipped:
#         with open(fname, 'rb') as f_in:
#             with gzip.open(f"{fname}.gz", 'wb') as f_out:
#                 shutil.copyfileobj(f_in, f_out)

#         os.remove(fname)


# def extract_loop(defect_indices, N, dilate_times=2,
#                  save_path='.', save_head=''):
    
#     '''
#     Given the defect indices in 3D grids, extract loops and calculate their number of holes.
#     Using binary dilation and Euler number calculation. 
#     Saves loop information and coordinates in files.

#     Parameters
#     ----------
#     defect_indices : array_like, (M, 3)
#                      Array containing the indices of defects in the 3D grid.
#                      M is the number of defects, and each row contains (x, y, z) indices.

#     N : int
#         Size of the 3D grid along each dimension.

#     dilate_times : int, optional
#                    Number of iterations for binary dilation to thicken the disclination lines.
#                    Default is 2.

#     save_path : str, optional
#                 Path to save loop information files. 
#                 Default is current directory.

#     save_head : str, optional
#                 Prefix for saved files. 
#                 Default is an empty string.

#     Saves
#     -----
#     'summary.xyz.gz' : summary, pandas file
#                        The information of each point of thickened disclination loops.
#                        Include x, y, z coordinates (indices), genus and label of this loop.

#     'summary_wrap.xyz.gz' : summary_wrap, pandas file
#                             Same with summary except the coordinates are wrapped in periodic boundary condition

#     '{lable}box_{g}.txt' : box, array-like, (3,2)
#                            The vortices of box containing one thickened disclination loop.
#                            g = genus of this loop

#     '{lable}index_{g}.txt' : coord, array-like, (M,3)
#                              The indices of each point in one thickened disclination loop.
#                              g = genus of this loop, M is the number of points in the loop.

#     'grid_g.npy' : grid_result, array-like, (N,N,N)
#                    The defects indices and genus represented by grid of box.
#                    N is the size of the 3D grid along each dimension.
#                    If one point is 0, there is no defect here.
#                    If one point is x (not zero), there is a defect here and the genus of this line is x-1.
    
#     Dependencies
#     ------------
#     - numpy : 1.22.0
#     - scipy : 1,7.3
#     - skimage : 0.19.1
#     - pandas : 2.0.2
#     '''


#     from scipy import ndimage
#     from skimage.measure import euler_number
#     import pandas as pd

#     # Ensure save_path ends with '/'
#     save_path = save_path + '/'

#     # Wrap the defect
#     defect_indices = (defect_indices%N).astype(int)

#     # Create a binary grid representing the defect indices
#     grid_origin = np.zeros((N, N, N))
#     grid_origin[tuple(defect_indices.T)] = 1

#     # Expand the defect grid to handle periodic boundary conditions
#     grid = np.zeros( (2*N, 2*N, 2*N ) )
#     grid[:N, :N, :N]            = grid_origin
#     grid[N:2*N, :N, :N]         = grid_origin
#     grid[:N, N:2*N, :N]         = grid_origin
#     grid[:N, :N, N:2*N]         = grid_origin
#     grid[N:2*N, N:2*N, :N]      = grid_origin
#     grid[:N, N:2*N, N:2*N]      = grid_origin
#     grid[N:2*N, :N, N:2*N]      = grid_origin
#     grid[N:2*N, N:2*N, N:2*N]   = grid_origin

#     # Perform binary dilation on the extended grid to thicken the lines
#     binimg = ndimage.binary_dilation(grid, iterations=dilate_times)

#     # Count the number of low-order points
#     num_pts = int(np.count_nonzero(binimg)/8)
#     print(f"Found {num_pts} low-order points")

#     # Prepare the array to record genus of each defect
#     grid_origin = grid_origin/2
#     grid_result = np.zeros((N, N, N))

#     # Initialize arrays to store loop information
#     summary = np.empty(8*num_pts,
#                        dtype=[
#                            ("pos-x", "uint16"),
#                            ("pos-y", "uint16"),
#                            ("pos-z", "uint16"),
#                            ("genus", "int16"),
#                            ("label", "uint16"),
#                        ]
#                        )
    
#     summary_wrap = np.empty(8*num_pts,
#                        dtype=[
#                            ("pos-x", "uint16"),
#                            ("pos-y", "uint16"),
#                            ("pos-z", "uint16"),
#                            ("genus", "int16"),
#                            ("label", "uint16"),
#                        ]
#                        )
    
#     # Label connected components in the binary grid
#     labels, num_objects = ndimage.label(binimg)

#     offset = 0
#     label = 0
#     loop = 0
#     for i, obj in enumerate(ndimage.find_objects(labels)):
        
#         # Ensure object boundaries are within the extended grid
#         xlo = max(obj[0].start, 0)
#         ylo = max(obj[1].start, 0)
#         zlo = max(obj[2].start, 0)
#         xhi = min(obj[0].stop, 2*N)
#         yhi = min(obj[1].stop, 2*N)
#         zhi = min(obj[2].stop, 2*N)
#         boundary = [xlo, ylo, zlo, xhi, yhi, zhi]
        
#         # Exclude the crosses and loops outside of the box and  
#         if (0 in boundary) or (2*N in boundary) or max(xlo, ylo, zlo)>N:
#             continue
        
#         label += 1
        
#         # Define the object within the extended grid
#         obj = (slice(xlo, xhi), slice(ylo, yhi), slice(zlo, zhi))
        
#         # Extract the object from the labeled grid
#         img = (labels[obj] == i+1)
    
#         # calculate Euler number, defined as # objects + # holes - # loops
#         # we do not have holes
#         g = euler_number(img, connectivity=1)

#         # calculate the number of loop by Euler number
#         g = 1 - g
        
#         # box : The vortices of box containing the loop
#         # coord : The indices of each point of thickened loop
#         pos = np.nonzero(img)
#         shift = pos[0].size
        
#         box = np.array([[xlo, xhi], [ylo, yhi], [zlo, zhi]], dtype=int)
#         coord = np.array([pos[0] + xlo, pos[1] + ylo, pos[2] + zlo])

#         np.savetxt(save_path + save_head + f"{label}box_{g}.txt", box, fmt="%d")
#         np.savetxt(save_path + save_head + f"{label}index_{g}.txt", coord, fmt="%d")

#         if g == 1:
#             loop += 1

#         # Update summary arrays with loop information
#         summary['pos-x'][offset:offset+shift] = coord[0]
#         summary['pos-y'][offset:offset+shift] = coord[1]
#         summary['pos-z'][offset:offset+shift] = coord[2]
#         summary['genus'][offset:offset+shift] = g
#         summary['label'][offset:offset+shift] = label
        
#         summary_wrap['pos-x'][offset:offset+shift] = coord[0]%N
#         summary_wrap['pos-y'][offset:offset+shift] = coord[1]%N
#         summary_wrap['pos-z'][offset:offset+shift] = coord[2]%N
#         summary_wrap['genus'][offset:offset+shift] = g
#         summary_wrap['label'][offset:offset+shift] = label
    
#         offset += shift

#         # Update the genus information of each defect
#         grid_result[tuple(coord%N)] = g+1
        
#     # grid_result : each point of defect labeled by the genus of lines that the defect belongs to.
#     grid_result = grid_result + grid_origin
#     grid_result[grid_result%1==0] = 0.5
#     grid_result = grid_result - 0.5

#     np.save(save_path + save_head + "grid_g.npy", grid_result)
    
#     # Trim summary arrays to remove unused space
#     summary = summary[:offset]
#     summary_wrap = summary_wrap[:offset]
#     summary = pd.DataFrame(summary)
#     summary_wrap = pd.DataFrame(summary_wrap)

#     save_xyz(save_path + save_head + "summary.xyz.gz", summary)
#     save_xyz(save_path + save_head + "summary_wrap.xyz.gz", summary_wrap)
                    
#     print(f'Found {loop} loops\n')


