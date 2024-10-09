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
                  is_boundary_periodic=0, planes=[1,1,1], print_time=False, return_test=False):
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

    is_boundary_periodic : bool, or array of three bools, optional
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

    is_boundary_periodic = array_from_single_or_list(is_boundary_periodic)

    from .field import add_periodic_boundary

    # Consider the periodic boundary condition
    n = add_periodic_boundary(n_origin, is_boundary_periodic=is_boundary_periodic)

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
    for i, if_periodic in enumerate(is_boundary_periodic):
        if if_periodic == True:
            defect_indices[:,i] = defect_indices[:,i] % np.shape(n_origin)[i]
    defect_indices, unique = np.unique(defect_indices, axis=0, return_index=True)
    test_result = test_result[unique]

    if return_test:
        return defect_indices, test_result, test_result_all
    else:
        return defect_indices
    

def defect_vinicity_grid(defect_indices, num_shell=2):

    square_size_list = np.arange(1, 2*num_shell+1, 2)
    square_num_list  = square_size_list + 1

    square_origin_list = np.arange(-0.5, -num_shell-0.5, -1)
    square_origin_list = np.broadcast_to(square_origin_list, (2,num_shell)).T
    square_origin_list = np.hstack([ np.zeros((num_shell, 1)), square_origin_list ])

    length = 4 * num_shell**2

    result = np.zeros( (np.shape(defect_indices)[0], length, 3) )

    indexx = np.isclose(defect_indices[:, 0], np.round(defect_indices[:, 0]))
    indexy = np.isclose(defect_indices[:, 1], np.round(defect_indices[:, 1]))
    indexz = np.isclose(defect_indices[:, 2], np.round(defect_indices[:, 2]))

    defectx = defect_indices[indexx]
    defecty = defect_indices[indexy]
    defectz = defect_indices[indexz]

    squarex = get_square(square_size_list, square_num_list, origin_list=square_origin_list , dim=3)
    squarey = squarex.copy()
    squarey[:, [0, 1]] = squarey[:, [1, 0]]
    squarez = squarex.copy()
    squarez[:, [0, 1]] = squarez[:, [1, 0]]
    squarez[:, [1, 2]] = squarez[:, [2, 1]]

    defectx = np.repeat(defectx, length, axis=0).reshape(np.shape(defectx)[0],length,3)
    defecty = np.repeat(defecty, length, axis=0).reshape(np.shape(defecty)[0],length,3)
    defectz = np.repeat(defectz, length, axis=0).reshape(np.shape(defectz)[0],length,3)

    defectx =  defectx + np.broadcast_to(squarex, (np.shape(defectx)[0], length,3))
    defecty =  defecty + np.broadcast_to(squarey, (np.shape(defecty)[0], length,3))
    defectz =  defectz + np.broadcast_to(squarez, (np.shape(defectz)[0], length,3))

    result[indexx] = defectx
    result[indexy] = defecty
    result[indexz] = defectz

    result = result.astype(int)

    return result


def defect_rotation(defect_indices, n, 
                    num_shell=1, box_size_periodic=[np.inf, np.inf, np.inf],
                    method='cross'):

    box_size_periodic = array_from_single_or_list(box_size_periodic)

    vic_grid = defect_vinicity_grid(defect_indices, num_shell=num_shell)
    vic_grid_wrap = np.where(box_size_periodic == np.inf, vic_grid, vic_grid%box_size_periodic)
    
    vic_n = n[*tuple(vic_grid_wrap.T)].transpose((1,0,2))

    if method == 'plane':
        Omega = get_plane(vic_n)
    elif method == 'cross':
        Omega = get_rotation_axis(vic_n)
    else: "method should be rather cross or plane"

    return Omega
   

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

    is_boundary_periodic = box_size_periodic!=np.inf
    defect1 = np.array(defect1)
    defect2 = np.array(defect2)
    defect1[is_boundary_periodic] = defect1[is_boundary_periodic] % box_size_periodic[is_boundary_periodic]
    defect2[is_boundary_periodic] = defect2[is_boundary_periodic] % box_size_periodic[is_boundary_periodic]
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
                threshold=4, box_size_periodic=[np.inf, np.inf, np.inf]):

    from scipy.spatial.distance import cdist
    from .field import unwrap_trajectory

    if len(lines) == 0:
        return "new", -1

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
                              min_length=40, is_boundary_periodic=(1,1,1), 
                              cross_window_length=31, loop_window_length=31):

    defect_indices = defect_detect(n, is_boundary_periodic=is_boundary_periodic)
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

