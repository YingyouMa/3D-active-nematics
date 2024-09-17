# def visual_loops_genus(lines_origin, N, grid_g, 
#                        window_loop=21, N_ratio_loop=1.5, radius=2,
#                        if_lines_added=False, wrap=True, 
#                        color_list=[]):
    
#     '''

#     #! Color and grid_g seperately

#     Visualize loops using Mayavi.
#     Loops are colored according to their genus.

#     Given the director field n, visualize the disclination loops by the genus as the following:

#     - Detect the defects (defect_indices) in the director field by defect_detect(n, boundary=True)

#     - Find the thickened disclination loops and their genus by extract_loop(defect_indices, N)
#       It will provide grid_g, labelling each defect with the genus of the line that this defect belongs to.
    
#     - Sort the defects into different lines by lines = defect_connected(defect_indices, N)
#       Need this step to smoothen the lines.

#     - visual_loops_genus(lines, N, grid_g)

#     Parameters
#     ----------
#     lines : list of arrays
#             List of disclination lines, where each line is represented as an array of coordinates.
#             Usually provided by defect_connected()

#     N : int
#         Size of the grid in each dimension.

#     grid_g : numpy.ndarray, shape (N, N, N)
#              The defects indices and genus represented by grid of box.
#              N is the size of the 3D grid along each dimension.
#              If one point is 0, there is no defect here.
#              If one point is x (not zero), there is a defect here and the genus of this line is x-1.
#              Usually provided by extract_loop().

#     window_loop : int, optional
#                   Window length for smoothening the loops using Savitzky-Golay filter.
#                   Default is 21.

#     N_ratio_loop : float, optional
#                    Ratio to determine the number of points in the output smoothened loop.
#                    N_out = N_ratio_loop * len(loop).
#                    Default is 1.5.

#     radius : int, optional
#              Scale factor for visualizing the loops.
#              Default is 2.

#     if_lines_added : bool, optional
#                      Flag indicating whether midpoints have been added to disclination lines.
#                      Default is False.

#     wrap : bool, optional
#            Flag indicating whether to wrap the loops in visualization
#            Default is True.

#     color_list : array-like, optional, shape (M,)
#                  The color for each disclination loop. Each value belongs to [0,1]
#                  The colormap is derived by blue_red_in_white_bg() in the same module.
#                  0 for blue, 1 for red.
#                  If the length of color_list is 0, the loops are colored by genus.
#                  Default is [] (loops colored by genus)


#     Dependencies
#     ------------
#     - mayavi: 4.7.4
#     - scipy: 1.7.3
#     - numpy: 1.22.0
#     '''

#     from mayavi import mlab
#     from scipy.spatial.distance import cdist

#     # If midpoints are not added to disclination lines, add them
#     lines = 1 * lines_origin
#     if if_lines_added == False:
#         for i, line in enumerate(lines):
#             lines[i] = add_mid_points_disclination(line)

#     # Initialize an array to store genus information for each line
#     genus_lines = np.zeros(len(lines))

#     # Get coordinates of defects in the grid
#     defect_g = np.array(np.nonzero(grid_g)).T

#     # Find genus for each line based on the proximity of its head to defect points
#     for i, line in enumerate(lines):
#         head = line[0]%N
#         dist = cdist([head], defect_g)
#         if np.min(dist) > 1:
#             genus_lines[i] = -1
#         else:
#             genus_lines[i] = grid_g[tuple(defect_g[np.argmin(dist)])] - 1

#     lines = np.array(lines, dtype=object)
#     loops = lines[genus_lines>-1]
#     genus_loops = genus_lines[genus_lines > -1]

#     mlab.figure(bgcolor=(1,1,1))

#     colormap = blue_red_in_white_bg()

#     for i, loop in enumerate(loops):
#         genus = genus_loops[i]
#         if len(color_list) == 0:
#             if genus == 0:
#                 color = (0,0,1)
#             elif genus == 1:
#                 color = (1,0,0)
#             elif genus > 1:
#                 color= (0,0,0)
#         else:
#             color = colormap[int(color_list[i]*510)]
#         loop = smoothen_line(loop, window_length=window_loop, mode='wrap', 
#                              N_out=int(N_ratio_loop*len(loop)))
        
#         if wrap == True:
#             loop = loop%N

#         mlab.points3d(*(loop.T), scale_factor=radius, color=tuple(color))








# def check_find_old(defect_here, defect_group, defect_box, N):

#     from scipy.spatial.distance import cdist
    
#     defect_plane_axis = np.where( ( defect_here % 1) == 0 )[0][0]

#     if_find = False

#     dist = cdist(defect_group, defect_box, metric='sqeuclidean')
#     defect_where = np.where(dist == 0.5)[1]
#     if len(defect_where) == 0:
#         defect_where = np.where(dist == 1)[1]
#         if len(defect_where) > 0:
#             for item in defect_where:
#                 defect_ordinal_next = item
#                 defect_next = defect_box[defect_ordinal_next]
#                 defect_diff = defect_next - defect_here
#                 if defect_plane_axis == (np.where( (np.abs(defect_diff) == 1) + (np.abs(defect_diff) == N-1) ))[0][0]:
#                     if_find = True
#                     break
#     else:
#         if_find = True
#         defect_ordinal_next = defect_where[0]
#         defect_next = defect_box[defect_ordinal_next]
        
#     if if_find == False:
#         defect_ordinal_next, defect_next = None, None
        
#     return if_find, defect_ordinal_next, defect_next

# def defect_connected_old(defect_indices, N, print_time=False, print_per=1000):
    
#     #! N_index for 3 different axes

#     """
#     Classify defects into different lines.

#     Parameters
#     ----------
#     defect_indices : numpy array, num_defects x 3
#                      Represents the locations of defects in the grid.
#                      For each location, there must be one integer (the index of plane) and two half-integers (the center of the loop on that plane)
#                      This is usually given by defect_detect()

    
    
#     """

#     #! It only works with boundary=True in defect_detect() if defects cross walls

#     index = 0
#     defect_num= len(defect_indices)
#     defect_left_num = defect_num
    
#     lines = []
    
#     check0 = defect_indices[:,0] < int(N/2)
#     check1 = defect_indices[:,1] < int(N/2)
#     check2 = defect_indices[:,2] < int(N/2)
    
#     defect_box = []
#     defect_box.append( defect_indices[ np.where( check0 * check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( check0 * ~check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( check0 * check1 * ~check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * ~check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * check1 * ~check2 ) ] )
#     defect_box.append( defect_indices[ np.where( check0 * ~check1 * ~check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * ~check1 * ~check2 ) ] )
    
#     start = time.time()
#     start_here = time.time()

#     while defect_left_num > 0:
        
#         loop_here = np.zeros( ( len(defect_indices),3 ) )
#         defect_ordinal_next = 0
#         cross_wall = np.array([0,0,0])
        
#         box_index = next((i for i, box in enumerate(defect_box) if len(box)>0), None)
#         defect_box_here = defect_box[box_index]
#         loop_here[0] = defect_box_here[0]
        
#         index_here = 0
        
#         while True:
            
#             defect_ordinal = defect_ordinal_next
#             defect_box_here = 1*defect_box[box_index]
#             defect_here = defect_box_here[defect_ordinal]
#             defect_box[box_index] = np.vstack(( defect_box_here[:defect_ordinal], defect_box_here[defect_ordinal+1:] ))
#             defect_box_here = 1*defect_box[box_index]
            
#             defect_group = np.array([defect_here])

#             if_find = False
            
#             if len(defect_box_here) > 0:
#                 if_find, defect_ordinal_next, defect_next = check_find_old(defect_here, defect_group, defect_box_here, N)

#             if if_find == False or len(defect_box_here) == 0:
                
#                 defect_box_all = np.concatenate([box for box in defect_box])
            
#                 bound_0 = np.where( (defect_here==0) + (defect_here==N-1) )[0]
#                 if len(bound_0) > 0:
#                     defect_bound = 1*defect_here
#                     defect_bound[bound_0[0]] = trans_period(defect_bound[bound_0[0]],N)
#                     defect_group = np.concatenate([defect_group, [defect_bound]])
#                 bound_1 = np.where(defect_here==(N-0.5))[0]
#                 for bound in bound_1:
#                     defect_bound = 1*defect_here
#                     defect_bound[bound] = -0.5
#                     defect_group = np.concatenate([defect_group, [defect_bound]])
#                 if len(bound_0) > 0 and len(bound_1) > 0:
#                     for bound in bound_1:
#                         defect_bound = 1*defect_here
#                         defect_bound[bound] = -0.5
#                         defect_bound[bound_0[0]] = trans_period(defect_bound[bound_0[0]],N)
#                         defect_group = np.concatenate([defect_group, [defect_bound]])
                        
#                 if_find, defect_ordinal_next, defect_next = check_find_old(defect_here, defect_group, defect_box_all, N)
#                 if if_find == True:
#                     box_index, defect_ordinal_next = find_box(defect_ordinal_next, [len(term) for term in defect_box])
                
#             if if_find == True:
#                 defect_diff = defect_next - defect_here
#                 cross_wall_here = np.trunc( defect_diff / (N-10) )
#                 cross_wall = cross_wall - cross_wall_here
#                 defect_next = defect_next + cross_wall * N
#                 loop_here[index_here+1] = defect_next
#                 index += 1
#                 index_here += 1
#                 if print_time == True:
#                     if index % print_per == 0:
#                         print(f'{index}/{defect_num} = {round(index/defect_num*100,2)}%, {round(time.time()-start_here,2)}s  ',
#                             f'{round(time.time()-start,2)}s in total' )
#                         start_here= time.time()
#             else:
#                 zero_loc = np.where(np.all([0,0,0] == loop_here, axis=1))[0]
#                 if len(zero_loc) > 0:
#                     loop_here = loop_here[:zero_loc[0]]
#                 lines.append(loop_here)
#                 defect_left_num = 0
#                 for term in defect_box:
#                     defect_left_num += len(term)
#                 break
    
#     return lines

# def lines_connect_closed(lines_list, box_size):

#     from .field import get_ghost_point
#     from itertools import product

#     closed_list = []
#     wait_list = []
#     closed_no_list=[]

#     for line in lines_list:
#         if is_defects_connected( line[0], line[-1]):
#             closed_list.append(line)
#         else:
#             point1 = get_ghost_point(line[0]%box_size, box_size)
#             point2 = get_ghost_point(line[-1]%box_size, box_size)
#             wait_list.append([line, [point1, point2]])

#     while len(wait_list)>1:
#         wait_line = wait_list[0]
#         for count in (0, 1):
#             for i in range(len(wait_list)-1,0,-1):
#                 test_line = wait_list[i]
#                 point_start = wait_line[1][count]
#                 for j in (0,1):
#                     if_find = False
#                     for point1, point2 in product(point_start, test_line[1][j]):
#                         #print(point1, point2)
#                         if is_defects_connected(point1, point2):
#                             print('found')
#                             if_find = True
#                             cross_box = (wait_line[0][-count] - test_line[0][-j])//box_size
#                             to_add = test_line[0] + cross_box * box_size
#                             if count==0 and j==1:
#                                 wait_list[0][0] = np.vstack([to_add, wait_line[0]])
#                                 wait_list[0][1][0] = test_line[1][0]
#                             if count==1 and j==0:
#                                 wait_list[0][0] = np.vstack([wait_line[0], to_add])
#                                 wait_list[0][1][1] = test_line[1][1]    
#                             wait_list.pop(i)
#                             break
#                     if if_find == True:
#                         break

# def defect_classify_into_lines_init(defect_indices, box_size, print_time=False, print_per=1000):
    
#     #! It only works with boundary=True in defect_detect() if defects cross walls
#     #! box_size, when defects are not throughout the whole box but gather in small region
#     #! set box_size to be optional value

#     """
#     The first step to classify defects into different lines.

#     Parameters
#     ----------
#     defect_indices : numpy array, (num_defects, 3)
#                      Represents the locations of defects in the grid.
#                      For each location, there must be one integer (the index of plane) and two half-integers (the center of the loop on that plane)
#                      This is usually given by defect_detect()

#     box_size : float or numpy of three floats
#                The largest index of the entire box in each dimension.
#                Used for periodic boundary condition and sub-box-division for acceleration.
#                If box_size is one integer as x, it is interprepted as (x,x,x).

#     Returns
#     -------
#     defect_sorted : (num_defects, 5)
#                     The defects classified into different disclination lines.
#                     For each defect, the first three values are the indices.
#                     The fourth one is the index of the line that this defect belongs to.
#                     The fifth one is the index of this defect in the line that this defect belongs to.

#     Dependencies
#     ------------
#     - scipy: 1.7.3
#     - numpy: 1.22.0 

#     Called by
#     ---------
#     - Disclination_line  

#     Internal functions
#     -------------------------
#     - trans_period()
#     - check_find()
#     - find_box()
#     """

#     # Begin internal functions
#     # ------------------------

#     def trans_period(n, N):
#         if n == 0:
#             return N
#         elif n == N-1:
#             return -1
    
#     def check_find(defect_here, defect_reservoir, defect_group=0,  box_size=0):
#         ''' 
#         To find if defect_group contains one defect neighboring to defect_here.
#         The periodic boundary condition could be put into consideration.
#         If several different neighboring defects are found in defect_reservoir, only one of them will be returned.

#         Parameters
#         ----------
#         defect_here : numpy array, (3,)
#                       the indices of the defect, provided by defect_detect().
#                       One of the index must be integer, representing the layer,
#                       and the other two indices must be half-integer, representing the center of one pixel in this layer
#                       Supposing defect_here = (layer, center1, center2), where layer is integer while center1 and center2 are half-integers,
#                       the set of all the possible neighboring defects is
#                       (layer+-1,     center1,        center2)
#                       (layer+-0.5,   center1+-0.5,   center2)
#                       (layer+-0.5,   center1,        center2+-0.5)
#                       here +- means plusminus, and the order is unneccessary as (+,+), (-,+), (+,-), (-,-) are all possible
#                       so there are 2+4+4=10 possible neighboring defects
#                       We use scipy.spatial.distance.cdist to find the neighboring defects, with metric='sqeuclidean'.
#                       If there exist neighboring defects, the distance will be 0.5**2 + 0.5**2 = 0.5 or 1.
#                       If the distance is 0.5, it must be the neighboring defect because (layer, conter1+-0.5, center2+-0.5) is not possible in defect_reservoir.
#                       If the distance is 1, we should check if the difference comes from the layer, as (layer, conter1+-1, center2) is possible in defect_reservoir but it's not neighboring defect.

#         defect_reservoir : numpy array, N x 3,
#                            The indices of other defects.
#                            The function will try to find if there is one defect in defect_reservoir such that the defect is neighboring to one of the defects in defect_group.
#                            Provided by defect_detect().
#                            The indices of each defect should have the same structure with defect_here, as one integer and two half-integers.

#         defect_group : numpy array, M x 3,
#                        The indices of the current defect and its ghost defects. In total there are M defects.
#                        The ghost defects come from the periodic boundary condition.     
#                        If there is no periodic boundary condition or defect_here is not near the boundary,
#                        defect_group should only contain defect_here
#                        Default is 0, where defect_group is interprepted as [defect_here]

#         box_size : int or numpy array of three ints
#                    The largest index of the entire box in each dimension.
#                    Used for periodic boundary condition.
#                    If box_size is one integer as x, it is interprepted as (x,x,x).
#                    Default is 0, where the periodic boundary condition is not considered

#         Return
#         ------
#         if_find : bool
#                 Whether find one neighboring defect

#         defect_next :  array of three ints:
#                     The indices of the neighboring defect

#         defect_ordinal_next : int
#                             The ordinal of the neighboring defect in defect_reservoir,
#                             such that defect_reservoir[defect_ordinal_next] = defect_next
#         '''

#         from scipy.spatial.distance import cdist
        
#         if len(np.shape([box_size])) == 1:
#             box_size = (box_size, box_size, box_size)
#         if len(np.shape([defect_group])) == 1:
#             defect_group = [defect_here]
        
#         defect_plane_axis = np.where( ( defect_here % 1) == 0 )[0][0] # find the integer index, as the layer

#         if_find = False

#         dist = cdist(defect_group, defect_reservoir, metric='sqeuclidean') # find the distance between each defect in defect_group and each defect in defect_reservoir
#         defect_where = np.where(dist == 0.5)[1] # If there exist dist==0.5 between one defect in defect_group and one defect in defect_reservoir, this defect in defect_reservoir is the neighboring defect
#         if len(defect_where) == 0: # If there is no dist==0.5, check if there is dist==1.
#             defect_where = np.where(dist == 1)[1] # If so, make sure the difference comes from the layer (from layer to layer+-1)
#             if len(defect_where) > 0:
#                 for item in defect_where:
#                     defect_ordinal_next = item
#                     defect_next = defect_reservoir[defect_ordinal_next]
#                     defect_diff = defect_next - defect_here
#                     # check in which axis the difference is 1 or size-1, where size is the largest index in the axis of layer
#                     # if the periodic boundary condition is not considered, size-1 will be -1, which will be automatically omiited due to np.abs()
#                     if defect_plane_axis == (np.where( (np.abs(defect_diff) == 1) + (np.abs(defect_diff) == box_size[defect_plane_axis]-1) ))[0][0]:
#                         if_find = True
#                         break
#         else:
#             if_find = True
#             defect_ordinal_next = defect_where[0]
#             defect_next = defect_reservoir[defect_ordinal_next]
            
#         if if_find == False:
#             defect_ordinal_next, defect_next = None, None
            
#         return if_find, defect_next, defect_ordinal_next

#     def find_box(value, length_list):
        
#         cumulative_sum = 0
#         position = 0

#         for i, num in enumerate(length_list):
#             cumulative_sum += num
#             if cumulative_sum-1 >= value:
#                 position = i
#                 break

#         index = value - (cumulative_sum - length_list[position])
#         return (position, index)


#     # Begin main function
#     # -------------------

#     from scipy.spatial.distance import cdist

#     if len(np.shape([box_size])) == 1:
#         box_size = np.array([box_size, box_size, box_size])
#     else:
#         box_size = np.array(box_size)
    
#     defect_indices = np.array(defect_indices)
#     defect_num = len(defect_indices)
#     defect_left_num = defect_num # the amount of unclassified defects. Initially it is the number of all defects

#     # We start each disclination line at the wall, so that we firstly select the defects at the wall as the start point.
#     # If the periodic boundary condition is considered, there might be no need to worry about it.
#     # To the opposite, if the periodic boundary condition is NOT considered, the disclination line must start at the wall,
#     # because here the cross line does NOT move back to the start point.
#     # Let's elaborate it. Imagine a line crossing the wall. If we do NOT start at the wall but start in the middle, the line will end at the wall.
#     # The rest of the line will turn to be another line. In other words, here the line will be splitted into two lines.

#     if_wall = (defect_indices[:,0]==0) | \
#               (defect_indices[:,1]==0) | \
#               (defect_indices[:,2]==0) | \
#               (defect_indices[:,0]==box_size[0]-1) | \
#               (defect_indices[:,1]==box_size[1]-1) | \
#               (defect_indices[:,2]==box_size[2]-1)

#     # For each defect, except the three coorinates, let's give it some other properties for convenience:
#     # the index of defects through all the defects
#     # if it is at the wall 

#     # In the output, we will still have the coordinates of all defects (sorted by lines), but there will also be:
#     # the index of the line that this defect belongs to
#     # the index of the defect within the line

#     defect_indices = np.hstack([defect_indices, np.zeros((defect_num,2))])
#     defect_indices[..., -2] = np.arange(defect_num)
#     defect_indices[if_wall, -1] = True

#     defect_sorted = np.zeros((defect_num, 5))
#     defect_sorted[..., -2] = -1
#     defect_sorted[..., -1] = -1

#     # to divide the defects into 8 subboxes to accelerate
#     check0 = defect_indices[:,0] < int(box_size[0]/2)
#     check1 = defect_indices[:,1] < int(box_size[1]/2)
#     check2 = defect_indices[:,2] < int(box_size[2]/2)

#     defect_box = []
#     defect_box.append( defect_indices[ np.where( check0 * check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( check0 * ~check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( check0 * check1 * ~check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * ~check1 * check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * check1 * ~check2 ) ] )
#     defect_box.append( defect_indices[ np.where( check0 * ~check1 * ~check2 ) ] )
#     defect_box.append( defect_indices[ np.where( ~check0 * ~check1 * ~check2 ) ] )
    
#     start = time.time()
#     start_here = time.time()

#     # loop when there are still unclassfied defects
#     index_line = -1 # representing the index of the new line
#     index = -1 # how many defects have been classfied
#     while defect_left_num > 0:
        
#         # to start to find a new discliantion line
#         cross_wall = np.array([0,0,0]) # the array recording that how many time have the line crossed wach wall due to periodic boudnary condition
#         index_here = 0 # representing the index of the next defect in the new line
#         index_line += 1
#         index += 1

#         # To see if there are still defects at the wall.
#         # If so, start the new line at such defect.
#         box_index = next((i for i, box in enumerate(defect_box) if np.sum(box[..., -1])>0), None)
        
#         # defect_box_here is the subbox containing the start defect
#         # defect_ordinal_next is the index of the new found defect in the subbox
#         if box_index != None:
#             defect_box_here = defect_box[box_index]
#             defect_ordinal_next = np.argmax(defect_box_here[..., -1]>0)
#         else:
#             box_index = next((i for i, box in enumerate(defect_box) if len(box)>0), None) # select the box which still contains unclassfied defects
#             defect_box_here = defect_box[box_index]
#             defect_ordinal_next = 0

#         defect_start = defect_box_here[defect_ordinal_next] # the defect that start a new line
#         defect_sorted[index, :3] = defect_start[:3]
#         defect_sorted[index, 3] = index_line
#         defect_sorted[index, 4] = 0

#         # Once start a line, try to find neighboring defects until there is no any
#         while True:
            
#             # update the defect and subbox in the loop
#             defect_ordinal = defect_ordinal_next
#             defect_box_here = 1*defect_box[box_index]
#             defect_here = defect_box_here[defect_ordinal]
#             defect_here_loc = defect_here[:3]
#             defect_box[box_index] = np.vstack(( defect_box_here[:defect_ordinal], defect_box_here[defect_ordinal+1:] ))
#             defect_box_here = 1*defect_box[box_index]
            
#             # Array of the locations of the current defect and its ghost defects coming from periodic boudnary
#             # Initially there is no ghost defect since we are searching within the subbox
#             defect_group = np.array([defect_here_loc])

#             if_find = False

#             defect_plane_axis = np.where( ( defect_here_loc % 1) == 0 )[0][0]
#             layer = defect_here_loc[defect_plane_axis]
              
#             # At first, try to find neighboring defect in the current subbox.
#             # Thus the periodic boundary condition is ignored here.
#             if len(defect_box_here) > 0:
#                 if_find, defect_next_loc, defect_ordinal_next = check_find(defect_here_loc, defect_box_here[:,:3])

#             # If there is no neighboring defect in the current subbox, expand the searching area to the entire box
#             if if_find == False or len(defect_box_here) == 0:
                
#                 defect_box_all = np.concatenate([box for box in defect_box])

#                 # Now let's consider the periodic boundary condition
#                 # We need to generate ghost defects if the defect is on the boundary
#                 if layer==0 or layer==box_size[defect_plane_axis]-1:
#                     defect_bound = 1*defect_here_loc
#                     defect_bound[defect_plane_axis] = trans_period(layer, box_size[defect_plane_axis])
#                     defect_group = np.concatenate([defect_group, [defect_bound]])
#                 bound_half = np.where(defect_here_loc==(box_size-0.5))[0]
#                 if len(bound_half) > 0:
#                     for bound in bound_half:
#                         defect_bound = 1*defect_here_loc
#                         defect_bound[bound] = -0.5
#                         defect_group = np.concatenate([defect_group, [defect_bound]])
#                     if layer==0 or layer==box_size[defect_plane_axis]-1:
#                         for bound in bound_half:
#                             defect_bound = 1*defect_here_loc
#                             defect_bound[bound] = -0.5
#                             defect_bound[defect_plane_axis] = trans_period(layer, box_size[defect_plane_axis])
#                             defect_group = np.concatenate([defect_group, [defect_bound]])
                        
#                 if_find, defect_next_loc, defect_ordinal_next = check_find(defect_here_loc, defect_box_all[:,:3], defect_group=defect_group, box_size=box_size)
#                 if if_find == True:
#                     box_index, defect_ordinal_next = find_box(defect_ordinal_next, [len(term) for term in defect_box])
                
#             if if_find == True:
#                 # If we find the next defect, store the data
#                 defect_diff = defect_next_loc - defect_here_loc
#                 cross_wall_here = np.trunc( defect_diff / (box_size-10) ) #! this 10
#                 cross_wall = cross_wall - cross_wall_here
#                 defect_next_loc = defect_next_loc + cross_wall * box_size
#                 index_here += 1
#                 index += 1
#                 defect_sorted[index, :3] = defect_next_loc
#                 defect_sorted[index, -2] = index_line
#                 defect_sorted[index, -1] = index_here
#                 defect_box_here = defect_box[box_index]
#                 if print_time == True:
#                     if index % print_per == 0:
#                         print(f'{index}/{defect_num} = {round(index/defect_num*100,2)}%, {round(time.time()-start_here,2)}s  ',
#                             f'{round(time.time()-start,2)}s in total' )
#                         start_here= time.time()
#             else:
#                 defect_left_num = 0
#                 for term in defect_box:
#                     defect_left_num += len(term)
#                 break
    
#     return defect_sorted, box_size


# def extract_lines(defect_sorted, box_size, is_add_mid=True):
#     '''
#     Extract the disclination lines into several instancies of class Disclination_line

#     Parameters
#     ----------
#     defect_sorted : numpy array, (num_defects, >=4)
#                     The information of each defect.
#                     For each defect, the first three values are the indices.
#                     The fourth one is the index of the line that this defect belongs to.

#     Returns
#     -------
#     lines : list,
#             The list of all the lines.
#             Each line is an instance of Disclination_line.
#     '''

#     from .classes.disclination_line import DisclinationLine

#     defect_sorted = defect_sorted[:, :4]

#     line_start = np.array(np.where(defect_sorted[1:,3] - defect_sorted[:-1,3] == 1)[0]) + 1
#     line_start = np.concatenate([[0], line_start])
#     line_start = np.concatenate([line_start, [np.shape(defect_sorted)[0]]])

#     line_num = int(defect_sorted[-1, 3] + 1)
#     lines = []
#     for i in range(line_num):
#         lines.append(DisclinationLine(defect_sorted[ line_start[i]:line_start[i+1] , :3],
#                                       box_size,
#                                       is_add_mid=is_add_mid))

#     return lines


# def defect_classify_into_lines1(defect_indices, box_size, print_time=False, print_per=1000):
#     '''
#     To classify defects into different lines.

#     Parameters
#     ----------
#     To read the document in defect_classify_into_lines_init() please.

#     Returns
#     -------
#     lines : list,
#             The list of all the lines.
#             Each line is an instance of Disclination_line.
#     '''
#     defect_sorted, box_size = defect_classify_into_lines_init(defect_indices, box_size, 
#                                                               print_time=print_time, print_per=print_per)
#     lines = extract_lines(defect_sorted, box_size)

#     return lines

# def visual_disclinations(lines, N, N_index=1, min_length=30, radius=0.5,
#                          window_cross=33, window_loop=21,
#                          N_ratio_cross=1, N_ratio_loop=1.5,
#                          loop_color=(0,0,0), cross_color=None,
#                          wrap=True, if_lines_added=False,
#                          new_figure=True, bgcolor=(1,1,1)):
#     #! plot3d to make lines as tubes
#     #! axes
#     #! N_index
#     #! different size, and f p p
#     '''
#     Visualize disclination lines in 3D using Mayavi.

#     Parameters
#     ----------
#     lines : list of arrays
#             List of disclination lines, where each line is represented as an array of coordinates.

#     N_index : int or array of 3 ints
#               Size of the cubic 3D grid in each dimension.
#               If only one int x is input, it is interpreted as (x,x,x)

#     min_length : int, optional
#                  Minimum length of disclination loops to consider. 
#                  Length calculated as the number of points conposing the loop-type disclinations.
#                  Default is 30.

#     radius : float, optional
#              Radius of the points representing the disclination lines in the visualization. 
#              Default is 0.5.

#     window_cross : int, optional
#                    Window length for Savitzky-Golay filter for cross-type disclinations. 
#                    Default is 33.

#     window_loop : int, optional
#                   Window length for Savitzky-Golay filter for loop-type disclinations. 
#                   Default is 21.

#     N_ratio_cross : float, optional
#                     Ratio to compute the number of points in the output smoothened line for crosses.
#                     N_out = N_ratio_cross * num_points_in_cross
#                     Default is 1.

#     N_ratio_loop : float, optional
#                    Ratio to compute the number of points in the output smoothened line for loops.
#                    N_out = N_ratio_loop * num_points_in_loop
#                    Default is 1.5.

#     loop_color : array like, shape (3,)  or None, optional
#                  The color of visualized loops in RGB.
#                  If None, the loops will follow same colormap with crosses.
#                  Default is (0,0,0)

#     wrap : bool, optional
#            If wrap the lines with periodic boundary condition.
#            Default is True

#     if_lines_added : bool, optional
#                      If the lines have already been added the mid-points
#                      Default is False.

#     new_figure : bool, optional
#                  If True, create a new figure for the plot. 
#                  Default is True.

#     bgcolor : array of three floats, optional
#               Background color of the plot in RGB.
#               Default is (0, 0, 0), white.

#     Returns
#     -------
#     None

#     Dependencies
#     ------------
#     - mayavi: 4.7.4
#     - numpy: 1.22.0

#     Functions in same module
#     ------------------------
#     - add_mid_points_disclination: Add mid-points to disclination lines.
#     - smoothen_line: Smoothen disclination lines.
#     - sample_far: Create a sequence where each number is trying to be far away from previous numbers.
#     - blue_red_in_white_bg: A colormap based on blue-red while dinstinct on white backgroud.    
#     '''

#     from mayavi import mlab

#     '''
#     if len(np.shape([N_index])) == 1:
#         N_index = (N_index, N_index, N_index)
#     '''

#     # Filter out short disclination lines
#     lines = np.array(lines, dtype=object)
#     lines = lines[ [len(line)>=min_length for line in lines] ] 

#     # Add mid-points to each disclination line
#     if if_lines_added == False:
#         for i, line in enumerate(lines):
#             lines[i] = add_mid_points_disclination(line)

#     # Separate lines into loops and crosses based on end-to-end distance and smoothen them.
#     loops = []
#     crosses = []
#     for i, line in enumerate(lines):
#         end_to_end = np.sum( (line[-1]-line[0])**2, axis=-1 )
#         if end_to_end > 2:
#             cross = smoothen_line(line, window_length=window_cross, 
#                                     N_out=int(N_ratio_cross * len(line)))
#             crosses.append(cross)
#         else:
#             loop = smoothen_line(line, window_length=window_loop, mode='wrap', 
#                                     N_out=int(N_ratio_loop*len(line)))
#             loops.append(loop)

#     # wrap the discliantions with periodic boundary conditions
#     if wrap == True:
#         loops = np.array(loops, dtype=object)%N
#         crosses = np.array(crosses, dtype=object)%N

#     # Generate a new figure with the given background color if needed
#     if new_figure==True:
#         mlab.figure(bgcolor=bgcolor) 

#     # Generate colormap
#     colormap = blue_red_in_white_bg()

#     if cross_color == None:
#         # Sort crosses by length for better visualization if the color is not set
#         crosses = np.array(crosses, dtype=object)
#         cross_length = [len(cross) for cross in crosses]
#         crosses = crosses[np.argsort(cross_length)[-1::-1]]
#     if loop_color == None:
#         loops = np.array(loops, dtype=object)
#         loop_length = [len(loop) for loop in loops]
#         loops = loops[np.argsort(loop_length)[-1::-1]]
#     if cross_color == None and loop_color == None:
#         color_index = ( sample_far(len(lines)) * 510 ).astype(int)
#         for i,cross in enumerate(crosses):
#             mlab.points3d(*(cross.T), scale_factor=radius, color=tuple(colormap[color_index[i]])) 
#         for j, loop in enumerate(loops):
#             mlab.points3d(*(loop.T), scale_factor=radius, color=tuple(colormap[color_index[len(crosses)+j]]))
#     elif cross_color != None and loop_color == None:
#         color_index = ( sample_far(len(loops)) * 510 ).astype(int)
#         for i,cross in enumerate(crosses):
#             mlab.points3d(*(cross.T), scale_factor=radius, color=cross_color) 
#         for j, loop in enumerate(loops):
#             mlab.points3d(*(loop.T), scale_factor=radius, color=tuple(colormap[color_index[j]]))
#     elif cross_color == None and loop_color != None:
#         color_index = ( sample_far(len(crosses)) * 510 ).astype(int)
#         for i,cross in enumerate(crosses):
#             print(i)
#             # return cross
#             mlab.points3d(*(cross.T), scale_factor=radius, color=tuple(colormap[color_index[i]])) 
#             #mlab.plot3d(*(cross.T), tube_radius=radius, color=tuple(colormap[color_index[i]]))
#         for j, loop in enumerate(loops):
#             mlab.points3d(*(loop.T), scale_factor=radius, color=loop_color)
#     else:
#         for i,cross in enumerate(crosses):
#             mlab.points3d(*(cross.T), scale_factor=radius, color=cross_color) 
#             #mlab.plot3d(*(cross.T), tube_radius=radius, color=cross_color)
#         for j, loop in enumerate(loops):
#             mlab.points3d(*(loop.T), scale_factor=radius, color=loop_color)  