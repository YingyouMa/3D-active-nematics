import numpy as np
import time

def trans_period_test1(n, N):
    if n == 0:
        return N
    elif n == N-1:
        return -1
    
def check_find_test1(defect_here, defect_reservoir, defect_group=0,  box_size=np.inf):
    ''' 
    To find if defect_group contains one defect neighboring to defect_here.
    The periodic boundary condition could be put into consideration.
    If several different neighboring defects are found in defect_reservoir, only one of them will be returned.

    Parameters
    ----------
    defect_here : numpy array, (3,)
                  the indices of the defect, provided by defect_detect().
                  One of the index must be integer, representing the layer,
                  and the other two indices must be half-integer, representing the center of one pixel in this layer
                  Supposing defect_here = (layer, center1, center2), where layer is integer while center1 and center2 are half-integers,
                  the set of all the possible neighboring defects is
                  (layer+-1,     center1,        center2)
                  (layer+-0.5,   center1+-0.5,   center2)
                  (layer+-0.5,   center1,        center2+-0.5)
                  here +- means plusminus, and the order is unneccessary as (+,+), (-,+), (+,-), (-,-) are all possible
                  so there are 2+4+4=10 possible neighboring defects
                  We use scipy.spatial.distance.cdist to find the neighboring defects, with metric='sqeuclidean'.
                  If there exist neighboring defects, the distance will be 0.5**2 + 0.5**2 = 0.5 or 1.
                  If the distance is 0.5, it must be the neighboring defect because (layer, conter1+-0.5, center2+-0.5) is not possible in defect_reservoir.
                  If the distance is 1, we should check if the difference comes from the layer, as (layer, conter1+-1, center2) is possible in defect_reservoir but it's not neighboring defect.

    defect_reservoir : numpy array, N x 3,
                       The indices of other defects.
                       The function will try to find if there is one defect in defect_reservoir such that the defect is neighboring to one of the defects in defect_group.
                       Provided by defect_detect().
                       The indices of each defect should have the same structure with defect_here, as one integer and two half-integers.

    defect_group : numpy array, M x 3,
                   The indices of the current defect and its ghost defects. In total there are M defects.
                   The ghost defects come from the periodic boundary condition.     
                   If there is no periodic boundary condition or defect_here is not near the boundary,
                   defect_group should only contain defect_here
                   Default is 0, where defect_group is interprepted as [defect_here]

    box_size : int or numpy array of three ints
               The largest index of the entire box in each dimension.
               Used for periodic boundary condition.
               If box_size is one integer as x, it is interprepted as (x,x,x).
               Default is np.inf, where the periodic boundary condition is not considered

    Return
    ------
    if_find : bool
              Whether find one neighboring defect

    defect_next :  array of three ints:
                   The indices of the neighboring defect

    defect_ordinal_next : int
                          The ordinal of the neighboring defect in defect_reservoir,
                          such that defect_reservoir[defect_ordinal_next] = defect_next
    '''

    from scipy.spatial.distance import cdist
    
    if len(np.shape([box_size])) == 1:
        box_size = (box_size, box_size, box_size)
    if len(np.shape([defect_group])) == 1:
        defect_group = [defect_here]
    
    defect_plane_axis = np.where( ( defect_here % 1) == 0 )[0][0] # find the integer index, as the layer

    dist = cdist(defect_group, defect_reservoir, metric='sqeuclidean') # find the distance between each defect in defect_group and each defect in defect_reservoir
    defect_where0 = np.where(dist == 0.5)[1] # If there exist dist==0.5 between one defect in defect_group and one defect in defect_reservoir, this defect in defect_reservoir is the neighboring defect
    defect_where1 = np.where(dist == 1)[1]
    defect_where = 1*defect_where0 # If so, make sure the difference comes from the layer (from layer to layer+-1)
    if len(defect_where1) > 0:
        for defect_ordinal_next in defect_where1:
            defect_next = defect_reservoir[defect_ordinal_next]
            defect_diff = defect_next - defect_here
            # check in which axis the difference is 1 or size-1, where size is the largest index in the axis of layer
            # if the periodic boundary condition is not considered, size-1 will be -1, which will be automatically omiited due to np.abs()
            if defect_plane_axis == (np.where( (np.abs(defect_diff) == 1) + (np.abs(defect_diff) == box_size[defect_plane_axis]-1) ))[0][0]:
                defect_where = np.concatenate([defect_where, [defect_ordinal_next]])
        
    return defect_where

def find_box_test1(value, length_list):
    
    cumulative_sum = 0
    position = 0

    for i, num in enumerate(length_list):
        cumulative_sum += num
        if cumulative_sum-1 >= value:
            position = i
            break

    index = value - (cumulative_sum - length_list[position])
    return (position, index)

def defect_index_find_box_test1(defect_index, box_size):
    # the stupid way to find the box which contains the defect
    x = defect_index[0] < int(box_size[0]/2)
    y = defect_index[1] < int(box_size[1]/2)
    z = defect_index[2] < int(box_size[2]/2)

    if x == 1 and y == 1 and z == 1:
        return 0
    elif x != 1 and y == 1 and z == 1:
        return 1
    elif x == 1 and y != 1 and z == 1:
        return 2
    elif x == 1 and y == 1 and z != 1:
        return 3
    elif x != 1 and y != 1 and z == 1:
        return 4
    elif x != 1 and y == 1 and z != 1:
        return 5
    elif x == 1 and y != 1 and z != 1:
        return 6
    elif x != 1 and y != 1 and z != 1:
        return 7

def defect_connected_test1(defect_indices, box_size=np.inf, print_time=False, print_per=1000):
    
    #! N_index for 3 different axes
    #! It only works with boundary=True in defect_detect() if defects cross walls
    #! no periodic

    """
    Classify defects into different lines.

    Parameters
    ----------
    defect_indices : numpy array, num_defects x 3
                     Represents the locations of defects in the grid.
                     For each location, there must be one integer (the index of plane) and two half-integers (the center of the loop on that plane)
                     This is usually given by defect_detect()

    
    
    """
    from scipy.spatial.distance import cdist

    if len(np.shape([box_size])) == 1:
        box_size = np.array([box_size, box_size, box_size])
    else:
        box_size = np.array(box_size)
    
    defect_indices = np.array(defect_indices)
    defect_num = len(defect_indices)
    defect_left_num = defect_num # the amount of unclassified defects. Initially it is the number of all defects

    # We start each disclination line at the wall, so that we firstly select the defects at the wall as the start point.
    # If the periodic boundary condition is considered, there might be no need to worry about it.
    # To the opposite, if the periodic boundary condition is NOT considered, the disclination line must start at the wall,
    # because here the cross line does NOT move back to the start point.
    # Let's elaborate it. Imagine a line crossing the wall. If we do NOT start at the wall but start in the middle, the line will end at the wall.
    # The rest of the line will turn to be another line. In other words, here the line will be splitted into two lines.

    if_wall = (defect_indices[:,0]==0) | \
              (defect_indices[:,1]==0) | \
              (defect_indices[:,2]==0) | \
              (defect_indices[:,0]==box_size[0]-1) | \
              (defect_indices[:,1]==box_size[1]-1) | \
              (defect_indices[:,2]==box_size[2]-1)

    # For each defect, except the three coorinates, let's give it some other properties for convenience:
    # the index of defects through all the defects
    # if it is at the wall 

    # In the output, we will still have the coordinates of all defects (sorted by lines), but there will also be:
    # the index of the line that this defect belongs to
    # the index of the defect within the line

    defect_indices = np.hstack([defect_indices, np.zeros((defect_num,2))])
    defect_indices[..., -2] = np.arange(defect_num)
    defect_indices[if_wall, -1] = True

    defect_sorted = np.zeros((defect_num, 5))
    defect_sorted[..., -2] = -1
    defect_sorted[..., -1] = -1

    # to divide the defects into 9 subboxes to accelerate
    # the first eight subboxes correspond to the 8 octants,
    # the last one is the "shell", including the wall and planes of axes
    check00 = defect_indices[:,0] < int(box_size[0]/2)  
    check01 = defect_indices[:,0] > int(box_size[0]/2)
    check10 = defect_indices[:,1] < int(box_size[1]/2)  
    check11 = defect_indices[:,1] > int(box_size[1]/2)
    check20 = defect_indices[:,2] < int(box_size[2]/2)  
    check21 = defect_indices[:,2] > int(box_size[2]/2)
    check3  = (defect_indices[:,0] == int(box_size[0]/2)) | \
            (defect_indices[:,1] == int(box_size[1]/2)) | \
            (defect_indices[:,2] == int(box_size[2]/2))


    defect_box = []
    defect_box.append( defect_indices[ np.where( check00 & check10 & check20 & ~if_wall ) ] )
    defect_box.append( defect_indices[ np.where( check01 & check10 & check20 & ~if_wall ) ] )
    defect_box.append( defect_indices[ np.where( check00 & check11 & check20 & ~if_wall ) ] )
    defect_box.append( defect_indices[ np.where( check00 & check10 & check21 & ~if_wall ) ] )
    defect_box.append( defect_indices[ np.where( check00 & check11 & check21 & ~if_wall ) ] )
    defect_box.append( defect_indices[ np.where( check01 & check10 & check21 & ~if_wall ) ] )
    defect_box.append( defect_indices[ np.where( check01 & check11 & check20 & ~if_wall ) ] )
    defect_box.append( defect_indices[ np.where( check01 & check11 & check21 & ~if_wall ) ] )
    defect_box.append( defect_indices[ np.where( check3 | if_wall  ) ] )

    
    start = time.time()
    start_here = time.time()

    # loop when there are still unclassfied defects
    index_line = -1 # representing the index of the new line
    index = -1 # how many defects have been classfied
    while defect_left_num > 0:
        
        # to start to find a new discliantion line
        cross_wall = np.array([0,0,0]) # the array recording that how many time have the line crossed wach wall due to periodic boudnary condition
        index_here = 0 # representing the index of the next defect in the new line
        index_line += 1
        index += 1

        # To see if there are still defects at the wall.
        # If so, start the new line at such defect.
        # defect_box_here is the subbox containing the start defect
        # defect_ordinal_here is the index of the current defect in the subbox
        if np.sum(defect_box[-1][:,-1]>0):
            box_index = 8
            defect_box_here = defect_box[box_index] 
            defect_ordinal_here = np.argmax(defect_box_here[..., -1]>0)
        else:
            box_index = next((i for i, box in enumerate(defect_box) if len(box)>0), None) # select the box which still contains unclassfied defects
            defect_box_here = defect_box[box_index]
            defect_ordinal_here = 0

        # defect here is the current defect, and now it's the defect that start a new line
        defect_here = defect_box_here[defect_ordinal_here] 
        defect_sorted[index, :3] = defect_here[:3]
        defect_sorted[index, 3] = index_line
        defect_sorted[index, 4] = 0

        # Whether to check the previous step
        if_history = False 

        # Once start a line, try to find neighboring defects until there is no any
        while True:
            
            # # update the defect and subbox in the loop
            # defect_ordinal = defect_ordinal_next
            # defect_box_here = 1*defect_box[box_index]
            # defect_here = defect_box_here[defect_ordinal]
            # defect_here_loc = defect_here[:3]
            # defect_box[box_index] = np.vstack(( defect_box_here[:defect_ordinal], defect_box_here[defect_ordinal+1:] ))
            # defect_box_here = 1*defect_box[box_index]
            
            # Array of the locations of the current defect and its ghost defects coming from periodic boudnary
            # Initially there is no ghost defect since we are searching within the subbox
            defect_here_loc = defect_here[:3]
            defect_group = np.array([defect_here_loc])

            if_find = False

            defect_plane_axis = np.where( ( defect_here_loc % 1) == 0 )[0][0]
            layer = defect_here_loc[defect_plane_axis]

            # If the defect is in one of octants (in the first 8 subboxes), 
            # try to find the next defect in the current subbox as well as the shell
            if box_index != 8 and len(defect_box_here) > 0:
                defect_reservoir_here = np.concatenate([defect_box_here[:,:3], defect_box[-1][:,:3]])
                defect_index_next = check_find_test1(defect_here_loc, defect_reservoir_here)
                if len(defect_index_next) > 0:
                    if_find = True
                    defect_where_next = np.zeros((len(defect_index_next,2)))
                    defect_where_next[:,1] = defect_index_next % len(defect_box_here)
                    defect_box_next = defect_index_next // len(defect_box_here)
                    defect_box_next[defect_box_next>0] = 8
                    defect_where_next[:,0] = defect_box_next
                    
            # If no defect is found in this octant, or the previous defect is in the shell,
            # the next defect should be searched throughout the entire box
            if if_find == False:
                defect_box_all = np.concatenate([box for box in defect_box])

                # Now let's consider the periodic boundary condition
                # We need to generate ghost defects if the defect is on the boundary
                if box_size[defect_plane_axis] != np.inf and (layer==0 or layer==box_size[defect_plane_axis]-1):
                    defect_bound = 1*defect_here_loc
                    defect_bound[defect_plane_axis] = trans_period_test1(layer, box_size[defect_plane_axis])
                    defect_group = np.concatenate([defect_group, [defect_bound]])
                bound_half = np.where(defect_here_loc==(box_size-0.5))[0]
                if len(bound_half) > 0:
                    for bound in bound_half:
                        defect_bound = 1*defect_here_loc
                        defect_bound[bound] = -0.5
                        defect_group = np.concatenate([defect_group, [defect_bound]])
                    if box_size[defect_plane_axis] != np.inf and (layer==0 or layer==box_size[defect_plane_axis]-1):
                        for bound in bound_half:
                            defect_bound = 1*defect_here_loc
                            defect_bound[bound] = -0.5
                            defect_bound[defect_plane_axis] = trans_period_test1(layer, box_size[defect_plane_axis])
                            defect_group = np.concatenate([defect_group, [defect_bound]])
                        
                defect_index_next = check_find_test1(defect_here_loc, defect_box_all[:,:3], defect_group=defect_group, box_size=box_size)
                if len(defect_index_next) > 0:
                    if_find = True
                    defect_where_next = np.zeros((len(defect_index_next,2)))
                    for i, item in enumerate(defect_index_next):
                        defect_where_next[i] = find_box_test1(item, [len(term) for term in defect_box])

                if if_find == True:
                    #! reverse
                    #! history
                    #! multi
                    # If we find the next defect ...

                    


        







                if if_find == True:
                    box_index, defect_ordinal_next = find_box1(defect_ordinal_next, [len(term) for term in defect_box])
                
            if if_find == True:
                # If we find the next defect, store the data
                defect_diff = defect_next_loc - defect_here_loc
                cross_wall_here = np.trunc( defect_diff / (box_size-10) ) #! this 10
                cross_wall = cross_wall - cross_wall_here
                defect_next_loc = defect_next_loc + cross_wall * box_size
                index_here += 1
                index += 1
                defect_sorted[index, :3] = defect_next_loc
                defect_sorted[index, -2] = index_line
                defect_sorted[index, -1] = index_here
                defect_box_here = defect_box[box_index]
                if print_time == True:
                    if index % print_per == 0:
                        print(f'{index}/{defect_num} = {round(index/defect_num*100,2)}%, {round(time.time()-start_here,2)}s  ',
                            f'{round(time.time()-start,2)}s in total' )
                        start_here= time.time()
            else:
                defect_left_num = 0
                for term in defect_box:
                    defect_left_num += len(term)
                break
    
    return defect_sorted