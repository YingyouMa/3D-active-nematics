# -----------------------------------------------
# Basic detection and analysis of disclinations
# Yingyou Ma, Physics @ Brandeis, 2023
# -----------------------------------------------

import numpy as np
import time

def defect_detect(n_origin, threshold=0, boundary_periodic=False, print_time=False):
    '''
    #! Introduce the format of 

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

    boundary_periodic : bool, optional
                        Flag to indicate whether to consider periodic boundaries. 
                        Default is False.

    print_time : bool, optional
                 Flag to print the time taken for each direction. 
                 Default is False.

    Returns
    -------
    defect_indices : numpy.ndarray, defect_num x 3
                     Array containing the indices of detected defects.
                     According to our algorithm, for each defect's location, there must be one inteter and two half-integers.

    Dependencies
    ------------
    - NumPy: 1.22.0
    '''

    # Consider the periodic boundary condition
    if not boundary_periodic:
        n = n_origin
    else:
        N, M, L = np.shape(n_origin)[:-1]
        n = np.zeros((N+1,M+1,L+1,3))
        n[:N, :M, :L] = n_origin
        n[N, :M, :L] = n[0, :M, :L]
        n[:, M, :L] = n[:, 0, :L]
        n[:,:,L] = n[:,:,0]

    now = time.time()

    # X-direction
    here = n[:, 1:, :-1]
    if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:, :-1, :-1], here))
    here = np.einsum('lmn, lmni -> lmni',if_parallel, n[:, 1:, :-1])
    if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:, 1:, 1:], here))
    here = np.einsum('lmn, lmni -> lmni',if_parallel, n[:, 1:, 1:])
    if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:, :-1, 1:], here))
    here = np.einsum('lmn, lmni -> lmni',if_parallel, n[:, :-1, 1:])
    test = np.einsum('lmni, lmni -> lmn', n[:, :-1, :-1], here)
    defect_indices = np.array(np.where(test<threshold)).transpose().astype(float)
    defect_indices[:,1:] = defect_indices[:,1:]+0.5
    if print_time:
        print('finish x-direction, with', str(round(time.time()-now,2))+'s')
    now = time.time()

    # Y-direction
    here = n[1:, :, :-1]
    if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:-1,:, :-1], here))
    here = np.einsum('lmn, lmni -> lmni',if_parallel, n[1:, :, :-1])
    if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[1:, :, 1:], here))
    here = np.einsum('lmn, lmni -> lmni',if_parallel, n[1:, :, 1:])
    if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:-1, :, 1:], here))
    here = np.einsum('lmn, lmni -> lmni',if_parallel, n[:-1, :, 1:])
    test = np.einsum('lmni, lmni -> lmn', n[:-1, :, :-1], here)
    temp = np.array(np.where(test<threshold)).transpose().astype(float)
    temp[:, [0,2]] = temp[:, [0,2]]+0.5
    defect_indices = np.concatenate([ defect_indices, temp ])
    if print_time:
        print('finish y-direction, with', str(round(time.time()-now,2))+'s')
    now = time.time()

    # Z-direction
    here = n[1:, :-1]
    if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:-1, :-1], here))
    here = np.einsum('lmn, lmni -> lmni',if_parallel, n[1:, :-1])
    if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[1:, 1:], here))
    here = np.einsum('lmn, lmni -> lmni',if_parallel, n[1:, 1:])
    if_parallel = np.sign(np.einsum('lmni, lmni -> lmn', n[:-1, 1:], here))
    here = np.einsum('lmn, lmni -> lmni',if_parallel, n[:-1, 1:])
    test = np.einsum('lmni, lmni -> lmn', n[:-1, :-1], here)
    temp = np.array(np.where(test<threshold)).transpose().astype(float)
    temp[:, :-1] = temp[:, :-1]+0.5
    defect_indices = np.concatenate([ defect_indices, temp ])
    if print_time:
        print('finish z-direction, with', str(round(time.time()-now,2))+'s')
    now = time.time()

    # Wrap with the periodic boundary condition
    if boundary_periodic:
        defect_indices[:,0] = defect_indices[:,0] % N
        defect_indices[:,1] = defect_indices[:,1] % M
        defect_indices[:,2] = defect_indices[:,2] % L
        defect_indices = np.unique(defect_indices, axis=0)

    return defect_indices



def sort_loop_indices(coords):
    '''
    Sort the indices of a loop based on their nearest neighbor order.

    Parameters
    ----------
    coords : array_like, (N, M)
             Array containing the indices representing a line.
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



def smoothen_line(line_coord, window_ratio=3, window_length=None, order=3, N_out=160, mode='interp'):
    '''
    Smoothen a line represented by coordinates using Savitzky-Golay filtering
    and cubic spline interpolation. Usually used for smoothening disclination lines.

    Parameters
    ----------
    line_coord : array_like, N x M
                 array containing the coordinates of the original line.
                 N is the number of coordinates, and M is the dimension.
    
    window_ratio : int, optional
                   Ratio to compute the Savitzky-Golay filter window length. 
                   window_length = number of coordinates (N) / window_ratio.
                   Default is 3.

    window_length: int, odd
                   If window_length is set directly, window_ratio would be ignored.
                   Default is None.

    order : int, optional
            Order of the Savitzky-Golay filter. 
            Default is 3.

    N_out : int, optional
            Number of points in the output smoothened line. 
            Default is 160.

    mode : str, optional
           Mode of extension for the Savitzky-Golay filter (usually 'interp' or 'wrap'). 
           Extension: Pad the signal with extension.
           Default is 'interp', with no extension.
           If 'wrap', the extension contains the values from the other end of the array.
           To smoothen loops, "wrap' must be used.

    Returns
    -------
    output : numpy.ndarray, N_out x M
             Array representing the smoothened line coordinates. M is the dimension.

    Dependencies
    ------------
    - scipy: 1.7.3
    - numpy: 1.22.0   
    '''

    # Calculate the window_length for the Savitzky-Golay filter
    # For some version of scipy, window_length must be odd
    if window_length == None:
        window_length = int(len(line_coord)/window_ratio/2)*2 + 1

    # Apply Savitzky-Golay filter to smoothen the line
    from scipy.signal import savgol_filter
    line_points = savgol_filter(line_coord, window_length, order, axis=0, mode=mode)

    # Generate the parameter values for cubic spline interpolation
    uspline = np.arange(len(line_coord))/len(line_coord)

    # Use cubic spline interpolation to obtain new coordinates
    from scipy.interpolate import splprep, splev
    tck = splprep(line_points.T, u=uspline, s=0)[0]
    output = np.array(splev(np.linspace(0,1,N_out), tck)).T

    return output


def get_plane(points):
    '''
    Calculate the normal vector of the best-fit plane to a set of 3D points 
    using Singular Value Decomposition (SVD).

    Parameters
    ----------
    points : numpy.ndarray, N x 3
             Array containing the 3D coordinates of the points.
             N is the number of points.

    Returns
    -------
    normal_vector : numpy.ndarray, 3
                    Array representing the normal vector of the best-fit plane.

    Dependencies
    ------------
    - numpy: 1.22.0   
    '''
    # Calculate the center of the points
    center    = points.mean(axis=0)

    # Translate the points to be relative to the center
    relative  = points - np.tile(center, (np.shape(points)[0],1))

    # Perform Singular Value Decomposition (SVD) on the transposed relative points
    svd  = np.linalg.svd(relative.T)

    # Extract the left singular vector corresponding to the smallest singular value
    normal_vector = svd[0][:, -1]

    return normal_vector



def plot_loop(
            loop_coord, 
            tube_radius=0.25, tube_opacity=0.5, tube_color=(0.5,0.5,0.5), if_add_head=True,
            if_norm=False, 
            norm_coord=[None,None,None], norm_color=(0,0,1), norm_length=20, 
            norm_opacity=0.5, norm_width=1.0, norm_orient=1,
            print_load_mayavi=False
            ):

    if print_load_mayavi == True:
        now = time.time()
        from mayavi import mlab
        print(f'loading mayavi cost {round(time.time()-now, 2)}s')
    else:
        from mayavi import mlab

    if if_add_head==True:
        loop_coord = np.concatenate([loop_coord, [loop_coord[0]]])

    mlab.plot3d(*(loop_coord.T), tube_radius=tube_radius, opacity=tube_opacity, color=tube_color)

    if if_norm == True:
        loop_N = get_plane(loop_coord) * norm_orient
        loop_center = loop_coord.mean(axis=0)
        for i, coord in enumerate(norm_coord):
            if coord != None:
                loop_center[i] = coord
        mlab.quiver3d(
        *(loop_center), *(loop_N),
        mode='arrow',
        color=norm_color,
        scale_factor=norm_length,
        opacity=norm_opacity,
        line_width=norm_width
        ) 
            


# -----------------------------------------------------------------------------
# Given a local director field. Visualize the disclination loop if there is any
# -----------------------------------------------------------------------------

def plot_loop_from_n(
                    n_box, 
                    origin=[0,0,0], N=1, width=1, 
                    tube_radius=0.25, tube_opacity=0.5, tube_color=(0.5,0.5,0.5), if_add_head=True,
                    if_smooth=True, window_ratio=3, order=3, N_out=160,
                    deform_funcs=[None,None,None],
                    if_norm=False, 
                    norm_coord=[None,None,None], norm_color=(0,0,1), norm_length=20, 
                    norm_opacity=0.5, norm_width=1.0, norm_orient=1

                    ):

    loop_indices = defect_detect(n_box)
    if len(loop_indices) > 0:
        loop_indices = loop_indices + np.tile(origin, (np.shape(loop_indices)[0],1) )
        loop_coord = sort_line_indices(loop_indices)/N*width
        for i, func in enumerate(deform_funcs):
            if func != None:
                loop_coord[:,i] = func(loop_coord[:,i])
        if if_smooth == True:
            loop_coord = smoothen_line(
                                    loop_coord,
                                    window_ratio=window_ratio, order=order, N_out=N_out,
                                    mode='wrap'
                                    )
        plot_loop(
                loop_coord, 
                tube_radius=tube_radius, tube_opacity=tube_opacity, tube_color=tube_color,
                if_add_head=if_add_head,
                if_norm=if_norm,
                norm_coord=norm_coord, norm_color=norm_color, norm_length=norm_length, 
                norm_opacity=norm_opacity, norm_width=norm_width, norm_orient=norm_orient
                    ) 


# -------------------------------------------------------------------------
# Visualize the disclination loop with directors lying on one cross section
# -------------------------------------------------------------------------

def show_loop_plane(
                    loop_box_indices, n_whole, 
                    width=0, margin_ratio=0.6, upper=0, down=0, norm_index=0, 
                    tube_radius=0.25, tube_opacity=0.5, scale_n=0.5, 
                    if_smooth=True,
                    print_load_mayavi=False
                    ):
    
    #! Unify origin (index or sapce unit)
    
    if print_load_mayavi == True:
        now = time.time()
        from mayavi import mlab
        print(f'loading mayavi cost {round(time.time()-now, 2)}s')
    else:
        from mayavi import mlab
    from Nematics3D.field import select_subbox, local_box_diagonalize

    def SLP_plot_plane(upper, down, d_box, grid, norm_vec, n_box, scale_n):

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
                scale_factor=scale_n,
                opacity = 1
                )
        vector.glyph.color_mode = 'color_by_scalar'
        lut_manager = mlab.colorbar(object=vector)
        lut_manager.data_range=(0,1)

    N = np.shape(n_whole)[0]
    if width == 0:
        width = N

    # Find the region enclosing the loop. The size of the region is controlled by margin_ratio
    sl0, sl1, sl2, _ = select_subbox(loop_box_indices, 
                                [N, N, N], 
                                margin_ratio=margin_ratio
                                )

    # Select the local n around the loop
    n_box = n_whole[sl0,sl1,sl2]

    eigvec, eigval = local_box_diagonalize(n_box)

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

    down, upper = np.sort([down, upper])
    if upper==down:
        upper = dmean + 0.5
        down  = dmean - 0.5
    else:
        upper = dmean + upper
        down  = dmean + down

    mlab.figure(bgcolor=(0,0,0))
    SLP_plot_plane(upper, down, d_box, grid, norm_vec, n_box, scale_n)
    plot_loop_from_n(
                    n_box, 
                    origin=loop_box_indices[:,0], N=N, width=width,
                    tube_radius=tube_radius, tube_opacity=tube_opacity,
                    if_smooth=if_smooth
                    )

    return dmean, eigvec, eigval

# -----------------------------------------------------
# Visualize the directors projected on principle planes
# ----------------------------------------------------- 

def show_plane_2Ddirector(
                        n_box, height, 
                        color_axis=(1,0), height_visual=0,
                        space=3, line_width=2, line_density=1.5, 
                        if_omega=True, S_box=0, S_threshold=0.18,
                        if_cb=True, colormap='blue-red',
                          ):
    
    from mayavi import mlab
    
    from plotdefect import get_streamlines
    
    color_axis1 = color_axis / np.linalg.norm(color_axis) 
    color_axis2 = np.cross( np.array([0,0,1]), np.concatenate( [color_axis1,[0]] ) )
    color_axis2 = color_axis2[:-1]
    
    x = np.arange(np.shape(n_box)[0])
    y = np.arange(np.shape(n_box)[1])
    z = np.arange(np.shape(n_box)[2])

    indexy = np.arange(0, np.shape(n_box)[1], space)
    indexz = np.arange(0, np.shape(n_box)[2], space)
    iny, inz = np.meshgrid(indexy, indexz, indexing='ij')
    ind = (iny, inz)

    n_plot = n_box[height]

    n_plane = np.array( [n_plot[:,:,1][ind], n_plot[:,:,2][ind] ] )
    n_plane = n_plane / np.linalg.norm( n_plane, axis=-1, keepdims=True)

    stl = get_streamlines(
                y[indexy], z[indexz], 
                n_plane[0].transpose(), n_plane[1].transpose(),
                density=line_density)
    stl = np.array(stl)

    connect_begin = np.where(np.abs( stl[1:,0] - stl[:-1,1]  ).sum(axis=-1) < 1e-5 )[0]
    connections = np.zeros((len(connect_begin),2))
    connections[:,0] = connect_begin
    connections[:,1] = connect_begin + 1

    lines_index = np.arange(np.shape(stl)[0])
    disconnect = lines_index[~np.isin(lines_index, connect_begin)]

    if height_visual == 0:
        src_x = stl[:, 0, 0] * 0 + height
    else:
        src_x = stl[:, 0, 0] * 0 + height_visual
    src_y = stl[:, 0, 0]
    src_z = stl[:, 0, 1]

    unit = stl[1:, 0] - stl[:-1, 0]
    unit = unit / np.linalg.norm(unit, axis=-1, keepdims=True)

    coe1 = np.einsum('ij, j -> i', unit, color_axis1)
    coe2 = np.einsum('ij, j -> i', unit, color_axis2)
    coe1 = np.concatenate([coe1, [coe1[-1]]])
    coe2 = np.concatenate([coe2, [coe2[-1]]])
    colors = np.arctan2(coe1,coe2)
    nan_index = np.array(np.where(np.isnan(colors)==1))
    colors[nan_index] = colors[nan_index-1]
    colors[disconnect] = colors[disconnect-1]

    src = mlab.pipeline.scalar_scatter(src_x, src_y, src_z, colors)
    src.mlab_source.dataset.lines = connections
    src.update()

    lines = mlab.pipeline.stripper(src)
    plot_lines = mlab.pipeline.surface(lines, line_width=line_width, colormap='blue-red')

    if type(colormap) == np.ndarray:
        lut = plot_lines.module_manager.scalar_lut_manager.lut.table.to_array()
        lut[:, :3] = colormap
        plot_lines.module_manager.scalar_lut_manager.lut.table = lut
    

    if if_cb == True:
        cb = mlab.colorbar(object=plot_lines, orientation='vertical', nb_labels=5, label_fmt='%.2f')
        cb.data_range = (0,1)
        cb.label_text_property.color = (0,0,0)
    

    if if_omega == True: 

        from scipy import ndimage

        binimg = S_box<S_threshold
        binimg[:height] = False
        binimg[(height+1):] = False
        binimg = ndimage.binary_dilation(binimg, iterations=1)
        if np.sum(binimg) > 0:
            labels, num_objects = ndimage.label(binimg, structure=np.zeros((3,3,3))+1)
            if num_objects > 2:
                raise NameError('more than two parts')
            elif num_objects == 1:
                print('only one part. Have to seperate points by hand')
                cross_coord = np.transpose(np.where(binimg==True))
                cross_center = np.mean(cross_coord, axis=0)
                cross_relative = cross_coord - np.tile(cross_center, (np.shape(cross_coord)[0],1))
                cross_dis = np.sum(cross_relative**2, axis=1)**(1/2)
                cross_to0 = cross_coord[cross_dis < np.percentile(cross_dis, 50), :]
                binimg[tuple(np.transpose(cross_to0))] = False
            labels, num_objects = ndimage.label(binimg, structure=np.zeros((3,3,3))+1)
            for i in range(1,3):  
                index = np.where(labels==i)
                X, Y, Z = np.meshgrid(x,y,z, indexing='ij')
                cord1 = X[index] - n_box[..., 0][index]/2
                cord2 = Y[index] - n_box[..., 1][index]/2
                cord3 = Z[index] - n_box[..., 2][index]/2
            
                pn = np.vstack((n_box[..., 0][index], n_box[..., 1][index], n_box[..., 2][index]))
                Omega = get_plane(pn.T)
                Omega= Omega * np.sign(Omega[0])
                scale_norm = 20
                if height_visual == 0:
                    xvisual = cord1.mean()
                else:
                    xvisual = height_visual
                mlab.quiver3d(
                        xvisual, cord2.mean(), cord3.mean(),
                        Omega[0], Omega[1], Omega[2],
                        mode='arrow',
                        color=(0,1,0),
                        scale_factor=scale_norm,
                        opacity=0.5
                        )

# ------------------------------------------------------------------------------------
# Visualize the disclination loop with directors projected on several principle planes
# ------------------------------------------------------------------------------------ 

def show_loop_plane_2Ddirector(
                                n_box, S_box,
                                height_list, if_omega_list=[1,1,1], plane_list=[1,1,1],
                                height_visual_list=0, if_rescale_loop=True,
                                figsize=(1920, 1360), bgcolor=(1,1,1), camera_set=0,
                                if_norm=True, norm_length=20, norm_orient=1, norm_color=(0,0,1),
                                line_width=2, line_density=1.5,
                                tube_radius=0.75, tube_opacity=1,
                                print_load_mayavi=False, if_cb=True, n_colormap='blue-red'
                                ):
    
    if height_visual_list == 0:
        height_visual_list = height_list
        if_rescale_loop = False
        parabola = None
    if if_rescale_loop == True:
        x, y, z = height_list
        coe_matrix = np.array([
                        [x**2, y**2, z**2],
                        [x, y, z],
                        [1,1,1]
                        ])
        del x, y, z
        coe_parabola = np.dot(height_visual_list, np.linalg.inv(coe_matrix))
        def parabola(x):
            return coe_parabola[0]*x**2 + coe_parabola[1]*x + coe_parabola[2]
    else:
        def parabola(x):
            return x


    if print_load_mayavi == True:
        now = time.time()
        from mayavi import mlab
        print(f'loading mayavi cost {round(time.time()-now, 2)}s')
    else:
        from mayavi import mlab

    mlab.figure(size=figsize, bgcolor=bgcolor)

    plot_loop_from_n(n_box, 
                     tube_radius=tube_radius, tube_opacity=tube_opacity, 
                     deform_funcs=[parabola,None,None],
                     if_norm=if_norm,
                     norm_coord=[height_visual_list[0],None,None], norm_length=norm_length, norm_orient=norm_orient, norm_color=norm_color,
                     )

    for i, if_plane in enumerate(plane_list):
        if if_plane == True:
            show_plane_2Ddirector(n_box, height_list[i], 
                                  height_visual=height_visual_list[i], if_omega=if_omega_list[i], 
                                  line_width=line_width, line_density=line_density,
                                  S_box=S_box, if_cb=if_cb, colormap=n_colormap)

    if camera_set != 0: 
        mlab.view(*camera_set[:3], roll=camera_set[3])

# -----------------------------------------------------
# Visualize the disclinations within the simulation box
# -----------------------------------------------------
        
def plot_defect(n, 
                origin=[0,0,0], grid=128, width=200,
                if_plot_defect=True, defect_threshold=0, defect_color=(0.2,0.2,0.2), scale_defect=2,
                plot_n=True, n_interval=1, ratio_n_dist = 5/6,
                print_load_mayavi=False
                ):
    
    if plot_n == False and plot_defect == False:
        print('no plot')
        return 0
    else:
        if print_load_mayavi == True:
            now = time.time()
            from mayavi import mlab
            print(f'loading mayavi cost {round(time.time()-now, 2)}s')
        else:
            from mayavi import mlab
        mlab.figure(bgcolor=(1,1,1))

    if plot_n == True:

        nx = n[:,:,:,0]
        ny = n[:,:,:,1]
        nz = n[:,:,:,2]

        N, M, L = np.shape(n)[:-1]

        indexx = np.arange(0, N, n_interval)
        indexy = np.arange(0, M, n_interval)
        indexz = np.arange(0, L, n_interval)
        ind = tuple(np.meshgrid(indexx, indexy, indexz, indexing='ij'))
        
        x = indexx / grid * width + origin[0]
        y = indexy / grid * width + origin[1]
        z = indexz / grid * width + origin[2]
        X, Y, Z = np.meshgrid(x,y,z, indexing='ij')

        distance = n_interval / grid * width
        n_length = distance * ratio_n_dist

        coordx = X - nx[ind] * n_length / 2
        coordy = Y - ny[ind] * n_length / 2
        coordz = Z - nz[ind] * n_length / 2

        phi = np.arccos(nx[ind])
        theta = np.arctan2(nz[ind], ny[ind])

        vector = mlab.quiver3d(
                                coordx, coordy, coordz,
                                nx[ind],ny[ind],nz[ind],
                                mode='cylinder',
                                scalars = (1-np.cos(2*phi))*(np.sin(theta%np.pi)+0.3),
                                scale_factor=n_length
                                )
        
        vector.glyph.color_mode = 'color_by_scalar'
        
        lut_manager = mlab.colorbar(object=vector)
        lut_manager.data_range=(0,1.3)

    if if_plot_defect == True:

        defect_indices = defect_detect(n, threshold=defect_threshold)
        defect_coords  = defect_indices / grid * width
        defect_coords = defect_coords + origin
        mlab.points3d(*(defect_coords.T), color=defect_color, scale_factor=scale_defect)



# -----------------------------------------------------------------------------------------------------
# Classify the disclinations into different lines and loops, by whether the disclinations are connected
# -----------------------------------------------------------------------------------------------------

def trans_period(n, N):
    if n == 0:
        return N
    elif n == N-1:
        return -1
    
def check_find(defect_here, defect_group, defect_box, N):

    from scipy.spatial.distance import cdist
    
    defect_plane_axis = np.where( ( defect_here % 1) == 0 )[0][0]

    if_find = False

    dist = cdist(defect_group, defect_box, metric='sqeuclidean')
    defect_where = np.where(dist == 0.5)[1]
    if len(defect_where) == 0:
        defect_where = np.where(dist == 1)[1]
        if len(defect_where) > 0:
            for item in defect_where:
                defect_ordinal_next = item
                defect_next = defect_box[defect_ordinal_next]
                defect_diff = defect_next - defect_here
                if defect_plane_axis == (np.where( (np.abs(defect_diff) == 1) + (np.abs(defect_diff) == N-1) ))[0][0]:
                    if_find = True
                    break
    else:
        if_find = True
        defect_ordinal_next = defect_where[0]
        defect_next = defect_box[defect_ordinal_next]
        
    if if_find == False:
        defect_ordinal_next, defect_next = None, None
        
    return if_find, defect_ordinal_next, defect_next

def find_box(value, length_list):
    
    cumulative_sum = 0
    position = 0

    for i, num in enumerate(length_list):
        cumulative_sum += num
        if cumulative_sum-1 >= value:
            position = i
            break

    index = value - (cumulative_sum - length_list[position])
    return (position, index)

def defect_connected(defect_indices, N, print_time=False, print_per=1000):
    """
    Classify defects into different lines.

    Parameters
    ----------
    defect_indices : numpy array, num_defects x 3
                     Represents the locations of defects in the grid.
                     For each location, there must be one integer (the index of plane) and two half-integers (the center of the loop on that plane)
                     This is usually given by defect_detect()

    
    
    """

    #! It only works with boundary=True in defect_detect() if defects cross walls

    index = 0
    defect_num= len(defect_indices)
    defect_left_num = defect_num
    
    lines = []
    
    check0 = defect_indices[:,0] < int(N/2)
    check1 = defect_indices[:,1] < int(N/2)
    check2 = defect_indices[:,2] < int(N/2)
    
    defect_box = []
    defect_box.append( defect_indices[ np.where( check0 * check1 * check2 ) ] )
    defect_box.append( defect_indices[ np.where( ~check0 * check1 * check2 ) ] )
    defect_box.append( defect_indices[ np.where( check0 * ~check1 * check2 ) ] )
    defect_box.append( defect_indices[ np.where( check0 * check1 * ~check2 ) ] )
    defect_box.append( defect_indices[ np.where( ~check0 * ~check1 * check2 ) ] )
    defect_box.append( defect_indices[ np.where( ~check0 * check1 * ~check2 ) ] )
    defect_box.append( defect_indices[ np.where( check0 * ~check1 * ~check2 ) ] )
    defect_box.append( defect_indices[ np.where( ~check0 * ~check1 * ~check2 ) ] )
    
    start = time.time()
    start_here = time.time()

    while defect_left_num > 0:
        
        loop_here = np.zeros( ( len(defect_indices),3 ) )
        defect_ordinal_next = 0
        cross_wall = np.array([0,0,0])
        
        box_index = next((i for i, box in enumerate(defect_box) if len(box)>0), None)
        defect_box_here = defect_box[box_index]
        loop_here[0] = defect_box_here[0]
        
        index_here = 0
        
        while True:
            
            defect_ordinal = defect_ordinal_next
            defect_box_here = 1*defect_box[box_index]
            defect_here = defect_box_here[defect_ordinal]
            defect_box[box_index] = np.vstack(( defect_box_here[:defect_ordinal], defect_box_here[defect_ordinal+1:] ))
            defect_box_here = 1*defect_box[box_index]
            
            defect_group = np.array([defect_here])

            if_find = False
            
            if len(defect_box_here) > 0:
                if_find, defect_ordinal_next, defect_next = check_find(defect_here, defect_group, defect_box_here, N)

            if if_find == False or len(defect_box_here) == 0:
                
                defect_box_all = np.concatenate([box for box in defect_box])
            
                bound_0 = np.where( (defect_here==0) + (defect_here==N-1) )[0]
                if len(bound_0) > 0:
                    defect_bound = 1*defect_here
                    defect_bound[bound_0[0]] = trans_period(defect_bound[bound_0[0]],N)
                    defect_group = np.concatenate([defect_group, [defect_bound]])
                bound_1 = np.where(defect_here==(N-0.5))[0]
                for bound in bound_1:
                    defect_bound = 1*defect_here
                    defect_bound[bound] = -0.5
                    defect_group = np.concatenate([defect_group, [defect_bound]])
                if len(bound_0) > 0 and len(bound_1) > 0:
                    for bound in bound_1:
                        defect_bound = 1*defect_here
                        defect_bound[bound] = -0.5
                        defect_bound[bound_0[0]] = trans_period(defect_bound[bound_0[0]],N)
                        defect_group = np.concatenate([defect_group, [defect_bound]])
                        
                if_find, defect_ordinal_next, defect_next = check_find(defect_here, defect_group, defect_box_all, N)
                if if_find == True:
                    box_index, defect_ordinal_next = find_box(defect_ordinal_next, [len(term) for term in defect_box])
                
            if if_find == True:
                defect_diff = defect_next - defect_here
                cross_wall_here = np.trunc( defect_diff / (N-10) )
                cross_wall = cross_wall - cross_wall_here
                defect_next = defect_next + cross_wall * N
                loop_here[index_here+1] = defect_next
                index += 1
                index_here += 1
                if print_time == True:
                    if index % print_per == 0:
                        print(f'{index}/{defect_num} = {round(index/defect_num*100,2)}%, {round(time.time()-start_here,2)}s  ',
                            f'{round(time.time()-start,2)}s in total' )
                        start_here= time.time()
            else:
                zero_loc = np.where(np.all([0,0,0] == loop_here, axis=1))[0]
                if len(zero_loc) > 0:
                    loop_here = loop_here[:zero_loc[0]]
                lines.append(loop_here)
                defect_left_num = 0
                for term in defect_box:
                    defect_left_num += len(term)
                break
    
    return lines

# ------------------------------------------
# Add mid-points into the disclination lines
# ------------------------------------------

def add_mid_points_disclination(line):

    line_new = np.zeros((2*len(line)-1,3))
    line_new[0::2] = line
    defect_diff = line[1:] - line[:-1]
    defect_diff_mid_value = np.sign(defect_diff[np.where( line[:-1]%1 == 0 )]) * 0.5
    defect_diff_mid_orient = (line[:-1]%1 == 0).astype(int)
    line_new[1::2] = line_new[0:-1:2] + np.array([defect_diff_mid_value]).T * defect_diff_mid_orient
    
    return line_new 

def blue_red_in_white_bg():
    '''
    Generate a colormap with a transition from blue to red. 
    The color is normalized to be distinct on white background.

    
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

    n = np.arange(2, num)
    a = 2**np.trunc(np.log2(n-1)+1)
    b = 2*n - a - 1

    result = np.zeros(num)

    result[0] = 0
    result[1] = 1
    result[2:] = b/a
    
    return result

def visual_disclinations(lines, N, min_length=30, radius=0.5,
                         window_cross=33, window_loop=21,
                         N_ratio_cross=1, N_ratio_loop=1.5,
                         loop_color=(0,0,0), wrap=True, if_lines_added=False):
    '''
    #! Color_list
    Visualize disclination lines in 3D using Mayavi.

    Parameters
    ----------
    lines : list of arrays
            List of disclination lines, where each line is represented as an array of coordinates.

    N : int
        Size of the cubic 3D grid in each dimension.

    min_length : int, optional
                 Minimum length of disclination loops to consider. 
                 Length calculated as the number of points conposing the loop-type disclinations.
                 Default is 30.

    radius : float, optional
             Radius of the points representing the disclination lines in the visualization. 
             Default is 0.5.

    window_cross : int, optional
                   Window length for Savitzky-Golay filter for cross-type disclinations. 
                   Default is 33.

    window_loop : int, optional
                  Window length for Savitzky-Golay filter for loop-type disclinations. 
                  Default is 21.

    N_ratio_cross : float, optional
                    Ratio to compute the number of points in the output smoothened line for crosses.
                    N_out = N_ratio_cross * num_points_in_cross
                    Default is 1.

    N_ratio_loop : float, optional
                   Ratio to compute the number of points in the output smoothened line for loops.
                   N_out = N_ratio_loop * num_points_in_loop
                   Default is 1.5.

    loop_color : array like, shape (3,)  or None, optional
                 The color of visualized loops.
                 If None, the loops will follow same colormap with crosses.
                 Default is (0,0,0)

    wrap : bool, optional
           If wrap the lines with periodic boundary condition.
           Default is True

    if_lines_added : bool, optional
                     If the lines have already been added the mid-points
                     Default is False.

    Returns
    -------
    None

    Dependencies
    ------------
    - mayavi: 4.7.4
    - numpy: 1.22.0

    Functions in same module
    ------------------------
    - add_mid_points_disclination: Add mid-points to disclination lines.
    - smoothen_line: Smoothen disclination lines.
    - sample_far: Create a sequence where each number is trying to be far away from previous numbers.
    - blue_red_in_white_bg: A colormap based on blue-red while dinstinct on white backgroud.    
    '''

    from mayavi import mlab

    # Filter out short disclination lines
    lines = np.array(lines, dtype=object)
    lines = lines[ [len(line)>=min_length for line in lines] ] 

    # Add mid-points to each disclination line
    if if_lines_added == False:
        for i, line in enumerate(lines):
            lines[i] = add_mid_points_disclination(line)

    # Separate lines into loops and crosses based on end-to-end distance and smoothen them.
    loops = []
    crosses = []
    for i, line in enumerate(lines):
        end_to_end = np.sum( (line[-1]-line[0])**2, axis=-1 )
        if end_to_end > 2:
            cross = smoothen_line(line, window_length=window_cross, 
                                    N_out=int(N_ratio_cross * len(line)))
            crosses.append(cross)
        else:
            loop = smoothen_line(line, window_length=window_loop, mode='wrap', 
                                    N_out=int(N_ratio_loop*len(line)))
            loops.append(loop)

    # Sort crosses by length for better visualization
    crosses = np.array(crosses, dtype=object)
    cross_length = [len(cross) for cross in crosses]
    crosses = crosses[np.argsort(cross_length)[-1::-1]]

    # Generate colormap
    colormap = blue_red_in_white_bg()

    # Generate colors for cross-type disclinations or all disclinations
    if loop_color != None:
        if len(crosses) > 0:
            color_index = ( sample_far(len(crosses)) * 510 ).astype(int)
    else:
        color_index = ( sample_far(len(lines)) * 510 ).astype(int)
    
    
    # Plotting
    mlab.figure(bgcolor=(1,1,1)) 

    # wrap the discliantions with periodic boundary conditions
    if wrap == True:
        loops = np.array(loops, dtype=object)%N
        crosses = np.array(crosses, dtype=object)%N

    for i,cross in enumerate(crosses):
        mlab.points3d(*(cross.T), scale_factor=radius, color=tuple(colormap[color_index[i]])) 

    if loop_color != None:
        for loop in loops:
            mlab.points3d(*(loop.T), scale_factor=radius, color=loop_color)
    else:
        for j, loop in enumerate(loops):
            mlab.points3d(*(loop.T), scale_factor=radius, color=tuple(colormap[color_index[len(crosses)+j]]))
        




def ordered_bulk_size(defect_indices, N, width, if_print=True):
    '''
    Compute the minimum distance from each point in a 3D grid the neirhboring defects.

    Parameters
    ----------
    defect_indices : numpy.ndarray, shape (N,3)
                     Array containing the coordinates of defect points. 
                     M is the number of defect points.

    N : int
        Size of the cubic 3D grid in each dimension.

    width : float
            Width of the simulation box in the unit of real length (not indices).

    if_print : bool, optional
              Flag to print the time taken for each octant.
              Default is True.

    Returns
    -------
    dist_min : numpy.ndarray, shape (N^3,)
               Array containing the minimum distances from each point in the 3D grid to the nearest defect.

    Dependencies
    ------------
    - numpy: 1.22.0
    - scipy: 1.7.3
    '''
    from itertools import product
    import time

    from scipy.spatial.distance import cdist

    # Ensure defect indices are within the periodic boundary conditions
    defect_indices = defect_indices % N

    # Generate the coordinates of each point in the 3D grid.
    grid = np.array(list(product(np.arange(N), np.arange(N), np.arange(N))))

    # Divide defect and grid points into octants based on their positions in each dimension
    defect_check0 = defect_indices[:,0] < int(N/2)
    defect_check1 = defect_indices[:,1] < int(N/2)
    defect_check2 = defect_indices[:,2] < int(N/2)

    defect_box = [defect_indices[ np.where(  defect_check0 *  defect_check1 *  defect_check2 ) ],
                  defect_indices[ np.where( ~defect_check0 *  defect_check1 *  defect_check2 ) ],
                  defect_indices[ np.where(  defect_check0 * ~defect_check1 *  defect_check2 ) ],
                  defect_indices[ np.where(  defect_check0 *  defect_check1 * ~defect_check2 ) ],
                  defect_indices[ np.where( ~defect_check0 * ~defect_check1 *  defect_check2 ) ],
                  defect_indices[ np.where( ~defect_check0 *  defect_check1 * ~defect_check2 ) ],
                  defect_indices[ np.where(  defect_check0 * ~defect_check1 * ~defect_check2 ) ],
                  defect_indices[ np.where( ~defect_check0 * ~defect_check1 * ~defect_check2 ) ]]

    grid_check0 = grid[:,0] < int(N/2)
    grid_check1 = grid[:,1] < int(N/2)
    grid_check2 = grid[:,2] < int(N/2)

    grid_box = []
    grid_box = [grid[ np.where(  grid_check0 *  grid_check1 *  grid_check2 ) ],
                grid[ np.where( ~grid_check0 *  grid_check1 *  grid_check2 ) ],
                grid[ np.where(  grid_check0 * ~grid_check1 *  grid_check2 ) ],
                grid[ np.where(  grid_check0 *  grid_check1 * ~grid_check2 ) ],
                grid[ np.where( ~grid_check0 * ~grid_check1 *  grid_check2 ) ],
                grid[ np.where( ~grid_check0 *  grid_check1 * ~grid_check2 ) ],
                grid[ np.where(  grid_check0 * ~grid_check1 * ~grid_check2 ) ],
                grid[ np.where( ~grid_check0 * ~grid_check1 * ~grid_check2 ) ]]

    # To same memory, consider half of grid points in each octant for each calculation
    size = len(grid_box[0])
    half = int(len(grid_box[0])/2)

    dist_min = np.zeros(N**3)

    # Calculate!
    for i, box in enumerate(grid_box):
        start = time.time()
        dist_min[ i*size:i*size+half ] = np.min(cdist(box[:half], defect_box[i]), axis=-1)
        dist_min[ i*size+half:(i+1)*size ] = np.min(cdist(box[half:], defect_box[i]), axis=-1)
        print(time.time()-start)

    # Change the unit of distances into real length,    
    dist_min = dist_min / N * width

    return dist_min



def save_xyz(fname, atoms, lattice=None):
    """
    Saves data to a file in extended XYZ format. This function will attempt to
    produce an extended XYZ file even if the input data is incomplete.
    Usually used to store information of disclination loops.
    Created by Matthew E. Peterson.

    Parameters
    ----------
    fname : str or pathlib.Path
            File to write data to.

    atoms : pandas.DataFrame
            Atomic data to be saved.

    lattice : array_like (default None)
              Lattice vectors of the system, so that lattice[0] is the first lattice vector.

    Raises
    ------
    KeyError
        If a key in `cols` does not appear in `data`.
    """

    import re
    import os
    import gzip
    import shutil

    gzipped = fname.endswith('.gz')
    if gzipped:
        fname = fname[:-3]

    def map_type(dtype):
        if np.issubdtype(dtype, np.integer):
            return 'I'
        elif np.issubdtype(dtype, np.floating):
            return 'R'
        else:
            return 'S'

    def collapse_names(names):
        cols = dict()
        for name in names:
            col = re.sub(r'-\w+$', '', name)
            try:
                cols[col].append(name)
            except:
                cols[col] = [name]
        return cols

    with open(fname, 'w') as f:
        # the first line is simply the number of atoms
        f.write(f"{len(atoms)}\n")

        # write lattice if given
        header = ""
        if lattice is not None:
            lattice = np.asarray(lattice)
            lattice = ' '.join(str(x) for x in lattice.ravel())
            header += f'Lattice="{lattice}" '

        # now determine property line
        cols = collapse_names(atoms.columns)
        props=[]
        for col, names in cols.items():
            name = names[0] 
            tp=map_type(atoms.dtypes[name])
            dim=len(names)
            props.append(f"{col}:{tp}:{dim}")
        props = ':'.join(props)
        header += f'Properties={props}\n'

        f.write(header)

        # writing the atomic data is easy provided we are using a DataFrame
        atoms.to_csv(f, header=None, sep=" ", float_format="%f", index=False)

    if gzipped:
        with open(fname, 'rb') as f_in:
            with gzip.open(f"{fname}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(fname)


def extract_loop(defect_indices, N, dilate_times=2,
                 save_path='.', save_head=''):
    
    '''
    Given the defect indices in 3D grids, extract loops and calculate their number of holes.
    Using binary dilation and Euler number calculation. 
    Saves loop information and coordinates in files.

    Parameters
    ----------
    defect_indices : array_like, (M, 3)
                     Array containing the indices of defects in the 3D grid.
                     M is the number of defects, and each row contains (x, y, z) indices.

    N : int
        Size of the 3D grid along each dimension.

    dilate_times : int, optional
                   Number of iterations for binary dilation to thicken the disclination lines.
                   Default is 2.

    save_path : str, optional
                Path to save loop information files. 
                Default is current directory.

    save_head : str, optional
                Prefix for saved files. 
                Default is an empty string.

    Saves
    -----
    'summary.xyz.gz' : summary, pandas file
                       The information of each point of thickened disclination loops.
                       Include x, y, z coordinates (indices), genus and label of this loop.

    'summary_wrap.xyz.gz' : summary_wrap, pandas file
                            Same with summary except the coordinates are wrapped in periodic boundary condition

    '{lable}box_{g}.txt' : box, array-like, (3,2)
                           The vortices of box containing one thickened disclination loop.
                           g = genus of this loop

    '{lable}index_{g}.txt' : coord, array-like, (M,3)
                             The indices of each point in one thickened disclination loop.
                             g = genus of this loop, M is the number of points in the loop.

    'grid_g.npy' : grid_result, array-like, (N,N,N)
                   The defects indices and genus represented by grid of box.
                   N is the size of the 3D grid along each dimension.
                   If one point is 0, there is no defect here.
                   If one point is x (not zero), there is a defect here and the genus of this line is x-1.
    
    Dependencies
    ------------
    - numpy : 1.22.0
    - scipy : 1,7.3
    - skimage : 0.19.1
    - pandas : 2.0.2
    '''


    from scipy import ndimage
    from skimage.measure import euler_number
    import pandas as pd

    # Ensure save_path ends with '/'
    save_path = save_path + '/'

    # Wrap the defect
    defect_indices = (defect_indices%N).astype(int)

    # Create a binary grid representing the defect indices
    grid_origin = np.zeros((N, N, N))
    grid_origin[tuple(defect_indices.T)] = 1

    # Expand the defect grid to handle periodic boundary conditions
    grid = np.zeros( (2*N, 2*N, 2*N ) )
    grid[:N, :N, :N]            = grid_origin
    grid[N:2*N, :N, :N]         = grid_origin
    grid[:N, N:2*N, :N]         = grid_origin
    grid[:N, :N, N:2*N]         = grid_origin
    grid[N:2*N, N:2*N, :N]      = grid_origin
    grid[:N, N:2*N, N:2*N]      = grid_origin
    grid[N:2*N, :N, N:2*N]      = grid_origin
    grid[N:2*N, N:2*N, N:2*N]   = grid_origin

    # Perform binary dilation on the extended grid to thicken the lines
    binimg = ndimage.binary_dilation(grid, iterations=dilate_times)

    # Count the number of low-order points
    num_pts = int(np.count_nonzero(binimg)/8)
    print(f"Found {num_pts} low-order points")

    # Prepare the array to record genus of each defect
    grid_origin = grid_origin/2
    grid_result = np.zeros((N, N, N))

    # Initialize arrays to store loop information
    summary = np.empty(8*num_pts,
                       dtype=[
                           ("pos-x", "uint16"),
                           ("pos-y", "uint16"),
                           ("pos-z", "uint16"),
                           ("genus", "int16"),
                           ("label", "uint16"),
                       ]
                       )
    
    summary_wrap = np.empty(8*num_pts,
                       dtype=[
                           ("pos-x", "uint16"),
                           ("pos-y", "uint16"),
                           ("pos-z", "uint16"),
                           ("genus", "int16"),
                           ("label", "uint16"),
                       ]
                       )
    
    # Label connected components in the binary grid
    labels, num_objects = ndimage.label(binimg)

    offset = 0
    label = 0
    loop = 0
    for i, obj in enumerate(ndimage.find_objects(labels)):
        
        # Ensure object boundaries are within the extended grid
        xlo = max(obj[0].start, 0)
        ylo = max(obj[1].start, 0)
        zlo = max(obj[2].start, 0)
        xhi = min(obj[0].stop, 2*N)
        yhi = min(obj[1].stop, 2*N)
        zhi = min(obj[2].stop, 2*N)
        boundary = [xlo, ylo, zlo, xhi, yhi, zhi]
        
        # Exclude the crosses and loops outside of the box and  
        if (0 in boundary) or (2*N in boundary) or max(xlo, ylo, zlo)>N:
            continue
        
        label += 1
        
        # Define the object within the extended grid
        obj = (slice(xlo, xhi), slice(ylo, yhi), slice(zlo, zhi))
        
        # Extract the object from the labeled grid
        img = (labels[obj] == i+1)
    
        # calculate Euler number, defined as # objects + # holes - # loops
        # we do not have holes
        g = euler_number(img, connectivity=1)

        # calculate the number of loop by Euler number
        g = 1 - g
        
        # box : The vortices of box containing the loop
        # coord : The indices of each point of thickened loop
        pos = np.nonzero(img)
        shift = pos[0].size
        
        box = np.array([[xlo, xhi], [ylo, yhi], [zlo, zhi]], dtype=int)
        coord = np.array([pos[0] + xlo, pos[1] + ylo, pos[2] + zlo])

        np.savetxt(save_path + save_head + f"{label}box_{g}.txt", box, fmt="%d")
        np.savetxt(save_path + save_head + f"{label}index_{g}.txt", coord, fmt="%d")

        if g == 1:
            loop += 1

        # Update summary arrays with loop information
        summary['pos-x'][offset:offset+shift] = coord[0]
        summary['pos-y'][offset:offset+shift] = coord[1]
        summary['pos-z'][offset:offset+shift] = coord[2]
        summary['genus'][offset:offset+shift] = g
        summary['label'][offset:offset+shift] = label
        
        summary_wrap['pos-x'][offset:offset+shift] = coord[0]%N
        summary_wrap['pos-y'][offset:offset+shift] = coord[1]%N
        summary_wrap['pos-z'][offset:offset+shift] = coord[2]%N
        summary_wrap['genus'][offset:offset+shift] = g
        summary_wrap['label'][offset:offset+shift] = label
    
        offset += shift

        # Update the genus information of each defect
        grid_result[tuple(coord%N)] = g+1
        
    # grid_result : each point of defect labeled by the genus of lines that the defect belongs to.
    grid_result = grid_result + grid_origin
    grid_result[grid_result%1==0] = 0.5
    grid_result = grid_result - 0.5

    np.save(save_path + save_head + "grid_g.npy", grid_result)
    
    # Trim summary arrays to remove unused space
    summary = summary[:offset]
    summary_wrap = summary_wrap[:offset]
    summary = pd.DataFrame(summary)
    summary_wrap = pd.DataFrame(summary_wrap)

    save_xyz(save_path + save_head + "summary.xyz.gz", summary)
    save_xyz(save_path + save_head + "summary_wrap.xyz.gz", summary_wrap)
                    
    print(f'Found {loop} loops\n')


def visual_loops_genus(lines_origin, N, grid_g, 
                       window_loop=21, N_ratio_loop=1.5, radius=2,
                       if_lines_added=False, wrap=True, 
                       color_list=[]):
    
    '''

    #! Color and grid_g seperately

    Visualize loops using Mayavi.
    Loops are colored according to their genus.

    Given the director field n, visualize the disclination loops by the genus as the following:

    - Detect the defects (defect_indices) in the director field by defect_detect(n, boundary=True)

    - Find the thickened disclination loops and their genus by extract_loop(defect_indices, N)
      It will provide grid_g, labelling each defect with the genus of the line that this defect belongs to.
    
    - Sort the defects into different lines by lines = defect_connected(defect_indices, N)
      Need this step to smoothen the lines.

    - visual_loops_genus(lines, N, grid_g)

    Parameters
    ----------
    lines : list of arrays
            List of disclination lines, where each line is represented as an array of coordinates.
            Usually provided by defect_connected()

    N : int
        Size of the grid in each dimension.

    grid_g : numpy.ndarray, shape (N, N, N)
             The defects indices and genus represented by grid of box.
             N is the size of the 3D grid along each dimension.
             If one point is 0, there is no defect here.
             If one point is x (not zero), there is a defect here and the genus of this line is x-1.
             Usually provided by extract_loop().

    window_loop : int, optional
                  Window length for smoothening the loops using Savitzky-Golay filter.
                  Default is 21.

    N_ratio_loop : float, optional
                   Ratio to determine the number of points in the output smoothened loop.
                   N_out = N_ratio_loop * len(loop).
                   Default is 1.5.

    radius : int, optional
             Scale factor for visualizing the loops.
             Default is 2.

    if_lines_added : bool, optional
                     Flag indicating whether midpoints have been added to disclination lines.
                     Default is False.

    wrap : bool, optional
           Flag indicating whether to wrap the loops in visualization
           Default is True.

    color_list : array-like, optional, shape (M,)
                 The color for each disclination loop. Each value belongs to [0,1]
                 The colormap is derived by blue_red_in_white_bg() in the same module.
                 0 for blue, 1 for red.
                 If the length of color_list is 0, the loops are colored by genus.
                 Default is [] (loops colored by genus)


    Dependencies
    ------------
    - mayavi: 4.7.4
    - scipy: 1.7.3
    - numpy: 1.22.0
    '''

    from mayavi import mlab
    from scipy.spatial.distance import cdist

    # If midpoints are not added to disclination lines, add them
    lines = 1 * lines_origin
    if if_lines_added == False:
        for i, line in enumerate(lines):
            lines[i] = add_mid_points_disclination(line)

    # Initialize an array to store genus information for each line
    genus_lines = np.zeros(len(lines))

    # Get coordinates of defects in the grid
    defect_g = np.array(np.nonzero(grid_g)).T

    # Find genus for each line based on the proximity of its head to defect points
    for i, line in enumerate(lines):
        head = line[0]%N
        dist = cdist([head], defect_g)
        if np.min(dist) > 1:
            genus_lines[i] = -1
        else:
            genus_lines[i] = grid_g[tuple(defect_g[np.argmin(dist)])] - 1

    lines = np.array(lines, dtype=object)
    loops = lines[genus_lines>-1]
    genus_loops = genus_lines[genus_lines > -1]

    mlab.figure(bgcolor=(1,1,1))

    colormap = blue_red_in_white_bg()

    for i, loop in enumerate(loops):
        genus = genus_loops[i]
        if len(color_list) == 0:
            if genus == 0:
                color = (0,0,1)
            elif genus == 1:
                color = (1,0,0)
            elif genus > 1:
                color= (0,0,0)
        else:
            color = colormap[int(color_list[i]*510)]
        loop = smoothen_line(loop, window_length=window_loop, mode='wrap', 
                             N_out=int(N_ratio_loop*len(loop)))
        
        if wrap == True:
            loop = loop%N

        mlab.points3d(*(loop.T), scale_factor=radius, color=tuple(color))
