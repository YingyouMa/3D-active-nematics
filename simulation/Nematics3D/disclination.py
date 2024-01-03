# -----------------------------------------------
# Basic detection and analysis of disclinations
# Yingyou Ma, Physics @ Brandeis, 2023
# -----------------------------------------------

import numpy as np
import time

# -------------------------------------------------
# Detect the disclinations in the 3D director field
# -------------------------------------------------

def find_defect(n, threshold=0, print_time=False):

  now = time.time()

  N, M, L = np.shape(n)[:3]

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
  if print_time == True:
    print('finish x-direction, with', str(round(time.time()-now,2))+'s')
  now = time.time()

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
  if print_time == True:
    print('finish y-direction, with', str(round(time.time()-now,2))+'s')
  now = time.time()

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
  if print_time == True:
    print('finish z-direction, with', str(round(time.time()-now,2))+'s')
  now = time.time()

  return defect_indices

# -------------------------------------------------------------------------
# Sort the index of defect points within a disclination loop
# Minimize the distance between the pair of points with neighboring indices
# -------------------------------------------------------------------------

def sort_loop_indices(defect_indices):
  loop_indices = defect_indices[nearest_neighbor_order(defect_indices)]
  return loop_indices

def nearest_neighbor_order(points):
    from scipy.spatial.distance import cdist
    num_points = len(points)
    dist = cdist(points, points) 

    visited = np.zeros(num_points, dtype=bool)
    visited[0] = True
    order = [0]  

    for i in range(num_points - 1):
        current_point = order[-1]
        nearest_neighbor = np.argmin(dist[current_point, :] + visited * np.max(dist))
        order.append(nearest_neighbor)
        visited[nearest_neighbor] = True

    return order

# ----------------------------
# Smoothen a disclination loop
# ----------------------------

def smoothen_loop(loop_coord, window_ratio=3, order=3, N_out=160):

    pad = int(len(loop_coord)/window_ratio/2)*2 + 1

    from scipy.signal import savgol_filter
    from scipy.interpolate import splprep, splev
    loop_points = savgol_filter(loop_coord, pad, order, axis=0, mode='wrap')
    uspline = np.arange(len(loop_coord))/len(loop_coord)
    tck = splprep(loop_points.T, u=uspline, s=0)[0]
    new_indices = np.array(splev(np.linspace(0,1,N_out), tck)).T

    return new_indices

# --------------------------------------------------------------
# Visualize a disclination loop by the coordinates of each point
# --------------------------------------------------------------

def plot_loop(
            loop_coord, 
            tube_radius=0.25, tube_opacity=0.5, if_add_head=True,
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

    mlab.plot3d(*(loop_coord.T), tube_radius=tube_radius, opacity=tube_opacity)

# -----------------------------------------------------------------------------
# Given a local director field. Visualize the disclination loop if there is any
# -----------------------------------------------------------------------------

def plot_loop_from_n(
                    n_box, 
                    origin=[0,0,0], N=1, width=1, 
                    tube_radius=0.25, tube_opacity=0.5, if_add_head=True,
                    if_smooth=True, window_ratio=3, order=3, N_out=160
                    ):

    loop_indices = find_defect(n_box)
    if len(loop_indices) > 0:
        loop_indices = loop_indices + np.tile(origin, (np.shape(loop_indices)[0],1) )
        loop_coord = sort_loop_indices(loop_indices)/N*width
        if if_smooth == True:
            loop_coord = smoothen_loop(
                                    loop_coord,
                                    window_ratio=window_ratio, order=order, N_out=N_out
                                    )
        plot_loop(
                loop_coord, 
                tube_radius=tube_radius, tube_opacity=tube_opacity, 
                if_add_head=if_add_head
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

# ---------------------------------------------------
# Derive the averaged norm vecor of given coordinates
# ---------------------------------------------------

def get_plane(points):

    svd  = np.linalg.svd(points.T)
    left = svd[0]

    return left[:, -1]

# ------------------------------------------------------------------------------------
# Visualize the disclination loop with directors projected on principle planes
# ------------------------------------------------------------------------------------ 

def show_plane_2Ddirector(
                        n_box, height, 
                        color_axis=(1,0), height_visual=0,
                        space=3, line_width=2, density=1.5, 
                        if_omega=True, S_box=0, S_threshold=0.2
                          ):
    
    from mayavi import mlab
    
    from plotdefect import get_streamlines
    
    color_axis = color_axis / np.linalg.norm(color_axis) 
    
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
                density=density)
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

    colors = np.abs(np.einsum('ij, j -> i', unit, color_axis))
    colors = np.concatenate([colors, [colors[-1]]])
    nan_index = np.array(np.where(np.isnan(colors)==1))
    colors[nan_index] = colors[nan_index-1]
    colors[disconnect] = colors[disconnect-1]

    src = mlab.pipeline.scalar_scatter(src_x, src_y, src_z, colors)
    src.mlab_source.dataset.lines = connections
    src.update()

    lines = mlab.pipeline.stripper(src)
    plot_lines = mlab.pipeline.surface(lines, line_width=line_width)
    cb = mlab.colorbar(object=plot_lines, orientation='vertical', nb_labels=5, label_fmt='%.2f')
    cb.data_range = (0,1)
    cb.label_text_property.color = (0,0,0)

    if if_omega == True: 

        from scipy import ndimage

        binimg = S_box<S_threshold
        binimg[:height] = False
        binimg[(height+1):] = False
        binimg = ndimage.binary_dilation(binimg, iterations=1)
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



def show_loop_plane_2Ddirector(
                                n_box, S_box,
                                height_list, if_omega_list,
                                height_visual_list=0, if_rescale_loop=True,
                                figsize=(1920, 1360), bgcolor=(1,1,1), camera_set=0,
                                norm_length=20, color_axis=(1,0),
                                print_load_mayavi=False
                                ):
    
    if height_visual_list == 0:
        height_visual_list = height_list
        if_rescale_loop =False
    elif if_rescale_loop == True:
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

    x = np.arange(np.shape(S_box)[0])
    y = np.arange(np.shape(S_box)[1])
    z = np.arange(np.shape(S_box)[2])
    X, Y, Z = np.meshgrid(x,y,z, indexing='ij')

    if print_load_mayavi == True:
        now = time.time()
        from mayavi import mlab
        print(f'loading mayavi cost {round(time.time()-now, 2)}s')
    else:
        from mayavi import mlab

    mlab.figure(size=figsize, bgcolor=bgcolor)
    defect = find_defect(n_box)
    if len(defect) > 0:
        loop_indices = defect[nearest_neighbor_order(defect)]
        if if_rescale_loop == True:
            loop_indices[:,0] = parabola(loop_indices[:,0])
        loop_N = get_plane(loop_indices)
        loop_smooth = smoothen_loop(loop_indices)
        plot_loop(loop_smooth, tube_radius=0.75, tube_opacity=1)

        loop_center = loop_smooth.mean(axis=0)
        mlab.quiver3d(
        height_visual_list[0], loop_center[1], loop_center[2],
        *(loop_N),
        mode='arrow',
        color=(0,0,1),
        scale_factor=norm_length,
        opacity=0.5
        ) 

    show_plane_2Ddirector(n_box, height_list[0], height_visual=height_visual_list[0], if_omega=True, S_box=S_box)
    show_plane_2Ddirector(n_box, height_list[1], height_visual=height_visual_list[1], if_omega=True, S_box=S_box)
    show_plane_2Ddirector(n_box, height_list[2], height_visual=height_visual_list[2], if_omega=True, S_box=S_box)
    if camera_set != 0: 
        mlab.view(*camera_set[:3], roll=camera_set[3])

 

