import numpy as np
import numba as nb
from numba import njit



@njit(cache=True, nogil=True)
def pose_to_rotation_matrix_numba(opk):
    # Omega, Phi, Kappa cos and sin
    co = np.cos(opk[0])
    so = np.sin(opk[0])
    cp = np.cos(opk[1])
    sp = np.sin(opk[1])
    ck = np.cos(opk[2])
    sk = np.sin(opk[2])

    # Omega, Phi, Kappa inverse rotation matries
    M_o = np.array([[1.0, 0.0, 0.0],
                    [0.0,  co, -so],
                    [0.0,  so,  co]], dtype=np.float64)

    M_p = np.array([[cp, 0.0,  sp],
                    [0.0, 1.0, 0.0],
                    [-sp, 0.0,  cp]], dtype=np.float64)

    M_k = np.array([[ck, -sk, 0.0],
                    [sk,  ck, 0.0],
                    [0.0, 0.0, 1.0]], dtype=np.float64)

    # Global inverse rotation matrix to go from cartesian to
    # camera-system spherical coordinates
    M = np.dot(M_o, np.dot(M_p, M_k))

    return M

#-------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def norms_numba(v):
    norms = np.sqrt((v**2).sum(axis=1))
    return norms

#-------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def float_pixels_numba(xyz_to_img, radius, img_rotation, img_shape):
    # Convert point to camera coordinate system
    v = xyz_to_img.dot(img_rotation.transpose())
    
    # Equirectangular projection
    t = np.arctan2(v[:,1], v[:,0])
    p = np.arccos(v[:,2] / radius)
    
    # Angles to pixel position
    width, height = img_shape
    w_pix = ((width - 1) * (1 - t / np.pi) / 2) % width
    h_pix = ((height - 1) * p / np.pi) % height
    
    return w_pix, h_pix

#-------------------------------------------------------------------------------

@njit(cache=True, nogil=True, fastmath=True)
def field_of_view(x_pix, y_pix, crop_top, crop_bottom, mask=None):
    in_fov = np.logical_and(crop_top <= y_pix, y_pix < crop_bottom)
    if not mask is None:
        n_points = x_pix.shape[0]
        x_int = np.floor(x_pix).astype(np.uint32)
        y_int = np.floor(y_pix).astype(np.uint32)
        for i in range(n_points):
            if in_fov[i] and not mask[x_int[i], y_int[i]]:
                in_fov[i] = False
    return np.where(in_fov)[0]

#-------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def array_pixel_width_numba(y_pix, dist, img_shape=(1024,512), voxel=0.03, k=0.2, d=10):
    # Compute angular width
    # Pixel are grown based on their distance
    # Small angular widths assumption: tan(x)~x
    # Close-by points are further grown with a heuristic based on k and d
    angular_width = (1 + k * np.exp(-dist / np.log(d))) * voxel / dist
    
    # Compute Y angular width
    # NB: constant for equirectangular projection
    angular_res_y = angular_width * img_shape[1] / np.pi
    
    # Compute X angular width
    # NB: function of latitude for equirectangular projection
    a = angular_width * img_shape[0] / (2.0 * np.pi)
    b = np.pi / img_shape[1]
    angular_res_x = a / (np.sin(b * y_pix) + 0.001)
    
    # NB: stack+transpose faster than column stack
    return np.stack((angular_res_x, angular_res_y)).transpose()

#-------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def pixel_masks_numba(x_pix, y_pix, width_pix):
    x_a = np.empty_like(x_pix, dtype=np.float32)
    x_b = np.empty_like(x_pix, dtype=np.float32)
    y_a = np.empty_like(x_pix, dtype=np.float32)
    y_b = np.empty_like(x_pix, dtype=np.float32)
    np.round(x_pix - width_pix[:,0] / 2,     0, x_a)
    np.round(x_pix + width_pix[:,0] / 2 + 1, 0, x_b)
    np.round(y_pix - width_pix[:,1] / 2,     0, y_a)
    np.round(y_pix + width_pix[:,1] / 2 + 1, 0, y_b)
    return np.stack((x_a, x_b, y_a, y_b)).transpose().astype(np.int32)

#-------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def border_pixel_masks_numba(pix_masks, x_min, x_max, y_min, y_max):
    for i in range(pix_masks.shape[0]):
        if pix_masks[i, 0] < x_min:
            pix_masks[i, 0] = x_min
        if pix_masks[i, 1] > x_max:
            pix_masks[i, 1] = x_max
        if pix_masks[i, 2] < y_min:
            pix_masks[i, 2] = y_min
        if pix_masks[i, 3] > y_max:
            pix_masks[i, 3] = y_max
    return pix_masks

#-------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def compute_depth_map(
        xyz_to_img,
        img_opk,
        img_mask=None,
        img_size=(2048, 1024),
        crop_top=0,
        crop_bottom=0,
        voxel=0.1,
        r_max=30,
        r_min=0.5,
        growth_k=0.2,
        growth_r=10,
        empty=0):
    # Rotation matrix from image Euler angle pose
    img_rotation = pose_to_rotation_matrix_numba(img_opk)
    
    # Remove points outside of image range
    distances = norms_numba(xyz_to_img)
    in_range = np.where(np.logical_and(r_min < distances, distances < r_max))[0]

    # Project points to float pixel coordinates
    x_pix, y_pix = float_pixels_numba(xyz_to_img[in_range], distances[in_range], img_rotation,
        img_size)

    # Remove points outside of camera field of view
    in_fov = field_of_view(x_pix, y_pix, crop_top, img_size[1] - crop_bottom, mask=img_mask)
    
    # Compute projection pixel patches sizes 
    width_pix = array_pixel_width_numba(y_pix[in_fov], distances[in_range][in_fov], 
        img_shape=img_size, voxel=voxel, k=growth_k, d=growth_r)
    pix_masks = pixel_masks_numba(x_pix[in_fov], y_pix[in_fov], width_pix)
    pix_masks = border_pixel_masks_numba(pix_masks, 0, img_size[0], crop_top,
        img_size[1] - crop_bottom)
    pix_masks[:,2:] -= crop_top  # Remove y-crop offset

    # Cropped maps initialization
    cropped_img_size = (img_size[0], img_size[1] - crop_bottom - crop_top)
    depth_map = np.full(cropped_img_size, r_max + 1, np.float32)
    undistort = np.sin(np.pi * np.arange(crop_top,
        img_size[1] - crop_bottom) / img_size[1]) + 0.001
    
    # Loop through indices for points in range and in FOV
    distances = distances[in_range][in_fov]
    for i_point in range(distances.shape[0]):  

        point_dist = distances[i_point]
        point_pix_mask = pix_masks[i_point] 
        
        # Update maps where point is closest recorded
        x_a, x_b, y_a, y_b = point_pix_mask
        for x in range(x_a, x_b):
            for y in range(y_a, y_b):
                if point_dist < depth_map[x, y]:
                    depth_map[x, y] = point_dist
    
    # Set empty pixels to default empty value
    for x in range(depth_map.shape[0]):
        for y in range(depth_map.shape[1]):
            if depth_map[x, y] > r_max:
                depth_map[x, y] = empty
    
    # Restore the cropped areas
    cropped_map_top = np.full((img_size[0], crop_top), empty, np.float32)
    cropped_map_bottom = np.full((img_size[0], crop_bottom), empty, np.float32)
    depth_map = np.concatenate((cropped_map_top, depth_map, cropped_map_bottom), axis=1)
    
    return depth_map

#-------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def compute_rgb_map(
        xyz_to_img,
        rgb,
        img_opk,
        img_mask=None,
        img_size=(2048, 1024),
        crop_top=0,
        crop_bottom=0,
        voxel=0.1,
        r_max=30,
        r_min=0.5,
        growth_k=0.2,
        growth_r=10,
        empty=0):
    # Rotation matrix from image Euler angle pose
    img_rotation = pose_to_rotation_matrix_numba(img_opk)
    
    # Remove points outside of image range
    distances = norms_numba(xyz_to_img)
    in_range = np.where(np.logical_and(r_min < distances, distances < r_max))[0]

    # Project points to float pixel coordinates
    x_pix, y_pix = float_pixels_numba(xyz_to_img[in_range], distances[in_range],
        img_rotation, img_size)

    # Remove points outside of camera field of view
    in_fov = field_of_view(x_pix, y_pix, crop_top, img_size[1] - crop_bottom, mask=img_mask)
    
    # Compute projection pixel patches sizes 
    width_pix = array_pixel_width_numba(y_pix[in_fov], distances[in_range][in_fov],
        img_shape=img_size, voxel=voxel, k=growth_k, d=growth_r)
    pix_masks = pixel_masks_numba(x_pix[in_fov], y_pix[in_fov], width_pix)
    pix_masks = border_pixel_masks_numba(pix_masks, 0, img_size[0], crop_top,
        img_size[1] - crop_bottom)
    pix_masks[:,2:] -= crop_top  # Remove y-crop offset

    # Cropped maps initialization
    cropped_img_size = (img_size[0], img_size[1] - crop_bottom - crop_top)
    depth_map = np.full(cropped_img_size, r_max + 1, np.float32)
    rgb_map = np.zeros((*cropped_img_size, 3), dtype=np.int16)
    undistort = np.sin(np.pi * np.arange(crop_top,
        img_size[1] - crop_bottom) / img_size[1]) + 0.001
    
    # Loop through indices for points in range and in FOV
    distances = distances[in_range][in_fov]
    rgb = rgb[in_range][in_fov]
    for i_point in range(distances.shape[0]):  

        point_dist = distances[i_point]
        point_rgb = rgb[i_point]
        point_pix_mask = pix_masks[i_point] 
        
        # Update maps where point is closest recorded
        x_a, x_b, y_a, y_b = point_pix_mask
        for x in range(x_a, x_b):
            for y in range(y_a, y_b):
                if point_dist < depth_map[x, y]:
                    depth_map[x, y] = point_dist
                    rgb_map[x, y] = point_rgb
    
    # Set empty pixels to default empty value
    for x in range(depth_map.shape[0]):
        for y in range(depth_map.shape[1]):
            if depth_map[x, y] > r_max:
                depth_map[x, y] = empty
    
    # Restore the cropped areas
    cropped_map_top = np.full((img_size[0], crop_top), empty, np.float32)
    cropped_map_bottom = np.full((img_size[0], crop_bottom), empty, np.float32)
    depth_map = np.concatenate((cropped_map_top, depth_map, cropped_map_bottom), axis=1)
    
    cropped_map_top = np.zeros((img_size[0], crop_top, 3), dtype=np.uint8)
    cropped_map_bottom = np.zeros((img_size[0], crop_bottom, 3), dtype=np.uint8)
    rgb_map = np.concatenate((cropped_map_top, rgb_map, cropped_map_bottom), axis=1)
    
    return rgb_map, depth_map

#-------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def compute_index_map(
        xyz_to_img,
        indices,
        img_opk,
        img_mask=None,
        img_size=(1024, 512),
        crop_top=0,
        crop_bottom=0,
        voxel=0.1,
        r_max=30,
        r_min=0.5,
        growth_k=0.2,
        growth_r=10,
        empty=0):
    # We store indices in int64 format so we only accept indices up to 
    # np.iinfo(np.int64).max
    num_points = xyz_to_img.shape[0]
    if num_points >= 9223372036854775807:
        raise OverflowError
    
    # Rotation matrix from image Euler angle pose
    img_rotation = pose_to_rotation_matrix_numba(img_opk)
    
    # Remove points outside of image range
    distances = norms_numba(xyz_to_img)
    in_range = np.where(np.logical_and(r_min < distances, distances < r_max))[0]

    # Project points to float pixel coordinates
    x_pix, y_pix = float_pixels_numba(xyz_to_img[in_range], distances[in_range],
        img_rotation, img_size)

    # Remove points outside of camera field of view
    in_fov = field_of_view(x_pix, y_pix, crop_top, img_size[1] - crop_bottom, mask=img_mask)
    
    # Compute projection pixel patches sizes 
    width_pix = array_pixel_width_numba(y_pix[in_fov], distances[in_range][in_fov],
        img_shape=img_size, voxel=voxel, k=growth_k, d=growth_r)
    pix_masks = pixel_masks_numba(x_pix[in_fov], y_pix[in_fov], width_pix)
    pix_masks = border_pixel_masks_numba(pix_masks, 0, img_size[0], crop_top,
        img_size[1] - crop_bottom)
    pix_masks[:,2:] -= crop_top  # Remove y-crop offset

    # Cropped depth map initialization
    cropped_img_size = (img_size[0], img_size[1] - crop_bottom - crop_top)
    depth_map = np.full(cropped_img_size, r_max + 1, np.float32)
    
    # Indices map intitialization
    # We store indices in int64 so we assumes point indices are lower
    # than max int64 ~ 2.14 x 10^9.
    # We need the negative for empty pixels
    no_idx = -1
    idx_map = np.full(cropped_img_size, no_idx, dtype=np.int64)
    
    # Loop through indices for points in range and in FOV
    distances = distances[in_range][in_fov]
    indices = indices[in_range][in_fov]
    for i_point in range(distances.shape[0]):  

        point_dist = distances[i_point]
        point_idx = indices[i_point]
        point_pix_mask = pix_masks[i_point] 
        
        # Update maps where point is closest recorded
        x_a, x_b, y_a, y_b = point_pix_mask
        for x in range(x_a, x_b):
            for y in range(y_a, y_b):
                if point_dist < depth_map[x, y]:
                    depth_map[x, y] = point_dist
                    idx_map[x, y] = point_idx
    
    # Set empty pixels to default empty value
    for x in range(depth_map.shape[0]):
        for y in range(depth_map.shape[1]):
            if depth_map[x, y] > r_max:
                depth_map[x, y] = empty
    
    # Restore the cropped areas
    cropped_map_top = np.full((img_size[0], crop_top), empty, np.float32)
    cropped_map_bottom = np.full((img_size[0], crop_bottom), empty, np.float32)
    depth_map = np.concatenate((cropped_map_top, depth_map, cropped_map_bottom), axis=1)
    
    cropped_map_top = np.full((img_size[0], crop_top), no_idx, np.int64)
    cropped_map_bottom = np.full((img_size[0], crop_bottom), no_idx, np.int64)
    idx_map = np.concatenate((cropped_map_top, idx_map, cropped_map_bottom), axis=1)
    
    return idx_map, depth_map
