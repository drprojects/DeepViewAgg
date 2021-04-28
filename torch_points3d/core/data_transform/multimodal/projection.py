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
                    [0.0, co, -so],
                    [0.0, so, co]], dtype=np.float32)

    M_p = np.array([[cp, 0.0, sp],
                    [0.0, 1.0, 0.0],
                    [-sp, 0.0, cp]], dtype=np.float32)

    M_k = np.array([[ck, -sk, 0.0],
                    [sk, ck, 0.0],
                    [0.0, 0.0, 1.0]], dtype=np.float32)

    # Global inverse rotation matrix to go from cartesian to
    # camera-system spherical coordinates
    M = np.dot(M_o, np.dot(M_p, M_k))

    return M


# -------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def norms_numba(v):
    norms = np.sqrt((v ** 2).sum(axis=1))
    return norms


# -------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def float_pixels_numba(xyz_to_img, radius, img_rotation, img_shape):
    # Convert point to camera coordinate system
    v = xyz_to_img.dot(img_rotation.transpose())

    # Equirectangular projection
    t = np.arctan2(v[:, 1], v[:, 0])
    p = np.arccos(v[:, 2] / radius)

    # Angles to pixel position
    width, height = img_shape
    w_pix = ((width - 1) * (1 - t / np.pi) / 2) % width
    h_pix = ((height - 1) * p / np.pi) % height

    return w_pix, h_pix


# -------------------------------------------------------------------------------

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


# -------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def array_pixel_width_numba(y_pix, dist, img_shape=(1024, 512), voxel=0.03,
        k=0.2, d=10):
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


# -------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def pixel_masks_numba(x_pix, y_pix, width_pix):
    x_a = np.empty_like(x_pix, dtype=np.float32)
    x_b = np.empty_like(x_pix, dtype=np.float32)
    y_a = np.empty_like(x_pix, dtype=np.float32)
    y_b = np.empty_like(x_pix, dtype=np.float32)
    np.round(x_pix - width_pix[:, 0] / 2, 0, x_a)
    np.round(x_pix + width_pix[:, 0] / 2 + 1, 0, x_b)
    np.round(y_pix - width_pix[:, 1] / 2, 0, y_a)
    np.round(y_pix + width_pix[:, 1] / 2 + 1, 0, y_b)
    return np.stack((x_a, x_b, y_a, y_b)).transpose().astype(np.int32)


# -------------------------------------------------------------------------------

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


# -------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def normalize_distance_numba(dist, low=None, high=None):
    d_min = low
    d_max = high
    dist = dist.astype(np.float32)
    if low is None:
        d_min = dist.min()
    if high is None:
        d_max = dist.max()
    return ((dist - d_min) / (d_max + 1e-4)).astype(np.float32)


# -------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def orientation_numba(u, v, requires_scaling=False):
    """Orientation is defined as |cos(theta)| with theta the angle
    between the u and v. By default, u and v are assumed to be already
    unit-scaled, use 'requires_scaling' if that is not the case.
    """
    orientation = np.zeros(u.shape[0], dtype=np.float32)
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    if v is None:
        return orientation

    if requires_scaling:
        u = u / (norms_numba(u) + 1e-4).reshape((-1, 1)).astype(np.float32)
        v = v / (norms_numba(v) + 1e-4).reshape((-1, 1)).astype(np.float32)

    orientation = np.abs((u * v).sum(axis=1))
    # idx = np.where(orientation > 1)[0]
    # orientation[idx] = 0

    return orientation


# -------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def normalize_height_numba(pixel_height, height):
    return (pixel_height / height).astype(np.float32)


# -------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def compute_depth_map(
        xyz_to_img,
        img_opk,
        img_mask=None,
        proj_size=(2048, 1024),
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
        img_rotation, proj_size)

    # Remove points outside of camera field of view
    in_fov = field_of_view(x_pix, y_pix, crop_top, proj_size[1] - crop_bottom,
        mask=img_mask)

    # Compute projection pixel patches sizes 
    width_pix = array_pixel_width_numba(y_pix[in_fov],
        distances[in_range][in_fov], img_shape=proj_size, voxel=voxel,
        k=growth_k, d=growth_r)
    pix_masks = pixel_masks_numba(x_pix[in_fov], y_pix[in_fov], width_pix)
    pix_masks = border_pixel_masks_numba(pix_masks, 0, proj_size[0], crop_top,
        proj_size[1] - crop_bottom)
    pix_masks[:, 2:] -= crop_top  # Remove y-crop offset

    # Cropped maps initialization
    cropped_img_size = (proj_size[0], proj_size[1] - crop_bottom - crop_top)
    depth_map = np.full(cropped_img_size, r_max + 1, np.float32)
    undistort = np.sin(np.pi * np.arange(crop_top,
        proj_size[1] - crop_bottom) / proj_size[1]) + 0.001

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
    cropped_map_top = np.full((proj_size[0], crop_top), empty, np.float32)
    cropped_map_bottom = np.full((proj_size[0], crop_bottom), empty, np.float32)
    depth_map = np.concatenate((cropped_map_top, depth_map, cropped_map_bottom),
        axis=1)

    return depth_map


# -------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def compute_rgb_map(
        xyz_to_img,
        rgb,
        img_opk,
        img_mask=None,
        proj_size=(2048, 1024),
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
                                      img_rotation, proj_size)

    # Remove points outside of camera field of view
    in_fov = field_of_view(x_pix, y_pix, crop_top, proj_size[1] - crop_bottom,
        mask=img_mask)

    # Compute projection pixel patches sizes 
    width_pix = array_pixel_width_numba(y_pix[in_fov],
        distances[in_range][in_fov], img_shape=proj_size, voxel=voxel,
        k=growth_k, d=growth_r)
    pix_masks = pixel_masks_numba(x_pix[in_fov], y_pix[in_fov], width_pix)
    pix_masks = border_pixel_masks_numba(pix_masks, 0, proj_size[0], crop_top,
                                         proj_size[1] - crop_bottom)
    pix_masks[:, 2:] -= crop_top  # Remove y-crop offset

    # Cropped maps initialization
    cropped_img_size = (proj_size[0], proj_size[1] - crop_bottom - crop_top)
    depth_map = np.full(cropped_img_size, r_max + 1, np.float32)
    rgb_map = np.zeros((*cropped_img_size, 3), dtype=np.int16)
    undistort = np.sin(np.pi * np.arange(crop_top,
        proj_size[1] - crop_bottom) / proj_size[1]) + 0.001

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
    cropped_map_top = np.full((proj_size[0], crop_top), empty, np.float32)
    cropped_map_bottom = np.full((proj_size[0], crop_bottom), empty, np.float32)
    depth_map = np.concatenate((cropped_map_top, depth_map, cropped_map_bottom),
        axis=1)

    cropped_map_top = np.zeros((proj_size[0], crop_top, 3), dtype=np.uint8)
    cropped_map_bottom = np.zeros((proj_size[0], crop_bottom, 3),
        dtype=np.uint8)
    rgb_map = np.concatenate((cropped_map_top, rgb_map, cropped_map_bottom),
        axis=1)

    return rgb_map, depth_map


# -------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def compute_index_map(
        xyz_to_img,
        indices,
        img_opk,
        img_mask=None,
        proj_size=(1024, 512),
        crop_top=0,
        crop_bottom=0,
        voxel=0.1,
        r_max=30,
        r_min=0.5,
        growth_k=0.2,
        growth_r=10,
        empty=0,
        no_id=-1):
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
                                      img_rotation, proj_size)

    # Remove points outside of camera field of view
    in_fov = field_of_view(x_pix, y_pix, crop_top, proj_size[1] - crop_bottom,
        mask=img_mask)

    # Compute projection pixel patches sizes
    width_pix = array_pixel_width_numba(y_pix[in_fov],
        distances[in_range][in_fov], img_shape=proj_size, voxel=voxel,
        k=growth_k, d=growth_r)
    pix_masks = pixel_masks_numba(x_pix[in_fov], y_pix[in_fov], width_pix)
    pix_masks = border_pixel_masks_numba(pix_masks, 0, proj_size[0], crop_top,
         proj_size[1] - crop_bottom)
    pix_masks[:, 2:] -= crop_top  # Remove y-crop offset

    # Cropped depth map initialization
    cropped_img_size = (proj_size[0], proj_size[1] - crop_bottom - crop_top)
    depth_map = np.full(cropped_img_size, r_max + 1, np.float32)

    # Indices map intitialization
    # We store indices in int64 so we assumes point indices are lower
    # than max int64 ~ 2.14 x 10^9.
    # We need the negative for empty pixels
    idx_map = np.full(cropped_img_size, no_id, dtype=np.int64)

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
    cropped_map_top = np.full((proj_size[0], crop_top), empty, np.float32)
    cropped_map_bottom = np.full((proj_size[0], crop_bottom), empty, np.float32)
    depth_map = np.concatenate((cropped_map_top, depth_map, cropped_map_bottom),
                               axis=1)

    cropped_map_top = np.full((proj_size[0], crop_top), no_id, np.int64)
    cropped_map_bottom = np.full((proj_size[0], crop_bottom), no_id, np.int64)
    idx_map = np.concatenate((cropped_map_top, idx_map, cropped_map_bottom),
                             axis=1)

    return idx_map, depth_map


# -------------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def compute_projection(
        xyz_to_img,
        indices,
        img_opk,
        linearity=None,
        planarity=None,
        scattering=None,
        normals=None,
        img_mask=None,
        proj_size=(1024, 512),
        crop_top=0,
        crop_bottom=0,
        voxel=0.1,
        r_max=30,
        r_min=0.5,
        growth_k=0.2,
        growth_r=10,
        empty=0,
        no_id=-1,
        exact=False):
    # We store indices in int64 format so we only accept indices up to
    # np.iinfo(np.int64).max
    num_points = xyz_to_img.shape[0]
    if num_points >= 9223372036854775807:
        raise OverflowError

    # Initialize pointwise geometric features to zero if not provided
    if linearity is None:
        linearity = np.zeros(xyz_to_img.shape[0], dtype=np.float32)
    if planarity is None:
        planarity = np.zeros(xyz_to_img.shape[0], dtype=np.float32)
    if scattering is None:
        scattering = np.zeros(xyz_to_img.shape[0], dtype=np.float32)
    if normals is None:
        normals = np.zeros((xyz_to_img.shape[0], 3), dtype=np.float32)
    linearity = linearity.astype(np.float32)
    planarity = planarity.astype(np.float32)
    scattering = scattering.astype(np.float32)
    normals = normals.astype(np.float32)

    # Rotation matrix from image Euler angle pose
    img_rotation = pose_to_rotation_matrix_numba(img_opk)

    # Remove points outside of image range
    distances = norms_numba(xyz_to_img)
    in_range = np.where(np.logical_and(r_min < distances, distances < r_max))[0]
    xyz_to_img = xyz_to_img[in_range]
    distances = distances[in_range]
    indices = indices[in_range]
    linearity = linearity[in_range]
    planarity = planarity[in_range]
    scattering = scattering[in_range]
    normals = normals[in_range]

    # Project points to float pixel coordinates
    x_pix, y_pix = float_pixels_numba(xyz_to_img, distances, img_rotation,
        proj_size)

    # Remove points outside of camera field of view
    in_fov = field_of_view(x_pix, y_pix, crop_top, proj_size[1] - crop_bottom,
        mask=img_mask)
    xyz_to_img = xyz_to_img[in_fov]
    distances = distances[in_fov]
    indices = indices[in_fov]
    linearity = linearity[in_fov]
    planarity = planarity[in_fov]
    scattering = scattering[in_fov]
    normals = normals[in_fov]
    x_pix = x_pix[in_fov]
    y_pix = y_pix[in_fov]

    # Compute projection pixel patches sizes
    width_pix = array_pixel_width_numba(y_pix, distances, img_shape=proj_size,
        voxel=voxel, k=growth_k, d=growth_r)
    pix_masks = pixel_masks_numba(x_pix, y_pix, width_pix)
    pix_masks = border_pixel_masks_numba(pix_masks, 0, proj_size[0], crop_top,
         proj_size[1] - crop_bottom)
    pix_masks[:, 2:] -= crop_top  # Remove y-crop offset

    # Compute the N x F array of pointwise projection features carrying:
    #     - normalized depth
    #     - linearity
    #     - planarity
    #     - scattering
    #     - orientation to the surface
    #     - normalized pixel height
    depth = normalize_distance_numba(distances, low=r_min, high=r_max)
    orientation = orientation_numba(
        xyz_to_img / (distances + 1e-4).reshape((-1, 1)),
        normals)
    height = normalize_height_numba(y_pix, proj_size[1])
    features = np.column_stack((
        depth,
        linearity,
        planarity,
        scattering,
        orientation,
        height))
    n_feat = features.shape[1]

    # Cropped depth map initialization
    cropped_img_size = (proj_size[0], proj_size[1] - crop_bottom - crop_top)
    depth_map = np.full(cropped_img_size, r_max + 1, dtype=np.float32)

    # Cropped indices map initialization
    # We store indices in int64 so we assumes point indices are lower
    # than max int64 ~ 2.14 x 10^9.
    # We need the negative for empty pixels
    idx_map = np.full(cropped_img_size, no_id, dtype=np.int64)

    # Cropped feature map initialization
    feat_map = np.zeros((*cropped_img_size, n_feat), dtype=np.float32)

    # Cropped local indices map initialization
    # This map is useful when 'exact=True', to keep track of seen
    # points' local indices. These indices can then be used to
    # efficiently build the 'exact' maps without the need for 'np.isin',
    # which is not supported un numba.
    if exact:
        local_idx_map = np.full(cropped_img_size, no_id, dtype=np.int64)

    # Loop through indices for points in range and in FOV
    for i_point in range(distances.shape[0]):

        point_dist = distances[i_point]
        point_idx = indices[i_point]
        point_pix_mask = pix_masks[i_point]
        point_feat = features[i_point]

        # Update maps where point is closest recorded
        x_a, x_b, y_a, y_b = point_pix_mask
        for x in range(x_a, x_b):
            for y in range(y_a, y_b):
                if point_dist < depth_map[x, y]:
                    depth_map[x, y] = point_dist
                    if exact:
                        # Store the local indices if 'exact=True' mode.
                        # These indices can then be used to efficiently
                        # build the 'exact' maps without the need for
                        # 'np.isin', which is not supported un numba.
                        idx_map[x, y] = i_point
                    else:
                        # Store the real point indices otherwise
                        idx_map[x, y] = point_idx
                        feat_map[x, y] = point_feat

    # When 'exact=True', we use the results from the previous projection
    # to extract the seen points. The output maps are sparse, as points
    # are only projected to their central pixel coordinates, without
    # artificial growth masks.
    if exact:
        # Recover the local indices of seen points
        idx_seen = np.unique(idx_map)
        idx_seen = idx_seen[idx_seen != no_id]

        # Reinitialize the output maps
        depth_map = np.full(cropped_img_size, r_max + 1, dtype=np.float32)
        idx_map = np.full(cropped_img_size, no_id, dtype=np.int64)

        # Convert the pixel projection coordinates to int
        x_pix = x_pix.astype(np.int32)
        y_pix = y_pix.astype(np.int32)

        # Loop through the seen points only and populate the maps
        # only at the central projection pixels (not the masks) and
        # without checking distances anymore.
        if idx_seen.shape[0] > 0:
            for i_point in idx_seen:

                point_dist = distances[i_point]
                point_idx = indices[i_point]
                point_feat = features[i_point]
                x = x_pix[i_point]
                y = y_pix[i_point]

                # Update maps without worrying about occlusions here
                depth_map[x, y] = point_dist
                idx_map[x, y] = point_idx
                feat_map[x, y] = point_feat

    # Set empty pixels to default empty value
    for x in range(depth_map.shape[0]):
        for y in range(depth_map.shape[1]):
            if depth_map[x, y] > r_max:
                depth_map[x, y] = empty

    # Restore the cropped areas
    cropped_map_top = np.full((proj_size[0], crop_top), empty, np.float32)
    cropped_map_bottom = np.full((proj_size[0], crop_bottom), empty, np.float32)
    depth_map = np.concatenate((cropped_map_top, depth_map, cropped_map_bottom),
        axis=1)

    cropped_map_top = np.full((proj_size[0], crop_top), no_id, np.int64)
    cropped_map_bottom = np.full((proj_size[0], crop_bottom), no_id, np.int64)
    idx_map = np.concatenate((cropped_map_top, idx_map, cropped_map_bottom),
        axis=1)

    cropped_map_top = np.zeros((proj_size[0], crop_top, n_feat), np.float32)
    cropped_map_bottom = np.zeros((proj_size[0], crop_bottom, n_feat), np.float32)
    feat_map = np.concatenate((cropped_map_top, feat_map, cropped_map_bottom),
        axis=1)

    return idx_map, depth_map, feat_map

# TODO : all-torch GPU-parallelized projection ? Rather than iteratively
#  populating the depth map, create a set of target pixel coordinates and
#  associated meta-data (distance, point ID, normal orientation, ...). Then
#  use torch-GPU operations to extract the meta-data with the smallest
#  distance for each pixel coordinate ? This seems possible with torch.scatter_.
#  To understand how, see:
#    - torch_points3d.utils.multimodal.lexargunique
#    - https://medium.com/@yang6367/understand-torch-scatter-b0fd6275331c
