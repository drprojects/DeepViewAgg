import torch


# -------------------------------------------------------------------- #
#                       Image Viewing Conditions                       #
# -------------------------------------------------------------------- #

def minmax_normalize(x, low=None, high=None):
    """Minmax rescaling of x to [0, 1]. Min and max bounds can be
    passed using low and high.

    :param x:
    :param low:
    :param high:
    :return:
    """
    if low is None:
        low = x.min()
    if high is None:
        high = x.max()
    return ((x.float() - low) / (high - low + 1e-4)).float()


def viewing_angle(u, v, requires_scaling=False):
    """Viewing angle is defined as |cos(theta)| with theta the angle
    between the u and v. By default, u and v are assumed to be already
    unit-scaled, use 'requires_scaling' if that is not the case.

    :param u:
    :param v:
    :param requires_scaling:
    :return:
    """
    orientation = torch.zeros(u.shape[0], device=u.device, dtype=torch.float)
    u = u.float()
    v = v.float()

    if v is None:
        return orientation

    if requires_scaling:
        u = u / (torch.linalg.norm(u, dim=1) + 1e-4).reshape((-1, 1)).float()
        v = v / (torch.linalg.norm(v, dim=1) + 1e-4).reshape((-1, 1)).float()

    orientation = (u * v).sum(dim=1).abs()
    # orientation[torch.where(orientation > 1)] = 0

    return orientation


# TODO: integrate the NeighborhoodBasedMappingFeatures computation here,
#  build this as a multimodal data trasnformer taking 1 image and the
#  points within range. This will make KNN much faster and easier. This
#  will induce slight NN errors for points near the search radius edge
#  but whatever...
def viewing_conditions(
        features=None, xyz_to_img=None, dist=None, x_proj=None, y_proj=None,
        normal=None, img_size=None, r_max=30, r_min=0.5):
    """Compute the N x F array of pointwise projection features carrying:
        - passed pointwise features
        - normalized distance
        - orientation to the normal
        - normalized pixel height
        - normalized pixel radius
    """
    if features is None:
        features = []
    elif len(features.shape) == 1:
        features = [features.view(-1, 1)]
    else:
        features = [features]

    # Normalized depth
    if dist is not None:
        features.append(minmax_normalize(dist, low=r_min, high=r_max))

    # Orientation to the normal
    if xyz_to_img is not None and normal is not None:
        features.append(viewing_angle(
            xyz_to_img, normal, requires_scaling=True).view(-1, 1))

    # Normalized pixel height
    if y_proj is not None and img_size is not None:
        features.append((y_proj / img_size[1]).float().view(-1, 1))

    # Normalized pixel radius
    if x_proj is not None and y_proj is not None and img_size is not None:
        radius = ((x_proj - img_size[0])**2 + (y_proj - img_size[1])**2).sqrt()
        radius = radius / max(img_size)
        features.append(radius.view(-1, 1))

    return torch.cat(features, 1)
