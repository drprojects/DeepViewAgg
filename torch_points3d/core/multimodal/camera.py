import torch
import numpy as np


class Camera:
    def __init__(self):

        # load intrinsics
        self.load_intrinsics(self.intrinsic_file)

        # load poses
        poses = np.loadtxt(self.pose_file)
        frames = poses[:, 0]
        poses = np.reshape(poses[:, 1:], [-1, 3, 4])
        self.cam2world = {}
        self.frames = frames
        for frame, pose in zip(frames, poses):
            pose = np.concatenate((pose, np.array([0., 0., 0., 1.]).reshape(1, 4)))
            # consider the rectification for perspective cameras
            if self.cam_id == 0 or self.cam_id == 1:
                self.cam2world[frame] = np.matmul(np.matmul(pose, self.camToPose),
                                                  np.linalg.inv(self.R_rect))
            # fisheye cameras
            elif self.cam_id == 2 or self.cam_id == 3:
                self.cam2world[frame] = np.matmul(pose, self.camToPose)
            else:
                raise RuntimeError('Unknown Camera ID!')

    def world_to_cam(self, points, R, T, inverse=False):
        assert (points.ndim == R.ndim)
        assert (T.ndim == R.ndim or T.ndim == (R.ndim - 1))
        ndim = R.ndim
        if ndim == 2:
            R = np.expand_dims(R, 0)
            T = np.reshape(T, [1, -1, 3])
            points = np.expand_dims(points, 0)
        if not inverse:
            points = np.matmul(R, points.transpose(0, 2, 1)).transpose(0, 2, 1) + T
        else:
            points = np.matmul(R.transpose(0, 2, 1), (points - T).transpose(0, 2, 1))

        if ndim == 2:
            points = points[0]

        return points

    def cam_to_image(self, points):
        raise NotImplementedError

    def load_intrinsics(self, intrinsic_file):
        raise NotImplementedError

    def project(self, vertices, frameId, inverse=True):

        # current camera pose
        curr_pose = self.cam2world[frameId]
        T = curr_pose[:3, 3]
        R = curr_pose[:3, :3]

        # convert points from world coordinate to local coordinate
        points_local = self.world2cam(vertices, R, T, inverse)

        # perspective projection
        u, v, depth = self.cam2image(points_local)

        return (u, v), depth

    def __call__(self, obj3d, frameId):

        vertices = obj3d.vertices

        uv, depth = self.project_vertices(vertices, frameId)

        obj3d.vertices_proj = uv
        obj3d.vertices_depth = depth
        obj3d.generateMeshes()