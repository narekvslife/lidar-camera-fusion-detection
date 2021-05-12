import torch
import numpy as np
from pyquaternion import Quaternion

from .settings import NUSCENES


def rotation_matrix(angles: torch.Tensor):
    """

    :param angles:
    :return:
    """

    theta = torch.deg2rad(angles)
    cos, sin = torch.cos(theta), torch.sin(theta)

    s1 = torch.stack((cos, -sin))
    s2 = torch.stack((cos, sin))

    s3 = torch.stack((s1, s2)).squeeze(2)
    return s3


def params_to_coordinates(point_center_params: torch.Tensor,
                          point_coordinates: torch.Tensor,
                          angles: torch.Tensor) -> torch.Tensor:  # 1
    """
        this function converts relative bounding box parameters to absolute coordinates

    :param point_center_params:
    :param point_coordinates:
    :param angles:
    :return:
    """

    rotation_matrices = rotation_matrix(angles)  # torch.Size([2, 2, N, 256, 32])
    rotation_matrices = rotation_matrices.permute(2, 4, 3, 0, 1)  # torch.Size([N, 32, 256, 2, 2])

#     print(rotation_matrices.shape, centerX_centerY.shape, coordinates.shape)
#     torch.Size([2, 2, 10, 256, 32]) torch.Size([10, 2, 256, 32]) torch.Size([N, 2, 256, 32])

    point_center_params = point_center_params.permute(0, 3, 2, 1).unsqueeze(4)  # TODO:
    point_coordinates = point_coordinates.permute(0, 3, 2, 1).unsqueeze(4)  # TODO:

#     torch.Size([2, 2, N, 256, 32]) torch.Size([N, 32, 256, 2, 1]) torch.Size([N, 32, 256, 2, 1])
#     print("1| AC", absolute_coordinates.shape, "RM", rotation_matrices.shape, "point_centers", point_centers.shape)
    abs_coords = point_coordinates + rotation_matrices @ point_center_params

    return abs_coords.squeeze(4).permute(0, 3, 2, 1)  # TODO:


def params_to_corners(bb_center_coords: torch.Tensor,
                      bb_orientations: torch.Tensor,
                      lengths: torch.Tensor,
                      widths: torch.Tensor) -> torch.Tensor:  # 2
    """

    :param bb_center_coords:
    :param bb_orientations:
    :param lengths:
    :param widths:
    :return:
    """
    R = rotation_matrix(bb_orientations).permute(2, 3, 4, 0, 1)

#     print("ROTATION_MATRIX:", R.shape, "lw:", torch.stack((lengths,widths)).unsqueeze(4).permute(1, 2, 3, 0, 4).shape)

    b1 = R @ torch.stack((lengths, widths)).unsqueeze(4).permute(1, 2, 3, 0, 4)  # TODO:
    b2 = R @ torch.stack((lengths, -widths)).unsqueeze(4).permute(1, 2, 3, 0, 4)  # TODO:
    b3 = R @ torch.stack((-lengths, -widths)).unsqueeze(4).permute(1, 2, 3, 0, 4)  # TODO:
    b4 = R @ torch.stack((-lengths, widths)).unsqueeze(4).permute(1, 2, 3, 0, 4)  # TODO:

#     print("bb_abs_center_coords:", bb_abs_center_coords.shape, "bn:", b1.squeeze(4).permute(0, 3, 1, 2).shape)

    b1 = bb_center_coords + b1.squeeze(4).permute(0, 3, 1, 2)  # TODO:
    b2 = bb_center_coords + b2.squeeze(4).permute(0, 3, 1, 2)  # TODO:
    b3 = bb_center_coords + b3.squeeze(4).permute(0, 3, 1, 2)  # TODO:
    b4 = bb_center_coords + b4.squeeze(4).permute(0, 3, 1, 2)  # TODO:

    b = torch.hstack((b1, b2, b3, b4)) / 2

    return b


def params_to_box_corners(bb_params: torch.Tensor,
                          point_coordinates: torch.Tensor,
                          angles: torch.Tensor) -> torch.Tensor:
    """
        This function turns relative predicted bounding box parameters
        into 4 coordinates of a box in an absolute coordinate system

    :param bb_params: tensor of size [N, 6, RV_WIDTH, RV_HEIGHT],
                      6 components are  [d_x, d_y, w_x, w_y, length, width]
    :param point_coordinates: point coordinates in vehicle ego space
    :param angles:
    :return: tensor of size    [N, 8, RV_WIDTH, RV_HEIGHT]
    """

    (point_centers_xy,
     w_x, w_y,
     lengths, widths) = (bb_params[:, :2],
                         bb_params[:, 2], bb_params[:, 3],
                         bb_params[:, 4], bb_params[:, 5])

    # TODO: to class(?)
    bb_abs_center_coords = params_to_coordinates(point_centers_xy, point_coordinates, angles)

    bb_abs_orientation = angles + torch.atan2(w_y, w_x)

    bb_abs_corners = params_to_corners(bb_abs_center_coords,
                                       bb_abs_orientation,
                                       lengths,
                                       widths)

    return bb_abs_corners


def get_front_bb(sample: dict):
    """
    Function computes and returns
    An array of points points of a bounding box in sensor coordinates.
    Each point is a (x, y) coordinate, each BB is 4 points

    :param sample: nuscenes sample dictionary
    :return: np.array of shape (N, 8, 3)
    """
    my_sample_lidar_data = NUSCENES.get('sample_data', sample['data']['LIDAR_TOP'])
    sample_annotation = NUSCENES.get_boxes(my_sample_lidar_data['token'])

    ego_record = NUSCENES.get('ego_pose', my_sample_lidar_data['ego_pose_token'])
    cs_record = NUSCENES.get('calibrated_sensor', my_sample_lidar_data['calibrated_sensor_token'])

    # first step: transform from absolute to ego
    ego_translation = -np.array(ego_record['translation'])

    # second step: transform from ego to sensor
    cs_translation = -np.array(cs_record['translation'])

    corners = []
    for box in sample_annotation:

        box.translate(ego_translation)
        box.rotate(Quaternion(ego_record['rotation']).inverse)

        box.translate(cs_translation)
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        # at this point bounding boxes are in sensor coordinate system
        # now we want to exclude such BB that do not have their center
        # lying in the front 90 degrees

        if box.center[1] <= 0:
            continue

        box.center_azimuth = np.degrees(np.arctan(box.center[0] / box.center[1]))

        # Transform front azimuth to be in range from 0 to 180
        box.center_azimuth = 90 - box.center_azimuth
        if not (45 < box.center_azimuth < 135):
            continue

        corners.append(box.bottom_corners())

    return np.transpose(np.array(corners).T, (2, 0, 1))


def get_bb_targets(coordinates, bounding_boxes):
    """
    coordinate.shape (N, 3, RV_WIDTH, RV_HEIGHT)
    bounding_boxes.shape (N, M_n, 4, 3) where M_n is different for each N
    """
    bb_targets = []
    n, _, rv_width, rv_height = coordinates.shape
    for cs_i in range(n):
        bb_targets_single_rv = np.zeros((4, 3, rv_width, rv_height))
        for bb in bounding_boxes[cs_i]:  # bounding_boxes[cs_i] - bounding box corners for according point cloud
            # bb[0] - left_top coordinates
            # bb[1] - left_bottom coordinates
            # bb[2] - right_bottom coordinates
            # bb[3] - right_top coordinates

            # bb[x][0] - x coordinate
            # bb[x][1] - y coordinate

            coords = coordinates[cs_i]
            c1 = np.logical_or(bb[0, 0] <= coords[0], bb[1, 0] <= coords[0])  # left_top/left_bottom.x <= coordinate.x
            c2 = np.logical_or(bb[2, 0] >= coords[0], bb[3, 0] >= coords[0])  # right_bottom/right_top.x >= coordinate.x
            c3 = np.logical_or(bb[1, 1] <= coords[1], bb[2, 1] <= coords[1])  # left/right_bottom.y <= coordinate.y
            c4 = np.logical_or(bb[3, 1] >= coords[1], bb[0, 1] >= coords[1])  # right_top/left_top.y >= coordinate.y

            c = np.logical_and(np.logical_and(c1, c2),
                               np.logical_and(c3, c4))

            bb_targets_single_rv[:, :, c] = np.expand_dims(bb, 2)
        bb_targets.append(bb_targets_single_rv.transpose((2, 3, 0, 1)))

    return np.array(bb_targets)
