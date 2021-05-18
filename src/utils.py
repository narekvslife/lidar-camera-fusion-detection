import torch
import numpy as np
from pyquaternion import Quaternion
import threading 
from src.settings import NUSCENES, RV_WIDTH, RV_HEIGHT, LABEL_NUMBER


def rotation_matrix(angles: torch.Tensor):
    """
    :param angles:
    :return:
    """

    cos, sin = torch.cos(angles), torch.sin(angles)
    s1 = torch.stack((cos, -sin))  # 2xNx128x32
    s2 = torch.stack((cos, sin))  # 2xNx128x32
    
    s3 = torch.stack((s1, s2)).squeeze(2) # 2x2xNx128x32

#     # check that stacking went as expected
#     assert s3[0, 1, 0, 0, 0] == s1[1, 0, 0, 0]
#     assert s3[1, 0, 0, 0, 0] == s2[0, 0, 0, 0]
#     assert s3[1, 1, 0, 0, 0] == s2[1, 0, 0, 0]

    return s3  # 2x2xNxRV_WIDTHxRV_HEIGHT 


def params_to_box_centers(point_center_params: torch.Tensor,
                          point_coordinates: torch.Tensor,
                          angles: torch.Tensor) -> torch.Tensor:  # 1
    """
    this function converts relative bounding box parameters to coordinates

    :param point_center_params:  N x 2 x RV_WIDTH x RV_HEIGHT
    :param point_coordinates: N x 2 x RV_WIDTH x RV_HEIGHT
    :param angles:
    :return:
    """
    rotation_matrices = rotation_matrix(angles)  # 2 x 2 x N x RV_WIDTH x RV_HEIGHT
    rotation_matrices = rotation_matrices.permute(2, 3, 4, 0, 1)  # N x RV_WIDTH x RV_HEIGHT x 2 x 2

    point_center_params = point_center_params.permute(0, 2, 3, 1).unsqueeze(4)  # N x RV_WIDTH x RV_HEIGHT x 2 x 1
    point_coordinates = point_coordinates.permute(0, 2, 3, 1).unsqueeze(4)  # N x RV_WIDTH x RV_HEIGHT x 2 x 1

    abs_coords = point_coordinates + rotation_matrices @ point_center_params  # N x RV_WIDTH x RV_HEIGHT x 2 x 1
    
    return abs_coords.squeeze(4).permute(0, 3, 1, 2)  # N x 2 x RV_WIDTH x RV_HEIGHT


def params_to_corners(bb_center_coords: torch.Tensor,
                      bb_orientations: torch.Tensor,
                      lengths: torch.Tensor,
                      widths: torch.Tensor) -> torch.Tensor:  # 2
    """

    :param bb_center_coords: N x 2 x RV_WIDTH x RV_HEIGHT
    :param bb_orientations:  N x RV_WIDTH x RV_HEIGHT
    :param lengths:  N x RV_WIDTH x RV_HEIGHT
    :param widths: N x RV_WIDTH x RV_HEIGHT
    :return:
    """
    

    R = rotation_matrix(bb_orientations).permute(2, 3, 4, 0, 1) # N x RV_WIDTH x RV_HEIGHT x 2 x 2

    print('|0|', torch.stack((lengths, widths)).shape)
    b1 = R @ torch.stack((lengths, widths)).permute(1, 2, 3, 0).unsqueeze(4) # N x RV_WIDTH x RV_HEIGHT x 2 x 1
    b2 = R @ torch.stack((lengths, -widths)).permute(1, 2, 3, 0).unsqueeze(4) 
    b3 = R @ torch.stack((-lengths, -widths)).permute(1, 2, 3, 0).unsqueeze(4) 
    b4 = R @ torch.stack((-lengths, widths)).permute(1, 2, 3, 0).unsqueeze(4)

    #     print("bb_abs_center_coords:", bb_abs_center_coords.shape, "bn:", b1.squeeze(4).permute(0, 3, 1, 2).shape)

    b1 = bb_center_coords + b1.squeeze(4).permute(0, 3, 1, 2) / 2 
    b2 = bb_center_coords + b2.squeeze(4).permute(0, 3, 1, 2) / 2
    b3 = bb_center_coords + b3.squeeze(4).permute(0, 3, 1, 2) / 2
    b4 = bb_center_coords + b4.squeeze(4).permute(0, 3, 1, 2) / 2

    b = torch.hstack((b1, b2, b3, b4))

    return b


def params_to_box_corners(bb_params: torch.Tensor,
                          point_coordinates: torch.Tensor,
                          angles: torch.Tensor) -> torch.Tensor:
    """
        This function turns relative predicted bounding box parameters
        into 4 coordinates of a box 
        
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
    
    angles = torch.deg2rad(angles)  # later in rotation matrix we will need radians, not degrees
    
    bb_center_coords = params_to_box_centers(point_centers_xy, point_coordinates, angles)  # N x 2 x RV_WIDTH x RV_HEIGHT

    bb_orientation = angles + torch.atan2(w_y, w_x)  # N x RV_WIDTH x RV_HEIGHT


    bb_corners = params_to_corners(bb_center_coords,
                                   bb_orientation,
                                   lengths,
                                   widths)

    return bb_corners


def get_front_bb(sample: dict, nusc):
    """
    Function computes and returns
    An array of points points of a bounding box in sensor coordinates.
    Each point is a (x, y) coordinate, each BB is 4 points

    :param sample: nuscenes sample dictionary
    :return: np.array of shape (N, 8, 3)
    """
    my_sample_lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    sample_annotation = nusc.get_boxes(my_sample_lidar_data['token'])

    ego_record = nusc.get('ego_pose', my_sample_lidar_data['ego_pose_token'])
    cs_record = nusc.get('calibrated_sensor', my_sample_lidar_data['calibrated_sensor_token'])

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


def get_bb_targets(coordinates, bounding_box_corners):
    """
    coordinate.shape (N, 3, RV_WIDTH, RV_HEIGHT)
    bounding_box_corners.shape (N, M_n, 4, 3) where M_n is different for each N
    bounding_box_labeles.shape (N, M_n, 1) where M_n is different for each N
    """

    bb_targets = []
    for sample_i in range(len(coordinates)):
        bbc_targets_single_rv = np.zeros((4, 3, RV_WIDTH, RV_HEIGHT))

        sample_coords = coordinates[sample_i]
        s_coords_x = sample_coords[0]
        s_coords_y = sample_coords[1]

        sample_boxes_corners = bounding_box_corners[sample_i]

        # bounding_box_corners[cs_i] - bounding box corners for according point cloud
        for bb_i in range(len(sample_boxes_corners)):
            bb_c = sample_boxes_corners[bb_i]  # bb_c[left_top, left_bottom, right_bottom, right_top]

            min_bb_x = bb_c[:, 0].min()
            max_bb_x = bb_c[:, 0].max()
            min_bb_y = bb_c[:, 1].min()
            max_bb_y = bb_c[:, 1].max()
        
            c1 = min_bb_x <= s_coords_x  # left_top/left_bottom.x <= coordinate.x
            c2 = max_bb_x >= s_coords_x  # right_bottom/right_top.x >= coordinate.x
            c3 = min_bb_y <= s_coords_y  # left/right_bottom.y <= coordinate.y
            c4 = max_bb_y >= s_coords_y  # right_top/left_top.y >= coordinate.y
            
            c = np.logical_and(np.logical_and(c1, c2),
                               np.logical_and(c3, c4))

            bbc_targets_single_rv[:, :, c] = np.expand_dims(bb_c, 2)

        bbc_targets_single_rv = bbc_targets_single_rv[:, :2].reshape((8, RV_WIDTH, RV_HEIGHT))

        bb_targets.append(bbc_targets_single_rv)

    return np.array(bb_targets)