import torch


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


def params_to_abs_center(point_centers: torch.Tensor,
                         absolute_coordinates: torch.Tensor,
                         angles: torch.Tensor) -> torch.Tensor:  # 1
    """
        this function converts relative bounding box center to absolute coordinates

        for this we need the absolute coordinates of the ego, and the ego rotation

    :param point_centers:
    :param absolute_coordinates:
    :param angles:
    :return:
    """

    rotation_matrices = rotation_matrix(angles)  # torch.Size([2, 2, 10, 256, 32])
    rotation_matrices = rotation_matrices.permute(2, 4, 3, 0, 1)  # torch.Size([10, 32, 256, 2, 2])

    #     print(rotation_matrices.shape, centerX_centerY.shape, coordinates.shape)
    #     torch.Size([2, 2, 10, 256, 32]) torch.Size([10, 2, 256, 32]) torch.Size([10, 2, 256, 32])

    point_centers = point_centers.permute(0, 3, 2, 1).unsqueeze(4)  # TODO:
    absolute_coordinates = absolute_coordinates.permute(0, 3, 2, 1).unsqueeze(4)  # TODO:

    #     torch.Size([2, 2, 10, 256, 32]) torch.Size([10, 32, 256, 2, 1]) torch.Size([10, 32, 256, 2, 1])
    #     print("1| AC", absolute_coordinates.shape, "RM", rotation_matrices.shape, "point_centers", point_centers.shape)
    abs_coords = absolute_coordinates + rotation_matrices @ point_centers

    return abs_coords.squeeze(4).permute(0, 3, 2, 1)  # TODO: transformation to separate function


def abs_params_to_corners(bb_abs_center_coords: torch.Tensor,
                          bb_abs_orientation: torch.Tensor,
                          lengths: torch.Tensor,
                          widths: torch.Tensor) -> torch.Tensor:  # 2
    """

    :param bb_abs_center_coords:
    :param bb_abs_orientation:
    :param lengths:
    :param widths:
    :return:
    """
    R = rotation_matrix(bb_abs_orientation).permute(2, 3, 4, 0, 1)

    #     print("ROTATION_MATRIX:", R.shape, "lw:", torch.stack((lengths,widths)).unsqueeze(4).permute(1, 2, 3, 0, 4).shape)

    b1 = R @ torch.stack((lengths, widths)).unsqueeze(4).permute(1, 2, 3, 0, 4)  # TODO:
    b2 = R @ torch.stack((lengths, -widths)).unsqueeze(4).permute(1, 2, 3, 0, 4)  # TODO:
    b3 = R @ torch.stack((-lengths, -widths)).unsqueeze(4).permute(1, 2, 3, 0, 4)  # TODO:
    b4 = R @ torch.stack((-lengths, widths)).unsqueeze(4).permute(1, 2, 3, 0, 4)  # TODO:

    #     print("bb_abs_center_coords:", bb_abs_center_coords.shape, "bn:", b1.squeeze(4).permute(0, 3, 1, 2).shape)

    b1 = bb_abs_center_coords + b1.squeeze(4).permute(0, 3, 1, 2)  # TODO:
    b2 = bb_abs_center_coords + b2.squeeze(4).permute(0, 3, 1, 2)  # TODO:
    b3 = bb_abs_center_coords + b3.squeeze(4).permute(0, 3, 1, 2)  # TODO:
    b4 = bb_abs_center_coords + b4.squeeze(4).permute(0, 3, 1, 2)  # TODO:

    b = torch.hstack((b1, b2, b3, b4)) / 2

    return b


def rel_params_to_abs_box_corners(bb_params: torch.Tensor,
                                  absolute_coordinates: torch.Tensor,
                                  angles: torch.Tensor,
                                  K: int) -> torch.Tensor:
    """
        This function turns relative predicted bounding box parameters
        into 4 coordinates of a box in an absolute coordinate system

    :param bb_params: tensor of size [N, K * 6, RV_WIDTH, RV_HEIGHT],
                      6 components are  [d_x, d_y, w_x, w_y, length, width]
    :param absolute_coordinates:
    :param angles:
    :return: tensor of size    [N, K * 8, RV_WIDTH, RV_HEIGHT]
    """

    #   bb_params.shape  === [10, 18, 128, 32]

    bb_param_number = 6
    N, _, RV_WIDTH, RV_HEIGHT = bb_params.shape

    bb_abs_corners = torch.zeros((N, K * 8, RV_WIDTH, RV_HEIGHT))

    #     print("KKKKKKKKKK", K)
    for k in range(K):
        index_margin = k * bb_param_number

        (centerX_centerY,
         w_x, w_y,
         lenghts, widths) = (bb_params[:, index_margin: index_margin + 2],
                             bb_params[:, index_margin + 2], bb_params[:, index_margin + 3],
                             bb_params[:, index_margin + 4], bb_params[:, index_margin + 5])

        # getting bb centers in absolute coordinates

        #         print("0|", centerX_centerY.shape, absolute_coordinates.shape, angles.shape)
        bb_abs_center_coords = params_to_abs_center(centerX_centerY, absolute_coordinates, angles)

        bb_abs_orientation = angles + torch.atan2(w_y, w_x)

        bb_abs_corners_K = abs_params_to_corners(bb_abs_center_coords,
                                                 bb_abs_orientation,
                                                 lenghts,
                                                 widths)
        #         torch.Size([10, 8, 128, 32]) torch.Size([10, 128, 32, 4, 2])

        #         print("INDEX MARGIN", index_margin)
        bb_abs_corners[:, k * 8: (k + 1) * 8, :, :] = bb_abs_corners_K
    return bb_abs_corners