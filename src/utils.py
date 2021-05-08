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
                         absolute_coordinates:     torch.Tensor,
                         angles:          torch.Tensor) -> torch.Tensor:
    """
        this function converts relative bounding box center to absolute coordinates

        for this we need the absolute coordinates of the ego, and the ego rotation

        then R
    :param point_centers:
    :param absolute_coordinates:
    :param angles:
    :return:
    """

    rotation_matrices = rotation_matrix(angles)  # torch.Size([2, 2, 10, 256, 32])
    rotation_matrices = rotation_matrices.permute(2, 4, 3, 0, 1)  # torch.Size([10, 32, 256, 2, 2])

    #         print(rotation_matrices.shape, centerX_centerY.shape, coordinates.shape)
    #         torch.Size([2, 2, 10, 256, 32]) torch.Size([10, 2, 256, 32]) torch.Size([10, 2, 256, 32])

    point_centers = point_centers.permute(0, 3, 2, 1).unsqueeze(4)  # TODO:
    absolute_coordinates = absolute_coordinates.permute(0, 3, 2, 1).unsqueeze(4)  # TODO:

    #         print(rotation_matrices.shape, centerX_centerY.shape, coordinates.shape)
    #         torch.Size([2, 2, 10, 256, 32]) torch.Size([10, 32, 256, 2, 1]) torch.Size([10, 32, 256, 2, 1])

    abs_coords = absolute_coordinates + rotation_matrices @ point_centers

    return abs_coords.squeeze(4).permute(0, 3, 2, 1)  # TODO: transformation to separate function


def abs_params_to_corners(bb_abs_center_coords: torch.Tensor,
                          bb_abs_orientation:   torch.Tensor,
                          lengths:              torch.Tensor,
                          widths:               torch.Tensor) -> torch.Tensor:
    """

    :param bb_abs_center_coords:
    :param bb_abs_orientation:
    :param lengths:
    :param widths:
    :return:
    """
    R = rotation_matrix(bb_abs_orientation).permute(2, 3, 4, 0, 1)

    #         print("ROTATION_MATRIX:", R.shape,\
    #               "lw:", torch.stack((lengths,widths)).unsqueeze(4).permute(1, 2, 3, 0, 4).shape)

    b1 = R @ torch.stack((lengths,   widths)).unsqueeze(4).permute(1, 2, 3, 0, 4)  # TODO:
    b2 = R @ torch.stack((lengths,  -widths)).unsqueeze(4).permute(1, 2, 3, 0, 4)  # TODO:
    b3 = R @ torch.stack((-lengths,  -widths)).unsqueeze(4).permute(1, 2, 3, 0, 4)  # TODO:
    b4 = R @ torch.stack((-lengths,   widths)).unsqueeze(4).permute(1, 2, 3, 0, 4)  # TODO:

    #         print("bb_abs_center_coords:", bb_abs_center_coords.shape,\
    #               "bn:", b1.squeeze(4).permute(0, 3, 1, 2).shape)
    b1 = bb_abs_center_coords + b1.squeeze(4).permute(0, 3, 1, 2)  # TODO:
    b2 = bb_abs_center_coords + b2.squeeze(4).permute(0, 3, 1, 2)  # TODO:
    b3 = bb_abs_center_coords + b3.squeeze(4).permute(0, 3, 1, 2)  # TODO:
    b4 = bb_abs_center_coords + b4.squeeze(4).permute(0, 3, 1, 2)  # TODO:

    b = torch.stack((b1, b2, b3, b4)) / 2

    return b.permute(1, 3, 4, 0, 2)


def params_to_abs_box_corners(bb_params:   torch.Tensor,
                              absolute_coordinates: torch.Tensor,
                              angles:      torch.Tensor) -> torch.Tensor:
    """
        This function turns relative predicted bounding box parameters
        into 4 coordinates of a box in an absolute coordinate system

    :param bb_params: tensor of size [N, K * 6, RV_WIDTH, RV_HEIGHT],
                      6 components are  [d_x, d_y, w_x, w_y, length, width]
    :param absolute_coordinates:
    :param angles:
    :return: tensor of size    [N, K * 4, RV_WIDTH, RV_HEIGHT]
    """

    (centerX_centerY,
     w_x,     w_y,
     lenghts, widths) = (bb_params[:, :2],
                         bb_params[:, 2], bb_params[:, 3],
                         bb_params[:, 4], bb_params[:, 5])

    # getting bb centers in absolute coordinates
    bb_abs_center_coords = params_to_abs_center(centerX_centerY, absolute_coordinates, angles)

    bb_abs_orientation = angles + torch.atan2(w_y, w_x)

    bb_abs_corners = abs_params_to_corners(bb_abs_center_coords,
                                           bb_abs_orientation,
                                           lenghts,
                                           widths)

    return bb_abs_corners
