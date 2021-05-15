import torch
import torch.nn as nn

from ..utils import params_to_box_corners
from ..settings import LABEL_NUMBER
from .dla import DeepLayerAggregation


class LaserNet(nn.Module):

    def __init__(self, num_classes: int = LABEL_NUMBER):
        """
            LaserNet implementation from https://arxiv.org/pdf/1903.08701.pdf almost..

            params:
                num_classes - number of target classes
        """
        super(LaserNet, self).__init__()

        self.num_classes = num_classes

        self.dla = DeepLayerAggregation()

        self.classes = nn.Sequential(
            nn.Conv2d(128, self.num_classes, kernel_size=(1, 1)))  # no Softmax!

        # relative center (x, y), relative orientation (wx, wy) = (cos w, sin w), and dimensions l, w
        # + one channel for each class label
        self.bb_params = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128,
                      out_channels=6,
                      kernel_size=(1, 1)),
            nn.LeakyReLU())

        self.log_stds = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(128, 32)),
            nn.LeakyReLU()
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
            x - tensor of size (N, 7, width, height) with main features
                7 for (x, y, height(z), intensity, aziumth, distance
        """
        
        coordinates = x[:, :2]
        x_range_view = x[:, 2:]
        
        dla_out = self.dla(x_range_view)

        class_preds = self.classes(dla_out)
        bb_preds = self.bb_params(dla_out)
        log_stds = self.log_stds(dla_out).squeeze()

        # azimuth angle is feature [4] out of 5 channels
        angles = x[:, 4, :, :]

        # bb_preds is of size [N, 6, RV_WIDTH, RV_HEIGHT]
        # for each point on the RV, and each mixture component K we predict 6 params of a bounding box
        # these are relative to the camera frame, we need to turn them to absolute space
        # and we want to get 4 box corners instead of 6 params
        
        bb_corners = params_to_box_corners(bb_preds,
                                           coordinates[:, :2],
                                           angles)

        return class_preds, bb_corners, log_stds
