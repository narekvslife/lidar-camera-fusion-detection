import torch
import torch.nn as nn

from src.utils import params_to_box_corners
from src.settings import LABEL_NUMBER
from src.models.dla import DeepLayerAggregation


class LaserNetSeg(nn.Module):

    def __init__(self, num_classes: int = LABEL_NUMBER):
        """
            LaserNet implementation from https://arxiv.org/pdf/1903.08701.pdf almost..

            params:
                num_classes - number of target classes
        """
        super(LaserNetSeg, self).__init__()

        self.num_classes = num_classes

        self.dla = DeepLayerAggregation()

        self.classes = nn.Conv2d(128, self.num_classes, kernel_size=(1, 1))


    def forward(self, x: torch.Tensor) -> tuple:
        """
            x - tensor of size (N, 7, width, height) with main features
                7 for (x, y, height(z), intensity, aziumth, distance
        """
        
        x_range_view = x[:, 2:]
        
        dla_out = self.dla(x_range_view)

        class_preds = self.classes(dla_out)

        return class_preds
