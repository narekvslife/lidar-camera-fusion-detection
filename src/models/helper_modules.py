import torch
import torch.nn as nn

def bn_conv3x3(in_channels: int,
               out_channels: int,
               stride: int = 1,
               padding: int = 1,
               activation=nn.LeakyReLU()):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        activation)


class ResidualConv3x3Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super(ResidualConv3x3Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = bn_conv3x3(self.in_channels, self.out_channels, stride=self.stride)(x)
        out = bn_conv3x3(self.out_channels, self.out_channels, stride=self.stride)(out)
        out = self.leaky_relu(out)

        out += x
        out = self.leaky_relu(out)

        return out
