import torch
import torch.nn as nn


def bn_conv3x3(in_channels: int,
               out_channels: int,
               stride: int = 1,
               padding: int = 1,
               activation=nn.LeakyReLU):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
        activation())


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


class FeatureExtractorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample: bool = True):
        super(FeatureExtractorBlock, self).__init__()

        if downsample:
            stride = (2, 1)
        else:
            stride = (1, 1)

        self.first_conv_block = nn.Sequential(
            bn_conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride),
            bn_conv3x3(in_channels=out_channels, out_channels=out_channels),
        )
        self.transformed_x = bn_conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.elu = nn.ELU()

        self.residual_block = ResidualConv3x3Block(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.first_conv_block(x)
        transformed_x = self.transformed_x(x)

        out += transformed_x
        out = self.elu(out)

        for _ in range(6):
            out = self.residual_block(out)

        return out


class FeatureAggregatorBlock(nn.Module):

    def __init__(self, in_channels_fine, in_channels_coarse, out_channels):
        super(FeatureAggregatorBlock, self).__init__()

        self.in_channels_fine = in_channels_fine
        self.in_channels_coarse = in_channels_coarse
        self.out_channels = out_channels

        self.coarse_deconv = nn.ConvTranspose2d(in_channels=self.in_channels_coarse,
                                                out_channels=self.in_channels_coarse,
                                                kernel_size=(2, 1),
                                                stride=(2, 1))

        self.first_conv_block = nn.Sequential(
            bn_conv3x3(in_channels=in_channels_fine + in_channels_coarse, out_channels=out_channels),
            bn_conv3x3(in_channels=out_channels, out_channels=out_channels)
        )

        self.transformed_x = bn_conv3x3(in_channels=in_channels_fine + in_channels_coarse,
                                        out_channels=out_channels)
        self.elu = nn.ELU()

        self.residual_block2 = ResidualConv3x3Block(out_channels, out_channels)

    def forward(self, fine_x: torch.Tensor, coarse_x: torch.Tensor) -> torch.Tensor:
        coarse_x = self.coarse_deconv(coarse_x)

        x = torch.cat((fine_x, coarse_x), 1)  # concatenate channels

        out = self.first_conv_block(x)
        transformed_x = self.transformed_x(x)
        out += transformed_x
        out = self.elu(out)

        out = self.residual_block2(out)

        return out
