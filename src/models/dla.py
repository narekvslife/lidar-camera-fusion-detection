from torch import nn, Tensor, cat
from src.models.helper_modules import bn_conv3x3, ResidualConv3x3Block


class FeatureExtractorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride: tuple, n: int = 6):
        super(FeatureExtractorBlock, self).__init__()

        self.first_conv_block = nn.Sequential(
            bn_conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride),
            bn_conv3x3(in_channels=out_channels, out_channels=out_channels),
        )
        self.transformed_x = bn_conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.residual_block = ResidualConv3x3Block(out_channels, out_channels)
        
        self.number_of_blocks = n

    def forward(self, x: Tensor) -> Tensor:
        
        out = self.first_conv_block(x)
        transformed_x = self.transformed_x(x)

        out += transformed_x

        for _ in range(self.number_of_blocks):
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

    def forward(self, fine_x: Tensor, coarse_x: Tensor) -> Tensor:
        coarse_x = self.coarse_deconv(coarse_x)

        x = cat((fine_x, coarse_x), 1)  # concatenate channels

        out = self.first_conv_block(x)
        transformed_x = self.transformed_x(x)
        out += transformed_x
        out = self.elu(out)

        out = self.residual_block2(out)

        return out


class DeepLayerAggregation(nn.Module):
    def __init__(self, in_channels):
        super(DeepLayerAggregation, self).__init__()

        self.fe_1a = FeatureExtractorBlock(in_channels=in_channels,
                                           out_channels=64,
                                           stride=(1, 1))

        self.fe_2a = FeatureExtractorBlock(in_channels=64,
                                           out_channels=64,
                                           stride=(2, 1))
        self.fa_1b = FeatureAggregatorBlock(in_channels_fine=64,
                                            in_channels_coarse=64,
                                            out_channels=64)

        self.fe_3a = FeatureExtractorBlock(in_channels=64,
                                           out_channels=128,
                                           stride=(2, 1))
        self.fa_2b = FeatureAggregatorBlock(in_channels_fine=64,
                                            in_channels_coarse=128,
                                            out_channels=128)
        self.fa_1c = FeatureAggregatorBlock(in_channels_fine=64,
                                            in_channels_coarse=128,
                                            out_channels=128)

    def forward(self, x: Tensor):
        out_1a = self.fe_1a(x)

        out_2a = self.fe_2a(out_1a)
        out_1b = self.fa_1b(out_1a, out_2a)

        out_3a = self.fe_3a(out_2a)
        out_2b = self.fa_2b(out_2a, out_3a)
        out_1c = self.fa_1c(out_1b, out_2b)

        return out_1c
