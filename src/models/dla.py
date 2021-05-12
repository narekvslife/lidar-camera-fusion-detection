from torch import nn, Tensor
from .helper_modules import FeatureExtractorBlock, FeatureAggregatorBlock


class DeepLayerAggregation(nn.Module):
    def __init__(self):
        super(DeepLayerAggregation, self).__init__()

        self.fe_1a = FeatureExtractorBlock(in_channels=5,
                                           out_channels=64,
                                           downsample=False)

        self.fe_2a = FeatureExtractorBlock(in_channels=64,
                                           out_channels=64)
        self.fa_1b = FeatureAggregatorBlock(in_channels_fine=64,
                                            in_channels_coarse=64,
                                            out_channels=64)

        self.fe_3a = FeatureExtractorBlock(in_channels=64,
                                           out_channels=128)
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
