import torch
import torch.nn as nn
import torch.nn.functional as F
import net_part
from net_part import FeatureExtractor
from submodule import convbn


class S_ShapedBlock(nn.Module):
    def __init__(self):
        super(S_ShapedBlock, self).__init__()
        #### First
        self.firstconv = nn.Sequential(convbn(1, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))




class ColorNet(nn.Module):
    """
    Colorization of IR image
    train with format WxH = 640x480
    """
    def __init__(self):
        self.concat_channels = 32
        super(ColorNet, self).__init__()
        self.featureExtractor = FeatureExtractor()
        self.hourglass = Hourglass(2)

    def forward(self,  x1):
        x_color = self.featureExtractor(x1)
        return x_color

