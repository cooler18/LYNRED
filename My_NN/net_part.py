#!/usr/bin/python
# full assembly of the sub-parts to form the complete net
from .submodule import *


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # self.reductionFactor = 3
        self.conv1 = nn.Sequential(convbn(1, 64, 3, 2, 1, 1), nn.ReLU(inplace=True),
                                  convbn(64, 128, 3, 1, 1, 1), nn.ReLU(inplace=True),
                                  convbn(128, 256, 3, 1, 1, 1), nn.ReLU(inplace=True),
                                  convbn(256, 512, 3, 1, 1, 1), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        gwc_feature = torch.cat((l2, l3, l4), dim=1)
        return {"gwc_feature": gwc_feature}
