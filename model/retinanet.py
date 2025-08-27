import torch
import torch.nn as nn
from backbone.resnet import ResNetBackbone
from .fpn import FPN
from .retinahead import RetinaHead


class RetinaNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.resnet50 = ResNetBackbone(pretrained=True)
        self.fpn = FPN(self.resnet50.out_channels, 256)
        self.head = RetinaHead(256, num_classes)


    def forward(self, x): # x: Batch*3*512*512
        C3, C4, C5 = self.resnet50(x)
        pylayers = self.fpn([C3, C4, C5])
        classifications, regressions = self.head(pylayers)

        return classifications, regressions