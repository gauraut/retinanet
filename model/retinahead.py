import torch
import torch.nn as nn
import torch.nn.functional as F
from .classifier import Classifier
from .box_regressor import BoxRegressor

class RetinaHead(nn.Module):
    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        self.classifier_head = Classifier(in_channels, num_classes)
        self.regression = BoxRegressor(in_channels)

    def forward(self, pylayers):
        classifications = self.classifier_head(pylayers)
        regressions = self.regression(pylayers)

        return classifications, regressions
        