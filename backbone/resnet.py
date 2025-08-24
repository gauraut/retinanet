import torch, torch.nn as nn
from torchvision.models import resnet50


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        print("Initializing ResNet50 backbone")
        model = resnet50(weights=None if not pretrained else "DEFAULT")