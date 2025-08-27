import torch, torch.nn as nn
from torchvision.models import resnet50


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        print("Initializing ResNet50 backbone")
        model = resnet50(weights=None if not pretrained else "DEFAULT")
        self.out_channels = [512, 1024, 2048]

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool) # First layer

        # C2-5 stages
        self.c2 = model.layer1
        self.c3 = model.layer2
        self.c4 = model.layer3
        self.c5 = model.layer4

    def forward(self, x):
        x = self.stem(x)
        conv2 = self.c2(x)
        conv3 = self.c3(conv2)
        conv4 = self.c4(conv3)
        conv5 = self.c5(conv4)

        return conv3, conv4, conv5