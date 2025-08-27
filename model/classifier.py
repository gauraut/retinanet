import torch
import torch.nn as nn
import torch.nn.functional as F
import math

conv_relu = lambda in_c, out_c, k=3, s=1, p=1: nn.Sequential(
    nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
    nn.ReLU(inplace=True)
)

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes=5):
        super().__init__()
        self.blocks = nn.ModuleList([(lambda: nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        ))() for _ in range(4)])

        self.classifier_head = nn.Conv2d(in_channels, 9*num_classes, 3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0.0)

        prior = 0.01
        bias_value = -math.log((1-prior)/prior)
        nn.init.constant_(self.classifier_head.bias, bias_value)
        
    def forward(self, pylayers):
        out = []

        for pylayer in pylayers:
            bl1 = self.blocks[0](pylayer)
            bl2 = self.blocks[1](bl1)
            bl3 = self.blocks[2](bl2)
            bl4 = self.blocks[3](bl3)

            class_with_logits = self.classifier_head(bl4)
            out.append(class_with_logits)

        return out