from backbone import ResNetBackbone
import torch, torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, channels, out_chn = 256):
        super().__init__()
        # self.l2 = nn.Conv2d(channels[0], channels[1], 1) # Not Needed
        self.l3 = nn.Conv2d(channels[0], out_chn, 1)
        self.l4 = nn.Conv2d(channels[1], out_chn, 1)
        self.l5 = nn.Conv2d(channels[2], out_chn, 1)
        self.p3 = nn.Conv2d(out_chn, out_chn, 3, padding=1)
        self.p4 = nn.Conv2d(out_chn, out_chn, 3, padding=1)
        self.p5 = nn.Conv2d(out_chn, out_chn, 3, padding=1)
        self.p6 = nn.Conv2d(out_chn, out_chn, 3, stride=2, padding=1)
        self.p7 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(out_chn, out_chn, 3, stride=2, padding=1))

    def forward(self, c3, c4, c5):
        lat5 = self.l5(c5)
        py5 = self.p5(lat5)
        lat4 = self.l4(c4) + F.interpolate(lat5, size=c4.shape[-2:], mode='nearest')
        py4 = self.p4(lat4)
        lat3 = self.l3(c3) + F.interpolate(lat4, size=c3.shape[-2:], mode='nearest')
        py3 = self.p3(lat3)
        py6 = self.p6(py5)
        py7 = self.p7(py6)

        return [py3, py4, py5, py6, py7]