import torch
import torch.nn as nn
import torch.nn.functional as F


class BoxRegressor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.blocks_list = nn.ModuleList([(lambda: nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        ))() for _ in range(4)])

        self.tower = nn.Sequential(self.blocks_list[0],
                                   self.blocks_list[1],
                                   self.blocks_list[2],
                                   self.blocks_list[3])
        
        self.box_regressor_head = nn.Conv2d(in_channels, 4*9, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, pylayers):
        outs = []
        for pl in pylayers:
            bl = self.tower(pl)
            bl = self.box_regressor_head(bl)

            outs.append(bl)

        return outs