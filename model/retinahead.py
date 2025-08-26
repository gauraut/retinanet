import torch, torch.nn as nn
import torch.nn.functional as F

class RetinaHead(nn.Module):
    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        