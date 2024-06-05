import torch
import torch.nn as nn


class SimpleDummyModel(nn.Module):
    def __init__(self):
        super(SimpleDummyModel, self).__init__()

    def forward(self, x):
        x = torch.mean(x, (1, 2, 3))
        x = torch.relu(x)
        return x
