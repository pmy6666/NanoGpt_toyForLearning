import torch
import torch.nn as nn


def sigmod(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))

class SiLUActivation(nn.Module):
    """
    SiLU(x) = x * sigmoid(x)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
