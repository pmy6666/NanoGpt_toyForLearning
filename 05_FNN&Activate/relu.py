import torch
import torch.nn as nn


class ReLUActivation(nn.Module):
    """
    只保留最核心的前向逻辑：
    ReLU(x) = max(0, x)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x, torch.zeros_like(x))
