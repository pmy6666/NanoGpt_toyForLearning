import math

import torch
import torch.nn as nn


class GELUActivation(nn.Module):
    """
    采用 Transformer 中非常常见的 tanh 近似：
    GELU(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """

    def __init__(self, approximate: str = "tanh"):
        super().__init__()
        # 教学版只支持 tanh 近似，接口上保留 approximate，方便类比 PyTorch。
        if approximate != "tanh":
            raise ValueError("This minimal GELU only supports approximate='tanh'.")
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeff = math.sqrt(2.0 / math.pi)
        inner = coeff * (x + 0.044715 * x.pow(3))
        return 0.5 * x * (1.0 + torch.tanh(inner))
