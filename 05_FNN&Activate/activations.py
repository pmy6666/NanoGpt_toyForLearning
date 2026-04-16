import torch.nn as nn

from gelu import GELUActivation
from relu import ReLUActivation
from silu import SiLUActivation


def build_activation(name: str, implementation: str = "torch") -> nn.Module:
    """
    统一的激活函数入口。
    - implementation='torch'：返回 PyTorch 内置模块
    - implementation='minimal'：返回当前目录下的教学版实现
    """

    name = name.lower()
    implementation = implementation.lower()

    if implementation == "torch":
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU(approximate="tanh")
        if name == "silu":
            return nn.SiLU()

    if implementation == "minimal":
        if name == "relu":
            return ReLUActivation()
        if name == "gelu":
            return GELUActivation()
        if name == "silu":
            return SiLUActivation()

    raise ValueError(
        f"Unsupported activation={name} or implementation={implementation}"
    )
