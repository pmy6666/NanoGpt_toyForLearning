from dataclasses import dataclass

import torch
import torch.nn as nn

from activations import build_activation


@dataclass
class FFNConfig:
    n_embd: int
    dropout: float = 0.1
    expansion_factor: int = 4
    activation: str = "gelu"
    activation_impl: str = "torch"
    gated: bool = False
    bias: bool = True


class PositionWiseFFN(nn.Module):
    """
    Transformer 中的 position-wise FFN。

    它对每个 token 的最后一维独立应用相同参数：
    [B, T, C] -> [B, T, hidden] -> [B, T, C]
    """

    def __init__(self, config: FFNConfig):
        super().__init__()
        hidden_dim = config.expansion_factor * config.n_embd
        self.n_embd = config.n_embd
        self.gated = config.gated

        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gate_proj = (
            nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            if config.gated
            else None
        )
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.activation = build_activation(
            config.activation, implementation=config.activation_impl
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.n_embd:
            raise ValueError(
                f"PositionWiseFFN expects last dim {self.n_embd}, got {x.size(-1)}"
            )

        if self.gated:
            hidden = self.activation(self.gate_proj(x)) * self.up_proj(x)
        else:
            hidden = self.activation(self.up_proj(x))

        out = self.down_proj(hidden)
        out = self.dropout(out)
        return out


class MLP(nn.Module):
    """
    与当前目录原始接口兼容的 FFN/MLP 封装。
    """

    def __init__(self, config):
        super().__init__()
        ffn_config = FFNConfig(
            n_embd=config.n_embd,
            dropout=getattr(config, "dropout", 0.1),
            expansion_factor=getattr(config, "expansion_factor", 4),
            activation=getattr(config, "activation", "gelu"),
            activation_impl=getattr(config, "activation_impl", "torch"),
            gated=getattr(config, "gated", False),
            bias=getattr(config, "bias", True),
        )
        self.ffn = PositionWiseFFN(ffn_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def print_activation_examples() -> None:
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    activation_names = ["relu", "gelu", "silu"]

    print("=== Activation outputs on a small vector ===")
    print(f"input: {x.tolist()}")
    for name in activation_names:
        y = build_activation(name, implementation="minimal")(x)
        row = ", ".join(f"{v:.4f}" for v in y.tolist())
        print(f"{name:>4s}: [{row}]")
    print()


def print_activation_compare() -> None:
    x = torch.linspace(-3.0, 3.0, steps=9)
    activation_names = ["relu", "gelu", "silu"]

    print("=== Minimal implementation vs torch implementation ===")
    for name in activation_names:
        y_minimal = build_activation(name, implementation="minimal")(x)
        y_torch = build_activation(name, implementation="torch")(x)
        max_error = (y_minimal - y_torch).abs().max().item()
        print(f"{name:>4s} max error: {max_error:.8f}")
    print()


def print_ffn_example() -> None:
    x = torch.randn(2, 4, 8)

    standard_ffn = PositionWiseFFN(
        FFNConfig(
            n_embd=8,
            dropout=0.1,
            expansion_factor=4,
            activation="gelu",
            activation_impl="minimal",
            gated=False,
        )
    )
    swiglu_ffn = PositionWiseFFN(
        FFNConfig(
            n_embd=8,
            dropout=0.1,
            expansion_factor=4,
            activation="silu",
            activation_impl="minimal",
            gated=True,
        )
    )

    y1 = standard_ffn(x)
    y2 = swiglu_ffn(x)

    print("=== Standard FFN ===")
    print(f"input shape : {tuple(x.shape)}")
    print(f"output shape: {tuple(y1.shape)}")
    print(f"params      : {count_parameters(standard_ffn)}")
    print()

    print("=== Gated FFN (SwiGLU style) ===")
    print(f"input shape : {tuple(x.shape)}")
    print(f"output shape: {tuple(y2.shape)}")
    print(f"params      : {count_parameters(swiglu_ffn)}")


def demo() -> None:
    print_activation_examples()
    print_activation_compare()
    print_ffn_example()


if __name__ == "__main__":
    demo()
