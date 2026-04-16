# FFN：Transformer 中的前馈网络

## 1. FFN 是做什么的

在一个标准 Transformer Block 中，最核心的两个子模块是：

- Self-Attention
- Feed-Forward Network (FFN)

Self-Attention 负责让一个 token 看见其他 token；
FFN 则负责对每个 token 的表示做更强的非线性映射。

所以它们的分工可以概括为：

- Attention：做 token 和 token 之间的信息交互；
- FFN：做 token 自身特征空间里的变换。

这也是为什么 FFN 经常被称为 `position-wise FFN`：
它对每个位置上的向量独立应用同一个 MLP，不在这个模块里混合不同位置。

如果输入张量形状是：

$$
x \in \mathbb{R}^{B \times T \times C}
$$

那么 FFN 的输入输出通常都是：

$$
\mathbb{R}^{B \times T \times C} \to \mathbb{R}^{B \times T \times C}
$$

其中：

- `B` 是 batch size
- `T` 是 sequence length
- `C` 是 embedding dimension

## 2. 标准 FFN 公式

最经典的 Transformer FFN 形式是：

$$
\mathrm{FFN}(x) = W_2 \, \phi(W_1 x + b_1) + b_2
$$

其中：

- `W1: C -> d_ff`
- `W2: d_ff -> C`
- `phi` 是激活函数，例如 `ReLU` 或 `GELU`
- `d_ff` 往往取 `4C`

写成更常见的结构就是：

$$
\mathrm{Linear}(C, d_{ff})
\rightarrow \mathrm{Activation}
\rightarrow \mathrm{Linear}(d_{ff}, C)
\rightarrow \mathrm{Dropout}
$$

如果把 `d_ff = 4C`，就得到最常见版本：

$$
\mathrm{Linear}(C, 4C)
\rightarrow \mathrm{Activation}
\rightarrow \mathrm{Linear}(4C, C)
$$

## 3. 为什么要先升维再降维

这是 FFN 的关键。

如果只是做：

$$
\mathrm{Linear}(C, C)
$$

那么它本质上只是一个线性变换，表达能力有限。

而标准 FFN 选择：

$$
C \rightarrow 4C \rightarrow C
$$

有两个主要原因：

1. 升维后，模型能在更大的隐藏空间里组合特征。
2. 中间加入非线性激活后，整个模块不再是简单的线性映射。

也就是说，FFN 的本质不是“再做一次投影”，而是：

- 先把特征展开；
- 在高维空间做非线性变换；
- 再压缩回原始维度。

这一步对表示能力提升非常重要。

## 4. 为什么说它是 position-wise

设输入为：

$$
x.\mathrm{shape} = [B, T, C]
$$

FFN 对每个 `x[b, t, :]` 单独应用同一组参数。

也就是说：

$$
y[b, t, :] = \mathrm{FFN}(x[b, t, :])
$$

注意这里不会发生：

- `t1` 和 `t2` 之间的信息交互；
- 序列维度上的卷积或池化；
- 不同 token 的混合。

所以：

- Attention 负责跨 token 交互；
- FFN 只负责 token 内部变换。

这两种能力恰好互补。

## 5. FFN 在 Transformer Block 中的位置

一个典型的 Pre-LN Transformer Block 写法是：

$$
x = x + \mathrm{Attention}(\mathrm{LN}(x))
$$

$$
x = x + \mathrm{FFN}(\mathrm{LN}(x))
$$

也就是：

1. 先归一化；
2. 进入注意力层；
3. 做残差连接；
4. 再归一化；
5. 进入 FFN；
6. 再做残差连接。

在这个结构里，FFN 不改变张量整体形状，但会显著改变表示内容。

## 6. 常见激活函数

### ReLU

最早期、最经典的激活函数：

$$
\mathrm{ReLU}(x) = \max(0, x)
$$

优点：

- 简单；
- 计算便宜；
- 容易理解。

缺点：

- 负半轴直接截断；
- 在一些任务里表达平滑性不如 GELU。

### GELU

现代 Transformer 非常常见：

$$
\mathrm{GELU}(x) = x \cdot \Phi(x)
$$

其中 `Phi(x)` 是标准高斯分布的累积分布函数。

直观上，GELU 不是简单地“砍掉负数”，而是做一种更平滑的门控。

优点：

- 更平滑；
- 在 Transformer/LLM 中表现通常优于 ReLU。

### SiLU / Swish

$$
\mathrm{SiLU}(x) = x \cdot \mathrm{sigmoid}(x)
$$

SiLU 本身常作为 gated FFN 的门控激活。

## 7. 主流变体：Gated FFN / SwiGLU

除了标准 FFN，现代大模型常使用 gated 结构。

一种典型写法是：

$$
\mathrm{hidden} = \mathrm{SiLU}(W_g x) \cdot (W_u x)
$$

$$
\mathrm{out} = W_o \, \mathrm{hidden}
$$

这里和标准 FFN 的区别在于：

- 不再只有一个升维分支；
- 多了一个 gate 分支；
- 两个分支逐元素相乘。

这种结构常被称为：

- GLU
- GEGLU
- SwiGLU

区别主要在 gate 分支激活函数不同：

- GLU：sigmoid
- GEGLU：GELU
- SwiGLU：SiLU/Swish

为什么它有效？

- 它相当于给隐藏特征加了更细粒度的门控；
- 不同维度的激活强度能被动态调节；
- 在很多 LLM 中比普通 FFN 更强。

## 8. 参数量怎么估算

假设标准 FFN：

- 输入维度 `C`
- 隐藏维度 `d_ff = 4C`

那么两层线性层的参数量大约是：

$$
C \cdot 4C + 4C \cdot C = 8C^2
$$

如果算上 bias，还要再加：

$$
4C + C
$$

这意味着 FFN 的参数量非常可观。

例如当 `C = 768` 时：

$$
8 \cdot 768^2 = 4{,}718{,}592
$$

仅一个 FFN 子模块就已经有数百万参数。

所以在很多 Transformer 中：

- Attention 很重要；
- FFN 同样是主要参数和计算来源。

## 9. 代码实现对应关系

`../mlp.py` 里的 `PositionWiseFFN` 对应的是：

```python
hidden = activation(up_proj(x))
out = down_proj(hidden)
out = dropout(out)
```

如果启用 `gated=True`，就会变成：

```python
hidden = activation(gate_proj(x)) * up_proj(x)
out = down_proj(hidden)
```

这对应现代 gated FFN 的基本结构。

## 10. 最小 PyTorch 版 FFN

```python
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

这和 `basic/model.py` 里的写法本质一致。

## 11. 学习 FFN 时最值得抓住的几点

如果你只记住最关键的内容，建议抓住下面五点：

1. FFN 作用在每个 token 上，是 `position-wise` 的。
2. FFN 不负责 token 间交互，这件事由 Attention 完成。
3. 标准结构是 `C -> 4C -> C`。
4. 升维 + 激活 + 降维，是 FFN 表达能力的来源。
5. 现代 LLM 常把普通 FFN 换成 gated 版本，例如 `SwiGLU`。

## 12. 一句话总结

FFN 是 Transformer 中“对单个 token 表示做非线性特征加工”的模块，经典形式是两层线性层配合激活函数，现代模型常进一步使用 gated 结构提升表达能力。
