# 激活函数：FFN 非线性的来源

## 1. 为什么学习 FFN 时必须一起看激活函数

FFN 的经典结构是：

$$
\mathrm{Linear}(C, 4C)
\rightarrow \mathrm{Activation}
\rightarrow \mathrm{Linear}(4C, C)
$$

这里真正让 FFN 变强的，不只是“升维再降维”，而是中间那一层 `Activation`。

如果去掉激活函数，那么：

$$
\mathrm{Linear}_2(\mathrm{Linear}_1(x))
$$

仍然等价于一层线性变换。

也就是说：

- 两层线性层叠加，依然只是线性；
- 只有加入激活函数，FFN 才真正具备非线性表达能力；
- 所以激活函数不是点缀，而是 FFN 的核心组件。

## 2. 从算法角度看激活函数在做什么

设线性层输出为：

$$
h = W x + b
$$

激活函数做的事情是：

$$
\tilde{h} = \phi(h)
$$

它会对每个元素逐点变换。

这类逐元素变换有三个重要作用：

1. 引入非线性，让网络能表达更复杂的函数。
2. 改变不同区间的响应强度，例如抑制负值或平滑放大正值。
3. 影响梯度传播，从而影响训练稳定性与优化效率。

所以激活函数不仅影响“输出长什么样”，也影响“模型好不好训”。

## 3. ReLU

公式：

$$
\mathrm{ReLU}(x) = \max(0, x)
$$

最小实现思路：

- 小于 0 的值全部截断成 0；
- 大于等于 0 的值保持不变。

代码上就是：

```python
def relu(x):
    return max(0, x)
```

如果处理张量，就是逐元素执行这个规则。

### 直觉理解

ReLU 相当于一个很硬的开关：

- 负值直接关掉；
- 正值原样通过。

### 优点

- 简单直接；
- 计算开销低；
- 在很多早期深度学习模型里效果很好。

### 缺点

- 负半轴梯度为 0；
- 某些神经元可能长期输出 0；
- 对 Transformer 这类模型来说，表达往往不如 GELU 平滑。

## 4. GELU

公式：

$$
\mathrm{GELU}(x) = x \cdot \Phi(x)
$$

其中 `\Phi(x)` 是标准高斯分布的累积分布函数。

在工程里常用 tanh 近似：

$$
\mathrm{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)
$$

### 直觉理解

ReLU 是“硬阈值”；
GELU 更像“软门控”。

它不会粗暴地把所有负数都截成 0，而是：

- 对较小值给更弱的通过率；
- 对较大正值更接近线性通过；
- 整体曲线更平滑。

### 为什么 Transformer 常用 GELU

- 比 ReLU 更平滑；
- 对小幅负值不是完全截断；
- 在语言模型和 Transformer 中通常表现更稳定。

## 5. SiLU / Swish

公式：

$$
\mathrm{SiLU}(x) = x \cdot \sigma(x)
$$

其中 `\sigma(x)` 是 sigmoid。

### 直觉理解

它相当于让输入先经过一个 `sigmoid` 门控，再乘回原值。

所以：

- 当 `x` 很大时，`sigmoid(x)` 接近 1；
- 当 `x` 很小时，`sigmoid(x)` 接近 0；
- 整体表现为一种平滑的自门控。

### 为什么它重要

SiLU 本身已经是一个不错的激活函数；
更重要的是，它经常被用于 gated FFN，例如 `SwiGLU`。

## 6. 三种激活函数怎么比较

如果只抓核心差异，可以这么记：

- `ReLU`：简单，硬截断。
- `GELU`：平滑，适合 Transformer。
- `SiLU`：平滑自门控，常用于更现代的 gated 结构。

从曲线形状看：

- ReLU 在 0 点有明显折角；
- GELU 和 SiLU 更平滑；
- 平滑激活在很多大模型里更常见。

## 7. 激活函数和 FFN 的关系

标准 FFN：

$$
\mathrm{FFN}(x) = W_2 \, \phi(W_1 x + b_1) + b_2
$$

这里的 `\phi` 就是激活函数。

也就是说：

- `W_1` 先把特征投影到高维空间；
- `\phi` 对高维特征逐元素做非线性变换；
- `W_2` 再把变换后的特征压回原维度。

如果 `\phi` 换掉，FFN 的行为也会跟着变。

所以实际工程里：

- 早期模型可能用 `ReLU`；
- 经典 Transformer / BERT / GPT 系列常见 `GELU`；
- 更现代的大模型里常见 `SiLU + gated FFN`。

## 8. 最小代码实现应该怎么写

本目录里把三种激活函数拆成了独立文件，方便你按源码阅读方式逐个看：

- [`../relu.py`](/Users/pangmy/Desktop/nanoGPT/nanoGPT_component/05_FNN&Activate/relu.py:1)
- [`../gelu.py`](/Users/pangmy/Desktop/nanoGPT/nanoGPT_component/05_FNN&Activate/gelu.py:1)
- [`../silu.py`](/Users/pangmy/Desktop/nanoGPT/nanoGPT_component/05_FNN&Activate/silu.py:1)
- [`../activations.py`](/Users/pangmy/Desktop/nanoGPT/nanoGPT_component/05_FNN&Activate/activations.py:1)

其中：

- 单个文件负责单个激活函数的最小实现；
- `activations.py` 负责统一构造入口；
- [`../mlp.py`](/Users/pangmy/Desktop/nanoGPT/nanoGPT_component/05_FNN&Activate/mlp.py:1) 只负责 FFN 结构和调用逻辑。

它们都遵循同一个思路：

1. 输入一个张量；
2. 对张量逐元素应用公式；
3. 返回同形状输出。

例如 `silu.py` 中的核心逻辑就是：

```python
class SiLUActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
```

这已经足够表达 SiLU 的核心算法。

## 9. 学习时建议怎么观察

最值得观察的不是公式本身，而是“同一组输入经过不同激活后会变成什么样”。

例如输入：

$$
[-3, -1, 0, 1, 3]
$$

你会看到：

- ReLU 会把负值全变成 0；
- GELU 会保留一部分负值，但幅度更小；
- SiLU 也会保留负值，并做平滑门控。

这正是为什么它们在 FFN 中表现不同。

## 10. 一句话总结

激活函数是 FFN 中非线性的真正来源；`ReLU` 更简单，`GELU` 更适合经典 Transformer，`SiLU` 则常服务于更现代的 gated FFN 结构。
