# 05 FFN & Activate

本目录聚焦 Transformer 中的 `FFN (Feed-Forward Network)` 以及 FFN 里常见的激活函数。

这一部分的目标是把两件事连起来理解：

- FFN 在 Transformer Block 里到底做什么；
- ReLU、GELU、SiLU 这类激活函数为什么会直接影响 FFN 的表达能力。

## 目录结构

- `relu.py`
  教学版 `ReLU`，只保留最核心的前向逻辑，接口风格接近 PyTorch。
- `gelu.py`
  教学版 `GELU`，使用 Transformer 中常见的 `tanh` 近似。
- `silu.py`
  教学版 `SiLU`，展示自门控形式 `x * sigmoid(x)`。
- `activations.py`
  激活函数统一入口，负责像小型工厂一样构造不同激活模块。
- `mlp.py`
  FFN 的教学实现，包含：
  - 经典 position-wise FFN；
  - 可选 gated FFN；
  - 最小对照示例，比较教学版激活与 PyTorch 内置实现。
- `docs/FFN.md`
  系统讲解 FFN 的作用、数学公式、参数规模、常见激活函数和主流变体。
- `docs/Activation.md`
  单独讲解激活函数的数学形式、直觉、优缺点，以及为什么现代 LLM 更常用 `GELU / SiLU`。

## 这一部分主要讲什么

在 Transformer Block 中，Attention 负责：

- 让 token 与其他 token 交互；
- 聚合上下文信息。

而 FFN 负责：

- 对每个 token 的特征做非线性变换；
- 把表示从 `C` 扩展到更高维的隐藏空间；
- 再压回 `C`，提升表达能力。

这里“非线性变换”是否有效，关键就在激活函数。

如果没有激活函数，那么：

- 多层线性层仍然可以合并成一层线性层；
- FFN 的表达能力会明显受限；
- 也就无法真正发挥 `C -> 4C -> C` 的作用。

可以把它理解成：

- Attention 更像“信息路由”；
- FFN 更像“特征加工”。

两者组合后，一个 Block 才同时具备：

- 跨 token 建模能力；
- 单 token 表征变换能力。

## 核心结论

- Transformer 里的 FFN 通常是 `position-wise` 的。
  也就是说，同一个前馈网络会独立作用到每个位置的 token 上，不会在 FFN 里混合序列维度。
- 最经典的结构是：
  `Linear(C, 4C) -> Activation -> Linear(4C, C) -> Dropout`
- `4C` 不是数学定理，而是经验上非常常见的 hidden size 扩展比例。
- 早期模型常用 `ReLU`，现代 LLM 更常用 `GELU` 或 gated 变体，例如 `SwiGLU`。
- FFN 的参数量通常不小，在很多 Transformer 中，它和注意力层一起构成主要计算开销。
- 激活函数不是附属细节，而是 FFN 是否具备强表达能力的核心组成部分。

## 与当前仓库 basic/model.py 的关系

`nanoGPT_component/basic/model.py` 中的 `FeedForward` 已经是标准 Transformer FFN：

- 先升维到 `4 * n_embd`
- 使用 `GELU`
- 再投影回 `n_embd`

本目录则把这个模块单独拆出来，便于只研究 FFN 本身。

同时，激活函数也被拆成独立文件，便于你像阅读 PyTorch 源码一样，分别查看每种激活的最小实现。

## 一句话总结

FFN 是 Transformer Block 里对每个 token 做非线性特征变换的模块，而激活函数决定了这种非线性如何产生；标准 FFN 常配合 `GELU`，更现代的结构常配合 `SiLU/SwiGLU`。
