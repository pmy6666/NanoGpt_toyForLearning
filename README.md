# NanoGPT_toy

`nanoGPT_component` 是一个按组件拆开的 nanoGPT 学习仓库，目标不是直接复刻完整工程，而是把 GPT 训练里最关键的几个部分拆出来单独理解：

- tokenizer 是怎么工作的
- 一个最小 GPT 模型如何搭起来
- 训练脚本从基础版到优化版是怎么演进的
- 位置编码为什么需要，以及常见方案在数学上做了什么
- 还在更新中...

这个目录更适合作为学习材料、教学示例和实验场，而不是即插即用的生产训练框架。

## 目录结构

```text
nanoGPT_component/
├── 01_tokenizer/
├── 02_trainer/
├── 03_postion-encode/
├── basic/
├── input.txt
└── README.md
```

### `basic/`

一个最小可运行的 GPT 训练实现，适合先把主干流程跑通。

- `config.py`
  训练超参数配置，包含 batch size、block size、层数、头数、dropout、tokenizer 类型、输出目录等。
- `model.py`
  GPT 主体实现，包含：
  - token embedding
  - position embedding
  - self-attention
  - feed-forward
  - transformer block
  - 文本生成 `generate`
- `train.py`
  从读取语料、构造 tokenizer、切分训练/验证集，到训练、评估、保存 checkpoint 和生成样例的完整流程。
- `utils.py`
  提供字符级 tokenizer、`tiktoken` tokenizer、文本读取和 JSON 保存等辅助函数。
- `train_log/`
  一些训练日志，用来对比不同 tokenizer 或 attention 配置的效果。

## 学习重点

如果把这个目录当作学习路线，可以重点关注下面几件事：

- `basic/`
  关注一个语言模型训练最小闭环到底由哪些步骤组成。
- `01_tokenizer/`
  关注 BPE 如何从字节逐步合并成更长的子词，以及 encode/decode 为什么必须可逆。
- `02_trainer/`
  关注 AMP、梯度累积、checkpointing、SDPA 分别在训练循环的什么位置介入，以及它们在显存、速度、稳定性上的权衡。
- `03_postion-encode/`
  关注位置信息为什么不能缺失，以及 sinusoidal 和 RoPE 为什么能表达相对位移。
- ...


## 后续可扩展方向

后续还会逐步的增加各个LLM的组件学习记录
