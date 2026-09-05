# 手撕 Transformer：从注意力计算到 Encoder–Decoder

这是我的 Transformer 手写实践项目。我把缩放点积注意力、多头拆分与合并、掩码、位置编码以及编码器和解码器逐层实现，再用 Multi30k 的**英语 → 德语翻译**任务连接数据处理与训练流程。

项目的重点是理解每一步张量运算：Q、K、V 如何参与计算，多头如何并行，解码器如何屏蔽未来信息，以及源句子如何通过交叉注意力影响目标输出。

## 手写实现了什么

模型核心由 PyTorch 张量运算和 `nn.Linear`、`nn.Embedding`、`nn.LayerNorm`、`nn.Dropout` 等基础组件组成。注意力及 Encoder–Decoder 结构自行搭建，未调用 `nn.Transformer`、`nn.MultiheadAttention` 或预训练翻译模型。梯度计算使用 PyTorch autograd，参数更新使用 Adam。

| 手写内容 | 关键实现 | 代码位置 |
|---|---|---|
| 掩码 Softmax | 将有效长度展开，利用广播屏蔽无效 key，再沿最后一维归一化 | `modules.py` · `masked_softmax` |
| 缩放点积注意力 | `QKᵀ / √d`、注意力权重、加权求和 | `modules.py` · `DotProductAttention` |
| 多头注意力 | Q/K/V 线性投影、拆头、并行计算、拼接和输出投影 | `modules.py` · `MultiHeadAttention` |
| 逐位置前馈网络 | 两层线性变换与 ReLU，对每个位置独立处理 | `modules.py` · `PositionWiseFFN` |
| 残差连接与归一化 | `LayerNorm(X + Dropout(Sublayer(X)))`，采用 Post-LN 结构 | `modules.py` · `AddNorm` |
| 正弦位置编码 | 使用不同频率的 sin/cos 构造位置向量 | `modules.py` · `PositionalEncoding` |
| 编码器 | 词嵌入缩放、位置编码、多层自注意力与前馈网络 | `EncoderDecoder.py` · `TransformerEncoder` |
| 解码器 | 因果自注意力、编码器—解码器交叉注意力、前馈网络和词表投影 | `EncoderDecoder.py` · `TransformerDecoder` |
| 整体模型 | 连接编码器和解码器，传递源序列有效长度 | `EncoderDecoder.py` · `Transformer` |

数据部分也包含自己的正则分词器、词频统计、词表、ID 转换、截断与 padding；`datasets` 负责读取数据，`DataLoader` 负责组成 batch。

## 实现中重点理解的内容

### 1. 把多个注意力头合并到 batch 维度

多头注意力在工程上通过改变张量形状实现并行计算。令 `B` 为 batch 大小、`L` 为序列长度、`H` 为头数，且 `d_head = d_model / H`：

```text
(B, L, d_model)
    ↓ reshape
(B, L, H, d_head)
    ↓ permute
(B, H, L, d_head)
    ↓ reshape
(B × H, L, d_head)
    ↓ 批量注意力计算，再合并各个头
(B, L, d_model)
```

Q 的长度可以与 K/V 不同，因此同一套实现也可以用于交叉注意力。`d_model` 必须能被 `num_heads` 整除。

### 2. 区分源 padding 掩码与目标因果掩码

| 使用位置 | 掩码依据 | 作用 |
|---|---|---|
| 编码器自注意力 | `src_valid_len`，形状 `(B,)` | 忽略源序列的 padding key |
| 解码器自注意力 | 每条序列的 `[1, 2, …, T]`，形状 `(B, T)` | 每个位置只能读取截至自身的目标前缀 |
| 解码器交叉注意力 | `src_valid_len`，形状 `(B,)` | 忽略 encoder 输出中的源 padding 位置 |

例如目标长度为 4，解码器自注意力的可见关系是：

```text
             key 位置
             0  1  2  3
query 位置 0  ✓  ×  ×  ×
           1  ✓  ✓  ×  ×
           2  ✓  ✓  ✓  ×
           3  ✓  ✓  ✓  ✓
```

交叉注意力的 Q 来自解码器，K 和 V 来自编码器，因此这里传入的是**源长度**。因果掩码在解码块内部生成。

### 3. 对齐解码器输入与预测标签

训练时使用目标句子的已知前缀，逐位置预测下一个 token：

```text
完整目标：<bos> ein Hund läuft <eos>
模型输入：<bos> ein Hund läuft
预测标签： ein Hund läuft <eos>
```

对应代码是 `tgt[:, :-1]` 和 `tgt[:, 1:]`。解码器输出未经 Softmax 的 logits，形状为 `(B, T, tgt_vocab_size)`，直接交给交叉熵计算损失。目标 padding 不参与损失，训练和验证的平均损失按有效目标 token 数统计。

## 项目结构

```text
11_transformer/
├── modules.py                 # 手写注意力、FFN、AddNorm 和位置编码
├── EncoderDecoder.py          # 编码块、解码块及完整 Transformer
├── processing.py              # 分词、词表、序列编码和 DataLoader
├── train.py                   # 英语到德语的训练与验证 demo
├── test.ipynb                 # 数据探索、分词与词表实验
├── Attention Is All You Need.pdf
└── README.md
```

建议按 `modules.py → EncoderDecoder.py → processing.py → train.py` 的顺序阅读。

## 数据预处理

使用 `datasets.load_dataset("bentrevett/multi30k")` 读取数据。英语转小写，德语保留大小写；两个语言分别建立词表，**只用训练集统计词频**，验证集和测试集复用训练词表。

特殊标记包括 `<unk>`、`<bos>`、`<eos>` 和 `<padding>`。标记 ID 通过词表查询，不假设固定编号。截断时先给 BOS/EOS 留出两个位置，再对正文截断，最后在右侧 padding。

```python
from processing import load_multi30k

loaders, src_vocab, tgt_vocab = load_multi30k(
    batch_size=32, num_steps=32, min_freq=2
)
src, src_valid_len, tgt, tgt_valid_len = next(iter(loaders["train"]))
```

如果已经通过 `datasets` 读取了数据，可以传入 `load_multi30k(dataset=dataset)`。

| 返回的 batch 张量 | 形状 | 含义 |
|---|---|---|
| `src` | `(B, num_steps)` | 英语 token ID，含起止标记和 padding |
| `src_valid_len` | `(B,)` | 源有效长度，含 BOS/EOS、不含 padding |
| `tgt` | `(B, num_steps)` | 德语 token ID，含起止标记和 padding |
| `tgt_valid_len` | `(B,)` | 目标有效长度，含 BOS/EOS、不含 padding |

以上张量均为 `torch.int64`。`loaders` 包含 `train`、`validation`、`test`，仅训练集打乱顺序。

## 运行 demo

使用 Python 3.10 或以上版本，在已激活的环境中安装依赖：

```bash
python -m pip install torch datasets numpy
```

进入本目录后运行：

```bash
python train.py
```

首次运行需要下载数据。程序按 CUDA → MPS → CPU 的顺序选择可用设备，并在每轮训练后计算验证集损失。

当前 demo 的配置直接写在 `train.py` 中：

| 参数 | 值 |
|---|---|
| 编码器 / 解码器层数 | 各 2 层 |
| `d_model` / 注意力头数 | 128 / 4 |
| FFN 隐藏维度 | 512 |
| Dropout | 0.1 |
| batch 大小 / 序列总长度 | 32 / 32，长度包含 BOS/EOS |
| 词表最低词频 | 2 |
| 优化器 / 学习率 | Adam / 0.001 |
| 训练轮数 | 10 |

## 当前进度

- 已实现 Transformer 核心结构、预处理与训练/验证 demo。
- 已用小样本验证前向、反向传播和参数更新，验证损失与整批有效 token 平均结果一致。
- 完整训练的收敛表现和翻译质量尚未验证，暂未报告 BLEU 等指标。
- 后续练习包括自回归翻译生成、模型与词表保存、测试集评估，以及 KV cache。当前解码器没有历史缓存，后续逐词生成时需要输入完整的已生成前缀。

本项目以理解和手写实现为目标；模型规模与训练配置用于实践，不代表对论文训练结果的复现。
