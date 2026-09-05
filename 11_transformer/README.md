# 从零开始实现 Transformer 架构

本项目旨在用仅用 `torch` 包中的基础组件从零开始实现一个 Transformer 架构。
包含了多头自注意力、掩码多头自注意力机制、 Encoder-Decoder 架构的实现等。
使用的 Python 版本 3.10

测试数据采用 `Multi30k`:

```python3
from datasets import load_dataset
# 英语 -> 德语 翻译任务
dataset = load_dataset("bentrevett/multi30k")

```



- `attention.py` 中 实现了缩放点积注意力机制`:label(DotProductAttention)` 与多头注意力机制 `:label(MultiHeadAttention)`

## Some Tricks




## 命名规范
在 `attention` 实现中，`num_queries` 代表的是查询的总次数，`query_size` 代表的每次查询的维度。
`values` 与 `keys` 同理





## 数据预处理

`processing.py` 的 `load_multi30k()` 返回 `(loaders, src_vocab, tgt_vocab)`。
`loaders` 包含 `train`、`validation` 和 `test`。可以传入已经读取的
`dataset=dataset`；词表仅从训练集建立，也可以同时传入现有两个词表。

```python
from processing import load_multi30k

loaders, src_vocab, tgt_vocab = load_multi30k(dataset=dataset)
src, src_valid_len, tgt, tgt_valid_len = next(iter(loaders['train']))
```

所有 batch 张量为 `torch.int64`。源、目标张量为 `(B, num_steps)`，
长度张量为 `(B,)`。长度包含 BOS/EOS、不包含右侧 padding；截断保留 EOS。
英语转小写，德语保留大小写。源长度传给编码器与交叉注意力。

回归检查：`python -m pytest -q test_processing.py`（需要 pytest）。
