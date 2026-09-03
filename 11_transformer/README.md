# 从零开始实现 Transformer 架构

本项目旨在用仅用 `torch` 包中的基础组件从零开始实现一个 Transformer 架构。
包含了多头自注意力、掩码多头自注意力机制、 Encoder-Decoder 架构的实现等。
使用的 Python 版本 3.10


- `attention.py` 中 实现了缩放点积注意力机制`:label(DotProductAttention)`