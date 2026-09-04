import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_softmax(X, valid_lens=None):
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)

        # valid_lens.shape -> (batch_size *  num_queries,)
        X = X.reshape(-1, shape[-1])  # (B*Q,V)
        position = torch.arange(shape[-1], device=X.device)
        # valid_lens 已经展开为 (B*Q,)
        # 变成 (B*Q, 1)，让每行的有效长度与 V 个位置编号比较
        # 运用广播机制 (1, V) < (B*Q, 1) → (B*Q, V) 的布尔掩码
        mask = position[None, :] < valid_lens[:, None]
        X = X.masked_fill(~mask, -1e6)
        return F.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """
    Attention = Softmax(QK^T/ srqt(d))V
    注意 如果两个矩阵要做内积运算 那么他们的维度一定要一样

    """

    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights: torch.Tensor | None = None

    def forward(self, queries, keys, values, valid_lens=None):
        # queries,keys,values 的形状:(batch_size,查询或者键值的个数,维度 d)
        # valid_lens:(batch_size,) 或 (batch_size,num_queries)
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)  # scores.shape:(B,Q,K)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)  # (batch_size,num_queries,d)


class MultiHeadAttention(nn.Module):
    """多头注意力机制实现"""

    def __init__(
        self, query_size, key_size, value_size, d_model, num_heads, dropout, bias=False, **kwargs
    ):
        # query_size 代表的是每个 query 的维度
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.W_q = nn.Linear(query_size, d_model, bias=bias)
        self.W_k = nn.Linear(key_size, d_model, bias=bias)
        self.W_v = nn.Linear(value_size, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.attention = DotProductAttention(dropout)  # 用一个 attention 可以实现多头的效果

    def forward(self, queries, keys, values, valid_lens=None):
        # 将Q，k，V 最后一个维度拆分成 num_heads 分别放入不同的注意力层中计算
        # 但是在工程实现中，为了保证并行计算，我们可以把 Q，K，V 的形状改为
        # (batch_size * num_heads,num_K/Q/V,d//num_heads) 放入一个 Attention 中进行计算
        Q = self._transpose_qkv(self.W_q(queries), self.num_heads)
        K = self._transpose_qkv(self.W_k(keys), self.num_heads)
        V = self._transpose_qkv(self.W_v(values), self.num_heads)

        # valid_lens 也要一起改变如果输入为 (batch_size,num_queries)
        # 需要在 dim=0 上重复 num_heads 次变为 (batch_size * num_heads,num_queries)以匹配输入
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(Q, K, V, valid_lens)
        return self.W_o(self._transpose_output(output, self.num_heads))

    def _transpose_qkv(self, X, num_heads):
        """为了多头注意力机制改变 Q、K、V 形状"""
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,d/num_heads)
        X = X.permute(0, 2, 1, 3)

        return X.reshape(
            -1, X.shape[2], X.shape[3]
        )  # X.shape = (batch_size * num_heads,查询或者“键-值“对的个数，d / num_heads)

    def _transpose_output(self, X, num_heads):
        """为了多头注意力的输出改变形状"""
        # 输入形状 (batch_size,num_heads,查询或“键-值“对的个数，d/num_heads)
        # 相当于 concat 操作了
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


class PositionWiseFFN(nn.Module):
    """前馈网络的实现 FFN"""

    def __init__(self, ffn_num_inputs, ffn_num_hiddens, ffn_num_outputs, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.W1 = nn.Linear(ffn_num_inputs, ffn_num_hiddens, bias=bias)
        self.relu = nn.ReLU()
        self.W2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs, bias=bias)

    def forward(self, X):
        return self.W2(self.relu(self.W1(X)))


class AddNorm(nn.Module):
    """Add & Layer Normalization 层的实现"""

    # 使用 dropout 作为正则化项
    def __init__(self, normalization_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(normalization_shape)

    def forward(self, X, Y):
        # Y: SubLayer(X)
        return self.norm(self.dropout(Y) + X)


class PositionalEncoding(nn.Module):
    """对嵌入层后的输入进行位置编码"""

    def __init__(self, d_model, dropout, max_len=10000, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

        self.PE = torch.zeros((1, max_len, d_model))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
        )
        self.PE[:, :, 0::2] = torch.sin(X)
        self.PE[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.PE[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)
