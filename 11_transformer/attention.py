import math
from turtle import forward

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

        # valid_lens.shape -> (batch_size * query_size,)
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

    def forward(self, queries, keys, values, valid_lens=None):
        # queries,keys,values 的形状:(batch_size,查询或者键值的个,维度 d)
        # valid_lens:(batch_size,) 或 (batch_size,query_size)
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)  # scores.shape:(B,Q,K)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)  # (batch_size,query_size,d)




class MultiHeadAttention(nn.Module):
    """多头注意力机制实现"""
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    def forward(self,queries,keys,values,valid_lens):
        pass


if __name__ == "__main__":
    # print('Test')
    # X = torch.zeros((2,3,4))
    # print(X)

    # valid_lens = torch.tensor([[2,3,3],[1,2,4]])
    # print(masked_softmax(X,valid_lens))

    # batch_size, query_size, key_size, value_size = 2, 2, 4, 4
    # d = 5
    # queries = torch.ones(batch_size, query_size, d)
    # keys = torch.ones(batch_size, key_size, d)
    # values = torch.ones(batch_size, value_size, d)

    # valid_lens = torch.tensor([1, 3])

    # atten = DotProductAttention(dropout=0.5)
    # atten.eval()
    # res = atten(queries, keys, values, valid_lens)
    # print(res.shape)
    # print(res)
    pass