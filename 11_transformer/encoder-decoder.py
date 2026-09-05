from numpy import block
import modules as mymodules
import math
import torch
from torch import nn

class TransformerEncoderBlock(nn.Module):
    """Transformer 编码块的实现"""
    #在 Transformer 编码块中 所有的 query_size,key_size,value_size 都和 d_model 相同
    # 所以可以大幅缩减超参数个数
    def __init__(
            self,
            d_model,
            ffn_num_hiddens,
            num_heads,
            dropout,
            use_bias=False,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.attention = mymodules.MultiHeadAttention(
            d_model,d_model,d_model,d_model,num_heads,dropout,bias=use_bias
        )
        self.addNorm1 = mymodules.AddNorm(d_model,dropout)
        self.ffn = mymodules.PositionWiseFFN(
            d_model,ffn_num_hiddens,d_model,bias=use_bias
            )
        self.addNorm2 = mymodules.AddNorm(d_model,dropout)

    def forward(self,X,valid_lens=None):
        Y = self.addNorm1(X,self.attention(X,X,X,valid_lens))
        return self.addNorm2(Y,self.ffn(Y))

class TransformerEncoder(nn.Module):
    """Transformer 编码器
            
        :超参数的默认值均来源于 "Attention is All You Need" 原文
    """
    def __init__(
            self,
            vocab_size,
            num_layers=6,
            num_heads=8,
            d_model=512,
            ffn_num_hiddens=2048,
            dropout=0.1,
            use_bias=False,
            **kwargs,
            ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size,d_model)        
        self.PE = mymodules.PositionalEncoding(d_model,dropout)
        self.num_layers = num_layers
        self.blocks = nn.Sequential()

        for _ in range(num_layers):
            self.blocks.add_module(
                name=f"Blocks {_}",
                module=TransformerEncoderBlock(d_model,ffn_num_hiddens,num_heads,dropout,use_bias=use_bias)
            )

    def forward(self,X,valid_lens):
        X = self.PE(self.embedding(X) * math.sqrt(self.d_model))
        for i, blk in enumerate(self.blocks):
            X = blk(X, valid_lens)
        return X



class TransformerDecoderBlock(nn.Module):
    """Transformer 解码块的实现"""
    def __init__(
            self,
            d_model,
            ffn_num_hiddens,
            num_heads,
            dropout, 
            use_bias=False,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.attention1 = mymodules.MultiHeadAttention(d_model,d_model,d_model,d_model,num_heads,dropout,use_bias)
        self.addNorm1 = mymodules.AddNorm(d_model,dropout)
        self.attention2 = mymodules.MultiHeadAttention(d_model,d_model,d_model,d_model,num_heads,dropout,use_bias)
        self.addNorm2 = mymodules.AddNorm(d_model,dropout)
        self.ffn = mymodules.PositionWiseFFN(d_model,ffn_num_hiddens,d_model,bias=use_bias)
        self.addNorm3 = mymodules.AddNorm(d_model,dropout)

    def forward(self,X,enc_outputs,valid_lens):

        batch_size, target_len, _ = X.shape
        # 每个目标位置分别只能看到前 1、2、...、target_len 个位置
        # dec_valid_lens: (batch_size, target_len)，dtype=torch.int64
        dec_valid_lens = torch.arange(
            1, target_len + 1, device=X.device
        ).repeat(batch_size, 1)
        Y = self.addNorm1(X,self.attention1(X,X,X,dec_valid_lens))
        Z = self.addNorm2(Y,self.attention2(Y,enc_outputs,enc_outputs,valid_lens))
        return self.addNorm3(Z,self.ffn(Z))



class TransformerDecoder(nn.Module):
    """Transformer 解码器"""
    def __init__(
            self,
            vocab_size,
            num_layers=6,
            num_heads=8,
            d_model=512,
            ffn_num_hiddens=2048,
            dropout=0.1,
            use_bias=False,
            **kwargs,
            ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.PE = mymodules.PositionalEncoding(d_model,dropout)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(
                name=f'Block {i}',
                module=TransformerDecoderBlock(d_model,ffn_num_hiddens,num_heads,dropout,use_bias)
            )

        self.fn = nn.Linear(d_model,vocab_size,use_bias)

    def forward(self,X,enc_outputs,valid_lens):
        X = self.PE(self.embedding(X) * math.sqrt(self.d_model))
        for blk in self.blocks:
            X = blk(X,enc_outputs,valid_lens)

        return self.fn(X)