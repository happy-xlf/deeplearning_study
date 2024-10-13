#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   Decoder_Block.py
@Time    :   2024/10/13 19:31:33
@Author  :   Lifeng Xu 
@desc :   
'''
import torch
import torch.nn as nn
import math

class SimpleDecoder(nn.Module):
    def __init__(self, hidden_dim, nums_head, dropout=0.1) -> None:
        super().__init__()
        self.nums_head = nums_head
        self.head_dim = hidden_dim // nums_head
        self.dropout = dropout

        self.layernorm_att = nn.LayerNorm(hidden_dim, eps=0.00001)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop_att = nn.Dropout(0.1)

        self.layernorm_ffn = nn.LayerNorm(hidden_dim, eps=0.00001)
        
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.down_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.act_fn = nn.ReLU()
        self.drop_fnn = nn.Dropout(0.1)

    def attn_output(self, query, key, value, attention_mask=None):
        att_weight = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if attention_mask is None:
            # 下三角矩阵
            attention_mask = torch.ones_like(att_weight).tril()
            att_weight = att_weight.masked_fill(attention_mask == 0, float("-1e20"))

        att_weight = torch.softmax(att_weight, dim=-1)
        
        att_weight = self.drop_att(att_weight)
        output = torch.matmul(att_weight, value)

        # contiguous函数作用：将tensor的内存调整为连续的，深拷贝一份
        output = output.transpose(1,2).contiguous()
        batch,seq,_,_ = output.size()
        output = output.view(batch, seq, -1)
        output = self.o_proj(output)

        return output

    def attn_block(self, X, attention_mask=None):
        batch, seq, _ = X.size()
        query = self.q_proj(X).view(batch, seq, -1, self.head_dim).transpose(1, 2)
        key = self.k_proj(X).view(batch, seq, -1, self.head_dim).transpose(1, 2)
        value = self.v_proj(X).view(batch, seq, -1, self.head_dim).transpose(1, 2)

        output = self.attn_output(query, key, value, attention_mask)
        
        return self.layernorm_att(X + output)
    
    def ffn_block(self, X):
        up_out = self.drop_fnn(self.act_fn(self.up_proj(X)))
        down_out = self.down_proj(up_out)
        
        return self.layernorm_ffn(X + down_out)
    
    def forward(self, X, attention_mask=None):
        attn_out = self.attn_block(X, attention_mask=attention_mask)
        ffn_out = self.ffn_block(attn_out)

        return ffn_out

x = torch.rand(10,20,64)
net = SimpleDecoder(64, 8)
out = net(x)
print(out.size())

