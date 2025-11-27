
# 文本 -> token embedding -> q k v -> attention 矩阵（整个） -> 分头(+位置矩阵) -> attention 计算 -> 映射回维度hiddel_dim
# forward 扩大，缩小，残差，norm
# 中间都有drop别忘了

from cycler import V
from numpy import mask_indices
import test
from model.LMConfig import LMConfig   
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


config = LMConfig()





class RMSNorm(nn.Module):
    def __init__(self,
                 dim: int = config.hidden_dim,
                 epsilon: float = config.norm_eps):
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(dim))
    
    # [B, T, model_dim]
    def forward(self, x):
        x_float = x.float()
        variance = x_float.pow(2).mean(dim = -1, keepdim = True)
        x_norms = x_float * torch.rsqrt(variance + self.epsilon)
        x_norms = x_norms.type_as(x)
        return x_norms * self.scale

test_input = torch.randn(2, 4, 512)
rmsnorm = RMSNorm(dim = 512)
output = rmsnorm(test_input)
print(output.shape)  # Expected output shape: (2, 4, 512)
print(output)

# x为k v 矩阵
# [B, T, self.n_kv_heads, self.head_dim] -> # [B, T, self.n_head * q_kv_head_ratio, self.head_dim]
def kv_expand(x, q_kv_head_ratio):
    if q_kv_head_ratio == 1:
        return x
    B,  T,  n_kv_heads,  head_dim = x.shape
   
    x = x.unsqueeze(3)  # [B, T, self.n_kv_heads, 1, self.head_dim]
    x = x.expand(B, T, n_kv_heads, q_kv_head_ratio, head_dim)
    x = x.reshape(B, T, n_kv_heads * q_kv_head_ratio, head_dim)
    return x



# 还没加 cache，不过加了kv 共享
class Attention(nn.Module):
    def __init__(self, config: LMConfig = config):
        super().__init__()

        self.dim = config.dim # embedding的维度
        self.hidden_dim = config.hidden_dim # attention 内部维度
        self.n_heads = config.n_heads # q 头数量
        self.head_dim = config.hidden_dim // config.n_heads
        self.gqa = config.gqa  # 是否kv共享
        self.n_kv_heads = config.n_kv_heads if self.gqa else config.n_heads 
        # self.kv_head_dim = self.head_dim

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(config.dropout)
        self.flash_attn = config.flash_attn
        self.max_seq_len = config.max_seq_len

        self.q_mat = nn.Linear(self.dim, self.hidden_dim, bias = False)
        self.k_mat = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.v_mat = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.o_mat = nn.Linear(self.hidden_dim, self.dim, bias = False)


        mask = torch.full((1, 1, self.max_seq_len, self.max_seq_len), float("-inf"))
        # 上三角不变，下三角全 0
        mask = torch.triu(mask, diagonal = 1)
        self.register_buffer("mask", mask, persistent = False)

    # x: [B, T, dim]
    def forward(self, x):

        # [B, T, hidden_dim] or[B, T, self.n_kv_heads * self.head_dim]
        q, k, v = self.q_mat(x), self.k_mat(x), self.v_mat(x)
        # [B, T, n_heads(kv_head), head_dim] head spllit 
        q = q.view(*q.shape[:-1], self.n_heads, -1)
        k = k.view(*k.shape[:-1], self.n_kv_heads, -1)
        v = v.view(*v.shape[:-1], self.n_kv_heads, -1)
        
        q_kv_head_ratio = self.n_heads // self.n_kv_heads

        # kv -> q alignment
        # [B, T, n_kv_head, head_dim] -> [B, T, n_heads, head_dim]. 
        k = kv_expand(k, q_kv_head_ratio)
        v = kv_expand(v, q_kv_head_ratio)
        # [B, n_head, T, T]

        q = q.permute(0, 2, 1, 3)   # [B, H, T, d]
        k = k.permute(0, 2, 1, 3)   # [B, H, T, d]
        v = v.permute(0, 2, 1, 3)   # [B, H, T, d]
        score = q @ k.transpose(-2, -1) * self.scale   #  [B, H, T, T]

        # mask 广播
        T = score.size(2)
        score = score + self.mask[:, :, :T, :T]
        score_float = score.float()
        score_float = F.softmax(score_float, dim = -1) @ v  # [B, n_heads, T, head_dim]
        score = score_float.type_as(score)
        score = self.dropout(score)
        # [B, T, hidden_dim]
        B, n_heads, T, head_dim = score.shape
        score = score.permute(0, 2, 1, 3).reshape(B, T, -1) # [B, T, n_heads * head_dim]
        score = self.o_mat(score)
        return score
    
# [B, T, dim]
# test_input = torch.randn(2, 17, 100)
# attn = Attention(dim = 100, hidden_dim = 512, n_heads = 8, n_kv_heads = 2, gqa = True)
# output = attn(test_input)
# print(output.shape)  # Expected output shape: (2, 17, 100)


class Forward(nn.Module):
    def __init__(self, config: LMConfig = config):
        super().__init__()
        mid_dim = int(config.hidden_dim * 4 / 3)
        mid_dim = (mid_dim + config.multiple_of - 1) // config.multiple_of * config.multiple_of  # 向上取整为multiple_of的倍数
        self.dropout = nn.Dropout(config.dropout)

        self.w1 = nn.Linear(config.dim, mid_dim, bias = False)
        self.w2 = nn.Linear(mid_dim, config.dim, bias = False)
        self.w3 = nn.Linear(config.dim, mid_dim, bias = False)
    
    # x: [B, T, dim]
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

test_input = torch.randn(2, 4, 64)
forward = Forward(dim = 64, hidden_dim = 128)
output = forward(test_input)
print(output.shape)  # Expected output shape: (2, 4, 64)
 


class ModelBlock(nn.Module):
    def __init__(self,
                 dim: int = config.dim,
                 hidden_dim: int = config.hidden_dim,
                 n_heads: int = config.n_heads,
                 n_kv_heads: int = config.n_kv_heads,
                 dropout: float = config.dropout,
                 flash_attn: bool = config.flash_attn,
                 gqa: bool = config.gqa,
                 max_seq_len: int = config.max_seq_len
                 ):
        super().__init__()
        self.attn_norm = RMSNorm(dim = dim, epsilon = config.norm_eps)
        self.attn = Attention()
        
        self.ffn_norm = RMSNorm(dim = dim, epsilon = config.norm_eps)
        self.ffn = Forward(dim = dim,
                           hidden_dim = hidden_dim,
                           dropout = dropout,
                           multiple_of = config.multiple_of)
    
    # x: [B, T, dim]
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x























