import math
import struct
import inspect
import time

from model.LMConfig import LMConfig
from typing import Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


# input  [B, T, embed_dim]
# output [B, T, embed_dim]
class RMSNorm(nn.Module):
    def __init__(self, dim, epsilon: float = 1e-5): # dim是embed_dim
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True) #[B, T, 1]
        # 用rsqrt，rsqrt(x) = 1 / sqrt(x)；但是是底层算子，更快，稳定，防止溢出
        rms = torch.rsqrt(rms + self.epsilon)
        # float 增加精度
        out = x.float() * rms * self.gamma # 广播，[B, T, embed_dim]
        return out.type_as(x) # 显存节约

# produce the RoPE matrix in complex-number form
# length(T): 预先计算的32 * 1024个  位置的 RoPE 旋转参数；遇到超出范围的，再计算
# [T, head_dim]
# RoPE 是作用于，每个head的；
# 因为整个Q K，实际包含 所有头的 Q 和 K, 每个头的Q中都对应token 1 2 3 4 5的q，所以都对应 位置变量pos 1 2 3 4 5
def RoPE_compl_matrix(head_dim, token_num: int = 32 * 1024, base: float = 1e5):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim)) # [head_dim/2]
    token_ids = torch.arange(token_num, device = theta.device)
    emb_length_matrix = torch.outer(token_ids, theta).float() # [T, head_dim/2]
    compl_matrix = torch.polar(torch.ones_like(emb_length_matrix), emb_length_matrix)
    return compl_matrix # [T, head_dim/2]


# add position embedding into Q K matrix
# input the entire matirx Q and K:  both [B, T, H, head_dim], H*head_dim = d_model(既是输入token的长度，也是每一层的隐藏维度)
# output the entire matrix Q and K: both [B, T, H, head_dim].
def RoPE_q_k_combine(RoPE_compl_mat, q_mat, k_mat):
    # change RoPE matrix shape, for broading
    def RoPE_shape_align(RoPE_compl_matrix, q_k_matrix):
        q_k_mat_dim = q_k_matrix.ndim
        assert RoPE_compl_matrix.shape == (q_k_matrix.shape[1], q_k_matrix.shape[-1])

        # [T, head_dim/2] -> [1, T, 1, head_dim/2]
        shape = [value if idx == 1 or idx == q_k_mat_dim - 1 else 1 for idx,  value in enumerate(q_k_matrix.shape)]
        RoPE_compl_matrix = RoPE_compl_matrix.reshape(*shape)
        return RoPE_compl_matrix

    q_mat = q_mat.reshape(*q_mat.shape[:-1], -1, 2)
    k_mat = k_mat.reshape(*k_mat.shape[:-1], -1, 2) # [B, T, H, head_dim/2, 2]

    q_mat = torch.view_as_complex(q_mat) # [B, T, H, head_dim/2]
    k_mat = torch.view_as_complex(k_mat)

    RoPE_compl_mat = RoPE_shape_align(RoPE_compl_mat, q_mat) # [T, head_dim/2] -> [1, T, 1, head_dim/2]

    q_mat = q_mat * RoPE_compl_mat # [B, T, H, head_dim/2]
    k_mat = k_mat * RoPE_compl_mat

    q_mat = torch.view_as_real(q_mat) # [B, T, H, head_dim/2, 2]
    k_mat = torch.view_as_real(k_mat)

    q_out = q_mat.flatten(3) # [B, T, H, head_dim]
    k_out = k_mat.flatten(3)

    return q_out, k_out


# B, T, H, head_dim = q_k_matrix.shape
q_mat, k_mat = torch.randn((2, 16, 8, 24)), torch.randn((2, 16, 8, 24 ))
RoPE_compl_mat = RoPE_compl_matrix(24, 16)
print(f"Right rope matrix shape: [16, 12]")
print(f"Actual rope matrix shape: {RoPE_compl_mat.shape}")

q_out, k_out = RoPE_q_k_combine(RoPE_compl_mat, q_mat, k_mat) 
print(f"Shape of q: {q_out.shape}")
print(f"Shape of k: {k_out.shape}")
print(f"Expected shape: [2, 16, 8, 24]")














# Group Querey Attention
def copy_kv(k, qk_ratio):
    B, T, H, head_dim = k.shape


    return (
        k[:, :, :, None, :]
        .expand(B, T, H, qk_ratio, head_dim) # logically repeated
        .reshape(B, T, H * qk_ratio, head_dim)
    )




class Attention(nn.Module):
    def __init__(self,  args: LMConfig):
        super().__init__()
        self.n_local_kv_heads = args.n_kv_heads if args.n_kv_heads is None else args.n_heads
        self.n_local_heads = args.n_heads
        assert self.n_local_heads % self.n_local_kv_heads == 0

        # ration of q heads ：k/v heads
        self.qk_ratio = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // self.n_local_heads

        self.wq = nn.Linear(args.dim, self.n_local_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_local_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_local_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_local_heads * self.head_dim, args.dim, bias=False)
        

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal = 1) # Triangle Upper

        self.register_buffer("mask", mask, persistent = False)

    # x: [B, T, model_dim]; RoPE_compl_mat: [T, head_dim/2]
    # 只有刚开始prompt_question 的时候T是>1，在reasoning过程中T一直等于1
    # 第一步，question 整个有很多token，之后reason过程中，每一次forward收到的x都只有1个 token
    # batch 可以>1,多个问题一起推理，不过每次还是每个问题，推理一个token
    def forward(self, x, RoPE_compl_mat, k_v_cache = None, use_cache = False):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # split head; [B, T, H, head_dim]
        q = q.view(*x.shape[:-1], self.n_local_heads, -1)
        k = k.view(*x.shape[:-1], self.n_local_kv_heads, -1)
        v = v.view(*x.shape[:-1], self.n_local_kv_heads, -1)
        
        # add pos_embed
        q, k = RoPE_q_k_combine(RoPE_compl_mat, q, k)

        # 问题在于 q_k_cache如何初始化，初始化成什么样子
        # q_k_cache [0]是k，v是1
        if k_v_cache is not None:
            k = torch.cat([k_v_cache[0], k], dim = 1)
            v = torch.cat([k_v_cache[1], v], dim = 1)
        past_kv = (x, v) if use_cache else None
        
        # exchange dims -> [B, H, T, head_dim]
        q = q.transpose(1, 2)
        # k v logically added heads
        k = copy_kv(k, self.qk_ratio).transpose(1, 2)
        v = copy_kv(v, self.qk_ratio).transpose(1, 2)

        # flash or normal attention computation
        if self.flash and x.shape[1] != 1:
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p = dropout_p, # 
                is_causal = True # 自动做casual mask
            )

        else: # [B, H, T, T]
            scores = (q @ k.transpose(-2,  -1)) / math.sqrt(self.head_dim)
            scores += self.mask[:,  :,  :x[1],  :x[1]] # mask matrix clip  
            scores = F.softmax(scores.float(),  dim=-1).type_as(q)
            scores = self.attn_dropout(scores)
            output = scores @ v

        output = output.transpose(1,  2).reshape(*x.shape[:-1],  -1)
        output = self.resid_dropout(self.wo(output))
        return output,  past_kv

# LMConfig_Dense = LMConfig(n_layers=2)
# attn = Attention(LMConfig_Dense)
# x = torch.randn((4,  16,  512)) # (batch size, seq len, embed dim)
# RoPE_compl_mat = RoPE_compl_matrix(64,  16) # (head dim, batch size) 其中 head dim = embed dim / num heads
# output,  past_kv = attn(x,  RoPE_compl_mat=RoPE_compl_mat,  use_cache=True)
# print(f'输入张量 x ：size = {x.shape}，RoPE 旋转角： size = {RoPE_compl_mat.shape}')
# print(f'输出 output: size = {output.shape},  kv_cache 基本信息：size_key = {past_kv[0].shape}, size_value = {past_kv[1].shape}')

class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim  = int((4 * config.dim) / 3 * 2)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
    


