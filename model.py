

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
def RoPE_compl_matrix(head_dim, token_length: int = 32 * 1024, base: float = 1e5):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim)) # [head_dim/2]
    token_length_list = torch.arange(token_length, device = theta.device)
    emb_length_matrix = torch.outer(token_length_list, theta).float # [T, head_dim/2]
    compl_matrix = torch.polar(torch.ones_like(emb_length_matrix), emb_length_matrix)
    return compl_matrix # [T, head_dim/2]


# add position embedding into Q K matrix
# input the entire matirx Q and K:  both [B, T, H, head_dim], H*head_dim = d_model(既是输入token的长度，也是每一层的隐藏维度)
# output the entire matrix Q and K: both [B, T, H, head_dim].
def RoPE_q_k_combine(RoPE_compl_matrix, q_matrix, k_matrix):
    # change RoPE matrix shape, for broading
    def RoPE_shape_align(RoPE_compl_matrix, q_k_matrix):
        q_k_dim = q_k_matrix.ndim
        assert q_k_dim > 1
        assert q_k_matrix.shape[1] == RoPE_compl_matrix.shape[0]
        assert q_k_matrix.shape[-1] == RoPE_compl_matrix.shape[1]
        B, T, H, head_dim = q_k_matrix.shape
        shape2be = [1, T, 1, head_dim/2]
        new_RoPE_compl_matrix = RoPE_compl_matrix.reshape(shape2be) # [T, head_dim/2] -> [1, T, 1, head_dim/2]
        return new_RoPE_compl_matrix
    
    assert q_matrix.shape == k_matrix.shape
    assert q_matrix.ndim == 4 and k_matrix.ndim == 4
    RoPE_compl_matrix = RoPE_shape_align(RoPE_compl_matrix, q_matrix) # [T, head_dim/2] -> [1, T, 1, head_dim/2]
    q_matrix = q_matrix.reshape(*q_matrix.shape[:-1], -1, 2)
    k_matrix = k_matrix.reshape(*k_matrix.shape[:-1], -1, 2) # [B, T, H, head_dim/2, 2]

    q_compl_matrix = torch.view_as_complex(q_matrix) # [B, T, H, head_dim/2]
    k_compl_matrix = torch.view_as_complex(k_matrix)

    q_matrix = q_compl_matrix * RoPE_compl_matrix # [B, T, H, head_dim/2]
    k_matrix = k_compl_matrix * RoPE_compl_matrix

    q_matrix = torch.view_as_real(q_matrix) # [B, T, H, head_dim/2, 2]
    k_matrix = torch.view_as_real(k_matrix)

    q_out = q_matrix.flatten(3) # [B, T, H, head_dim]
    k_out = k_matrix.flatten(3)

    return q_out, k_out














