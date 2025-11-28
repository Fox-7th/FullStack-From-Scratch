
# 文本 -> token embedding -> q k v -> attention 矩阵（整个） -> 分头(+位置矩阵) -> attention 计算 -> 映射回维度hiddel_dim
# forward 扩大，缩小，残差，norm
# 中间都有drop别忘了

from turtle import pos
from cycler import V
from numpy import mask_indices
import test
from model.LMConfig import LMConfig   
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from flash_attn.flash_attn_interface import flash_attn_func


config = LMConfig()


class RMSNorm(nn.Module):
    def __init__(self, config: LMConfig = config):
        super().__init__()
        self.epsilon = config.norm_eps
        self.scale = nn.Parameter(torch.ones(config.head_dim))

    # [B, T, model_dim]
    def forward(self, x):
        x_float = x.float()
        variance = x_float.pow(2).mean(dim = -1, keepdim = True)
        x_norms = x_float * torch.rsqrt(variance + self.epsilon)
        x_norms = x_norms.type_as(x)
        return x_norms * self.scale

# test_input = torch.randn(2, 4, 512)
# rmsnorm = RMSNorm()
# output = rmsnorm(test_input)
# print(output.shape)  # Expected output shape: (2, 4, 512)
# print(output)

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

        dim = config.head_dim # attention 内部维度
        gqa = config.gqa  # 是否kv共享头
        
        self.n_heads = config.n_heads # q 头数量
        self.head_dim = dim // self.n_heads
        self.n_kv_heads = config.n_kv_heads if gqa else self.n_heads 

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(config.dropout)
        self.flash_attn = config.flash_attn
        self.max_seq_len = config.max_seq_len
        self.flash = hasattr(torch.nn.functional,  'scaled_dot_product_attention') and config.flash_attn

        self.q_mat = nn.Linear(dim, config.n_heads * self.head_dim, bias = False)
        self.k_mat = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias = False)
        self.v_mat = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias = False)
        self.o_mat = nn.Linear(config.n_heads * self.head_dim, dim, bias = False)

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

        if not self.flash:
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
        
        # flash attention
        # TODO: Implement flash attention here
        else:
            B, T, _, _ = q.shape
            # [B, n_heads, T, head_dim]
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                # attn_mask = self.mask[:, :, :T, :T], 
                dropout_p = config.dropout, 
                is_causal = True
            )  # [B, n_heads, T, head_dim]
            attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T, -1)
            attn_output = self.o_mat(attn_output)
            return attn_output
        
        # flash attention
        # else:
        #     # 此时 q, k, v = [B, T, n_heads, head_dim]
        #     B, T, _, _ = q.shape

        #     # FlashAttention 不需要 permute！
        #     attn_output = flash_attn_func(
        #         q, k, v,
        #         causal=True,
        #         dropout_p=config.dropout
        #     )  # -> [B, T, n_heads, head_dim]

        #     attn_output = attn_output.reshape(B, T, -1)
        #     attn_output = self.o_mat(attn_output)
        #     return attn_output

# [B, T, dim]
# test_input = torch.randn(2, 17, config.dim)
# attn = Attention()
# output = attn(test_input)
# print(output.shape)  # Expected output shape: (2, 17, config.dim)


class Forward(nn.Module):
    def __init__(self, config: LMConfig = config):
        super().__init__()

        if config.hidden_dim is None:
            hidden_dim = int(config.head_dim * 4 / 3)
            hidden_dim = (hidden_dim + config.multiple_of - 1) // config.multiple_of * config.multiple_of  # 向上取整为multiple_of的倍数
        else:
            hidden_dim = config.hidden_dim
        self.dropout = nn.Dropout(config.dropout)
        self.w1 = nn.Linear(config.head_dim, hidden_dim, bias = False)
        self.w2 = nn.Linear(hidden_dim, config.head_dim, bias = False)
        self.w3 = nn.Linear(config.head_dim, hidden_dim, bias = False)
    
    # x: [B, T, dim]
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# test_input = torch.randn(2, 4, config.dim)
# forward = Forward()
# output = forward(test_input)
# print(output.shape)  # Expected output shape: (2, 4, config.dim)
 

class ModelBlock(nn.Module):
    def __init__(self, config: LMConfig = config):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.attn = Attention()
        self.ffn_norm = RMSNorm()
        self.ffn = Forward()
    
    # x: [B, T, dim]
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

# test_input = torch.randn(2, 4, config.dim)
# model_block = ModelBlock()
# output = model_block(test_input)
# print(output.shape)  # Expected output shape: (2, 4, config.dim)



# RoPE
# def pos_cis(config: LMConfig = config):
#     freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.dim, 2).float() / config.dim))
#     t = torch.arange(config.max_seq_len, dtype = freq.dtype)
#     freqs = torch.outer(t, freq)  # [max_seq_len, dim/2]
#     emb = torch.polar(torch.ones_like(freqs), freqs)  # [max_seq_len, dim/2]
#     return emb  # cis


# input shape 
# hidden_dim, max_seq_len, device
def pos_emb(config: LMConfig = config):
    # shape: [dim/2]
    head_dim = config.dim // config.n_heads # 256//8 = 64
    freqs = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    position = torch.arange(config.max_seq_len, device = freqs.device, dtype = freqs.dtype)
    pos_mat = torch.outer(position, freqs)  # [max_seq_len, head_dim/2]
    pos_mat = torch.polar(torch.ones_like(pos_mat), pos_mat)  # [max_seq_len, head_dim/2]
    return pos_mat  # shape: [max_seq_len, head_dim/2] = [8192, 32]

# x: [B, T, n_heads, head_dim]
# 这里的pos_emb 的维度应该是 [T, head_dim/2]
def apply_rope(q, k, pos_emb):
    assert q.dim() == 4 and k.dim() == 4
    assert pos_emb.shape == (q.size(1), q.size(3) // 2)

    # x: [B, T, n_heads, head_dim]
    B, T, n_heads, head_dim = q.shape
    q = q.view(B, T, n_heads, head_dim // 2, 2)
    k = k.view(B, T, n_heads, head_dim // 2, 2)
    q_complex = torch.view_as_complex(q)  # [B, T, n_heads, head_dim/2]
    k_complex = torch.view_as_complex(k)  # [B, T, n_heads, head_dim/2]

    # pos_emb: [T, head_dim/2]
    pos_emb = pos_emb[None, :, None, :]  # [1, T, 1, head_dim/2]
    q_rot = q_complex * pos_emb  # [B, T, n_heads, head_dim/2]
    k_rot = k_complex * pos_emb  # [B, T, n_heads, head_dim/2]
    q_rel = torch.view_as_real(q_rot).view(B, T, n_heads, head_dim) # [B, T, n_heads, head_dim]
    k_rel = torch.view_as_real(k_rot).view(B, T, n_heads, head_dim) # [B, T, n_heads, head_dim]
    return q_rel, k_rel
    
# test_input_q = torch.randn(2, config.max_seq_len, config.n_heads, config.dim//config.n_heads)
# test_input_k = torch.randn(2, config.max_seq_len, config.n_heads, config.dim//config.n_heads)
# pos_embedding = pos_emb()
# print(pos_embedding.shape)  # Expected output shape: (8192, 256)
# q_out, k_out = apply_rope(test_input_q, test_input_k, pos_embedding)
# print(q_out.shape)  # Expected output shape: (2, 8192, 8, 64)
# print(k_out.shape)  # Expected output shape: (2, 8192, 8, 64)



class Model(nn.Module):
    def __init__(self, config: LMConfig = config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.head_dim)
        # position embedding 可选

        #
        self.pos_emb = nn.Embedding(config.block_size, config.head_dim)


        self.blocks = nn.ModuleList([ModelBlock() for _ in range(config.n_layers)])
        self.norm = RMSNorm()
        self.output_head = nn.Linear(config.head_dim, config.vocab_size, bias = False)
        self.output_head.weight = self.token_emb.weight  # weight tying
        # PyTorch中共享权重时只需要“参数对象相同”，不需要显式转置，因为在计算时维度会自动以不同方式被使用。
        # nn.Linear(in, out) 的 weight 形状是 [out, in]，而不是 [in, out]
        # nn.Linear(in, out) 的计算公式：y = x @ W.T + b
 
    # x: [B, T]
    def forward(self, x):
        x = self.token_emb(x)  # [B, T, dim]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.output_head(x)  # [B, T, vocab_size]
        return logits

# test_input = torch.randint(0, config.vocab_size, (2, 17))
# model = Model()
# output = model(test_input)
# print(output.shape)  # Expected output shape: (2, 17, config.vocab_size)

















