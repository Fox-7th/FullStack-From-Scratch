
# 文本 -> token embedding -> q k v -> attention 矩阵（整个） -> 分头(+位置矩阵) -> attention 计算 -> 映射回维度hiddel_dim
# forward 扩大，缩小，残差，norm
# 中间都有drop别忘了

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










































