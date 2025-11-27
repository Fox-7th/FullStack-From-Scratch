
# 文本 -> token embedding -> q k v -> attention 矩阵（整个） -> 分头(+位置矩阵) -> attention 计算 -> 映射回维度hiddel_dim
# forward 扩大，缩小，残差，norm
中间都有drop别忘了

from model.LMConfig import LMConfig   

LMConfig = LMConfig()

class RMSNorm(nn.Module):
    def __init__(self, dim: int = 512, epsilon: float = 1e6):
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(dim))
    
    # [B, T, model_dim]
    def forward(self, x ):
        











































