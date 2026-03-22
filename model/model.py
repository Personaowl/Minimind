from transformers import PretrainedConfig
from typing import Optional

import math
import torch
import torch.nn as nn


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def __norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return self.weight * self.__norm(x.float()).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: Optional[dict] = None):
    """
    预计算 RoPE (旋转位置编码) 需要用到的 cos 和 sin 查表，同时包含 YaRN 长文本扩展逻辑。
    
    :param dim: 每个注意力头(Head)的特征维度大小
    :param end: 预计算的最大文本长度 (例如 32K)
    :param rope_base: 旋转公式的底数 (传统是 10000，现代大模型常设为 1e6)
    :param rope_scaling: YaRN 的配置参数字典
    """

    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0

    # YARN 的 RoPE 长文本扩展逻辑：
    if rope_scaling is not None:
        # 配置项
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16) # 要把长度扩大几倍
        beta_fast = rope_scaling.get("beta_fast", 32.0) # 高频区波长阈值 
        beta_slow = rope_scaling.get("beta_slow", 1.0)  # 低频区波长乘数
        attn_factor = rope_scaling.get("attention_factor", 1.0) # 温度缩放因子，恢复注意力锐度

        if end / orig_max > 1.0:
            # 从波长反向推出维度索引
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            
            low = max(math.floor(inv_dim(beta_fast)), 0)            # 维度序号小于 low 的是高频区
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1) # 维度序号大于 high 的是低频区
            
            # 构造Ramp: 一个在 low 和 high 之间平滑过渡的线性函数，用于混合原始频率和扩展频率
            # - 高频区 (i < low) 的 ramp 值为 0
            # - 低频区 (i > high) 的 ramp 值为 1
            # - 中频区 (low < i < high) 的 ramp 值在 0~1 之间平滑过渡
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 
                0, 1
            )
            
            # 依据频区，重新计算步长
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    
    # 矩阵外积：位置 m 乘以 基础步长 theta，算出每个位置、每个维度需要转的绝对角度 (弧度)
    freqs = torch.outer(t, freqs).float() 
    
    # 因为之前 freqs 只有一半的维度 (dim/2)，现在通过 torch.cat 复制拼接成完整的 dim
    # freqs_cos 形状: (end, dim)，存储了所有位置的所有特征维度的 cos 值
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    在模型的前向传播 (Forward) 过程中，将预计算好的 cos 和 sin 实际作用到 Q 和 K 上，完成旋转。
    
    :param q: Query 张量
    :param k: Key 张量
    :param cos: 从 precompute 函数查到的当前位置的 cos 值
    :param sin: 从 precompute 函数查到的当前位置的 sin 值
    """
    
    def rotate_half(x):
        half_idx = x.shape[-1] // 2
        return torch.cat((-x[..., half_idx:], x[..., :half_idx]), dim=-1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed