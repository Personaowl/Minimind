from transformers import PretrainedConfig
from typing import Optional, Tuple, List, Union
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class MtyMindConfig(PretrainedConfig):
    model_type = "mtymind"

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


def repeat_kv(x: torch.Tensor, n_rep: int):
    """
    对KV进行复制拓展
    :param bs: (Batch Size): 批次大小
    :param slen: (Sequence Length): 序列长度
    :param num_key_value_heads: KV 头的数量
    :param head_dim: 每个头处理的特征维度
    """

    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return(
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args:MtyMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        # 断言保护，确保总的注意力头数量能够被 KV 头数量整除
        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads # 这就是Q头
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # hidden_size:Embedding Dimension，词嵌入维度大小；
        self.head_dim = args.hidden_size // args.num_attention_heads # 每个头负责多大的维度

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        ) # 输出层

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention") 
            and args.flash_attention
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        前向传播函数，执行注意力计算。
        
        :param x: 输入张量，形状为 (Batch Size, Sequence Length, Hidden Size)
        :param position_embeddings: 预计算的 RoPE 位置编码，包含 cos 和 sin 两个张量
        :param past_key_value: 可选的缓存键值对，用于加速自回归生成
        :param use_cache: 是否返回新的键值对以供后续使用
        :param attention_mask: 可选的注意力掩码，用于屏蔽特定位置的注意力
        """
        # 投影，计算 Q、K、V
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 把输入拆分成多个头，用view
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim) 
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # q和k， 使用roPE
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos=cos[:seq_len], sin=sin[:seq_len])
        # 对于k和v， 使用repeat(注意kv cache)
        if past_key_value is not None:
            # 如果有缓存，说明是在生成阶段，每次只处理一个新位置
            xk = torch.cat([past_key_value[0], xk], dim=1) # 在序列长度维度拼接新的k
            xv = torch.cat([past_key_value[1], xv], dim=1) # 在序列长度维度拼接新的v
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            # PyTorch 的矩阵乘法 @ 只会对最后两个维度进行运算。为了让“句子里的词”互相计算分数，我们必须把“句子长度”和“头维度”放到最后面去！
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # Flash加速，调库
        if(
            self.flash
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq, 
                xk, 
                xv, 
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) # 计算注意力分数
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device), 
                diagonal=1
            )
            # 最后拼接头，输出投影，返回

            #为了凑齐一个 Batch，我们会补一些没用的 <PAD> 词。这段代码把这些废词的注意力分数也赋值-1e9 近似于负无穷大
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv
        

        # -1表示把最后两个维度合并成一个维度，变回 (Batch Size, Sequence Length, Hidden Size)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
    

class FeedForward(nn.Module):
    # 初始化, 升维， 降维， 门控， dropout， 激活函数
    def __init__(self, config: MtyMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))
     
class MtyMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MtyMindConfig):
        super().__init__()
        self.layer_id = layer_id
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = self.layer_id
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.mlp = (
            FeedForward(config) if not config.use_moe else MoEFeedForward(config)
        )
    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value
    
class MtyMindModel(nn.Module):
    def __init__(self, config: MtyMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [MtyMindBlock(i, config) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        #RoPE 预计算
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        
        batch_size, seq_length = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算start_pos：如果存在past，则start_pos为已有past序列长度
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # Embedding + dropout
        hidden_states = self.dropout(
            self.embed_tokens(input_ids)
        )  # [bsz, seq_len, hidden]

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )
        presents = []
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device)
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
            if self.config.use_moe and hasattr(layer.mlp, "aux_loss"):
                total_aux_loss += layer.mlp.aux_loss

        hidden_states = self.norm(hidden_states)

        return hidden_states, presents, total_aux_loss


class MtyMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MtyMindConfig

    def __init__(self, config: MtyMindConfig):
        self.config = config
        super().__init__(config)
        self.model = MtyMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #权重共享，输出层的权重和输入嵌入层的权重是同一个矩阵，这样可以减少模型参数量，同时也有助于模型学习更好的词向量表示。
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )

        # logits to keep是整数，那就保留最后n个位置
        # 生成的时候只需要最后的logits来预测下一个token
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(-logits_to_keep, int) 
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if aux_loss is not None:
                loss += aux_loss

        return CausalLMOutputWithPast(
            loss = loss,
            logits = logits,
            past_key_values = past_key_values,
            hidden_states = hidden_states
        )

class MoEGate(nn.Module):
    def __init__(self, config: MtyMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个 token 路由到的专家数量
        self.n_routed_experts = config.n_routed_experts  # 总的可路由专家数量

        self.scoring_func = config.scoring_func  # 评分函数（如 softmax），用于计算被选中专家的权重
        self.alpha = config.aux_loss_alpha  # 辅助损失系数，用于强制多个专家之间负载均衡，避免资源浪费
        self.seq_aux = config.seq_aux  # 是否在整个序列层面计算负载均衡损失

        self.norm_topk_prob = config.norm_topk_prob  # 是否对选出的 Top-k 专家的概率进行重新归一化
        self.gating_dim = config.hidden_size  # 门控网络的输入维度，通常等于模型的隐藏层维度
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # kaiming初始化方法，可以方便的初始化一些合适的参数，以便更好的训练网络
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # moe的时候，只看token值，不关心其位置，所以合并batch和seq_len维度
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)  # 将输入展平为 (Batch Size * Sequence Length, Hidden Size)
        logits = F.linear(hidden_states, self.weight, None)  # 计算每个 token 对每个专家的打分，得到 (Batch Size * Sequence Length, n_routed_experts)

        # 使用softmax对打分进行归一化
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )
        
        # 第一种方法, 序列级别的aux_loss
        # 第二种方法， batch级别的aux_loss

        # 从所有专家的打分(scores)中，挑出分数最高的 top_k (比如 2) 个专家。
        # topk_weight: 被选中专家的具体分数 (例如: [0.6, 0.3])
        # topk_idx: 被选中专家的编号索引 (例如: [1, 3] 代表选了 1号和3号专家)
        # dim=-1 表示在最后一个维度(专家维度)上进行挑选，sorted=False 表示挑出来的结果不需要按大小排好序(为了省算力)。
        topk_weight, topk_idx = torch.topk(
            scores, self.top_k, dim=-1, sorted=False
        )

        # 第一步：权重归一化
        # 当选择多个专家时，需要对选中专家的权重进行标准化
        # 目的：确保每个token对多个专家的权重和为1，避免权重积累过大
        if self.top_k > 1 and self.norm_topk_prob:
            # keepdim=True 是为了保持矩阵维度不变，方便后续做除法。
            # + 1e-20 (极其微小的数) 是为了防止分数加起来刚好等于 0，导致后面除以 0 报错。
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20

            # 用每个专家的原始分数除以总分。
            # 这样新的权重加起来就绝对等于 1.0 (100%) 了。
            topk_weight = topk_weight / denominator


        # 第二步: 计算辅助损失 (Auxiliary Loss) 仅在训练时
        # 目的：防止“胜者为王”现象，确保所有专家都能得到充分训练（负载均衡）
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k

            # topk_idx 原本的形状是 [总词数(bsz * seq_len), 2]。
            # 现在把它强行改变形状为 [批次大小(bsz), 剩下的维度自动算(-1)]。
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            if self.seq_aux:
                # 策略 A: 序列级辅助损失 (Sequence-level Auxiliary Loss)
                # 计算每个 Batch 内每个序列的专家使用频率
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # ce 的意思是 "count of experts"，用来统计每句话里，每个专家到底被翻了多少次牌子。
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                # 统计每个专家被选中的次数

                # 1 表示沿着第 1 维度 (专家维度) 操作。
                # topk_idx_for_aux_loss (每个词选中的专家编号)。
                # 这行代码的意思是：顺着选中的专家编号，把 1 累加到空白记账本 `ce` 上。
                # 算完后，`ce` 里可能长这样：[[3, 1, 2, 0], [1, 2, 1, 2]] (代表第一句话里，0号专家被喊了 3次，3号一次没喊)。
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                )
                
                # ce 除以总的选择次数，得到每个专家的使用频率 (fi)，
                ce=ce.div_(seq_len * aux_topk / self.n_routed_experts)
                
                # 计算专家得分与使用频率的乘积，鼓励均匀分布
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                # 策略 B: 全局/Batch级辅助损失
                # 使用 One-hot 编码统计所有专家被选中的整体比例
                # 把所有被选中的专家编号展平变成一维数组，比如 [1, 3, 0, 1, 2, 1...]
                # 然后 F.one_hot 会把每个编号变成 [0,1,0,0] 这样的独立卡片。
                # mask_ce 是一个巨大的表格：行数是被选出的总人次(12)，列数是专家总数(4)。
                # 被选中的专家那一列是 1，其他列是 0。
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)  # 每个专家被选中的实际频率 (fi)
                Pi = scores_for_aux.mean(0)   # 每个专家的平均门控得分 (Pi)
                fi = ce * self.n_routed_experts
                # 损失函数 = sum(Pi * fi)，当 Pi 和 fi 均为均匀分布时该值最小
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            # 非训练模式或 alpha 为 0 时，不计算辅助损失
            # 直接新建一个数值为 0 的 1维张量作为 aux_loss 返回去，省下算力。
            aux_loss = scores.new_zeros(1).squeeze()
            
        return topk_idx, topk_weight, aux_loss


class MoEFeedForward(nn.Module):
    """
    混合专家 (MoE) 前馈神经网络层。
    通过门控机制将输入路由到多个专家网络中的一部分，以实现参数量的扩展同时保持计算量可控。
    """
    def __init__(self, config: MtyMindConfig):
        super().__init__()
        self.config = config
        # 路由专家列表：每个专家都是一个标准的前馈神经网络 (FeedForward)
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )
        # 门控网络：负责决定每个 token 应该由哪些专家处理
        self.gate = MoEGate(config)
        # 共享专家：所有 token 都会经过这些专家，用于捕获通用知识
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        """
        前向传播函数。
        
        :param x: 输入张量，形状为 (Batch Size, Sequence Length, Hidden Size)
        """
        identity = x
        orig_shape = x.shape
        bsz, seq_len, h = orig_shape

        # 1. 门控决策：计算每个 token 的专家索引、权重以及辅助损失
        topk_idx, topk_weight, aux_loss = self.gate(x)
        
        # 展平输入以方便按 token 处理
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # 训练模式：采用易于求导的方式处理
            # 对于每个 token，重复输入以匹配 top-k 专家数量，然后分别送入对应的专家处理，最后根据门控权重加权求和。
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # 预先创建一个空的输出张量，形状与输入相同，用于存储专家的输出结果
            y = torch.empty_like(x, dtype=x.dtype)
            
            # 遍历所有专家，处理被路由到该专家的 token
            for i, expert in enumerate(self.experts):
                # 1. 挑选出被路由到当前专家 i 的 token
                # 通过比较 flat_topk_idx 和当前专家编号 i，创建一个布尔掩码，标记哪些 token 被路由到该专家。
                # 例如，如果 flat_topk_idx 是 [1, 3, 0, 1, 2]，当 i=1 时，mask 就是 [True, False, False, True, False]，表示第 0 和第 3 个 token 被路由到专家 1。
                mask = (flat_topk_idx == i)
                # 2. 专家处理：将被路由到当前专家的 token 输入到该专家网络中，得到输出结果。
                if mask.any():
                    expert_out = expert(x[mask])
                    y[mask] = expert_out.to(y.dtype)
                else:
                    # 把所有参数加起来，并乘以0.0，得到一个带有计算图的标量0
                    dummy_grad = sum(p.sum() for p in expert.parameters()) * 0.0
                    y = y + dummy_grad.to(y.dtype)
            
            # 2. 加权求和：根据门控权重合并多个专家的输出
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式：使用高效的 moe_infer 方法
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )

        # 3. 结合共享专家
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        MoE 高效推理方法。
        通过对专家索引排序和分包处理，减少显存拷贝和调用的碎片化。
        """
        expert_cache = torch.zeros_like(x)
        
        # 排序：将指向同一专家的 token 聚集在一起
        # 按专家编号从小到大排序！
        # argsort 返回的是“排序后的元素在原来列表里的位置(索引)”
        # flat_expert_indices 里是: [1, 3, 0, 1, 2, 1]
        # 排序后应该是: [0, 1, 1, 1, 2, 3]
        # 它们原本的位置(idxs)是: [2, 0, 3, 5, 4, 1]
        idxs = flat_expert_indices.argsort()
        
        # 统计每个专家分配到的 token 总数（累加得到边界）
        # bincount 统计每个专家接了几个活：0号接1个，1号接3个，2号接1个，3号接1个 -> [1, 3, 1, 1]
        # cumsum 累加求和，算出在排序列表里的“切割线” -> [1, 4, 5, 6]
        tokens_per_expert = flat_expert_indices.bincount(minlength=len(self.experts)).cpu().numpy().cumsum(0)
        
        # 4. 追踪原词的主人
        # idxs 里存的是展平后的任务序号，比如 0,1 属于词0；2,3属于词1；4,5属于词2。
        # 直接整除 top_k (这里是2)，瞬间还原出这个任务到底是哪个词的！
        # 比如 idxs 里的 [2, 0, 3, 5, 4, 1] // 2 -> 变成了 [1, 0, 1, 2, 2, 0]
        token_idxs = idxs // self.config.num_experts_per_tok
        
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            
            # 批量读取并处理
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            
            # 乘上对应的门控权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            
            # 写回结果缓存
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache