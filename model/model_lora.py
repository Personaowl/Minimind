import torch
from torch import nn


# 定义Lora网络结构，用于参数高效微调 (Parameter-Efficient Fine-Tuning)
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小，秩越小参数越少
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A (降维)
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B (升维)
        # 矩阵A高斯初始化，打乱初始状态
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化，确保训练开始时LoRA模块的输出为0，不影响原模型
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    """
    遍历模型并将LoRA模块注入到符合条件的线性层中
    :param model: 需要注入LoRA的原模型
    :param rank: LoRA的秩大小
    """
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        # 判断条件：目前只对输入输出维度一致的平方Linear层(通常是Attention中的Wq, Wk, Wv等)做LoRA注入
        if (
            isinstance(module, nn.Linear)
            and module.weight.shape[0] == module.weight.shape[1]
        ):
            # 初始化LoRA层并放到和原模型同设备
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(
                device
            )
            # 将LoRA模块作为子模块挂载到当前层上
            setattr(module, "lora", lora)
            
            # 保存原有的前向传播函数
            original_forward = module.forward

            # 显式绑定：定义新的前向传播，结果等于 "原层输出 + LoRA输出"
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            # 替换原有的前向传播
            module.forward = forward_with_lora


def load_lora(model, path):
    """
    从磁盘加载已训练的LoRA权重到模型中
    :param model: 已经调用过 apply_lora 的模型
    :param path: LoRA权重的pth文件路径
    """
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)
    
    # 去除DDP分布式训练时可能引入的 "module." 前缀
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }

    # 遍历模型，从总体 state_dict 中提取属于特定 lora 模块的权重并加载
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            # 筛选出当前层对应的 lora 权重字典
            lora_state = {
                k.replace(f"{name}.lora.", ""): v
                for k, v in state_dict.items()
                if f"{name}.lora." in k
            }
            # 调用 load_state_dict 将权重应用到该层对应的 lora 上
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """
    仅抽取和保存模型中的LoRA层权重
    :param model: 训练后带LoRA模块的模型
    :param path: 保存路径
    """
    # 如果使用了 torch.compile，获取原始模型
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    
    # 遍历原始模型的所有模块
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            # 去除DDP前缀
            clean_name = name[7:] if name.startswith("module.") else name
            
            # 构造带完整命名空间的权重字典
            lora_state = {
                f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
            
    # 将字典包含的所有LoRA层权重保存到磁盘
    torch.save(state_dict, path)