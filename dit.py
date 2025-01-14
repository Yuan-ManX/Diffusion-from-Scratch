import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.vision_transformer import PatchEmbed
from typing import List

from utils import (CaptionProjection, CrossAttention, Mlp, SelfAttention, T2IFinalLayer,
                   TimestepEmbedder, create_norm, get_2d_sincos_pos_embed, get_mask,
                   mask_out_token, modulate, unmask_tokens)


class AttentionBlockPromptEmbedding(nn.Module):
    """
    专门用于处理提示嵌入（prompt embeddings）的注意力模块。

    Args:
        dim (int): 输入和输出的维度大小。
        head_dim (int): 每个注意力头的维度大小。
        mlp_ratio (float): 前馈网络隐藏层维度相对于输入维度的倍数。
        multiple_of (int): 将前馈网络隐藏层维度向上取整到该值的最近倍数。
        norm_eps (float): 层归一化中的 epsilon 值。
        use_bias (bool): 是否在线性层中使用偏置项。
    """
    def __init__(
        self,
        dim: int,
        head_dim: int, 
        mlp_ratio: float,
        multiple_of: int,
        norm_eps: float,
        use_bias: bool,
    ) -> None:
        super().__init__()
        # 确保输入维度可以被头维度整除，确保每个头有整数个维度
        assert dim % head_dim == 0, 'Hidden dimension must be divisible by head dim'
        
        # 保存参数
        self.dim = dim
        # 计算注意力头的数量
        self.num_heads = dim // head_dim
        
        # 创建层归一化层
        self.norm1 = create_norm('layernorm', dim, eps=norm_eps)
        # 创建自注意力机制
        self.attn = SelfAttention(
            dim=dim,
            num_heads=self.num_heads,
            qkv_bias=use_bias,
            norm_eps=norm_eps,
        )

        # 创建第二个层归一化层
        self.norm2 = create_norm('layernorm', dim, eps=norm_eps)
        # 创建前馈神经网络
        self.mlp = FeedForward(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio), # 计算前馈网络的隐藏层维度
            multiple_of=multiple_of, # 将隐藏层维度向上取整到 multiple_of 的最近倍数
            use_bias=use_bias,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播方法。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        # 第一个残差连接块：层归一化 + 自注意力机制 + 残差连接
        x = x + self.attn(self.norm1(x))
        # 第二个残差连接块：层归一化 + 前馈神经网络 + 残差连接
        x = x + self.mlp(self.norm2(x))
        return x

    def custom_init(self, init_std: float = 0.02) -> None:
        """
        自定义初始化方法。

        Args:
            init_std (float): 初始化标准差，默认为0.02。
        """
        # 对自注意力机制进行自定义初始化
        self.attn.custom_init(init_std)
        # 对前馈神经网络进行自定义初始化
        self.mlp.custom_init(init_std)


class FeedForward(nn.Module):
    """
    前馈神经网络模块，使用 SiLU（Swish）激活函数。

    Args:
        dim (int): 输入和输出的维度大小。
        hidden_dim (int): 第一个和第二个线性层之间的隐藏层维度。
        multiple_of (int): 将隐藏层维度向上取整到该值的最近倍数。
        use_bias (bool): 是否在线性层中使用偏置项。
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        use_bias: bool,
    ):
        super().__init__()
        # 保存输入和输出的维度大小
        self.dim = dim
        # 计算隐藏层维度，并向上取整到 multiple_of 的最近倍数
        hidden_dim = int(2 * hidden_dim / 3) # 假设隐藏层维度为输入维度的 2/3
        # 向上取整
        self.hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        # 创建第一个线性层，将输入维度映射到隐藏层维度
        self.w1 = nn.Linear(dim, self.hidden_dim, bias=use_bias)
        # 创建第二个线性层，将输入维度映射到隐藏层维度
        self.w2 = nn.Linear(dim, self.hidden_dim, bias=use_bias)
        # 创建第三个线性层，将隐藏层维度映射回输入维度
        self.w3 = nn.Linear(self.hidden_dim, dim, bias=use_bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

    def custom_init(self, init_std: float) -> None:
        """
        自定义初始化方法。

        Args:
            init_std (float): 初始化标准差，用于权重初始化。
        """
        # 对第一个线性层的权重进行截断正态分布初始化，均值为0，标准差为0.02
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        # 对第二个和第三个线性层的权重进行截断正态分布初始化，均值为0，标准差为 init_std
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class FeedForwardECMoe(nn.Module):
    """
    基于专家选择（Expert-Choice）风格的混合专家（Mixture of Experts, MoE）前馈层，使用 GELU 激活函数。

    Args:
        num_experts (int): 层中专家的数量。
        expert_capacity (float): 容量因子，决定每个专家处理的 token 数量。
        dim (int): 输入和输出的维度大小。
        hidden_dim (int): 两个线性层之间的隐藏层维度。
        multiple_of (int): 将隐藏层维度向上取整到该值的最近倍数。
    """
    def __init__(
        self,
        num_experts: int,
        expert_capacity: float,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        # 保存专家数量
        self.num_experts = num_experts
        # 保存容量因子
        self.expert_capacity = expert_capacity
        # 保存输入和输出的维度大小
        self.dim = dim
        # 计算隐藏层维度，并向上取整到 multiple_of 的最近倍数
        self.hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 创建第一个线性层的权重参数，形状为 [num_experts, dim, hidden_dim]
        # 使用全1初始化，实际训练过程中会通过反向传播更新
        self.w1 = nn.Parameter(torch.ones(num_experts, dim, self.hidden_dim))
        # 创建第二个线性层的权重参数，形状为 [num_experts, hidden_dim, dim]
        # 使用全1初始化，实际训练过程中会通过反向传播更新
        self.w2 = nn.Parameter(torch.ones(num_experts, self.hidden_dim, dim))
        # 创建门控网络，将输入维度映射到专家数量，形状为 [dim, num_experts]
        # 不使用偏置项
        self.gate = nn.Linear(dim, num_experts, bias=False)
        # 创建 GELU 激活函数
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法。

        Args:
            x (torch.Tensor): 输入张量，形状为 [n, t, d]，其中 n 是批次大小，t 是序列长度，d 是维度。

        Returns:
            torch.Tensor: 输出张量，形状为 [n, t, d]。
        """
        assert len(x.shape) == 3
        # 解析输入张量的维度，n 是批次大小，t 是序列长度，d 是维度
        n, t, d = x.shape
        # 计算每个专家处理的 token 数量
        tokens_per_expert = int(self.expert_capacity * t / self.num_experts)

        # 计算门控分数，形状为 [n, t, e]，其中 e 是专家数量
        scores = self.gate(x)  # [n, t, e]
        # 对门控分数应用 softmax，得到概率分布，形状为 [n, t, e]
        probs = F.softmax(scores, dim=-1)  # [n, t, e]

        # 对概率分布进行 topk 操作，选择每个专家处理的前 k 个 token
        # permute 后形状为 [n, e, t]，然后在最后一个维度上取 topk
        # 返回值 g 是 topk 的值，形状为 [n, e, k]
        # m 是 topk 的索引，形状为 [n, e, k]
        g, m = torch.topk(probs.permute(0, 2, 1), tokens_per_expert, dim=-1)  # [n, e, k], [n, e, k]
        # 将索引 m 转换为 one-hot 编码，形状为 [n, e, k, t]
        p = F.one_hot(m, num_classes=t).float()  # [n, e, k, t]

        # 将输入张量 x 与 one-hot 编码 p 进行 einsum 操作，选择每个专家处理的 token
        # 结果 xin 的形状为 [n, e, k, d]
        xin = torch.einsum('nekt, ntd -> nekd', p, x)  # [n, e, k, d]
        # 对选择的 token 进行第一个线性变换，形状为 [n, e, k, hidden_dim]
        h = torch.einsum('nekd, edf -> nekf', xin, self.w1)  # [n, e, k, 4d]
        # 应用 GELU 激活函数
        h = self.gelu(h)
        # 对激活后的结果进行第二个线性变换，形状为 [n, e, k, d]
        h = torch.einsum('nekf, efd -> nekd', h, self.w2)  # [n, e, k, d]

        # 将门控分数 g 与变换后的结果 h 相乘，形状为 [n, e, k, d]
        out = g.unsqueeze(dim=-1) * h  # [n, e, k, d]
        # 将结果与 one-hot 编码 p 进行 einsum 操作，重塑回原始形状 [n, t, d]
        out = torch.einsum('nekt, nekd -> ntd', p, out)
        return out
    
    def custom_init(self, init_std: float):
        """
        自定义初始化方法。

        Args:
            init_std (float): 初始化标准差，用于权重初始化。
        """
        # 对门控网络的权重进行截断正态分布初始化，均值为0，标准差为0.02
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=0.02)
        # 对第一个线性层的权重进行截断正态分布初始化，均值为0，标准差为0.02
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        # 对第二个线性层的权重进行截断正态分布初始化，均值为0，标准差为 init_std
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)


class DiTBlock(nn.Module):
    """
    DiT（Diffusion Transformer）模块，由注意力机制和 MLP 块组成。
    支持在 MLP 块中选择使用密集前馈（dense feed-forward）或专家选择风格的混合专家（MoE）前馈块。

    Args:
        dim (int): 块的输入和输出维度。
        head_dim (int): 每个注意力头的维度大小。
        mlp_ratio (float): MLP 块中线性层之间隐藏层的比例。
        qkv_ratio (float): 注意力块中 qkv 层维度的比例。
        multiple_of (int): 在 MLP 块中将隐藏层维度向上取整到该值的最近倍数。
        pooled_emb_dim (int): 池化后的标题嵌入维度，用于 AdaLN 调制。
        norm_eps (float): 层归一化的 epsilon 值。
        depth_init (bool): 是否基于块索引初始化 MLP/注意力块中最后一层的权重。
        layer_id (int): 当前块在 DiT 模型中的索引。
        num_layers (int): DiT 模型中块的总数量。
        compress_xattn (bool): 是否使用 qkv_ratio 缩放交叉注意力 qkv 维度。
        use_bias (bool): 是否在线性层中使用偏置项。
        moe_block (bool): 是否在 MLP 块中使用混合专家。
        num_experts (int): 如果使用 MoE 块，专家的数量。
        expert_capacity (float): 如果使用 MoE 块，每个专家的容量因子。
    """
    def __init__(
        self,
        dim: int,
        head_dim: int,
        mlp_ratio: float,
        qkv_ratio: float,
        multiple_of: int,
        pooled_emb_dim: int,
        norm_eps: float,
        depth_init: bool,
        layer_id: int,
        num_layers: int,
        compress_xattn: bool,
        use_bias: bool,
        moe_block: bool,
        num_experts: int,
        expert_capacity: float,
    ):
        super().__init__()
        # 保存输入和输出维度
        self.dim = dim

        # 计算 qkv 隐藏层维度，如果 qkv_ratio 不为1，则根据 qkv_ratio 计算；否则，设为 dim
        qkv_hidden_dim = (
            (head_dim * 2) * ((int(dim * qkv_ratio) + head_dim * 2 - 1) // (head_dim * 2))
            if qkv_ratio != 1 else dim
        )
        # 计算 MLP 隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)

        # 创建层归一化层
        self.norm1 = create_norm('layernorm', dim, eps=norm_eps)
        # 创建自注意力机制
        self.attn = SelfAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim,  # 计算注意力头的数量
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_hidden_dim,  # qkv 隐藏层维度
        )
        # 创建交叉注意力机制
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=qkv_hidden_dim // head_dim if compress_xattn else dim // head_dim, # 计算交叉注意力头的数量
            qkv_bias=use_bias,
            norm_eps=norm_eps,
            hidden_dim=qkv_hidden_dim if compress_xattn else dim, # qkv 隐藏层维度
        )
        # 创建第二个和第三个层归一化层
        self.norm2 = create_norm('layernorm', dim, eps=norm_eps)
        self.norm3 = create_norm('layernorm', dim, eps=norm_eps)
        
        # 根据是否使用 MoE 块，选择不同的 MLP 实现
        self.mlp = (
            FeedForwardECMoe(num_experts, expert_capacity, dim, mlp_hidden_dim, multiple_of)
            if moe_block else
            FeedForward(dim, mlp_hidden_dim, multiple_of, use_bias)
        )

        # 创建 AdaLN 调制模块
        self.adaLN_modulation = nn.Sequential(
            nn.GELU(approximate="tanh"), # 使用近似 tanh 的 GELU 激活函数
            nn.Linear(pooled_emb_dim, 6 * dim, bias=True), # 线性层，将 pooled_emb_dim 映射到 6 * dim
        )
        
        # 计算权重初始化标准差
        self.weight_init_std = (
            0.02 / (2 * (layer_id + 1)) ** 0.5 if depth_init else 
            0.02 / (2 * num_layers) ** 0.5
        )
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播方法。

        Args:
            x (torch.Tensor): 输入张量。
            y (torch.Tensor): 交叉注意力机制的输入。
            c (torch.Tensor): 用于 AdaLN 调制的条件张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        # 对条件张量 c 进行 AdaLN 调制，并拆分成六个部分
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        # 应用自注意力机制并进行 AdaLN 调制
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # 应用交叉注意力机制
        x = x + self.cross_attn(self.norm2(x), y)
        # 应用 MLP 并进行 AdaLN 调制
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x

    def custom_init(self):
        """
        自定义初始化方法。
        """
        # 重置所有归一化层的参数
        for norm in (self.norm1, self.norm2, self.norm3):
            norm.reset_parameters()
        # 对自注意力机制进行自定义初始化
        self.attn.custom_init(self.weight_init_std)
        # 对交叉注意力机制进行自定义初始化
        self.cross_attn.custom_init(self.weight_init_std)
        # 对 MLP 进行自定义初始化
        self.mlp.custom_init(self.weight_init_std)
    

class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) 模型，支持基于嵌入的条件输入，用于文本生成图像任务。

    Args:
        input_size (int, 默认: 32): 输入图像的大小（假设为正方形）。
        patch_size (int, 默认: 2): 图像分块的大小，用于分块嵌入。
        in_channels (int, 默认: 4): 输入图像的通道数（默认假设为四通道的潜在空间）。
        dim (int, 默认: 1152): Transformer 主干网络的维度，即主要 Transformer 层的维度。
        depth (int, 默认: 28): Transformer 块的数量。
        head_dim (int, 默认: 64): 每个注意力头的维度大小。
        multiple_of (int, 默认: 256): 在 MLP 块中将隐藏层维度向上取整到该值的最近倍数。
        caption_channels (int, 默认: 1024): 标题嵌入的通道数。
        pos_interp_scale (float, 默认: 1.0): 位置嵌入插值缩放因子（1.0 对应 256x256，2.0 对应 512x512）。
        norm_eps (float, 默认: 1e-6): 层归一化的 epsilon 值。
        depth_init (bool, 默认: True): 是否在 DiT 块中使用基于深度的初始化。
        qkv_multipliers (List[float], 默认: [1.0]): DiT 块中 QKV 投影维度的乘数列表。
        ffn_multipliers (List[float], 默认: [4.0]): DiT 块中 FFN 隐藏层维度的乘数列表。
        use_patch_mixer (bool, 默认: True): 是否使用分块混合器层。
        patch_mixer_depth (int, 默认: 4): 分块混合器块的数量。
        patch_mixer_dim (int, 默认: 512): 分块混合器层的维度大小。
        patch_mixer_qkv_ratio (float, 默认: 1.0): 分块混合器块中 QKV 投影维度的乘数。
        patch_mixer_mlp_ratio (float, 默认: 1.0): 分块混合器块中 FFN 隐藏层维度的乘数。
        use_bias (bool, 默认: True): 是否在线性层中使用偏置项。
        num_experts (int, 默认: 8): 如果使用 MoE 块，专家的数量。
        expert_capacity (int, 默认: 1): 如果使用 MoE FFN 层，每个专家的容量因子。
        experts_every_n (int, 默认: 2): 每隔 n 个块添加一个 MoE FFN 层。
    """
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 1152,
        depth: int = 28,
        head_dim: int = 64,
        multiple_of: int = 256,
        caption_channels: int = 1024,
        pos_interp_scale: float = 1.0,
        norm_eps: float = 1e-6,
        depth_init: bool = True,
        qkv_multipliers: List[float] = [1.0],
        ffn_multipliers: List[float] = [4.0],
        use_patch_mixer: bool = True,
        patch_mixer_depth: int = 4,
        patch_mixer_dim: int = 512,
        patch_mixer_qkv_ratio: float = 1.0,
        patch_mixer_mlp_ratio: float = 1.0,
        use_bias: bool = True,
        num_experts: int = 8,
        expert_capacity: int = 1,
        experts_every_n: int = 2
    ):
        super().__init__()
        self.input_size = input_size  # 保存输入图像的大小
        self.in_channels = in_channels  # 保存输入图像的通道数
        self.out_channels = in_channels  # 保存输出图像的通道数
        self.patch_size = patch_size  # 保存图像分块的大小
        self.head_dim = head_dim  # 保存每个注意力头的维度大小
        self.pos_interp_scale = pos_interp_scale  # 保存位置嵌入插值缩放因子
        self.use_patch_mixer = use_patch_mixer  # 保存是否使用分块混合器层的标志
        
        # 定义近似 GELU 激活函数
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        # 创建分块嵌入层，将输入图像分块并嵌入到高维空间
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, dim, bias=True
        )
        # 创建时间步嵌入层，将时间步信息嵌入到高维空间
        self.t_embedder = TimestepEmbedder(dim, approx_gelu)

        # 计算分块的数量
        num_patches = self.x_embedder.num_patches
        # 计算基础尺寸（图像大小除以分块大小）
        self.base_size = input_size // self.patch_size
        # 注册位置嵌入张量，形状为 [1, num_patches, dim]
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, dim))

        # 创建标题嵌入投影层，将标题嵌入投影到 Transformer 维度
        self.y_embedder = CaptionProjection(
            in_channels=caption_channels,
            hidden_size=dim,
            act_layer=approx_gelu,
            norm_layer=create_norm('layernorm', dim, eps=norm_eps)
        )

        # 创建标题嵌入预处理注意力块，用于处理标题嵌入
        self.y_emb_preprocess = AttentionBlockPromptEmbedding(
            dim,
            head_dim,
            mlp_ratio=4.0,  # MLP 块的隐藏层比例，默认为4.0
            multiple_of=multiple_of,  # MLP 块的隐藏层维度向上取整的倍数
            norm_eps=norm_eps,  # 层归一化的 epsilon 值
            use_bias=use_bias  # 是否在线性层中使用偏置项
        )
        
        # 创建标题嵌入池化后的处理 MLP，用于进一步处理池化后的标题嵌入
        self.pooled_y_emb_process = Mlp(
            dim,
            dim,
            dim,
            approx_gelu, # 使用近似 GELU 激活函数
            norm_layer=create_norm('layernorm', dim, eps=norm_eps) # 使用层归一化
        )

        # 如果使用分块混合器层
        if self.use_patch_mixer:
            # 计算需要使用 MoE 块的块的索引
            expert_blocks_idx = [
                i for i in range(1, patch_mixer_depth) 
                if (i+1) % experts_every_n == 0
            ]
            # 生成一个布尔列表，指示每个块是否使用 MoE
            is_moe_block = [
                True if i in expert_blocks_idx else False 
                for i in range(patch_mixer_depth)
            ]
            
            # 创建分块混合器模块列表
            self.patch_mixer = nn.ModuleList([
                DiTBlock(
                    dim=patch_mixer_dim,  # 输入和输出维度
                    head_dim=head_dim,  # 每个注意力头的维度大小
                    mlp_ratio=patch_mixer_mlp_ratio,  # MLP 块的隐藏层比例
                    qkv_ratio=patch_mixer_qkv_ratio,  # QKV 投影维度的比例
                    multiple_of=multiple_of,  # MLP 块的隐藏层维度向上取整的倍数
                    pooled_emb_dim=dim,  # 池化后的标题嵌入维度
                    norm_eps=norm_eps,  # 层归一化的 epsilon 值
                    depth_init=False,  # 是否使用基于深度的初始化
                    layer_id=0,  # 当前块的索引
                    num_layers=depth,  # Transformer 块的总数量
                    compress_xattn=False,  # 是否压缩交叉注意力 qkv 维度
                    use_bias=use_bias,  # 是否在线性层中使用偏置项
                    moe_block=is_moe_block[i],  # 是否使用 MoE 块
                    num_experts=num_experts,  # 专家的数量
                    expert_capacity=expert_capacity  # 每个专家的容量因子
                ) for i in range(patch_mixer_depth)
            ])

            # Couple of projection layers
            # 如果分块混合器层的维度与 Transformer 主干网络的维度不同，则创建投影层
            if patch_mixer_dim != dim:
                self.patch_mixer_map_xin = nn.Sequential(
                    # 创建层归一化层
                    create_norm('layernorm', dim, eps=norm_eps),
                    # 创建线性层，将 dim 映射到 patch_mixer_dim
                    nn.Linear(dim, patch_mixer_dim, bias=use_bias)
                )
                self.patch_mixer_map_xout = nn.Sequential(
                    # 创建层归一化层
                    create_norm('layernorm', patch_mixer_dim, eps=norm_eps),
                    # 创建线性层，将 patch_mixer_dim 映射到 dim
                    nn.Linear(patch_mixer_dim, dim, bias=use_bias)
                )
                self.patch_mixer_map_y = nn.Sequential(
                    # 创建层归一化层
                    create_norm('layernorm', dim, eps=norm_eps),
                    # 创建线性层，将 dim 映射到 patch_mixer_dim
                    nn.Linear(dim, patch_mixer_dim, bias=use_bias)
                )
            else:
                # 如果维度相同，则使用恒等映射
                self.patch_mixer_map_xin = nn.Identity()
                self.patch_mixer_map_xout = nn.Identity()
                self.patch_mixer_map_y = nn.Identity()

        # 确保 ffn_multipliers 和 qkv_multipliers 列表长度相同
        assert len(ffn_multipliers) == len(qkv_multipliers)
        if len(ffn_multipliers) == depth: 
            # 如果列表长度与深度相同，则直接使用 qkv_multipliers 作为 qkv_ratios
            qkv_ratios = qkv_multipliers
            # 如果列表长度与深度相同，则直接使用 ffn_multipliers 作为 mlp_ratios
            mlp_ratios = ffn_multipliers
        else:
            # Distribute the multiplers across each partition
            # 如果列表长度与深度不同，则将乘数分配到每个分割的块中
            num_splits = len(ffn_multipliers)
            assert depth % num_splits == 0, 'number of blocks should be divisible by number of splits'
            # 计算每个分割的块的数量
            depth_per_split = depth // num_splits
            
            # 使用 NumPy 数组操作，将每个乘数重复 depth_per_split 次，然后展平为一维列表
            qkv_ratios = list(np.array([
                [m]*depth_per_split for m in qkv_multipliers
            ]).reshape(-1))
            
            mlp_ratios = list(np.array([
                [m]*depth_per_split for m in ffn_multipliers
            ]).reshape(-1))

        # Don't use MoE in last block
        # 确保不使用 MoE 在最后一个块中
        expert_blocks_idx = [
            i for i in range(0, depth - 1) 
            if (i+1) % experts_every_n == 0
        ]
        is_moe_block = [
            True if i in expert_blocks_idx else False 
            for i in range(depth)
        ]
        
        # 创建 Transformer 块列表
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,  # 输入和输出维度
                head_dim=head_dim,  # 每个注意力头的维度大小
                mlp_ratio=mlp_ratios[i],  # MLP 块的隐藏层比例
                qkv_ratio=qkv_ratios[i],  # QKV 投影维度的比例
                multiple_of=multiple_of,  # MLP 块的隐藏层维度向上取整的倍数
                pooled_emb_dim=dim,  # 池化后的标题嵌入维度
                norm_eps=norm_eps,  # 层归一化的 epsilon 值
                depth_init=depth_init,  # 是否使用基于深度的初始化
                layer_id=i,  # 当前块的索引
                num_layers=depth,  # Transformer 块的总数量
                compress_xattn=False,  # 是否压缩交叉注意力 qkv 维度
                use_bias=use_bias,  # 是否在线性层中使用偏置项
                moe_block=is_moe_block[i],  # 是否使用 MoE 块
                num_experts=num_experts,  # 专家的数量
                expert_capacity=expert_capacity  # 每个专家的容量因子
            ) for i in range(depth)
        ])
        
        # 注册掩码 token
        self.register_buffer(
            "mask_token", 
            torch.zeros(1, 1, patch_size ** 2 * self.out_channels)
        )
        # 创建最终层，用于生成最终图像
        self.final_layer = T2IFinalLayer(
            dim,  # 输入和输出维度
            dim,  # 输入和输出维度
            patch_size,  # 分块大小
            self.out_channels,  # 输出通道数
            approx_gelu,  # 使用近似 GELU 激活函数
            create_norm('layernorm', dim, eps=norm_eps)  # 使用层归一化
        )

        # 初始化模型权重
        self.initialize_weights()

    def forward_without_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        mask_ratio: float = 0,
        **kwargs
    ) -> dict:
        """
        不使用分类器自由指导的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
            t (torch.Tensor): 时间步张量，形状为 (batch_size,)。
            y (torch.Tensor): 标题嵌入张量，形状为 (batch_size, 1, seq_len, dim)。
            mask_ratio (float): 训练时掩码的比例（0到1之间），默认为0。

        Returns:
            dict: 包含以下键的字典：
                - 'sample': 输出张量，形状为 (batch_size, out_channels, height, width)。
                - 'mask': 可选的二进制掩码张量，如果应用了掩码则返回，否则为 None。
        """
        self.h = x.shape[-2] // self.patch_size  # 计算高度方向上的分块数量
        self.w = x.shape[-1] // self.patch_size  # 计算宽度方向上的分块数量

        # 对输入图像进行分块嵌入并添加位置嵌入，形状为 (N, T, D)，其中 T = H * W / patch_size ** 2
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        # 对时间步进行嵌入，形状为 (N, D)
        t = self.t_embedder(t.expand(x.shape[0]))  # (N, D)

        # 对标题嵌入进行投影，形状为 (N, 1, L, D)
        y = self.y_embedder(y)  # (N, 1, L, D)
        # 对标题嵌入进行预处理，形状从 (N, 1, L, D) 变为 (N, D)
        y = self.y_emb_preprocess(y.squeeze(dim=1)).unsqueeze(dim=1)  # (N, 1, L, D) -> (N, D)
        # 对标题嵌入进行池化处理
        y_pooled = self.pooled_y_emb_process(y.mean(dim=-2).squeeze(dim=1))
        # 将池化后的标题嵌入添加到时间步嵌入中
        t = t + y_pooled

        # 初始化掩码为 None
        mask = None
        
        if self.use_patch_mixer:
            # 对输入张量进行投影映射
            x = self.patch_mixer_map_xin(x)
            # 对标题嵌入进行投影映射
            y_mixer = self.patch_mixer_map_y(y)
            for block in self.patch_mixer:
                # 通过分块混合器块处理输入张量，形状为 (N, T, D_mixer)
                x = block(x, y_mixer, t)  # (N, T, D_mixer)
        
        if mask_ratio > 0:
            mask_dict = get_mask(
                x.shape[0], x.shape[1], 
                mask_ratio=mask_ratio, 
                device=x.device
            ) # 获取掩码字典
            # 获取保留的 token ID
            ids_keep = mask_dict['ids_keep']
            # 获取恢复的 token ID
            ids_restore = mask_dict['ids_restore']
            # 获取掩码张量
            mask = mask_dict['mask']
            # 对输入张量应用掩码
            x = mask_out_token(x, ids_keep)
        
        if self.use_patch_mixer:
            # Project mixer out to backbone transformer dim (do after masking to save compute)
            # 将混合器输出投影回主干 Transformer 维度（在应用掩码后进行以节省计算）
            x = self.patch_mixer_map_xout(x)

        for block in self.blocks:
            # 通过 Transformer 块处理输入张量，形状为 (N, T, D)
            x = block(x, y, t)  # (N, T, D)
        
        # 通过最终层处理输入张量，形状为 (N, T, patch_size ** 2 * out_channels)
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        
        if mask_ratio > 0:
            # 恢复被掩码的 token
            x = unmask_tokens(x, ids_restore, self.mask_token)

        # 将分块嵌入恢复为原始图像形状，形状为 (N, out_channels, H, W)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return {'sample': x, 'mask': mask}
    
    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg: float = 1.0,
        mask_ratio: float = 0,
        **kwargs
    ) -> dict:
        """
        使用分类器自由指导的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
            t (torch.Tensor): 时间步张量，形状为 (batch_size,)。
            y (torch.Tensor): 标题嵌入张量，形状为 (batch_size, 1, seq_len, dim)。
            cfg (float): 分类器自由指导的尺度（1.0 表示无指导）。
            mask_ratio (float): 训练时掩码的比例（0到1之间），默认为0。

        Returns:
            dict: 包含输出张量的字典，形状为 (batch_size, out_channels, height, width)。
        """
        # 将输入张量 x 和标题嵌入 y 分别拼接为原来的两倍大小
        # 拼接后的形状为 (2 * batch_size, channels, height, width)
        x = torch.cat([x, x], 0) 
        # 拼接后的形状为 (2 * batch_size, 1, seq_len, dim)
        y = torch.cat([y, torch.zeros_like(y)], 0)
        if len(t) != 1:
            # 如果时间步张量长度不为1，则拼接为原来的两倍大小
            t = torch.cat([t, t], 0)
        
        # 调用不使用分类器自由指导的前向传播方法，获取条件和无条件的 epsilon
        eps = self.forward_without_cfg(x, t, y, mask_ratio, **kwargs)['sample']
        # 将 epsilon 拆分为条件和无条件两部分
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        # 计算最终的 epsilon，结合了无条件部分和条件部分与 cfg 的乘积
        eps = uncond_eps + cfg * (cond_eps - uncond_eps)
        return {'sample': eps}

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg: float = 1.0,
        **kwargs
    ) -> dict:
        """
        根据分类器自由指导的值选择适当的前向传播方法。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
            t (torch.Tensor): 时间步张量，形状为 (batch_size,)。
            y (torch.Tensor): 标题嵌入张量，形状为 (batch_size, 1, seq_len, dim)。
            cfg (float): 分类器自由指导的尺度，默认为1.0。

        Returns:
            dict: 包含输出张量的字典。
        """
        if cfg != 1.0:
            # 如果 cfg 不为1.0，则使用带有分类器自由指导的前向传播方法
            return self.forward_with_cfg(x, t, y, cfg, **kwargs)
        else:
            # 否则，使用不带分类器自由指导的前向传播方法
            return self.forward_without_cfg(x, t, y, **kwargs)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        逆转分块嵌入过程，重建原始图像尺寸。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, T, patch_size ** 2 * out_channels)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, out_channels, height, width)。
        """
        # 输出通道数
        c = self.out_channels
        # 分块大小
        p = self.x_embedder.patch_size[0]
        # 计算高度和宽度，假设高度和宽度相等
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1]

        # 重塑张量为 (batch_size, height, width, patch_size, patch_size, out_channels)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        # 转置张量为 (batch_size, out_channels, height, patch_size, width, patch_size)
        x = torch.einsum('nhwpqc->nchpwq', x)
        # 重塑张量为 (batch_size, out_channels, height * patch_size, width * patch_size)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def initialize_weights(self) -> None:
        """
        使用自定义初始化方案初始化模型权重。
        """
        def zero_bias(m: nn.Module) -> None:
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                zero_bias(module)  # 所有线性层的偏置初始化为零

        # Baseline init of all parameters
        # 对所有参数进行基础初始化
        self.apply(_basic_init)

        # 生成 2D 正弦余弦位置嵌入，并赋值给位置嵌入张量
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches**0.5),
            pos_interp_scale=self.pos_interp_scale,
            base_size=self.base_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 对 x_embedder 的投影权重进行初始化
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # 对时间步嵌入器的 MLP 权重进行初始化
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # 对标题嵌入池化后的处理 MLP 权重进行初始化
        nn.init.normal_(self.pooled_y_emb_process.fc1.weight, std=0.02)
        nn.init.normal_(self.pooled_y_emb_process.fc2.weight, std=0.02)
        # 对标题嵌入投影层的 MLP 权重进行初始化
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)
        
        # Custom init of blocks
        # 对 Transformer 块和分块混合器块进行自定义初始化
        for block in self.blocks:
            block.custom_init()
        for block in self.patch_mixer:
            block.custom_init()

        # 对 AdaLN 调制模块的权重进行初始化
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)

        for block in self.patch_mixer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)

        # 对标题嵌入预处理注意力块的权重进行初始化
        self.y_emb_preprocess.custom_init()
        nn.init.constant_(self.y_emb_preprocess.attn.proj.weight, 0)
        nn.init.constant_(self.y_emb_preprocess.mlp.w3.weight, 0)

        # Zero-out output layers
        # 对最终层的权重进行初始化
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)


def MicroDiT_Tiny_2(
    caption_channels: int = 1024,
    qkv_ratio: List[float] = [0.5, 1.0],
    mlp_ratio: List[float] = [0.5, 4.0],
    pos_interp_scale: float = 1.0,
    input_size: int = 32,
    num_experts: int = 8,
    expert_capacity: float = 2.0,
    experts_every_n: int = 2,
    in_channels: int = 4,
    **kwargs
) -> DiT:
    """
    创建 MicroDiT_Tiny_2 模型实例。

    Args:
        caption_channels (int, 默认: 1024): 标题嵌入的通道数。
        qkv_ratio (List[float], 默认: [0.5, 1.0]): QKV 投影维度的比例列表。
        mlp_ratio (List[float], 默认: [0.5, 4.0]): MLP 隐藏层维度的比例列表。
        pos_interp_scale (float, 默认: 1.0): 位置嵌入插值缩放因子。
        input_size (int, 默认: 32): 输入图像的大小。
        num_experts (int, 默认: 8): 如果使用 MoE 块，专家的数量。
        expert_capacity (float, 默认: 2.0): 如果使用 MoE FFN 层，每个专家的容量因子。
        experts_every_n (int, 默认: 2): 每隔 n 个块添加一个 MoE FFN 层。
        in_channels (int, 默认: 4): 输入图像的通道数。
        **kwargs: 其他可选的关键字参数。

    Returns:
        DiT: 配置好的 MicroDiT_Tiny_2 模型实例。
    """
    # Transformer 块的数量，设置为16
    depth = 16

    # 创建 DiT 模型实例
    model = DiT(
        input_size=input_size,  # 输入图像的大小，默认为32
        patch_size=2,  # 图像分块的大小，设置为2
        in_channels=in_channels,  # 输入图像的通道数，默认为4
        dim=512,  # Transformer 主干网络的维度，设置为512
        depth=depth,  # Transformer 块的数量，设置为16
        head_dim=32,  # 每个注意力头的维度大小，设置为32
        multiple_of=256,  # MLP 块中隐藏层维度向上取整的倍数，设置为256
        caption_channels=caption_channels,  # 标题嵌入的通道数，默认为1024
        pos_interp_scale=pos_interp_scale,  # 位置嵌入插值缩放因子，默认为1.0
        norm_eps=1e-6,  # 层归一化的 epsilon 值，设置为1e-6
        depth_init=True,  # 是否在 DiT 块中使用基于深度的初始化，设置为True
        qkv_multipliers=np.linspace(qkv_ratio[0], qkv_ratio[1], num=depth, dtype=float),  # QKV 投影维度的比例列表，使用线性插值生成
        ffn_multipliers=np.linspace(mlp_ratio[0], mlp_ratio[1], num=depth, dtype=float),  # MLP 隐藏层维度的比例列表，使用线性插值生成
        use_patch_mixer=True,  # 是否使用分块混合器层，设置为True
        patch_mixer_depth=4,  # 分块混合器块的数量，设置为4
        patch_mixer_dim=512,  # 分块混合器层的维度大小，设置为512，分配更多的预算给混合器层
        patch_mixer_qkv_ratio=1.0,  # 分块混合器块中 QKV 投影维度的比例，设置为1.0
        patch_mixer_mlp_ratio=4.0,  # 分块混合器块中 MLP 隐藏层维度的比例，设置为4.0
        use_bias=False,  # 是否在线性层中使用偏置项，设置为False
        num_experts=num_experts,  # 如果使用 MoE 块，专家的数量，默认为8
        expert_capacity=expert_capacity,  # 如果使用 MoE FFN 层，每个专家的容量因子，默认为2.0
        experts_every_n=experts_every_n,  # 每隔 n 个块添加一个 MoE FFN 层，默认为2
        **kwargs  # 其他可选的关键字参数
    )
    return model


def MicroDiT_XL_2(
    caption_channels: int = 1024,
    qkv_ratio: List[float] = [0.5, 1.0],
    mlp_ratio: List[float] = [0.5, 4.0],
    pos_interp_scale: float = 1.0,
    input_size: int = 32,
    num_experts: int = 8,
    expert_capacity: float = 2.0,
    experts_every_n: int = 2,
    in_channels: int = 4,
    **kwargs
) -> DiT:
    """
    创建 MicroDiT_XL_2 模型实例。

    Args:
        caption_channels (int, 默认: 1024): 标题嵌入的通道数。
        qkv_ratio (List[float], 默认: [0.5, 1.0]): QKV 投影维度的比例列表。
        mlp_ratio (List[float], 默认: [0.5, 4.0]): MLP 隐藏层维度的比例列表。
        pos_interp_scale (float, 默认: 1.0): 位置嵌入插值缩放因子。
        input_size (int, 默认: 32): 输入图像的大小。
        num_experts (int, 默认: 8): 如果使用 MoE 块，专家的数量。
        expert_capacity (float, 默认: 2.0): 如果使用 MoE FFN 层，每个专家的容量因子。
        experts_every_n (int, 默认: 2): 每隔 n 个块添加一个 MoE FFN 层。
        in_channels (int, 默认: 4): 输入图像的通道数。
        **kwargs: 其他可选的关键字参数。

    Returns:
        DiT: 配置好的 MicroDiT_XL_2 模型实例。
    """
    # Transformer 块的数量，设置为28
    depth = 28

    # 创建 DiT 模型实例
    model = DiT(
        input_size=input_size,  # 输入图像的大小，默认为32
        patch_size=2,  # 图像分块的大小，设置为2
        in_channels=in_channels,  # 输入图像的通道数，默认为4
        dim=1024,  # Transformer 主干网络的维度，设置为1024
        depth=depth,  # Transformer 块的数量，设置为28
        head_dim=64,  # 每个注意力头的维度大小，设置为64
        multiple_of=256,  # MLP 块中隐藏层维度向上取整的倍数，设置为256
        caption_channels=caption_channels,  # 标题嵌入的通道数，默认为1024
        pos_interp_scale=pos_interp_scale,  # 位置嵌入插值缩放因子，默认为1.0
        norm_eps=1e-6,  # 层归一化的 epsilon 值，设置为1e-6
        depth_init=True,  # 是否在 DiT 块中使用基于深度的初始化，设置为True
        qkv_multipliers=np.linspace(qkv_ratio[0], qkv_ratio[1], num=depth, dtype=float),  # QKV 投影维度的比例列表，使用线性插值生成
        ffn_multipliers=np.linspace(mlp_ratio[0], mlp_ratio[1], num=depth, dtype=float),  # MLP 隐藏层维度的比例列表，使用线性插值生成
        use_patch_mixer=True,  # 是否使用分块混合器层，设置为True
        patch_mixer_depth=6,  # 分块混合器块的数量，设置为6
        patch_mixer_dim=768,  # 分块混合器层的维度大小，设置为768，分配更多的预算给混合器层
        patch_mixer_qkv_ratio=1.0,  # 分块混合器块中 QKV 投影维度的比例，设置为1.0
        patch_mixer_mlp_ratio=4.0,  # 分块混合器块中 MLP 隐藏层维度的比例，设置为4.0
        use_bias=False,  # 是否在线性层中使用偏置项，设置为False
        num_experts=num_experts,  # 如果使用 MoE 块，专家的数量，默认为8
        expert_capacity=expert_capacity,  # 如果使用 MoE FFN 层，每个专家的容量因子，默认为2.0
        experts_every_n=experts_every_n,  # 每隔 n 个块添加一个 MoE FFN 层，默认为2
        **kwargs  # 其他可选的关键字参数
    )

    return model
