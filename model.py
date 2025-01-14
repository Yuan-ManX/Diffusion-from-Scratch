from functools import partial
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models import ComposerModel
from diffusers import AutoencoderKL
from easydict import EasyDict

import dit as model_zoo
from utils import (
    DATA_TYPES,
    DistLoss,
    UniversalTextEncoder,
    UniversalTokenizer,
    text_encoder_embedding_format,
)


class LatentDiffusion(ComposerModel):
    """
    潜在扩散模型，用于从文本提示生成图像。

    该模型结合了用于去噪图像潜在向量的 DiT（Diffusion Transformer）模型、
    用于在潜在空间中对图像进行编码/解码的 VAE（变分自编码器），
    以及用于将文本提示转换为嵌入的文本编码器。
    它实现了 EDM（Elucidated Diffusion Model）采样过程。

    参数:
        dit (nn.Module): 扩散 Transformer 模型，用于去噪图像潜在向量。
        vae (AutoencoderKL): 来自 diffusers 的 VAE 模型，用于编码/解码图像。
        text_encoder (UniversalTextEncoder): 文本编码器，用于将提示转换为嵌入。
        tokenizer (UniversalTokenizer): 分词器，用于处理文本提示。
        image_key (str, optional): 批次字典中图像数据的键。默认为 'image'。
        text_key (str, optional): 批次字典中文本数据的键。默认为 'captions'。
        image_latents_key (str, optional): 批次字典中预计算的图像潜在向量的键。默认为 'image_latents'。
        text_latents_key (str, optional): 批次字典中预计算的文本潜在向量的键。默认为 'caption_latents'。
        precomputed_latents (bool, optional): 是否使用预计算的潜在向量（必须在批次中）。默认为 True。
        dtype (str, optional): 模型操作的数据类型。默认为 'bfloat16'。
        latent_res (int, optional): 潜在空间的分辨率，假设 VAE 进行 8 倍下采样。默认为 32。
        p_mean (float, optional): EDM 对数正态噪声的均值。默认为 -0.6。
        p_std (float, optional): EDM 对数正态噪声的标准差。默认为 1.2。
        train_mask_ratio (float, optional): 训练期间要掩码的补丁比例。默认为 0。
    """

    def __init__(
        self,
        dit: nn.Module,
        vae: AutoencoderKL,
        text_encoder: UniversalTextEncoder,
        tokenizer: UniversalTokenizer,
        image_key: str = 'image',
        text_key: str = 'captions',
        image_latents_key: str = 'image_latents',
        text_latents_key: str = 'caption_latents',
        precomputed_latents: bool = True,
        dtype: str = 'bfloat16',
        latent_res: int = 32,
        p_mean: float = -0.6,
        p_std: float = 1.2,
        train_mask_ratio: float = 0.
    ):
        super().__init__()
        self.dit = dit  # 初始化 DiT 模型
        self.vae = vae  # 初始化 VAE 模型
        self.image_key = image_key  # 图像数据在批次字典中的键
        self.text_key = text_key  # 文本数据在批次字典中的键
        self.image_latents_key = image_latents_key  # 预计算的图像潜在向量在批次字典中的键
        self.text_latents_key = text_latents_key  # 预计算的文本潜在向量在批次字典中的键
        self.precomputed_latents = precomputed_latents  # 是否使用预计算的潜在向量
        self.dtype = dtype  # 模型操作的数据类型
        self.latent_res = latent_res  # 潜在空间的分辨率
        self.edm_config = EasyDict({  # 初始化 EDM 配置
            'sigma_min': 0.002,  # EDM 的最小 sigma 值
            'sigma_max': 80,      # EDM 的最大 sigma 值
            'P_mean': p_mean,     # EDM 对数正态噪声的均值
            'P_std': p_std,       # EDM 对数正态噪声的标准差
            'sigma_data': 0.9,    # 数据 sigma 值
            'num_steps': 18,      # 采样步骤数
            'rho': 7,             # EDM 的 rho 参数
            'S_churn': 0,         # EDM 的 S_churn 参数
            'S_min': 0,           # EDM 的 S_min 参数
            'S_max': float('inf'),# EDM 的 S_max 参数
            'S_noise': 1          # EDM 的 S_noise 参数
        })
        self.train_mask_ratio = train_mask_ratio  # 训练期间的掩码比例
        self.eval_mask_ratio = 0.  # 评估期间不进行掩码
        assert self.train_mask_ratio >= 0, 'Masking ratio must be non-negative!'

        # 定义 randn_like 函数
        self.randn_like = torch.randn_like
        # 获取 VAE 的缩放因子
        self.latent_scale = self.vae.config.scaling_factor

        # 初始化文本编码器
        self.text_encoder = text_encoder
        # 初始化分词器
        self.tokenizer = tokenizer
        # freeze vae and text_encoder during training
        # 在训练期间冻结 VAE 和文本编码器
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        # avoid wrapping the models that we aren't training
        # 避免包装我们不训练的模型
        self.text_encoder._fsdp_wrap = False
        self.vae._fsdp_wrap = False
        # 对 DiT 模型进行 FSDP 包装
        self.dit._fsdp_wrap = True

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播方法。

        从批次字典中获取图像潜在向量和文本嵌入，并计算损失。

        参数:
            batch (dict): 输入批次字典，包含图像、文本和其他相关信息。

        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 损失、图像潜在向量和文本嵌入。
        """
        # 获取图像潜在向量
        if self.precomputed_latents and self.image_latents_key in batch:
            # Assuming that latents have already been scaled, i.e., multiplied with the scaling factor
            # 假设潜在向量已经过缩放，即乘以缩放因子
            latents = batch[self.image_latents_key]
        else:
            with torch.no_grad():
                # 获取图像数据
                images = batch[self.image_key]
                latents = self.vae.encode(
                    images.to(DATA_TYPES[self.dtype]) # 将图像数据转换为指定的数据类型
                )['latent_dist'].sample().data # 对图像进行编码并采样潜在向量
                # 缩放潜在向量
                latents *= self.latent_scale

        # Get text embeddings
        # 获取文本嵌入
        if self.precomputed_latents and self.text_latents_key in batch:
            conditioning = batch[self.text_latents_key]
        else:
            # 获取文本数据
            captions = batch[self.text_key]
            # 重塑为 (batch_size, seq_length)
            captions = captions.view(-1, captions.shape[-1])
            if 'attention_mask' in batch:
                # 如果存在注意力掩码，则使用注意力掩码进行编码
                conditioning = self.text_encoder.encode(
                    captions,
                    attention_mask=batch['attention_mask'].view(-1, captions.shape[-1])
                )[0]
            else:
                # 否则，直接进行编码
                conditioning = self.text_encoder.encode(captions)[0]

        # Zero out dropped captions. Needed for classifier-free guidance during inference.
        # 用于推理期间的分类器自由指导。
        if 'drop_caption_mask' in batch.keys():
            conditioning *= batch['drop_caption_mask'].view(
                [-1] + [1] * (len(conditioning.shape) - 1)
            ) # 将掩码应用于条件嵌入

        # 计算损失
        loss = self.edm_loss(
            latents.float(),  # 将潜在向量转换为浮点数
            conditioning.float(),  # 将条件嵌入转换为浮点数
            mask_ratio=self.train_mask_ratio if self.training else self.eval_mask_ratio
        )  # 根据训练状态选择掩码比例
        # 返回损失、潜在向量和条件嵌入
        return (loss, latents, conditioning)

    def model_forward_wrapper(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        y: torch.Tensor,
        model_forward_fxn: callable,
        mask_ratio: float,
        **kwargs
    ) -> dict:
        """
        EDM 模型调用的包装器。

        该方法实现了 EDM 采样过程中的前向传播步骤，包括对输入进行缩放、计算缩放因子，并调用模型的前向传播方法。

        参数:
            x (torch.Tensor): 输入张量。
            sigma (torch.Tensor): 标准差张量。
            y (torch.Tensor): 条件嵌入。
            model_forward_fxn (callable): 模型的前向传播函数。
            mask_ratio (float): 掩码比例。
            **kwargs: 其他关键字参数。

        返回:
            Dict: 包含模型输出和其他相关信息。
        """
        # 将 sigma 转换为与 x 相同的 dtype 并重塑为 (batch_size, 1, 1, 1)
        sigma = sigma.to(x.dtype).reshape(-1, 1, 1, 1)
        # 计算缩放因子 c_skip, c_out, c_in 和 c_noise
        c_skip = (
            self.edm_config.sigma_data ** 2 /
            (sigma ** 2 + self.edm_config.sigma_data ** 2)
        )
        c_out = (
            sigma * self.edm_config.sigma_data /
            (sigma ** 2 + self.edm_config.sigma_data ** 2).sqrt()
        )
        c_in = 1 / (self.edm_config.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # 调用模型的前向传播函数
        out = model_forward_fxn(
            (c_in * x).to(x.dtype),   # 对输入进行缩放
            c_noise.flatten(),        # 将 c_noise 展平并传递给模型
            y,                        # 传递条件嵌入
            mask_ratio=mask_ratio,    # 传递掩码比例
            **kwargs                  # 传递其他关键字参数
        )
        # 获取模型的输出样本
        F_x = out['sample']
        # 将 c_skip 移动到与输出样本相同的设备
        c_skip = c_skip.to(F_x.device)
        # 将输入 x 移动到与输出样本相同的设备
        x = x.to(F_x.device)
        # 将 c_out 移动到与输出样本相同的设备
        c_out = c_out.to(F_x.device)
        # 计算最终的输出 D_x
        D_x = c_skip * x + c_out * F_x
        # 将 D_x 赋值给输出字典中的 'sample'
        out['sample'] = D_x
        # 返回输出字典
        return out

    def edm_loss(self, x: torch.Tensor, y: torch.Tensor, mask_ratio: float = 0, **kwargs) -> torch.Tensor:
        """
        EDM 损失计算方法。

        该方法计算去噪过程中每个噪声水平的损失。

        参数:
            x (torch.Tensor): 输入张量。
            y (torch.Tensor): 条件嵌入。
            mask_ratio (float, optional): 掩码比例。默认为 0。
            **kwargs: 其他关键字参数。

        返回:
            torch.Tensor: 计算得到的损失。
        """
        # 生成随机噪声
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        # 计算 sigma 值
        sigma = (rnd_normal * self.edm_config.P_std + self.edm_config.P_mean).exp()
        # 计算权重
        weight = (
            (sigma ** 2 + self.edm_config.sigma_data ** 2) /
            (sigma * self.edm_config.sigma_data) ** 2
        )
        # 生成噪声
        n = self.randn_like(x) * sigma

        # 调用模型前向传播包装器进行前向传播
        model_out = self.model_forward_wrapper(
            x + n,
            sigma,
            y,
            self.dit,
            mask_ratio=mask_ratio,
            **kwargs
        )
        # 获取去噪后的输出
        D_xn = model_out['sample']
        # 计算损失 (N, C, H, W)
        loss = weight * ((D_xn - x) ** 2)  # (N, C, H, W)

        if mask_ratio > 0:
            # Masking is not feasible during image generation as it only returns denoised version
            # for non-masked patches. Image generation requires all patches to be denoised.
            # 在图像生成过程中，掩码是不可行的，因为它只返回去噪后的非掩码补丁。图像生成需要所有补丁都被去噪。
            assert (
                self.dit.training and 'mask' in model_out
            ), 'Masking is only recommended during training'
            # 对损失进行平均池化
            loss = F.avg_pool2d(loss.mean(dim=1), self.dit.patch_size).flatten(1)
            # 获取未掩码的区域
            unmask = 1 - model_out['mask']
            # 计算平均损失 (N,)
            loss = (loss * unmask).sum(dim=1) / unmask.sum(dim=1)  # (N,)
        # 返回平均损失
        return loss.mean()

    # Composer specific formatting of model loss and eval functions.
    # Composer 特定的模型损失和评估函数格式。
    def loss(self, outputs: tuple, batch: dict) -> torch.Tensor:
        # 前向传递已经计算了损失函数
        return outputs[0]

    def eval_forward(self, batch: dict, outputs: Optional[tuple] = None) -> tuple:
        # Skip if output already calculated (e.g., during training forward pass)
        # 如果输出已经计算（例如，在训练前向传递期间），则跳过
        if outputs is not None:
            return outputs
        loss, _, _ = self.forward(batch)
        return loss, None, None

    def get_metrics(self, is_train: bool = False) -> dict:
        # get_metrics expected to return a dict in composer
        # get_metrics 预期返回一个字典在 composer 中
        return {'loss': DistLoss()}

    def update_metric(self, batch: dict, outputs: tuple, metric: DistLoss):
        """
        更新度量指标。

        Args:
            batch (dict): 当前批次的数据。
            outputs (tuple): 模型的前向传播输出。
            metric (DistLoss): 用于跟踪和更新损失度量的对象。
        """
        # 使用当前批次的前向输出更新度量指标
        metric.update(outputs[0])

    @torch.no_grad()
    def edm_sampler_loop(
        self, x: torch.Tensor, 
        y: torch.Tensor, 
        steps: Optional[int] = None, 
        cfg: float = 1.0, 
        **kwargs
    ) -> torch.Tensor:
        """
        EDM（Euler-Maruyama Diffusion Model）采样循环。

        Args:
            x (torch.Tensor): 初始的噪声张量，通常是随机生成的。
            y (torch.Tensor): 条件信息，如类别标签，用于指导生成过程。
            steps (Optional[int]): 采样步数。如果未提供，则使用配置中的步数。
            cfg (float): 控制采样过程的配置参数，默认为1.0。
            **kwargs: 其他可选的关键字参数。

        Returns:
            torch.Tensor: 生成的结果张量。
        """
        # 初始化掩码比例为，设为0表示在图像生成过程中不进行掩码操作
        mask_ratio = 0  # no masking during image generation

        # 根据cfg参数选择模型的前向传播函数
        model_forward_fxn = (
            partial(self.dit.forward, cfg=cfg) if cfg > 1.0  # 如果cfg大于1.0，使用partial包装dit.forward，并传入cfg参数
            else self.dit.forward  # 否则，直接使用dit.forward
        )

        # Time step discretization.
        # 时间步长的离散化
        num_steps = self.edm_config.num_steps if steps is None else steps
        # 生成时间步长的索引，从0到num_steps-1
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=x.device)
        # 计算时间步长的sigma值，使用逆时间步长的幂律缩放
        t_steps = (
            self.edm_config.sigma_max ** (1 / self.edm_config.rho) +
            step_indices / (num_steps - 1) *
            (self.edm_config.sigma_min ** (1 / self.edm_config.rho) -
             self.edm_config.sigma_max ** (1 / self.edm_config.rho))
        ) ** self.edm_config.rho
        # 在t_steps的末尾添加一个0，确保循环中最后一个时间步的处理
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])

        # Main sampling loop.
        # 将初始噪声张量x转换为float64，并乘以初始的t_steps值
        x_next = x.to(torch.float64) * t_steps[0]
        # 主采样循环，遍历每一个时间步
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next
            # Increase noise temporarily.
            # 计算gamma值，用于增加噪声
            gamma = (
                min(self.edm_config.S_churn / num_steps, np.sqrt(2) - 1)
                if self.edm_config.S_min <= t_cur <= self.edm_config.S_max else 0
            )
            # 计算调整后的t_hat
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            # 计算调整后的噪声张量x_hat
            x_hat = (
                x_cur +
                (t_hat ** 2 - t_cur ** 2).sqrt() *
                self.edm_config.S_noise *
                self.randn_like(x_cur)
            )

            # Euler step.
            # 执行Euler步长
            denoised = self.model_forward_wrapper(
                x_hat.to(torch.float32),
                t_hat.to(torch.float32),
                y,
                model_forward_fxn,
                mask_ratio=mask_ratio,
                **kwargs
            )['sample'].to(torch.float64)
            # 计算当前步的导数d_cur
            d_cur = (x_hat - denoised) / t_hat
            # 更新x_next为下一个时间步的预测值
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            # 应用二阶校正
            if i < num_steps - 1:
                # 再次使用更新后的x_next进行前向传播
                denoised = self.model_forward_wrapper(
                    x_next.to(torch.float32),
                    t_next.to(torch.float32),
                    y,
                    model_forward_fxn,
                    mask_ratio=mask_ratio,
                    **kwargs
                )['sample'].to(torch.float64)
                # 计算二阶导数d_prime
                d_prime = (x_next - denoised) / t_next
                # 应用二阶校正，更新x_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        # 将最终结果转换回float32并返回
        return x_next.to(torch.float32)

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[list] = None,
        tokenized_prompts: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        guidance_scale: Optional[float] = 5.0,
        num_inference_steps: Optional[int] = 30,
        seed: Optional[int] = None,
        return_only_latents: Optional[bool] = False,
        **kwargs
    ) -> torch.Tensor:
        """
        使用扩散模型生成图像。

        Args:
            prompt (Optional[List[str]]): 文本提示，可以是字符串列表。
            tokenized_prompts (Optional[torch.LongTensor]): 预分词的 token ID 序列。如果提供，则跳过分词步骤。
            attention_mask (Optional[torch.LongTensor]): 注意力掩码，用于指示哪些 token 是有效的。
            guidance_scale (Optional[float]): 指导尺度，默认为5.0。较高的值会使生成结果更符合文本提示。
            num_inference_steps (Optional[int]): 推理步数，默认为30。步数越多，生成质量通常越高，但计算时间也越长。
            seed (Optional[int]): 随机种子，用于结果可复现。如果提供，则设置随机数生成器的种子。
            return_only_latents (Optional[bool]): 是否仅返回潜在向量。默认为False，表示返回解码后的图像。
            **kwargs: 其他可选的关键字参数。

        Returns:
            torch.Tensor: 生成的结果图像或潜在向量。
        """
        # 检查是否提供了必要的输入
        # _check_prompt_given(prompt, tokenized_prompts, prompt_embeds=None)  
        # 假设有一个检查函数
        assert prompt or tokenized_prompts, "Must provide either prompt or tokenized prompts"
        # 通过 VAE 获取设备，假设所有组件在同一设备上
        device = self.vae.device  # hack to identify model device during training
        # 初始化随机数生成器
        rng_generator = torch.Generator(device=device)
        if seed:
            # 如果提供了种子，则设置随机数生成器的种子以确保结果可复现
            rng_generator = rng_generator.manual_seed(seed)

        # Convert prompt text to embeddings (zero out embeddings for classifier-free guidance)
        # 如果没有提供预分词的 token ID，则进行分词
        if tokenized_prompts is None:
            # 使用分词器对文本提示进行分词
            out = self.tokenizer.tokenize(prompt)
            tokenized_prompts = out['input_ids']
            # 如果分词结果中包含注意力掩码，则使用它；否则，设置为 None
            attention_mask = (
                out['attention_mask'] if 'attention_mask' in out else None
            )
        # 使用文本编码器将 token ID 序列转换为嵌入向量
        text_embeddings = self.text_encoder.encode(
            tokenized_prompts.to(device),
            attention_mask=attention_mask.to(device) if attention_mask is not None else None
        )[0]

        # 生成随机潜在向量，形状为 (batch_size, 通道数, 高度, 宽度)
        latents = torch.randn(
            (len(text_embeddings), self.dit.in_channels, self.latent_res, self.latent_res),
            device=device,
            generator=rng_generator,
        )

        # iteratively denoise latents
        # 使用 EDM 采样循环对潜在向量进行逐步去噪
        latents = self.edm_sampler_loop(
            latents,
            text_embeddings,
            num_inference_steps,
            cfg=guidance_scale
        )

        # 如果只需要潜在向量，则直接返回
        if return_only_latents:
            return latents

        # Decode latents with VAE
        # 对潜在向量进行解码，生成图像
        # 对潜在向量进行缩放
        latents = 1 / self.latent_scale * latents
        # 将潜在向量的数据类型转换为 VAE 所需的类型
        torch_dtype = DATA_TYPES[self.dtype]
        # 使用 VAE 对潜在向量进行解码，生成图像样本
        image = self.vae.decode(latents.to(torch_dtype)).sample
        # 将图像像素值归一化到 [0, 1] 范围
        image = (image / 2 + 0.5).clamp(0, 1)
        # 将图像转换为 float 类型并分离出计算图
        image = image.float().detach()
        # 返回生成的图像
        return image
    
    
def create_latent_diffusion(
    vae_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
    text_encoder_name: str = 'openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378', 
    dit_arch: str = 'MicroDiT_XL_2',
    latent_res: int = 32,
    in_channels: int = 4,
    pos_interp_scale: float = 1.0,
    dtype: str = 'bfloat16',
    precomputed_latents: bool = True,
    p_mean: float = -0.6,
    p_std: float = 1.2,
    train_mask_ratio: float = 0.
) -> LatentDiffusion:
    """
    创建并返回一个 LatentDiffusion 模型实例。

    Args:
        vae_name (str): VAE 的预训练模型名称或路径。默认为 'stabilityai/stable-diffusion-xl-base-1.0'。
        text_encoder_name (str): 文本编码器的名称或路径。默认为 'openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378'。
        dit_arch (str): 扩散模型的架构名称。默认为 'MicroDiT_XL_2'。
        latent_res (int): 潜在向量的分辨率（高度和宽度）。默认为32。
        in_channels (int): 输入通道数。默认为4。
        pos_interp_scale (float): 位置插值缩放因子。默认为1.0。
        dtype (str): 数据类型。默认为 'bfloat16'。
        precomputed_latents (bool): 是否使用预计算的潜在向量。默认为True。
        p_mean (float): 正态分布的均值，用于初始化潜在向量。默认为-0.6。
        p_std (float): 正态分布的标准差，用于初始化潜在向量。默认为1.2。
        train_mask_ratio (float): 训练时的掩码比例。默认为0.0。

    Returns:
        LatentDiffusion: 配置好的 LatentDiffusion 模型实例。
    """
    # 从文本编码器获取最大序列长度 (s) 和 token 嵌入维度 (d)
    s, d = text_encoder_embedding_format(text_encoder_name)

    # 从预训练模型库中获取并初始化扩散模型的主干网络 (DiT)
    dit = getattr(model_zoo, dit_arch)(      # 使用 getattr 从 model_zoo 获取指定的模型架构
        input_size=latent_res,               # 输入大小为潜在向量的分辨率
        caption_channels=d,                  # 标题通道数为 token 嵌入维度
        pos_interp_scale=pos_interp_scale,   # 位置插值缩放因子
        in_channels=in_channels              # 输入通道数
    )

    # 从预训练模型库中加载 VAE 模型
    vae = AutoencoderKL.from_pretrained(
        vae_name,
        subfolder=None if vae_name=='ostris/vae-kl-f8-d16' else 'vae',
        torch_dtype=DATA_TYPES[dtype],
        pretrained=True
    )

    # 初始化文本编码器
    text_encoder = UniversalTextEncoder(
        text_encoder_name,
        dtype=dtype,
        pretrained=True
    )
    # 初始化分词器
    tokenizer = UniversalTokenizer(text_encoder_name)

    # 创建 LatentDiffusion 模型实例
    model = LatentDiffusion(
        dit=dit,                                  # 扩散模型的主干网络
        vae=vae,                                  # 变分自编码器
        text_encoder=text_encoder,                # 文本编码器
        tokenizer=tokenizer,                      # 分词器
        precomputed_latents=precomputed_latents,  # 是否使用预计算的潜在向量
        dtype=dtype,                              # 数据类型
        latent_res=latent_res,                    # 潜在向量的分辨率
        p_mean=p_mean,                            # 正态分布的均值，用于初始化潜在向量
        p_std=p_std,                              # 正态分布的标准差，用于初始化潜在向量
        train_mask_ratio=train_mask_ratio         # 训练时的掩码比例
    )
    
    return model
