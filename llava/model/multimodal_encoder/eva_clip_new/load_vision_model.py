import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from functools import partial
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from .hf_model import HFTextEncoder
except:
    HFTextEncoder = None
from .timm_model import TimmModel
from .eva_vit_model import EVAVisionTransformer
from .transformer import LayerNorm, QuickGELU, Attention, VisionTransformer, LayerNormFp32

try:
    from apex.normalization import FusedLayerNorm
except:
    FusedLayerNorm = LayerNorm
    print("Please 'pip install apex'")

try:
    import xformers.ops as xops
except ImportError:
    xops = None
    print("Please 'pip install xformers'")

from .transform import image_transform

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0. # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    global_average_pool: bool = False # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    drop_path_rate: Optional[float] = None  # drop path rate
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = True  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'token'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    eva_model_name: str = None # a valid eva model name overrides layers, width, patch_size
    qkv_bias: bool = True
    fusedLN: bool = False
    xattn: bool = False
    postnorm: bool = False
    rope: bool = False
    pt_hw_seq_len: int = 16   # 224/14
    intp_freq: bool = False
    naiveswiglu: bool = False
    subln: bool = False

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU
    if vision_cfg.eva_model_name:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNorm
        
        visual = EVAVisionTransformer(
            img_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            num_classes=embed_dim,
            use_mean_pooling=vision_cfg.global_average_pool, #False
            init_values=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            embed_dim=vision_cfg.width,
            depth=vision_cfg.layers,
            num_heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            qkv_bias=vision_cfg.qkv_bias,
            drop_path_rate=vision_cfg.drop_path_rate,
            norm_layer= partial(FusedLayerNorm, eps=1e-6) if vision_cfg.fusedLN else partial(norm_layer, eps=1e-6),
            xattn=vision_cfg.xattn,
            rope=vision_cfg.rope,
            postnorm=vision_cfg.postnorm,
            pt_hw_seq_len= vision_cfg.pt_hw_seq_len,   # 224/14
            intp_freq= vision_cfg.intp_freq,
            naiveswiglu= vision_cfg.naiveswiglu,
            subln= vision_cfg.subln
        )
    elif vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size
        )
        act_layer = nn.GELU  # so that text transformer doesn't use QuickGELU w/ timm models
    # elif isinstance(vision_cfg.layers, (tuple, list)):
    #     vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
    #     visual = ModifiedResNet(
    #         layers=vision_cfg.layers,
    #         output_dim=embed_dim,
    #         heads=vision_heads,
    #         image_size=vision_cfg.image_size,
    #         width=vision_cfg.width
    #     )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            global_average_pool=vision_cfg.global_average_pool,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual



class EVA_VISION(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            itm_task: bool = False,
    ):
        super().__init__()
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'logit_scale'}

    def encode_image(self, image, normalize: bool = False, forward_head=False):
        if forward_head:
            features = self.visual.head(image)
        else:
            features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features


    def forward(self, image):
        image_features = self.encode_image(image, normalize=True, forward_head=False)
        return image_features


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, nn.Parameter):
            l.data = l.data.to(dtype)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name) and isinstance(l, nn.Parameter):
                attr = getattr(l, name, None)
                if attr is not None:
                    attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)




def create_transforms(
        config: CLIPVisionCfg,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
):
    model = EVA_VISION(
        embed_dim=config.width,
        vision_cfg=config,
        quick_gelu=False,
        cast_dtype=None,
        itm_task=False,
    )

    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess_train = image_transform(
        model.visual.image_size,
        is_train=True,
        mean=image_mean,
        std=image_std
    )
    preprocess_val = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std
    )

    return model, preprocess_train, preprocess_val
config = CLIPVisionCfg(**json.load(open('eva_clip/model_configs/EVA02-CLIP-L-14-336.json')))
model, preprocess_train, preprocess_val = create_transforms(config)

