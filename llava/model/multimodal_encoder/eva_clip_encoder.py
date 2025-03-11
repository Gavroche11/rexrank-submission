import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from functools import partial
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
from PIL import Image
try:
    from .eva_clip_new.hf_model import HFTextEncoder
except:
    HFTextEncoder = None
from .eva_clip_new.timm_model import TimmModel
from .eva_clip_new.eva_vit_model import EVAVisionTransformer
from .eva_clip_new.transformer import LayerNorm, QuickGELU, Attention, VisionTransformer, LayerNormFp32
from .eva_clip_new.factory import _rescan_model_configs, get_model_config
from .eva_clip_new.transform import EvaClipImageTrainProcessor

from llava.utils import rank0_print

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

    def encode_image(self, image, normalize: bool = False, forward_head=False, **kwargs):
        if forward_head:
            features = self.visual.head(image)
        else:
            features = self.visual(image, **kwargs)
        return F.normalize(features, dim=-1) if normalize else features
    
    def get_patch_embeddings(self, image):
        """Extract patch embeddings before the projection head"""
        # Access the patch embeddings directly
        x = self.visual.patch_embed(image)
        print(f"patch_embed: {x.shape}")
        # Check if model uses class tokens (common in ViT models)
        if hasattr(self.visual, 'cls_token') and self.visual.cls_token is not None:
            cls_tokens = self.visual.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        print(f"cls_token: {x.shape}")
        # Add positional embeddings if the model has them
        if hasattr(self.visual, 'pos_embed') and self.visual.pos_embed is not None:
            # Make sure dimensions match
            print(x.shape, self.visual.pos_embed.shape)
            if x.size(1) != self.visual.pos_embed.size(1):
                # For models where pos_embed includes the class token position
                # but we're not using class token
                if not hasattr(self.visual, 'cls_token') and x.size(1) + 1 == self.visual.pos_embed.size(1):
                    x = torch.cat([torch.zeros(x.shape[0], 1, x.shape[2], device=x.device), x], dim=1)
                # For other dimension mismatches, use only the patch position embeddings
                elif hasattr(self.visual, 'cls_token') and x.size(1) == self.visual.pos_embed.size(1):
                    x = x + self.visual.pos_embed
                else:
                    # Skip positional embeddings if dimensions don't match
                    print(f"Warning: Skipping positional embeddings due to dimension mismatch. x: {x.shape}, pos_embed: {self.visual.pos_embed.shape}")
            else:
                x = x + self.visual.pos_embed
        print(f"pos_embed: {x.shape}")
        # Apply any initial normalization or dropout
        if hasattr(self.visual, 'pos_drop'):
            x = self.visual.pos_drop(x)
        print(x.shape)
        # Process through transformer blocks
        for blk in self.visual.blocks:
            x = blk(x)
        print(x.shape)
        # Apply final normalization if present
        if hasattr(self.visual, 'norm') and self.visual.norm is not None:
            x = self.visual.norm(x)
        print(x.shape)
        
        return x

    def forward(self, image, **kwargs):
        image_features = self.encode_image(image, normalize=True, forward_head=False, **kwargs)
        return image_features
    
    # def get_patch_embedding(self, image):
    #     image_features = self.visual.patch_embedding(image)
    #     return image_features


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


# def _create_model_and_transforms(
#         embed_dim:int,
#         config: CLIPVisionCfg,
#         image_mean: Optional[Tuple[float, ...]] = None,
#         image_std: Optional[Tuple[float, ...]] = None,
# ):
#     model = EVA_VISION(
#         embed_dim=embed_dim,
#         vision_cfg=config,
#         quick_gelu=False,
#         cast_dtype=None,
#         itm_task=False,
#     )

#     image_mean = image_mean or getattr(model.visual, 'image_mean', None)
#     image_std = image_std or getattr(model.visual, 'image_std', None)
#     preprocess_train = image_transform(
#         model.visual.image_size,
#         is_train=True,
#         mean=image_mean,
#         std=image_std
#     )
#     preprocess_val = image_transform(
#         model.visual.image_size,
#         is_train=False,
#         mean=image_mean,
#         std=image_std
#     )

#     return model, preprocess_train, preprocess_val


class EvaClipEncoderWrapper(nn.Module):
    def __init__(self, vision_tower_pretrained: str, config: dict):
        super(EvaClipEncoderWrapper, self).__init__()
        self.vision_tower_pretrained = vision_tower_pretrained
        self.config = config
        self.vision_config = CLIPVisionCfg(**self.config["vision_cfg"])
        self.model = EVA_VISION(
                                    embed_dim=self.config["embed_dim"],
                                    vision_cfg=self.vision_config,
                                    quick_gelu=False,
                                    cast_dtype=None,
                                    itm_task=False,
                                )
        
        self.model.visual.load_state_dict(torch.load(vision_tower_pretrained), strict=False)

    def forward(self, image):
        return self.model(image, return_all_features=True)

    @property
    def dtype(self):
        return list(self.parameters())[-1].dtype
    
    @property
    def device(self):
        return list(self.parameters())[-1].device


class EvaClipVisionTower(nn.Module):    
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.vision_tower_pretrained = args.vision_tower_pretrained

        _rescan_model_configs()
        self.config = get_model_config(vision_tower) # dict
        self.vision_config = CLIPVisionCfg(**self.config['vision_cfg']) # vision config

        if not delay_load:
            rank0_print(f"Loading EVA-CLIP: {self.vision_tower_name}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        rank0_print(f"Pretrained: {self.vision_tower_pretrained}")
        # self.image_processor = EvaXImageTrainProcessor(self.config["vision_cfg"]["image_size"])
        # self.vision_tower = EVAXEncoderWrapper(self.vision_tower_pretrained, self.config)

        self.vision_tower = EvaClipEncoderWrapper(
            vision_tower_pretrained=self.vision_tower_pretrained,
            config=self.config
        ).to(torch.bfloat16)

        image_mean = getattr(self.vision_tower.model.visual, "image_mean", None)
        image_std = getattr(self.vision_tower.model.visual, "image_std", None)

        self.image_processor = EvaClipImageTrainProcessor(
            image_size=self.vision_tower.model.visual.image_size,
            mean=image_mean,
            std=image_std
        )

        rank0_print(f"Loaded image processor: {self.image_processor}")
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0)).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype)).to(images.dtype)

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.config["vision_cfg"]["width"]

    @property
    def image_size(self):
        return self.config["vision_cfg"]["image_size"]

# device = 'cuda'
# config = json.load(open('eva_clip/model_configs/EVA02-CLIP-L-14-336.json'))
# vision_config = CLIPVisionCfg(**config['vision_cfg'])
# model, preprocess_train, preprocess_val = _create_model_and_transforms(config['embed_dim'], vision_config)
# CSV_FILE = '/home/BARO_Cluster/data/data/csv/llm2clip/mimic_clip_test.csv'  # Replace with your CSV file path
# MODEL_CHECKPOINT_PATH = '/home/BARO_Cluster/data/data/research/model/llm2clip/pytorch_model.bin'  # Replace with your EVA-CLIP model checkpoint path

# # After creating the model and loading the checkpoint
# torch.cuda.empty_cache()  # Clear CUDA cache before loading model
# model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH), strict=False)
# model.to(device)
# model = model.to(torch.bfloat16)
# model.eval()