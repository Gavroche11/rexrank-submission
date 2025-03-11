"""
Use EVA-X series as your backbone. You could get 
EVA-X representations simply with timm. Try them 
with your own X-ray tasks. 
Enjoy!

Reference:
    https://github.com/baaivision/EVA
    https://github.com/huggingface/pytorch-image-models
Thanks for their work!
    
by Jingfeng Yao 
from HUST-VL
"""

import torch
import torch.nn as nn
from copy import deepcopy
from timm.models.eva import Eva

from pathlib import Path
from typing import Tuple, Union
from dataclasses import dataclass
from llava.utils import rank0_print
from timm.layers import resample_abs_pos_embed, resample_patch_embed

from .eva_clip.eva_clip_processors import EvaXImageTrainProcessor
from .eva_clip.factory import _rescan_model_configs, get_model_config

def checkpoint_filter_fn(
        state_dict,
        model,
        interpolation='bicubic',
        antialias=True,
):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    state_dict = state_dict.get('model_ema', state_dict)
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    # prefix for loading OpenCLIP compatible weights
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    elif 'visual.pos_embed' in state_dict:
        prefix = 'visual.'
    else:
        prefix = ''
    mim_weights = prefix + 'mask_token' in state_dict
    no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue

        if 'rope' in k:
            # fixed embedding no need to load buffer from checkpoint
            continue

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )

        k = k.replace('mlp.ffn_ln', 'mlp.norm')
        k = k.replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1')
        k = k.replace('mlp.w1', 'mlp.fc1_g')
        k = k.replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if no_qkv:
            k = k.replace('q_bias', 'q_proj.bias')
            k = k.replace('v_bias', 'v_proj.bias')

        if mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
            if k == 'norm.weight' or k == 'norm.bias':
                # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                k = k.replace('norm', 'fc_norm')
            else:
                # skip pretrain mask token & head weights
                continue

        out_dict[k] = v

    return out_dict


class EVA_X(Eva):
    def __init__(self, **kwargs):
        super(EVA_X, self).__init__(**kwargs)
        self.grad_checkpointing = False

    def forward_features(self, x, return_all_features=False):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.blocks:
            if self.grad_checkpointing:
                x = checkpoint(blk, x, (rot_pos_embed, ))
            else:
                x = blk(x, rope=rot_pos_embed)

        if not return_all_features:
            x = self.norm(x)
            if self.fc_norm is not None:
                return self.fc_norm(x.mean(1))
            else:
                return x[:, 0]
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, return_all_features):
        if return_all_features:
            return self.forward_features(x, return_all_features)
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}
    
    def get_num_layers(self):
        return len(self.blocks)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

    model.apply(_convert_weights_to_fp16)


@dataclass
class EVAXVisionCfg:
    precision: str = "fp16"
    layers: Union[Tuple[int, int, int, int], int] = 12
    image_size: Union[Tuple[int, int], int] = 224
    embed_dim: int = 768
    mlp_ratio: float = 4 * 2 / 3
    patch_size: int = 16
    num_heads: int = 12
    eva_model_name: "str" = "eva-x-base"
    qkv_fused: bool = False
    swiglu_mlp: bool = True
    scale_mlp: bool = True
    use_rot_pos_emb: bool = True
    ref_feat_shape=(14, 14)  # 224 / 16
    width: int = 2048  # hidden_size
    num_classes: int = 1000


def _build_vision_tower(vision_tower_path: str, vision_cfg: EVAXVisionCfg, **kwargs):
    vision_cfg = EVAXVisionCfg(**vision_cfg)

    if vision_cfg.eva_model_name.startswith("eva-x-base"):
        visual = EVA_X(
            img_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            embed_dim=vision_cfg.embed_dim,
            depth=vision_cfg.layers,
            num_heads=vision_cfg.num_heads,
            qkv_fused=vision_cfg.qkv_fused,
            mlp_ratio=vision_cfg.mlp_ratio,
            swiglu_mlp=vision_cfg.swiglu_mlp,
            scale_mlp=vision_cfg.scale_mlp,
            use_rot_pos_emb=vision_cfg.use_rot_pos_emb,
            ref_feat_shape=vision_cfg.ref_feat_shape,
            num_classes=vision_cfg.num_classes,
        )

        eva_ckpt = checkpoint_filter_fn(
            torch.load(
                vision_tower_path,
                map_location='cpu'
            ),
            visual
        )

        # if vision_cfg.precision == "fp16":
            # convert_weights_to_fp16(visual)

        new_eva_ckpt = {}

        for k, v in eva_ckpt.items():
            if k.startswith('module.'):
                new_eva_ckpt[k[7:]] = v
            else:
                new_eva_ckpt[k] = v

        incompatible_keys = visual.load_state_dict(new_eva_ckpt, strict=False)
        rank0_print("EVA-X incompatible_keys:", incompatible_keys)
    elif vision_cfg.eva_model_name == "eva-x-small-480":
        visual = EVA_X(
            img_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            embed_dim=vision_cfg.embed_dim,
            depth=vision_cfg.layers,
            num_heads=vision_cfg.num_heads,
            qkv_fused=vision_cfg.qkv_fused,
            mlp_ratio=vision_cfg.mlp_ratio,
            swiglu_mlp=vision_cfg.swiglu_mlp,
            scale_mlp=vision_cfg.scale_mlp,
            use_rot_pos_emb=vision_cfg.use_rot_pos_emb,
            ref_feat_shape=vision_cfg.ref_feat_shape,
            num_classes=vision_cfg.num_classes,
        )

        eva_ckpt = checkpoint_filter_fn(
            torch.load(
                vision_tower_path,
                map_location='cpu'
            ),
            visual
        )

        # if vision_cfg.precision == "fp16":
            # convert_weights_to_fp16(visual)

        incompatible_keys = visual.load_state_dict(eva_ckpt, strict=False)
        rank0_print("EVA-X incompatible_keys:", incompatible_keys)
    else:
        raise NotImplementedError("You can only use EVA-X Base model now. If you want to use EVA-X Base, you should use 'eva-x-base'.")

    return visual


class EVAXEncoderWrapper(nn.Module):
    def __init__(self, vision_tower_pretrained, config):
        super(EVAXEncoderWrapper, self).__init__()
        self.config = config
        self.config["vision_tower_path"] = vision_tower_pretrained
        self.model = _build_vision_tower(**self.config)

    def forward(self, image, **kwargs):
        encode = self.model(image, return_all_features=True)
        return encode

    @property
    def dtype(self):
        return list(self.parameters())[-1].dtype

    @property
    def device(self):
        return list(self.parameters())[-1].device


class EvaXVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.vision_tower_pretrained = args.vision_tower_pretrained

        _rescan_model_configs()
        self.config = get_model_config(vision_tower)

        if not delay_load:
            rank0_print(f"Loading EVA-X: {self.vision_tower_name}")
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
        self.image_processor = EvaXImageTrainProcessor(self.config["vision_cfg"]["image_size"])
        self.vision_tower = EVAXEncoderWrapper(self.vision_tower_pretrained, self.config)

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
        # return 2048

    # @property
    # def num_patches(self):
    #     return (self.config["vision_cfg"]["image_size"] // self.config["vision_cfg"]["patch_size"]) ** 2

    # @property
    # def num_patches_per_side(self):
    #     return self.config["vision_cfg"]["image_size"] // self.config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config["vision_cfg"]["image_size"]
