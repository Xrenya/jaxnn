"""Vision Transformer (ViT) in Jax/Falx

A Jax implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision

Adapted from PyTorch/timm's resnet implementation.
Copyright of original work: 2019 Ross Wightman

Hacked together by / Copyright 2026, Rinat Shaymukhametov
"""

import jax
from flax import nnx
from flax.typing import Dtype, PromoteDtypeFn, PrecisionLike
from flax.nnx.nn import dtypes as flax_dtypes
import jax.numpy as jnp
from functools import partial

from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable, Literal

from jaxnn.layers import (
    to_ntuple,
    Activation,
    MaxPool2D,
    AvgPool2D,
    Identity,
    Attention,
    DiffAttention,
    # AttentionPoolLatent,
    # AttentionPoolPrr,
    PatchEmbed,
    Mlp,
    # SwiGLUPacked,
    # SwiGLU,
    # LayerNorm,
    # RmsNorm,
    DropPath,
)

from jaxnn.models._registry import register_model, generate_default_cfgs
from jaxnn.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from jaxnn.models._builder import build_model_with_cfg


LayerType = Union[str, Callable, Type[nnx.Module]]


__all__ = ['VisionTransformer']


ATTN_LAYERS = {
    "": Attention,
    "attn": Attention,
    'diff': DiffAttention,
}


class PatchEmbed(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, x: jax.Array):
        return x
    
class Block(nnx.Module):
    def __init__(self,):
        pass

    def __call__(self, x: jax.Array):
        return x
    

class VisionTransformer(nnx.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "avgmax", "max", "token", "map", "prr"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_toke: bool = True,
        pos_embed: str = "learn",
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        pool_include_prefix: bool = False,
        dynamic_img_size: bool = False,
        dunamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "reset", "jax", "jax_nlhb", "moco", ""] = "",
        fix_init: bool = False,
        embed_layer: Callable = PatchEmbed,
        embed_norm_layer: Optional[LayerType] = None,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nnx.Module] = Block,
        mlp_layer: Type[nnx.Module] = Mlp,
        attn_layer: LayerType = Attention,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        preferred_element_type: Optional[Dtype] = None,
        norm_dtype: Optional[Dtype] = jnp.float32,
        norm_param_dtype: Dtype = jnp.float32,
        norm_promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        *,
        rngs: nnx.Rngs,
    ):
        pass