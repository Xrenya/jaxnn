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
import jax.numpy as jnp
from functools import partial

from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

from jaxnn.layers import (
    to_ntuple, Activation, MaxPool2D, AvgPool2D, Identity,
    Attention,
    # DiffAttention,
    # AttentionPoolLatent,
    # AttentionPoolPrr,
    # PatchEmbed,
    # Mlp,
    # SwiGLUPacked,
    # SwiGLU,
    # LayerNorm,
    # RmsNorm,
    # DropPath,
)

from jaxnn.models._registry import register_model, generate_default_cfgs
from jaxnn.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from jaxnn.models._builder import build_model_with_cfg

LayerType = Union[str, Callable, Type[nnx.Module]]


ATTN_LAYERS = {
    '': Attention,
    'attn': Attention,
    # 'diff': DiffAttention,
}
