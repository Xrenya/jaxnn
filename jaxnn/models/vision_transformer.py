"""Vision Transformer (ViT) in Jax/Flax

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

import math
from functools import partial
from typing import Callable, Literal, Optional, Tuple, Type, Union

import jax
from flax import nnx
from flax.typing import Dtype, PromoteDtypeFn, PrecisionLike
from flax.nnx.nn import dtypes as flax_dtypes
import jax.numpy as jnp
from flax import linen as nn
from flax.nnx import initializers
from functools import partial
import flax.linen.dtypes as flax_dtypes


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
    PatchDropout,
    LayerScale,
    DropPath,
    # SwiGLUPacked,
    # SwiGLU,
    # LayerNorm,
    # RmsNorm,
    calculate_drop_path_rates,
    get_norm_layer,
    get_act_layer,
    wrap_norm_layer
)

from jaxnn.models._registry import register_model, generate_default_cfgs
from jaxnn.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from jaxnn.models._builder import build_model_with_cfg


LayerType = Union[str, Callable, Type[nnx.Module]]

__all__ = ["VisionTransformer"]


ATTN_LAYERS = {
    "": Attention,
    "attn": Attention,
    "diff": DiffAttention,
}


class PatchEmbed(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, x: jax.Array):
        return x


class Block(nnx.Module):
    def __init__(
        self,
    ):
        pass

    def __call__(self, x: jax.Array):
        return x


class VisionTransformer(nnx.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal[
            "", "avg", "avgmax", "max", "token", "map", "prr"
        ] = "token",
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
        class_token: bool = True,
        pos_embed: str = "learn",
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        pool_include_prefix: bool = False,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "reset", "jax", "jax_nlhb", "moco", ""] = "",
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
        # Common conv kwargs for dtype/precision control
        common_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            norm_promote_dtype=norm_promote_dtype,
            preferred_element_type=preferred_element_type,
        )

        assert global_pool in ("", "avg", "avgmax", "max", "token", "map", "prr")
        assert class_token or global_pool != "token"
        assert pos_embed in ("", "none", "learn")
        use_fc_norm = (
            global_pool in ("avg", "avgmax", "max") if fc_norm is None else fc_norm
        )
        
        norm_layer = (
            wrap_norm_layer(
                get_norm_layer(norm_layer),
                dtype=norm_dtype,
                param_dtype=norm_param_dtype,
                promote_dtype=norm_promote_dtype,
            )(num_features=embed_dim, rngs=rngs)
            or 
            wrap_norm_layer(
                nnx.LayerNorm,
                dtype=norm_dtype,
                param_dtype=norm_param_dtype,
                promote_dtype=norm_promote_dtype,
            )(num_features=embed_dim, rngs=rngs, epsilon=1e-6)
        )
        embed_norm_layer = wrap_norm_layer(
                get_norm_layer(embed_norm_layer),
                dtype=norm_dtype,
                param_dtype=norm_param_dtype,
                promote_dtype=norm_promote_dtype,
            )(num_features=embed_dim, rngs=rngs)
        act_layer = get_act_layer(act_layer) or nnx.gelu

        # Config
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.no_ebed_class = no_embed_class
        self.pool_include_prefix = pool_include_prefix
        self.dynamic_img_size = dynamic_img_pad
        self.grad_checkpointing = False

        # Number of prefix tokens (cls + reg)
        self.num_prefix_tokens = (1 if class_token else 0) + reg_tokens
        self.has_class_token = reg_tokens

        # Patch embedding
        embed_args = {}
        if dynamic_img_size:
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        if embed_norm_layer is not None:
            embed_args["norm_layer"] = embed_norm_layer
        common_kwargs.update(embed_args)
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **common_kwargs,
            rngs=rngs,
        )
        num_patches = self.patch_embed.num_patches
        reduction = (
            self.patch_embed.feat_ration()
            if hasattr(self.patch_embed, "feat_ratio")
            else patch_size
        )

        # prefix tokens
        self.cls_token = (
            nnx.Param(jnp.zeros(1, 1, embed_dim), dtype=param_dtype)
            if class_token
            else None
        )
        self.reg_token = (
            nnx.Param(jnp.zeros(1, reg_tokens, embed_dim), dtype=param_dtype)
            if reg_tokens
            else None
        )

        # positional embedding
        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )

        if not pos_embed or pos_embed == "none":
            self.pos_embed = None
        else:
            self.pos_embed = nnx.Param(
                initializers.normal(stddev=0.02)(
                    rngs.params(), (1, embed_len, embed_dim), param_dtype
                )
            )
        self.pos_drop = nnx.Dropout(rate=pos_drop_rate, rngs=rngs)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                rate=patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = Identity()
        self.norm_pre = (
            wrap_norm_layer(
                norm_layer,
                dtype=norm_dtype,
                param_dtype=norm_param_dtype,
                promote_dtype=norm_promote_dtype,
            )(num_features=embed_dim, rngs=rngs)
            if pre_norm
            else Identity()
        )

        dpr = calculate_drop_path_rates(drop_path_rate, depth)
        self.blocks = nnx.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    scale_attn_norm=scale_attn_norm,
                    scale_mlp_norm=scale_mlp_norm,
                    proj_bias=proj_bias,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    attn_layer=attn_layer,
                    depth=i,
                    **common_kwargs,
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )


# Transoformer Block
class Block(nnx.Module):
    """Transformer block with pre-normalization."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Callable = nnx.gelu,
        norm_layer: Callable = partial(nnx.LayerNorm, epsilon=1e-6),
        mlp_layer: Type[nnx.Module] = Mlp,
        attn_layer: Type[nnx.Module] = Attention,
        depth: int = 0,
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
        """Initialize Block.

        Pre-Norm Residual Connections

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            qk_norm: If True, apply normalization to query and key.
            proj_bias: If True, add bias to output projection.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            init_values: Initial values for layer scale.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            mlp_layer: MLP layer.
            attn_layer: Attention layer type (class or string).
            depth: Block index, passed to attention layer for depth-dependent init.
        """
        self.init_values = init_values
        self.norm1 = wrap_norm_layer(
            norm_layer,
            dtype=norm_dtype,
            param_dtype=norm_param_dtype,
            promote_dtype=norm_promote_dtype,
        )(num_features=dim, rngs=rngs)
        self.attn = _create_attn(
            attn_layer=attn_layer,
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
            depth=depth,
            rngs=rngs,
        )
        self.ls1 = LayerScale(
            dim=dim,
            init_values=init_values,
            param_dtype=norm_param_dtype,
            dtype=norm_dtype,
            rngs=rngs
        )
        self.drop_path1 = DropPath(drop_path, rngs=rngs) if drop_path > 0. else Identity()
        
        self.norm2 = wrap_norm_layer(
            norm_layer,
            dtype=norm_dtype,
            param_dtype=norm_param_dtype,
            promote_dtype=norm_promote_dtype,
        )(num_features=dim, rngs=rngs)

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
            rngs=rngs,
        )

        self.ls2 = LayerScale(
            dim=dim,
            init_values=init_values,
            param_dtype=norm_param_dtype,
            dtype=norm_dtype,
            rngs=rngs
        ) if init_values else Identity()
        self.drop_path2 = DropPath(rate=drop_path, rngs=rngs) if drop_path > 0. else Identity()

    def __call__(
        self,
        x: jax.Array,
        attn_mask: Optional[jax.Array] = None,
        is_causal: bool = False,
    ) -> jax.Array:
        x = x + self.drop_path1(self.ls1(self.attn(
            self.norm1(x), attn_mask=attn_mask, is_causal=is_causal
        )))
        x = x = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class ResPostBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Callable = nnx.gelu,
        norm_layer: Callable = partial(nnx.LayerNorm, epsilon=1e-6),
        mlp_layer: Type[nnx.Module] = Mlp,
        attn_layer: Type[nnx.Module] = Attention,
        depth: int = 0,
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
        """Initialize Block.

        Support for adapter tuning
        Residual Post-Norm

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            qk_norm: If True, apply normalization to query and key.
            proj_bias: If True, add bias to output projection.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            init_values: Initial values for layer scale.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            mlp_layer: MLP layer.
            attn_layer: Attention layer type (class or string).
            depth: Block index, passed to attention layer for depth-dependent init.
        """
        self.init_values = init_values
        self.attn = _create_attn(
            attn_layer=attn_layer,
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
            depth=depth,
            rngs=rngs,
        )
        self.norm1 = wrap_norm_layer(
            norm_layer,
            dtype=norm_dtype,
            param_dtype=norm_param_dtype,
            promote_dtype=norm_promote_dtype,
        )(num_features=dim, rngs=rngs)
        self.drop_path1 = DropPath(drop_path, rngs=rngs) if drop_path > 0. else Identity()
        
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
            rngs=rngs,
        )
        self.norm2 = wrap_norm_layer(
            norm_layer,
            dtype=norm_dtype,
            param_dtype=norm_param_dtype,
            promote_dtype=norm_promote_dtype,
        )(num_features=dim, rngs=rngs)
        self.drop_path2 = DropPath(rate=drop_path, rngs=rngs) if drop_path > 0. else Identity()


    def __call__(
        self,
        x: jax.Array,
        attn_mask: Optional[jax.Array] = None,
        is_causal: bool = False,
    ) -> jax.Array:
        x = x + self.drop_path1(self.norm1(self.attn(
            x, attn_mask=attn_mask, is_causal=is_causal
        )))
        x = x = self.drop_path2(self.norm2(self.mlp(x)))
        return x


def _create_attn(
    attn_layer: LayerType,
    dim: int,
    num_heads: int,
    qkv_bias: bool = False,
    qk_norm: bool = False,
    scale_norm: bool = False,
    proj_bias: bool = True,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
    norm_layer: Optional[Type[nnx.Module]] = None,
    depth: int = 0,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
    param_dtype: Dtype = jnp.float32,
    norm_dtype: Optional[Dtype] = jnp.float32,
    norm_param_dtype: Dtype = jnp.float32,
    norm_promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
    preferred_element_type: Optional[Dtype] = None,
    *,
    rngs: nnx.Rngs,
    **kwargs,
) -> nnx.Module:
    
    if isinstance(attn_layer, str):
        attn_layer = ATTN_LAYERS.get(attn_layer, None)
        assert attn_layer is not None, f'Unknown attn_layer: {attn_layer}'

    # Only pass depth to attention layers that use it
    if issubclass(attn_layer, DiffAttention):
        kwargs['depth'] = depth

    conv_kwargs = dict(
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        promote_dtype=promote_dtype,
        preferred_element_type=preferred_element_type,
        norm_dtype=norm_dtype,
        norm_param_dtype=norm_param_dtype,
        norm_promote_dtype=norm_promote_dtype,
    )
    
    kwargs.update(conv_kwargs)
    return attn_layer(
        dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        qk_norm=qk_norm,
        scale_norm=scale_norm,
        proj_bias=proj_bias,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
        norm_layer=norm_layer,
        **kwargs,
    )