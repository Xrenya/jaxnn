"""Vision Transformer (ViT) in JAX/Flax.

A JAX implementation of Vision Transformers as described in:

"An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale"
    https://arxiv.org/abs/2010.11929

"How to Train Your ViT? Data, Augmentation, and Regularization in Vision Transformers"
    https://arxiv.org/abs/2106.10270

"FlexiViT: One Model for All Patch Sizes"
    https://arxiv.org/abs/2212.08013

The official JAX code is released and available at:
    https://github.com/google-research/vision_transformer
    https://github.com/google-research/big_vision

Adapted from PyTorch/timm's Vision Transformer implementation.
Copyright of original work: 2019 Ross Wightman

Copyright 2026 Rinat Shaymukhametov
"""

from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Set, Tuple, Type, Union

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers
from flax.nnx.nn import dtypes as flax_dtypes
from flax.typing import Dtype, PrecisionLike, PromoteDtypeFn

from jaxnn.layers import (
    Attention,
    AttentionPoolLatent,
    AttentionPoolPrr,
    DiffAttention,
    DropPath,
    Identity,
    LayerScale,
    Mlp,
    PatchDropout,
    PatchEmbed,
    calculate_drop_path_rates,
    get_act_layer,
    get_norm_layer,
    wrap_norm_layer,
)
from jaxnn.models._builder import build_model_with_cfg
from jaxnn.models._registry import generate_default_cfgs, register_model


LayerType = Union[str, Callable, Type[nnx.Module]]

__all__ = ["VisionTransformer", "Block", "ResPostBlock"]


ATTN_LAYERS = {
    "": Attention,
    "attn": Attention,
    "diff": DiffAttention,
}


def _cfg(url: str = "", **kwargs) -> Dict[str, Any]:
    return {
        "url": url,
        "hf_hub_id": "JaxNN/",
        "input_size": (224, 224, 3),
        "fixed_input_size": True,
        "interpolation": "bicubic",
        "crop_pct": 0.9,
        "crop_mode": "center",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "num_classes": 1000,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


def _vit_pretrained_cfg(name: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if name.endswith(".dino") or name.endswith(".mae"):
        kwargs.update(
            num_classes=0,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    elif name.endswith(".orig_in21k"):
        kwargs["num_classes"] = 0
    elif name.endswith(".augreg_in21k"):
        kwargs["num_classes"] = 21843
    return _cfg(url=f"https://huggingface.co/JaxNN/{name}", **kwargs)


_VIT_PRETRAINED = (
    "vit_base_patch16_224.orig_in21k_ft_in1k",
    "vit_base_patch16_224.augreg_in1k",
    "vit_base_patch32_224.sam_in1k",
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit_small_patch16_224.augreg_in1k",
    "vit_base_patch32_224.augreg_in21k",
    "vit_base_patch16_224.dino",
    "vit_large_patch32_224.orig_in21k",
    "vit_base_patch32_224.orig_in21k",
    "vit_base_patch16_224.augreg_in21k",
    "vit_large_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "vit_tiny_patch16_224.augreg_in21k",
    "vit_base_patch16_224.mae",
    "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_small_patch16_224.augreg_in21k",
    "vit_large_patch16_224.orig_in21k",
    "vit_base_patch16_224.orig_in21k",
    "vit_base_patch32_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224.sam_in1k",
    "vit_base_patch32_224.augreg_in1k",
    "vit_small_patch16_224.dino",
    "vit_large_patch16_224.mae",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "vit_large_patch16_224.augreg_in21k",
)


default_cfgs = generate_default_cfgs(
    {
        name: _vit_pretrained_cfg(name)
        for name in _VIT_PRETRAINED
    }
)


def _resolve_norm_layer(norm_layer: Optional[LayerType]) -> Callable:
    norm_layer = get_norm_layer(norm_layer)
    return norm_layer or partial(nnx.LayerNorm, epsilon=1e-6)


def _make_norm(
    norm_layer: Callable,
    num_features: int,
    *,
    dtype: Optional[Dtype],
    param_dtype: Dtype,
    promote_dtype: PromoteDtypeFn,
    rngs: nnx.Rngs,
) -> nnx.Module:
    return wrap_norm_layer(
        norm_layer,
        dtype=dtype,
        param_dtype=param_dtype,
        promote_dtype=promote_dtype,
    )(num_features=num_features, rngs=rngs)


def _to_broadcast_token(token: nnx.Param, batch_size: int) -> jax.Array:
    return jnp.broadcast_to(token.value, (batch_size,) + token.value.shape[1:])


def _drop_path(module: nnx.Module, x: jax.Array, deterministic: bool) -> jax.Array:
    if isinstance(module, DropPath):
        return module(x, deterministic=deterministic)
    return module(x)


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
        attn_layer: LayerType = Attention,
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
        self.norm1 = _make_norm(
            norm_layer,
            dim,
            dtype=norm_dtype,
            param_dtype=norm_param_dtype,
            promote_dtype=norm_promote_dtype,
            rngs=rngs,
        )
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
            norm_dtype=norm_dtype,
            norm_param_dtype=norm_param_dtype,
            norm_promote_dtype=norm_promote_dtype,
            depth=depth,
            rngs=rngs,
        )
        self.ls1 = (
            LayerScale(
                dim=dim,
                init_values=init_values,
                param_dtype=norm_param_dtype,
                dtype=norm_dtype,
                rngs=rngs,
            )
            if init_values
            else Identity()
        )
        self.drop_path1 = (
            DropPath(rate=drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        )

        self.norm2 = _make_norm(
            norm_layer,
            dim,
            dtype=norm_dtype,
            param_dtype=norm_param_dtype,
            promote_dtype=norm_promote_dtype,
            rngs=rngs,
        )
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
            rngs=rngs,
        )
        self.ls2 = (
            LayerScale(
                dim=dim,
                init_values=init_values,
                param_dtype=norm_param_dtype,
                dtype=norm_dtype,
                rngs=rngs,
            )
            if init_values
            else Identity()
        )
        self.drop_path2 = (
            DropPath(rate=drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        )

    def __call__(
        self,
        x: jax.Array,
        attn_mask: Optional[jax.Array] = None,
        is_causal: bool = False,
        deterministic: bool = False,
    ) -> jax.Array:
        x = x + _drop_path(
            self.drop_path1,
            self.ls1(
                self.attn(
                    self.norm1(x),
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                    deterministic=deterministic,
                )
            ),
            deterministic=deterministic,
        )
        x = x + _drop_path(
            self.drop_path2,
            self.ls2(self.mlp(self.norm2(x))),
            deterministic=deterministic,
        )
        return x


class ResPostBlock(nnx.Module):
    """Transformer block with residual post-normalization."""

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
        attn_layer: LayerType = Attention,
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
            norm_dtype=norm_dtype,
            norm_param_dtype=norm_param_dtype,
            norm_promote_dtype=norm_promote_dtype,
            depth=depth,
            rngs=rngs,
        )
        self.norm1 = _make_norm(
            norm_layer,
            dim,
            dtype=norm_dtype,
            param_dtype=norm_param_dtype,
            promote_dtype=norm_promote_dtype,
            rngs=rngs,
        )
        self.drop_path1 = (
            DropPath(rate=drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        )
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
            rngs=rngs,
        )
        self.norm2 = _make_norm(
            norm_layer,
            dim,
            dtype=norm_dtype,
            param_dtype=norm_param_dtype,
            promote_dtype=norm_promote_dtype,
            rngs=rngs,
        )
        self.drop_path2 = (
            DropPath(rate=drop_path, rngs=rngs) if drop_path > 0.0 else Identity()
        )

    def __call__(
        self,
        x: jax.Array,
        attn_mask: Optional[jax.Array] = None,
        is_causal: bool = False,
        deterministic: bool = False,
    ) -> jax.Array:
        x = x + _drop_path(
            self.drop_path1,
            self.norm1(
                self.attn(
                    x,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                    deterministic=deterministic,
                )
            ),
            deterministic=deterministic,
        )
        x = x + _drop_path(
            self.drop_path2,
            self.norm2(self.mlp(x)),
            deterministic=deterministic,
        )
        return x


class VisionTransformer(nnx.Module):
    """Vision Transformer following timm's model structure."""

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
        assert global_pool in ("", "avg", "avgmax", "max", "token", "map", "prr")
        assert class_token or global_pool != "token"
        assert pos_embed in ("", "none", "learn")

        del weight_init
        self.rngs = rngs
        self.dtype_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
        )
        self.norm_kwargs = dict(
            dtype=norm_dtype,
            param_dtype=norm_param_dtype,
            promote_dtype=norm_promote_dtype,
        )

        norm_layer = _resolve_norm_layer(norm_layer)
        embed_norm_cls = get_norm_layer(embed_norm_layer)
        if embed_norm_cls:
            def embed_norm_factory(num_features: int, *, rngs: nnx.Rngs) -> nnx.Module:
                return _make_norm(
                    embed_norm_cls,
                    num_features,
                    dtype=norm_dtype,
                    param_dtype=norm_param_dtype,
                    promote_dtype=norm_promote_dtype,
                    rngs=rngs,
                )
        else:
            embed_norm_factory = None
        act_layer = get_act_layer(act_layer) or nnx.gelu
        use_fc_norm = (
            global_pool in ("avg", "avgmax", "max") if fc_norm is None else fc_norm
        )

        self.num_classes = num_classes
        self.in_chans = in_chans
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.no_embed_class = no_embed_class
        self.pool_include_prefix = pool_include_prefix
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.num_prefix_tokens = (1 if class_token else 0) + reg_tokens
        self.has_class_token = class_token

        embed_args = {}
        if dynamic_img_size:
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        if embed_norm_factory is not None:
            embed_args["norm_layer"] = embed_norm_factory

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
            **self.dtype_kwargs,
            rngs=rngs,
        )
        num_patches = self.patch_embed.num_patches
        reduction = (
            self.patch_embed.feat_ratio()
            if hasattr(self.patch_embed, "feat_ratio")
            else patch_size
        )

        self.cls_token = (
            nnx.Param(jnp.zeros((1, 1, embed_dim), dtype=param_dtype))
            if class_token
            else None
        )
        self.reg_token = (
            nnx.Param(jnp.zeros((1, reg_tokens, embed_dim), dtype=param_dtype))
            if reg_tokens
            else None
        )

        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = (
            nnx.Param(
                initializers.normal(stddev=0.02)(
                    rngs.params(), (1, embed_len, embed_dim), param_dtype
                )
            )
            if pos_embed and pos_embed != "none"
            else None
        )
        self.pos_drop = nnx.Dropout(rate=pos_drop_rate, rngs=rngs)
        self.patch_drop = (
            PatchDropout(
                rate=patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
                rngs=rngs,
            )
            if patch_drop_rate > 0.0
            else Identity()
        )
        self.norm_pre = (
            _make_norm(norm_layer, embed_dim, **self.norm_kwargs, rngs=rngs)
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
                    drop_path=float(dpr[i]),
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    attn_layer=attn_layer,
                    depth=i,
                    **self.dtype_kwargs,
                    norm_dtype=norm_dtype,
                    norm_param_dtype=norm_param_dtype,
                    norm_promote_dtype=norm_promote_dtype,
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )
        self.feature_info = [
            dict(module=f"blocks.{i}", num_chs=embed_dim, reduction=reduction)
            for i in range(depth)
        ]

        self.norm = (
            _make_norm(norm_layer, embed_dim, **self.norm_kwargs, rngs=rngs)
            if final_norm and not use_fc_norm
            else Identity()
        )

        if global_pool == "map":
            self.attn_pool = AttentionPoolLatent(
                in_features=embed_dim,
                out_features=embed_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                **self.dtype_kwargs,
                norm_dtype=norm_dtype,
                norm_param_dtype=norm_param_dtype,
                norm_promote_dtype=norm_promote_dtype,
                rngs=rngs,
            )
        elif global_pool == "prr":
            self.attn_pool = AttentionPoolPrr(
                dim=embed_dim,
                num_heads=num_heads,
                pool_type="token" if class_token else "avg",
                norm_layer=norm_layer,
                **self.dtype_kwargs,
                norm_dtype=norm_dtype,
                norm_param_dtype=norm_param_dtype,
                norm_promote_dtype=norm_promote_dtype,
                rngs=rngs,
            )
            self.pool_include_prefix = True
        else:
            self.attn_pool = None

        self.fc_norm = (
            _make_norm(norm_layer, embed_dim, **self.norm_kwargs, rngs=rngs)
            if use_fc_norm
            else Identity()
        )
        self.head_drop = nnx.Dropout(rate=drop_rate, rngs=rngs)
        self.head = (
            nnx.Linear(
                in_features=embed_dim,
                out_features=num_classes,
                **self.dtype_kwargs,
                rngs=rngs,
            )
            if num_classes > 0
            else Identity()
        )

    def no_weight_decay(self) -> Set[str]:
        return {"pos_embed", "cls_token", "reg_token"}

    def get_classifier(self) -> nnx.Module:
        return self.head

    def reset_classifier(
        self,
        num_classes: int,
        global_pool: Optional[str] = None,
        *,
        rngs: Optional[nnx.Rngs] = None,
    ) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "avgmax", "max", "token", "map", "prr")
            if global_pool in ("map", "prr") and self.attn_pool is None:
                raise ValueError("Cannot add attention pooling in reset_classifier().")
            if global_pool not in ("map", "prr") and self.attn_pool is not None:
                self.attn_pool = None
            elif global_pool in ("map", "prr") and self.global_pool != global_pool:
                raise ValueError("Cannot change attention pooling type in reset_classifier().")
            self.global_pool = global_pool

        rngs = rngs or self.rngs
        self.head = (
            nnx.Linear(
                in_features=self.embed_dim,
                out_features=num_classes,
                **self.dtype_kwargs,
                rngs=rngs,
            )
            if num_classes > 0
            else Identity()
        )

    def _pos_embed(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        B = x.shape[0]
        pos_embed = self.pos_embed.value if self.pos_embed is not None else None

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(_to_broadcast_token(self.cls_token, B))
        if self.reg_token is not None:
            to_cat.append(_to_broadcast_token(self.reg_token, B))

        if self.no_embed_class:
            if pos_embed is not None:
                x = x + pos_embed
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=1)
        else:
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=1)
            if pos_embed is not None:
                x = x + pos_embed
        return self.pos_drop(x, deterministic=deterministic)

    def forward_intermediates(
        self,
        x: jax.Array,
        indices: Optional[Tuple[int, ...]] = None,
        *,
        deterministic: bool = False,
    ):
        x = self.patch_embed(x)
        x = self._pos_embed(x, deterministic=deterministic)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        intermediates = []
        take_all = indices is None
        indices = set(indices or ())
        for i, block in enumerate(self.blocks.layers):
            x = block(x, deterministic=deterministic)
            if take_all or i in indices:
                intermediates.append(x)
        x = self.norm(x)
        return x, intermediates

    def forward_features(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        x = self.patch_embed(x)
        x = self._pos_embed(x, deterministic=deterministic)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for block in self.blocks.layers:
            x = block(x, deterministic=deterministic)
        x = self.norm(x)
        return x

    def pool(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        if self.attn_pool is not None:
            return self.attn_pool(x, deterministic=deterministic)

        if self.global_pool == "token":
            return x[:, 0]
        if self.global_pool == "":
            return x

        start = 0 if self.pool_include_prefix else self.num_prefix_tokens
        x = x[:, start:]
        if self.global_pool == "avg":
            return x.mean(axis=1)
        if self.global_pool == "max":
            return x.max(axis=1)
        if self.global_pool == "avgmax":
            return 0.5 * (x.mean(axis=1) + x.max(axis=1))
        raise ValueError(f"Unsupported global_pool: {self.global_pool}")

    def forward_head(
        self,
        x: jax.Array,
        *,
        pre_logits: bool = False,
        deterministic: bool = False,
    ) -> jax.Array:
        x = self.pool(x, deterministic=deterministic)
        x = self.fc_norm(x)
        x = self.head_drop(x, deterministic=deterministic)
        return x if pre_logits else self.head(x)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        x = self.forward_features(x, deterministic=deterministic)
        return self.forward_head(x, deterministic=deterministic)


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
    norm_layer: Optional[Callable] = None,
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
        attn_layer = ATTN_LAYERS.get(attn_layer)
        assert attn_layer is not None, f"Unknown attn_layer: {attn_layer}"

    if issubclass(attn_layer, DiffAttention):
        kwargs["depth"] = depth

    kwargs.update(
        dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
            norm_dtype=norm_dtype,
            norm_param_dtype=norm_param_dtype,
            norm_promote_dtype=norm_promote_dtype,
        )
    )
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
        rngs=rngs,
    )


def _create_vision_transformer(
    variant: str,
    pretrained: bool = False,
    **kwargs,
) -> VisionTransformer:
    return build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_strict=False,
        **kwargs,
    )


@register_model
def vit_tiny_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    return _create_vision_transformer(
        "vit_tiny_patch16_224",
        pretrained,
        **dict(model_args, **kwargs),
    )


@register_model
def vit_small_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    return _create_vision_transformer(
        "vit_small_patch16_224",
        pretrained,
        **dict(model_args, **kwargs),
    )


@register_model
def vit_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    return _create_vision_transformer(
        "vit_base_patch16_224",
        pretrained,
        **dict(model_args, **kwargs),
    )


@register_model
def vit_base_patch32_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
    return _create_vision_transformer(
        "vit_base_patch32_224",
        pretrained,
        **dict(model_args, **kwargs),
    )


@register_model
def vit_large_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    return _create_vision_transformer(
        "vit_large_patch16_224",
        pretrained,
        **dict(model_args, **kwargs),
    )


@register_model
def vit_large_patch32_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16)
    return _create_vision_transformer(
        "vit_large_patch32_224",
        pretrained,
        **dict(model_args, **kwargs),
    )
