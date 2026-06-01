"""Layers

Hacked together by / Copyright 2026 Rinat Shaymukhametov
"""

from typing import Optional, Tuple, Type, Union, Callable

import jax
from flax import nnx
import jax.numpy as jnp
from flax.typing import Dtype, PromoteDtypeFn, PrecisionLike
from flax.nnx.nn import dtypes as flax_dtypes

from .attention import maybe_add_mask
from .create_norm import get_norm_layer
from .helpers import wrap_norm_layer
from .identity import Identity
from .mlp import Mlp
LayerType = Union[str, Callable, Type[nnx.Module]]

__all__ = ["AttentionPoolLatent", "AttentionPoolPrr"]


class AttentionPoolLatent(nnx.Module):
    """ Attention pooling w/ latent query

    Setting out_features=0 disables the output projection, norm, and MLP layers (pre_logits mode).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        embed_dim: int = None,
        num_heads: int = 8,
        feat_size: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        latent_len: int = 1,
        latent_dim: int = None,
        pos_embed: str = '',
        pool_type: str = 'token',
        norm_layer: Optional[Type[nnx.Module]] = None,
        act_layer: Optional[Type[nnx.Module]] = nnx.gelu,
        drop: float = 0.0,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        preferred_element_type: Optional[Dtype] = None,
        norm_dtype: Optional[Dtype] = jnp.float32,
        norm_param_dtype: Dtype = jnp.float32,
        norm_promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        fused_attn: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.rngs = rngs
        dtype_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
        )
        
        embed_dim = embed_dim or in_features
        if out_features is None:
            out_features = in_features
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feat_size = feat_size
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.fused_attn = fused_attn
        self.dtype_kwargs = dtype_kwargs

        if pos_embed == 'abs':
            assert feat_size is not None
            # 1. calc std
            std = in_features ** -0.5
        
            # 2. lower/upper: -2*std и 2*std
            initializer = nnx.initializers.truncated_normal(
                stddev=std, 
                lower=-2.0 * std, 
                upper=2.0 * std
            )
            # 3. init with rngs
            shape = (feat_size, in_features)
            initial_value = initializer(rngs.params(), shape)
            self.pos_embed = nnx.Param(initial_value)
        else:
            self.pos_embed = None

        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len

        # 1. calc std
        std = self.latent_dim ** -0.5
    
        # 2. lower/upper: -2*std и 2*std
        initializer = nnx.initializers.truncated_normal(
            stddev=std, 
            lower=-2.0 * std, 
            upper=2.0 * std
        )
        # 3. init with rngs
        shape = (1, latent_len, embed_dim)
        initial_value = initializer(rngs.params(), shape)
        self.latent = nnx.Param(initial_value)
        
        self.q = nnx.Linear(embed_dim, embed_dim, use_bias=qkv_bias, **dtype_kwargs, rngs=rngs)
        self.kv = nnx.Linear(embed_dim, embed_dim * 2, use_bias=qkv_bias, **dtype_kwargs, rngs=rngs)
        if qk_norm:
            norm_layer = get_norm_layer(norm_layer)
            if norm_layer:
                self.q_norm = (
                    wrap_norm_layer(
                        norm_layer,
                        dtype=norm_dtype,
                        param_dtype=norm_param_dtype,
                        promote_dtype=norm_promote_dtype,
                    )(num_features=self.head_dim, rngs=rngs)
                )
                self.k_norm = (
                    wrap_norm_layer(
                        norm_layer,
                        dtype=norm_dtype,
                        param_dtype=norm_param_dtype,
                        promote_dtype=norm_promote_dtype,
                    )(num_features=self.head_dim, rngs=rngs)
                )
            else:
                self.q_norm = wrap_norm_layer(
                    nnx.LayerNorm,
                    dtype=norm_dtype,
                    param_dtype=norm_param_dtype,
                    promote_dtype=norm_promote_dtype,
                )(num_features=self.head_dim, rngs=rngs, epsilon=1e-6)
                self.k_norm = wrap_norm_layer(
                    nnx.LayerNorm,
                    dtype=norm_dtype,
                    param_dtype=norm_param_dtype,
                    promote_dtype=norm_promote_dtype,
                )(num_features=self.head_dim, rngs=rngs, epsilon=1e-6)
        else:
            self.q_norm = Identity()
            self.k_norm = Identity()

        if out_features > 0:
            self.proj = nnx.Linear(embed_dim, out_features, use_bias=qkv_bias, **dtype_kwargs, rngs=rngs)
            self.proj_drop = nnx.Dropout(rate=drop, rngs=rngs)
            norm = get_norm_layer(norm_layer)
            self.norm = (
                wrap_norm_layer(
                    norm,
                    dtype=norm_dtype,
                    param_dtype=norm_param_dtype,
                    promote_dtype=norm_promote_dtype,
                )(num_features=embed_dim, rngs=rngs)
                if norm
                else Identity()
            )
            self.mlp = Mlp(out_features, int(out_features * mlp_ratio), out_features=out_features, act_layer=act_layer, **dtype_kwargs, rngs=rngs)
        else:
            self.proj = Identity()
            self.proj_drop = nnx.Dropout(rate=drop, rngs=rngs)
            self.norm = Identity()
            self.mlp = None
            out_features = embed_dim

        self.out_features = out_features

    def __call__(
        self,
        x: jax.Array,
        attn_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
    ):
        B, N, C = x.shape

        if self.pos_embed is not None:
            x = x + self.pos_embed[jnp.newaxis, ...]

        q_latent = self.latent.repeat(B, axis=0)

        q = (
            self.q(q_latent)
            .reshape(B, self.latent_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .transpose(2, 0, 3, 1, 4)
        )
        k, v = kv

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            # JAX attention expects (B, tokens, heads, head_dim)
            x = nnx.dot_product_attention(
                query=q.transpose(0, 2, 1, 3),
                key=k.transpose(0, 2, 1, 3),
                value=v.transpose(0, 2, 1, 3),
                bias=None,
                mask=attn_mask,
                broadcast_dropout=True,
                dropout_rng=None,
                dropout_rate=0.0,
                deterministic=deterministic,
                module=None,
                is_causal=False,
                dtype=self.dtype_kwargs.get("dtype", None),
                precision=self.dtype_kwargs.get("precision", None),
                promote_dtype=self.dtype_kwargs.get(
                    "promote_dtype", flax_dtypes.promote_dtype
                ),
            )
            x = x.reshape(B, self.latent_len, self.embed_dim)
        else:
            
            q = q * self.scale
            attn = jnp.einsum("bhnd,bhmd->bhnm", q, k)  # JAX-style matmul
            attn = maybe_add_mask(attn, attn_mask)
            attn = jax.nn.softmax(attn, axis=-1)
            x = jnp.einsum("bhnm,bhmd->bhnd", attn, v)
            x = x.transpose(0, 2, 1, 3).reshape(B, self.latent_len, self.embed_dim)
        
        
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=deterministic)

        if self.mlp is not None:
            x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)
        return x


class AttentionPoolPrr(nnx.Module):
    """ Patch Representation Refinement (PRR) attention pool.

    From "Locality-Attending Vision Transformer" (ICLR 2026).

    Parameter-free multi-head self-attention that refines all patch representations
    before pooling. No Q/K/V projections — input is reshaped directly into multi-head
    format for self-attention.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        pool_type: str = 'token',
        pre_norm: bool = False,
        post_norm: bool = False,
        norm_layer: Optional[Type[nnx.Module]] = None,
        fused_attn: bool = True,
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
        dtype_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
        )
        self.dtype_kwargs = dtype_kwargs
        self.rngs = rngs 
        assert pool_type in ('token', 'avg'), f"pool_type must be 'token' or 'avg', got '{pool_type}'"
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        norm_layer = get_norm_layer(norm_layer)
        if norm_layer is None and (pre_norm or post_norm):
            norm_layer = nnx.LayerNorm

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.fused_attn = fused_attn
        self.out_features = dim

        self.pre_norm = (
            wrap_norm_layer(
                norm_layer,
                dtype=norm_dtype,
                param_dtype=norm_param_dtype,
                promote_dtype=norm_promote_dtype,
            )(num_features=dim, rngs=rngs)
            if pre_norm
            else Identity()
        )
        self.post_norm = (
            wrap_norm_layer(
                norm_layer,
                dtype=norm_dtype,
                param_dtype=norm_param_dtype,
                promote_dtype=norm_promote_dtype,
            )(num_features=dim, rngs=rngs)
            if post_norm
            else Identity()
        )

    def __call__(
        self,
        x: jax.Array,
        attn_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
    ) -> jax.Array:
        B, N, C = x.shape

        x = self.pre_norm(x)

        # Parameter-free self-attention: reshape into multi-head format
        qkv = x.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)  # (B, H, N, D)
        if self.fused_attn:
            # JAX attention expects (B, tokens, heads, head_dim)
            x = nnx.dot_product_attention(
                query=qkv.transpose(0, 2, 1, 3),
                key=qkv.transpose(0, 2, 1, 3),
                value=qkv.transpose(0, 2, 1, 3),
                bias=None,
                mask=attn_mask,
                broadcast_dropout=True,
                dropout_rng=None,
                dropout_rate=0.0,
                deterministic=deterministic,
                module=None,
                is_causal=False,
                dtype=self.dtype_kwargs.get("dtype", None),
                precision=self.dtype_kwargs.get("precision", None),
                promote_dtype=self.dtype_kwargs.get(
                    "promote_dtype", flax_dtypes.promote_dtype
                ),
            )
            x = x.reshape(B, N, C)
        else:
            attn = (qkv * self.scale) @ qkv.transpose(0, 1, 3, 2)
            attn = maybe_add_mask(attn, attn_mask)
            attn = jax.nn.softmax(attn, axis=-1)
            x = jnp.einsum("bhnm,bhmd->bhnd", attn, qkv)
            x = x.transpose(0, 2, 1, 3).reshape(B, N, C)

        x = self.post_norm(x)

        # Pool
        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)

        return x
