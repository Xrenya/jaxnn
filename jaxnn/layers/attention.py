from typing import Final, Optional, Type

import jax
from flax import nnx
import jax.numpy as jnp

from .identity import Identity

__all__ = ['Attention'] #, 'AttentionRope', 'maybe_add_mask', 'resolve_self_attn_mask']


# Make option to select nnx.MultiHeadAttention
# layer = nnx.MultiHeadAttention(num_heads=8, in_features=5, qkv_features=16,
#                                decode=False, rngs=nnx.Rngs(0)) 
class Attention(nnx.Module):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_head_dim: Optional[int] = None,
        dim_out: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: Optional[Type[nnx.Module]] = None,
        precision: str = "float32",
        dtype=None,
        *,
        rngs: nnx.Rngs
    ):
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings.
            num_heads: Number of attention heads.
            attn_head_dim: Dimension of each attention head. If None, computed as dim // num_heads.
            dim_out: Output dimension. If None, same as dim.
            qkv_bias: Whether to use bias in the query, key, value projections.
            qk_norm: Whether to apply normalization to query and key vectors.
            scale_norm: Whether to apply normalization to attention output before projection.
            proj_bias: Whether to use bias in the output projection.
            attn_drop: Dropout rate applied to the attention weights.
            proj_drop: Dropout rate applied after the output projection.
            norm_layer: Normalization layer constructor for QK normalization if enabled.
            precision: jax.lax.Precision is an enumeration used to control
                       the tradeoff between computational speed and numerical
                       accuracy for matrix multiplications
                       (`float32`, `tensorfloat32`, `bfloat16`)
                       for ref: https://kolonist26-jax-kr.readthedocs.io/en/latest/jax.lax.html
        """
        dim_out = dim_out or dim
        head_dim = attn_head_dim
        if head_dim is None:
            assert dim % num_heads == 0, "dim should be divisible by num_heads"
            head_dim = dim // num_heads
        if qk_norm or scale_norm:
            assert norm_layer is not None, "norm_layer must be provided if qk_norm or scale_norm is True"

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim

        self.qkv = nnx.Linear(
            in_features=dim,
            out_features=self.attn_dim * 3,
            use_bias=qkv_bias,
            dtype=dtype,
            rngs=rngs,
        )
        self.q_norm = norm_layer(head_dim) if qk_norm else Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else Identity()
        self.attn_drop = nnx.Dropout(attn_drop, rngs=rngs)
        self.norm = norm_layer(self.attn_dim) if scale_norm else Identity()
        self.proj = nnx.Linear(
            in_features=self.attn_dim,
            out_features=dim_out,
            use_bias=proj_bias,
            dtype=dtype,
            rngs=rngs,
        )
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)

        # jax.lax.Precision accepts 'default', 'high', or 'highest' or its aliases
        if isinstance(precision, str):
            self.precision = jax.lax.Precision(precision)
        else:
            self.precision = precision

        self.rngs = rngs

    def __call__(
        self,
        x: jax.Array,
        attn_mask: jax.Array | None = None,
        is_causal: bool = False,
        deterministic: bool = False,
    ):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .transpose(2, 0, 3, 1, 4)
        )
        q, k, v = qkv  # each: (B, num_heads, N, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        # fused attention; scale is handled internally
        x = nnx.dot_product_attention(
            query=q,
            key=k,
            value=v,
            bias=None,
            mask=attn_mask,
            broadcast_dropout=True,
            dropout_rng=self.rngs.dropout() if not deterministic else None,
            dropout_rate=self.attn_drop.rate,
            deterministic=deterministic,
            dtype=x.dtype,
            precision=self.precision,
            module=None,
            is_causal=is_causal,
        )
        # (B, num_heads, N, head_dim) -> (B, N, attn_dim)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, self.attn_dim)

        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=deterministic)
        return x
        