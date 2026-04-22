from typing import Final, Optional, Type
import logging

import jax
from flax import nnx
import jax.numpy as jnp
from flax.typing import Dtype, PromoteDtypeFn, PrecisionLike
from flax.nnx.nn import dtypes as flax_dtypes

from .identity import Identity
from .pos_embed_sincon import apply_rot_embed_cat

_logger = logging.getLogger(__name__)

__all__ = ["Attention", "AttentionRope", "maybe_add_mask", "resolve_self_attn_mask"]


def maybe_add_mask(score: jax.Array, attn_mask: Optional[jax.Array] = None):
    return score if attn_mask is None else score + attn_mask


def resolve_self_attn_mask(
    seq_len: int,
    attn: jax.Array,
    attn_mask: Optional[jax.Array] = None,
    is_causal: bool = False,
) -> Optional[jax.Array]:
    """Build additive bias matching dot_product_attention semantics for self-attention.
    is_causal and attn_mask are mutually exclusive (is_causal takes precedence).
    """
    if is_causal:
        # upper-triangular filled with -inf, diagonal = 0 (same as triu_(1))
        attn_bias = jnp.full((seq_len, seq_len), float("-inf"))
        attn_bias = jnp.triu(attn_bias, k=1)
    elif attn_mask is None:
        attn_bias = None
    elif attn_mask.dtype == jnp.bool_:
        # bool mask: True = keep, False = mask out (same as masked_fill_(~mask, -inf))
        attn_bias = jnp.where(
            attn_mask, jnp.zeros_like(attn_mask, dtype=attn.dtype), float("-inf")
        )
    else:
        attn_bias = attn_mask.astype(attn.dtype)
    return attn_bias


def attention_to_mha(attn: "Attention") -> nnx.MultiHeadAttention:
    """Convert a custom Attention module to nnx.MultiHeadAttention.

    Maps the fused QKV weight [dim, attn_dim*3] back to separate Q/K/V
    projections expected by nnx.MultiHeadAttention.
    """
    dim = attn.qkv.in_features
    attn_dim = attn.attn_dim
    num_heads = attn.num_heads
    head_dim = attn.head_dim

    mha = nnx.MultiHeadAttention(
        num_heads=num_heads,
        in_features=dim,
        qkv_features=attn_dim,
        out_features=attn.proj.out_features,
        dropout_rate=attn.attn_drop.rate,
        deterministic=False,
        dtype=attn.qkv.dtype,
        param_dtype=attn.qkv.kernel.value.dtype,
        precision=attn.dtype_kwargs.get("precision", None),
        kernel_init=nnx.initializers.zeros,
        rngs=attn.rngs,
    )

    w_qkv = attn.qkv.kernel[...]
    wq, wk, wv = jnp.split(w_qkv, 3, axis=-1)

    mha.query.kernel[...] = wq.reshape(dim, num_heads, head_dim)
    mha.key.kernel[...] = wk.reshape(dim, num_heads, head_dim)
    mha.value.kernel[...] = wv.reshape(dim, num_heads, head_dim)

    if attn.qkv.bias is not None:
        b_qkv = attn.qkv.bias[...]
        bq, bk, bv = jnp.split(b_qkv, 3, axis=0)
        mha.query.bias[...] = bq.reshape(num_heads, head_dim)
        mha.key.bias[...] = bk.reshape(num_heads, head_dim)
        mha.value.bias[...] = bv.reshape(num_heads, head_dim)

    mha.out.kernel[...] = attn.proj.kernel[...].reshape(num_heads, head_dim, -1)
    if attn.proj.bias is not None:
        mha.out.bias[...] = attn.proj.bias[...]

    return mha


def mha_to_attention(
    mha: nnx.MultiHeadAttention,
    qk_norm: bool = False,
    scale_norm: bool = False,
    norm_layer: Optional[Type[nnx.Module]] = None,
    **kwargs,
) -> "Attention":
    """Create an Attention module from nnx.MultiHeadAttention.

    Fuses the separate Q/K/V projections into a single QKV linear layer.
    Pass qk_norm/scale_norm and a norm_layer to add those features on top.
    """
    # nnx stores query kernel as (in_features, num_heads, head_dim)
    in_features, num_heads, head_dim = mha.query.kernel.value[...].shape
    out_features = mha.out.kernel.value[...].shape[-1]  # (num_heads, head_dim, out)
    has_bias = mha.query.bias is not None

    attn = Attention(
        dim=in_features,
        num_heads=num_heads,
        dim_out=out_features,
        qkv_bias=has_bias,
        qk_norm=qk_norm,
        scale_norm=scale_norm,
        proj_bias=(mha.out.bias is not None),
        norm_layer=norm_layer,
        rngs=mha.rngs,
        **kwargs,
    )

    attn_dim = num_heads * head_dim
    wq = mha.query.kernel[...].reshape(in_features, attn_dim)
    wk = mha.key.kernel[...].reshape(in_features, attn_dim)
    wv = mha.value.kernel[...].reshape(in_features, attn_dim)
    attn.qkv.kernel[...] = jnp.concatenate([wq, wk, wv], axis=-1)

    if has_bias:
        bq = mha.query.bias[...].reshape(-1)
        bk = mha.key.bias[...].reshape(-1)
        bv = mha.value.bias[...].reshape(-1)
        attn.qkv.bias[...] = jnp.concatenate([bq, bk, bv], axis=0)

    attn.proj.kernel[...] = mha.out.kernel[...].reshape(attn_dim, out_features)
    if mha.out.bias is not None:
        attn.proj.bias[...] = mha.out.bias[...]

    return attn


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
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Optional[Type[nnx.Module]] = None,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        preferred_element_type: Optional[Dtype] = None,
        fused_attn: bool = True,
        *,
        rngs: nnx.Rngs,
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

            ``precision``
                XLA dot-product precision passed directly to ``nnx.Conv``.
                Accepts ``jax.lax.Precision`` enum values, string shortcuts
                (``"highest"``, ``"high"``, ``"default"``), or a 2-tuple for
                asymmetric LHS/RHS precision.  ``None`` (default) lets XLA
                choose, which is typically ``"default"`` precision.  On TPUs,
                ``"default"`` maps to bfloat16 matrix units; use
                ``"highest"`` for full float32 accumulation when numerical
                fidelity matters more than throughput.
                (`float32`, `tensorfloat32`, `bfloat16`)
                for ref: https://kolonist26-jax-kr.readthedocs.io/en/latest/jax.lax.html

            ``promote_dtype``
                A callable ``(inputs, kernel, bias, *, dtype) -> (inputs, kernel, bias)``
                that casts operands before the convolution.  The default
                (``flax.nnx.nn.dtypes.promote_dtype``) promotes all operands
                to a common dtype derived from the inputs.  Pass a custom
                function to implement mixed-precision strategies, e.g. keeping
                weights in ``float32`` while inputs are ``bfloat16``.

            ``preferred_element_type``
                Passed to ``jax.lax.conv_general_dilated`` as
                ``preferred_element_type``.  Controls the *output* accumulation
                dtype of the dot product independently of the operand dtypes,
                e.g. ``jnp.float32`` with ``bfloat16`` weights to accumulate
                in higher precision.  ``None`` (default) lets JAX infer this
                from the operand types.
        """
        dim_out = dim_out or dim
        head_dim = attn_head_dim
        if head_dim is None:
            assert dim % num_heads == 0, "dim should be divisible by num_heads"
            head_dim = dim // num_heads
        if qk_norm or scale_norm:
            assert norm_layer is not None, (
                "norm_layer must be provided if qk_norm or scale_norm is True"
            )

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim

        self.qk_norm = qk_norm
        self.scale_norm = scale_norm

        # Common conv kwargs for dtype/precision control
        dtype_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
        )
        self.dtype_kwargs = dtype_kwargs

        self.qkv = nnx.Linear(
            in_features=dim,
            out_features=self.attn_dim * 3,
            use_bias=qkv_bias,
            rngs=rngs,
            **dtype_kwargs,
        )
        self.q_norm = norm_layer(head_dim) if qk_norm else Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else Identity()
        self.attn_drop = nnx.Dropout(attn_drop, rngs=rngs)
        self.norm = norm_layer(self.attn_dim) if scale_norm else Identity()
        self.proj = nnx.Linear(
            in_features=self.attn_dim,
            out_features=dim_out,
            use_bias=proj_bias,
            rngs=rngs,
            **dtype_kwargs,
        )
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)
        self.fused_attn = fused_attn
        self.rngs = rngs

    def to_nnx_mha(self) -> nnx.MultiHeadAttention:
        """Convert to standard nnx.MultiHeadAttention (loses extra features)."""
        if self.qk_norm or self.scale_norm:
            _logger.warning("qk_norm and scale_norm will be lost in conversion")
        return attention_to_mha(self)

    @classmethod
    def from_nnx_mha(
        cls,
        mha: nnx.MultiHeadAttention,
        qk_norm: bool = False,
        scale_norm: bool = False,
        **kwargs,
    ) -> "Attention":
        """Create from nnx.MultiHeadAttention with optional extra features"""
        return mha_to_attention(mha, qk_norm=qk_norm, scale_norm=scale_norm, **kwargs)

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
        if self.fused_attn:
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
                module=None,
                is_causal=is_causal,
                dtype=self.dtype_kwargs.get("dtype", None),
                precision=self.dtype_kwargs.get("precision", None),
                promote_dtype=self.dtype_kwargs.get(
                    "promote_dtype", flax_dtypes.promote_dtype
                ),
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(0, 2, 1)
            attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
            attn = maybe_add_mask(attn, attn_bias)
            attn = jax.nn.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        # (B, num_heads, N, head_dim) -> (B, N, attn_dim)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, self.attn_dim)

        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=deterministic)
        return x


class AttentionRope(nnx.Module):
    """A Self Attention module with ROPE support.

    Includes options for:
     * QK normalization option
     * Attention output (scale) normalization
     * Fused or unfused QKV projection support
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_out: Optional[int] = None,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        num_prefix_tokens: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
        norm_layer: Optional[Type[nnx.Module]] = None,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        rotate_half: bool = False,
        fused_attn: bool = True,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        preferred_element_type: Optional[Dtype] = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            dim_out: Output dimension. If None, same as dim.
            qkv_bias: Whether to add a bias term to the query, key, and value projections
            qkv_fused: Whether to use fused QKV projection (single linear) or separate projections
            num_prefix_tokens: Number of reg/cls tokens at the beginning of the sequence that
                should not have position embeddings applied
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for the output projection
            attn_head_dim: Dimension of each attention head. If None, computed as dim // num_heads.
            norm_layer: Normalization layer constructor to use for QK and scale normalization
            qk_norm: Enable normalization of query (Q) and key (K) vectors with norm_layer
            scale_norm: Enable normalization (scaling) of attention output with norm_layer
            proj_bias: Whether to use bias in the output projection
            rotate_half: Use 'half' ROPE layout instead of default 'interleaved'

            ``precision``
                XLA dot-product precision passed directly to ``nnx.Conv``.
                Accepts ``jax.lax.Precision`` enum values, string shortcuts
                (``"highest"``, ``"high"``, ``"default"``), or a 2-tuple for
                asymmetric LHS/RHS precision.  ``None`` (default) lets XLA
                choose, which is typically ``"default"`` precision.  On TPUs,
                ``"default"`` maps to bfloat16 matrix units; use
                ``"highest"`` for full float32 accumulation when numerical
                fidelity matters more than throughput.
                (`float32`, `tensorfloat32`, `bfloat16`)
                for ref: https://kolonist26-jax-kr.readthedocs.io/en/latest/jax.lax.html

            ``promote_dtype``
                A callable ``(inputs, kernel, bias, *, dtype) -> (inputs, kernel, bias)``
                that casts operands before the convolution.  The default
                (``flax.nnx.nn.dtypes.promote_dtype``) promotes all operands
                to a common dtype derived from the inputs.  Pass a custom
                function to implement mixed-precision strategies, e.g. keeping
                weights in ``float32`` while inputs are ``bfloat16``.

            ``preferred_element_type``
                Passed to ``jax.lax.conv_general_dilated`` as
                ``preferred_element_type``.  Controls the *output* accumulation
                dtype of the dot product independently of the operand dtypes,
                e.g. ``jnp.float32`` with ``bfloat16`` weights to accumulate
                in higher precision.  ``None`` (default) lets JAX infer this
                from the operand types.
        """
        # Common conv kwargs for dtype/precision control
        dtype_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
        )
        self.dtype_kwargs = dtype_kwargs

        dim_out = dim_out or dim
        head_dim = attn_head_dim
        if head_dim is None:
            assert dim % num_heads == 0, "dim should be divisible by num_heads"
            head_dim = dim // num_heads
        if scale_norm or qk_norm:
            assert norm_layer is not None, (
                "norm_layer must be provided if qk_norm or scale_norm is True"
            )

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = head_dim * num_heads
        self.scale = head_dim**-0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.fused_attn = fused_attn
        self.rotate_half = rotate_half

        if qkv_fused:
            self.qkv = nnx.Linear(
                in_features=dim,
                out_features=self.head_dim * 3,
                use_bias=qkv_bias,
                rngs=rngs,
                **dtype_kwargs,
            )
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            self.qkv = None
            self.q_proj = nnx.Linear(
                in_features=dim,
                out_features=self.attn_dim,
                bias=qkv_bias,
                rngs=rngs,
                **dtype_kwargs,
            )
            self.k_proj = nnx.Linear(
                in_features=dim,
                out_features=self.attn_dim,
                bias=qkv_bias,
                rngs=rngs,
                **dtype_kwargs,
            )
            self.v_proj = nnx.Linear(
                in_features=dim,
                out_features=self.attn_dim,
                bias=qkv_bias,
                rngs=rngs,
                **dtype_kwargs,
            )

        self.q_norm = norm_layer(head_dim) if qk_norm else Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else Identity()
        self.attn_drop = nnx.Dropout(rate=attn_drop)
        self.norm = norm_layer(self.attn_dim) if scale_norm else Identity()
        self.proj = nnx.Linear(
            in_features=self.attn_dim,
            out_features=dim_out,
            bias=proj_bias,
            rngs=rngs,
            **dtype_kwargs,
        )
        self.proj_drop = nnx.Dropout(rate=proj_drop)

        self.qkv_fused = qkv_fused
        self.rngs = rngs

    def __call__(
        self,
        x: jax.Array,
        rope: Optional[jax.Array] = None,
        attn_mask: Optional[jax.Array] = None,
        is_causal: bool = False,
        deterministic: bool = False,
    ):
        """Forward pass for the attention module.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            rope: Rotary position embeddings tensor for position-aware attention
            attn_mask: Optional attention mask to apply during attention computation
            is_causal: If True, use causal (autoregressive) masking

        Returns:
            Tensor of shape (batch_size, sequence_length, dim_out)
        """
        B, N, C = x.shape

        if self.qkv is not None:
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, self.head_dim)
                .tranpose(2, 0, 3, 1, 4)
            )
            q, k, v = qkv  # each: (B, num_heads, N, head_dim)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q = (
                self.q_proj(x)
                .reshape(B, N, self.num_heads, self.head_dim)
                .tranpose(0, 2, 1, 3)
            )
            k = (
                self.k_proj(x)
                .reshape(B, N, self.num_heads, self.head_dim)
                .tranpose(0, 2, 1, 3)
            )
            v = (
                self.v_proj(x)
                .reshape(B, N, self.num_heads, self.head_dim)
                .tranpose(0, 2, 1, 3)
            )

        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            npt = self.num_heads
            half = getattr(self, "rotate_half", False)
            q = jnp.concatenate(
                [
                    q[:, :, :npt, :],
                    apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half),
                ],
                axis=2,
            )

            k = jnp.concatenate(
                [
                    k[:, :, :npt, :],
                    apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half),
                ],
                axis=2,
            )

        if self.fused_attn:
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
                module=None,
                is_causal=is_causal,
                dtype=self.dtype_kwargs.get("dtype", None),
                precision=self.dtype_kwargs.get("precision", None),
                promote_dtype=self.dtype_kwargs.get(
                    "promote_dtype", flax_dtypes.promote_dtype
                ),
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(0, 2, 1)
            attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
            attn = maybe_add_mask(attn, attn_bias)
            attn = jax.nn.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        # (B, num_heads, N, head_dim) -> (B, N, attn_dim)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, self.attn_dim)

        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=deterministic)
        return x
