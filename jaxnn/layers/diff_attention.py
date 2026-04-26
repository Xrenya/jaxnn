"""Differential Attention

Paper: 'Differential Transformer' - https://arxiv.org/abs/2410.05258

Reference impl: https://github.com/microsoft/unilm/tree/master/Diff-Transformer
"""

import math
from typing import Optional, Type

import jax
from flax import nnx
import jax.numpy as jnp
from flax.typing import Dtype, PromoteDtypeFn, PrecisionLike
from flax.nnx.nn import dtypes as flax_dtypes
from flax.linen.attention import dot_product_attention_weights

from .identity import Identity
from .helpers import wrap_norm_layer
from .attention import maybe_add_mask, resolve_self_attn_mask


class DiffAttention(nnx.Module):
    """Differential Attention module.

    Computes attention as the difference between two softmax attention maps, which helps
    cancel out noise and promotes sparse attention patterns. The module splits Q and K
    into two groups, computes separate attention maps, and subtracts one from the other
    scaled by a learnable lambda parameter.

    The attention output is computed as:
        Attn = softmax(Q1 @ K1^T) - lambda * softmax(Q2 @ K2^T)
        Output = Attn @ V

    Supports both fused (scaled_dot_product_attention) and manual implementations.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Optional[Type[nnx.Module]] = None,
        depth: int = 0,
        dual_lambda: bool = False,
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
        """Initialize the DiffAttention module.

        Args:
            dim: Input dimension of the token embeddings.
            num_heads: Number of attention heads.
            qkv_bias: Whether to use bias in the query, key, value projections.
            qk_norm: Whether to apply normalization to query and key vectors.
            scale_norm: Whether to apply normalization before the output projection.
            proj_bias: Whether to use bias in the output projection.
            attn_drop: Dropout rate applied to the attention weights.
            proj_drop: Dropout rate applied after the output projection.
            norm_layer: Normalization layer constructor (defaults to RmsNorm).
            depth: Block depth index, used to compute depth-dependent lambda_init.
            dual_lambda: If True, use simplified dual scalar lambda parameterization
                (2 params). If False, use the paper's original formulation with
                lambda_q/k vectors (4 * head_dim params).
            dtype: The dtype of the computation.
            param_dtype: The dtype of the parameters.
            precision: XLA dot-product precision.
            promote_dtype: Function to promote dtypes before operations.
            preferred_element_type: Output accumulation dtype for dot products.
            norm_dtype: Dtype for normalization layers.
            norm_param_dtype: Param dtype for normalization layers.
            norm_promote_dtype: Promote dtype function for normalization.
            fused_attn: Use fused attention implementation.
            rngs: NNX RNGs for random operations.
        """
        self.fused_attn = fused_attn

        # Common kwargs for dtype/precision control
        dtype_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
        )
        self.dtype_kwargs = dtype_kwargs

        dtype_norm_kwargs = dict(
            dtype=norm_dtype,
            param_dtype=norm_param_dtype,
            promote_dtype=norm_promote_dtype,
        )
        self.dtype_norm_kwargs = dtype_norm_kwargs
        self.rngs = rngs

        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        if norm_layer is None:
            norm_layer = nnx.RMSNorm

        self.num_heads = num_heads
        self.head_dim = dim // num_heads // 2  # Split heads into 2 groups
        self.scale = self.head_dim**-0.5
        self.attn_dim = self.num_heads * 2 * self.head_dim  # Total attention dim

        # QKV projection - outputs dim * 3 (standard)
        self.qkv = nnx.Linear(
            in_features=dim,
            out_features=dim * 3,
            use_bias=qkv_bias,
            rngs=rngs,
            **dtype_kwargs,
        )

        # QK normalization
        self.q_norm = (
            wrap_norm_layer(
                norm_layer,
                dtype=norm_dtype,
                param_dtype=norm_param_dtype,
                promote_dtype=norm_promote_dtype,
            )(num_features=self.head_dim, rngs=rngs)
            if qk_norm
            else Identity()
        )
        self.k_norm = (
            wrap_norm_layer(
                norm_layer,
                dtype=norm_dtype,
                param_dtype=norm_param_dtype,
                promote_dtype=norm_promote_dtype,
            )(num_features=self.head_dim, rngs=rngs)
            if qk_norm
            else Identity()
        )

        self.attn_drop = nnx.Dropout(attn_drop, rngs=rngs)
        self.attn_drop_p = attn_drop

        # Output normalization
        self.norm = (
            wrap_norm_layer(
                norm_layer,
                dtype=norm_dtype,
                param_dtype=norm_param_dtype,
                promote_dtype=norm_promote_dtype,
            )(num_features=dim, rngs=rngs)
            if scale_norm
            else Identity()
        )

        self.proj = nnx.Linear(
            in_features=dim,
            out_features=dim,
            use_bias=proj_bias,
            rngs=rngs,
            **dtype_kwargs,
        )
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)

        # Lambda parameters for differential attention
        self.dual_lambda = dual_lambda
        lambda_dtype = norm_param_dtype

        if dual_lambda:
            # Simplified: 2 scalar parameters
            self.lambda_a = nnx.Param(
                nnx.initializers.zeros(rngs.params(), (), dtype=lambda_dtype)
            )
            self.lambda_b = nnx.Param(
                nnx.initializers.zeros(rngs.params(), (), dtype=lambda_dtype)
            )
            self.lambda_q1 = self.lambda_k1 = self.lambda_q2 = self.lambda_k2 = None
        else:
            # Original paper: 4 vectors of size head_dim
            initializer = nnx.initializers.normal(stddev=0.1, dtype=lambda_dtype)
            self.lambda_a = self.lambda_b = None
            self.lambda_q1 = nnx.Param(initializer(rngs.params(), (self.head_dim,)))
            self.lambda_k1 = nnx.Param(initializer(rngs.params(), (self.head_dim,)))
            self.lambda_q2 = nnx.Param(initializer(rngs.params(), (self.head_dim,)))
            self.lambda_k2 = nnx.Param(initializer(rngs.params(), (self.head_dim,)))

        # Sub-layer normalization for attention output
        self.sub_norm = wrap_norm_layer(
            nnx.RMSNorm,
            dtype=norm_dtype,
            param_dtype=norm_param_dtype,
            promote_dtype=norm_promote_dtype,
        )(num_features=2 * self.head_dim, epsilon=1e-5, rngs=rngs)

        # Lambda initialization based on depth
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)

    def _compute_lambda(self) -> jax.Array:
        """Compute the lambda scaling factor for differential attention."""
        if self.dual_lambda:
            # Scalar lambda from two parameters
            lambda_1 = jnp.exp(self.lambda_a.value)
            lambda_2 = jnp.exp(self.lambda_b.value)
        else:
            # Vector lambda from dot products
            lambda_1 = jnp.exp(
                jnp.sum(self.lambda_q1.value * self.lambda_k1.value, axis=-1)
            )
            lambda_2 = jnp.exp(
                jnp.sum(self.lambda_q2.value * self.lambda_k2.value, axis=-1)
            )
        return lambda_1 - lambda_2 + self.lambda_init

    def __call__(
        self,
        x: jax.Array,
        attn_mask: Optional[jax.Array] = None,
        is_causal: bool = False,
        deterministic: bool = False,
    ):
        B, N, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=2)

        # Reshape for multi-head: split heads into 2 groups for differential
        # Q, K: (B, N, 2*num_heads, head_dim) -> (B, 2*num_heads, N, head_dim)
        q = q.reshape(B, N, 2 * self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, 2 * self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        # V: (B, N, num_heads, 2*head_dim) -> (B, num_heads, N, 2*head_dim)
        v = v.reshape(B, N, self.num_heads, 2 * self.head_dim).transpose(0, 2, 1, 3)

        # Apply QK normalization
        q, k = self.q_norm(q), self.k_norm(k)

        # Split into two groups for differential attention
        q1, q2 = jnp.split(q, 2, axis=1)  # Each: (B, num_heads, N, head_dim)
        k1, k2 = jnp.split(k, 2, axis=1)  # Each: (B, num_heads, N, head_dim)

        # Compute lambda scaling factor
        lambda_full = self._compute_lambda()

        attn_bias = resolve_self_attn_mask(
            seq_len=N,
            attn=q1,  # shape used only for dtype
            attn_mask=attn_mask,
            is_causal=is_causal,
        )

        if self.fused_attn:
            # Transpose for flax.linen attention API: (B, N, num_heads, head_dim)
            q1_ = q1.transpose(0, 2, 1, 3)
            q2_ = q2.transpose(0, 2, 1, 3)
            k1_ = k1.transpose(0, 2, 1, 3)
            k2_ = k2.transpose(0, 2, 1, 3)
            v_ = v.transpose(0, 2, 1, 3)

            dropout_rng = self.rngs.dropout() if not deterministic else None

            # Compute attention weights separately for both groups
            attn_w1 = dot_product_attention_weights(
                query=q1_,
                key=k1_,
                bias=attn_bias,
                mask=None,
                broadcast_dropout=True,
                dropout_rng=dropout_rng,
                dropout_rate=self.attn_drop.rate,
                deterministic=deterministic,
                dtype=self.dtype_kwargs.get("dtype", None),
                precision=self.dtype_kwargs.get("precision", None),
            )
            attn_w2 = dot_product_attention_weights(
                query=q2_,
                key=k2_,
                bias=attn_bias,
                mask=None,
                broadcast_dropout=True,
                dropout_rng=dropout_rng,
                dropout_rate=self.attn_drop.rate,
                deterministic=deterministic,
                dtype=self.dtype_kwargs.get("dtype", None),
                precision=self.dtype_kwargs.get("precision", None),
            )

            # Differential attention: subtract scaled second attention
            diff_w = attn_w1 - lambda_full * attn_w2

            # Apply differential weights to values
            x = jnp.einsum(
                "...hqk,...khd->...qhd",
                diff_w,
                v_,
                precision=self.dtype_kwargs.get("precision", None),
            )
            # Transpose back: (B, N, num_heads, 2*head_dim)
            x = x.transpose(0, 2, 1, 3)
        else:
            # Manual attention computation
            scale = self.scale
            attn1 = (q1 * scale) @ k1.transpose(0, 1, 3, 2)
            attn2 = (q2 * scale) @ k2.transpose(0, 1, 3, 2)

            attn1 = maybe_add_mask(attn1, attn_bias)
            attn2 = maybe_add_mask(attn2, attn_bias)

            attn1 = jax.nn.softmax(attn1, axis=-1)
            attn2 = jax.nn.softmax(attn2, axis=-1)

            attn1 = self.attn_drop(attn1, deterministic=deterministic)
            attn2 = self.attn_drop(attn2, deterministic=deterministic)

            x = (attn1 - lambda_full * attn2) @ v

        # Apply sub-layer normalization and lambda scaling
        x = self.sub_norm(x)
        x = x * (1 - self.lambda_init)

        # Reshape to (B, N, attn_dim)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, self.attn_dim)

        # Final projection
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=deterministic)
        return x
