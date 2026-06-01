"""MLP module w/ dropout and configurable activation layer

JAX/Flax implementation.

Hacked together by / Copyright 2020 Ross Wightman
Hacked together by / Copyright 2026, Rinat Shaymukhametov
"""

from functools import partial
from typing import Optional, Type, Union, Tuple, Callable

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype, PromoteDtypeFn, PrecisionLike
from flax.nnx.nn import dtypes as flax_dtypes
from flax.nnx import initializers

from .grn import GlobalResponseNorm
from .helpers import to_2tuple
from .identity import Identity


class Mlp(nnx.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks.

    NOTE: When use_conv=True, expects 4D NHWC tensors, otherwise N*C expected.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nnx.gelu,
        norm_layer: Optional[Callable] = None,
        bias: Union[bool, Tuple[bool, bool]] = True,
        drop: Union[float, Tuple[float, float]] = 0.0,
        use_conv: bool = False,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        preferred_element_type: Optional[Dtype] = None,
        *,
        rngs: nnx.Rngs,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        conv_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
        )

        if use_conv:
            # 1x1 conv preserves spatial dims (NHWC layout in JAX)
            linear_layer = partial(
                nnx.Conv,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
            )
        else:
            linear_layer = nnx.Linear

        self.fc1 = linear_layer(
            in_features, hidden_features, use_bias=bias[0], **conv_kwargs, rngs=rngs
        )
        self.act = act_layer
        self.drop1 = nnx.Dropout(rate=drop_probs[0], rngs=rngs)
        self.norm = (
            norm_layer(hidden_features, rngs=rngs)
            if norm_layer is not None
            else Identity()
        )
        self.fc2 = linear_layer(
            hidden_features, out_features, use_bias=bias[1], **conv_kwargs, rngs=rngs
        )
        self.drop2 = nnx.Dropout(rate=drop_probs[1], rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GluMlp(nnx.Module):
    """MLP w/ GLU style gating.
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202

    NOTE: When use_conv=True, expects 4D NHWC tensors, otherwise N*C expected.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nnx.sigmoid,
        norm_layer: Optional[Callable] = None,
        bias: Union[bool, Tuple[bool, bool]] = True,
        drop: Union[float, Tuple[float, float]] = 0.0,
        use_conv: bool = False,
        gate_last: bool = True,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        preferred_element_type: Optional[Dtype] = None,
        *,
        rngs: nnx.Rngs,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.chunk_dim = -1 if not use_conv else -1  # both NHWC and N*C chunk on last dim in JAX
        self.gate_last = gate_last

        conv_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
        )

        if use_conv:
            linear_layer = partial(
                nnx.Conv,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
            )
        else:
            linear_layer = nnx.Linear

        self.fc1 = linear_layer(
            in_features, hidden_features, use_bias=bias[0], **conv_kwargs, rngs=rngs
        )
        self.act = act_layer
        self.drop1 = nnx.Dropout(rate=drop_probs[0], rngs=rngs)
        self.norm = (
            norm_layer(hidden_features // 2, rngs=rngs)
            if norm_layer is not None
            else Identity()
        )
        self.fc2 = linear_layer(
            hidden_features // 2, out_features, use_bias=bias[1], **conv_kwargs, rngs=rngs
        )
        self.drop2 = nnx.Dropout(rate=drop_probs[1], rngs=rngs)
        self.rngs = rngs

    def init_weights(self):
        # Initialize gate portion: bias=1, weights near zero
        if self.fc1.bias is not None:
            half = self.fc1.bias.get_value().shape[0] // 2
            bias_val = self.fc1.bias.get_value().at[half:].set(jnp.ones_like(self.fc1.bias.get_value()[half:]))
            self.fc1.bias = nnx.Param(bias_val)
        half = self.fc1.kernel.get_value().shape[-1] // 2
        kernel_val = self.fc1.kernel.get_value().at[..., half:].set(
            jax.random.normal(self.rngs.params(), self.fc1.kernel.get_value()[..., half:].shape, dtype=self.fc1.kernel.get_value().dtype) * 1e-6
        )
        self.fc1.kernel = nnx.Param(kernel_val)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.fc1(x)
        x1, x2 = jnp.split(x, 2, axis=self.chunk_dim)
        x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


SwiGLUPacked = partial(GluMlp, act_layer=nnx.silu, gate_last=False)


class SwiGLU(nnx.Module):
    """SwiGLU with split fc1 projections.

    NOTE: GluMlp can implement SwiGLU, but this impl has separate fc1_g/fc1_x
    which simplifies checkpoint mapping with other common implementations.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nnx.silu,
        norm_layer: Optional[Callable] = None,
        bias: Union[bool, Tuple[bool, bool]] = True,
        drop: Union[float, Tuple[float, float]] = 0.0,
        align_to: int = 0,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        preferred_element_type: Optional[Dtype] = None,
        *,
        rngs: nnx.Rngs,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        if align_to:
            hidden_features = hidden_features + (-hidden_features % align_to)

        conv_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
        )

        self.fc1_g = nnx.Linear(in_features, hidden_features, use_bias=bias[0], **conv_kwargs, rngs=rngs)
        self.fc1_x = nnx.Linear(in_features, hidden_features, use_bias=bias[0], **conv_kwargs, rngs=rngs)
        self.act = act_layer
        self.drop1 = nnx.Dropout(rate=drop_probs[0], rngs=rngs)
        self.norm = (
            norm_layer(hidden_features, rngs=rngs)
            if norm_layer is not None
            else Identity()
        )
        self.fc2 = nnx.Linear(hidden_features, out_features, use_bias=bias[1], **conv_kwargs, rngs=rngs)
        self.drop2 = nnx.Dropout(rate=drop_probs[1], rngs=rngs)
        self.rngs = rngs

    def init_weights(self):
        if self.fc1_g.bias is not None:
            self.fc1_g.bias = nnx.Param(jnp.ones_like(self.fc1_g.bias.value))
        self.fc1_g.kernel = nnx.Param(
            jax.random.normal(self.rngs.params(), self.fc1_g.kernel.value.shape, dtype=self.fc1_g.kernel.value.dtype) * 1e-6
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedMlp(nnx.Module):
    """MLP as used in gMLP."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nnx.gelu,
        norm_layer: Optional[Callable] = None,
        gate_layer: Optional[Callable] = None,
        bias: Union[bool, Tuple[bool, bool]] = True,
        drop: Union[float, Tuple[float, float]] = 0.0,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        preferred_element_type: Optional[Dtype] = None,
        *,
        rngs: nnx.Rngs,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        conv_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
        )

        self.fc1 = nnx.Linear(in_features, hidden_features, use_bias=bias[0], **conv_kwargs, rngs=rngs)
        self.act = act_layer
        self.drop1 = nnx.Dropout(rate=drop_probs[0], rngs=rngs)
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features, rngs=rngs)
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = Identity()
        self.norm = (
            norm_layer(hidden_features, rngs=rngs)
            if norm_layer is not None
            else Identity()
        )
        self.fc2 = nnx.Linear(hidden_features, out_features, use_bias=bias[1], **conv_kwargs, rngs=rngs)
        self.drop2 = nnx.Dropout(rate=drop_probs[1], rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvMlp(nnx.Module):
    """MLP using 1x1 convs that keeps spatial dims (for 4D NHWC tensors)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nnx.relu,
        norm_layer: Optional[Callable] = None,
        bias: Union[bool, Tuple[bool, bool]] = True,
        drop: float = 0.0,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        preferred_element_type: Optional[Dtype] = None,
        *,
        rngs: nnx.Rngs,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        conv_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
        )

        self.fc1 = nnx.Conv(
            in_features, hidden_features,
            kernel_size=(1, 1), strides=(1, 1), padding="VALID",
            use_bias=bias[0], **conv_kwargs, rngs=rngs,
        )
        self.norm = norm_layer(hidden_features, rngs=rngs) if norm_layer else Identity()
        self.act = act_layer
        self.drop = nnx.Dropout(rate=drop, rngs=rngs)
        self.fc2 = nnx.Conv(
            hidden_features, out_features,
            kernel_size=(1, 1), strides=(1, 1), padding="VALID",
            use_bias=bias[1], **conv_kwargs, rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class GlobalResponseNormMlp(nnx.Module):
    """MLP w/ Global Response Norm (see grn.py), nn.Linear or 1x1 Conv.

    NOTE: Intended for 4D NHWC (use_conv=False, channels-last) or
    NCHW-equivalent (use_conv=True) tensor layouts.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nnx.gelu,
        bias: Union[bool, Tuple[bool, bool]] = True,
        drop: Union[float, Tuple[float, float]] = 0.0,
        use_conv: bool = False,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        preferred_element_type: Optional[Dtype] = None,
        *,
        rngs: nnx.Rngs,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        conv_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
        )
        self.promote_dtype = promote_dtype
        self.preferred_element_type = preferred_element_type

        if use_conv:
            linear_layer = partial(
                nnx.Conv,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
            )
        else:
            linear_layer = nnx.Linear

        self.fc1 = linear_layer(in_features, hidden_features, use_bias=bias[0], **conv_kwargs, rngs=rngs)
        self.act = act_layer
        self.drop1 = nnx.Dropout(rate=drop_probs[0], rngs=rngs)
        self.grn = GlobalResponseNorm(
            hidden_features,
            channels_last=not use_conv,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.fc2 = linear_layer(hidden_features, out_features, use_bias=bias[1], **conv_kwargs, rngs=rngs)
        self.drop2 = nnx.Dropout(rate=drop_probs[1], rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.to(self.preferred_element_type)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.grn(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = x.to(self.promote_dtype)
        return x