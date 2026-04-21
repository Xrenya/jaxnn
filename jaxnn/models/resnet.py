"""Flax/JAX ResNet

Adapted from PyTorch/timm's resnet implementation.
Copyright of original work: 2019 Ross Wightman

Hacked together by / Copyright 2026 Rinat Shaymukhametov
"""

import functools
import jax
from flax import nnx
from flax.typing import Dtype, PromoteDtypeFn, PrecisionLike
from flax.nnx.nn import dtypes as flax_dtypes
import jax.numpy as jnp
from functools import partial

from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

from ..layers import to_ntuple, Activation, MaxPool2D, AvgPool2D, Identity

from jaxnn.models._registry import register_model, generate_default_cfgs
from jaxnn.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from jaxnn.models._builder import build_model_with_cfg

LayerType = Union[str, Callable, Type[nnx.Module]]


__all__ = ["ResNet", "BasicBlock", "Bottleneck"]


# Supported norm layer string aliases.
# All callables share the interface: cls(num_features, *, rngs).
# e.g. GroupNorm: nnx.GroupNorm(num_features, num_groups=32, *, rngs) - compatible.
_NORM_LAYER_MAP: Dict[str, Type[nnx.Module]] = {
    "batchnorm": nnx.BatchNorm,
    "batch_norm": nnx.BatchNorm,
    "bn": nnx.BatchNorm,
    "groupnorm": nnx.GroupNorm,
    "group_norm": nnx.GroupNorm,
    "gn": nnx.GroupNorm,
    "layernorm": nnx.LayerNorm,
    "layer_norm": nnx.LayerNorm,
    "ln": nnx.LayerNorm,
}


def get_norm_layer(norm_layer: LayerType) -> Type[nnx.Module]:
    """Resolve a norm layer string alias or pass a callable through unchanged.

    Args:
        norm_layer: A string alias (e.g. ``"groupnorm"``, ``"bn"``) or an
            already-callable class such as ``nnx.BatchNorm``.

    Returns:
        A callable class with the signature ``cls(num_features, *, rngs)``.

    Raises:
        ValueError: If a string alias is not recognised.
        TypeError: If the argument is neither a string nor a callable.
    """
    if isinstance(norm_layer, str):
        key = norm_layer.lower().replace("-", "").replace(" ", "")
        if key not in _NORM_LAYER_MAP:
            raise ValueError(
                f"Unknown norm_layer string {norm_layer!r}. "
                f"Supported aliases: {sorted(_NORM_LAYER_MAP)}"
            )
        return _NORM_LAYER_MAP[key]
    if callable(norm_layer):
        return norm_layer
    raise TypeError(
        f"norm_layer must be a string alias or a callable class, "
        f"got {type(norm_layer)}"
    )


def wrap_norm_layer(
    norm_cls: Type[nnx.Module],
    dtype: Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
) -> Callable:
    """Return a norm-layer factory with dtype/param_dtype/promote_dtype pre-bound.

    The returned callable has the same ``(num_features, *, rngs)`` signature
    expected everywhere in this file, so all call sites remain unchanged.

    Args:
        norm_cls: A norm-layer class (``nnx.BatchNorm``, ``nnx.LayerNorm``,
            ``nnx.GroupNorm``, or any compatible custom class).
        dtype: Output (activation) dtype of the norm layer.  ``None`` lets
            Flax infer it from the inputs (default behaviour).  Pass
            ``jnp.float32`` to keep activations in full precision when the
            model otherwise runs in a lower-precision dtype (e.g. bfloat16).
        param_dtype: Storage dtype for the learnable scale/bias parameters.
            Defaults to ``jnp.float32`` — matching Flax's own default and
            the standard mixed-precision recommendation (keep parameters in
            float32 regardless of activation dtype).
        promote_dtype: Callable that casts ``(x, scale, bias)`` before the
            normalisation arithmetic.  Defaults to Flax's built-in promotion
            function.

    Returns:
        A callable ``factory(num_features, *, rngs)`` that constructs the
        norm layer with the pre-bound dtype settings.
    """
    @functools.wraps(norm_cls)
    def factory(num_features: int, *, rngs: nnx.Rngs) -> nnx.Module:
        return norm_cls(
            num_features,
            dtype=dtype,
            param_dtype=param_dtype,
            promote_dtype=promote_dtype,
            rngs=rngs,
        )
    return factory


class Downsample(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        strides: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        first_dilation: Optional[Tuple[int, int]] = None,
        norm_layer: Optional[Type[nnx.Module]] = None,
        padding: Union[str, Tuple[int, int]] = "SAME",
        use_bias: bool = False,
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
    ) -> None:
        kernel_size = (
            (1, 1) if strides == (1, 1) and dilation == (1, 1) else kernel_size
        )
        first_dilation = (
            (first_dilation or dilation) if kernel_size != (1, 1) else (1, 1)
        )

        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            kernel_dilation=first_dilation,
            padding=padding,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
            rngs=rngs,
        )
        norm_layer = norm_layer or nnx.BatchNorm
        self.bn = wrap_norm_layer(
            norm_layer,
            dtype=norm_dtype,
            param_dtype=norm_param_dtype,
            promote_dtype=norm_promote_dtype,
        )(num_features=out_channels, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)
        x = self.bn(x)
        return x


class AntiAliasingLayer(nnx.Module):
    def __init__(
        self,
        aa_layer: LayerType,
        strides: Tuple[int, int] = (2, 2),
        window_shape: Tuple[int, int] = (2, 2),
        enable: bool = True,
        padding: str = "VALID",
        noop: Optional[Type[nnx.Module]] = nnx.identity,
    ) -> None:
        if isinstance(strides, int):
            strides = (strides, strides)

        self.pooling = True
        if not aa_layer or not enable:
            self.pooling = False
            aa_layer = noop if noop is not None else None
        elif isinstance(aa_layer, str):
            aa_layer = aa_layer.lower().replace("_", "").replace("-", "")
            if aa_layer in ("avg", "avgpool"):
                aa_layer = nnx.avg_pool
            else:
                raise ValueError(f"Unknown anti-aliasing layer ({aa_layer}).")

        self.pool = aa_layer
        self.window_shape = window_shape
        self.strides = strides
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.pooling:
            return self.pool(
                inputs=x,
                window_shape=self.window_shape,
                strides=self.strides,
                padding=self.padding,
            )
        return self.pool(x)


class AdaptiveAvgPool2D(nnx.Module):
    def __init__(self, output_size: Union[int, Tuple[int, int]]) -> None:
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size: Tuple[int, int] = output_size

    def __call__(self, x: jax.Array) -> jax.Array:
        _, input_height, input_width, _ = x.shape

        h_stride = input_height // self.output_size[0]
        w_stride = input_width // self.output_size[1]

        h_kernel = input_height - (self.output_size[0] - 1) * h_stride
        w_kernel = input_width - (self.output_size[1] - 1) * w_stride

        return nnx.avg_pool(
            x,
            window_shape=(h_kernel, w_kernel),
            strides=(h_stride, w_stride),
        )


class Classifier(nnx.Module):
    def __init__(
        self,
        num_pooled_features: int,
        num_classes: int,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        preferred_element_type: Optional[Dtype] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.in_features = num_pooled_features
        self.out_features = num_classes
        if num_classes <= 0:
            self.fc = Identity()
        else:
            self.fc = nnx.Linear(
                in_features=num_pooled_features,
                out_features=num_classes,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                promote_dtype=promote_dtype,
                preferred_element_type=preferred_element_type,
                rngs=rngs,
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc(x)


class BasicBlock(nnx.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        strides: Tuple[int, int] = (1, 1),
        downsample: Optional[nnx.Module] = None,
        cardinality: int = 1,
        base_width: int = 64,
        reduce_first: int = 1,
        dilation: Tuple[int, int] = (1, 1),
        *,
        rngs: nnx.Rngs,
        first_dilation: Optional[Tuple[int, int]] = None,
        act_layer: Type[nnx.Module] = nnx.relu,
        norm_layer: Type[nnx.Module] = nnx.BatchNorm,
        attn_layer: Optional[Callable] = None,
        aa_layer: Optional[Type[nnx.Module]] = None,
        drop_block: Optional[Type[nnx.Module]] = None,
        drop_path: Optional[nnx.Module] = None,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        preferred_element_type: Optional[Dtype] = None,
        norm_dtype: Optional[Dtype] = jnp.float32,
        norm_param_dtype: Dtype = jnp.float32,
        norm_promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
    ) -> None:
        assert cardinality == 1, "BasicBlock only supports cardinality of 1"
        assert base_width == 64, "BasicBlock does not support changing base width"

        first_planes = planes // reduce_first
        out_planes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (
            strides[0] == 2 or first_dilation[0] != dilation[0]
        )

        # Common conv kwargs for dtype/precision control
        conv_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
        )
        _norm = wrap_norm_layer(norm_layer, dtype=norm_dtype, param_dtype=norm_param_dtype, promote_dtype=norm_promote_dtype)

        self.conv1 = nnx.Conv(
            in_features=inplanes,
            out_features=first_planes,
            kernel_size=(3, 3),
            strides=(1, 1) if use_aa else strides,
            padding=first_dilation,
            kernel_dilation=first_dilation,
            use_bias=False,
            rngs=rngs,
            **conv_kwargs,
        )
        self.bn1 = _norm(num_features=first_planes, rngs=rngs)
        self.drop_block = (
            drop_block(rngs=rngs) if drop_block is not None else nnx.identity
        )
        self.act1 = Activation(act_layer)
        self.aa = AntiAliasingLayer(aa_layer, strides=strides, enable=use_aa)

        self.conv2 = nnx.Conv(
            in_features=first_planes,
            out_features=out_planes,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=dilation,
            kernel_dilation=dilation,
            use_bias=False,
            rngs=rngs,
            **conv_kwargs,
        )
        self.bn2 = _norm(num_features=out_planes, rngs=rngs)

        self.se = attn_layer(out_planes, rngs=rngs) if attn_layer is not None else None
        self.act2 = Activation(act_layer)
        self.downsample = downsample
        self.strides = strides
        self.dilation = dilation
        self.drop_path = drop_path

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x, training=training)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x = x + shortcut
        x = self.act2(x)

        return x


class Bottleneck(nnx.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        strides: Tuple[int, int] = (1, 1),
        downsample: Optional[nnx.Module] = None,
        cardinality: int = 1,
        base_width: int = 64,
        reduce_first: int = 1,
        dilation: Tuple[int, int] = (1, 1),
        *,
        rngs: nnx.Rngs,
        first_dilation: Optional[Tuple[int, int]] = None,
        act_layer: Type[nnx.Module] = nnx.relu,
        norm_layer: Type[nnx.Module] = nnx.BatchNorm,
        attn_layer: Optional[Callable] = None,
        aa_layer: Optional[Type[nnx.Module]] = None,
        drop_block: Optional[Type[nnx.Module]] = None,
        drop_path: Optional[nnx.Module] = None,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        preferred_element_type: Optional[Dtype] = None,
        norm_dtype: Optional[Dtype] = jnp.float32,
        norm_param_dtype: Dtype = jnp.float32,
        norm_promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
    ) -> None:
        width = int(planes * (base_width / 64.0)) * cardinality
        first_planes = width // reduce_first
        out_planes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (
            strides[0] == 2 or first_dilation[0] != dilation[0]
        )

        # Common conv kwargs for dtype/precision control
        conv_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
        )
        _norm = wrap_norm_layer(norm_layer, dtype=norm_dtype, param_dtype=norm_param_dtype, promote_dtype=norm_promote_dtype)

        self.conv1 = nnx.Conv(
            in_features=inplanes,
            out_features=first_planes,
            kernel_size=(1, 1),
            use_bias=False,
            rngs=rngs,
            **conv_kwargs,
        )
        self.bn1 = _norm(num_features=first_planes, rngs=rngs)
        self.act1 = Activation(act_layer)

        self.conv2 = nnx.Conv(
            in_features=first_planes,
            out_features=width,
            kernel_size=(3, 3),
            strides=(1, 1) if use_aa else strides,
            padding=first_dilation,
            kernel_dilation=first_dilation,
            feature_group_count=cardinality,
            use_bias=False,
            rngs=rngs,
            **conv_kwargs,
        )
        self.bn2 = _norm(num_features=width, rngs=rngs)
        self.drop_block = (
            drop_block(rngs=rngs) if drop_block is not None else nnx.identity
        )
        self.act2 = Activation(act_layer)
        self.aa = AntiAliasingLayer(aa_layer, strides=strides, enable=use_aa)

        self.conv3 = nnx.Conv(
            in_features=width,
            out_features=out_planes,
            kernel_size=(1, 1),
            use_bias=False,
            rngs=rngs,
            **conv_kwargs,
        )
        self.bn3 = _norm(num_features=out_planes, rngs=rngs)

        self.se = attn_layer(out_planes, rngs=rngs) if attn_layer is not None else None
        self.act3 = Activation(act_layer)
        self.downsample = downsample
        self.strides = strides
        self.dilation = dilation
        self.drop_path = drop_path

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x, training=training)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x = x + shortcut
        x = self.act3(x)

        return x


class DropPath(nnx.Module):
    def __init__(
        self,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True,
        *,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.rngs = rngs

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        if self.drop_prob == 0.0 or not training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        key = self.rngs.dropout() if self.rngs is not None else jax.random.PRNGKey(0)
        random_tensor = jax.random.bernoulli(key, keep_prob, shape)

        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor = random_tensor / keep_prob

        return x * random_tensor


def downsample_avg(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    strides: Tuple[int, int] = (1, 1),
    dilation: Tuple[int, int] = (1, 1),
    first_dilation: Optional[Tuple[int, int]] = None,
    norm_layer: Optional[Type[nnx.Module]] = None,
    padding: Union[str, Tuple[int, int]] = "SAME",
    use_bias: bool = False,
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
) -> nnx.Module:
    norm_layer = norm_layer or nnx.BatchNorm
    avg_stride: Tuple[int, int] = strides if dilation == (1, 1) else (1, 1)
    need_pool = avg_stride[0] > 1 or (dilation[0] > 1 and strides[0] > 1)
    _norm = wrap_norm_layer(norm_layer, dtype=norm_dtype, param_dtype=norm_param_dtype, promote_dtype=norm_promote_dtype)

    class AvgDownsample(nnx.Module):
        def __init__(self):
            if need_pool:
                self.pool = AvgPool2D(
                    kernel_size=(2, 2),
                    strides=avg_stride,
                    padding="SAME",
                )
            else:
                self.pool = None
            self.conv = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                use_bias=use_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                promote_dtype=promote_dtype,
                preferred_element_type=preferred_element_type,
                rngs=rngs,
            )
            self.bn = _norm(num_features=out_channels, rngs=rngs)

        def __call__(self, x: jax.Array) -> jax.Array:
            if self.pool is not None:
                x = self.pool(x)
            x = self.conv(x)
            x = self.bn(x)
            return x

    return AvgDownsample()


def drop_blocks(drop_prob: float = 0.0) -> List[Optional[partial]]:
    """Generate per-stage DropBlock configs (4 stages).

    DropBlock is only applied to stages 3 and 4 (index 2, 3) following
    the original paper.
    """
    return [
        None,
        None,
        partial(DropPath, drop_prob=drop_prob) if drop_prob else None,
        partial(DropPath, drop_prob=drop_prob) if drop_prob else None,
    ]


def make_blocks(
    block_fns: Tuple[Union[Type[BasicBlock], Type[Bottleneck]], ...],
    channels: Tuple[int, ...],
    block_repeats: Tuple[int, ...],
    inplanes: int,
    reduce_first: int = 1,
    output_stride: int = 32,
    down_kernel_size: int = 1,
    avg_down: bool = False,
    drop_block_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    **kwargs,
) -> Tuple[List[Tuple[str, nnx.Module]], List[Dict[str, Any]]]:
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    # dilation= and first_dilation= are used as
    # padding= / kernel_dilation= in nnx.Conv.
    dilation: Tuple[int, int] = (1, 1)
    prev_dilation: Tuple[int, int] = (1, 1)

    for stage_idx, (block_fn, planes, num_blocks, db) in enumerate(
        zip(block_fns, channels, block_repeats, drop_blocks(drop_block_rate))
    ):
        stage_name = f"layer{stage_idx + 1}"
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation = (dilation[0] * stride, dilation[1] * stride)
            stride = 1
        else:
            net_stride *= stride

        stride_t: Tuple[int, int] = to_ntuple(2)(stride)

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=to_ntuple(2)(down_kernel_size),
                strides=stride_t,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get("norm_layer"),
                dtype=kwargs.get("dtype"),
                param_dtype=kwargs.get("param_dtype", jnp.float32),
                precision=kwargs.get("precision"),
                promote_dtype=kwargs.get("promote_dtype", flax_dtypes.promote_dtype),
                preferred_element_type=kwargs.get("preferred_element_type"),
                norm_dtype=kwargs.get("norm_dtype", jnp.float32),
                norm_param_dtype=kwargs.get("norm_param_dtype", jnp.float32),
                norm_promote_dtype=kwargs.get("norm_promote_dtype", flax_dtypes.promote_dtype),
                rngs=kwargs.get("rngs"),
            )
            downsample = (
                downsample_avg(**down_kwargs) if avg_down else Downsample(**down_kwargs)
            )

        block_kwargs = dict(
            reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs
        )
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            block_stride: Tuple[int, int] = stride_t if block_idx == 0 else (1, 1)
            block_dpr = (
                drop_path_rate * net_block_idx / (net_num_blocks - 1)
                if net_num_blocks > 1
                else 0.0
            )
            blocks.append(
                block_fn(
                    inplanes,
                    planes,
                    block_stride,
                    downsample,
                    first_dilation=prev_dilation,
                    drop_path=DropPath(block_dpr) if block_dpr > 0.0 else None,
                    **block_kwargs,
                )
            )
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nnx.Sequential(*blocks)))
        feature_info.append(
            dict(num_chs=inplanes, reduction=net_stride, module=stage_name)
        )

    return stages, feature_info


def _build_stem(
    in_chans: int,
    inplanes: int,
    stem_width: int,
    stem_type: str,
    act_layer: LayerType,
    norm_layer: LayerType,
    rngs: nnx.Rngs,
    dtype: Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
    preferred_element_type: Optional[Dtype] = None,
    norm_dtype: Optional[Dtype] = jnp.float32,
    norm_param_dtype: Dtype = jnp.float32,
    norm_promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
) -> nnx.Module:
    """Build the stem conv block for all ResNet variants.

    stem_type        widths (first, middle, final)    avg_down

    "" / "b"        : normal 7x7 conv, out=64
    "deep"          : (stem_width, stem_width, stem_width*2)
                     c  stem_width=32  (32, 32, 64)
                     d  stem_width=32  (32, 32, 64),  avg_down=True
                     e  stem_width=64  (64, 64, 128), avg_down=True
                     s  stem_width=64  (64, 64, 128)
    "deep_tiered"   : (3*(sw//4), 3*(sw//2), sw*2)
                     t  stem_width=32  (24, 48, 64),  avg_down=True
    "deep_tieredn"  : (3*(sw//4), sw, sw*2)  - matches timm's formula exactly
                      tn stem_width=32  (24, 32, 64),  avg_down=True
    """
    deep_stem = "deep" in stem_type

    # Common conv kwargs for dtype/precision control
    conv_kwargs = dict(
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        promote_dtype=promote_dtype,
        preferred_element_type=preferred_element_type,
    )
    _norm = wrap_norm_layer(norm_layer, dtype=norm_dtype, param_dtype=norm_param_dtype, promote_dtype=norm_promote_dtype)

    if deep_stem:
        if "tiered" in stem_type:
            # Isolate the optional "n" suffix after stripping known tokens.
            suffix = (
                stem_type.replace("deep", "")
                .replace("tiered", "")
                .replace("_", "")
                .strip()
            )
            if suffix == "n":
                # tn: matches timm's formula (3*(sw//4), sw) -> (24, 32) for sw=32
                stem_chs = (3 * (stem_width // 4), stem_width)
            else:
                # t: extended variant (3*(sw//4), 3*(sw//2)) -> (24, 48) for sw=32
                stem_chs = (3 * (stem_width // 4), 3 * (stem_width // 2))
        else:
            # c / d / e / s
            stem_chs = (stem_width, stem_width)

        out_chs = stem_width * 2

        return nnx.Sequential(
            nnx.Conv(
                in_features=in_chans,
                out_features=stem_chs[0],
                kernel_size=(3, 3),
                strides=(2, 2),
                padding=(1, 1),
                use_bias=False,
                rngs=rngs,
                **conv_kwargs,
            ),
            _norm(num_features=stem_chs[0], rngs=rngs),
            Activation(act_layer),
            nnx.Conv(
                in_features=stem_chs[0],
                out_features=stem_chs[1],
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=(1, 1),
                use_bias=False,
                rngs=rngs,
                **conv_kwargs,
            ),
            _norm(num_features=stem_chs[1], rngs=rngs),
            Activation(act_layer),
            nnx.Conv(
                in_features=stem_chs[1],
                out_features=out_chs,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=(1, 1),
                use_bias=False,
                rngs=rngs,
                **conv_kwargs,
            ),
        )
    else:
        return nnx.Conv(
            in_features=in_chans,
            out_features=inplanes,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=(3, 3),
            use_bias=False,
            rngs=rngs,
            **conv_kwargs,
        )


class ResNet(nnx.Module):
    """ResNet / ResNeXt / SE-ResNeXt supporting variants b/c/d/e/s/t/tn.

    Stem variant is selected via ``stem_type``:
      ``""`` or ``"b"``   - normal 7x7 stem (stem_width ignored)
      ``"deep"``          - variant c  (stem_width=32, widths 32-32-64)
                          - variant d  (stem_width=32, widths 32-32-64,  avg_down=True)
                          - variant e  (stem_width=64, widths 64-64-128, avg_down=True)
                          - variant s  (stem_width=64, widths 64-64-128)
      ``"deep_tiered"``   - variant t  (stem_width=32, widths 24-48-64, avg_down=True)
      ``"deep_tieredn"``  - variant tn (stem_width=32, widths 24-32-64, avg_down=True)

    Dtype / precision knobs (applied uniformly to all ``nnx.Conv`` calls):

      ``precision``
          XLA dot-product precision passed directly to ``nnx.Conv``.
          Accepts ``jax.lax.Precision`` enum values, string shortcuts
          (``"highest"``, ``"high"``, ``"default"``), or a 2-tuple for
          asymmetric LHS/RHS precision.  ``None`` (default) lets XLA
          choose, which is typically ``"default"`` precision.  On TPUs,
          ``"default"`` maps to bfloat16 matrix units; use
          ``"highest"`` for full float32 accumulation when numerical
          fidelity matters more than throughput.

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

    def __init__(
        self,
        block: Union[Type[BasicBlock], Type[Bottleneck]],
        layers: Tuple[int, ...],
        num_classes: int = 1000,
        in_chans: int = 3,
        output_stride: int = 32,
        global_pool: str = "avg",
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        replace_stem_pool: bool = False,
        block_reduce_first: int = 1,
        down_kernel_size: int = 1,
        avg_down: bool = False,
        channels: Optional[Tuple[int, ...]] = (64, 128, 256, 512),
        act_layer: LayerType = nnx.relu,
        norm_layer: LayerType = nnx.BatchNorm,
        aa_layer: Optional[Type[nnx.Module]] = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_block_rate: float = 0.0,
        zero_init_last: bool = True,
        block_args: Optional[Dict[str, Any]] = None,
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
    ) -> None:
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.global_pool_type = global_pool
        self.grad_checkpointing = False

        norm_layer = get_norm_layer(norm_layer)

        deep_stem = "deep" in stem_type
        inplanes = stem_width * 2 if deep_stem else 64

        # Shared dtype/precision kwargs — forwarded to every nnx.Conv and nnx.Linear
        dtype_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
        )
        # Norm-layer dtype kwargs — forwarded to every norm layer via wrap_norm_layer
        norm_kwargs = dict(
            norm_dtype=norm_dtype,
            norm_param_dtype=norm_param_dtype,
            norm_promote_dtype=norm_promote_dtype,
        )

        self.conv1 = _build_stem(
            in_chans=in_chans,
            inplanes=inplanes,
            stem_width=stem_width,
            stem_type=stem_type,
            act_layer=act_layer,
            norm_layer=norm_layer,
            rngs=rngs,
            **dtype_kwargs,
            **norm_kwargs,
        )
        self.bn1 = wrap_norm_layer(
            norm_layer,
            dtype=norm_dtype,
            param_dtype=norm_param_dtype,
            promote_dtype=norm_promote_dtype,
        )(num_features=inplanes, rngs=rngs)
        self.act1 = Activation(act_layer)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module="act1")]

        # Stem pooling
        if replace_stem_pool:
            self.maxpool = nnx.Sequential(
                nnx.Conv(
                    in_features=inplanes,
                    out_features=inplanes,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding=(1, 1),
                    use_bias=False,
                    rngs=rngs,
                    **dtype_kwargs,
                ),
                wrap_norm_layer(
                    norm_layer,
                    dtype=norm_dtype,
                    param_dtype=norm_param_dtype,
                    promote_dtype=norm_promote_dtype,
                )(num_features=inplanes, rngs=rngs),
                Activation(act_layer),
            )
        else:
            if aa_layer is not None:
                self.maxpool = nnx.Sequential(
                    MaxPool2D(
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding=((1, 1), (1, 1)),
                    ),
                    AntiAliasingLayer(aa_layer, strides=(2, 2)),
                )
            else:
                self.maxpool = MaxPool2D(
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding=((1, 1), (1, 1)),
                )

        block_fns = to_ntuple(len(channels))(block)
        stage_modules, stage_feature_info = make_blocks(
            block_fns,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **dtype_kwargs,
            **norm_kwargs,
            **block_args,
            rngs=rngs,
        )
        self.stage_modules = nnx.List([m for _, m in stage_modules])
        self.feature_info += stage_feature_info
        self.num_features = self.head_hidden_size = (
            channels[-1] * block_fns[-1].expansion
        )
        self.global_pool = AdaptiveAvgPool2D(output_size=(1, 1))
        self.head_drop = (
            nnx.Dropout(rate=drop_rate, rngs=rngs) if drop_rate > 0.0 else None
        )
        self.fc = Classifier(
            self.num_features, self.num_classes, rngs=rngs, **dtype_kwargs
        )

        if zero_init_last:
            for module in self.stage_modules:
                for block in module.layers:
                    if hasattr(block, "bn3"):
                        block.bn3.scale.value = jnp.zeros_like(block.bn3.scale.value)
                    elif hasattr(block, "bn2"):
                        block.bn2.scale.value = jnp.zeros_like(block.bn2.scale.value)

    def _feature_take_indices(
        self,
        indices: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[Tuple[int, ...], int]:
        num_features = len(self.feature_info)

        if indices is None:
            take = tuple(range(num_features))
            return take, num_features - 1

        if isinstance(indices, int):
            assert 0 < indices <= num_features, (
                f"last-n ({indices}) out of range (1..{num_features})"
            )
            take = tuple(range(num_features - indices, num_features))
            return take, num_features - 1

        take = tuple(num_features + i if i < 0 else i for i in indices)
        assert all(0 <= i < num_features for i in take), (
            f"index out of range (0..{num_features - 1}): {take}"
        )
        return take, max(take)

    def forward_intermediates(
        self,
        x: jax.Array,
        indices: Optional[Union[int, Tuple[int, ...]]] = None,
        stop_early: bool = False,
        intermediates_only: bool = False,
        training: bool = False,
    ) -> Union[List[jax.Array], Tuple[jax.Array, List[jax.Array]]]:
        take_indices, max_index = self._feature_take_indices(indices)
        take_set = set(take_indices)
        intermediates: List[jax.Array] = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if 0 in take_set:
            intermediates.append(x)
        x = self.maxpool(x)

        for i, layer in enumerate(self.stage_modules):
            stage_idx = i + 1
            x = layer(x)
            if stage_idx in take_set:
                intermediates.append(x)
            if stop_early and stage_idx >= max_index:
                remaining = sum(1 for t in take_indices if t > stage_idx)
                intermediates.extend([x] * remaining)
                break

        if intermediates_only:
            return intermediates
        return x, intermediates

    def prune_intermediate_layers(
        self,
        indices: Union[int, Tuple[int, ...]] = 1,
        prune_head: bool = True,
    ) -> Tuple[int, ...]:
        take_indices, max_index = self._feature_take_indices(indices)
        for i in range(max_index + 1, len(self.stage_modules)):
            self.stage_modules[i] = nnx.Sequential(nnx.identity)
        if prune_head:
            self.global_pool = nnx.identity
            self.fc = nnx.identity
        return take_indices

    def forward_features(self, x: jax.Array, training: bool = False) -> jax.Array:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        for layer in self.stage_modules:
            x = layer(x, training=training)
        return x

    def forward_head(
        self, x: jax.Array, pre_logits: bool = False, training: bool = False
    ) -> jax.Array:
        x = self.global_pool(x)
        x = x.reshape(x.shape[0], -1)
        if pre_logits:
            return x
        if self.head_drop is not None:
            x = self.head_drop(x, deterministic=not training)
        x = self.fc(x)
        return x

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        x = self.forward_features(x, training=training)
        x = self.forward_head(x, training=training)
        return x


# Config helpers
def _cfg(url: str = "", **kwargs) -> Dict[str, Any]:
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (224, 224, 3),
        "pool_size": (7, 7),
        "crop_pct": 0.875,
        "interpolation": "bilinear",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "conv1",
        "classifier": "fc",
        "license": "apache-2.0",
        **kwargs,
    }


def _tcfg(url: str = "", **kwargs) -> Dict[str, Any]:
    return _cfg(
        url=url,
        **dict(
            {
                "interpolation": "bicubic",
                "pool_size": (8, 8),
                "input_size": (256, 256, 3),
                "crop_pct": 0.94,
                "test_input_size": (320, 320, 3),
                "test_crop_pct": 1.0,
            },
            **kwargs,
        ),
    )


def _ttcfg(url: str = "", **kwargs) -> Dict[str, Any]:
    return _cfg(
        url=url,
        **dict(
            {
                "interpolation": "bicubic",
                "pool_size": (7, 7),
                "input_size": (224, 224, 3),
                "crop_pct": 0.95,
                "test_input_size": (288, 288, 3),
                "test_crop_pct": 1.0,
            },
            **kwargs,
        ),
    )


def _rcfg(url: str = "", **kwargs) -> Dict[str, Any]:
    return _cfg(
        url=url,
        **dict(
            {
                "interpolation": "bicubic",
                "crop_pct": 0.95,
                "test_input_size": (288, 288, 3),
                "test_crop_pct": 1.0,
                "origin_url": "https://github.com/Xrenya/JaxNN",
                "paper_ids": "arXiv:2110.00476",
            },
            **kwargs,
        ),
    )


def _r3cfg(url: str = "", **kwargs) -> Dict[str, Any]:
    return _cfg(
        url=url,
        **dict(
            {
                "interpolation": "bicubic",
                "input_size": (160, 160, 3),
                "pool_size": (5, 5),
                "crop_pct": 0.91,
                "test_input_size": (224, 224, 3),
                "test_crop_pct": 0.95,
                "origin_url": "https://github.com/Xrenya/JaxNN",
                "paper_ids": "arXiv:2110.00476",
            },
            **kwargs,
        ),
    )


def _ra4cfg(url: str = "", **kwargs) -> Dict[str, Any]:
    """Config for the ra4/e3600 training series.

    These models use mean=(0.5, 0.5, 0.5) / std=(0.5, 0.5, 0.5) and crop_pct=0.9.
    Using standard ImageNet stats instead is the most common cause of output
    mismatch when comparing against timm for this checkpoint family.
    """
    return _rcfg(
        url=url,
        **dict(
            {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5), "crop_pct": 0.9}, **kwargs
        ),
    )


def _gcfg(url: str = "", **kwargs) -> Dict[str, Any]:
    return _cfg(
        url=url,
        **dict(
            {
                "interpolation": "bicubic",
            },
            **kwargs,
        ),
    )


def _create_resnet(variant: str, pretrained: bool = False, **kwargs) -> ResNet:
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


# Default configs - all JaxNN ResNet variants
# NOTE: input_size uses HWC (Flax convention), not CHW (PyTorch)
# TODO: add the remaining weights
default_cfgs = generate_default_cfgs(
    {
        # ResNet (BasicBlock)
        "resnet10t.c3_in1k": _ttcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet10t.c3_in1k",
            input_size=(176, 176, 3), pool_size=(6, 6), test_crop_pct=0.95, test_input_size=(224, 224, 3),
            first_conv="conv1.0",
        ),
        "resnet14t.c3_in1k": _ttcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet14t.c3_in1k",
            input_size=(176, 176, 3), pool_size=(6, 6), test_crop_pct=0.95, test_input_size=(224, 224, 3),
            first_conv="conv1.0",
        ),
        "resnet18.a1_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet18.a1_in1k",
        ),
        "resnet18.a2_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet18.a2_in1k",
        ),
        "resnet18.a3_in1k": _r3cfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet18.a3_in1k",
        ),
        "resnet18.fb_ssl_yfcc100m_ft_in1k": _cfg(
            hf_hub_id="JaxNN/",
            url="JaxNN/resnet18.fb_ssl_yfcc100m_ft_in1k",
            license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'
        ),
        "resnet18.fb_swsl_ig1b_ft_in1k": _cfg(
            hf_hub_id="JaxNN/",
            url="JaxNN/resnet18.fb_swsl_ig1b_ft_in1k",
            license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'
        ),
        "resnet18.gluon_in1k": _gcfg(
            hf_hub_id="JaxNN/",
            url="JaxNN/resnet18.gluon_in1k",
        ),
        "resnet18.tv_in1k": _cfg(
            hf_hub_id="JaxNN/",
            url="JaxNN/resnet18.tv_in1k",
            license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'
        ),
        "resnet18d.ra4_e3600_r224_in1k": _ra4cfg(
            hf_hub_id="JaxNN/",
            url="JaxNN/ra4_e3600_r224_in1k.tv_in1k",
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=0.9, first_conv='conv1.0'
        ),
        "resnet18d.ra2_in1k": _ttcfg(
            hf_hub_id="JaxNN/",
            url="JaxNN/resnet18d.ra2_in1k",
            first_conv="conv1.0",
        ),
        "resnet34.a1_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet34.a1_in1k",
        ),
        "resnet34.a2_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet34.a2_in1k",
        ),
        "resnet34.a3_in1k": _r3cfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet34.a3_in1k",
            crop_pct=0.95,
        ),
        "resnet34.bt_in1k": _ttcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet34.bt_in1k",
        ),
        "resnet34.gluon_in1k": _gcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet34.gluon_in1k",
        ),
        "resnet34.tv_in1k": _cfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet34.tv_in1k",
            license='bsd-3-clause', origin_url='https://github.com/pytorch/vision',
        ),
        "resnet34d.ra2_in1k": _ttcfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            url="https://huggingface.co/JaxNN/resnet34.ra2_in1k",
        ),
        "resnet34.ra4_e3600_r224_in1k": _ra4cfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet34.ra4_e3600_r224_in1k",
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=0.9,
        ),
        # ResNet (Bottleneck)
        "resnet26.bt_in1k": _ttcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet26.bt_in1k",
        ),
        "resnet26d.bt_in1k": _ttcfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            url="https://huggingface.co/JaxNN/resnet26d.bt_in1k",
        ),
        "resnet26t.ra2_in1k": _ttcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet26t.ra2_in1k",
            first_conv='conv1.0', input_size=(256, 256, 3), pool_size=(8, 8),
            crop_pct=0.94, test_input_size=(320, 320, 3), test_crop_pct=1.0,
        ),
        "resnet50.a1_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.a1_in1k",
        ),
        "resnet50.a1h_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            input_size=(176, 176, 3), pool_size=(6, 6), crop_pct=0.9, test_input_size=(224, 224, 3), test_crop_pct=1.0
        ),
        "resnet50.a2_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.a2_in1k",
        ),
        "resnet50.a3_in1k": _r3cfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.a3_in1k",
        ),
        "resnet50.b1k_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.b1k_in1k",
        ),
        "resnet50.b2k_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.b2k_in1k",
        ),
        "resnet50.bt_in1k": _ttcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.bt_in1k",
        ),
        "resnet50.c1_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.c1_in1k",
        ),
        "resnet50.c2_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.c2_in1k",
        ),
        "resnet50.d_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.d_in1k",
        ),
        "resnet50.ram_in1k": _ttcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.ram_in1k",
        ),
        "resnet50.am_in1k": _tcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.am_in1k",
        ),
        "resnet50.ra_in1k": _ttcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.ra_in1k",
        ),
        "resnet50.fb_ssl_yfcc100m_ft_in1k": _cfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.fb_ssl_yfcc100m_ft_in1k",
            license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models',
        ),
        "resnet50.fb_swsl_ig1b_ft_in1k": _cfg(
            hf_hub_id="JaxNN/",
            license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'
        ),
        "resnet50.gluon_in1k": _gcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.gluon_in1k",
        ),
        "resnet50.tv_in1k": _cfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.tv_in1k",
            license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'
        ),
        "resnet50.tv2_in1k": _cfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50.tv2_in1k",
            input_size=(176, 176, 3),
            pool_size=(6, 6),
            test_input_size=(224, 224, 3),
            test_crop_pct=0.965,
            interpolation="bilinear",
            crop_pct=0.875,
        ),
        "resnet50c.gluon_in1k": _gcfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            url="https://huggingface.co/JaxNN/resnet50c.gluon_in1k",
        ),
        "resnet50d.a1_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            url="https://huggingface.co/JaxNN/resnet50d.a1_in1k",
        ),
        "resnet50d.a2_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            url="https://huggingface.co/JaxNN/resnet50d.a2_in1k",
        ),
        "resnet50d.a3_in1k": _r3cfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            url="https://huggingface.co/JaxNN/resnet50d.a3_in1k",
        ),
        "resnet50d.ra2_in1k": _ttcfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            url="https://huggingface.co/JaxNN/resnet50d.ra2_in1k",
        ),
        "resnet50d.ra4_e3600_r224_in1k": _ra4cfg(
            hf_hub_id="JaxNN/", first_conv="conv1.0",
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
            crop_pct=0.95, test_input_size=(288, 288, 3), test_crop_pct=1.0,
        ),
        "resnet50s.gluon_in1k": _gcfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            url="https://huggingface.co/JaxNN/resnet50s.gluon_in1k",
        ),
        "resnet50d.gluon_in1k": _gcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50d.gluon_in1k",
            first_conv="conv1.0",
        ),
        "resnet50t.untrained": _ttcfg(first_conv="conv1.0"),
        "resnet101.a1_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet101.a1_in1k",
        ),
        "resnet101.a1h_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet101.a1h_in1k",
        ),
        "resnet101.a2_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet101.a2_in1k",
        ),
        "resnet101.a3_in1k": _r3cfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet101.a3_in1k",
        ),
        "resnet101.tv_in1k": _cfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet101.tv_in1k",
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "resnet101.tv2_in1k": _cfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet101.tv2_in1k",
            input_size=(176, 176, 3),
            pool_size=(6, 6),
            test_input_size=(224, 224, 3),
            test_crop_pct=0.965,
            interpolation="bilinear",
            crop_pct=0.875,
        ),
        "resnet101.gluon_in1k": _gcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet101.gluon_in1k",
        ),
        "resnet101c.gluon_in1k": _gcfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            url="https://huggingface.co/JaxNN/resnet101c.gluon_in1k",
        ),
        "resnet101d.gluon_in1k": _gcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet101d.gluon_in1k",
            first_conv="conv1.0",
        ),
        "resnet101d.ra2_in1k": _ttcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet101d.ra2_in1k",
            first_conv="conv1.0", input_size=(256, 256, 3), pool_size=(8, 8), crop_pct=0.95,
            test_crop_pct=1.0, test_input_size=(320, 320, 3)
        ),
        "resnet101s.gluon_in1k": _gcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet101s.gluon_in1k",
            first_conv="conv1.0",
        ),
        "resnet152.a1_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet152.a1_in1k",
        ),
        "resnet152.a1h_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet152.a1h_in1k",
        ),
        "resnet152.a2_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet152.a2_in1k",
        ),
        "resnet152.a3_in1k": _r3cfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet152.a3_in1k",
        ),
        "resnet152.gluon_in1k": _gcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet152.gluon_in1k",
        ),
        "resnet152.tv2_in1k": _cfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet152.tv2_in1k",
            input_size=(176, 176, 3),
            pool_size=(6, 6),
            test_input_size=(224, 224, 3),
            test_crop_pct=0.965,
            interpolation="bilinear",
            crop_pct=0.875,
        ),
        "resnet152.tv_in1k": _cfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet152.tv_in1k",
            license="bsd-3-clause",
            origin_url="https://github.com/pytorch/vision",
        ),
        "resnet152c.gluon_in1k": _gcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet152c.gluon_in1k",
            first_conv="conv1.0",
        ),
        "resnet152d.gluon_in1k": _gcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet152d.gluon_in1k",
            first_conv="conv1.0",
        ),
        "resnet152d.ra2_in1k": _rcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet152d.ra2_in1k",
            first_conv="conv1.0",
            input_size=(256, 256, 3),
            pool_size=(8, 8),
            test_input_size=(320, 320, 3),
            test_crop_pct=1.0,
        ),
        "resnet152s.gluon_in1k": _gcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet152s.gluon_in1k",
            first_conv="conv1.0",
        ),
        "resnet200.untrained": _ttcfg(),
        "resnet200d.ra2_in1k": _ttcfg(
            hf_hub_id="JaxNN/", first_conv="conv1.0",
            input_size=(256, 256, 3), pool_size=(8, 8), crop_pct=0.95,
            test_crop_pct=1.0, test_input_size=(320, 320, 3)
        ),
        # Wide ResNet
        "wide_resnet50_2.racm_in1k": _ttcfg(hf_hub_id="JaxNN/"),
        "wide_resnet50_2.tv2_in1k": _cfg(
            hf_hub_id="JaxNN/",
            input_size=(176, 176, 3),
            pool_size=(6, 6),
            test_input_size=(224, 224, 3),
            test_crop_pct=0.965,
            interpolation="bilinear",
            crop_pct=0.875,
        ),
        "wide_resnet101_2.tv2_in1k": _cfg(
            hf_hub_id="JaxNN/",
            input_size=(176, 176, 3),
            pool_size=(6, 6),
            test_input_size=(224, 224, 3),
            test_crop_pct=0.965,
            interpolation="bilinear",
            crop_pct=0.875,
        ),
        # ResNeXt
        "resnet50_gn.a1h_in1k": _ttcfg(
            hf_hub_id="JaxNN/",
            url="https://huggingface.co/JaxNN/resnet50_gn.a1h_in1k",
            crop_pct=0.94,
        ),
        "resnext50_32x4d.a1_in1k": _rcfg(hf_hub_id="JaxNN/"),
        "resnext50_32x4d.a1h_in1k": _rcfg(hf_hub_id="JaxNN/"),
        "resnext50_32x4d.a2_in1k": _rcfg(hf_hub_id="JaxNN/"),
        "resnext50_32x4d.a3_in1k": _r3cfg(hf_hub_id="JaxNN/"),
        "resnext50_32x4d.fb_ssl_yfcc100m_ft_in1k": _cfg(hf_hub_id="JaxNN/"),
        "resnext50_32x4d.fb_swsl_ig1b_ft_in1k": _cfg(hf_hub_id="JaxNN/"),
        "resnext50_32x4d.gluon_in1k": _gcfg(hf_hub_id="JaxNN/"),
        "resnext50_32x4d.ra_in1k": _ttcfg(hf_hub_id="JaxNN/"),
        "resnext50_32x4d.tv_in1k": _cfg(hf_hub_id="JaxNN/"),
        "resnext50_32x4d.tv2_in1k": _cfg(
            hf_hub_id="JaxNN/",
            input_size=(176, 176, 3),
            pool_size=(6, 6),
            test_input_size=(224, 224, 3),
            test_crop_pct=0.965,
            interpolation="bilinear",
            crop_pct=0.875,
        ),
        "resnext50d_32x4d.bt_in1k": _ttcfg(hf_hub_id="JaxNN/", first_conv="conv1.0"),
        "resnext101_32x4d.fb_ssl_yfcc100m_ft_in1k": _cfg(hf_hub_id="JaxNN/"),
        "resnext101_32x4d.fb_swsl_ig1b_ft_in1k": _cfg(hf_hub_id="JaxNN/"),
        "resnext101_32x4d.gluon_in1k": _gcfg(hf_hub_id="JaxNN/"),
        "resnext101_32x8d.fb_ssl_yfcc100m_ft_in1k": _cfg(hf_hub_id="JaxNN/"),
        "resnext101_32x8d.fb_swsl_ig1b_ft_in1k": _cfg(hf_hub_id="JaxNN/"),
        "resnext101_32x8d.tv_in1k": _cfg(hf_hub_id="JaxNN/"),
        "resnext101_32x8d.tv2_in1k": _cfg(
            hf_hub_id="JaxNN/",
            input_size=(176, 176, 3),
            pool_size=(6, 6),
            test_input_size=(224, 224, 3),
            test_crop_pct=0.965,
            interpolation="bilinear",
            crop_pct=0.875,
        ),
        "resnext101_64x4d.gluon_in1k": _gcfg(hf_hub_id="JaxNN/"),
        "resnext101_64x4d.tv_in1k": _cfg(hf_hub_id="JaxNN/"),
        # Anti-aliased ResNets
        "resnetblur18.untrained": _cfg(),
        "resnetblur50.bt_in1k": _ttcfg(hf_hub_id="JaxNN/"),
        "resnetblur50d.untrained": _ttcfg(first_conv="conv1.0"),
        "resnetblur101d.untrained": _ttcfg(first_conv="conv1.0"),
        "resnetaa34d.ra2_in1k": _ttcfg(hf_hub_id="JaxNN/", first_conv="conv1.0"),
        "resnetaa50.a1h_in1k": _rcfg(hf_hub_id="JaxNN/"),
        "resnetaa50d.ra2_in1k": _ttcfg(hf_hub_id="JaxNN/", first_conv="conv1.0"),
        "resnetaa50d.d_in12k": _ttcfg(
            hf_hub_id="JaxNN/", first_conv="conv1.0", num_classes=11821
        ),
        "resnetaa50d.d_in12k_ft_in1k": _ttcfg(hf_hub_id="JaxNN/", first_conv="conv1.0"),
        "resnetaa101d.sw_in12k": _ttcfg(
            hf_hub_id="JaxNN/", first_conv="conv1.0", num_classes=11821
        ),
        "resnetaa101d.sw_in12k_ft_in1k": _ttcfg(
            hf_hub_id="JaxNN/", first_conv="conv1.0"
        ),
        # SE-ResNet
        "seresnet18.untrained": _cfg(),
        "seresnet34.untrained": _cfg(),
        "seresnet50.a1_in1k": _rcfg(hf_hub_id="JaxNN/"),
        "seresnet50.a2_in1k": _rcfg(hf_hub_id="JaxNN/"),
        "seresnet50.a3_in1k": _r3cfg(hf_hub_id="JaxNN/"),
        "seresnet50.ra2_in1k": _ttcfg(hf_hub_id="JaxNN/"),
        "seresnet50t.untrained": _ttcfg(first_conv="conv1.0"),
        "seresnet101.untrained": _ttcfg(),
        "seresnet152.untrained": _ttcfg(),
        "seresnet152d.ra2_in1k": _ttcfg(hf_hub_id="JaxNN/", first_conv="conv1.0"),
        "seresnet200d.untrained": _ttcfg(first_conv="conv1.0"),
        "seresnet269d.untrained": _ttcfg(first_conv="conv1.0"),
        "seresnetaa50d.untrained": _ttcfg(first_conv="conv1.0"),
        # SE-ResNeXt
        "seresnext26d_32x4d.bt_in1k": _ttcfg(hf_hub_id="JaxNN/", first_conv="conv1.0"),
        "seresnext26t_32x4d.bt_in1k": _ttcfg(hf_hub_id="JaxNN/", first_conv="conv1.0"),
        "seresnext50_32x4d.racm_in1k": _ttcfg(hf_hub_id="JaxNN/"),
        "seresnext50_32x4d.gluon_in1k": _gcfg(hf_hub_id="JaxNN/"),
        "seresnext101_32x4d.gluon_in1k": _gcfg(hf_hub_id="JaxNN/"),
        "seresnext101_32x8d.ah_in1k": _rcfg(hf_hub_id="JaxNN/"),
        "seresnext101d_32x8d.ah_in1k": _rcfg(hf_hub_id="JaxNN/", first_conv="conv1.0"),
        "seresnext101_64x4d.gluon_in1k": _gcfg(hf_hub_id="JaxNN/"),
        "seresnextaa101d_32x8d.ah_in1k": _rcfg(
            hf_hub_id="JaxNN/", first_conv="conv1.0"
        ),
        "seresnextaa101d_32x8d.sw_in12k": _rcfg(
            hf_hub_id="JaxNN/", first_conv="conv1.0", num_classes=11821
        ),
        "seresnextaa101d_32x8d.sw_in12k_ft_in1k": _rcfg(
            hf_hub_id="JaxNN/", first_conv="conv1.0"
        ),
        "seresnextaa101d_32x8d.sw_in12k_ft_in1k_288": _rcfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            input_size=(288, 288, 3),
            pool_size=(9, 9),
            test_input_size=(320, 320, 3),
            crop_pct=1.0,
        ),
        "seresnextaa201d_32x8d.untrained": _ttcfg(first_conv="conv1.0"),
        "senet154.gluon_in1k": _gcfg(hf_hub_id="JaxNN/", first_conv="conv1.0"),
        # ECA-ResNet
        "ecaresnet26t.ra2_in1k": _ttcfg(hf_hub_id="JaxNN/", first_conv="conv1.0"),
        "ecaresnet50d.miil_in1k": _cfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            interpolation="bilinear",
            crop_pct=0.875,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        ),
        "ecaresnet50d_pruned.miil_in1k": _cfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            interpolation="bilinear",
            crop_pct=0.875,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        ),
        "ecaresnet50t.a1_in1k": _rcfg(hf_hub_id="JaxNN/", first_conv="conv1.0"),
        "ecaresnet50t.a2_in1k": _rcfg(hf_hub_id="JaxNN/", first_conv="conv1.0"),
        "ecaresnet50t.a3_in1k": _r3cfg(hf_hub_id="JaxNN/", first_conv="conv1.0"),
        "ecaresnet50t.ra2_in1k": _ttcfg(hf_hub_id="JaxNN/", first_conv="conv1.0"),
        "ecaresnetlight.miil_in1k": _cfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            interpolation="bilinear",
            crop_pct=0.875,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        ),
        "ecaresnet101d.miil_in1k": _cfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            interpolation="bilinear",
            crop_pct=0.875,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        ),
        "ecaresnet101d_pruned.miil_in1k": _cfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            interpolation="bilinear",
            crop_pct=0.875,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        ),
        "ecaresnet200d.untrained": _ttcfg(first_conv="conv1.0"),
        "ecaresnet269d.ra2_in1k": _ttcfg(hf_hub_id="JaxNN/", first_conv="conv1.0"),
        "ecaresnext26t_32x4d.untrained": _ttcfg(first_conv="conv1.0"),
        "ecaresnext50t_32x4d.untrained": _ttcfg(first_conv="conv1.0"),
        # ResNet-RS
        "resnetrs50.tf_in1k": _cfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            input_size=(160, 160, 3),
            pool_size=(5, 5),
            crop_pct=0.91,
            test_input_size=(224, 224, 3),
            interpolation="bicubic",
        ),
        "resnetrs101.tf_in1k": _cfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            input_size=(192, 192, 3),
            pool_size=(6, 6),
            crop_pct=0.94,
            test_input_size=(288, 288, 3),
            interpolation="bicubic",
        ),
        "resnetrs152.tf_in1k": _cfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            input_size=(256, 256, 3),
            pool_size=(8, 8),
            crop_pct=1.0,
            test_input_size=(320, 320, 3),
            interpolation="bicubic",
        ),
        "resnetrs200.tf_in1k": _cfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            input_size=(256, 256, 3),
            pool_size=(8, 8),
            crop_pct=1.0,
            test_input_size=(320, 320, 3),
            interpolation="bicubic",
        ),
        "resnetrs270.tf_in1k": _cfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            input_size=(256, 256, 3),
            pool_size=(8, 8),
            crop_pct=1.0,
            test_input_size=(352, 352, 3),
            interpolation="bicubic",
        ),
        "resnetrs350.tf_in1k": _cfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            input_size=(288, 288, 3),
            pool_size=(9, 9),
            crop_pct=1.0,
            test_input_size=(384, 384, 3),
            interpolation="bicubic",
        ),
        "resnetrs420.tf_in1k": _cfg(
            hf_hub_id="JaxNN/",
            first_conv="conv1.0",
            input_size=(320, 320, 3),
            pool_size=(10, 10),
            crop_pct=1.0,
            test_input_size=(416, 416, 3),
            interpolation="bicubic",
        ),
    }
)


# Model registration functions
# BasicBlock models
@register_model
def resnet10t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-10-T model."""
    model_args = dict(
        block=BasicBlock,
        layers=(1, 1, 1, 1),
        stem_width=32,
        stem_type="deep_tiered",
        avg_down=True,
    )
    return _create_resnet("resnet10t", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet14t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-14-T model."""
    model_args = dict(
        block=Bottleneck,
        layers=(1, 1, 1, 1),
        stem_width=32,
        stem_type="deep_tieredn",
        avg_down=True,
    )
    return _create_resnet("resnet14t", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet18(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-18 model."""
    model_args = dict(block=BasicBlock, layers=(2, 2, 2, 2))
    return _create_resnet("resnet18", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet18d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-18-D model."""
    model_args = dict(
        block=BasicBlock,
        layers=(2, 2, 2, 2),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet("resnet18d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet34(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-34 model."""
    model_args = dict(block=BasicBlock, layers=(3, 4, 6, 3))
    return _create_resnet("resnet34", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet34d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-34-D model."""
    model_args = dict(
        block=BasicBlock,
        layers=(3, 4, 6, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet("resnet34d", pretrained, **dict(model_args, **kwargs))


# Bottleneck models


@register_model
def resnet26(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-26 model."""
    model_args = dict(block=Bottleneck, layers=(2, 2, 2, 2))
    return _create_resnet("resnet26", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet26d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-26-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=(2, 2, 2, 2),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet("resnet26d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet26t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-26-T model."""
    model_args = dict(
        block=Bottleneck,
        layers=(2, 2, 2, 2),
        stem_width=32,
        stem_type="deep_tiered",
        avg_down=True,
    )
    return _create_resnet("resnet26t", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model."""
    model_args = dict(block=Bottleneck, layers=(3, 4, 6, 3))
    return _create_resnet("resnet50", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50c(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-C model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 6, 3), stem_width=32, stem_type="deep"
    )
    return _create_resnet("resnet50c", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet("resnet50d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50s(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-S model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 6, 3), stem_width=64, stem_type="deep"
    )
    return _create_resnet("resnet50s", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-T model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        stem_width=32,
        stem_type="deep_tiered",
        avg_down=True,
    )
    return _create_resnet("resnet50t", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101 model."""
    model_args = dict(block=Bottleneck, layers=(3, 4, 23, 3))
    return _create_resnet("resnet101", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101c(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-C model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 23, 3), stem_width=32, stem_type="deep"
    )
    return _create_resnet("resnet101c", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 23, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet("resnet101d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101s(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-S model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 23, 3), stem_width=64, stem_type="deep"
    )
    return _create_resnet("resnet101s", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-152 model."""
    model_args = dict(block=Bottleneck, layers=(3, 8, 36, 3))
    return _create_resnet("resnet152", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152c(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-152-C model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 8, 36, 3), stem_width=32, stem_type="deep"
    )
    return _create_resnet("resnet152c", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-152-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 8, 36, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet("resnet152d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152s(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-152-S model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 8, 36, 3), stem_width=64, stem_type="deep"
    )
    return _create_resnet("resnet152s", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet200(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-200 model."""
    model_args = dict(block=Bottleneck, layers=(3, 24, 36, 3))
    return _create_resnet("resnet200", pretrained, **dict(model_args, **kwargs))


@register_model
def resnet200d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-200-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 24, 36, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet("resnet200d", pretrained, **dict(model_args, **kwargs))


# Wide ResNet


@register_model
def wide_resnet50_2(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a Wide ResNet-50-2 model."""
    model_args = dict(block=Bottleneck, layers=(3, 4, 6, 3), base_width=128)
    return _create_resnet("wide_resnet50_2", pretrained, **dict(model_args, **kwargs))


@register_model
def wide_resnet101_2(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a Wide ResNet-101-2 model."""
    model_args = dict(block=Bottleneck, layers=(3, 4, 23, 3), base_width=128)
    return _create_resnet("wide_resnet101_2", pretrained, **dict(model_args, **kwargs))


# ResNeXt
@register_model
def resnet50_gn(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model w/ GroupNorm"""
    model_args = dict(block=Bottleneck, layers=(3, 4, 6, 3), norm_layer="groupnorm")
    return _create_resnet("resnet50_gn", pretrained, **dict(model_args, **kwargs))


@register_model
def resnext50_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-50 32x4d model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 6, 3), cardinality=32, base_width=4
    )
    return _create_resnet("resnext50_32x4d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnext50d_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-50-D 32x4d model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        cardinality=32,
        base_width=4,
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet("resnext50d_32x4d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x4d model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 23, 3), cardinality=32, base_width=4
    )
    return _create_resnet("resnext101_32x4d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x8d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x8d model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 23, 3), cardinality=32, base_width=8
    )
    return _create_resnet("resnext101_32x8d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x16d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x16d model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 23, 3), cardinality=32, base_width=16
    )
    return _create_resnet("resnext101_32x16d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x32d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x32d model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 23, 3), cardinality=32, base_width=32
    )
    return _create_resnet("resnext101_32x32d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_64x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 64x4d model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 23, 3), cardinality=64, base_width=4
    )
    return _create_resnet("resnext101_64x4d", pretrained, **dict(model_args, **kwargs))


# Anti-aliased ResNets


@register_model
def resnetblur18(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-18 with anti-aliasing (blur pool)."""
    model_args = dict(
        block=BasicBlock, layers=(2, 2, 2, 2), aa_layer=partial(nnx.avg_pool)
    )
    return _create_resnet("resnetblur18", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetblur50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 with anti-aliasing (blur pool)."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 6, 3), aa_layer=partial(nnx.avg_pool)
    )
    return _create_resnet("resnetblur50", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetblur50d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D with anti-aliasing (blur pool)."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        aa_layer=partial(nnx.avg_pool),
    )
    return _create_resnet("resnetblur50d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetblur101d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D with anti-aliasing (blur pool)."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 23, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        aa_layer=partial(nnx.avg_pool),
    )
    return _create_resnet("resnetblur101d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa34d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-34-D with anti-aliasing (avg pool)."""
    model_args = dict(
        block=BasicBlock,
        layers=(3, 4, 6, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        aa_layer=partial(nnx.avg_pool),
    )
    return _create_resnet("resnetaa34d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 with anti-aliasing (avg pool)."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 6, 3), aa_layer=partial(nnx.avg_pool)
    )
    return _create_resnet("resnetaa50", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa50d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D with anti-aliasing (avg pool)."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        aa_layer=partial(nnx.avg_pool),
    )
    return _create_resnet("resnetaa50d", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa101d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D with anti-aliasing (avg pool)."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 23, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        aa_layer=partial(nnx.avg_pool),
    )
    return _create_resnet("resnetaa101d", pretrained, **dict(model_args, **kwargs))


# SE-ResNet (requires attn_layer support)


@register_model
def seresnet18(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNet-18 model."""
    model_args = dict(block=BasicBlock, layers=(2, 2, 2, 2))
    return _create_resnet("seresnet18", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet34(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNet-34 model."""
    model_args = dict(block=BasicBlock, layers=(3, 4, 6, 3))
    return _create_resnet("seresnet34", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNet-50 model."""
    model_args = dict(block=Bottleneck, layers=(3, 4, 6, 3))
    return _create_resnet("seresnet50", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet50t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNet-50-T model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        stem_width=32,
        stem_type="deep_tiered",
        avg_down=True,
    )
    return _create_resnet("seresnet50t", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet101(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNet-101 model."""
    model_args = dict(block=Bottleneck, layers=(3, 4, 23, 3))
    return _create_resnet("seresnet101", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet152(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNet-152 model."""
    model_args = dict(block=Bottleneck, layers=(3, 8, 36, 3))
    return _create_resnet("seresnet152", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet152d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNet-152-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 8, 36, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet("seresnet152d", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet200d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNet-200-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 24, 36, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet("seresnet200d", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet269d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNet-269-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 30, 48, 8),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet("seresnet269d", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnetaa50d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNet-AA-50-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        aa_layer=partial(nnx.avg_pool),
    )
    return _create_resnet("seresnetaa50d", pretrained, **dict(model_args, **kwargs))


# SE-ResNeXt


@register_model
def seresnext26d_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNeXt-26-D 32x4d model."""
    model_args = dict(
        block=Bottleneck,
        layers=(2, 2, 2, 2),
        cardinality=32,
        base_width=4,
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet(
        "seresnext26d_32x4d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def seresnext26t_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNeXt-26-T 32x4d model."""
    model_args = dict(
        block=Bottleneck,
        layers=(2, 2, 2, 2),
        cardinality=32,
        base_width=4,
        stem_width=32,
        stem_type="deep_tiered",
        avg_down=True,
    )
    return _create_resnet(
        "seresnext26t_32x4d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def seresnext50_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNeXt-50 32x4d model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 6, 3), cardinality=32, base_width=4
    )
    return _create_resnet("seresnext50_32x4d", pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext101_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNeXt-101 32x4d model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 23, 3), cardinality=32, base_width=4
    )
    return _create_resnet(
        "seresnext101_32x4d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def seresnext101_32x8d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNeXt-101 32x8d model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 23, 3), cardinality=32, base_width=8
    )
    return _create_resnet(
        "seresnext101_32x8d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def seresnext101d_32x8d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNeXt-101-D 32x8d model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 23, 3),
        cardinality=32,
        base_width=8,
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet(
        "seresnext101d_32x8d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def seresnext101_64x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNeXt-101 64x4d model."""
    model_args = dict(
        block=Bottleneck, layers=(3, 4, 23, 3), cardinality=64, base_width=4
    )
    return _create_resnet(
        "seresnext101_64x4d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def seresnextaa101d_32x8d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNeXt-AA-101-D 32x8d model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 23, 3),
        cardinality=32,
        base_width=8,
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        aa_layer=partial(nnx.avg_pool),
    )
    return _create_resnet(
        "seresnextaa101d_32x8d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def seresnextaa201d_32x8d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNeXt-AA-201-D 32x8d model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 24, 36, 4),
        cardinality=32,
        base_width=8,
        stem_width=64,
        stem_type="deep",
        avg_down=True,
        aa_layer=partial(nnx.avg_pool),
    )
    return _create_resnet(
        "seresnextaa201d_32x8d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def senet154(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SENet-154 model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 8, 36, 3),
        cardinality=64,
        base_width=4,
        stem_type="deep",
        down_kernel_size=3,
        block_reduce_first=2,
    )
    return _create_resnet("senet154", pretrained, **dict(model_args, **kwargs))


# ECA-ResNet (requires attn_layer support)


@register_model
def ecaresnet26t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNet-26-T model."""
    model_args = dict(
        block=Bottleneck,
        layers=(2, 2, 2, 2),
        stem_width=32,
        stem_type="deep_tiered",
        avg_down=True,
    )
    return _create_resnet("ecaresnet26t", pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet50d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNet-50-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet("ecaresnet50d", pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet50d_pruned(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a pruned ECA-ResNet-50-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet(
        "ecaresnet50d_pruned", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def ecaresnet50t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNet-50-T model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        stem_width=32,
        stem_type="deep_tiered",
        avg_down=True,
    )
    return _create_resnet("ecaresnet50t", pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnetlight(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNet-Light model."""
    model_args = dict(
        block=Bottleneck, layers=(1, 1, 11, 3), stem_width=32, avg_down=True
    )
    return _create_resnet("ecaresnetlight", pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet101d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNet-101-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 23, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet("ecaresnet101d", pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet101d_pruned(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a pruned ECA-ResNet-101-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 23, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet(
        "ecaresnet101d_pruned", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def ecaresnet200d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNet-200-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 24, 36, 3),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet("ecaresnet200d", pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet269d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNet-269-D model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 30, 48, 8),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    return _create_resnet("ecaresnet269d", pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnext26t_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNeXt-26-T 32x4d model."""
    model_args = dict(
        block=Bottleneck,
        layers=(2, 2, 2, 2),
        cardinality=32,
        base_width=4,
        stem_width=32,
        stem_type="deep_tiered",
        avg_down=True,
    )
    return _create_resnet(
        "ecaresnext26t_32x4d", pretrained, **dict(model_args, **kwargs)
    )


@register_model
def ecaresnext50t_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNeXt-50-T 32x4d model."""
    model_args = dict(
        block=Bottleneck,
        layers=(2, 2, 2, 2),
        cardinality=32,
        base_width=4,
        stem_width=32,
        stem_type="deep_tiered",
        avg_down=True,
    )
    return _create_resnet(
        "ecaresnext50t_32x4d", pretrained, **dict(model_args, **kwargs)
    )


# ResNet-RS


@register_model
def resnetrs50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-50 model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        stem_width=32,
        stem_type="deep",
        replace_stem_pool=True,
        avg_down=True,
    )
    return _create_resnet("resnetrs50", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs101(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-101 model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 4, 23, 3),
        stem_width=32,
        stem_type="deep",
        replace_stem_pool=True,
        avg_down=True,
    )
    return _create_resnet("resnetrs101", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs152(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-152 model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 8, 36, 3),
        stem_width=32,
        stem_type="deep",
        replace_stem_pool=True,
        avg_down=True,
    )
    return _create_resnet("resnetrs152", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs200(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-200 model."""
    model_args = dict(
        block=Bottleneck,
        layers=(3, 24, 36, 3),
        stem_width=32,
        stem_type="deep",
        replace_stem_pool=True,
        avg_down=True,
    )
    return _create_resnet("resnetrs200", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs270(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-270 model."""
    model_args = dict(
        block=Bottleneck,
        layers=(4, 29, 53, 4),
        stem_width=32,
        stem_type="deep",
        replace_stem_pool=True,
        avg_down=True,
    )
    return _create_resnet("resnetrs270", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs350(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-350 model."""
    model_args = dict(
        block=Bottleneck,
        layers=(4, 36, 72, 4),
        stem_width=32,
        stem_type="deep",
        replace_stem_pool=True,
        avg_down=True,
    )
    return _create_resnet("resnetrs350", pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs420(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-420 model."""
    model_args = dict(
        block=Bottleneck,
        layers=(4, 44, 87, 4),
        stem_width=32,
        stem_type="deep",
        replace_stem_pool=True,
        avg_down=True,
    )
    return _create_resnet("resnetrs420", pretrained, **dict(model_args, **kwargs))
