import functools
from typing import Callable, Optional, Type

from flax import nnx
from flax.typing import Dtype, PromoteDtypeFn
import jax.numpy as jnp
import collections.abc
from itertools import repeat


def _to_ntuple(n: int):
    """Cast an integer or tuple to a tuple of length n."""

    def _parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return _parse


to_1tuple = _to_ntuple(1)
to_2tuple = _to_ntuple(2)
to_3tuple = _to_ntuple(3)
to_4tuple = _to_ntuple(4)
to_ntuple = _to_ntuple


def wrap_norm_layer(
    norm_cls: Type[nnx.Module],
    dtype: Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    promote_dtype: PromoteDtypeFn = jnp.float32,
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
    def factory(num_features: int, rngs: nnx.Rngs, **kwargs) -> nnx.Module:
        return norm_cls(
            num_features,
            dtype=dtype,
            param_dtype=param_dtype,
            promote_dtype=promote_dtype,
            rngs=rngs,
            **kwargs,
        )

    return factory
