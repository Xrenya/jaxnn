"""Layers

TODO: move into serparate folder jaxnn/models/layers/..

Hacked together by / Copyright 2026 Rinat Shaymukhametov
"""
from typing import Optional, Tuple, Type, Union, Callable, Sequence

import jax
from flax import nnx

LayerType = Union[str, Callable, Type[nnx.Module]]


def to_ntuple(n: int):
    """Cast an integer or tuple to a tuple of length n."""

    def _parse(x):
        if isinstance(x, Sequence):
            return tuple(x)
        return tuple([x] * n)

    return _parse


def create_aa(aa_layer: Optional[Type[nnx.Module]], channels: int, stride: int):
    """Helper to create anti-aliasing layers."""
    if aa_layer is None:
        return nnx.identity
    return aa_layer(channels=channels, stride=stride)


class Activation(nnx.Module):
    """Wraps a stateless activation function so it prints cleanly.

    Storing a raw JAX function (e.g. ``nnx.relu``) as an attribute shows
    ``<jax._src.custom_derivatives.custom_jvp object at 0x…>``.  This
    wrapper gives a readable repr and is a proper ``nnx.Module`` so it
    participates correctly in NNX graph traversal (with no state leaves).
    """

    def __init__(self, fn: Callable) -> None:
        self.fn = fn

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fn(x)

    def __repr__(self) -> str:
        name = getattr(self.fn, "__name__", None) or type(self.fn).__name__
        return f"Activation({name})"


class MaxPool2D(nnx.Module):
    """2-D max pooling - replaces anonymous lambdas wrapping ``nnx.max_pool``."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (2, 2),
        padding: Union[str, Tuple] = ((1, 1), (1, 1)),
    ) -> None:
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.max_pool(
            inputs=x,
            window_shape=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )

    def __repr__(self) -> str:
        return f"MaxPool2D(kernel_size={self.kernel_size}, strides={self.strides})"


class AvgPool2D(nnx.Module):
    """2-D average pooling - replaces anonymous lambdas wrapping ``nnx.avg_pool``."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (2, 2),
        strides: Tuple[int, int] = (2, 2),
        padding: Union[str, Tuple] = "VALID",
    ) -> None:
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.avg_pool(
            x, window_shape=self.kernel_size, strides=self.strides, padding=self.padding
        )

    def __repr__(self) -> str:
        return f"AvgPool2D(kernel_size={self.kernel_size}, strides={self.strides}, padding={self.padding})"
