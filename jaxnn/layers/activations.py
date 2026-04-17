"""Layers

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
