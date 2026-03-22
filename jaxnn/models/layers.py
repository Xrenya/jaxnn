from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable, Sequence

import jax
from flax import nnx
import jax.numpy as jnp

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