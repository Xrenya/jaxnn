from typing import Optional, Type, Union, Callable

from flax import nnx

LayerType = Union[str, Callable, Type[nnx.Module]]


def create_aa(aa_layer: Optional[Type[nnx.Module]], channels: int, stride: int):
    """Helper to create anti-aliasing layers."""
    if aa_layer is None:
        return nnx.identity
    return aa_layer(channels=channels, stride=stride)