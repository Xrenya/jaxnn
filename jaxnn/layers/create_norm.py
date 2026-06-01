import functools
import types
from typing import Type, Dict, Optional, Union, Callable

from flax import nnx
import jax

LayerType = Union[str, Callable, Type[nnx.Module]]


_NORM_LAYER_MAP: Dict[str, Type[nnx.Module]] = {
    "batchnorm": nnx.BatchNorm,
    "bn": nnx.BatchNorm,
    "groupnorm": nnx.GroupNorm,
    "gn": nnx.GroupNorm,
    "layernorm": nnx.LayerNorm,
    "ln": nnx.LayerNorm,
    "rmsnorm": nnx.RMSNorm,
}


def get_norm_layer(norm_layer: Optional[LayerType]):
    if norm_layer is None:
        return None
    assert isinstance(norm_layer, (type, str, types.FunctionType, functools.partial))
    norm_kwrgs = {}

    if isinstance(norm_layer, functools.partial):
        norm_kwrgs.update(norm_layer.keywords)
        norm_layer = norm_layer.func

    if isinstance(norm_layer, str):
        if not norm_layer:
            return None
        layer_name = norm_layer.replace("_", "").lower()
        norm_layer = _NORM_LAYER_MAP[layer_name]
    else:
        norm_layer = norm_layer

    if norm_kwrgs:
        norm_layer = functools.partial(norm_layer, **norm_kwrgs)
    return norm_layer


def create_norm_layer(layer_name, num_features, **kwargs):
    layer = get_norm_layer(layer_name)
    layer_instance = layer(num_features, **kwargs)
    return layer_instance
