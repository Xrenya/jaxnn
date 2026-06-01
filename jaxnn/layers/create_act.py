import functools
import types
from typing import Type, Dict, Union, Optional, Callable

from flax import nnx
import jax

LayerType = Union[str, Callable, Type[nnx.Module]]


_ACT_FN_MAP: Dict[str, Type[nnx.Module]] = {
    "celu": nnx.celu,
    "elu": nnx.elu,
    "gelu": nnx.gelu,
    "glu": nnx.glu,
    "hard_sigmoid": nnx.hard_sigmoid,
    "hard_silu": nnx.hard_silu,
    "hard_swish": nnx.hard_swish,
    "hard_tanh": nnx.hard_tanh,
    "leaky_relu": nnx.leaky_relu,
    "log_sigmoid": nnx.log_sigmoid,
    "log_softmax": nnx.log_softmax,
    "logsumexp": nnx.logsumexp,
    "one_hot": nnx.one_hot,
    "relu": nnx.relu,
    "selu": nnx.selu,
    "sigmoid": nnx.sigmoid,
    "identity": nnx.identity,
    "silu": nnx.silu,
    "soft_sign": nnx.soft_sign,
    "softmax": nnx.softmax,
    "softplus": nnx.softplus,
    "standardize": nnx.standardize,
    "swish": nnx.swish,
    "tanh": nnx.tanh,
}


def get_act_layer(name: Optional[LayerType] = "relu"):
    """Activation Layer Factory
    Fetching activation layers by name with this function allows export
    """
    if name is None:
        return None
    if not isinstance(name, str):
        # callable, module, etc
        return name
    if not name:
        return None
    name = name.lower()
    return _ACT_FN_MAP[name]


def create_act_layer(
    name: Optional[LayerType] = "relu",
    **kwargs,
):
    act_layer = get_act_layer(name)
    if act_layer is None:
        return None
    # no inplace operation
    return act_layer(**kwargs)
