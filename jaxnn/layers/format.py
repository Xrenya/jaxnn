from enum import Enum
from typing import Union

import jax
from flax import nnx
from einops import rearrange


class Format(str, Enum):
    NCHW: str = "NCHW"
    NHWC: str = "NHWC"
    NCL: str = "NCL"
    NLC: str = "NLC"


FormatT = Union[str, Format]


def get_spartial_dim(fmt: FormatT):
    """Return spatial dimension indices for a given tensor format.

    Args:
        fmt: Tensor format (NCHW, NHWC, NCL, or NLC).

    Returns:
        Tuple of spatial dimension indices.
    """
    fmt = Format(fmt)
    if fmt == Format.NLC:
        dim = (1,)
    elif fmt == Format.NCL:
        dim = (2,)
    elif fmt == Format.NHWC:
        dim = (1, 2)
    else:
        dim = (2, 3)
    return dim


def get_channel_dim(fmt: FormatT):
    """Return channel dimension index for a given tensor format.

    Args:
        fmt: Tensor format (NCHW, NHWC, NCL, or NLC).

    Returns:
        Channel dimension index.
    """
    fmt = Format(fmt)
    if fmt == Format.NHWC:
        dim = 3
    elif fmt == Format.NCL:
        dim = 2
    else:
        dim = 1
    return dim


def nchw_to(x: jax.Array, fmt: Format):
    """Convert tensor from NCHW format to specified format.

    Args:
        x: Input tensor in NCHW format.
        fmt: Target format.

    Returns:
        Tensor in target format.
    """
    if fmt == Format.NHWC:
        x = rearrange(x, "b c h w -> b w h c")
    elif fmt == Format.NCL:
        x = rearrange(x, "b c h w -> b c (w h)")
    elif fmt == Format.NLC:
        x = rearrange(x, "b c h w -> b (w h) c")
    return x


def nhwc_to(x: jax.Array, fmt: Format):
    """Convert tensor from NHWC format to specified format.

    Args:
        x: Input tensor in NHWC format.
        fmt: Target format.

    Returns:
        Tensor in target format.
    """
    if fmt == Format.NCHW:
        x = rearrange(x, "b h w c -> b c w h")
    elif fmt == Format.NCL:
        x = rearrange(x, "b h w c -> b c (w h)")
    elif fmt == Format.NLC:
        x = rearrange(x, "b h w c -> b (w h) c")
    return x
