from typing import Final, Optional, Type, Callable, Union, Tuple, List
import logging

import jax
from flax import nnx
import jax.numpy as jnp
from flax.typing import Dtype, PromoteDtypeFn, PrecisionLike
from flax.nnx.nn import dtypes as flax_dtypes
import jax.image as jimg
import numpy as np

from .helpers import to_2tuple
from .format import Format, nhwc_to
from .trace_utils import _assert
from .identity import Identity

_logger = logging.getLogger(__name__)


def patch_dropout_forward(
    x: jax.Array,
    rate: float,
    num_prefix_tokens: int,
    ordered: bool,
    *,
    rngs: nnx.Rngs,
) -> Tuple[jax.Array, Optional[jax.Array]]:
    """
    Common forward logic for patch dropout.

    Args:
        x: Input tensor of shape (B, L, D)
        prob: Dropout probability
        num_prefix_tokens: Number of prefix tokens to preserve
        ordered: Whether to maintain patch order
        training: Whether in training mode

    Returns:
        Tuple of (output tensor, keep_indices or None)
    """
    if rate == 0:
        return x, None

    if num_prefix_tokens:
        prefix_tokens, x = x[:, :prefix_tokens], x[:, prefix_tokens:]
    else:
        prefix_tokens = None

    B = x.shape[0]
    L = x.shape[1]
    num_keep = max(1, int(L * (1.0 - rate)))
    keep_indices = jnp.argsort(jax.random.normal(rngs.dropout(), (B, L)), axis=-1)[
        :, :num_keep
    ]

    if ordered:
        # NOTE does not need to maintain patch order in typical transformer use,
        # but possibly useful for debug / visualization
        keep_indices = keep_indices.sort(axis=-1)
    keep_indices_expanded = jnp.expand_dims(
        keep_indices,
        axis=tuple(range(2, x.ndim)),  # expand into the trailing dims
    )
    x = jnp.take_along_axis(x, keep_indices_expanded, axis=1)

    if prefix_tokens is not None:
        x = jnp.concatenate((prefix_tokens, x), axis=1)

    return x, keep_indices


class PatchDropout(nnx.Module):
    """
    Patch Dropout without returning indices.
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """

    def __init__(
        self,
        rate: float = 0.5,
        num_prefix_tokens: int = 1,
        ordered: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        assert 0 <= rate < 1.0
        self.prob = rate
        self.num_prefix_tokens = (
            num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        )
        self.ordered = ordered
        self.rngs = rngs

    def __call__(self, x: jax.Array) -> jax.Array:
        output, _ = patch_dropout_forward(
            x,
            self.prob,
            self.num_prefix_tokens,
            self.ordered,
            self.rngs,
        )
        return output
