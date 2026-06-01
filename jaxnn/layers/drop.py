"""DropBlock, DropPath

PyTorch implementations of DropBlock and DropPath (Stochastic Depth) regularization layers.

Papers:
DropBlock: A regularization method for convolutional networks (https://arxiv.org/abs/1810.12890)

Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)

Code:
DropBlock impl inspired by two Tensorflow impl that I liked:
 - https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py#L74
 - https://github.com/clovaai/assembled-cnn/blob/master/nets/blocks.py

Adapted from PyTorch/timm's resnet implementation.
Copyright of original work: 2020 Ross Wightman

Hacked together by / Copyright 2026 Rinat Shaymukhametov
"""

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
from .trace_utils import _assert
from .identity import Identity

_logger = logging.getLogger(__name__)


def calculate_drop_path_rates(
    drop_path_rate: float,
    depths: Union[int, List[int]],
    stagewise: bool = False,
) -> Union[List[float], List[List[float]]]:
    """Generate drop path rates for stochastic depth.

    This function handles two common patterns for drop path rate scheduling:
    1. Per-block: Linear increase from 0 to drop_path_rate across all blocks
    2. Stage-wise: Linear increase across stages, with same rate within each stage

    Args:
        drop_path_rate: Maximum drop path rate (at the end).
        depths: Either a single int for total depth (per-block mode) or
                list of ints for depths per stage (stage-wise mode).
        stagewise: If True, use stage-wise pattern. If False, use per-block pattern.
                   When depths is a list, stagewise defaults to True.

    Returns:
        For per-block mode: List of drop rates, one per block.
        For stage-wise mode: List of lists, drop rates per stage.
    """

    if isinstance(depths, int):
        # Single depth value - per-block pattern
        if stagewise:
            raise ValueError(
                "stagewise=True requires depths to be a list of stage depths"
            )
        dpr = [x.item() for x in jnp.linspace(0, drop_path_rate, depths)]
        return dpr
    else:
        total_depth = sum(depths)
        if stagewise:
            # Stage-wise pattern: same drop rate within each stage
            rates = jnp.linspace(0, drop_path_rate, total_depth)
            # remove the last for correct splitting
            indices = jnp.cumsum(jnp.array(depths))[:-1]
            dpr = jnp.split(rates, indices)
            return dpr
        else:
            # Per-block pattern across all stages
            dpr = [x for x in jnp.linspace(0, drop_path_rate, total_depth)]
            return dpr
