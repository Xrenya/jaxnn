from typing import Final, Optional, Type
import logging

import jax
from flax import nnx
import jax.numpy as jnp
from flax.typing import Dtype, PromoteDtypeFn, PrecisionLike
from flax.nnx.nn import dtypes as flax_dtypes


class PatchEmbed(nnx.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,):
        pass

    def __call__(self, x: jax.Array):
        return x