from typing import Optional

import jax
from flax import nnx
import jax.numpy as jnp
from flax.typing import Dtype


class LayerScale(nnx.Module):
    """Apply per-channel scaling to a tensor (channels in last dimension).

    Stabilize training by learning a small initial scale per channel.

    Args:
        dim: Number of channels (last dimension of input tensor).
        init_values: Initial value for each scale parameter.
        param_dtype: Dtype for the learnable scale parameter.
        dtype: Optional dtype for the output (JAX will promote; usually not needed).
        rngs: NNX RNGs for parameter initialization.
    """

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        param_dtype: Dtype = jnp.float32,
        dtype: Optional[Dtype] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.init_values = init_values
        self.rngs = rngs

        self.dtype = dtype

        # Learnable per-channel scale
        self.gamma = nnx.Param(
            nnx.initializers.constant(init_values, dtype=param_dtype)(
                rngs.params(), (dim,)
            )
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x * self.gamma.get_value()
        if self.dtype is not None:
            y = y.astype(self.dtype)
        return y
