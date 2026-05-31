from typing import Optional, Tuple

from flax import nnx
from flax.typing import Dtype, PromoteDtypeFn, PrecisionLike
from flax.nnx.nn import dtypes as flax_dtypes
import jax.numpy as jnp


__all__ = ["GlobalResponseNorm"]


class GlobalResponseNorm(nnx.Module):
    """Global Response Normalization layer."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        channels_last: bool = True,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        *,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.eps = eps
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.promote_dtype = promote_dtype
        self.rngs = rngs

        if channels_last:
            self.spatial_dim: Tuple[int, ...] = (1, 2)
            self.channel_dim: int = -1
            self.wb_shape: Tuple[int, ...] = (1, 1, 1, dim)
        else:
            self.spatial_dim = (2, 3)
            self.channel_dim = 1
            self.wb_shape = (1, dim, 1, 1)

        self.weight = nnx.Param(jnp.zeros((dim,), dtype=self.param_dtype))
        self.bias = nnx.Param(jnp.zeros((dim,), dtype=self.param_dtype))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = jnp.reshape(self.weight.get_value(), self.wb_shape)
        bias = jnp.reshape(self.bias.get_value(), self.wb_shape)
        x, weight, bias = self.promote_dtype((x, weight, bias), dtype=self.dtype)

        # 1. L2: ||X|| with jnp.linalg.norm
        x_g = jnp.linalg.norm(x, ord=2, axis=self.spatial_dim, keepdims=True)

        # 2. Channel normalizaton: Nx = ||X|| / (Mean(||X||) + eps)
        x_n = x_g / (jnp.mean(x_g, axis=self.channel_dim, keepdims=True) + self.eps)

        # 3. torch.addcmul(b, w, x * x_n) -> b + w * (x * x_n)
        return x + bias + weight * (x * x_n)
