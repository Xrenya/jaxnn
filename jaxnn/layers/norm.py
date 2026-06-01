import numbers
import functools
import types
from typing import Type, Dict, Optional, Union, Callable

from flax import nnx
import jax.numpy as jnp
import jax

LayerType = Union[str, Callable, Type[nnx.Module]]


# class RmsNorm(nnx.Module):
#     def __init__(
#         self,
#         channels: int,
#         epsilon: float = 1e-6,
#         affine: bool = True,
#         norm_dtype: Optional[Dtype] = jnp.float32,
#         norm_param_dtype: Dtype = jnp.float32,
#         norm_promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
#     ):
#         normalized_shape = channels
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         self.normalized_shape = normalized_shape
#         self.eps = eps
#         self.elementwise_affine = affine

#         if self.elementwise_affine:
#             self.weight = nnx.Param(jnp.zeros(self.normalized_shape))
#         else:
#             self.weight = None

#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         if self.elementwise_affine:
#             nnx.initializers.ones_init(self.weight)
