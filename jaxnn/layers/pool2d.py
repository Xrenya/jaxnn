from typing import Tuple, Type, Union, Callable

import jax
from flax import nnx

LayerType = Union[str, Callable, Type[nnx.Module]]


class MaxPool2D(nnx.Module):
    """2-D max pooling - replaces anonymous lambdas wrapping ``nnx.max_pool``."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (2, 2),
        padding: Union[str, Tuple] = ((1, 1), (1, 1)),
    ) -> None:
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.max_pool(
            inputs=x,
            window_shape=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )

    def __repr__(self) -> str:
        return f"MaxPool2D(kernel_size={self.kernel_size}, strides={self.strides})"


class AvgPool2D(nnx.Module):
    """2-D average pooling - replaces anonymous lambdas wrapping ``nnx.avg_pool``."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (2, 2),
        strides: Tuple[int, int] = (2, 2),
        padding: Union[str, Tuple] = "VALID",
    ) -> None:
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.avg_pool(
            x, window_shape=self.kernel_size, strides=self.strides, padding=self.padding
        )

    def __repr__(self) -> str:
        return f"AvgPool2D(kernel_size={self.kernel_size}, strides={self.strides}, padding={self.padding})"
