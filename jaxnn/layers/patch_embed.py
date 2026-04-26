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


class PatchEmbed(nnx.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
        output_fmt: Optional[str] = None,
        # Data types and precision
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        preferred_element_type: Optional[Dtype] = None,
        norm_dtype: Optional[Dtype] = jnp.float32,
        norm_param_dtype: Dtype = jnp.float32,
        norm_promote_dtype: PromoteDtypeFn = flax_dtypes.promote_dtype,
        *,
        rngs: nnx.Rngs,
    ):
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)
        self.embed_dim = embed_dim
        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            self.flatten = flatten
            self.output_fmt = Format.NHWC
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nnx.Conv(
            in_features=in_chans,
            out_features=embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            promote_dtype=promote_dtype,
            preferred_element_type=preferred_element_type,
            rngs=rngs,
        )
        self.norm = norm_layer(embed_dim, rngs=rngs) if norm_layer else Identity()
        self.rngs = rngs

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        B, H, W, C = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(
                    H == self.img_size[0],
                    f"Input height ({H}) doesn't match model ({self.img_size[0]}).",
                )
                _assert(
                    W == self.img_size[1],
                    f"Input width ({W}) doesn't match model ({self.img_size[1]}).",
                )
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]}).",
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]}).",
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))

        x = self.proj(x)

        if self.flatten:
            x = x.reshape(B, -1, self.embed_dim)  # (B, N, embed_dim)
        elif self.output_fmt != Format.NHWC:
            x = nhwc_to(x, self.output_fmt)
        x = self.norm(x)
        return x

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def set_input_size(
        self,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        rngs: Optional[nnx.Rngs] = None,
    ):
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = to_2tuple(patch_size)

        if new_patch_size is not None and new_patch_size != self.patch_size:
            if rngs is not None:
                rngs = self.rngs

            old_weight = self.proj.kernel.get_value()
            old_bias = (
                self.proj.bias.get_value() if self.proj.bias is not None else None
            )

            new_proj = nnx.Conv(
                self.proj.in_features,
                self.proj.out_features,
                kernel_size=new_patch_size,
                strides=new_patch_size,
                use_bias=old_bias is not None,
                rngs=rngs,
            )

            new_kernel = resample_patch_embed(old_weight, new_patch_size)

            new_proj.kernel[...] = nnx.Param(new_kernel)

            if old_bias is not None:
                new_proj.bias[...] = nnx.Param(old_bias)

            self.proj = new_proj
            self.patch_size = new_patch_size

        img_size = img_size or self.img_size

        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(
                img_size
            )


def resample_patch_embed(
    patch_embed: jnp.ndarray,
    new_size: List[int] | Tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
    verbose: bool = False,
) -> jnp.ndarray:
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    Args:
        patch_embed: original kernel (JAX array), expected shape (out_channels, in_channels, h, w)
        new_size: target shape (height, width)-only (list or tuple)
        interpolation: interpolation for resize ('bilinear', 'bicubic', 'nearest')
        antialias: use anti-aliasing filter in resize
        verbose: log operation
    Returns:
        Resized patch embedding kernel.
    """
    assert patch_embed.ndim == 4, "Four dimensions expected (out_ch, in_ch, h, w)"
    assert len(new_size) == 2, "New shape should only be (h, w)"
    old_size = tuple(patch_embed.shape[-2:])
    new_size = tuple(new_size)
    if old_size == new_size:
        return patch_embed

    if verbose:
        _logger.info(
            f"Resize patch embedding {patch_embed.shape} to {new_size}, w/ {interpolation} interpolation."
        )

    def get_resize_mat(
        _old_size: Tuple[int, int], _new_size: Tuple[int, int]
    ) -> jnp.ndarray:
        n_old = int(np.prod(_old_size))
        eye = jnp.eye(n_old, dtype=jnp.float32).reshape(
            n_old, *_old_size
        )  # (n_old, h, w)

        def resize_basis(basis_vec):  # (h, w) -> (new_h*new_w,)
            x = basis_vec[None, None, ...]  # (1,1,h,w)
            y = jimg.resize(
                x,
                (1, 1, _new_size[0], _new_size[1]),
                method=interpolation,
                antialias=antialias,
            )
            return y[0, 0, ...].reshape(-1)

        mat = jax.vmap(resize_basis)(eye)  # (n_old, n_new)
        return mat.T

    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv_np = np.linalg.pinv(resize_mat.T)
    resize_mat_pinv = jnp.array(resize_mat_pinv_np, dtype=patch_embed.dtype)

    def resample_kernel(kernel: jnp.ndarray) -> jnp.ndarray:
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    v_resample_kernel = jax.vmap(
        jax.vmap(resample_kernel, in_axes=0, out_axes=0), in_axes=1, out_axes=1
    )

    orig_dtype = patch_embed.dtype
    patch_embed_f = patch_embed.astype(jnp.float32)
    patch_embed_resampled_f = v_resample_kernel(patch_embed_f)
    patch_embed_resampled = patch_embed_resampled_f.astype(orig_dtype)

    return patch_embed_resampled
