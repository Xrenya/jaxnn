from typing import Final, Optional, Type
import logging

import jax
from flax import nnx
import jax.numpy as jnp
from flax.typing import Dtype, PromoteDtypeFn, PrecisionLike
from flax.nnx.nn import dtypes as flax_dtypes


def rope_rotate_half(x: jax.Array) -> jax.Array:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, -x1], -1).reshape(x.shape)


def rot(x: jax.Array) -> jax.Array:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x1  x0 -x3  x2 -x5  x4]
    return jnp.concatenate([-x[..., 1::2], -x[..., 0::2]], -1).reshape(x.shape)


def apply_rot_embed_cat(
    x: jax.Array,
    emb: jax.Array,
    half: bool = False,
) -> jax.Array:
    sin_emb, cos_emb = jnp.split(emb, 2, axis=-1)
    if half:
        # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
        # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2
        # rope_rotate_half(x), eg [-x3, -x4, -x5, x0, x1, x2]
        return x * cos_emb + rope_rotate_half(x) * sin_emb
    else:
        # sin: [..., D], eg [sin0, sin0, sin1, sin1, sin2, sin2]
        # cos: [..., D], eg [cos0, cos0, cos1, cos1, cos2, cos2]
        # rot(x), eg [-x1, x0, -x3, x2, -x5, x4]
        return x * cos_emb + rot(x) * sin_emb
