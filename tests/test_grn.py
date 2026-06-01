import jax.numpy as jnp
from flax import nnx

from jaxnn.layers.grn import GlobalResponseNorm


def test_grn_channels_last_identity_at_init():
    x = jnp.arange(2 * 4 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 4, 3)

    grn = GlobalResponseNorm(dim=3, channels_last=True, rngs=nnx.Rngs(0))
    y = grn(x)

    assert y.shape == x.shape
    assert jnp.allclose(y, x)
    assert jnp.isfinite(y).all()


def test_grn_channels_first_identity_at_init():
    x = jnp.arange(2 * 3 * 4 * 4, dtype=jnp.float32).reshape(2, 3, 4, 4)

    grn = GlobalResponseNorm(dim=3, channels_last=False, rngs=nnx.Rngs(1))
    y = grn(x)

    assert y.shape == x.shape
    assert jnp.allclose(y, x)
    assert jnp.isfinite(y).all()


def test_grn_matches_reference_when_params_nonzero():
    x = jnp.arange(2 * 3 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 3, 4)

    grn = GlobalResponseNorm(dim=4, channels_last=True, rngs=nnx.Rngs(2))
    grn.weight[...] = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
    grn.bias[...] = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)

    y = grn(x)

    x_g = jnp.linalg.norm(x, ord=2, axis=(1, 2), keepdims=True)
    x_n = x_g / (jnp.mean(x_g, axis=-1, keepdims=True) + 1e-6)
    weight = grn.weight.get_value().reshape(1, 1, 1, 4)
    bias = grn.bias.get_value().reshape(1, 1, 1, 4)
    expected = x + bias + weight * (x * x_n)

    assert y.shape == x.shape
    assert jnp.allclose(y, expected, rtol=1e-5, atol=1e-5)
    assert jnp.isfinite(y).all()