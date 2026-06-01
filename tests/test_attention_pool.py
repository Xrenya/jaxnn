import jax
import jax.numpy as jnp
from flax import nnx

from jaxnn.layers.attention_pool import AttentionPoolLatent


def test_attention_pool_latent_smoke():
    rngs = nnx.Rngs(42)

    layer = AttentionPoolLatent(
        in_features=32,
        out_features=64,
        embed_dim=32,
        num_heads=4,
        latent_len=1,
        pool_type="token",
        rngs=rngs,
    )

    x = jnp.ones((2, 16, 32))  # B, N, C
    y = layer(x, deterministic=True)

    assert y.shape == (2, 64)
    assert jnp.isfinite(y).all()


def test_attention_pool_latent_multi_latent_avg():
    rngs = nnx.Rngs(42)

    layer = AttentionPoolLatent(
        in_features=32,
        out_features=32,
        embed_dim=32,
        num_heads=4,
        latent_len=4,
        pool_type="avg",
        fused_attn=False,
        rngs=rngs,
    )

    x = jnp.ones((2, 16, 32))
    y = layer(x, deterministic=True)

    assert y.shape == (2, 32)
    assert jnp.isfinite(y).all()


def test_attention_pool_latent_mask_manual():
    rngs = nnx.Rngs(42)

    layer = AttentionPoolLatent(
        in_features=32,
        embed_dim=32,
        num_heads=4,
        latent_len=2,
        pool_type="avg",
        fused_attn=False,
        rngs=rngs,
    )

    x = jnp.ones((2, 8, 32))
    mask = jnp.zeros((1, 1, 2, 8))
    mask = mask.at[:, :, :, -1].set(float("-inf"))

    y = layer(x, attn_mask=mask, deterministic=True)

    assert y.shape == (2, 32)
    assert jnp.isfinite(y).all()
