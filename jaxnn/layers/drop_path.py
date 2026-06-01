from flax import nnx
import jax
import jax.numpy as jnp


class DropPath(nnx.Module):
    """DropPath randomly drops entire samples in a batch along the residual path"""

    def __init__(self, rate: float, scale_by_keep: bool = True, *, rngs: nnx.Rngs):
        self.rate = rate
        self.scale_by_keep = scale_by_keep
        self.rngs = rngs

    def __call__(self, x: jax.Array, deterministic: bool = False):
        if deterministic or self.rate == 0:
            return x

        if self.rate >= 1.0:
            return jnp.zeros_like(x)

        keep_prob = 1.0 - self.rate
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rng = self.rngs.droppath()
        # Sample mask: 1 = keep path, 0 = drop path (probability of keeping is 1-p)
        mask = jax.random.bernoulli(rng, p=keep_prob, shape=mask_shape).astype(x.dtype)
        if keep_prob > 0.0 and self.scale_by_keep:
            return x * mask / keep_prob
        return x * mask
