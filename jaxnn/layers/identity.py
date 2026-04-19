from flax import nnx


class Identity(nnx.Module):
    def __call__(self, x):
        return x
