from typing import Sequence

from flax import nnx


def to_ntuple(n: int):
    """Cast an integer or tuple to a tuple of length n."""

    def _parse(x):
        if isinstance(x, Sequence):
            return tuple(x)
        return tuple([x] * n)

    return _parse
