"""Differential Attention

Paper: 'Differential Transformer' - https://arxiv.org/abs/2410.05258

Reference impl: https://github.com/microsoft/unilm/tree/master/Diff-Transformer
"""

import math
from typing import Optional, Type

import jax
from flax import nnx


class DiffAttention(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, x: jax.Array):
        return x
