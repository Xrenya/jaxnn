"""JAX/Flax Feature Extraction Helpers.

This design is inspired by timm / torchvision's IntermediateLayerGetter:
https://github.com/pytorch/vision/blob/d88d8961ae51507d0cb680329d985b1488b1b76b/torchvision/models/_utils.py

Ported/adapted to JAX/Flax by Rinat Shaymukhametov, with modifications to support
Flax/JAX.

Hacked together by / Copyright 2026 Rinat Shaymukhametov
"""

from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import jax
from flax import nnx

OutIndicesT = Union[int, Tuple[int, ...]]


def _out_indices_as_tuple(x: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    if isinstance(x, int):
        # if indices is an int, take last N features
        return tuple(range(-x, 0))
    return tuple(x)


class FeatureInfo:
    def __init__(
        self,
        feature_info: List[Dict],
        out_indices: OutIndicesT,
    ):
        out_indices = _out_indices_as_tuple(out_indices)
        prev_reduction = 1
        for i, fi in enumerate(feature_info):
            # sanity check the mandatory fields, there may be additional fields depending on the model
            assert "num_chs" in fi and fi["num_chs"] > 0
            assert "reduction" in fi and fi["reduction"] >= prev_reduction
            prev_reduction = fi["reduction"]
            assert "module" in fi
            fi.setdefault("index", i)
        self.out_indices = out_indices
        self.info = feature_info

    def from_other(self, out_indices: OutIndicesT):
        out_indices = _out_indices_as_tuple(out_indices)
        return FeatureInfo(deepcopy(self.info), out_indices)

    def get(self, key: str, idx: Optional[Union[int, List[int]]] = None):
        """Get value by key at specified index (indices)
        if idx == None, returns value for key at each output index
        if idx is an integer, return value for that feature module index (ignoring output indices)
        if idx is a list/tuple, return value for each module index (ignoring output indices)
        """
        if idx is None:
            return [self.info[i][key] for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [self.info[i][key] for i in idx]
        else:
            return self.info[idx][key]

    def get_dicts(
        self,
        keys: Optional[List[str]] = None,
        idx: Optional[Union[int, List[int]]] = None,
    ):
        """return info dicts for specified keys (or all if None) at specified indices (or out_indices if None)"""
        if idx is None:
            if keys is None:
                return [self.info[i] for i in self.out_indices]
            else:
                return [{k: self.info[i][k] for k in keys} for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [
                self.info[i] if keys is None else {k: self.info[i][k] for k in keys}
                for i in idx
            ]
        else:
            return (
                self.info[idx] if keys is None else {k: self.info[idx][k] for k in keys}
            )

    def channels(self, idx: Optional[Union[int, List[int]]] = None):
        """feature channels accessor"""
        return self.get("num_chs", idx)

    def reduction(self, idx: Optional[Union[int, List[int]]] = None):
        """feature reduction (output stride) accessor"""
        return self.get("reduction", idx)

    def module_name(self, idx: Optional[Union[int, List[int]]] = None):
        """feature module name accessor"""
        return self.get("module", idx)

    def __getitem__(self, item):
        return self.info[item]

    def __len__(self):
        return len(self.info)


def _get_feature_info(net, out_indices: OutIndicesT):
    feature_info = getattr(net, "feature_info")
    if isinstance(feature_info, FeatureInfo):
        return feature_info.from_other(out_indices)
    elif isinstance(feature_info, (list, tuple)):
        return FeatureInfo(net.feature_info, out_indices)
    else:
        assert False, "Provided feature_info is not valid"


class FeatureGetterNet(nnx.Module):
    """Feature extraction wrapper using the model's forward_intermediates().

    This is the JAX/Flax equivalent of timm's FeatureGetterNet. Instead of
    PyTorch hooks or module rewriting, it delegates to the model's own
    `forward_intermediates()` method.

    The wrapped model must implement:
        forward_intermediates(x, indices) -> (final_feature, list_of_intermediates)
    """

    def __init__(
        self,
        model: nnx.Module,
        out_indices: OutIndicesT = (0, 1, 2, 3, 4),
        out_map: Optional[Sequence[Union[int, str]]] = None,
        return_dict: bool = False,
    ):
        self.model = model
        self.return_dict = return_dict
        self.feature_info = _get_feature_info(model, out_indices)
        self.out_indices = _out_indices_as_tuple(out_indices)
        if out_map is not None:
            assert len(out_map) == len(self.out_indices)
            self.out_map = out_map
        else:
            self.out_map = None

    def __call__(self, x: jax.Array):
        _, intermediates = self.model.forward_intermediates(
            x,
            indices=self.out_indices,
        )
        if self.return_dict:
            if self.out_map is not None:
                return OrderedDict(zip(self.out_map, intermediates))
            return OrderedDict(
                (str(i), f) for i, f in zip(self.out_indices, intermediates)
            )
        return intermediates
