from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from timm.layers import Format, _assert
from ._manipulate import checkpoint


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
            assert 'num_chs' in fi and fi['num_chs'] > 0
            assert 'reduction' in fi and fi['reduction'] >= prev_reduction
            prev_reduction = fi['reduction']
            assert 'module' in fi
            fi.setdefault('index', i)
        self.out_indices = out_indices
        self.info = feature_info

    def from_other(self, out_indices: OutIndicesT):
        out_indices = _out_indices_as_tuple(out_indices)
        return FeatureInfo(deepcopy(self.info), out_indices)

    def get(self, key: str, idx: Optional[Union[int, List[int]]] = None):
        """ Get value by key at specified index (indices)
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

    def get_dicts(self, keys: Optional[List[str]] = None, idx: Optional[Union[int, List[int]]] = None):
        """ return info dicts for specified keys (or all if None) at specified indices (or out_indices if None)
        """
        if idx is None:
            if keys is None:
                return [self.info[i] for i in self.out_indices]
            else:
                return [{k: self.info[i][k] for k in keys} for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [self.info[i] if keys is None else {k: self.info[i][k] for k in keys} for i in idx]
        else:
            return self.info[idx] if keys is None else {k: self.info[idx][k] for k in keys}

    def channels(self, idx: Optional[Union[int, List[int]]] = None):
        """ feature channels accessor
        """
        return self.get('num_chs', idx)

    def reduction(self, idx: Optional[Union[int, List[int]]] = None):
        """ feature reduction (output stride) accessor
        """
        return self.get('reduction', idx)

    def module_name(self, idx: Optional[Union[int, List[int]]] = None):
        """ feature module name accessor
        """
        return self.get('module', idx)

    def __getitem__(self, item):
        return self.info[item]

    def __len__(self):
        return len(self.info)
    

def _get_feature_info(net, out_indices: OutIndicesT):
    feature_info = getattr(net, 'feature_info')
    if isinstance(feature_info, FeatureInfo):
        return feature_info.from_other(out_indices)
    elif isinstance(feature_info, (list, tuple)):
        return FeatureInfo(net.feature_info, out_indices)
    else:
        assert False, "Provided feature_info is not valid"


class FeatureHookNet(nn.ModuleDict):
    """ FeatureHookNet

    Wrap a model and extract features specified by the out indices using forward/forward-pre hooks.

    If `no_rewrite` is True, features are extracted via hooks without modifying the underlying
    network in any way.

    If `no_rewrite` is False, the model will be re-written as in the
    FeatureList/FeatureDict case by folding first to second (Sequential only) level modules into this one.

    """
    def __init__(
        self,
        model: nn.Module,
        out_indices: OutIndicesT = (0, 1, 2, 3, 4),
        out_map: Optional[Sequence[Union[int, str]]] = None,
        return_dict: bool = False,
        output_fmt: str = 'HWCN',
        no_rewrite: Optional[bool] = None,
        flatten_sequential: bool = False,
        default_hook_type: str = 'forward',
    ):
        """

        Args:
            model: Model from which to extract features.
            out_indices: Output indices of the model features to extract.
            out_map: Return id mapping for each output index, otherwise str(index) is used.
            return_dict: Output features as a dict.
            no_rewrite: Enforce that model is not re-written if True, ie no modules are removed / changed.
                flatten_sequential arg must also be False if this is set True.
            flatten_sequential: Re-write modules by flattening first two levels of nn.Sequential containers.
            default_hook_type: The default hook type to use if not specified in model.feature_info.
        """
        super().__init__()
        assert not torch.jit.is_scripting()
        self.feature_info = _get_feature_info(model, out_indices)
        self.return_dict = return_dict
        self.output_fmt = Format(output_fmt)
        self.grad_checkpointing = False
        if no_rewrite is None:
            no_rewrite = not flatten_sequential
        layers = OrderedDict()
        hooks = []
        if no_rewrite:
            assert not flatten_sequential
            if hasattr(model, 'reset_classifier'):  # make sure classifier is removed?
                model.reset_classifier(0)
            layers['body'] = model
            hooks.extend(self.feature_info.get_dicts())
        else:
            modules = _module_list(model, flatten_sequential=flatten_sequential)
            remaining = {
                f['module']: f['hook_type'] if 'hook_type' in f else default_hook_type
                for f in self.feature_info.get_dicts()
            }
            for new_name, old_name, module in modules:
                layers[new_name] = module
                for fn, fm in module.named_modules(prefix=old_name):
                    if fn in remaining:
                        hooks.append(dict(module=fn, hook_type=remaining[fn]))
                        del remaining[fn]
                if not remaining:
                    break
            assert not remaining, f'Return layers ({remaining}) are not present in model'
        self.update(layers)
        self.hooks = FeatureHooks(hooks, model.named_modules(), out_map=out_map)

    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    def forward(self, x):
        for i, (name, module) in enumerate(self.items()):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # Skipping checkpoint of first module because need a gradient at input
                # Skipping last because networks with in-place ops might fail w/ checkpointing enabled
                # NOTE: first_or_last module could be static, but recalc in is_scripting guard to avoid jit issues
                first_or_last_module = i == 0 or i == max(len(self) - 1, 0)
                x = module(x) if first_or_last_module else checkpoint(module, x)
            else:
                x = module(x)
        out = self.hooks.get_output(x.device)
        return out if self.return_dict else list(out.values())