"""Model Registry

Adapted from PyTorch/timm's model registry implementation.
Copyright of original work: 2020 Ross Wightman

Rewritten and extended by Rinat Shaymukhametov, with minor modifications

Hacked together by / Copyright 2026 Rinat Shaymukhametov
"""

import fnmatch
import re
import sys
import warnings
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import replace
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Sequence,
    Union,
    Tuple,
)

from ._pretrained import PretrainedCfg, DefaultCfg

__all__ = [
    "split_model_name_tag",
    "get_arch_name",
    "register_model",
    "generate_default_cfgs",
    "list_models",
    "list_pretrained",
    "is_model",
    "model_entrypoint",
    "list_modules",
    "is_model_in_modules",
    "get_pretrained_cfg_value",
    "is_model_pretrained",
    "get_arch_pretrained_cfgs",
]

_module_to_models: Dict[str, Set[str]] = defaultdict(
    set
)  # dict of sets to check membership of model in module
_model_to_module: Dict[str, str] = {}  # mapping of model names to module names
_model_entrypoints: Dict[
    str, Callable[..., Any]
] = {}  # mapping of model names to architecture entrypoint fns
_model_has_pretrained: Set[str] = (
    set()
)  # set of model names that have pretrained weight url present
_model_default_cfgs: Dict[
    str, PretrainedCfg
] = {}  # central repo for model arch -> default cfg objects
_model_pretrained_cfgs: Dict[
    str, PretrainedCfg
] = {}  # central repo for model arch.tag -> pretrained cfgs
_model_with_tags: Dict[str, List[str]] = defaultdict(
    list
)  # shortcut to map each model arch to all model + tag names


def split_model_name_tag(model_name: str, no_tag: str = "") -> Dict[str, str]:
    model_name, *tag_list = model_name.split(".", 1)
    model_name = model_name.split("/")[-1] if "/" in model_name else model_name
    tag = tag_list[0] if tag_list else no_tag
    return {"model_name": model_name, "tag": tag}


def get_arch_name(model_name: str) -> str:
    return split_model_name_tag(model_name).get("model_name", None)


def model_entrypoint(
    model_name: str, module_filter: Optional[str] = None
) -> Callable[..., Any]:
    """Fetch a model entrypoint for specified model name"""
    arch_name = get_arch_name(model_name)
    if module_filter and arch_name not in _module_to_models.get(module_filter, {}):
        raise RuntimeError(f"Model ({model_name} not found in module {module_filter}.")
    return _model_entrypoints[arch_name]


def register_model(fn: Callable[..., Any]) -> Callable[..., Any]:
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split(".")
    module_name = module_name_split[-1] if len(module_name_split) else ""

    model_name = fn.__name__
    if hasattr(mod, "__all__"):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    if model_name in _model_entrypoints:
        warnings.warn(
            f"Overwriting {model_name} in registry with {fn.__module__}.{model_name}. This is because the name being "
            "registered conflicts with an existing name. Please check if this is not expected.",
            stacklevel=2,
        )
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)

    if hasattr(mod, "default_cfgs"):
        default_cfg = mod.default_cfgs.get(model_name, None)

        # If not found by base name, the default_cfgs might use tagged keys
        # like 'resnet34.a1_in1k' instead of being pre-processed by
        # generate_default_cfgs into {'resnet34': DefaultCfg(...)}.
        # Build a DefaultCfg on the fly from all matching tagged entries.
        if default_cfg is None:
            matched = {}
            for k, v in mod.default_cfgs.items():
                cfg_base = k.split(".", 1)[0]
                if cfg_base == model_name:
                    matched[k] = v
            if matched:
                # Use generate_default_cfgs to build proper DefaultCfg
                generated = generate_default_cfgs(matched)
                default_cfg = generated.get(model_name, None)

        if default_cfg is None:
            return fn

        if not isinstance(default_cfg, DefaultCfg):
            # Wrap a bare PretrainedCfg into DefaultCfg
            if isinstance(default_cfg, PretrainedCfg):
                default_cfg = DefaultCfg(
                    tags=deque([""]),
                    cfgs={"": default_cfg},
                    is_pretrained=default_cfg.has_weights,
                )
            elif isinstance(default_cfg, dict):
                pcfg = PretrainedCfg(**default_cfg)
                default_cfg = DefaultCfg(
                    tags=deque([""]),
                    cfgs={"": pcfg},
                    is_pretrained=pcfg.has_weights,
                )
            else:
                raise TypeError(
                    f"{model_name}: default_cfgs entry must be DefaultCfg, "
                    f"PretrainedCfg, or dict. Got {type(default_cfg).__name__}."
                )

        # Validate
        if not default_cfg.tags:
            raise ValueError(f"{model_name}: DefaultCfg must have at least one tag.")
        if default_cfg.tags[0] not in default_cfg.cfgs:
            raise ValueError(
                f"{model_name}: Default tag '{default_cfg.tags[0]}' not in cfgs."
            )

        for tag_idx, tag in enumerate(default_cfg.tags):
            is_default = tag_idx == 0
            pretrained_cfg = default_cfg.cfgs[tag]
            model_name_tag = ".".join([model_name, tag]) if tag else model_name
            replace_items = dict(architecture=model_name, tag=tag if tag else None)
            if pretrained_cfg.hf_hub_id and pretrained_cfg.hf_hub_id == "JaxNN/":
                # auto-complete hub name w/ architecture.tag
                replace_items["hf_hub_id"] = pretrained_cfg.hf_hub_id + model_name_tag
            pretrained_cfg = replace(pretrained_cfg, **replace_items)

            if is_default:
                _model_pretrained_cfgs[model_name] = pretrained_cfg
                if pretrained_cfg.has_weights:
                    _model_has_pretrained.add(model_name)

            if tag:
                _model_pretrained_cfgs[model_name_tag] = pretrained_cfg
                if pretrained_cfg.has_weights:
                    _model_has_pretrained.add(model_name_tag)
                _model_with_tags[model_name].append(model_name_tag)
            else:
                _model_with_tags[model_name].append(model_name)

        _model_default_cfgs[model_name] = default_cfg

    return fn


def _expand_filter(filter: str):
    parsed = split_model_name_tag(filter)
    filter_base = parsed["model_name"]
    filter_tag = parsed["tag"]
    if not filter_tag:
        return [".".join([filter_base, "*"]), filter]
    else:
        return [filter]


def _natural_key(string_: str) -> List[Union[int, str]]:
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def list_models(
    filter: Union[str, List[str]] = "",
    module: Union[str, List[str]] = "",
    pretrained: bool = False,
    exclude_filters: Union[str, List[str]] = "",
    name_matches_cfg: bool = False,
    include_tags: Optional[bool] = None,
) -> List[str]:
    """Return list of available model names, sorted alphabetically

    Args:
        filter - Wildcard filter string that works with fnmatch
        module - Limit model selection to a specific submodule (ie 'vision_transformer')
        pretrained - Include only models with valid pretrained weights if True
        exclude_filters - Wildcard filters to exclude models after including them with filter
        name_matches_cfg - Include only models w/ model_name matching default_cfg name (excludes some aliases)
        include_tags - Include pretrained tags in model names (model.tag). Defaults
            set to True when pretrained=True else False (default: bool)

    Returns:
        models - The sorted list of models

    Example:
        model_list('gluon_resnet*') -- returns all models starting with 'gluon_resnet'
        model_list('*resnext*, 'resnet') -- returns all models with 'resnext' in 'resnet' module
    """
    if filter:
        include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
    else:
        include_filters = []

    if include_tags:
        include_tags = pretrained

    if not module:
        all_models: Set[str] = set(_model_entrypoints.keys())
    else:
        assert isinstance(module, Sequence)
        all_models: Set[str] = set()
        for m in module:
            all_models.update(_module_to_models[m])
    all_models = all_models

    if include_tags:
        models_with_tags: Set[str] = set()
        for m in all_models:
            models_with_tags.update(_model_with_tags[m])
        all_models = models_with_tags
        include_filters = [ef for f in include_filters for ef in _expand_filter(f)]
        exclude_filters = [ef for f in exclude_filters for ef in _expand_filter(f)]

    if include_filters:
        models: Set[str] = set()
        for f in include_filters:
            include_models = fnmatch.filter(all_models, f)
            if len(include_models):
                models = models.union(include_models)
    else:
        models = all_models

    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)  # exclude these models
            if len(exclude_models):
                models = models.difference(exclude_models)

    if pretrained:
        models = _model_has_pretrained.intersection(models)

    if name_matches_cfg:
        models = set(_model_pretrained_cfgs).intersection(models)

    return sorted(models, key=_natural_key)


def list_pretrained(
    filter: Union[str, List[str]] = "",
    exclude_filters: str = "",
) -> List[str]:
    return list_models(
        filter=filter,
        pretrained=True,
        exclude_filters=exclude_filters,
        include_tags=True,
    )


def is_model(model_name: str) -> bool:
    """Check if a model name exists"""
    arch_name = get_arch_name(model_name=model_name)
    return arch_name in _model_entrypoints


def is_model_in_modules(
    model_name: str, module_names: Union[Tuple[str, ...], List[str], Set[str]]
) -> bool:
    """Check if a model exists within a subset of modules

    Args:
        model_name - name of model to check
        module_names - names of modules to search in
    """
    arch_name = get_arch_name(model_name)
    assert isinstance(module_names, (tuple, list, set))
    return any(arch_name in _module_to_models[n] for n in module_names)


def is_model_pretrained(model_name: str) -> bool:
    return model_name in _model_has_pretrained


def get_pretrained_cfg(
    model_name: str,
    allow_unregistered: bool = True,
) -> Optional[PretrainedCfg]:
    """Look up PretrainedCfg by model name, with or without a tag.

    Examples:
        get_pretrained_cfg('resnet34')          → default PretrainedCfg
        get_pretrained_cfg('resnet34.a1_in1k')  → specific tag PretrainedCfg
    """
    if model_name in _model_pretrained_cfgs:
        return deepcopy(_model_pretrained_cfgs[model_name])

    # Try base name fallback
    arch_name = get_arch_name(model_name)
    if arch_name in _model_pretrained_cfgs:
        return deepcopy(_model_pretrained_cfgs[arch_name])

    # Fall back to DefaultCfg
    if arch_name in _model_default_cfgs:
        default_cfg = _model_default_cfgs[arch_name]
        if isinstance(default_cfg, DefaultCfg) and default_cfg.cfgs:
            return deepcopy(default_cfg.default)

    if allow_unregistered:
        return None

    raise RuntimeError(f"No pretrained config for '{model_name}'")


def get_pretrained_cfg_value(model_name: str, cfg_key: str) -> Optional[Any]:
    """Get a specific model default_cfg value by key. None if key doesn't exist."""
    cfg = get_pretrained_cfg(model_name, allow_unregistered=True)
    return getattr(cfg, cfg_key, None)


def get_arch_pretrained_cfgs(model_name: str) -> Dict[str, PretrainedCfg]:
    parsed = split_model_name_tag(model_name)
    arch_name = parsed["model_name"]
    model_names = _model_with_tags[arch_name]
    cfgs = {m: _model_pretrained_cfgs[m] for m in model_names}
    return cfgs


def generate_default_cfgs(cfgs: Dict[str, Union[Dict[str, Any], PretrainedCfg]]):
    out = defaultdict(DefaultCfg)
    default_set = set()  # no tag and tags ending with * are prioritized as default

    for k, v in cfgs.items():
        if isinstance(v, dict):
            v = PretrainedCfg(**v)
        has_weights = v.has_weights

        parsed = split_model_name_tag(k)
        model = parsed["model_name"]
        tag = parsed["tag"]

        is_default_set = model in default_set
        priority = (has_weights and not tag) or (
            tag.endswith("*") and not is_default_set
        )
        tag = tag.strip("*")

        default_cfg = out[model]

        if priority:
            default_cfg.tags.appendleft(tag)
            default_set.add(model)
        elif has_weights and not default_cfg.is_pretrained:
            default_cfg.tags.appendleft(tag)
        else:
            default_cfg.tags.append(tag)

        if has_weights:
            default_cfg.is_pretrained = True

        default_cfg.cfgs[tag] = v

    return out


def list_modules() -> List[str]:
    """Return list of module names that contain models / model entrypoints"""
    modules = _module_to_models.keys()
    return sorted(modules)
