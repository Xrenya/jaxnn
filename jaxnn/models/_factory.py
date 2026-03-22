import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from flax import nnx

from ._hub import load_model_config_from_hf, load_model_config_from_path
from ._pretrained import PretrainedCfg
from ._registry import is_model, model_entrypoint, split_model_name_tag
from ._helpers import load_checkpoint

__all__ = ["parse_model_name", "create_model"]


def parse_model_name(model_name: str) -> Tuple[Optional[str], str]:
    if model_name.startswith('hf-hub:'):
        return 'hf-hub', model_name[len('hf-hub:'):]
    elif model_name.startswith('local-dir:'):
        return 'local-dir', model_name[len('local-dir:'):]
    else:
        model_name = os.path.split(model_name)[-1]
        return None, model_name


def create_model(
    model_name: str,
    pretrained: bool = False,
    pretrained_cfg: Optional[Union[str, Dict[str, Any], PretrainedCfg]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> nnx.Module:
    """Create a Flax model, optionally loading pretrained weights."""
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_id = parse_model_name(model_name)
    if model_source:
        assert not pretrained_cfg, (
            'pretrained_cfg should not be set when sourcing model from '
            'Hugging Face Hub or local directory.'
        )
        if model_source == 'hf-hub':
            pretrained_cfg, model_name, model_args = load_model_config_from_hf(
                model_id,
                cache_dir=cache_dir,
            )
        elif model_source == 'local-dir':
            pretrained_cfg, model_name, model_args = load_model_config_from_path(
                model_id,
            )
        else:
            assert False, f'Unknown model_source {model_source}'
        if model_args:
            for k, v in model_args.items():
                kwargs.setdefault(k, v)
    else:
        data = split_model_name_tag(model_id)
        model_name = data["model_name"]
        pretrained_tag = data["tag"]
        if pretrained_tag and not pretrained_cfg:
            pretrained_cfg = pretrained_tag

    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    create_fn = model_entrypoint(model_name)

    model = create_fn(
        pretrained=pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay=pretrained_cfg_overlay,
        cache_dir=cache_dir,
        **kwargs,
    )

    if checkpoint_path:
        model = load_checkpoint(model, checkpoint_path, pretrained_cfg)

    return model