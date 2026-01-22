import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlsplit

from flax import nnx

from ._pretrained import PretrainedCfg
from ._registry import is_model, model_entrypoint, split_model_name_tag


__all__ = ["parse_model_name", "create_model"]


def parse_model_name(model_name: str) -> Tuple[Optional[str], str]:
    parsed = urlsplit(model_name)
    assert parsed.scheme in ('', 'hf-hub', 'local-dir')
    if parsed.scheme == 'hf-hub':
        return parsed.scheme, parsed.path
    elif parsed.scheme == 'local-dir':
        return parsed.scheme, parsed.path
    else:
        model_name = os.path.split(parsed.path)[-1]
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
    """Create a model.

    Lookup model's entrypoint function and pass relevant args to create a new model.

    Tip:
        **kwargs will be passed through entrypoint fn to ``flaxm.models.build_model_with_cfg()``
        and then the model class __init__(). kwargs values set to None are pruned before passing.

    Args:
        model_name: Name of model to instantiate.
        pretrained: If set to `True`, load pretrained ImageNet-1k weights.
        pretrained_cfg: Pass in an external pretrained_cfg for model.
        pretrained_cfg_overlay: Replace key-values in base pretrained_cfg with these.
        checkpoint_path: Path of checkpoint to load _after_ the model is initialized.
        cache_dir: Override model cache dir for Hugging Face Hub and Torch checkpoints.

    Keyword Args:
        drop_rate (float): Classifier dropout rate for training.
        drop_path_rate (float): Stochastic depth drop rate for training.
        global_pool (str): Classifier global pooling type.

    Example:

    ```py
    >>> from timm import create_model

    >>> # Create a MobileNetV3-Large model with no pretrained weights.
    >>> model = create_model('mobilenetv3_large_100')

    >>> # Create a MobileNetV3-Large model with pretrained weights.
    >>> model = create_model('mobilenetv3_large_100', pretrained=True)
    >>> model.num_classes
    1000

    >>> # Create a MobileNetV3-Large model with pretrained weights and a new head with 10 classes.
    >>> model = create_model('mobilenetv3_large_100', pretrained=True, num_classes=10)
    >>> model.num_classes
    10

    >>> # Create a Dinov2 small model with pretrained weights and save weights in a custom directory.
    >>> model = create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, cache_dir="/data/my-models")
    >>> # Data will be stored at `/data/my-models/models--timm--vit_small_patch14_dinov2.lvd142m/`
    ```
    """
    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_id = parse_model_name(model_name)
    if model_source:
        assert not pretrained_cfg, 'pretrained_cfg should not be set when sourcing model from Hugging Face Hub.'
        if model_source == 'hf-hub':
            # For model names specified in the form `hf-hub:path/architecture_name@revision`,
            # load model weights + pretrained_cfg from Hugging Face hub.
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
        model_name, pretrained_tag = split_model_name_tag(model_id)
        if pretrained_tag and not pretrained_cfg:
            # a valid pretrained_cfg argument takes priority over tag in model name
            pretrained_cfg = pretrained_tag

    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        model = create_fn(
            pretrained=pretrained,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            cache_dir=cache_dir,
            **kwargs,
        )

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model
