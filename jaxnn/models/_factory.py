"""JaxNN model factory

Model name conventions

``resnet34``
    Plain registry lookup.

``resnet34.a1_in1k``
    Registry lookup with pretrained tag.

``hf-hub:JaxNN/resnet34.a1_in1k``  (or legacy ``hf_hub:...``)
    Load architecture + weights from a Hugging Face Hub repository.

``local-dir:/path/to/resnet34.a1_in1k``
    Load architecture + weights from a local JaxNN checkpoint directory.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlsplit

from jaxnn.models._builder import build_model_with_cfg
from jaxnn.models._hub import load_model_config_from_hf, load_model_config_from_path
from jaxnn.models._pretrained import PretrainedCfg
from jaxnn.models._registry import is_model, model_entrypoint, split_model_name_tag

_logger = logging.getLogger(__name__)

__all__ = ["parse_model_name", "safe_model_name", "create_model"]


def parse_model_name(model_name: str) -> Tuple[Optional[str], str]:
    """Parse source and name from a potentially-prefixed model name.

    Returns (source, bare_name) where source is one of:
        None         - plain registry name
        'hf-hub'     - Hugging Face Hub repo id
        'local-dir'  - local directory path
    """
    if model_name.startswith("hf_hub"):
        model_name = model_name.replace("hf_hub", "hf-hub", 1)

    parsed = urlsplit(model_name)

    if parsed.scheme not in ("", "hf-hub", "local-dir"):
        raise ValueError(
            f"Unknown model name scheme {parsed.scheme!r} in {model_name!r}. "
            "Supported: '' (registry), 'hf-hub:', 'local-dir:'."
        )

    if parsed.scheme in ("hf-hub", "local-dir"):
        return parsed.scheme, parsed.path

    return None, parsed.path


def safe_model_name(model_name: str, remove_source: bool = True) -> str:
    """Return a filesystem-safe version of a model name."""
    if remove_source:
        _, model_name = parse_model_name(model_name)
    return model_name.replace("/", "_").replace(":", "_")


def create_model(
    model_name: str,
    pretrained: bool = False,
    pretrained_cfg: Optional[Union[str, Dict[str, Any], PretrainedCfg]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Any:
    """Create a JaxNN model.

    Parameters
    ----------
    model_name:
        Supports four forms:
        - ``'resnet34'``                              plain registry name
        - ``'resnet34.a1_in1k'``                      registry + pretrained tag
        - ``'hf-hub:JaxNN/resnet34.a1_in1k'``         load from HF Hub
        - ``'local-dir:/path/to/resnet34.a1_in1k'``   load from local dir

    pretrained:
        Load pretrained weights.  Implied True for hf-hub: and local-dir:.

    pretrained_cfg:
        Override the entire pretrained config.

    pretrained_cfg_overlay:
        Shallow-merge on top of resolved pretrained_cfg (highest priority).

    checkpoint_path:
        Local JaxNN checkpoint directory.  config.json is read and merged
        into model.pretrained_cfg; weights are loaded from state/.

    cache_dir:
        Override cache directory for Hub downloads.
    """
    # 1. Parse scheme from model name
    source, model_name_bare = parse_model_name(model_name)

    # 2. Source-specific config loading
    if source == "hf-hub":
        pretrained_cfg, model_name_bare, _cfg_model_args = load_model_config_from_hf(
            model_name_bare, cache_dir=cache_dir
        )
        if _cfg_model_args:
            pretrained_cfg_overlay = {
                **_cfg_model_args,
                **(pretrained_cfg_overlay or {}),
            }
        pretrained = True

    elif source == "local-dir":
        _dir = checkpoint_path if checkpoint_path is not None else model_name_bare
        pretrained_cfg, model_name_bare, _cfg_model_args = load_model_config_from_path(
            _dir
        )
        if _cfg_model_args:
            pretrained_cfg_overlay = {
                **_cfg_model_args,
                **(pretrained_cfg_overlay or {}),
            }
        pretrained = True
        checkpoint_path = None  # consumed - prevent double load in builder

    # 3. Registry lookup
    #
    # split_model_name_tag returns a TUPLE (model_name, tag) - not a dict.
    # e.g. 'resnet152d.ra2_in1k'  -> ('resnet152d', 'ra2_in1k')
    #      'resnet34'             -> ('resnet34', '')
    #
    # The pretrained_tag is critical: it selects the correct registered
    # pretrained_cfg entry (mean, std, input_size, crop_pct, …).
    # Without it, resolve_pretrained_cfg falls back to a bare PretrainedCfg()
    # with library defaults instead of the checkpoint-specific values.
    data = split_model_name_tag(model_name_bare)
    model_name_only = data.get("model_name", "")
    pretrained_tag = data.get("tag", "")
    if not is_model(model_name_only):
        raise RuntimeError(
            f"Unknown model {model_name_only!r}. "
            "Check jaxnn.list_models() for available names.\n"
            f"  model_name={model_name!r}, parsed bare={model_name_bare!r}"
        )

    create_fn = model_entrypoint(model_name_only)

    # Pass the tag as the pretrained_cfg selector so resolve_pretrained_cfg
    # looks up 'resnet152d.ra2_in1k' rather than just 'resnet152d'.
    if pretrained_tag and pretrained_cfg is None:
        pretrained_cfg = pretrained_tag

    # 4. Build (delegates to build_model_with_cfg via the factory fn)
    return create_fn(
        pretrained=pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay=pretrained_cfg_overlay,
        checkpoint_path=checkpoint_path,
        cache_dir=cache_dir,
        **kwargs,
    )
