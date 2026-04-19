"""JaxNN model factory - mirrors timm's _factory.py.

Model name conventions (identical to timm)
------------------------------------------
``resnet34``
    Plain registry lookup.

``resnet34.a1_in1k``
    Registry lookup with pretrained tag.

``hf-hub:JaxNN/resnet34.a1_in1k``  (or legacy ``hf_hub:…``)
    Load architecture + weights from a Hugging Face Hub repository.
    config.json in the repo drives both model construction and
    preprocessing metadata.

``local-dir:/path/to/resnet34.a1_in1k``
    Load architecture + weights from a local directory that was produced
    by the JaxNN converter (contains config.json + state/).
    config.json is read and merged into model.pretrained_cfg so that
    mean/std/input_size reflect the actual checkpoint, not the library default.

The old JaxNN convention ``'JaxNN/resnet34.a1_in1k'`` (no scheme prefix)
was NOT a valid timm model name - urlsplit gives it scheme='' and treats it
as an unrecognised registry name.  Use ``'hf-hub:JaxNN/resnet34.a1_in1k'``
instead, just as you would with timm.
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


# ---------------------------------------------------------------------------
# parse_model_name
# ---------------------------------------------------------------------------


def parse_model_name(model_name: str) -> Tuple[Optional[str], str]:
    """Parse source and name from a potentially-prefixed model name.

    Recognised schemes (identical to timm):
        ``''``          - plain registry name, source is None.
        ``'hf-hub'``    - Hugging Face Hub repo id follows the colon.
        ``'local-dir'`` - absolute or relative local directory path.

    Legacy ``hf_hub:…`` is silently normalised to ``hf-hub:…``.

    Examples::

        parse_model_name('resnet34')
        # -> (None, 'resnet34')

        parse_model_name('resnet34.a1_in1k')
        # -> (None, 'resnet34.a1_in1k')

        parse_model_name('hf-hub:JaxNN/resnet34.a1_in1k')
        # -> ('hf-hub', 'JaxNN/resnet34.a1_in1k')

        parse_model_name('local-dir:/mnt/c/.../resnet34.a1_in1k')
        # -> ('local-dir', '/mnt/c/.../resnet34.a1_in1k')
    """
    # Backwards-compat: normalise legacy hf_hub prefix
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

    # No scheme - plain registry name (possibly with pretrained tag)
    return None, parsed.path


def safe_model_name(model_name: str, remove_source: bool = True) -> str:
    """Return a filesystem-safe version of a model name.

    Strips the source prefix and replaces characters that are invalid in
    filenames (``/``, ``:``).
    """
    if remove_source:
        _, model_name = parse_model_name(model_name)
    return model_name.replace("/", "_").replace(":", "_")


# ---------------------------------------------------------------------------
# create_model
# ---------------------------------------------------------------------------


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
        Model identifier.  Supports four forms:

        * ``'resnet34'`` - registered name, random initialisation.
        * ``'resnet34.a1_in1k'`` - registered name with pretrained tag.
        * ``'hf-hub:JaxNN/resnet34.a1_in1k'`` - load config + weights from
          Hugging Face Hub.  ``pretrained=True`` is implied.
        * ``'local-dir:/path/to/dir'`` - load config + weights from a local
          JaxNN checkpoint directory.  ``pretrained=True`` is implied.

    pretrained:
        Load pretrained weights.  Always True for ``hf-hub:`` and
        ``local-dir:`` sources regardless of this flag.

    pretrained_cfg:
        Override the entire pretrained config (dict, string tag, or
        PretrainedCfg instance).

    pretrained_cfg_overlay:
        Shallow-merge these keys on top of the resolved pretrained_cfg.
        Highest priority - wins over config.json and registry defaults.

    checkpoint_path:
        Path to a *local* JaxNN checkpoint directory (contains
        ``config.json`` and ``state/``).  The directory's config.json is
        read and merged into ``model.pretrained_cfg``, then weights are
        loaded.  Use this when you already know the variant name but want
        to load weights from a local path instead of downloading.

        If you want both the architecture *and* the config sourced from a
        local directory without knowing the variant name upfront, use the
        ``'local-dir:/path'`` model_name prefix instead.

    cache_dir:
        Override the cache directory for Hub downloads.

    **kwargs:
        Forwarded to the model constructor (e.g. ``num_classes``,
        ``drop_rate``).
    """
    # 1. Parse the model name to extract source and the bare name
    source, model_name_bare = parse_model_name(model_name)

    # 2. Handle source-specific config loading
    if source == "hf-hub":
        # Load config.json from the Hub repo; this returns a PretrainedCfg
        # and the architecture name stored inside it.
        pretrained_cfg, model_name_bare, _local_model_args = load_model_config_from_hf(
            model_name_bare, cache_dir=cache_dir
        )
        # model_args from config.json are passed via pretrained_cfg_overlay so
        # build_model_with_cfg can forward them to the model constructor.
        if _local_model_args:
            pretrained_cfg_overlay = {
                **_local_model_args,
                **(pretrained_cfg_overlay or {}),
            }
        # Hub source always implies loading weights
        pretrained = True

    elif source == "local-dir":
        if checkpoint_path is not None:
            _dir = checkpoint_path
        else:
            _dir = model_name_bare
        pretrained_cfg, model_name_bare, _local_model_args = (
            load_model_config_from_path(_dir)
        )
        if _local_model_args:
            pretrained_cfg_overlay = {
                **_local_model_args,
                **(pretrained_cfg_overlay or {}),
            }
        pretrained = True
        # checkpoint_path already consumed - clear it so build_model_with_cfg
        # does not try to load weights a second time via the checkpoint branch.
        checkpoint_path = None

    # 3. Look up the model in the registry
    data = split_model_name_tag(model_name_bare)
    model_name_only = data.get("model_name", "")
    pretrained_tag = data.get("tag", "")
    if not is_model(model_name_only):
        raise RuntimeError(
            f"Unknown model {model_name_only!r}. "
            "Check jaxnn.list_models() for available names."
        )

    create_fn = model_entrypoint(model_name_only)

    # If a pretrained tag was embedded in the name and no explicit cfg was
    # given, pass the tag as the pretrained_cfg selector.
    if pretrained_tag and pretrained_cfg is None:
        pretrained_cfg = pretrained_tag

    # 4. Delegate to build_model_with_cfg (via the registered factory fn)
    return create_fn(
        pretrained=pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay=pretrained_cfg_overlay,
        checkpoint_path=checkpoint_path,
        cache_dir=cache_dir,
        **kwargs,
    )
