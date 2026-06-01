"""Data configuration utilities"""

from typing import Any, Dict, Optional

from .constants import (
    DEFAULT_CROP_MODE,
    DEFAULT_CROP_PCT,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)


def _extract_input_size(cfg: Dict[str, Any], use_test_size: bool) -> tuple:
    """Return (img_h, img_w, crop_pct) from a pretrained_cfg dict.

    JaxNN stores input_size in HWC order: (H, W, C).
    """
    if use_test_size and "test_input_size" in cfg:
        raw = cfg["test_input_size"]
        crop_pct = float(
            cfg.get("test_crop_pct") or cfg.get("crop_pct", DEFAULT_CROP_PCT)
        )
    else:
        raw = cfg.get("input_size", (224, 224, 3))
        crop_pct = float(cfg.get("crop_pct", DEFAULT_CROP_PCT))

    # raw is (H, W, C) — drop the channel dim
    img_h = int(raw[0])
    img_w = int(raw[1])
    return img_h, img_w, crop_pct


def resolve_model_data_config(
    model,
    args: Optional[Any] = None,
    use_test_size: bool = False,
) -> Dict[str, Any]:
    """Extract preprocessing config from a model's pretrained_cfg.

    Parameters
    ----------
    model:
        A JaxNN model with a ``pretrained_cfg`` attribute (set automatically
        by :func:`~jaxnn.models.create_model`).
    args:
        Optional namespace or dict of user overrides.  Any key present in the
        returned dict will override the value from ``pretrained_cfg`` when not
        None.  Useful for CLI-driven scripts.
    use_test_size:
        When True, prefer ``test_input_size`` / ``test_crop_pct`` over the
        training-time values — use for inference at test resolution.
        Defaults to False (training resolution) to mirror timm's default.

    Returns
    -------
    dict
        Keys: ``input_size (H,W)``, ``interpolation``, ``mean``, ``std``,
        ``crop_pct``, ``crop_mode``.  Pass with ``**`` to
        :func:`~jaxnn.data.create_transform`.

    Notes
    -----
    The ``pretrained_cfg`` on the model is populated by
    ``build_model_with_cfg`` from the registered ``default_cfgs`` entry for
    the variant (e.g. ``'resnet152d.ra2_in1k'``).  If ``create_model`` was
    called with a bare name and no pretrained tag (e.g.
    ``create_model('resnet152d')``), the cfg will contain library defaults
    rather than checkpoint-specific values.  Always include the tag:
    ``create_model('resnet152d.ra2_in1k', pretrained=True)``.
    """
    cfg: Dict[str, Any] = getattr(model, "pretrained_cfg", None) or {}
    return _build_data_cfg(cfg, args, use_test_size)


def resolve_data_config(
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    args: Optional[Any] = None,
    use_test_size: bool = False,
) -> Dict[str, Any]:
    """Lower-level variant that takes a pretrained_cfg dict directly.

    Useful when you have the config dict but not the model instance.
    Identical behaviour to :func:`resolve_model_data_config`.
    """
    cfg = pretrained_cfg or {}
    return _build_data_cfg(cfg, args, use_test_size)


def _build_data_cfg(
    cfg: Dict[str, Any],
    args: Optional[Any],
    use_test_size: bool,
) -> Dict[str, Any]:
    """Shared implementation for both resolve_* functions."""
    img_h, img_w, crop_pct = _extract_input_size(cfg, use_test_size)

    data_cfg = dict(
        input_size=(img_h, img_w),
        interpolation=cfg.get("interpolation", "bicubic"),
        mean=tuple(cfg.get("mean", IMAGENET_DEFAULT_MEAN)),
        std=tuple(cfg.get("std", IMAGENET_DEFAULT_STD)),
        crop_pct=crop_pct,
        crop_mode=cfg.get("crop_mode", DEFAULT_CROP_MODE),
    )

    if args is not None:
        overrides = args if isinstance(args, dict) else vars(args)
        for key in data_cfg:
            if key in overrides and overrides[key] is not None:
                data_cfg[key] = overrides[key]

    return data_cfg
