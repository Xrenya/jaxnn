"""Data configuration utilities - mirrors timm.data.config."""

from typing import Any, Dict, Optional

from .constants import (
    DEFAULT_CROP_MODE,
    DEFAULT_CROP_PCT,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)


def resolve_model_data_config(
    model,
    args: Optional[Any] = None,
    use_test_size: bool = False,
) -> Dict[str, Any]:
    """Extract preprocessing config from a model's pretrained_cfg.

    Mirrors timm's ``resolve_model_data_config``.  The returned dict can be
    passed directly to :func:`create_transform`.

    Args:
        model: A JaxNN model with a ``pretrained_cfg`` attribute.
        args: Optional namespace / dict with user overrides.  Keys match the
              returned dict (``input_size``, ``interpolation``, …).
        use_test_size: If True, prefer ``test_input_size`` / ``test_crop_pct``
                       over the training-time values.

    Returns:
        Dict with keys: ``input_size``, ``interpolation``, ``mean``, ``std``,
        ``crop_pct``, ``crop_mode``.
    """
    cfg: Dict[str, Any] = getattr(model, "pretrained_cfg", None) or {}

    # ── input size ────────────────────────────────────────────────────────
    if use_test_size:
        input_size = cfg.get("test_input_size") or cfg.get("input_size", (224, 224, 3))
    else:
        input_size = cfg.get("input_size", (224, 224, 3))

    # JaxNN uses HWC tuples; extract (H, W) for the transforms
    if len(input_size) == 3:
        img_h, img_w = input_size[0], input_size[1]
    else:
        img_h = img_w = input_size[0]

    # ── preprocessing knobs ───────────────────────────────────────────────
    if use_test_size:
        crop_pct = cfg.get("test_crop_pct") or cfg.get("crop_pct", DEFAULT_CROP_PCT)
    else:
        crop_pct = cfg.get("crop_pct", DEFAULT_CROP_PCT)

    data_cfg = dict(
        input_size=(img_h, img_w),
        interpolation=cfg.get("interpolation", "bicubic"),
        mean=tuple(cfg.get("mean", IMAGENET_DEFAULT_MEAN)),
        std=tuple(cfg.get("std", IMAGENET_DEFAULT_STD)),
        crop_pct=float(crop_pct),
        crop_mode=cfg.get("crop_mode", DEFAULT_CROP_MODE),
    )

    # ── apply user overrides ──────────────────────────────────────────────
    if args is not None:
        overrides = args if isinstance(args, dict) else vars(args)
        for key in data_cfg:
            if key in overrides and overrides[key] is not None:
                data_cfg[key] = overrides[key]

    return data_cfg


def resolve_data_config(
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    args: Optional[Any] = None,
    use_test_size: bool = False,
) -> Dict[str, Any]:
    """Lower-level variant that takes a pretrained_cfg dict directly.

    Useful when you have the config but not the model instance.
    """
    cfg = pretrained_cfg or {}

    if use_test_size:
        input_size = cfg.get("test_input_size") or cfg.get("input_size", (224, 224, 3))
        crop_pct = cfg.get("test_crop_pct") or cfg.get("crop_pct", DEFAULT_CROP_PCT)
    else:
        input_size = cfg.get("input_size", (224, 224, 3))
        crop_pct = cfg.get("crop_pct", DEFAULT_CROP_PCT)

    if len(input_size) == 3:
        img_h, img_w = input_size[0], input_size[1]
    else:
        img_h = img_w = input_size[0]

    data_cfg = dict(
        input_size=(img_h, img_w),
        interpolation=cfg.get("interpolation", "bicubic"),
        mean=tuple(cfg.get("mean", IMAGENET_DEFAULT_MEAN)),
        std=tuple(cfg.get("std", IMAGENET_DEFAULT_STD)),
        crop_pct=float(crop_pct),
        crop_mode=cfg.get("crop_mode", DEFAULT_CROP_MODE),
    )

    if args is not None:
        overrides = args if isinstance(args, dict) else vars(args)
        for key in data_cfg:
            if key in overrides and overrides[key] is not None:
                data_cfg[key] = overrides[key]

    return data_cfg
