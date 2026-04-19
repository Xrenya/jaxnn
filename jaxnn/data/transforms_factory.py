"""Transform factory

Usage
-------------
    from jaxnn.data import create_transform, resolve_data_config
    import jax.numpy as jnp

    cfg   = model.pretrained_cfg          # dict attached by build_model_with_cfg
    data  = resolve_data_config(cfg)      # picks test_input_size when present
    xform = create_transform(**data, is_training=False)

    img_np = xform(pil_image)             # np.ndarray (H, W, C) float32
    batch  = jnp.array(img_np)[None]      # (1, H, W, C)

Output format
-------------
Every transform returned by this module outputs a 
**numpy array of shape (H, W, C), dtype float32**.
Pass it to ``jnp.array()`` and add a batch dim with ``[None]``.
"""

import warnings
from typing import Optional, Tuple, Union

from .constants import (
    DEFAULT_CROP_MODE,
    DEFAULT_CROP_PCT,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from .transforms import ImagenetEvalTransform


def resolve_data_config(
    pretrained_cfg: dict,
    *,
    use_test_size: bool = True,
) -> dict:
    """Extract preprocessing kwargs from a JaxNN ``pretrained_cfg`` dict.

    JaxNN stores ``input_size`` in HWC order ``(H, W, C)``; the channel dim
    is stripped here so ``create_transform`` receives a plain ``(H, W)`` tuple.

    Parameters
    ----------
    pretrained_cfg:
        ``model.pretrained_cfg`` dict as attached by ``build_model_with_cfg``.
    use_test_size:
        If True and ``test_input_size`` is present, use it (with
        ``test_crop_pct``) instead of the training resolution.
        Set to False to always use training size/crop_pct.

    Returns
    -------
    dict
        Keys: ``input_size``, ``interpolation``, ``mean``, ``std``,
        ``crop_pct``, ``crop_mode`` — ready for ``**``-unpacking into
        ``create_transform``.
    """
    cfg = pretrained_cfg

    if use_test_size and "test_input_size" in cfg:
        # HWC tuple — drop channel dim
        raw = cfg["test_input_size"]
        input_size: Tuple[int, int] = (int(raw[0]), int(raw[1]))
        crop_pct = float(cfg.get("test_crop_pct", cfg.get("crop_pct", DEFAULT_CROP_PCT)))
    else:
        raw = cfg["input_size"]
        input_size = (int(raw[0]), int(raw[1]))
        crop_pct = float(cfg.get("crop_pct", DEFAULT_CROP_PCT))

    return dict(
        input_size=input_size,
        interpolation=cfg.get("interpolation", "bicubic"),
        mean=tuple(cfg.get("mean", IMAGENET_DEFAULT_MEAN)),
        std=tuple(cfg.get("std",  IMAGENET_DEFAULT_STD)),
        crop_pct=crop_pct,
        crop_mode=cfg.get("crop_mode", DEFAULT_CROP_MODE),
    )


def create_transform(
    input_size: Union[int, Tuple[int, int]] = 224,
    is_training: bool = False,
    interpolation: str = "bicubic",
    mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
    crop_pct:  Optional[float] = None,
    crop_mode: Optional[str] = None,
    # training-only kwargs, currently ignored
    scale: Optional[Tuple[float, float]] = None,
    ratio: Optional[Tuple[float, float]] = None,
    hflip: float = 0.5,
    vflip: float = 0.0,
    color_jitter: float = 0.4,
    auto_augment: Optional[str] = None,
    re_prob: float = 0.0,
    re_mode: str = "const",
    re_count: int = 1,
    no_aug: bool = False,
) -> ImagenetEvalTransform:
    """Create an eval preprocessing transform matching timm's API.

    The returned callable has the signature::

        transform(img: PIL.Image | np.ndarray) -> np.ndarray  # (H, W, C) float32

    Parameters
    ----------
    input_size:
        Target ``(H, W)`` or a single int for square crops.  Do **not** pass
        a three-tuple ``(H, W, C)`` — use ``resolve_data_config`` to strip
        the channel dim from ``pretrained_cfg["input_size"]`` automatically.
    is_training:
        When True a ``UserWarning`` is raised and the eval transform is
        returned.  Training augmentations are not yet implemented.
    interpolation:
        Resize filter passed to torchvision — ``'bicubic'``, ``'bilinear'``,
        ``'nearest'``, ``'lanczos'``.
    mean, std:
        Per-channel normalisation (RGB order).
    crop_pct:
        Crop fraction.  ``scale_size = floor(crop_size / crop_pct)``.
        Defaults to ``DEFAULT_CROP_PCT`` (0.875).
    crop_mode:
        ``'center'`` (default), ``'squash'``, or ``'border'``.

    Crop mode details
    -----------------
    ``'center'``
        Square ``input_size``: torchvision ``Resize(scale_size)`` (shorter-edge)
        then ``CenterCrop``.
        Rectangular ``input_size``: ``Resize((scale_H, scale_W))`` then
        ``CenterCrop``.
    ``'squash'``
        ``Resize((scale_H, scale_W))`` ignoring aspect ratio, then
        ``CenterCrop``.
    ``'border'``
        Longest-edge resize to ``max(scale_H, scale_W)``, zero-pad to
        ``(scale_H, scale_W)``, then ``CenterCrop``.
    """
    if is_training:
        warnings.warn(
            "Training transforms are not yet implemented in jaxnn.data. "
            "Returning eval transform. Apply your own augmentations before "
            "calling this transform.",
            UserWarning,
            stacklevel=2,
        )

    crop_pct = crop_pct  if crop_pct  is not None else DEFAULT_CROP_PCT
    crop_mode = crop_mode if crop_mode is not None else DEFAULT_CROP_MODE

    return ImagenetEvalTransform(
        input_size=input_size,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        crop_mode=crop_mode,
    )
