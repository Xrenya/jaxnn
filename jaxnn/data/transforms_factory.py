"""Transform factory - mirrors timm.data.transforms_factory."""

from typing import Optional, Tuple, Union

from .constants import (
    DEFAULT_CROP_MODE,
    DEFAULT_CROP_PCT,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from .transforms import ImagenetEvalTransform


def create_transform(
    input_size: Union[int, Tuple[int, int]] = 224,
    is_training: bool = False,
    interpolation: str = "bicubic",
    mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
    crop_pct: Optional[float] = None,
    crop_mode: Optional[str] = None,
    # training-only args (accepted but ignored when is_training=False)
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
):
    """Create a preprocessing transform matching timm's API.

    For ``is_training=False`` returns an :class:`ImagenetEvalTransform` that:

    1. Resizes the shorter side to ``int(crop_size / crop_pct)``
    2. Center-crops to ``input_size``
    3. Converts to float32 HWC ``[0, 1]``
    4. Normalises with ``mean`` / ``std``

    The returned callable accepts a PIL Image or uint8 numpy array and
    returns a float32 numpy array of shape ``(H, W, C)``, ready for
    ``jnp.array()`` and batching.

    Args:
        input_size: Target ``(H, W)`` or single int for square crops.
        is_training: If True, training augmentations will be applied.
            Currently only eval transforms are implemented.
        interpolation: Resize interpolation - ``'bicubic'``, ``'bilinear'``,
            ``'nearest'``, ``'lanczos'``.
        mean: Per-channel normalisation mean (RGB order).
        std: Per-channel normalisation std (RGB order).
        crop_pct: Fraction of the resized image used for the crop.
            Defaults to :data:`DEFAULT_CROP_PCT` (0.875).
        crop_mode: ``'center'`` (default) or ``'squash'``.

    Returns:
        A callable ``transform(img) -> np.ndarray``.
    """
    crop_pct = crop_pct or DEFAULT_CROP_PCT
    crop_mode = crop_mode or DEFAULT_CROP_MODE

    if is_training:
        # Training augmentations are not yet implemented in the pure-numpy
        # stack.  Return the eval transform so the pipeline is still usable
        # (users can add their own augmentations before calling the transform).
        import warnings

        warnings.warn(
            "Training transforms are not yet implemented in jaxnn.data. "
            "Returning eval transform. Apply your own augmentations before "
            "calling this transform.",
            UserWarning,
            stacklevel=2,
        )

    return ImagenetEvalTransform(
        input_size=input_size,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        crop_mode=crop_mode,
    )
