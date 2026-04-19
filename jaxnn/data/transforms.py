"""Image transforms for JaxNN - torchvision backend.

Pipeline (identical structure to timm's transforms_imagenet_eval)
-----------------------------------------------------------------
center mode (square input_size):
    Resize(scale_size, antialias=True)          # shorter-edge resize
    CenterCrop((H, W))
    ToTensor()                                  # uint8 PIL => CHW float32 / 255.
    Normalize(mean, std)
    permute(1,2,0).numpy()                      # CHW => HWC for JAX

center mode (rectangular input_size):
    Resize((scale_H, scale_W), antialias=True)  # fixed-size resize
    CenterCrop((H, W))

squash mode:
    Resize((scale_H, scale_W), antialias=True)  # both axes, no aspect ratio
    CenterCrop((H, W))

border mode:
    TF.resize(img, max(scale_H, scale_W), ...)  # longest-edge resize
    TF.pad(img, ...)                            # zero-pad to (scale_H, scale_W)
    TF.center_crop(img, (H, W))

Output
------
np.ndarray  shape (H, W, C)  dtype float32   - channels-last for JAX/Flax.
Batch with ``jnp.array(arr)[None]`` or ``np.stack([...])``.
"""

import math
from typing import Tuple, Union

import numpy as np
from PIL import Image

import torchvision.transforms.functional as TF
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)


# Interpolation string => torchvision InterpolationMode
_INTERP_MAP = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
    "lanczos": InterpolationMode.LANCZOS,
    "linear": InterpolationMode.BILINEAR,
}


def _tv_interp(mode: str) -> InterpolationMode:
    key = mode.lower()
    if key not in _INTERP_MAP:
        raise ValueError(
            f"Unknown interpolation mode {mode!r}. Supported: {sorted(_INTERP_MAP)}"
        )
    return _INTERP_MAP[key]


# Padding helper for border crop mode
def _pad_to_size(img: Image.Image, target_h: int, target_w: int) -> Image.Image:
    """Zero-pad a PIL image to at least (target_h, target_w), centred.

    Mirrors timm's CenterCropOrPad: if the image already meets or exceeds the
    target on a given axis, that axis is left untouched.
    torchvision TF.pad expects padding as (left, top, right, bottom).
    """
    w, h = img.size
    pad_top = max((target_h - h) // 2, 0)
    pad_left = max((target_w - w) // 2, 0)
    pad_bottom = max(target_h - h - pad_top, 0)
    pad_right = max(target_w - w - pad_left, 0)
    if pad_top == pad_left == pad_bottom == pad_right == 0:
        return img
    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)


# Public transform class
class ImagenetEvalTransform:
    """Eval-time preprocessing identical to timm's ``transforms_imagenet_eval``.

    Uses torchvision for all spatial operations (Resize, CenterCrop).
    Returns a numpy (H, W, C) float32 array for JAX/Flax.

    Parameters
    ----------
    input_size:
        Target crop size as ``(H, W)`` or a single int for square crops.
    interpolation:
        Resize filter - ``'bicubic'``, ``'bilinear'``, ``'nearest'``,
        ``'lanczos'``.
    mean, std:
        Per-channel normalisation (RGB order).
    crop_pct:
        Fraction of the resize size used for the crop.
        ``scale_size = floor(crop_size / crop_pct)`` - timm's exact formula.
    crop_mode:
        ``'center'`` (default), ``'squash'``, or ``'border'``.
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 224,
        interpolation: str = "bicubic",
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        crop_pct: float = 0.875,
        crop_mode: str = "center",
    ) -> None:
        if isinstance(input_size, int):
            self.crop_h = self.crop_w = input_size
        else:
            self.crop_h, self.crop_w = int(input_size[0]), int(input_size[1])

        self.interpolation = interpolation
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.crop_pct = crop_pct
        self.crop_mode = crop_mode
        self._square = self.crop_h == self.crop_w

        self.scale_h = math.floor(self.crop_h / crop_pct)
        self.scale_w = math.floor(self.crop_w / crop_pct)

        tv_interp = _tv_interp(interpolation)
        self._tv_interp = tv_interp

        self._to_tensor = ToTensor()  # PIL uint8 => CHW float32 / 255.
        self._normalize = Normalize(list(mean), list(std))

        if crop_mode == "squash":
            self._spatial = Compose(
                [
                    Resize(
                        (self.scale_h, self.scale_w),
                        interpolation=tv_interp,
                        antialias=True,
                    ),
                    CenterCrop((self.crop_h, self.crop_w)),
                ]
            )

        elif crop_mode == "border":
            self._spatial = None  # built dynamically per image

        else:
            # center mode (default)
            if self._square:
                # Resize(int) => torchvision shorter-edge resize
                # Resize(scale_size) + CenterCrop(img_size)
                resize = Resize(self.scale_h, interpolation=tv_interp, antialias=True)
            else:
                # Resize((H,W)) => fixed two-axis resize, then crop
                resize = Resize(
                    (self.scale_h, self.scale_w),
                    interpolation=tv_interp,
                    antialias=True,
                )
            self._spatial = Compose(
                [
                    resize,
                    CenterCrop((self.crop_h, self.crop_w)),
                ]
            )

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Apply eval transform.

        Parameters
        ----------
        img : PIL.Image or numpy uint8 HWC array

        Returns
        -------
        np.ndarray  (H, W, C)  float32
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 1. Spatial transform (resize + crop) - returns PIL Image
        if self.crop_mode == "border":
            img = self._border_spatial(img)
        else:
            img = self._spatial(img)  # Compose returns PIL when input is PIL

        # 2. PIL => CHW float32 tensor => normalise
        tensor = self._normalize(self._to_tensor(img))

        # 3. CHW => HWC numpy for JAX/Flax
        return tensor.permute(1, 2, 0).numpy()

    # ------------------------------------------------------------------

    def _border_spatial(self, img: Image.Image) -> Image.Image:
        """Border-mode spatial ops: longest-edge resize, pad, crop."""
        # Resize so the longest edge equals max(scale_H, scale_W).
        # TF.resize(img, int) does shorter-edge resize - we want longest-edge,
        # so we compute the target size explicitly.
        scale_max = max(self.scale_h, self.scale_w)
        w, h = img.size
        if w >= h:
            new_w = scale_max
            new_h = round(h * scale_max / w)
        else:
            new_h = scale_max
            new_w = round(w * scale_max / h)
        img = TF.resize(
            img, (new_h, new_w), interpolation=self._tv_interp, antialias=True
        )
        # Zero-pad to (scale_H, scale_W)
        img = _pad_to_size(img, self.scale_h, self.scale_w)
        # Center-crop to final size
        return TF.center_crop(img, (self.crop_h, self.crop_w))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"size=({self.crop_h},{self.crop_w}), "
            f"scale_size=({self.scale_h},{self.scale_w}), "
            f"interp={self.interpolation}, "
            f"crop_pct={self.crop_pct}, "
            f"crop_mode={self.crop_mode}, "
            f"mean={self.mean}, "
            f"std={self.std})"
        )
