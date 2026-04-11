"""Image transforms for JaxNN - pure numpy/PIL.

All transforms operate on PIL Images and return numpy arrays (HWC, float32).
The pipeline matches timm's eval pipeline exactly so that pretrained weights
produce the same logits as the PyTorch originals.
"""

from typing import Tuple, Union

import numpy as np
from PIL import Image


# Resize helpers
_INTERP_MAP = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "nearest": Image.NEAREST,
    "lanczos": Image.LANCZOS,
}


def _pil_interp(mode: str) -> int:
    return _INTERP_MAP.get(mode.lower(), Image.BICUBIC)


def _size_tuple(size) -> Tuple[int, int]:
    """Normalise size to (H, W)."""
    if isinstance(size, int):
        return size, size
    return int(size[0]), int(size[1])


# Individual transform ops
def resize_shortest(
    img: Image.Image,
    size: int,
    interpolation: str = "bicubic",
) -> Image.Image:
    """Resize so the shorter side == size, keeping aspect ratio."""
    w, h = img.size
    if w <= h:
        new_w = size
        new_h = int(h * size / w)
    else:
        new_h = size
        new_w = int(w * size / h)
    return img.resize((new_w, new_h), _pil_interp(interpolation))


def center_crop(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Center-crop PIL image to (H, W)."""
    crop_h, crop_w = size
    w, h = img.size
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return img.crop((left, top, left + crop_w, top + crop_h))


def squash_resize(
    img: Image.Image,
    size: Tuple[int, int],
    interpolation: str = "bicubic",
) -> Image.Image:
    """Resize (squash) directly to (H, W) ignoring aspect ratio."""
    h, w = size
    return img.resize((w, h), _pil_interp(interpolation))


def to_numpy_float(img: Image.Image) -> np.ndarray:
    """PIL Image → float32 numpy HWC in [0, 1]."""
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return arr / 255.0


def normalize(
    img: np.ndarray,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> np.ndarray:
    """Normalize HWC float32 image in-place (broadcast over channels)."""
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    return (img - mean_arr) / std_arr


# Composed eval transform
class ImagenetEvalTransform:
    """Eval-time transform matching timm's transforms_imagenet_eval.

    Pipeline:
        1. Resize shorter side to `resize_size` (= crop_size / crop_pct)
        2. Center-crop to `crop_size`  (or squash if crop_mode='squash')
        3. Convert to float32 HWC in [0, 1]
        4. Normalize with mean / std

    Returns a numpy array of shape (H, W, C) float32, ready to be batched
    and passed to jnp.array().
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]],
        interpolation: str = "bicubic",
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        crop_pct: float = 0.875,
        crop_mode: str = "center",
    ) -> None:
        self.crop_h, self.crop_w = _size_tuple(input_size)
        self.interpolation = interpolation
        self.mean = mean
        self.std = std
        self.crop_pct = crop_pct
        self.crop_mode = crop_mode

        if crop_mode == "squash":
            self.resize_h = self.crop_h
            self.resize_w = self.crop_w
        else:
            # shorter side is resized so that the crop fits exactly
            scale_size = int(self.crop_h / crop_pct)
            self.resize_size = scale_size

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.crop_mode == "squash":
            img = squash_resize(img, (self.resize_h, self.resize_w), self.interpolation)
        else:
            img = resize_shortest(img, self.resize_size, self.interpolation)
            img = center_crop(img, (self.crop_h, self.crop_w))

        arr = to_numpy_float(img)
        arr = normalize(arr, self.mean, self.std)
        return arr

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"size=({self.crop_h},{self.crop_w}), "
            f"interp={self.interpolation}, "
            f"crop_pct={self.crop_pct}, "
            f"crop_mode={self.crop_mode})"
        )
