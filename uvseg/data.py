"""Dataset and input construction.

Each sample is a 5-channel uint8 image (H, W, 5): R, G, B, illumination, depth.

* illumination: the L* channel of CIELAB (``skimage.color.rgb2lab``), in [0, 100]
  and stored as uint8 without rescaling, exactly as in the original experiments.
* depth: the Depth Anything map (uint8, 0-255), see ``uvseg.depth``.

All channels are resized to ``image_size`` (default 160 x 240) with bilinear
interpolation; masks are resized with nearest neighbour.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
from skimage import color
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(directory: str | os.PathLike, exclude: Sequence[str] = ()) -> list[Path]:
    """Sorted list of image files in ``directory``. ``exclude`` holds file names or stems to skip."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    excluded = set(exclude)
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
        and p.name not in excluded and p.stem not in excluded
    )


def find_by_stem(directory: str | os.PathLike, stem: str) -> Path | None:
    """Find ``<directory>/<stem>.<any image extension>``."""
    directory = Path(directory)
    for ext in IMAGE_EXTENSIONS + tuple(e.upper() for e in IMAGE_EXTENSIONS):
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def read_exclude_list(path: str | os.PathLike | None) -> list[str]:
    if not path:
        return []
    lines = Path(path).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def build_input(image_rgb: np.ndarray, depth: np.ndarray, image_size: tuple[int, int] = (160, 240)) -> np.ndarray:
    """Build the 5-channel network input.

    Args:
        image_rgb: uint8 RGB image (H, W, 3) at original resolution.
        depth: uint8 depth map (any resolution; normally the same as the image).
        image_size: output (height, width).

    Returns:
        uint8 array (height, width, 5) with channels R, G, B, illumination, depth.
    """
    height, width = image_size
    size = (width, height)  # OpenCV expects (width, height)
    illumination = color.rgb2lab(image_rgb)[:, :, 0]  # L* in [0, 100]

    channels = [cv2.resize(np.ascontiguousarray(image_rgb[:, :, c]), size, interpolation=cv2.INTER_LINEAR) for c in range(3)]
    channels.append(cv2.resize(illumination, size, interpolation=cv2.INTER_LINEAR))
    channels.append(cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR))
    # Stacking as float64 and casting truncates L* to an integer, as in the original code.
    return np.stack(channels, axis=-1).astype(np.float64).astype(np.uint8)


def load_mask(path: str | os.PathLike, image_size: tuple[int, int] = (160, 240)) -> np.ndarray:
    """Binary mask (height, width) float32 in {0, 1}. Accepts 0/255 or 0/1 masks."""
    mask = np.array(Image.open(path).convert("L"), dtype=np.float32)
    threshold = 127.0 if mask.max() > 1 else 0.0
    mask = (mask > threshold).astype(np.float32)
    height, width = image_size
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


class SegDataset(Dataset):
    """Segmentation dataset returning (image, mask) or (image, name) pairs.

    Images, masks and depth maps are matched by file stem, so ``v13_00769.jpg`` uses
    ``masks/v13_00769.png`` and ``depth/v13_00769.png``. Depth maps must exist; use
    ``uvseg.depth.prepare_depth_maps`` (called automatically by the training,
    evaluation and prediction scripts) to create missing ones.

    Args:
        image_dir: folder with RGB images.
        depth_dir: folder with depth maps.
        mask_dir: folder with binary masks, or ``None`` for inference.
        image_size: (height, width) of the network input.
        transform: an albumentations transform applied to ``image`` and ``mask``.
        exclude: file names or stems to leave out.
        return_name: also return the image file name (``(image, mask, name)``).
    """

    def __init__(self, image_dir, depth_dir, mask_dir=None, image_size=(160, 240), transform=None, exclude=(),
                 return_name=False):
        self.return_name = return_name
        self.images = list_images(image_dir, exclude)
        if not self.images:
            raise RuntimeError(f"No images found in {image_dir}")
        self.depth_dir = Path(depth_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.image_size = tuple(image_size)
        self.transform = transform

        missing_depth = [p.name for p in self.images if find_by_stem(self.depth_dir, p.stem) is None]
        if missing_depth:
            raise FileNotFoundError(f"{len(missing_depth)} depth map(s) missing in {self.depth_dir}, e.g. {missing_depth[:3]}")
        if self.mask_dir is not None:
            missing_masks = [p.name for p in self.images if find_by_stem(self.mask_dir, p.stem) is None]
            if missing_masks:
                raise FileNotFoundError(f"{len(missing_masks)} mask(s) missing in {self.mask_dir}, e.g. {missing_masks[:3]}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        image = np.array(Image.open(path).convert("RGB"))
        depth = np.array(Image.open(find_by_stem(self.depth_dir, path.stem)).convert("L"))
        x = build_input(image, depth, self.image_size)

        if self.mask_dir is None:
            if self.transform is not None:
                x = self.transform(image=x)["image"]
            return x, path.name

        mask = load_mask(find_by_stem(self.mask_dir, path.stem), self.image_size)
        if self.transform is not None:
            out = self.transform(image=x, mask=mask)
            x, mask = out["image"], out["mask"]
        return (x, mask, path.name) if self.return_name else (x, mask)
