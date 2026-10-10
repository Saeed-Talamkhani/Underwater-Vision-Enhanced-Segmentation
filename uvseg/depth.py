"""Monocular depth maps with Depth Anything (ViT-L).

This reproduces what the original experiments did with the official repository::

    python run.py --encoder vitl --img-path <images> --outdir <depth> --pred-only --grayscale

i.e. the same input resizing (lower-bound resize to 518 px, multiple of 14, bicubic),
ImageNet normalisation, bilinear upsampling back to the original resolution and
per-image min-max scaling to an 8-bit grayscale map. The network weights are the
official Depth Anything ViT-L weights, loaded through Hugging Face ``transformers``
(``LiheYoung/depth-anything-large-hf``) so no repository clone is needed.

Depth maps are cached as single-channel PNGs (``<image stem>.png``) so they are
computed only once per image.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_DEPTH_MODEL = "LiheYoung/depth-anything-large-hf"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _constrain_to_multiple_of(x: float, multiple_of: int, min_val: int = 0) -> int:
    y = int(np.round(x / multiple_of) * multiple_of)
    if y < min_val:
        y = int(math.ceil(x / multiple_of) * multiple_of)
    return y


def lower_bound_size(height: int, width: int, target: int = 518, multiple_of: int = 14) -> tuple[int, int]:
    """Output (height, width) of Depth Anything's ``Resize(keep_aspect_ratio=True,
    resize_method='lower_bound', ensure_multiple_of=14)``."""
    scale = max(target / height, target / width)
    new_h = _constrain_to_multiple_of(scale * height, multiple_of, min_val=target)
    new_w = _constrain_to_multiple_of(scale * width, multiple_of, min_val=target)
    return new_h, new_w


def preprocess_for_depth(image_rgb: np.ndarray, input_size: int = 518) -> torch.Tensor:
    """uint8 RGB image (H, W, 3) -> normalised float tensor (1, 3, h, w)."""
    image = image_rgb.astype(np.float64) / 255.0
    new_h, new_w = lower_bound_size(image.shape[0], image.shape[1], target=input_size)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1))).astype(np.float32)
    return torch.from_numpy(image).unsqueeze(0)


def postprocess_depth(predicted: torch.Tensor, height: int, width: int) -> np.ndarray:
    """Relative depth prediction (1, h, w) -> uint8 map (H, W) scaled to 0-255 per image."""
    depth = F.interpolate(predicted[None].float(), (height, width), mode="bilinear", align_corners=False)[0, 0]
    d_min, d_max = depth.min(), depth.max()
    if float(d_max - d_min) == 0.0:
        return np.zeros((height, width), dtype=np.uint8)
    depth = (depth - d_min) / (d_max - d_min) * 255.0
    return depth.cpu().numpy().astype(np.uint8)


class DepthEstimator:
    """Wraps Depth Anything for single-image inference.

    Example:
        >>> estimator = DepthEstimator()           # downloads weights on first use
        >>> depth = estimator(image_rgb)            # uint8 (H, W)
    """

    def __init__(self, model_id: str = DEFAULT_DEPTH_MODEL, device: str | None = None, input_size: int = 518):
        try:
            from transformers import AutoModelForDepthEstimation
        except ImportError as e:  # pragma: no cover
            raise ImportError("Depth estimation needs `transformers`: pip install transformers") from e

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        pixel_values = preprocess_for_depth(image_rgb, self.input_size).to(self.device)
        predicted = self.model(pixel_values=pixel_values).predicted_depth  # (1, h, w)
        return postprocess_depth(predicted, height, width)


def read_rgb(path: str | os.PathLike) -> np.ndarray:
    """Read an image as uint8 RGB (H, W, 3)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def prepare_depth_maps(
    image_paths: Iterable[Path],
    depth_dir: str | os.PathLike,
    estimator: DepthEstimator | None = None,
    model_id: str = DEFAULT_DEPTH_MODEL,
    device: str | None = None,
    overwrite: bool = False,
) -> int:
    """Compute and cache a depth map for every image that does not have one yet.

    Returns the number of depth maps written. The (large) depth model is loaded only
    if at least one map is missing.
    """
    from tqdm import tqdm

    from .data import find_by_stem  # local import to avoid a circular import

    depth_dir = Path(depth_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in image_paths if overwrite or find_by_stem(depth_dir, Path(p).stem) is None]
    if not todo:
        return 0

    if estimator is None:
        print(f"Computing {len(todo)} depth map(s) with {model_id} -> {depth_dir}")
        estimator = DepthEstimator(model_id=model_id, device=device)
    for path in tqdm(todo, desc="depth"):
        depth = estimator(read_rgb(path))
        cv2.imwrite(str(depth_dir / f"{Path(path).stem}.png"), depth)
    return len(todo)
