"""Configuration, reproducibility, checkpoints and I/O helpers."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from .data import SegDataset, list_images, read_exclude_list
from .depth import DEFAULT_DEPTH_MODEL, prepare_depth_maps

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def load_config(path: str | os.PathLike, overrides: dict | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for dotted_key, value in (overrides or {}).items():
        if value is None:
            continue
        node = cfg
        *parents, leaf = dotted_key.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return cfg


def split_dirs(cfg: dict, split: str) -> dict:
    """Image / mask / depth folders of a split, following the patterns in the config."""
    data = cfg["data"]
    root = Path(data["root"])
    return {key: root / data[f"{key}_dir"].format(split=split) for key in ("images", "masks", "depth")}


# --------------------------------------------------------------------------------------
# Reproducibility and device
# --------------------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str | None = None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def build_dataset(cfg: dict, split: str, transform, with_masks: bool = True, device: str | None = None,
                  exclude_list: str | None = None, return_name: bool = False) -> SegDataset:
    """Create the dataset of a split, computing any missing depth maps first."""
    dirs = split_dirs(cfg, split)
    exclude = read_exclude_list(exclude_list)
    images = list_images(dirs["images"], exclude)
    depth_cfg = cfg.get("depth", {})
    prepare_depth_maps(images, dirs["depth"], model_id=depth_cfg.get("model_id", DEFAULT_DEPTH_MODEL), device=device)
    return SegDataset(
        image_dir=dirs["images"],
        depth_dir=dirs["depth"],
        mask_dir=dirs["masks"] if with_masks else None,
        image_size=tuple(cfg["data"]["image_size"]),
        transform=transform,
        exclude=exclude,
        return_name=return_name,
    )


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------


def save_checkpoint(path: str | os.PathLike, model, optimizer=None, **extra) -> None:
    state = {"state_dict": model.state_dict(), **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_weights(model, path: str | os.PathLike, map_location="cpu") -> dict:
    """Load model weights from a training checkpoint ({'state_dict': ...}) or a plain state dict.

    Also accepts checkpoints written by the original notebook (``my_checkpoint.pth.tar``).
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def export_weights(checkpoint_path: str | os.PathLike, out_path: str | os.PathLike, half: bool = False) -> None:
    """Strip optimizer state from a checkpoint so it can be shared (about 4x smaller)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if half:
        state_dict = {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict}, out_path)


# --------------------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------------------


def overlay(image_rgb: np.ndarray, mask: np.ndarray, color=(0, 200, 255), alpha: float = 0.45) -> np.ndarray:
    """Blend a binary mask (H, W) over an RGB image (H, W, 3)."""
    out = image_rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def save_prediction(out_dir: Path, name: str, image_rgb: np.ndarray, pred_mask: np.ndarray) -> None:
    """Save the binary mask (0/255 PNG) and an overlay at the original image resolution."""
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = image_rgb.shape[:2]
    mask = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    stem = Path(name).stem
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask * 255)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), cv2.cvtColor(overlay(image_rgb, mask), cv2.COLOR_RGB2BGR))


def dump_config(cfg: dict, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False)
