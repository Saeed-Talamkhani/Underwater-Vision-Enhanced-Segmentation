"""Segment bridge components in new underwater images (no masks or depth maps needed).

Examples:
    python predict.py --checkpoint weights/unet5c.pth --input samples/ --output outputs/
    python predict.py --checkpoint weights/unet5c.pth --input samples/pier_01.jpg --output outputs/

For every image this writes <name>_mask.png (0/255, original resolution) and
<name>_overlay.jpg. Depth maps are computed with Depth Anything and cached in
<output>/depth (or --depth-dir).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from uvseg.data import IMAGE_EXTENSIONS, build_input, find_by_stem, list_images
from uvseg.depth import DEFAULT_DEPTH_MODEL, prepare_depth_maps
from uvseg.metrics import binarize
from uvseg.model import UNet5C
from uvseg.transforms import eval_transform
from uvseg.utils import get_device, load_config, load_weights, save_prediction


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="an image file or a folder of images")
    p.add_argument("--output", default="outputs")
    p.add_argument("--depth-dir", help="where to cache depth maps (default: <output>/depth)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(args.device)
    image_size = tuple(cfg["data"]["image_size"])

    src = Path(args.input)
    if src.is_file():
        if src.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {src}")
        images = [src]
    else:
        images = list_images(src)
    out_dir = Path(args.output)
    depth_dir = Path(args.depth_dir) if args.depth_dir else out_dir / "depth"
    prepare_depth_maps(images, depth_dir, model_id=cfg.get("depth", {}).get("model_id", DEFAULT_DEPTH_MODEL),
                       device=device)

    model = UNet5C(out_channels=cfg["model"]["out_channels"], features=cfg["model"]["features"]).to(device)
    load_weights(model, args.checkpoint, map_location=device)
    model.eval()
    transform = eval_transform()

    for path in tqdm(images, desc="predict"):
        image = np.array(Image.open(path).convert("RGB"))
        depth = np.array(Image.open(find_by_stem(depth_dir, path.stem)).convert("L"))
        x = transform(image=build_input(image, depth, image_size))["image"].unsqueeze(0).to(device)
        pred = binarize(model(x), args.threshold)[0, 0].cpu().numpy()
        save_prediction(out_dir, path.name, image, pred)

    print(f"Saved {len(images)} prediction(s) to {out_dir}")


if __name__ == "__main__":
    main()
