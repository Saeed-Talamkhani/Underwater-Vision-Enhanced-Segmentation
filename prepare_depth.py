"""Pre-compute Depth Anything depth maps (optional: train/evaluate/predict do this automatically).

Examples:
    # every split defined in the config (train, val, test)
    python prepare_depth.py --config configs/default.yaml --data-root data
    # a single folder
    python prepare_depth.py --images data/train/images --out data/train/depth
"""

from __future__ import annotations

import argparse

from uvseg.data import find_by_stem, list_images
from uvseg.depth import DEFAULT_DEPTH_MODEL, DepthEstimator, prepare_depth_maps
from uvseg.utils import get_device, load_config, split_dirs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data-root", help="overrides data.root")
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--images", help="a single image folder (instead of the config splits)")
    p.add_argument("--out", help="output folder for --images")
    p.add_argument("--model-id", default=None, help=f"default: {DEFAULT_DEPTH_MODEL}")
    p.add_argument("--overwrite", action="store_true", help="recompute existing depth maps")
    p.add_argument("--device")
    args = p.parse_args()

    cfg = load_config(args.config, {"data.root": args.data_root})
    model_id = args.model_id or cfg.get("depth", {}).get("model_id", DEFAULT_DEPTH_MODEL)
    device = get_device(args.device)

    if args.images:
        if not args.out:
            p.error("--out is required with --images")
        jobs = [(args.images, args.out)]
    else:
        jobs = [(split_dirs(cfg, s)["images"], split_dirs(cfg, s)["depth"]) for s in args.splits.split(",")]

    estimator = None  # loaded once, and only if some depth map is missing
    for image_dir, depth_dir in jobs:
        images = list_images(image_dir)
        todo = images if args.overwrite else [p for p in images if find_by_stem(depth_dir, p.stem) is None]
        if todo and estimator is None:
            estimator = DepthEstimator(model_id=model_id, device=device)
        n = prepare_depth_maps(todo, depth_dir, estimator=estimator, overwrite=True) if todo else 0
        print(f"{image_dir}: {n} depth map(s) written to {depth_dir}")


if __name__ == "__main__":
    main()
