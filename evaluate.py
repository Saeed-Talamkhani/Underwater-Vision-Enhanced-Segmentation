"""Evaluate a trained UNet5C on one or more splits.

Examples:
    python evaluate.py --checkpoint weights/unet5c.pth --split test
    python evaluate.py --checkpoint weights/unet5c.pth --split val,test \
        --exclude-list data/excluded_images.txt --per-image-csv results/iou_per_image.csv

Splits given as a comma-separated list are evaluated together as one set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from uvseg.metrics import ConfusionCounts, binarize, format_metrics, per_image_iou
from uvseg.model import UNet5C
from uvseg.transforms import eval_transform
from uvseg.utils import build_dataset, get_device, load_config, load_weights, save_prediction, split_dirs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data-root", help="overrides data.root")
    p.add_argument("--split", default="test", help="split name, or comma-separated list (e.g. val,test)")
    p.add_argument("--exclude-list", help="text file with image names to leave out")
    p.add_argument("--per-image-csv", help="write per-image IoU to this CSV")
    p.add_argument("--save-dir", help="save predicted masks and overlays here")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config, {"data.root": args.data_root})
    device = get_device(args.device)
    splits = [s.strip() for s in args.split.split(",") if s.strip()]

    datasets = [build_dataset(cfg, s, eval_transform(), device=device, exclude_list=args.exclude_list,
                              return_name=True) for s in splits]
    image_dirs = {}
    for split, ds in zip(splits, datasets):
        for p in ds.images:
            image_dirs[p.name] = split_dirs(cfg, split)["images"]
    loader = DataLoader(ConcatDataset(datasets), batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg["data"]["num_workers"])

    model = UNet5C(out_channels=cfg["model"]["out_channels"], features=cfg["model"]["features"]).to(device)
    load_weights(model, args.checkpoint, map_location=device)
    model.eval()

    counts, rows = ConfusionCounts(), []
    for x, y, names in tqdm(loader, desc="evaluate"):
        preds = binarize(model(x.to(device)))
        target = y.float().unsqueeze(1).to(device)
        counts.update(preds, target)
        for name, iou in zip(names, per_image_iou(preds, target)):
            rows.append({"image": name, **iou})
        if args.save_dir:
            for name, pred in zip(names, preds[:, 0].cpu().numpy()):
                image = Image.open(image_dirs[name] / name).convert("RGB")
                save_prediction(Path(args.save_dir), name, np.array(image), pred)

    metrics = counts.compute()
    print(f"{'+'.join(splits)} ({len(rows)} images): {format_metrics(metrics)}")
    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))

    if args.per_image_csv:
        Path(args.per_image_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.per_image_csv, index=False)
        print(f"Per-image IoU written to {args.per_image_csv}")


if __name__ == "__main__":
    main()
