"""Train UNet5C.

Example:
    python train.py --config configs/default.yaml --data-root data

Depth maps that are missing from data/<split>/depth are computed with Depth Anything
before training starts. The best model (lowest validation loss) is saved to
<output_dir>/best.pth and per-epoch results to <output_dir>/log.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from uvseg.losses import build_loss
from uvseg.metrics import ConfusionCounts, binarize, format_metrics
from uvseg.model import UNet5C, count_parameters
from uvseg.transforms import eval_transform, train_transform
from uvseg.utils import build_dataset, dump_config, get_device, load_config, save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data-root", help="overrides data.root")
    p.add_argument("--output-dir", help="overrides output_dir")
    p.add_argument("--epochs", type=int, help="overrides train.epochs")
    p.add_argument("--batch-size", type=int, help="overrides train.batch_size")
    p.add_argument("--lr", type=float, help="overrides train.lr")
    p.add_argument("--num-workers", type=int, help="overrides data.num_workers")
    p.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    return p.parse_args()


def run_epoch(model, loader, loss_fn, optimizer, scaler, device, use_amp):
    model.train()
    total = 0.0
    loop = tqdm(loader, leave=False)
    for x, y in loop:
        x = x.to(device)
        y = y.float().unsqueeze(1).to(device)
        with torch.autocast(device_type=device.split(":")[0], enabled=use_amp):
            loss = loss_fn(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total, counts = 0.0, ConfusionCounts()
    for x, y in loader:
        x = x.to(device)
        y = y.float().unsqueeze(1).to(device)
        logits = model(x)
        total += loss_fn(logits, y).item()
        counts.update(binarize(logits), y)
    return total / len(loader), counts.compute()


def main():
    args = parse_args()
    cfg = load_config(args.config, {
        "data.root": args.data_root, "output_dir": args.output_dir, "train.epochs": args.epochs,
        "train.batch_size": args.batch_size, "train.lr": args.lr, "data.num_workers": args.num_workers,
    })
    device = get_device(args.device)
    tcfg = cfg["train"]
    out_dir = Path(cfg["output_dir"])
    dump_config(cfg, out_dir / "config.yaml")

    train_ds = build_dataset(cfg, "train", train_transform(), device=device)
    val_ds = build_dataset(cfg, "val", eval_transform(), device=device)
    print(f"train: {len(train_ds)} images | val: {len(val_ds)} images | device: {device}")

    set_seed(cfg["seed"])
    model = UNet5C(out_channels=cfg["model"]["out_channels"], features=cfg["model"]["features"]).to(device)
    print(f"UNet5C: {count_parameters(model) / 1e6:.1f}M parameters")
    loss_cfg = dict(tcfg["loss"])
    loss_fn = build_loss(loss_cfg.pop("name"), **loss_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    use_amp = bool(tcfg.get("amp", True)) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loader_kw = dict(batch_size=tcfg["batch_size"], num_workers=cfg["data"]["num_workers"],
                     pin_memory=device.startswith("cuda"))
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    log_path = out_dir / "log.csv"
    best_val, patience_left = float("inf"), tcfg["patience"]
    with open(log_path, "w", newline="") as log_file:
        writer = None
        for epoch in range(1, tcfg["epochs"] + 1):
            train_loss = run_epoch(model, train_loader, loss_fn, optimizer, scaler, device, use_amp)
            val_loss, metrics = validate(model, val_loader, loss_fn, device)
            improved = val_loss < best_val
            print(f"epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
                  f"{format_metrics(metrics)}{'  *saved*' if improved else ''}")

            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
            if writer is None:
                writer = csv.DictWriter(log_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            log_file.flush()

            if improved:
                best_val, patience_left = val_loss, tcfg["patience"]
                save_checkpoint(out_dir / "best.pth", model, optimizer, epoch=epoch, val_loss=val_loss,
                                metrics=metrics, config=cfg)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"Early stopping: no improvement for {tcfg['patience']} epochs")
                    break

    print(f"Best validation loss {best_val:.4f}. Weights: {out_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
