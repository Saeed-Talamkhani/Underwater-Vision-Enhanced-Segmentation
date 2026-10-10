"""Binary segmentation metrics accumulated over a whole dataset.

All metrics are computed from the dataset-level confusion matrix (not averaged over
batches). Class 1 is the foreground (mask value 255, the bridge structure) and class 0
the background.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ConfusionCounts:
    tp: int = 0  # predicted 1, true 1
    fp: int = 0  # predicted 1, true 0
    fn: int = 0  # predicted 0, true 1
    tn: int = 0  # predicted 0, true 0

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds, target = preds.bool(), target.bool()
        self.tp += int((preds & target).sum())
        self.fp += int((preds & ~target).sum())
        self.fn += int((~preds & target).sum())
        self.tn += int((~preds & ~target).sum())

    def compute(self, eps: float = 1e-8) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        total = tp + fp + fn + tn
        iou_fg = tp / (tp + fp + fn + eps)
        iou_bg = tn / (tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        n_fg, n_bg = tp + fn, tn + fp  # ground-truth pixel counts
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "dice": 2 * tp / (2 * tp + fp + fn + eps),
            "accuracy": (tp + tn) / (total + eps),
            "iou_background": iou_bg,
            "iou_structure": iou_fg,
            "miou": (iou_bg + iou_fg) / 2,
            "fwiou": (iou_bg * n_bg + iou_fg * n_fg) / (total + eps),
        }


def binarize(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > threshold).float()


def per_image_iou(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> list[dict]:
    """Background / structure / mean IoU for each image in a batch of shape (B, 1, H, W)."""
    rows = []
    for p, t in zip(preds.bool(), target.bool()):
        inter_fg, union_fg = (p & t).sum().item(), (p | t).sum().item()
        inter_bg, union_bg = (~p & ~t).sum().item(), (~p | ~t).sum().item()
        iou_bg, iou_fg = inter_bg / (union_bg + eps), inter_fg / (union_fg + eps)
        rows.append({"iou_background": iou_bg, "iou_structure": iou_fg, "miou": (iou_bg + iou_fg) / 2})
    return rows


def format_metrics(m: dict) -> str:
    keys = ["miou", "iou_structure", "iou_background", "fwiou", "dice", "f1", "precision", "recall", "accuracy"]
    return " | ".join(f"{k} {m[k]:.4f}" for k in keys if k in m)
