"""Augmentations (albumentations), identical to the original training setup.

Inputs are already resized to the network size by ``SegDataset``. All five channels are
scaled from [0, 255] to [0, 1] by dividing by 255 (note that the illumination channel
therefore lies in [0, 0.39]).
"""

from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

_NORMALIZE = dict(mean=[0.0] * 5, std=[1.0] * 5, max_pixel_value=255.0)


def train_transform():
    return A.Compose(
        [
            # border_mode is set explicitly: it is the albumentations 1.x default used in the
            # paper, while albumentations 2.x changed the default to a constant border.
            A.Rotate(limit=35, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(**_NORMALIZE),
            ToTensorV2(),
        ],
        is_check_shapes=False,
    )


def eval_transform():
    return A.Compose([A.Normalize(**_NORMALIZE), ToTensorV2()], is_check_shapes=False)
