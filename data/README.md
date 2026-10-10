# Data layout

Put your data here (or anywhere, and pass `--data-root`):

```
data/
├── train/
│   ├── images/   RGB images (.png / .jpg)
│   ├── masks/    binary masks with the same file stem (structure = 255, background = 0)
│   └── depth/    created automatically by Depth Anything on the first run
├── val/
│   └── images/ masks/ depth/
└── test/
    └── images/ masks/ depth/
```

Images, masks and depth maps are matched by file stem, so `train/images/v13_00769.jpg`
uses `train/masks/v13_00769.png`.

A flat layout such as `data/train_images`, `data/train_masks` also works: set
`images_dir: "{split}_images"` (and likewise for masks and depth) in `configs/default.yaml`.

If you already have depth maps, put them in the `depth/` folders and they will be reused.
