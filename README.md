# Underwater Vision-Enhanced Image Segmentation for Underwater Bridge Inspection

[![Paper](https://img.shields.io/badge/Automation%20in%20Construction-2025-0a5c8a)](https://doi.org/10.1016/j.autcon.2025.106230)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](requirements.txt)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Saeed-Talamkhani/underwater-vision-segmentation/blob/main/notebooks/train_colab.ipynb)

Official PyTorch implementation of

> **Underwater vision-enhanced image segmentation for supporting automated inspection of underwater bridge components**<br>
> Saeed Talamkhani and Kaijian Liu<br>
> *Automation in Construction*, 175, 106230, 2025. [doi:10.1016/j.autcon.2025.106230](https://doi.org/10.1016/j.autcon.2025.106230)

<!-- TODO: add a teaser figure (input image | depth | prediction) and uncomment:
<p align="center"><img src="assets/teaser.png" width="90%"></p>
-->

## Overview

Underwater inspection images suffer from light scattering, wavelength-dependent light
attenuation and low light, which makes it hard for segmentation models to localize bridge
components such as piers and piles. This repository segments bridge components in
underwater inspection images with **UNet5C**, a five-stream attention U-Net that learns from
quality-specific representations of each image instead of the raw RGB image alone.

Every image is decomposed into five maps, each encoded by its own CBAM-attention encoder.
The streams are fused at every scale and decoded by a U-Net decoder.

| Channel | Source | Encoder kernel |
|---|---|---|
| Red, green, blue | the RGB image, one stream per colour | 3×3, 5×5, 7×7 |
| Illumination | L\* channel of CIELAB | 3×3 |
| Depth | relative depth from [Depth Anything](https://github.com/LiheYoung/Depth-Anything) (ViT-L) | 3×3 |

The pipeline is end to end: depth maps are computed automatically the first time an image
is used and cached next to the data, so you only need images and masks.

<!-- TODO: copy the main results table from the paper, e.g.

## Results

| Method | mIoU | Structure IoU | F1 |
|---|---|---|---|
| U-Net (RGB) | | | |
| UNet5C (ours) | | | |
-->

## Installation

```bash
git clone https://github.com/Saeed-Talamkhani/underwater-vision-segmentation.git
cd underwater-vision-segmentation
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.x. A GPU is recommended for training. The
Depth Anything weights (about 1.3 GB) are downloaded from Hugging Face on first use.

## Quick start: segment your own images

Download the trained weights `unet5c.pth` from
[Hugging Face](https://huggingface.co/Saeed-Talamkhani/underwater-vision-segmentation) into `weights/`, then:

```bash
python predict.py --checkpoint weights/unet5c.pth --input path/to/images --output outputs/
```

For each image this writes a binary mask (`*_mask.png`, structure = 255) and an overlay
(`*_overlay.jpg`) at the original resolution. No depth maps or masks are needed.

## Data

Organise your data as below (details in [`data/README.md`](data/README.md)). Masks are
binary (structure = 255, background = 0) and share the file stem of their image.

```
data/
├── train/  images/  masks/
├── val/    images/  masks/
└── test/   images/  masks/
```

The underwater inspection images used in the paper are not distributed with this repository.
<!-- TODO: add a data availability statement, e.g. "available from the corresponding author on reasonable request". -->

## Training

```bash
python train.py --config configs/default.yaml --data-root data --output-dir runs/unet5c
```

[`configs/default.yaml`](configs/default.yaml) holds the settings used in the paper: input
160×240, batch size 16, Adam (lr 1e-4), focal loss (α = 1, γ = 2), up to 60 epochs with
early stopping after 20 epochs without improvement in validation loss, random rotation
(±35°) and horizontal / vertical flips, and mixed precision on GPU.

The best model is saved to `runs/unet5c/best.pth`, and per-epoch losses and validation
metrics to `runs/unet5c/log.csv`. Missing depth maps are created in `data/<split>/depth/`
before training starts; to compute them ahead of time, run `python prepare_depth.py --data-root data`.

To train on Google Colab, open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb).

## Evaluation

```bash
python evaluate.py --checkpoint runs/unet5c/best.pth --data-root data --split test \
    --per-image-csv results/iou_per_image.csv --save-dir results/predictions
```

Reported metrics are computed over the whole set (not averaged over batches): mean IoU,
IoU of the structure and background classes, frequency-weighted IoU, Dice / F1, precision,
recall and pixel accuracy. Several splits can be evaluated as one set with `--split val,test`;
[`data/excluded_images.txt`](data/excluded_images.txt) lists the images that were left out of
that combined evaluation in the original experiments (`--exclude-list data/excluded_images.txt`).

To share a trained model without the optimizer state:

```bash
python export_weights.py --checkpoint runs/unet5c/best.pth --out weights/unet5c.pth
```

## Repository structure

```
├── configs/default.yaml      training and data settings used in the paper
├── uvseg/
│   ├── model.py              UNet5C and the CBAM attention block
│   ├── data.py               5-channel input construction and dataset
│   ├── depth.py              Depth Anything depth maps (computed and cached automatically)
│   ├── transforms.py         augmentations
│   ├── losses.py             focal loss
│   ├── metrics.py            segmentation metrics
│   └── utils.py              config, checkpoints, visualisation
├── train.py · evaluate.py · predict.py
├── prepare_depth.py          optional: pre-compute depth maps
├── export_weights.py         strip optimizer state for sharing
└── notebooks/train_colab.ipynb
```

## Reproducibility notes

* **Depth maps** reproduce the official Depth Anything command used in the paper
  (`run.py --encoder vitl --pred-only --grayscale`): identical resizing, normalisation,
  upsampling and per-image 0–255 scaling, with the official ViT-L weights loaded through
  Hugging Face `transformers`. Existing depth maps placed in `data/<split>/depth/` are reused.
* **Albumentations** is pinned to 1.4.20, the version used in the paper; the rotation border
  mode is also set explicitly because version 2 changed its default.
* UNet5C has 154.7M parameters. Results can vary slightly between runs and hardware
  because of non-deterministic GPU operations.

## Citation

If you use this code, please cite:

```bibtex
@article{talamkhani2025underwater,
  title   = {Underwater vision-enhanced image segmentation for supporting automated inspection of underwater bridge components},
  author  = {Talamkhani, Saeed and Liu, Kaijian},
  journal = {Automation in Construction},
  volume  = {175},
  pages   = {106230},
  year    = {2025},
  doi     = {10.1016/j.autcon.2025.106230}
}
```

## Acknowledgments

* [Depth Anything](https://github.com/LiheYoung/Depth-Anything) (Apache-2.0) for monocular depth estimation.
* The CBAM block is adapted from [Jongchan/attention-module](https://github.com/Jongchan/attention-module) (MIT).
* The U-Net training scaffold follows [Aladdin Persson's U-Net tutorial](https://github.com/aladdinpersson/Machine-Learning-Collection) (MIT).
<!-- TODO: add funding acknowledgment if required by the grant. -->

## License

Released under the [MIT License](LICENSE).

## Contact

Saeed Talamkhani, Stevens Institute of Technology. Email stalamkh@stevens.edu or open an issue.
