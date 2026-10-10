"""Underwater vision-enhanced image segmentation (Talamkhani & Liu, Automation in Construction, 2025)."""

from .model import UNet5C, UNET_5C, count_parameters

__all__ = ["UNet5C", "UNET_5C", "count_parameters"]
__version__ = "1.0.0"
