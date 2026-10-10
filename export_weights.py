"""Strip the optimizer state from a training checkpoint before sharing it.

Example:
    python export_weights.py --checkpoint runs/unet5c/best.pth --out weights/unet5c.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uvseg.utils import export_weights


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="training checkpoint, e.g. best.pth or my_checkpoint.pth.tar")
    p.add_argument("--out", required=True)
    p.add_argument("--half", action="store_true", help="store weights in float16 (half the size)")
    args = p.parse_args()

    export_weights(args.checkpoint, args.out, half=args.half)
    size_mb = Path(args.out).stat().st_size / 2**20
    print(f"Wrote {args.out} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
