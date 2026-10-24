"""Render the crop-stress result as a picture: NDVI before, NDVI after, flagged stress."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stress", default="outputs/crop_stress/stress_arrays.npz")
    ap.add_argument("--out", default="outputs/crop_stress/overview.png")
    args = ap.parse_args()
    z = np.load(args.stress)
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    for a, k, t in zip(ax[:2], ["ndvi_before", "ndvi_after"], ["NDVI before", "NDVI after"]):
        im = a.imshow(z[k], cmap="RdYlGn", vmin=0, vmax=0.9)
        a.set_title(t)
        a.axis("off")
    fig.colorbar(im, ax=ax[:2], shrink=0.7, label="NDVI")
    ax[2].imshow(z["ndvi_after"], cmap="gray", vmin=0, vmax=0.9)
    ax[2].imshow(np.ma.masked_where(~z["mask"], z["score"]), cmap="autumn_r", vmin=0, vmax=1)
    ax[2].set_title("Flagged vegetation loss")
    ax[2].axis("off")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
