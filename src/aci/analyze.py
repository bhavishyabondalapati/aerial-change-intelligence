"""End-to-end for one LEVIR-CD-style pair: predict mask -> polygons -> area in hectares. Writes a GeoJSON, summary and overlay PNG."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from aci.geo import area_summary, mask_to_polygons, pixel_size_transform
from aci.predict import load_model, predict_probability

PIXEL_SIZE_M = 0.5  # LEVIR-CD ground resolution; the images carry no real coordinates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/analysis")
    args = ap.parse_args()

    before = np.array(Image.open(args.before).convert("RGB"))
    after = np.array(Image.open(args.after).convert("RGB"))
    prob = predict_probability(load_model(args.ckpt), before, after)
    mask = prob > args.threshold
    # LEVIR-CD has no CRS, so use a local metric grid (EPSG:3857 is only a label here; areas come from pixel size)
    gdf = mask_to_polygons(mask, prob, pixel_size_transform(PIXEL_SIZE_M, mask.shape[0]), "EPSG:3857")
    summary = area_summary(gdf)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(gdf):
        gdf.to_file(out / "change_polygons.geojson", driver="GeoJSON")
    (out / "area_summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    for a, im, t in zip(ax, [before, after, prob, after], ["Before", "After", "Change probability", "Detected change"]):
        a.imshow(im, cmap="magma" if im.ndim == 2 else None, vmin=0, vmax=1) if im.ndim == 2 else a.imshow(im)
        a.set_title(t)
        a.axis("off")
    ax[3].contour(mask, levels=[0.5], colors="red", linewidths=1)
    fig.suptitle(f"{summary['regions']} regions, {summary['area_m2']:.0f} m² changed, mean confidence {summary['mean_confidence']:.2f}")
    fig.tight_layout()
    fig.savefig(out / "overlay.png", dpi=110)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
