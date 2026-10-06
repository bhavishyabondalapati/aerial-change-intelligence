"""Crop-stress track: NDVI drop between two Sentinel-2 dates -> stress polygons -> affected hectares."""
import argparse
import json
from pathlib import Path

import numpy as np
from rasterio.transform import Affine

from aci.geo import area_summary, mask_to_polygons
from aci.ndvi import ndvi, stress_mask


def load(npz):
    z = np.load(npz)
    return {k: z[k] for k in z.files}


def run(data_dir, out_dir, drop=0.15):
    b, a = load(Path(data_dir) / "before.npz"), load(Path(data_dir) / "after.npz")
    h, w = min(b["red"].shape[0], a["red"].shape[0]), min(b["red"].shape[1], a["red"].shape[1])
    crop = lambda z, k: z[k][:h, :w]
    nb, na = ndvi(crop(b, "red"), crop(b, "nir")), ndvi(crop(a, "red"), crop(a, "nir"))
    valid = crop(b, "valid") & crop(a, "valid")
    mask, score = stress_mask(nb, na, valid, drop_threshold=drop)
    transform, crs = Affine(*b["transform"]), str(b["crs"])
    gdf = mask_to_polygons(mask, score, transform, crs, min_pixels=5)  # 5 px at 10 m = 0.05 ha
    summary = area_summary(gdf)
    summary.update(before_date=str(b["date"]), after_date=str(a["date"]), valid_fraction=float(valid.mean()),
                   drop_threshold=drop)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out / "stress_polygons.geojson", driver="GeoJSON") if len(gdf) else None
    np.savez_compressed(out / "stress_arrays.npz", ndvi_before=nb, ndvi_after=na, mask=mask, score=score)
    (out / "crop_stress_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sentinel/godavari")
    ap.add_argument("--out", default="outputs/crop_stress")
    ap.add_argument("--drop", type=float, default=0.15)
    args = ap.parse_args()
    print(json.dumps(run(args.data, args.out, args.drop), indent=2))


if __name__ == "__main__":
    main()
