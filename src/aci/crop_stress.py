"""Crop-stress track: NDVI drop between two Sentinel-2 dates -> stress polygons -> affected hectares."""
import argparse
import json
from pathlib import Path

import numpy as np
from rasterio.transform import Affine

from aci.geo import area_summary, mask_to_polygons
from aci.ndvi import has_offset, scene_ndvi, stress_mask


def load(npz):
    z = np.load(npz)
    if "baseline" not in z.files:
        raise ValueError(f"{npz} has no processing baseline (downloaded by an older version); re-run python -m aci.sentinel")
    return {k: z[k] for k in z.files}


def crop_to(scene, h, w):
    """Cut every 2-D array in a scene to h x w so two dates line up pixel for pixel."""
    return {k: (v[:h, :w] if v.ndim == 2 else v) for k, v in scene.items()}


def run(data_dir, out_dir, drop=0.15):
    b, a = load(Path(data_dir) / "before.npz"), load(Path(data_dir) / "after.npz")
    h, w = min(b["red"].shape[0], a["red"].shape[0]), min(b["red"].shape[1], a["red"].shape[1])
    b, a = crop_to(b, h, w), crop_to(a, h, w)
    nb, na = scene_ndvi(b, "before"), scene_ndvi(a, "after")
    valid = b["valid"] & a["valid"]
    mask, score = stress_mask(nb, na, valid, drop_threshold=drop)
    transform, crs = Affine(*b["transform"]), str(b["crs"])
    gdf = mask_to_polygons(mask, score, transform, crs, min_pixels=5)  # 5 px at 10 m = 0.05 ha
    summary = area_summary(gdf)
    summary.update(before_date=str(b["date"]), after_date=str(a["date"]), valid_fraction=float(valid.mean()),
                   drop_threshold=drop,
                   offset_removed={"before": has_offset(str(b["baseline"])), "after": has_offset(str(a["baseline"]))})
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
