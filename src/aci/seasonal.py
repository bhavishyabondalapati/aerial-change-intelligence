"""Seasonal crop-stress check: compare one year's NDVI with each pixel's own "normal" from earlier years.

Same window (Feb 1 - Mar 10) every year and the same Sentinel-2 tile, so the crop calendar and the pixel grid match.
A 2025 pixel is flagged only if its NDVI fell further below its 2021-24 average than its usual year-to-year swing.
"""
import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rasterio.transform import Affine

from aci.crop_stress import crop_to, load
from aci.geo import area_summary, mask_to_polygons
from aci.ndvi import buffer_invalid, has_offset, scene_ndvi

WINDOW = ("02-01", "03-10")
REFERENCE_DAY = "03-01"  # within the window, prefer the clear scene nearest this date so every year is at the same crop stage


def fetch_year(year, bbox, out_dir, tile=None, min_usable=0.6, tries=5):
    """Save the scene nearest the reference day that is clear over the area. Returns it, or None (year skipped).

    Catalog cloud cover is for the whole 110 km tile, so up to `tries` candidates are checked against the area itself.
    """
    from aci.sentinel import find_scenes, read_scene, save_scene  # needs network packages; keep tests independent

    start, end = f"{year}-{WINDOW[0]}", f"{year}-{WINDOW[1]}"
    ref = date.fromisoformat(f"{year}-{REFERENCE_DAY}")
    candidates = sorted(find_scenes(bbox, start, end, tile=tile), key=lambda i: abs((i.datetime.date() - ref).days))
    for item in candidates[:tries]:
        scene = read_scene(item, bbox)
        if scene["valid"].mean() >= min_usable:
            save_scene(scene, Path(out_dir) / f"{year}.npz")
            return scene
        print(f"{year}: {scene['date']} only {scene['valid'].mean():.0%} usable over the area, trying next")
    print(f"{year}: no clear scene in {start}..{end}, skipping this year")
    return None


def normal_ndvi(ndvis, valids):
    """Per-pixel mean and year-to-year standard deviation of NDVI, using only cloud-free years.

    ndvis, valids: arrays shaped (years, H, W). Returns (mean, std, count); std is NaN where count < 2.
    """
    count = valids.sum(0)
    total = np.where(valids, ndvis, 0).sum(0)
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    sq = np.where(valids, (ndvis - mean) ** 2, 0).sum(0)
    std = np.where(count > 1, np.sqrt(sq / np.maximum(count - 1, 1)), np.nan)
    return mean.astype(np.float32), std.astype(np.float32), count


def anomaly_mask(ndvi_now, valid_now, mean, std, count, k=2.0, min_std=0.05, min_mean=0.3, min_years=3):
    """Flag pixels whose NDVI drop below normal is more than k times their usual year-to-year swing.

    min_std stops a pixel that happened to be very steady for 3-4 years from being flagged over a tiny dip.
    min_mean keeps only pixels that are normally vegetated (not water, roads or towns).
    Returns (mask, z, score): z = drop in units of normal swing; score = z scaled to 0..1 (z of 4 -> 1.0).
    """
    sigma = np.maximum(np.nan_to_num(std, nan=min_std), min_std)
    z = np.nan_to_num((mean - ndvi_now) / sigma, nan=0.0).astype(np.float32)
    mask = valid_now & (count >= min_years) & (np.nan_to_num(mean) >= min_mean) & (z > k)
    return mask, z, np.clip(z / 4, 0, 1).astype(np.float32)


def run(data_dir, out_dir, target=2025, k=2.0, cloud_buffer=5, min_years=3):
    scenes = {int(p.stem): load(p) for p in sorted(Path(data_dir).glob("*.npz"))}
    if target not in scenes:
        raise FileNotFoundError(f"No {target}.npz in {data_dir}; run python -m aci.seasonal --fetch first")
    baseline_years = [y for y in scenes if y < target]
    if len(baseline_years) < min_years:
        raise RuntimeError(f"Only {len(baseline_years)} clear baseline years ({baseline_years}); need at least {min_years}")
    tiles = {str(s["tile"]) for s in scenes.values()}
    if len(tiles) > 1:
        raise RuntimeError(f"Scenes come from different tiles {tiles}; pixels would not line up")

    h = min(s["red"].shape[0] for s in scenes.values())
    w = min(s["red"].shape[1] for s in scenes.values())
    scenes = {y: crop_to(s, h, w) for y, s in scenes.items()}
    ndvi = {y: scene_ndvi(s, str(y)) for y, s in scenes.items()}
    valid = {y: buffer_invalid(s["valid"], cloud_buffer) for y, s in scenes.items()}

    mean, std, count = normal_ndvi(np.stack([ndvi[y] for y in baseline_years]), np.stack([valid[y] for y in baseline_years]))
    mask, z, score = anomaly_mask(ndvi[target], valid[target], mean, std, count, k=k, min_years=min_years)

    t = scenes[target]
    gdf = mask_to_polygons(mask, score, Affine(*t["transform"]), str(t["crs"]), min_pixels=5)
    comparable = valid[target] & (count >= min_years) & (mean >= 0.3)  # normally-vegetated pixels we could judge
    summary = area_summary(gdf)
    summary.update(
        target_year=target, baseline_years=baseline_years, window=f"{WINDOW[0]}..{WINDOW[1]}", tile=tiles.pop(),
        reference_day=REFERENCE_DAY, k_sigma=k, cloud_buffer_m=cloud_buffer * 10,
        comparable_area_ha=float(comparable.sum() * 0.01),  # 10 m pixels = 0.01 ha
        flagged_fraction_of_comparable=float(mask.sum() / max(comparable.sum(), 1)),
        mean_ndvi_normal=float(mean[comparable].mean()) if comparable.any() else None,
        mean_ndvi_target=float(ndvi[target][comparable].mean()) if comparable.any() else None,
        median_normal_swing=float(np.nanmedian(std[comparable])) if comparable.any() else None,
        scenes={str(y): {"date": str(s["date"]), "processing_baseline": str(s["baseline"]),
                         "offset_removed": has_offset(str(s["baseline"])), "usable_fraction": float(valid[y].mean())}
                for y, s in scenes.items()},
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out / "anomaly_polygons.geojson", driver="GeoJSON") if len(gdf) else None
    (out / "seasonal_summary.json").write_text(json.dumps(summary, indent=2))
    plot(ndvi, valid, mean, std, mask, score, target, out / "overview.png")
    return summary


def plot(ndvi, valid, mean, std, mask, score, target, path):
    """Top row: each year's NDVI (grey = cloud). Bottom row: normal NDVI, normal swing, flagged pixels."""
    years = sorted(ndvi)
    fig = plt.figure(figsize=(3.2 * max(len(years), 3), 7.5))
    grid = fig.add_gridspec(2, max(len(years), 3) * 2)
    cmap = plt.get_cmap("RdYlGn").with_extremes(bad="lightgrey")
    for i, y in enumerate(years):
        ax = fig.add_subplot(grid[0, 2 * i : 2 * i + 2])
        ax.imshow(np.ma.masked_where(~valid[y], ndvi[y]), cmap=cmap, vmin=0, vmax=0.9)
        ax.set_title(f"{y}" + (" (target)" if y == target else ""))
        ax.axis("off")
    span = max(len(years), 3) * 2 // 3
    panels = [(mean, cmap, 0, 0.9, "Normal NDVI (baseline mean)"), (std, "viridis", 0, 0.2, "Normal year-to-year swing")]
    for j, (arr, cm, lo, hi, title) in enumerate(panels):
        ax = fig.add_subplot(grid[1, j * span : (j + 1) * span])
        im = ax.imshow(arr, cmap=cm, vmin=lo, vmax=hi)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.axis("off")
    ax = fig.add_subplot(grid[1, 2 * span : 3 * span])
    ax.imshow(ndvi[target], cmap="gray", vmin=0, vmax=0.9)
    ax.imshow(np.ma.masked_where(~mask, score), cmap="autumn_r", vmin=0, vmax=1)
    ax.set_title(f"{target}: drop beyond normal swing")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    from aci.sentinel import GODAVARI_AOI

    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download the yearly scenes first")
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--target", type=int, default=2025)
    ap.add_argument("--k", type=float, default=2.0, help="flag drops bigger than k x the pixel's normal swing")
    ap.add_argument("--cloud-buffer", type=int, default=5, help="pixels around clouds to also ignore (10 m each)")
    ap.add_argument("--data", default="data/sentinel/godavari_seasonal")
    ap.add_argument("--out", default="outputs/seasonal")
    args = ap.parse_args()
    if args.fetch:
        bbox = GODAVARI_AOI
        # Target year first; every other year must use its tile so all pixels line up
        first = fetch_year(args.target, bbox, args.data)
        if first is None:
            raise SystemExit(f"No clear scene for the target year {args.target}")
        for y in sorted(set(args.years) - {args.target}):
            fetch_year(y, bbox, args.data, tile=str(first["tile"]))
    print(json.dumps(run(args.data, args.out, args.target, args.k, args.cloud_buffer), indent=2))


if __name__ == "__main__":
    main()
