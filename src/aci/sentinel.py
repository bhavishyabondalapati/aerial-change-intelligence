"""Fetch cloud-free Sentinel-2 red/NIR crops from the free Microsoft Planetary Computer catalog (no account needed)."""
import argparse
from pathlib import Path

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

from aci.ndvi import valid_from_scl

# Godavari delta, Andhra Pradesh, India: rice paddies and aquaculture near Amalapuram (lon_min, lat_min, lon_max, lat_max)
GODAVARI_AOI = (81.95, 16.55, 82.10, 16.70)
CATALOG = "https://planetarycomputer.microsoft.com/api/stac/v1"


def covers(item, bbox):
    """True if the scene footprint fully contains the bbox (scenes at tile edges can cover only part of it)."""
    l, b, r, t = item.bbox
    return l <= bbox[0] and b <= bbox[1] and r >= bbox[2] and t >= bbox[3]


def find_best_scene(bbox, start, end, max_cloud=20, tile=None):
    """Least-cloudy Sentinel-2 L2A scene fully covering the bbox between two dates (optionally in a given MGRS tile)."""
    catalog = pystac_client.Client.open(CATALOG, modifier=pc.sign_inplace)
    items = catalog.search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).item_collection()
    items = [i for i in items if covers(i, bbox) and (tile is None or i.properties["s2:mgrs_tile"] == tile)]
    if len(items) == 0:
        raise RuntimeError(f"No Sentinel-2 scene fully covering the area under {max_cloud}% cloud for {start}..{end}; widen the dates")
    return min(items, key=lambda i: i.properties["eo:cloud_cover"])


def read_band(item, band, bbox, out_shape=None):
    """Read one band clipped to bbox (lon/lat). Returns (array, transform, crs)."""
    with rasterio.open(item.assets[band].href) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(left, bottom, right, top, src.transform)
        arr = src.read(1, window=win, out_shape=out_shape, resampling=rasterio.enums.Resampling.nearest)
        transform = src.window_transform(win)
        if out_shape:
            transform = transform * transform.scale(win.width / out_shape[1], win.height / out_shape[0])
        return arr, transform, src.crs


def fetch_scene(bbox, start, end, out_dir, tag, tile=None):
    """Download one scene; returns its MGRS tile so the second date can use the exact same grid."""
    item = find_best_scene(bbox, start, end, tile=tile)
    red, transform, crs = read_band(item, "B04", bbox)
    nir, _, _ = read_band(item, "B08", bbox, out_shape=red.shape)
    scl, _, _ = read_band(item, "SCL", bbox, out_shape=red.shape)  # 20m -> resampled to 10m grid
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{tag}.npz", red=red, nir=nir, valid=valid_from_scl(scl),
                        transform=np.array(transform)[:6], crs=crs.to_string(),
                        date=item.datetime.date().isoformat(), cloud=item.properties["eo:cloud_cover"],
                        baseline=item.properties["s2:processing_baseline"], tile=item.properties["s2:mgrs_tile"])
    print(f"{tag}: {item.datetime.date()} cloud={item.properties['eo:cloud_cover']:.1f}% "
          f"baseline={item.properties['s2:processing_baseline']} shape={red.shape} -> {out / (tag + '.npz')}")
    return item.properties["s2:mgrs_tile"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", nargs=2, default=["2024-12-01", "2025-01-15"], metavar=("START", "END"))
    ap.add_argument("--after", nargs=2, default=["2025-03-01", "2025-03-31"], metavar=("START", "END"))
    ap.add_argument("--bbox", nargs=4, type=float, default=GODAVARI_AOI)
    ap.add_argument("--out", default="data/sentinel/godavari")
    args = ap.parse_args()
    tile = fetch_scene(tuple(args.bbox), *args.before, args.out, "before")
    fetch_scene(tuple(args.bbox), *args.after, args.out, "after", tile=tile)


if __name__ == "__main__":
    main()
