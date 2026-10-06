"""Turn a pixel change mask into georeferenced polygons and compute affected area with GeoPandas."""
import geopandas as gpd
import numpy as np
from rasterio import features
from rasterio.transform import Affine, from_origin
from shapely.geometry import shape


def pixel_size_transform(pixel_size_m: float, height: int) -> Affine:
    """For images with no geo-info (LEVIR-CD): pretend a flat grid in metres, `pixel_size_m` per pixel."""
    return from_origin(0, height * pixel_size_m, pixel_size_m, pixel_size_m)


def mask_to_polygons(mask: np.ndarray, prob: np.ndarray, transform: Affine, crs, min_pixels: int = 10) -> gpd.GeoDataFrame:
    """One polygon per connected changed region, with its pixel count and mean confidence."""
    rows = []
    labeled = features.shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform, connectivity=8)
    inv = ~transform
    for geom, _ in labeled:
        poly = shape(geom)
        # Mean confidence inside this polygon: rasterise it back on the grid
        region = features.rasterize([(poly, 1)], out_shape=mask.shape, transform=transform).astype(bool)
        n = int(region.sum())
        if n < min_pixels:
            continue
        rows.append({"geometry": poly, "pixels": n, "confidence": float(prob[region].mean())})
    return gpd.GeoDataFrame(rows, columns=["geometry", "pixels", "confidence"], geometry="geometry", crs=crs)


def area_summary(gdf: gpd.GeoDataFrame) -> dict:
    """Areas are computed in a metric (projected) CRS. Geographic (lat/lon) data is reprojected to its UTM zone first."""
    if len(gdf) == 0:
        return {"regions": 0, "area_m2": 0.0, "area_ha": 0.0, "mean_confidence": 0.0, "max_region_m2": 0.0}
    g = gdf
    if g.crs is not None and g.crs.is_geographic:
        g = g.to_crs(g.estimate_utm_crs())
    areas = g.geometry.area
    w = areas / areas.sum()
    return {
        "regions": int(len(g)),
        "area_m2": float(areas.sum()),
        "area_ha": float(areas.sum() / 10_000),
        "mean_confidence": float((g["confidence"] * w).sum()),  # area-weighted
        "max_region_m2": float(areas.max()),
    }
