import numpy as np

from aci.geo import area_summary, mask_to_polygons, pixel_size_transform


def test_area_from_pixels():
    mask = np.zeros((100, 100), bool)
    mask[10:30, 10:30] = True  # 20x20 px
    prob = np.where(mask, 0.9, 0.1).astype(np.float32)
    gdf = mask_to_polygons(mask, prob, pixel_size_transform(0.5, 100), "EPSG:32614")
    s = area_summary(gdf)
    assert s["regions"] == 1
    assert abs(s["area_m2"] - 400 * 0.25) < 1e-6  # 400 px * 0.5m * 0.5m = 100 m2
    assert abs(s["mean_confidence"] - 0.9) < 1e-6


def test_small_regions_dropped_and_empty_ok():
    mask = np.zeros((20, 20), bool)
    mask[0, 0] = True
    gdf = mask_to_polygons(mask, np.ones((20, 20), np.float32), pixel_size_transform(1, 20), "EPSG:32614", min_pixels=5)
    assert area_summary(gdf)["regions"] == 0
