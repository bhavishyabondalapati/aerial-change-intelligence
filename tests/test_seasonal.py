import json

import numpy as np
import pytest

from aci.seasonal import anomaly_mask, normal_ndvi, run


def test_normal_ndvi_ignores_cloudy_years():
    ndvis = np.array([[0.6], [0.8], [0.0], [0.7]])  # year 3 is a cloud (value is junk)
    valids = np.array([[True], [True], [False], [True]])
    mean, std, count = normal_ndvi(ndvis, valids)
    assert count[0] == 3
    assert abs(mean[0] - 0.7) < 1e-6
    assert abs(std[0] - 0.1) < 1e-6  # sample std of 0.6, 0.8, 0.7


def test_same_drop_flagged_only_where_it_is_unusual():
    # Two pixels, both normally 0.7, both drop to 0.45 this year (a 0.25 drop)
    mean = np.array([0.7, 0.7], np.float32)
    std = np.array([0.06, 0.20], np.float32)  # pixel 0 is steady, pixel 1 swings a lot every year
    count = np.array([4, 4])
    mask, z, _ = anomaly_mask(np.array([0.45, 0.45]), np.array([True, True]), mean, std, count, k=2)
    assert mask.tolist() == [True, False]
    assert z[0] > 4 and z[1] < 2


def test_not_flagged_without_enough_history_or_vegetation_or_when_cloudy():
    mean = np.array([0.7, 0.2, 0.7], np.float32)  # pixel 1 is normally bare/water
    std = np.full(3, 0.05, np.float32)
    count = np.array([2, 4, 4])  # pixel 0 has only 2 clear years
    now = np.array([0.1, 0.0, 0.1])
    mask, _, _ = anomaly_mask(now, np.array([True, True, False]), mean, std, count)
    assert mask.tolist() == [False, False, False]


def test_min_std_floor_stops_tiny_dips_being_flagged():
    mask, _, _ = anomaly_mask(np.array([0.68]), np.array([True]), np.array([0.7]), np.array([0.001]), np.array([4]))
    assert not mask[0]


def _scene(path, ndvi_value, baseline, valid=True, year=2025):
    """Write a tiny fake saved scene whose NDVI is `ndvi_value` everywhere (red=1000, nir chosen to match)."""
    red = np.full((20, 20), 1000, np.float32)
    nir = red * (1 + ndvi_value) / (1 - ndvi_value)
    offset = 1000 if float(baseline) >= 4 else 0
    np.savez(path, red=(red + offset).astype(np.uint16), nir=(nir + offset).astype(np.uint16),
             valid=np.full((20, 20), valid), transform=np.array([10.0, 0, 500000, 0, -10.0, 1800000]),
             crs="EPSG:32644", date=f"{year}-02-20", cloud=1.0, baseline=baseline, tile="44QPD")


def test_run_end_to_end_with_mixed_baselines_and_a_skipped_year(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    # 2021 is old-format (no offset); 2022 was skipped (no file); 2023-24 new-format; 2025 crashes to 0.3
    for year, v, bl in [(2021, 0.70, "02.12"), (2023, 0.74, "05.00"), (2024, 0.66, "05.10"), (2025, 0.30, "05.11")]:
        _scene(data / f"{year}.npz", v, bl, year=year)
    s = run(data, tmp_path / "out", cloud_buffer=0)
    assert s["baseline_years"] == [2021, 2023, 2024]
    assert s["scenes"]["2021"]["offset_removed"] is False and s["scenes"]["2025"]["offset_removed"] is True
    assert abs(s["mean_ndvi_normal"] - 0.70) < 0.01  # proves the offset was handled per year
    assert abs(s["area_ha"] - 4.0) < 1e-6  # all 20x20 pixels x 0.01 ha flagged
    assert (tmp_path / "out" / "overview.png").exists()
    assert json.loads((tmp_path / "out" / "seasonal_summary.json").read_text())["target_year"] == 2025


def test_run_refuses_too_few_baseline_years(tmp_path):
    for year in (2024, 2025):
        _scene(tmp_path / f"{year}.npz", 0.7, "05.11", year=year)
    with pytest.raises(RuntimeError, match="baseline years"):
        run(tmp_path, tmp_path / "out")


def test_small_drop_not_flagged_even_if_pixel_is_unusually_steady():
    # Normal 0.70 with swing 0.02 (floored to 0.05): a 0.11 drop is z = 2.2 -> flagged, a 0.09 drop is z = 1.8 + below 0.1
    mean, std, count = np.full(3, 0.7, np.float32), np.full(3, 0.02, np.float32), np.full(3, 4)
    now = np.array([0.59, 0.61, 0.62])
    mask, z, _ = anomaly_mask(now, np.ones(3, bool), mean, std, count, k=2, min_std=0.05, min_drop=0.1)
    assert mask.tolist() == [True, False, False]
    # With a looser k the 0.09 drop passes the swing test, but the 0.1 minimum drop still blocks it
    mask, _, _ = anomaly_mask(now, np.ones(3, bool), mean, std, count, k=1.5, min_std=0.05, min_drop=0.1)
    assert mask.tolist() == [True, False, False]
