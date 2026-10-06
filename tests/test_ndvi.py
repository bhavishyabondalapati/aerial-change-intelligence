import numpy as np

from aci.ndvi import has_offset, ndvi, remove_offset, scene_ndvi, stress_mask, valid_from_scl


def test_ndvi_values():
    assert abs(float(ndvi(np.array([100]), np.array([300]))[0]) - 0.5) < 1e-6


def test_stress_flags_only_vegetated_drops_and_respects_clouds():
    before = np.array([0.8, 0.8, 0.1, 0.8])
    after = np.array([0.4, 0.75, 0.0, 0.4])
    valid = np.array([True, True, True, False])
    mask, _ = stress_mask(before, after, valid)
    assert mask.tolist() == [True, False, False, False]


def test_scl_cloud_pixels_invalid():
    assert valid_from_scl(np.array([4, 9, 3])).tolist() == [True, False, False]


def test_offset_removed_only_for_baseline_04_and_later():
    dn = np.array([1100, 1300, 900], np.uint16)
    assert remove_offset(dn, "05.11").tolist() == [100, 300, 0]  # new baseline: subtract 1000, never below 0
    assert remove_offset(dn, "02.12").tolist() == [1100, 1300, 900]  # old baseline: untouched
    assert has_offset("04.00") and not has_offset("03.01")


def test_scene_ndvi_same_field_gives_same_answer_in_both_baselines():
    old = {"red": np.array([100]), "nir": np.array([300]), "baseline": "02.12", "date": "2021-03-01"}
    new = {"red": np.array([1100]), "nir": np.array([1300]), "baseline": "05.11", "date": "2025-03-01"}
    assert abs(float(scene_ndvi(old, "old")[0]) - 0.5) < 1e-6
    assert abs(float(scene_ndvi(new, "new")[0]) - 0.5) < 1e-6
    # Without the fix the new scene would look much less green: (1300-1100)/(2400) = 0.083
    assert float(ndvi(new["red"], new["nir"])[0]) < 0.1
