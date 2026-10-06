import numpy as np

from aci.ndvi import ndvi, stress_mask, valid_from_scl


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
