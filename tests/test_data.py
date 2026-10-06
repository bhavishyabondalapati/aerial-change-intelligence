import numpy as np

from aci.data import normalize, random_crop_flip


def test_normalize_shape():
    assert normalize(np.zeros((8, 8, 3), np.uint8)).shape == (3, 8, 8)


def test_crop_keeps_arrays_aligned():
    a = np.random.randint(0, 255, (64, 64, 3), np.uint8)
    m = a[..., 0] > 128
    a2, b2, m2 = random_crop_flip(a, a.copy(), m, 32, train=True)
    assert a2.shape == (32, 32, 3) and m2.shape == (32, 32)
