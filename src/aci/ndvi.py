"""Vegetation-index math. NDVI = (NIR - Red) / (NIR + Red); healthy crops are high, bare soil/stressed crops are low."""
import numpy as np


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    red, nir = red.astype(np.float32), nir.astype(np.float32)
    return (nir - red) / np.maximum(nir + red, 1e-6)


def stress_mask(ndvi_before, ndvi_after, valid, drop_threshold=0.15, min_before=0.3):
    """Flag pixels that were vegetated before and lost at least `drop_threshold` NDVI.

    Returns (mask, score) where score = NDVI drop clipped to [0, 1] (bigger = stronger signal).
    `valid` is False where clouds/shadows make a pixel unusable.
    """
    drop = ndvi_before - ndvi_after
    mask = valid & (ndvi_before >= min_before) & (drop >= drop_threshold)
    score = np.clip(drop / 0.5, 0, 1).astype(np.float32)  # 0.5 NDVI drop -> score 1.0
    return mask, score


# Sentinel-2 Scene Classification Layer codes we treat as usable: vegetation(4), bare soil(5), water(6), unclassified(7)
GOOD_SCL = (4, 5, 6, 7)


def valid_from_scl(scl: np.ndarray) -> np.ndarray:
    return np.isin(scl, GOOD_SCL)
