"""Vegetation-index math. NDVI = (NIR - Red) / (NIR + Red); healthy crops are high, bare soil/stressed crops are low."""
import numpy as np
from scipy import ndimage

# From processing baseline 04.00 (January 2022) on, ESA adds 1000 to every Sentinel-2 L2A value so that
# slightly negative reflectances can be stored. Older scenes (e.g. baseline 02.12 from 2021) have no offset.
BOA_OFFSET = 1000


def has_offset(baseline: str) -> bool:
    return float(baseline) >= 4.0


def remove_offset(dn: np.ndarray, baseline: str) -> np.ndarray:
    """Raw Sentinel-2 numbers -> offset-free values (still x10000 reflectance). Only touches baseline 04.00+."""
    x = dn.astype(np.float32)
    return np.clip(x - BOA_OFFSET, 0, None) if has_offset(baseline) else x


def scene_ndvi(scene: dict, name: str) -> np.ndarray:
    """NDVI for one saved scene (dict with red, nir, baseline, date). Prints whether the offset was removed."""
    baseline = str(scene["baseline"])
    action = "removed +1000 offset" if has_offset(baseline) else "no offset to remove"
    print(f"{name} ({scene['date']}): processing baseline {baseline} -> {action}")
    return ndvi(remove_offset(scene["red"], baseline), remove_offset(scene["nir"], baseline))


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


def buffer_invalid(valid: np.ndarray, pixels: int = 5) -> np.ndarray:
    """Also mark pixels within `pixels` of a cloud/shadow as unusable: the scene classification misses hazy cloud edges."""
    if pixels <= 0:
        return valid
    return ~ndimage.binary_dilation(~valid, iterations=pixels)
