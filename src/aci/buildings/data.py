"""LEVIR-CD loader: pairs of before (A) / after (B) images and a binary change mask (label)."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(img: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 CxHxW, normalised with ImageNet statistics."""
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)


class LevirCD(Dataset):
    """Expects root/{A,B,label}/<split>_<n>.png (the layout the LEVIR-CD zips extract to). Training uses random crops and flips."""

    def __init__(self, root, split="train", crop=256):
        self.dir = Path(root)
        self.names = sorted(p.name for p in (self.dir / "A").glob(f"{split}_*.png"))
        if not self.names:
            raise FileNotFoundError(f"No {split} images found in {self.dir / 'A'}")
        self.crop = crop
        self.train = split == "train"

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        a = np.array(Image.open(self.dir / "A" / n).convert("RGB"))
        b = np.array(Image.open(self.dir / "B" / n).convert("RGB"))
        m = np.array(Image.open(self.dir / "label" / n).convert("L")) > 127
        a, b, m = random_crop_flip(a, b, m, self.crop, self.train)
        return (
            torch.from_numpy(normalize(a)),
            torch.from_numpy(normalize(b)),
            torch.from_numpy(m[None].astype(np.float32)),
        )


def random_crop_flip(a, b, m, crop, train, rng=np.random):
    """Same crop/flip on all three arrays. At eval time, take the top-left crop (or full image if crop is None)."""
    if crop is None:
        return a, b, m
    h, w = m.shape
    if train:
        y, x = rng.randint(0, h - crop + 1), rng.randint(0, w - crop + 1)
    else:
        y = x = 0
    a, b, m = (t[y : y + crop, x : x + crop] for t in (a, b, m))
    if train:
        if rng.rand() < 0.5:
            a, b, m = a[:, ::-1], b[:, ::-1], m[:, ::-1]
        if rng.rand() < 0.5:
            a, b, m = a[::-1], b[::-1], m[::-1]
        # Swapping before/after teaches the model that "change" is symmetric
        if rng.rand() < 0.5:
            a, b = b, a
    return a.copy(), b.copy(), m.copy()
