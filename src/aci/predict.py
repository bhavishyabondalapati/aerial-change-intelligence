"""Predict a change mask + per-pixel confidence for one before/after pair."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aci.data import normalize
from aci.device import get_device
from aci.model import SiameseUNet


def load_model(ckpt="checkpoints/best.pt", device=None):
    device = device or get_device()
    model = SiameseUNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.eval()


@torch.no_grad()
def predict_probability(model, before: np.ndarray, after: np.ndarray, device=None) -> np.ndarray:
    """uint8 HxWx3 images -> float32 HxW map: probability that each pixel changed."""
    device = device or next(model.parameters()).device
    a = torch.from_numpy(normalize(before))[None].to(device)
    b = torch.from_numpy(normalize(after))[None].to(device)
    return torch.sigmoid(model(a, b))[0, 0].cpu().numpy()


def region_confidence(prob: np.ndarray, mask: np.ndarray):
    """Average predicted probability inside the changed pixels (0 if nothing changed)."""
    return float(prob[mask].mean()) if mask.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/prediction")
    args = ap.parse_args()

    model = load_model(args.ckpt)
    before = np.array(Image.open(args.before).convert("RGB"))
    after = np.array(Image.open(args.after).convert("RGB"))
    prob = predict_probability(model, before, after)
    mask = prob > args.threshold

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "probability.npy", prob)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out / "mask.png")
    Image.fromarray((prob * 255).astype(np.uint8)).save(out / "confidence.png")
    summary = {"changed_fraction": float(mask.mean()), "mean_confidence": region_confidence(prob, mask)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
