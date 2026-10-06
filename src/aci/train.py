"""Train the Siamese U-Net on LEVIR-CD and write checkpoints/best.pt + outputs/metrics.json."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from aci.data import LevirCD
from aci.device import get_device
from aci.metrics import confusion, expected_calibration_error, scores
from aci.model import SiameseUNet


def loss_fn(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return bce + dice


@torch.no_grad()
def evaluate(model, ds, device, threshold=0.5, max_items=None):
    """Run on full-size (1024x1024) images, one at a time. Returns metrics dict."""
    model.eval()
    tp = fp = fn = tn = 0
    probs, targets = [], []
    n = len(ds) if max_items is None else min(max_items, len(ds))
    for i in range(n):
        a, b, y = ds[i]
        p = torch.sigmoid(model(a[None].to(device), b[None].to(device)))[0, 0].cpu().numpy()
        t = y[0].numpy() > 0.5
        c = confusion(p > threshold, t)
        tp, fp, fn, tn = tp + c[0], fp + c[1], fn + c[2], tn + c[3]
        probs.append(p[::4, ::4])  # subsample pixels for the calibration estimate
        targets.append(t[::4, ::4])
    out = scores(tp, fp, fn, tn)
    out["ece"] = expected_calibration_error(np.concatenate([x.ravel() for x in probs]), np.concatenate([x.ravel() for x in targets]))
    out["threshold"] = threshold
    out["n_images"] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/levir-cd")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--subset", type=int, default=None, help="use only N training images (quick smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = get_device()
    print("device:", device)
    torch.manual_seed(0)
    np.random.seed(0)

    train_full = LevirCD(args.data, "train", crop=256)
    train_ds = Subset(train_full, range(args.subset)) if args.subset else train_full
    val_ds = LevirCD(args.data, "val", crop=None)
    # Each 1024x1024 image gives many 256 crops; sample several crops per image per epoch
    crops_per_image = 4
    loader = DataLoader(
        torch.utils.data.ConcatDataset([train_ds] * crops_per_image),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0,
    )

    model = SiameseUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Path("checkpoints").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    best_f1, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for a, b, y in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            a, b, y = a.to(device), b.to(device), y.to(device)
            loss = loss_fn(model(a, b), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        m = evaluate(model, val_ds, device, max_items=32)
        m.update(epoch=epoch + 1, train_loss=running / len(loader), seconds=round(time.time() - t0))
        history.append(m)
        print(json.dumps(m))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(model.state_dict(), "checkpoints/best.pt")
        Path("outputs/train_history.json").write_text(json.dumps(history, indent=2))

    # Final numbers on the untouched test split, using the best checkpoint
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
    test = evaluate(model, LevirCD(args.data, "test", crop=None), device)
    Path("outputs/metrics.json").write_text(json.dumps(test, indent=2))
    print("TEST:", json.dumps(test))


if __name__ == "__main__":
    main()
