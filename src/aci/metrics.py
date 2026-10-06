"""Segmentation metrics for binary change masks."""
import numpy as np


def confusion(pred: np.ndarray, target: np.ndarray):
    pred, target = pred.astype(bool), target.astype(bool)
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    tn = int((~pred & ~target).sum())
    return tp, fp, fn, tn


def scores(tp, fp, fn, tn=0):
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + eps),
        "iou": tp / (tp + fp + fn + eps),
    }


def expected_calibration_error(prob: np.ndarray, target: np.ndarray, bins=10):
    """Average gap between predicted confidence and observed frequency of change. 0 = perfectly calibrated."""
    prob, target = prob.ravel(), target.ravel().astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    ece = 0.0
    for k in range(bins):
        sel = idx == k
        if sel.any():
            ece += sel.mean() * abs(prob[sel].mean() - target[sel].mean())
    return float(ece)
