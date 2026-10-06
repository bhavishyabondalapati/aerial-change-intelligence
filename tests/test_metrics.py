import numpy as np

from aci.metrics import confusion, expected_calibration_error, scores


def test_perfect_prediction():
    m = np.array([[1, 0], [0, 1]])
    s = scores(*confusion(m, m))
    assert s["f1"] > 0.999 and s["iou"] > 0.999


def test_known_iou():
    pred = np.array([1, 1, 0, 0])
    target = np.array([1, 0, 1, 0])
    tp, fp, fn, tn = confusion(pred, target)
    assert (tp, fp, fn, tn) == (1, 1, 1, 1)
    assert abs(scores(tp, fp, fn, tn)["iou"] - 1 / 3) < 1e-6


def test_ece_zero_for_calibrated_and_high_for_overconfident():
    target = np.array([1] * 50 + [0] * 50)
    assert expected_calibration_error(np.full(100, 0.5), target) < 1e-9
    assert expected_calibration_error(np.full(100, 0.99), target) > 0.4
