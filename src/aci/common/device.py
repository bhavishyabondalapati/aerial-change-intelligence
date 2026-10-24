"""Pick the best compute device. On a Mac this is Apple's GPU ("mps")."""
import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
