import torch

from aci.model import SiameseUNet


def test_output_shape_and_identical_inputs():
    model = SiameseUNet(pretrained=False).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x, x)
    assert out.shape == (1, 1, 64, 64)
