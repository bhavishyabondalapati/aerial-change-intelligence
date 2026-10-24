"""Siamese U-Net: one shared ResNet18 encoder reads both images, a decoder reads their feature differences."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SiameseUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)  # 1/2, 64ch
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4  # 1/4..1/32
        # Each skip is |featA - featB| concatenated with featA + featB -> 2x channels
        self.d4 = conv_block(512 * 2 + 256 * 2, 256)
        self.d3 = conv_block(256 + 128 * 2, 128)
        self.d2 = conv_block(128 + 64 * 2, 64)
        self.d1 = conv_block(64 + 64 * 2, 64)
        self.head = nn.Conv2d(64, 1, 1)

    def encode(self, x):
        f0 = self.stem(x)
        f1 = self.l1(self.pool(f0))
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        return f0, f1, f2, f3, f4

    @staticmethod
    def fuse(fa, fb):
        return torch.cat([(fa - fb).abs(), fa + fb], dim=1)

    def forward(self, a, b):
        """Returns logits (B,1,H,W). Apply sigmoid for per-pixel change probability."""
        size = a.shape[-2:]
        fa, fb = self.encode(a), self.encode(b)
        s = [self.fuse(x, y) for x, y in zip(fa, fb)]
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d4(torch.cat([up(s[4], s[3]), s[3]], 1))
        x = self.d3(torch.cat([up(x, s[2]), s[2]], 1))
        x = self.d2(torch.cat([up(x, s[1]), s[1]], 1))
        x = self.d1(torch.cat([up(x, s[0]), s[0]], 1))
        return F.interpolate(self.head(x), size=size, mode="bilinear", align_corners=False)
