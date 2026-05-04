"""
custom_modules.py  (extended)
==============================
Registers all custom modules for Ultralytics YAML parsing:
  - DSConv          : single depthwise-separable conv + BN + ReLU
  - DSBlock         : 2× DSConv + MaxPool (TinyissimoYOLO block)
  - CBAM            : channel + spatial attention
  - DSInception     : depthwise-separable Inception module (no pool)
  - DSInceptionBlock: DSInception + MaxPool (strided, use in backbone)
  - DSResBlock      : depthwise-separable residual block (same dims)
  - DSResDownBlock  : DS residual block with stride-2 downsampling

Import ONCE before loading any YOLO yaml:
    import custom_modules
    from ultralytics import YOLO
    model = YOLO("tinyissimo_ds_inception.yaml")
"""

import torch
import torch.nn as nn
from torchvision import models

from ultralytics.nn.modules.conv import Conv

# ============================================================
# Base block
# ============================================================


class DSConv(Conv):
    """
    Depthwise Separable Conv.
    DW(k×k, s) + BN + ReLU  →  PW(1×1) + BN + ReLU.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        # use Conv module for PW conv to get same padding, activation etc.
        super().__init__(c1, c1, k, s, g=c1, p=k // 2)  # depthwise
        self.pw = Conv(c1, c2, 1, 1)  # pointwise

    def forward(self, x):
        x = super().forward(x)  # depthwise conv + BN + ReLU
        x = self.pw(x)  # pointwise conv + BN + ReLU
        return x

    def forward_fuse(self, x):
        return self.pw(
            super().forward_fuse(x)
        )  # for export: fuse DW conv, then PW conv


class DSBlock(nn.Module):
    """2× DSConv + MaxPool(2,2). TinyissimoYOLO backbone stage."""

    def __init__(self, c1: int, c2: int, c_mid: int = -1):
        super().__init__()
        c_mid = c_mid if c_mid > 0 else c1
        self.block = nn.Sequential(
            DSConv(c1, c_mid),
            DSConv(c_mid, c2),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# DS-Inception
# ============================================================


class DSInception(nn.Module):
    """
    Depthwise-Separable Inception module. No pooling — spatial size preserved.

    Four parallel branches concatenated along the channel dimension:

        B1: 1×1 conv                       channels = c_b1
        B2: 1×1 bottleneck → DS 3×3        channels = c_b2
        B3: 1×1 bottleneck → DS3×3 → DS3×3 channels = c_b3
            (two stacked 3×3 gives 5×5 receptive field at DS cost)
        B4: MaxPool 3×3 (s=1) → 1×1        channels = c_b4

    Output channels = c_b1 + c_b2 + c_b3 + c_b4

    Args
    ----
    c1         : input channels
    c_b1..c_b4 : per-branch output channels
    bottleneck : fraction of output channels used as bottleneck. Default 0.5.
    """

    def __init__(
        self,
        c1: int,
        c_b1: int,
        c_b2: int,
        c_b3: int,
        c_b4: int,
        bottleneck: float = 0.5,
    ):
        super().__init__()
        mid2 = max(int(c_b2 * bottleneck), 4)
        mid3 = max(int(c_b3 * bottleneck), 4)

        # B1 — pure 1×1
        self.b1 = nn.Sequential(
            nn.Conv2d(c1, c_b1, 1, bias=False),
            nn.BatchNorm2d(c_b1),
            nn.ReLU(inplace=True),
        )

        # B2 — 1×1 bottleneck + DS 3×3
        self.b2 = nn.Sequential(
            nn.Conv2d(c1, mid2, 1, bias=False),
            nn.BatchNorm2d(mid2),
            nn.ReLU(inplace=True),
            DSConv(mid2, c_b2, k=3),
        )

        # B3 — 1×1 bottleneck + DS 3×3 + DS 3×3  (≈ 5×5 RF)
        self.b3 = nn.Sequential(
            nn.Conv2d(c1, mid3, 1, bias=False),
            nn.BatchNorm2d(mid3),
            nn.ReLU(inplace=True),
            DSConv(mid3, mid3, k=3),
            DSConv(mid3, c_b3, k=3),
        )

        # B4 — pool + 1×1 projection
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(c1, c_b4, 1, bias=False),
            nn.BatchNorm2d(c_b4),
            nn.ReLU(inplace=True),
        )

        self.out_channels = c_b1 + c_b2 + c_b3 + c_b4

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class DSInceptionBlock(nn.Module):
    """
    DSInception + optional MaxPool(2,2).
    Use as a backbone stage: one block = one spatial downsampling.

    Args
    ----
    c1..c_b4   : same as DSInception
    bottleneck : same as DSInception
    pool       : apply MaxPool after inception (default True)
    """

    def __init__(
        self,
        c1: int,
        c_b1: int,
        c_b2: int,
        c_b3: int,
        c_b4: int,
        bottleneck: float = 0.5,
        pool: bool = True,
    ):
        super().__init__()
        self.inception = DSInception(c1, c_b1, c_b2, c_b3, c_b4, bottleneck)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()
        self.out_channels = self.inception.out_channels

    def forward(self, x):
        return self.pool(self.inception(x))


# ============================================================
# DS-ResNet blocks
# ============================================================


class DSResBlock(nn.Module):
    """
    DS Residual Block — identity shortcut (same channels, no downsampling).

        out = ReLU(BN( DSConv→DSConv(x) + x ))

    Args
    ----
    c1 : input = output channels
    """

    def __init__(self, c1: int):
        super().__init__()
        self.block = nn.Sequential(DSConv(c1, c1), DSConv(c1, c1))
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.block(x) + x))


class DSResDownBlock(nn.Module):
    """
    DS Residual Block with stride-2 downsampling + channel expansion.

    Main path : DSConv(c1→c2, s=2) → DSConv(c2→c2)
    Skip path : Conv1×1(c1→c2, s=2)   projection shortcut

        out = ReLU(BN( main(x) + skip(x) ))

    Args
    ----
    c1 : input channels
    c2 : output channels (typically 2× c1)
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.main = nn.Sequential(DSConv(c1, c2, k=3, s=2), DSConv(c2, c2))
        self.skip = nn.Sequential(
            nn.Conv2d(c1, c2, 1, stride=2, bias=False), nn.BatchNorm2d(c2)
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.main(x) + self.skip(x)))


class Add(nn.Module):
    """Element-wise addition of two tensors. Handles spatial dimension mismatches by upsampling."""

    def forward(self, x):
        # Accept a list/tuple of two tensors (provided by the model runner when 'from' is a list)
        if isinstance(x, (list, tuple)):
            if len(x) != 2:
                raise ValueError("Add expects exactly two input tensors")
            x1, x2 = x[0], x[1]
        else:
            try:
                x1, x2 = x
            except Exception:
                raise TypeError("Add.forward expects a list/tuple of two tensors")
        
        # Handle spatial dimension mismatches by upsampling smaller to larger
        h1, w1 = x1.shape[-2:]
        h2, w2 = x2.shape[-2:]
        
        if (h1, w1) != (h2, w2):
            # Upsample the smaller tensor to match the larger one
            if h1 * w1 < h2 * w2:
                x1 = torch.nn.functional.interpolate(x1, size=(h2, w2), mode='nearest')
            else:
                x2 = torch.nn.functional.interpolate(x2, size=(h1, w1), mode='nearest')
        try:
            return x1 + x2
        except Exception as e:
            print(f"Error in Add module: {e}")
            raise Exception(f"Add module failed to add tensors. Check shapes and compatibility. x1 shape: {x1.shape}, x2 shape: {x2.shape}")
        
    
class MobileNetV3Backbone(nn.Module):
    """Use MobileNetV3 backbone 0.35 width on ImageNet. Output features from 3 stages."""

    def __init__(self, variant="small", width_mult=0.35):
        super().__init__()
        if variant == "small":
            mobilenet = models.mobilenet_v3_small(width_mult=width_mult)
        elif variant == "large":
            mobilenet = models.mobilenet_v3_large(width_mult=width_mult)
        else:
            raise ValueError("variant must be 'small' or 'large'")
        
        # Extract features from three stages (after each downsampling)
        self.stage1 = nn.Sequential(
            mobilenet.features[0],  # ConvBNReLU (stride=2)
            mobilenet.features[1],  # InvertedResidual (stride=2)
            mobilenet.features[2],  # InvertedResidual (stride=1)
        )  # Output stride 4
        self.stage2 = nn.Sequential(
            mobilenet.features[3],  # InvertedResidual (stride=2)
            mobilenet.features[4],  # InvertedResidual (stride=1)
            mobilenet.features[5],  # InvertedResidual (stride=1)
        )  # Output stride 8
        self.stage3 = nn.Sequential(
            mobilenet.features[6],  # InvertedResidual (stride=2)
            mobilenet.features[7],  # InvertedResidual (stride=1)
            mobilenet.features[8],  # InvertedResidual (stride=1)
        )  # Output stride 16

    def forward(self, x):
        f1 = self.stage1(x)  # low-level features (C1)
        f2 = self.stage2(f1)  # mid-level features (C2)
        f3 = self.stage3(f2)  # high-level features (C3)
        return f1, f2, f3  # return as tuple for model runner to unpack
        


# ============================================================
# Register into Ultralytics
# ============================================================


def register_modules() -> None:
    try:
        import ultralytics.nn.tasks as tasks

        _map = {
            "DSConv": DSConv,
            "DSBlock": DSBlock,
            "DSInception": DSInception,
            "DSInceptionBlock": DSInceptionBlock,
            "DSResBlock": DSResBlock,
            "DSResDownBlock": DSResDownBlock,
            "Add": Add,
        }
        for name, cls in _map.items():
            setattr(tasks, name, cls)
            tasks.__dict__[name] = cls
        print(f"✓ Registered: {', '.join(_map)}")
    except ImportError:
        raise ImportError("pip install ultralytics")


register_modules()
