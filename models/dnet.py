import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_normalization(channels, normalization):
    normalization = str(normalization).lower()
    if normalization == "batch":
        return nn.BatchNorm2d(channels)
    if normalization == "instance":
        return nn.InstanceNorm2d(
            channels,
            affine=False,
            track_running_stats=False,
        )
    raise ValueError("Unsupported D-Net normalization: " + str(normalization))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization="batch"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            build_normalization(out_channels, normalization),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            build_normalization(out_channels, normalization),
            nn.ReLU(inplace=True),
        )

    def forward(self, value):
        return self.net(value)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, value):
        batch, channels, _, _ = value.shape
        weights = self.fc(self.avg_pool(value).view(batch, channels))
        return value * weights.view(batch, channels, 1, 1)


class LargeKernelBlock(nn.Module):
    def __init__(self, channels, normalization="batch"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.dw_conv = nn.Conv2d(
            channels, channels, 7, padding=3, groups=channels, bias=False
        )
        self.bn = build_normalization(channels, normalization)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.bn2 = build_normalization(channels, normalization)
        self.act2 = nn.ReLU(inplace=True)
        self.se = SEBlock(channels)

    def forward(self, value):
        residual = value
        value = self.act(self.bn(self.dw_conv(self.conv1(value))))
        value = self.se(self.bn2(self.conv2(value)))
        return self.act2(value + residual)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, normalization="batch"):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, normalization)

    def forward(self, value):
        return self.conv(self.pool(value))


class Up(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        normalization="batch",
    ):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels,
            normalization,
        )

    def forward(self, value, skip):
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, self.conv1x1(value)], dim=1))


class DNet(nn.Module):
    """Diffuse update network. Its input is Cat[I_l, M^(t-1), S^(t-1)]."""

    def __init__(
        self,
        in_channels=9,
        out_channels=3,
        features=(64, 128, 256, 512, 1024),
        initial_fraction=0.35,
        freeze_batch_norm=True,
        normalization="batch",
    ):
        super().__init__()
        self.initial_fraction = float(initial_fraction)
        self.freeze_batch_norm = bool(freeze_batch_norm)
        self.normalization = str(normalization).lower()
        if not 0.0 < self.initial_fraction < 1.0:
            raise ValueError("initial_fraction must be in (0, 1)")
        f1, f2, f3, f4, f5 = [int(value) for value in features]
        self.inc = ConvBlock(in_channels, f1, self.normalization)
        self.down1 = Down(f1, f2, self.normalization)
        self.down2 = Down(f2, f3, self.normalization)
        self.down3 = Down(f3, f4, self.normalization)
        self.down4 = Down(f4, f5, self.normalization)
        self.bot = LargeKernelBlock(f5, self.normalization)
        self.up4 = Up(f5, f4, f4, self.normalization)
        self.up3 = Up(f4, f3, f3, self.normalization)
        self.up2 = Up(f3, f2, f2, self.normalization)
        self.up1 = Up(f2, f1, f1, self.normalization)
        self.outc = nn.Sequential(nn.Conv2d(f1, out_channels, 1), nn.Sigmoid())
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # A neutral output prevents the sigmoid head from entering a
        # zero-gradient saturated state after partial checkpoint loading.
        output_conv = self.outc[0]
        nn.init.zeros_(output_conv.weight)
        nn.init.constant_(
            output_conv.bias,
            math.log(self.initial_fraction / (1.0 - self.initial_fraction)),
        )

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_batch_norm:
            # Keep the checkpointed statistics fixed while the affine BN
            # parameters and the remaining D-Net weights are fine-tuned.
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, value):
        x1 = self.inc(value)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        value = self.bot(x5)
        value = self.up4(value, x4)
        value = self.up3(value, x3)
        value = self.up2(value, x2)
        value = self.up1(value, x1)
        return self.outc(value)
