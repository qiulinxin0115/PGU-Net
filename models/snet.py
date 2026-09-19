import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, value):
        return self.net(value)


class GlobalContextBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.conv_context = nn.Conv2d(channels, 1, 1)
        self.softmax = nn.Softmax(dim=2)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, value):
        batch, channels, height, width = value.shape
        weights = self.softmax(self.conv_context(value).view(batch, 1, height * width))
        flattened = value.view(batch, channels, height * width)
        context = torch.bmm(flattened, weights.transpose(1, 2)).view(batch, channels, 1, 1)
        return self.transform(context)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, value):
        return self.conv(self.pool(value))


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, value, skip):
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, self.conv1x1(value)], dim=1))


class SNet(nn.Module):
    """Low-frequency compensation update. Input: Cat[I_l, D^t, M^(t-1)]."""

    def __init__(
        self,
        in_channels=9,
        out_channels=3,
        features=(32, 64, 128, 256, 256),
        attention_scale=0.2,
        lowpass_factor=16,
        max_amplitude=0.25,
        initial_fraction=0.2,
    ):
        super().__init__()
        f1, f2, f3, f4, f5 = [int(value) for value in features]
        self.inc = ConvBlock(in_channels, f1)
        self.down1 = Down(f1, f2)
        self.down2 = Down(f2, f3)
        self.down3 = Down(f3, f4)
        self.down4 = Down(f4, f5)
        self.bot_conv = ConvBlock(f5, f5)
        self.bot_attn = GlobalContextBlock(f5)
        self.attn_scale = float(attention_scale)
        self.lowpass_factor = max(int(lowpass_factor), 1)
        self.max_amplitude = float(max_amplitude)
        self.initial_fraction = float(initial_fraction)
        if not 0.0 < self.initial_fraction < 1.0:
            raise ValueError("initial_fraction must be in (0, 1)")
        self.up4 = Up(f5, f4, f4)
        self.up3 = Up(f4, f3, f3)
        self.up2 = Up(f3, f2, f2)
        self.up1 = Up(f2, f1, f1)
        self.outc = nn.Sequential(nn.Conv2d(f1, out_channels, 1), nn.Sigmoid())
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        output_conv = self.outc[0]
        nn.init.zeros_(output_conv.weight)
        nn.init.constant_(
            output_conv.bias,
            math.log(self.initial_fraction / (1.0 - self.initial_fraction)),
        )

    def forward(self, value):
        x1 = self.inc(value)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        value = self.bot_conv(x5)
        value = value + self.attn_scale * self.bot_attn(value)
        value = self.up4(value, x4)
        value = self.up3(value, x3)
        value = self.up2(value, x2)
        value = self.up1(value, x1)
        value = self.outc(value)
        height, width = value.shape[-2:]
        low_height = max(height // self.lowpass_factor, 1)
        low_width = max(width // self.lowpass_factor, 1)
        if (low_height, low_width) != (height, width):
            value = F.interpolate(value, size=(low_height, low_width), mode="area")
            value = F.interpolate(
                value, size=(height, width), mode="bilinear", align_corners=False
            )
        return self.max_amplitude * value
