import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value):
        return self.net(value)


class MirrorNet(nn.Module):
    def __init__(self, min_exponent=2.0, max_exponent=96.0):
        super().__init__()
        self.min_exponent = float(min_exponent)
        self.max_exponent = float(max_exponent)
        self.enc1 = ConvBlock(7, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 32, 32)
        self.ks_head = nn.Conv2d(32, 1, 1)
        self.n_head = nn.Conv2d(32, 1, 1)

    def forward(self, image, light):
        if light.ndim != 2 or light.shape[1] != 4:
            raise ValueError("light must have shape [B, 4]")
        batch, _, height, width = image.shape
        light_map = light.view(batch, 4, 1, 1).expand(batch, 4, height, width)
        x1 = self.enc1(torch.cat([image, light_map], dim=1))
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        value = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec2(torch.cat([value, x2], dim=1))
        value = F.interpolate(value, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec1(torch.cat([value, x1], dim=1))
        ks = torch.sigmoid(self.ks_head(value))
        exponent01 = torch.sigmoid(self.n_head(value))
        exponent = self.min_exponent + (
            self.max_exponent - self.min_exponent
        ) * exponent01
        return ks, exponent


class LightPredictor(nn.Module):
    def __init__(self, min_exponent=2.0, max_exponent=96.0, max_intensity=2.5):
        super().__init__()
        self.min_exponent = float(min_exponent)
        self.max_exponent = float(max_exponent)
        self.max_intensity = float(max_intensity)
        base = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(5, base, 3, stride=2, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(base * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
        )

    def forward(self, image, ks, exponent):
        denominator = max(self.max_exponent - self.min_exponent, 1e-6)
        exponent01 = ((exponent - self.min_exponent) / denominator).clamp(0.0, 1.0)
        raw = self.fc(self.encoder(torch.cat([image, ks, exponent01], dim=1)).flatten(1))
        direction = F.normalize(raw[:, :3], dim=1, eps=1e-6)
        intensity = torch.sigmoid(raw[:, 3:4]) * self.max_intensity
        return torch.cat([direction, intensity], dim=1)


class PBSIRModule(nn.Module):
    """Coarse-to-fine PBSIR and differentiable Blinn-Phong renderer."""

    def __init__(
        self,
        checkpoint_path=None,
        min_exponent=2.0,
        max_exponent=96.0,
        max_intensity=2.5,
        freeze_batch_norm=True,
    ):
        super().__init__()
        self.min_exponent = float(min_exponent)
        self.max_exponent = float(max_exponent)
        self.max_intensity = float(max_intensity)
        self.freeze_batch_norm = bool(freeze_batch_norm)
        self.mirror = MirrorNet(min_exponent, max_exponent)
        self.predictor = LightPredictor(min_exponent, max_exponent, max_intensity)
        if checkpoint_path:
            self.load_pretrained(checkpoint_path)

    def load_pretrained(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError("PBSIR checkpoint not found: " + checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Expected a joint PBSIR checkpoint dictionary")
        if "mirror_net" not in checkpoint or "light_predictor" not in checkpoint:
            raise RuntimeError(
                "Checkpoint must contain mirror_net and light_predictor: " + checkpoint_path
            )
        self.mirror.load_state_dict(checkpoint["mirror_net"], strict=True)
        self.predictor.load_state_dict(checkpoint["light_predictor"], strict=True)

    def set_trainable(self, trainable):
        for parameter in self.parameters():
            parameter.requires_grad_(bool(trainable))

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_batch_norm:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def predict_parameters(self, image, coarse_only=False):
        batch = image.shape[0]
        zero_light = torch.zeros(batch, 4, dtype=image.dtype, device=image.device)
        coarse_ks, coarse_n = self.mirror(image, zero_light)
        light = self.predictor(image, coarse_ks, coarse_n)
        if coarse_only:
            ks, exponent = coarse_ks, coarse_n
        else:
            ks, exponent = self.mirror(image, light)
        return {
            "ks": ks,
            "n": exponent,
            "light": light,
            "coarse_ks": coarse_ks,
            "coarse_n": coarse_n,
        }

    def mirror_with_light(self, image, light):
        ks, exponent = self.mirror(image, light)
        return {"ks": ks, "n": exponent, "light": light}

    def perturb_light(self, light, sigma=0.05):
        direction = F.normalize(
            light[:, :3] + float(sigma) * torch.randn_like(light[:, :3]),
            dim=1,
            eps=1e-6,
        )
        intensity = (
            light[:, 3:4] + float(sigma) * torch.randn_like(light[:, 3:4])
        ).clamp(0.0, self.max_intensity)
        return torch.cat([direction, intensity], dim=1)

    def render(self, normal, ks, exponent, light):
        if normal.shape[-2:] != ks.shape[-2:]:
            normal = F.interpolate(normal, size=ks.shape[-2:], mode="bilinear", align_corners=False)
        normal = F.normalize(normal, dim=1, eps=1e-6)
        batch, _, height, width = normal.shape
        light_direction = F.normalize(light[:, :3], dim=1, eps=1e-6).view(batch, 3, 1, 1)
        view_direction = normal.new_tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1)
        view_direction = view_direction.expand(batch, 3, height, width)
        halfway = F.normalize(light_direction + view_direction, dim=1, eps=1e-6)
        ndoth = (normal * halfway).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        intensity = light[:, 3:4].clamp(0.0, self.max_intensity).view(batch, 1, 1, 1)
        specular = intensity * ks * ndoth.pow(exponent)
        return specular.expand(-1, 3, -1, -1)

    def forward(self, image, normal, coarse_only=False):
        parameters = self.predict_parameters(image, coarse_only=coarse_only)
        parameters["M"] = self.render(
            normal, parameters["ks"], parameters["n"], parameters["light"]
        )
        return parameters
