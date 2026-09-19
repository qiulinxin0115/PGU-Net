import math

import torch
import torch.nn as nn

from .dnet import DNet
from .pbsir import PBSIRModule
from .snet import SNet


class SpecularProximalNet(nn.Module):
    """Learned bounded modulation P_M; not an exact analytical proximal map."""

    def __init__(
        self,
        channels=(32, 32, 16),
        correction_scale=0.1,
        gate_floor=0.25,
        initial_gate=0.8,
    ):
        super().__init__()
        c1, c2, c3 = [int(value) for value in channels]
        self.correction_scale = float(correction_scale)
        self.gate_floor = float(gate_floor)
        if not 0.0 <= self.gate_floor < 1.0:
            raise ValueError("gate_floor must be in [0, 1)")
        self.features = nn.Sequential(
            nn.Conv2d(12, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(c3, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        initial_gate = min(max(float(initial_gate), self.gate_floor + 1e-4), 1.0 - 1e-4)
        gate = (initial_gate - self.gate_floor) / (1.0 - self.gate_floor)
        with torch.no_grad():
            self.output.bias[0] = math.log(gate / (1.0 - gate))

    def forward(self, low, diffuse, indirect, physics_specular):
        raw = self.output(
            self.features(torch.cat([low, diffuse, indirect, physics_specular], dim=1))
        )
        gate = self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(raw[:, :1])
        relative_correction = self.correction_scale * torch.tanh(raw[:, 1:2])
        modulation = (gate + relative_correction).clamp(
            self.gate_floor * 0.5, 1.0 + self.correction_scale
        )
        specular = (physics_specular * modulation).clamp(0.0, 1.0)
        correction = specular - gate.expand_as(specular) * physics_specular
        return {
            "M": specular,
            "gate": gate,
            "modulation": modulation,
            "correction": correction,
        }


class PGUNet(nn.Module):
    """Shared-weight PGU-Net with stage-wise PBSIR and learned modulation."""

    def __init__(
        self,
        pbsir_checkpoint=None,
        stages=4,
        d_features=(64, 128, 256, 512, 1024),
        d_initial_fraction=0.35,
        d_freeze_batch_norm=True,
        d_normalization="batch",
        s_features=(32, 64, 128, 256, 256),
        s_attention_scale=0.2,
        s_lowpass_factor=16,
        s_max_amplitude=0.25,
        s_initial_fraction=0.2,
        pm_channels=(32, 32, 16),
        pm_correction_scale=0.1,
        pm_gate_floor=0.25,
        min_exponent=2.0,
        max_exponent=96.0,
        max_light_intensity=2.5,
    ):
        super().__init__()
        if int(stages) < 1:
            raise ValueError("stages must be positive")
        self.stages = int(stages)
        self.d_net = DNet(
            in_channels=9,
            features=d_features,
            initial_fraction=d_initial_fraction,
            freeze_batch_norm=d_freeze_batch_norm,
            normalization=d_normalization,
        )
        self.s_net = SNet(
            in_channels=9,
            features=s_features,
            attention_scale=s_attention_scale,
            lowpass_factor=s_lowpass_factor,
            max_amplitude=s_max_amplitude,
            initial_fraction=s_initial_fraction,
        )
        self.pbsir = PBSIRModule(
            pbsir_checkpoint,
            min_exponent=min_exponent,
            max_exponent=max_exponent,
            max_intensity=max_light_intensity,
            freeze_batch_norm=True,
        )
        self.p_m = SpecularProximalNet(
            channels=pm_channels,
            correction_scale=pm_correction_scale,
            gate_floor=pm_gate_floor,
        )

    def set_pbsir_trainable(self, trainable):
        self.pbsir.set_trainable(bool(trainable))

    def _physics(self, image, normal):
        return self.pbsir(image, normal)

    def forward(self, low, normal):
        if low.ndim != 4 or low.shape[1] != 3:
            raise ValueError("low must have shape [B, 3, H, W]")
        if normal.ndim != 4 or normal.shape[1] != 3:
            raise ValueError("normal must have shape [B, 3, H, W]")

        if low.shape != normal.shape:
            raise ValueError("low and normal must have the same shape")
        if min(low.shape[-2:]) < 32:
            raise ValueError("Both spatial dimensions must be at least 32")

        initial_physics = self._physics(low, normal)
        previous_m = initial_physics["M"]
        previous_s = torch.zeros_like(low)
        history = {
            key: []
            for key in (
                "I",
                "D",
                "S",
                "M",
                "M_pbsir",
                "ks",
                "n",
                "light",
                "coarse_ks",
                "coarse_n",
                "I_int",
                "pm_modulation",
                "pm_gate",
                "pm_correction",
            )
        }
        history["initial_M"] = previous_m
        history["initial_physics"] = initial_physics

        for _ in range(self.stages):
            diffuse = self.d_net(torch.cat([low, previous_m, previous_s], dim=1))
            indirect = self.s_net(torch.cat([low, diffuse, previous_m], dim=1))
            intermediate = (diffuse + previous_m + indirect).clamp(0.0, 1.0)
            physics = self._physics(intermediate, normal)
            refined = self.p_m(low, diffuse, indirect, physics["M"])
            specular = refined["M"]
            gate = refined["gate"]
            correction = refined["correction"]
            reconstruction = diffuse + indirect + specular
            values = {
                "I": reconstruction,
                "D": diffuse,
                "S": indirect,
                "M": specular,
                "M_pbsir": physics["M"],
                "ks": physics["ks"],
                "n": physics["n"],
                "light": physics["light"],
                "coarse_ks": physics["coarse_ks"],
                "coarse_n": physics["coarse_n"],
                "I_int": intermediate,
                "pm_modulation": refined["modulation"],
                "pm_gate": gate,
                "pm_correction": correction,
            }
            for key, value in values.items():
                history[key].append(value)
            previous_m, previous_s = specular, indirect
        return history["I"][-1], history
