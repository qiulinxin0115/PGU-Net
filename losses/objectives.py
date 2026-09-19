import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim_index(prediction, target, window_size=11):
    channels = prediction.shape[1]
    coordinates = torch.arange(
        window_size, dtype=prediction.dtype, device=prediction.device
    )
    coordinates = coordinates - window_size // 2
    kernel_1d = torch.exp(-coordinates.square() / (2.0 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    padding = window_size // 2
    mean_x = F.conv2d(prediction, kernel, padding=padding, groups=channels)
    mean_y = F.conv2d(target, kernel, padding=padding, groups=channels)
    variance_x = (
        F.conv2d(prediction.square(), kernel, padding=padding, groups=channels)
        - mean_x.square()
    )
    variance_y = (
        F.conv2d(target.square(), kernel, padding=padding, groups=channels)
        - mean_y.square()
    )
    covariance = (
        F.conv2d(prediction * target, kernel, padding=padding, groups=channels)
        - mean_x * mean_y
    )
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
    denominator = (
        (mean_x.square() + mean_y.square() + c1)
        * (variance_x + variance_y + c2)
        + 1e-8
    )
    return (numerator / denominator).mean()


def low_frequency_projection(value, factor=16):
    height, width = value.shape[-2:]
    low_height = max(height // max(int(factor), 1), 1)
    low_width = max(width // max(int(factor), 1), 1)
    if (low_height, low_width) == (height, width):
        return value
    value = F.interpolate(value, size=(low_height, low_width), mode="area")
    return F.interpolate(value, size=(height, width), mode="bilinear", align_corners=False)


def gradient_consistency(prediction, target):
    prediction_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    prediction_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(prediction_x, target_x) + F.l1_loss(prediction_y, target_y)


def total_variation(value):
    variation_x = (value[..., :, 1:] - value[..., :, :-1]).abs().mean()
    variation_y = (value[..., 1:, :] - value[..., :-1, :]).abs().mean()
    return variation_x + variation_y


def supported_physics_distance(specular, physics_specular):
    magnitude = physics_specular.detach().abs().mean(dim=1, keepdim=True)
    scale = torch.quantile(magnitude.flatten(1), 0.95, dim=1).view(-1, 1, 1, 1)
    support = (magnitude / scale.clamp_min(1e-4)).clamp(0.0, 1.0).square()
    error = (specular - physics_specular.detach()).abs().mean(dim=1, keepdim=True)
    numerator = (support * error).flatten(1).sum(dim=1)
    denominator = support.flatten(1).sum(dim=1).clamp_min(1.0)
    return (numerator / denominator).mean()


class PGUNetLoss(nn.Module):
    """Enhancement, component and PBSIR losses (manuscript Eqs. 18-25).

    Coefficients are expanded products, e.g. lambda_d_structure=lambda_D*beta_D.
    The source implementation uses 0.01 for S-TV; beta_S is rounded in the text.
    Optional parameter-map terms require the corresponding reference maps.
    """
    def __init__(
        self,
        lambda_ssim=0.1,
        lambda_m=0.0005,
        lambda_inv=0.03,
        lambda_rob=0.01,
        lambda_d_structure=0.05,
        lambda_d_allocation=0.5,
        lambda_s_reference=1.5,
        lambda_s_smooth=0.01,
        lambda_m_physics=0.05,
        s_reference_scale=0.25,
        s_lowpass_factor=16,
        s_max_amplitude=0.25,
        min_exponent=2.0,
        max_exponent=96.0,
    ):
        super().__init__()
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_m = float(lambda_m)
        self.lambda_inv = float(lambda_inv)
        self.lambda_rob = float(lambda_rob)
        self.lambda_d_structure = float(lambda_d_structure)
        self.lambda_d_allocation = float(lambda_d_allocation)
        self.lambda_s_reference = float(lambda_s_reference)
        self.lambda_s_smooth = float(lambda_s_smooth)
        self.lambda_m_physics = float(lambda_m_physics)
        self.s_reference_scale = float(s_reference_scale)
        self.s_lowpass_factor = int(s_lowpass_factor)
        self.s_max_amplitude = float(s_max_amplitude)
        self.min_exponent = float(min_exponent)
        self.max_exponent = float(max_exponent)

    def normalize_exponent(self, value):
        denominator = max(self.max_exponent - self.min_exponent, 1e-6)
        return ((value - self.min_exponent) / denominator).clamp(0.0, 1.0)

    def parameter_distance(self, first, second, detach_second=False):
        target_ks = second["ks"].detach() if detach_second else second["ks"]
        target_n = second["n"].detach() if detach_second else second["n"]
        return F.l1_loss(first["ks"], target_ks) + F.l1_loss(
            self.normalize_exponent(first["n"]),
            self.normalize_exponent(target_n),
        )

    def component_references(self, target, low_input, final_specular):
        indirect_reference = self.s_reference_scale * low_frequency_projection(
            (target - low_input).clamp_min(0.0), self.s_lowpass_factor
        )
        indirect_reference = indirect_reference.clamp(0.0, self.s_max_amplitude)
        diffuse_reference = (
            target - indirect_reference - final_specular.detach()
        ).clamp(0.0, 1.0)
        return diffuse_reference.detach(), indirect_reference.detach()

    def forward(
        self,
        reconstruction,
        target,
        low_input,
        final_diffuse,
        final_indirect,
        final_specular,
        physics_specular=None,
        low_parameters=None,
        high_parameters=None,
        robust_parameters=None,
    ):
        prediction = reconstruction.clamp(0.0, 1.0)
        enhancement = F.l1_loss(reconstruction, target)
        ssim = 1.0 - ssim_index(prediction, target)
        sparse_m = final_specular.abs().mean()
        diffuse_structure = gradient_consistency(final_diffuse, target)
        diffuse_reference, indirect_reference = self.component_references(
            target, low_input, final_specular
        )
        diffuse_allocation = F.l1_loss(final_diffuse, diffuse_reference)
        indirect_consistency = F.l1_loss(final_indirect, indirect_reference)
        indirect_smoothness = total_variation(final_indirect)
        zero = reconstruction.new_zeros(())
        physics_consistency = (
            supported_physics_distance(final_specular, physics_specular)
            if physics_specular is not None
            else zero
        )
        invariance = (
            self.parameter_distance(low_parameters, high_parameters, detach_second=True)
            if low_parameters is not None and high_parameters is not None
            else zero
        )
        robustness = (
            self.parameter_distance(robust_parameters, low_parameters, detach_second=True)
            if robust_parameters is not None and low_parameters is not None
            else zero
        )
        total = (
            enhancement
            + self.lambda_ssim * ssim
            + self.lambda_m * sparse_m
            + self.lambda_inv * invariance
            + self.lambda_rob * robustness
            + self.lambda_d_structure * diffuse_structure
            + self.lambda_d_allocation * diffuse_allocation
            + self.lambda_s_reference * indirect_consistency
            + self.lambda_s_smooth * indirect_smoothness
            + self.lambda_m_physics * physics_consistency
        )
        return {
            "total": total,
            "enhancement": enhancement,
            "ssim": ssim,
            "sparse_m": sparse_m,
            "invariance": invariance,
            "robustness": robustness,
            "d_structure": diffuse_structure,
            "d_allocation": diffuse_allocation,
            "s_reference": indirect_consistency,
            "s_smooth": indirect_smoothness,
            "m_physics": physics_consistency,
            "d_mean": final_diffuse.detach().abs().mean(),
            "s_mean": final_indirect.detach().abs().mean(),
            "d_reference_mean": diffuse_reference.abs().mean(),
            "s_reference_mean": indirect_reference.abs().mean(),
            "m_mean": final_specular.detach().abs().mean(),
            "m_pbsir_mean": (
                physics_specular.detach().abs().mean()
                if physics_specular is not None
                else zero
            ),
        }
