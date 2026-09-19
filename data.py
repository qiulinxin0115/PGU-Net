"""Paired RGB images and signed camera-space normal maps."""
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_index(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(str(folder))
    result = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in result:
                raise ValueError("Duplicate image stem: " + path.stem)
            result[path.stem] = path
    if not result:
        raise ValueError("No images found in " + str(folder))
    return result


def read_rgb(path):
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def read_normal(path):
    # Use vector data, not a rendered/colorized normal visualization.
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError("Normal arrays must contain signed floating-point vectors.")
        value = torch.from_numpy(np.array(array, dtype=np.float32))
    else:
        value = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            value = value.get("normal", value.get("normals"))
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise ValueError("Normal .pt must contain a float tensor or a normal/normals entry.")
        value = value.float()
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 3 and value.shape[-1] == 3 and value.shape[0] != 3:
        value = value.permute(2, 0, 1)
    if value.ndim != 3 or value.shape[0] != 3 or not torch.isfinite(value).all():
        raise ValueError("Expected finite normals with shape [3,H,W] or [H,W,3]: " + str(path))
    if value.abs().max() > 1.001:
        raise ValueError("Expected signed normal coordinates in [-1,1]: " + str(path))
    if (value.square().sum(0) < 1e-12).any():
        raise ValueError("Zero-length normal vectors found: " + str(path))
    return F.normalize(value, dim=0).contiguous()


class ImageNormalDataset(Dataset):
    def __init__(self, low_dir, normal_dir, high_dir=None, crop_size=0,
                 augment=False, low_prefix="", high_prefix="", normal_prefix=""):
        low = image_index(low_dir)
        high = image_index(high_dir) if high_dir else None
        normal_dir = Path(normal_dir)
        if not normal_dir.is_dir():
            raise FileNotFoundError(str(normal_dir))
        self.samples = []
        self.crop_size = int(crop_size)
        self.augment = augment
        if self.crop_size and self.crop_size < 32:
            raise ValueError("crop_size must be 0 or at least 32.")
        for stem, low_path in low.items():
            if not stem.startswith(low_prefix):
                raise ValueError("Image does not start with --low-prefix: " + stem)
            key = stem[len(low_prefix):]
            high_path = high.get(high_prefix + key) if high is not None else None
            if high is not None and high_path is None:
                raise FileNotFoundError("Missing reference for " + stem)
            candidates = [normal_dir / (normal_prefix + key + ext) for ext in (".npy", ".pt")]
            candidates = [p for p in candidates if p.is_file()]
            if len(candidates) != 1:
                raise ValueError("Expected exactly one .npy or .pt normal for " + stem)
            self.samples.append((stem, low_path, high_path, candidates[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        stem, low_path, high_path, normal_path = self.samples[index]
        low = read_rgb(low_path)
        normal = read_normal(normal_path)
        high = read_rgb(high_path) if high_path else None
        if high is not None and low.shape != high.shape:
            raise ValueError("Low/high dimensions differ: " + stem)
        if low.shape != normal.shape:
            raise ValueError("Normal map must be aligned to the original RGB resolution: " + stem)
        height, width = low.shape[-2:]
        if min(height, width) < max(self.crop_size, 32):
            raise ValueError("Image smaller than requested crop/minimum size: " + stem)
        values = {"low": low, "normal": normal}
        if high is not None:
            values["high"] = high
        if self.crop_size:
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
            values = {k: v[:, top:top+self.crop_size, left:left+self.crop_size]
                      for k, v in values.items()}
        if self.augment:
            if random.random() < 0.5:
                values = {k: v.flip(-1) for k, v in values.items()}
                values["normal"][0].neg_()
            if random.random() < 0.5:
                values = {k: v.flip(-2) for k, v in values.items()}
                values["normal"][1].neg_()
        values = {k: v.contiguous() for k, v in values.items()}
        values["name"] = stem
        return values

