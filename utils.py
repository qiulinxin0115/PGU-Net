"""Shared checkpoint, image and evaluation helpers."""
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from losses.objectives import ssim_index
from models import PGUNet

FORMAT = "pgunet_public_train_v1"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def select_device(name):
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_checkpoint(path):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        raise ValueError("Expected a checkpoint saved by this release's train.py.")
    return checkpoint


def model_from_checkpoint(checkpoint):
    model = PGUNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, str(temporary))
    temporary.replace(path)


def save_rgb(path, tensor):
    value = tensor.detach().cpu().clamp(0, 1)
    if value.ndim == 4:
        value = value[0]
    array = (value.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(str(path))


def metrics(prediction, target):
    prediction = prediction.clamp(0, 1)
    mse = (prediction - target).square().mean().clamp_min(1e-12)
    return {"psnr": float(-10 * torch.log10(mse)),
            "ssim": float(ssim_index(prediction, target)),
            "mae": float((prediction - target).abs().mean())}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rng_state(generator):
    np_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same CUDA device count.")
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])

