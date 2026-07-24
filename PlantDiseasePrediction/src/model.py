from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from utils import MODELS_DIR


def create_alexnet(num_classes: int) -> nn.Module:
    try:
        model = models.alexnet(weights=None)
    except TypeError:
        model = models.alexnet(pretrained=False)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _infer_num_classes(state_dict: dict) -> int:
    for key in ("classifier.6.weight", "module.classifier.6.weight"):
        if key in state_dict:
            return state_dict[key].shape[0]
    raise KeyError("Could not infer number of classes from state_dict")


def _extract_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if any(key.endswith(".weight") for key in checkpoint):
            return checkpoint
    raise ValueError("Unsupported checkpoint format")


def load_pretrained_weights(
    model: nn.Module,
    path: str | Path,
    strict: bool = False,
) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        source_state = checkpoint.state_dict()
    else:
        source_state = _strip_module_prefix(_extract_state_dict(checkpoint))

    model_state = model.state_dict()
    filtered_state = {
        key: value
        for key, value in source_state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model.load_state_dict(filtered_state, strict=False)
    if strict:
        missing = set(model_state) - set(filtered_state)
        if missing:
            raise RuntimeError(f"Missing keys when loading with strict=True: {missing}")
    return model


def load_model_from_checkpoint(
    path: str | Path,
    num_classes: int | None = None,
    device: torch.device | str = "cpu",
) -> nn.Module:
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        model.to(device)
        model.eval()
        return model

    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
    if num_classes is None:
        num_classes = _infer_num_classes(state_dict)

    model = create_alexnet(num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_inference_model(
    path: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> nn.Module:
    path = path or MODELS_DIR / "alexnet.pkl"
    return load_model_from_checkpoint(path, device=device)
