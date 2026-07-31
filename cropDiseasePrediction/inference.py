from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from PIL import Image as PILImage

from utils.class_names import CLASS_NAMES


def load_model(model_path: str = "best_plantdoc_model.pth") -> Tuple[torch.nn.Module, torch.device]:
    """Recreate the ResNet50 model, load state_dict and return model and device.

    Args:
        model_path: Path to the state_dict file.

    Returns:
        A tuple of (model, device).

    Raises:
        FileNotFoundError: If the checkpoint file is missing.
        RuntimeError: If the checkpoint cannot be loaded or keys mismatch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate base model with ImageNet weights for proper backbone initialization
    model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Recreate classifier head used during training
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, len(CLASS_NAMES)))

    try:
        state = torch.load(model_path, map_location=device)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Model checkpoint not found at '{model_path}'.") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint: {exc}") from exc

    try:
        model.load_state_dict(state)
    except Exception as exc:
        raise RuntimeError(f"Failed to load state_dict into model: {exc}") from exc

    model.to(device)
    model.eval()

    return model, device


def preprocess_image(image: PILImage.Image) -> torch.Tensor:
    """Preprocess a PIL image for ResNet50 inference.

    Steps: Resize to 256, center crop 224, convert to tensor and normalize with ImageNet stats.

    Args:
        image: PIL Image in RGB.

    Returns:
        Torch tensor with shape (1, 3, 224, 224).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=ResNet50_Weights.DEFAULT.transforms.mean,  # type: ignore
                             std=ResNet50_Weights.DEFAULT.transforms.std),   # type: ignore
    ])

    tensor = preprocess(image).unsqueeze(0)
    return tensor


def predict(image: PILImage.Image, model_device: Tuple[torch.nn.Module, torch.device] = None) -> Tuple[str, float, List[tuple], Dict[str, float]]:
    """Run inference on a PIL image and return predictions.

    Args:
        image: PIL Image
        model_device: Optional tuple (model, device). If None, the function will call load_model().

    Returns:
        predicted_class: str
        confidence: float (0-1)
        top5: List of (class, probability)
        probabilities: Dict[class_name, probability]
    """
    if model_device is None:
        model, device = load_model()
    else:
        model, device = model_device

    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)

    probs = probs.squeeze(0).cpu()

    top5_prob, top5_idx = torch.topk(probs, k=5)
    top5_prob = top5_prob.numpy().tolist()
    top5_idx = top5_idx.numpy().tolist()

    top5 = [(CLASS_NAMES[idx], float(top5_prob[i])) for i, idx in enumerate(top5_idx)]

    probabilities = {CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))}

    predicted_class = top5[0][0]
    confidence = top5[0][1]

    return predicted_class, confidence, top5, probabilities
