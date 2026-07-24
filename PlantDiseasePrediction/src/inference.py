import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import get_val_transforms
from model import create_alexnet
from utils import MODELS_DIR, get_device

DEFAULT_MODEL_PATH = MODELS_DIR / "alexnet.pkl"

ALEXNET_CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
]


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _load_alexnet(model_path: Path, device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Expected state_dict checkpoint: {model_path}")

    state_dict = _strip_module_prefix(checkpoint["state_dict"])
    num_classes = state_dict["classifier.6.weight"].shape[0]

    model = create_alexnet(num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    class_names = checkpoint.get("class_names") or checkpoint.get("classes")
    if class_names is None:
        if num_classes == len(ALEXNET_CLASS_NAMES):
            class_names = ALEXNET_CLASS_NAMES
        else:
            class_names = [f"Class_{index}" for index in range(num_classes)]

    return model, class_names


def predict_image(
    image_path: str | Path,
    model_path: str | Path | None = None,
    device: torch.device | None = None,
) -> dict:
    device = device or get_device()
    model_path = Path(model_path or DEFAULT_MODEL_PATH)
    model, class_names = _load_alexnet(model_path, device)

    transform = get_val_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)

    confidence, predicted_idx = torch.max(probabilities, dim=0)
    predicted_idx = predicted_idx.item()

    top_k = min(3, len(class_names))
    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_predictions = [
        {
            "index": index.item(),
            "class_name": class_names[index.item()],
            "probability": prob.item(),
        }
        for prob, index in zip(top_probs, top_indices)
    ]

    return {
        "image": str(image_path),
        "class_index": predicted_idx,
        "class_name": class_names[predicted_idx],
        "confidence": confidence.item(),
        "top_predictions": top_predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a plant leaf image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to pretrained model (default: models/alexnet.pkl)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = predict_image(args.image, model_path=args.model_path)

    print(f"Image: {result['image']}")
    print(f"Class Index: {result['class_index']}")
    print(f"Class Name: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Top-3 Predictions:")
    for rank, pred in enumerate(result["top_predictions"], start=1):
        print(
            f"  {rank}. {pred['class_name']} "
            f"(index: {pred['index']}, probability: {pred['probability']:.4f})"
        )
