from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_NUM_CLASSES = 39

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_data_dir() -> Path:
    return PROJECT_ROOT / "dataset" / "PlantVillage"


def load_classes(data_dir: Path | None = None) -> list[str]:
    data_dir = data_dir or get_data_dir()
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    return sorted(p.name for p in train_dir.iterdir() if p.is_dir())


def save_checkpoint(state: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
