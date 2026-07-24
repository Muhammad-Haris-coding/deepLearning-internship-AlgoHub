from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import IMAGENET_MEAN, IMAGENET_STD, get_data_dir

IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4


def get_train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_val_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_train_loader(
    data_dir: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> DataLoader:
    data_dir = data_dir or get_data_dir()
    dataset = datasets.ImageFolder(
        root=data_dir / "train",
        transform=get_train_transforms(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_val_loader(
    data_dir: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> DataLoader:
    data_dir = data_dir or get_data_dir()
    dataset = datasets.ImageFolder(
        root=data_dir / "val",
        transform=get_val_transforms(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
