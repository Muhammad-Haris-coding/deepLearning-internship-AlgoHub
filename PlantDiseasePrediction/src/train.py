import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_train_loader, get_val_loader
from model import create_alexnet, load_pretrained_weights
from utils import (
    DEFAULT_NUM_CLASSES,
    RESULTS_DIR,
    get_device,
    load_classes,
    save_checkpoint,
)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total


def validate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, running_correct / total


def train(args: argparse.Namespace) -> None:
    device = get_device()
    class_names = load_classes()
    num_classes = len(class_names) if class_names else DEFAULT_NUM_CLASSES

    train_loader = get_train_loader(batch_size=args.batch_size)
    val_loader = get_val_loader(batch_size=args.batch_size)

    model = create_alexnet(num_classes).to(device)

    if args.pretrained:
        load_pretrained_weights(model, args.pretrained)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    save_path = Path(args.output) if args.output else RESULTS_DIR / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "arch": "alexnet",
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": val_acc,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                save_path,
            )
            print(f"Saved best model to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AlexNet for plant disease classification")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Optional path to pretrained weights (e.g. models/alexnet.pkl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "best_model.pth"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
