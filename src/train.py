
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


def load_split(name):
    data = torch.load(
        DATA_DIR / f"{name}.pt",
        map_location="cpu",
        weights_only=False,
    )
    return TensorDataset(data["images"], data["labels"])


class CNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, cfg["conv1_channels"], kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg["conv1_channels"]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                cfg["conv1_channels"],
                cfg["conv2_channels"],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(cfg["conv2_channels"], cfg["dense_units"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["dense_units"], 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = count = 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)

    return total_loss / count, correct / count


def main():
    torch.manual_seed(PARAMS["seed"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = PARAMS["training"]["batch_size"]

    train_loader = DataLoader(
        load_split("train"), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        load_split("val"), batch_size=batch_size
    )

    model = CNN(PARAMS["model"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=PARAMS["training"]["learning_rate"],
        weight_decay=PARAMS["training"]["weight_decay"],
    )

    history = []
    for epoch in range(PARAMS["training"]["epochs"]):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

    torch.save(model.state_dict(), MODEL_DIR / "model.pth")
    pd.DataFrame(history).to_csv(MODEL_DIR / "history.csv", index=False)
    print(f"Saved model and history. Device: {device}")


if __name__ == "__main__":
    main()
