
from pathlib import Path
import json

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(
        DATA_DIR / "test.pt",
        map_location="cpu",
        weights_only=False,
    )
    loader = DataLoader(
        TensorDataset(data["images"], data["labels"]),
        batch_size=PARAMS["training"]["batch_size"],
    )

    model = CNN(PARAMS["model"]).to(device)
    model.load_state_dict(
        torch.load(
            MODEL_DIR / "model.pth",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * len(labels)
            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    test_loss = total_loss / len(all_labels)
    test_accuracy = sum(
        a == b for a, b in zip(all_labels, all_predictions)
    ) / len(all_labels)

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(10)))
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(10)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names
    )
    display.plot(xticks_rotation=45, cmap="Blues")
    plt.title("CIFAR-10 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_examples": len(all_labels),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_plot": "figures/confusion_matrix.png",
    }

    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4%}")
    print(f"Saved metrics to {ROOT / 'metrics.json'}")
    print(f"Saved confusion matrix to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
