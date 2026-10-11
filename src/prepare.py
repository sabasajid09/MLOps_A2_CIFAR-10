
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for split, train in [("train", True), ("test", False)]:
        dataset = CIFAR10(
            root=str(RAW_DIR / "download"),
            train=train,
            download=True,
        )

        output = {
            "images": torch.from_numpy(dataset.data.copy()),
            "labels": torch.tensor(dataset.targets, dtype=torch.long),
        }

        torch.save(output, RAW_DIR / f"{split}.pt")
        print(f"Saved {split} dataset: {len(dataset)} images")


if __name__ == "__main__":
    main()
