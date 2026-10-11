
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


def load_images(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    images = data["images"].permute(0, 3, 1, 2).float() / 255.0
    labels = data["labels"]
    return images, labels


def standardize(images):
    cfg = PARAMS["preprocessing"]
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    return (images - mean) / std


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images, labels = load_images(RAW_DIR / "train.pt")
    test_images, test_labels = load_images(RAW_DIR / "test.pt")

    images = standardize(images)
    test_images = standardize(test_images)

    seed = PARAMS["seed"]
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(labels), generator=generator)

    val_size = round(
        len(labels) * PARAMS["preprocessing"]["validation_split"]
    )
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    datasets = {
        "train": (images[train_idx], labels[train_idx]),
        "val": (images[val_idx], labels[val_idx]),
        "test": (test_images, test_labels),
    }

    for split, (x, y) in datasets.items():
        torch.save(
            {"images": x, "labels": y},
            OUT_DIR / f"{split}.pt",
        )
        print(f"Saved {split}: {len(y)} images")


if __name__ == "__main__":
    main()
