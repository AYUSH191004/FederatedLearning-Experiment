import torch
from torchvision import transforms
from medmnist import PathMNIST

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "medmnist"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_pathmnist_datasets(data_dir=DATA_DIR, download=True):
    """
    Load the official PathMNIST train, validation, and test datasets.

    Images are converted to tensors with values scaled to [0, 1].
    No augmentation is applied at this stage.
    """

    transform = transforms.ToTensor()

    train_dataset = PathMNIST(
        split="train",
        root=data_dir,
        download=download,
        transform=transform
    )

    val_dataset = PathMNIST(
        split="val",
        root=data_dir,
        download=download,
        transform=transform
    )

    test_dataset = PathMNIST(
        split="test",
        root=data_dir,
        download=download,
        transform=transform
    )

    return train_dataset, val_dataset, test_dataset


def inspect_dataset(dataset, split_name):
    """Print basic dataset properties."""

    image, label = dataset[0]

    print(f"\n{split_name} Dataset")
    print("-" * 40)
    print(f"Number of samples : {len(dataset)}")
    print(f"Image shape       : {image.shape}")
    print(f"Label             : {label}")
    print(f"Label shape       : {label.shape}")
    print(f"Image dtype       : {image.dtype}")
    print(f"Image min         : {image.min().item():.4f}")
    print(f"Image max         : {image.max().item():.4f}")


if __name__ == "__main__":

    train_dataset, val_dataset, test_dataset = get_pathmnist_datasets()

    inspect_dataset(train_dataset, "Train")
    inspect_dataset(val_dataset, "Validation")
    inspect_dataset(test_dataset, "Test")