from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loaders(
    data_dir: str | Path,
    batch_size: int = 32,
    img_size: tuple[int, int] = (224, 224),
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Loads the skin lesion dataset and returns train and test DataLoaders.

    Args:
        data_dir: Path to the root directory containing the class subdirectories.
        batch_size: The number of images per batch.
        img_size: The target image size as a (height, width) tuple.
        num_workers: The number of subprocesses to use for data loading.
        seed: The random seed for the train-test split to ensure reproducibility.

    Returns:
        A tuple containing:
            - train_loader: The DataLoader for the training subset.
            - test_loader: The DataLoader for the testing subset.
            - class_names: A list of class names inferred from the directory structure.

    Raises:
        ValueError: If the directory does not exist, is empty, or contains no usable data.
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise ValueError(f"The data directory does not exist: {data_path}")

    # Check if directory is empty
    if not any(data_path.iterdir()):
        raise ValueError(f"The data directory is empty: {data_path}")

    # ImageNet normalization stats
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = datasets.ImageFolder(root=str(data_path), transform=transform)
    class_names = full_dataset.classes

    if not full_dataset:
        raise ValueError(f"No valid images found in the data directory: {data_path}")

    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    test_size = total_size - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, class_names
