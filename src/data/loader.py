from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(
    data_dir: str | Path,
    batch_size: int = 32,
    img_size: tuple[int, int] = (224, 224),
    num_workers: int = 4,
    subset_fraction: float = 0.0,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Loads the train and val skin lesion datasets and returns DataLoaders.

    Args:
        data_dir: Path to the root split directory containing 'train' and 'val' subfolders.
        batch_size: The number of images per batch.
        img_size: The target image size as a (height, width) tuple.
        num_workers: The number of subprocesses to use for data loading.
        subset_fraction: If > 0.0, limits the datasets to this percentage of samples (e.g., 0.1 for 10%).

    Returns:
        A tuple containing:
            - train_loader: The DataLoader for the training subset.
            - val_loader: The DataLoader for the validation subset.
            - class_names: A list of class names inferred from the directory structure.

    Raises:
        ValueError: If the required subdirectories are missing.
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    for d in [train_dir, val_dir]:
        if not d.exists():
            raise ValueError(f"Required data directory does not exist: {d}")

    # ImageNet normalization stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Strong augmentations for training
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Standard resizes/crops for validation
    val_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

    class_names = train_dataset.classes
    pin_memory = torch.cuda.is_available()

    if subset_fraction > 0.0:
        # Use first N% of samples per class to simulate a small dataset quickly
        # and ensure all classes are represented (stratified subset)
        from collections import defaultdict
        
        def get_stratified_indices(dataset: datasets.ImageFolder) -> list[int]:
            class_indices = defaultdict(list)
            for idx, (_, class_idx) in enumerate(dataset.samples):
                class_indices[class_idx].append(idx)
            
            stratified_indices = []
            for class_idx, indices in class_indices.items():
                num_samples = max(1, int(len(indices) * subset_fraction))
                stratified_indices.extend(indices[:num_samples])
            return stratified_indices
            
        train_indices = get_stratified_indices(train_dataset)
        val_indices = get_stratified_indices(val_dataset)
        
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    # Handle class imbalance for the entire loader via sampling
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler

    # Calculate class counts from the actual training subset
    if isinstance(train_dataset, torch.utils.data.Subset):
        train_targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
    else:
        train_targets = train_dataset.targets

    class_counts = Counter(train_targets)
    total_samples = len(train_targets)
    
    # Calculate weights per class and assign to each sample in the dataset
    class_weights = {
        cls: total_samples / count for cls, count in class_counts.items()
    }
    sample_weights = [class_weights[t] for t in train_targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, class_names


def get_test_loader(
    data_dir: str | Path,
    batch_size: int = 32,
    img_size: tuple[int, int] = (224, 224),
    num_workers: int = 4,
) -> tuple[DataLoader, list[str]]:
    """Loads the test skin lesion dataset and returns a DataLoader.

    Args:
        data_dir: Path to the root split directory containing 'test' subfolder.
        batch_size: The number of images per batch.
        img_size: The target image size as a (height, width) tuple.
        num_workers: The number of subprocesses to use for data loading.

    Returns:
        A tuple containing:
            - test_loader: The DataLoader for the testing subset.
            - class_names: A list of class names inferred from the directory structure.

    Raises:
        ValueError: If the test directory does not exist.
    """
    test_dir = Path(data_dir) / "test"

    if not test_dir.exists():
        raise ValueError(f"Required test directory does not exist: {test_dir}")

    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=transform)
    pin_memory = torch.cuda.is_available()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return test_loader, test_dataset.classes


if __name__ == "__main__":
    dataloader_train, dataloader_val, class_names = get_data_loaders(
        data_dir="dataset/split"
    )
    print("Classes:", class_names)
    print("Train batches:", len(dataloader_train))
    print("Val batches:", len(dataloader_val))

    dataloader_test, test_classes = get_test_loader(data_dir="dataset/split")
    print("Test batches:", len(dataloader_test))
