"""
Data splitting script for the HAM10000 dataset.
Splits the processed class-wise directories into train, val, and test sets.
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def split_dataset(
    processed_dir: Path,
    output_dir: Path,
    test_size: float = 0.1,
    val_ratio: float = 0.3,
    seed: int = 42,
) -> None:
    """Splits the processed dataset into train, val, and test subsets.

    The testing data will exactly contain `test_size` fraction of the total data.
    The remaining data will be split according to `val_ratio`, which dictates
    the validation fraction out of that remaining subgroup.

    Args:
        processed_dir: Path to the processed dataset containing class subfolders.
        output_dir: Path to the output directory where splits will be created.
        test_size: Proportion of the dataset to hold out strictly for testing.
        val_ratio: Proportion of the remaining data to allocate to validation.
        seed: Random seed for reproducibility.

    Raises:
        FileNotFoundError: If the processed directory doesn't exist.
        ValueError: If no class directories are found.
    """
    random.seed(seed)

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class directories found in {processed_dir}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = list(class_dir.glob("*.jpg"))

        # Sort for reproducibility before shuffling
        images.sort()
        random.shuffle(images)

        total_images = len(images)

        # Take exactly test_size out of the full dataset
        num_test = int(total_images * test_size)

        # Calculate train/val from the REMAINING data
        remaining_images_count = total_images - num_test
        num_val = int(remaining_images_count * val_ratio)

        test_images = images[:num_test]
        val_images = images[num_test : num_test + num_val]
        train_images = images[num_test + num_val :]

        # Copy images to their respective split directories
        _copy_images(test_images, test_dir / class_name)
        _copy_images(val_images, val_dir / class_name)
        _copy_images(train_images, train_dir / class_name)

        logging.info(
            f"Class '{class_name}': {len(train_images)} train, "
            f"{len(val_images)} val, {len(test_images)} test."
        )

    logging.info(f"Dataset splitting complete. Files saved to {output_dir}")


def _copy_images(images: list[Path], target_dir: Path) -> None:
    """Helper function to create a directory and copy images.

    Args:
        images: List of image paths to copy.
        target_dir: Destination directory path.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    for img_path in images:
        shutil.copy2(img_path, target_dir / img_path.name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split the processed HAM10000 dataset into train/val/test."
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="dataset/processed",
        help="Path to the processed class-organised dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/split",
        help="Output directory for the train/val/test split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of data to hold out for testing (default: 0.1).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.3,
        help="Fraction of remaining data for validation (default: 0.3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    split_dataset(
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        test_size=args.test_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
