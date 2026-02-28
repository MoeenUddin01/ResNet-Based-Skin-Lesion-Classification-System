"""
Script to count the number of images in each class folder.
"""

from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)


def count_images_per_class(processed_dir: Path) -> dict[str, int]:
    """Counts the number of images in each class directory.

    Args:
        processed_dir: Path to the processed dataset directory
            containing class subdirectories.

    Returns:
        A dictionary mapping the class name to the count of images.

    Raises:
        FileNotFoundError: If the processed directory does not exist.
    """
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed dataset directory not found at {processed_dir}"
        )

    class_counts: dict[str, int] = {}
    total_images = 0

    logging.info(f"Scanning directory: {processed_dir}")
    logging.info("-" * 40)

    for class_dir in processed_dir.iterdir():
        if class_dir.is_dir():
            # Count the number of files (images) in the class directory
            count = sum(1 for _ in class_dir.iterdir() if _.is_file())
            class_counts[class_dir.name] = count
            total_images += count

    # Print the counts sorted alphabetically by class name
    logging.info("Image counts per class:")
    for class_name, count in sorted(class_counts.items()):
        logging.info(f"  - {class_name}: {count} images")

    logging.info("-" * 40)
    logging.info(f"Total images across all classes: {total_images}")

    return class_counts


if __name__ == "__main__":
    BASE_DIR = Path(
        "/home/moeenuddin/Desktop/Deep_learning/Skin_Lesion"
        "/ResNet-Based-Skin-Lesion-Classification-System"
    )
    PROCESSED_DATA_DIR = BASE_DIR / "dataset" / "processed"

    try:
        count_images_per_class(processed_dir=PROCESSED_DATA_DIR)
    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
