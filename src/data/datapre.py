"""
Data combining and preprocessing script for the HAM10000 dataset.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def combine_and_organize_data(
    raw_dir: Path,
    processed_dir: Path,
    metadata_csv: Path,
) -> None:
    """Combines raw HAM10000 images and organizes them into class-specific directories.

    Args:
        raw_dir: Path to the raw dataset directory containing the parts.
        processed_dir: Path to the output directory for processed class folders.
        metadata_csv: Path to the HAM10000 metadata CSV file.

    Raises:
        FileNotFoundError: If the metadata CSV or required raw directories do not exist.
        KeyError: If the required columns ('image_id', 'dx') are missing from the CSV.
    """
    part_1_dir = raw_dir / "ham10000_images_part_1"
    part_2_dir = raw_dir / "ham10000_images_part_2"
    combined_dir = raw_dir / "combined_images"

    # 1. Move images from parts 1 and 2 to a combined folder
    combined_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created combined directory at {combined_dir}")

    for part_dir in [part_1_dir, part_2_dir]:
        if not part_dir.exists():
            logging.warning(f"Directory {part_dir} not found. Skipping.")
            continue

        logging.info(f"Moving images from {part_dir} to {combined_dir}...")
        for img_path in part_dir.glob("*.jpg"):
            shutil.move(str(img_path), str(combined_dir / img_path.name))

    # 2. Read image_id and dx from the metadata CSV
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_csv}")

    logging.info("Reading metadata CSV...")
    df = pd.read_csv(metadata_csv)

    if "image_id" not in df.columns or "dx" not in df.columns:
        raise KeyError("Metadata CSV must contain 'image_id' and 'dx' columns.")

    # 3 & 4. Move images to class-specific folders inside the processed directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Organizing images into class folders at {processed_dir}...")

    moved_count = 0
    missing_count = 0

    for _, row in df.iterrows():
        image_id = row["image_id"]
        class_label = row["dx"]

        img_name = f"{image_id}.jpg"
        source_path = combined_dir / img_name
        class_dir = processed_dir / class_label
        dest_path = class_dir / img_name

        if source_path.exists():
            # Create the class directory if it doesn't exist
            class_dir.mkdir(parents=True, exist_ok=True)
            # Move the image to the final location
            shutil.move(str(source_path), str(dest_path))
            moved_count += 1
        else:
            missing_count += 1

    logging.info(f"Successfully moved {moved_count} images to class folders.")
    if missing_count > 0:
        logging.warning(f"{missing_count} images from metadata were not found.")

    # Cleanup: Remove the empty combined directory after everything is moved
    if combined_dir.exists() and not any(combined_dir.iterdir()):
        combined_dir.rmdir()
        logging.info("Removed empty temporary combined directory.")


if __name__ == "__main__":
    # Base paths properly split across lines to remain under 88 characters
    BASE_DIR = Path(
        "/home/moeenuddin/Desktop/Deep_learning/Skin_Lesion"
        "/ResNet-Based-Skin-Lesion-Classification-System"
    )
    RAW_DATA_DIR = BASE_DIR / "dataset" / "raw"
    PROCESSED_DATA_DIR = BASE_DIR / "dataset" / "processed"
    METADATA_FILE = RAW_DATA_DIR / "HAM10000_metadata.csv"

    combine_and_organize_data(
        raw_dir=RAW_DATA_DIR,
        processed_dir=PROCESSED_DATA_DIR,
        metadata_csv=METADATA_FILE,
    )
