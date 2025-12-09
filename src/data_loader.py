"""
Data Loading and Preprocessing Module
Handles loading seismic images, masks, and depth information
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_IMAGES_DIR,
    TRAIN_MASKS_DIR,
    TEST_IMAGES_DIR,
    DEPTHS_CSV,
    IMG_HEIGHT,
    IMG_WIDTH,
    TRAINING_CONFIG,
)


class DataLoader:
    """
    Data loader for TGS Salt Identification dataset.
    Handles loading, preprocessing, and splitting of seismic data.
    """

    def __init__(self):
        self.depths_df = None
        self.train_ids = None
        self.test_ids = None
        self._load_depths()

    def _load_depths(self) -> None:
        """Load depth information from CSV."""
        self.depths_df = pd.read_csv(DEPTHS_CSV)
        self.depths_df.set_index("id", inplace=True)
        print(f"Loaded depth information for {len(self.depths_df)} samples")
        print(f"Depth range: {self.depths_df['z'].min()} - {self.depths_df['z'].max()} meters")

    def get_depth(self, image_id: str) -> float:
        """Get depth value for a given image ID."""
        return self.depths_df.loc[image_id, "z"]

    def load_image(self, image_id: str, is_train: bool = True) -> np.ndarray:
        """
        Load and preprocess a single seismic image.

        Args:
            image_id: Image identifier
            is_train: Whether to load from train or test directory

        Returns:
            Grayscale image as numpy array (H, W)
        """
        if is_train:
            img_path = os.path.join(TRAIN_IMAGES_DIR, f"{image_id}.png")
        else:
            img_path = os.path.join(TEST_IMAGES_DIR, f"{image_id}.png")

        img = Image.open(img_path)
        # Convert to grayscale if needed
        if img.mode == "RGBA":
            img = img.convert("L")
        elif img.mode == "RGB":
            img = img.convert("L")

        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array

    def load_mask(self, image_id: str) -> np.ndarray:
        """
        Load salt mask for a given image.

        Args:
            image_id: Image identifier

        Returns:
            Binary mask as numpy array (H, W)
        """
        mask_path = os.path.join(TRAIN_MASKS_DIR, f"{image_id}.png")
        mask = Image.open(mask_path)
        mask_array = np.array(mask, dtype=np.float32) / 255.0
        # Binarize the mask
        mask_array = (mask_array > 0.5).astype(np.float32)
        return mask_array

    def get_train_ids(self) -> List[str]:
        """Get list of training image IDs."""
        if self.train_ids is None:
            self.train_ids = [
                f.replace(".png", "")
                for f in os.listdir(TRAIN_IMAGES_DIR)
                if f.endswith(".png")
            ]
        return self.train_ids

    def get_test_ids(self) -> List[str]:
        """Get list of test image IDs."""
        if self.test_ids is None:
            self.test_ids = [
                f.replace(".png", "")
                for f in os.listdir(TEST_IMAGES_DIR)
                if f.endswith(".png")
            ]
        return self.test_ids

    def load_all_train_data(
        self, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load all training data.

        Returns:
            images: Array of shape (N, H, W)
            masks: Array of shape (N, H, W)
            depths: Array of shape (N,)
            ids: List of image IDs
        """
        train_ids = self.get_train_ids()
        n_samples = len(train_ids)

        images = np.zeros((n_samples, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        masks = np.zeros((n_samples, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        depths = np.zeros(n_samples, dtype=np.float32)

        for i, img_id in enumerate(train_ids):
            if verbose and (i + 1) % 500 == 0:
                print(f"Loading: {i + 1}/{n_samples}")

            images[i] = self.load_image(img_id, is_train=True)
            masks[i] = self.load_mask(img_id)
            depths[i] = self.get_depth(img_id)

        if verbose:
            print(f"Loaded {n_samples} training samples")
            print(f"Images shape: {images.shape}")
            print(f"Masks shape: {masks.shape}")
            print(f"Salt coverage: {masks.mean() * 100:.2f}%")

        return images, masks, depths, train_ids

    def compute_coverage_class(self, mask: np.ndarray) -> int:
        """
        Compute coverage class for stratified splitting.

        Args:
            mask: Binary mask array

        Returns:
            Coverage class (0-10)
        """
        coverage = mask.mean()
        # Create 11 coverage classes (0-10)
        return min(int(coverage * 10), 10)

    def get_stratified_split(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        depths: np.ndarray,
        ids: List[str],
        test_size: float = None,
    ) -> Dict:
        """
        Create stratified train/validation split based on salt coverage.

        Returns:
            Dictionary with train and validation data
        """
        if test_size is None:
            test_size = TRAINING_CONFIG["test_size"]

        # Compute coverage classes for stratification (use fewer bins for small datasets)
        n_samples = len(masks)
        if n_samples < 500:
            # Use binary stratification for small datasets
            coverage_classes = np.array([(m.mean() > 0).astype(int) for m in masks])
        else:
            coverage_classes = np.array([self.compute_coverage_class(m) for m in masks])

        # Check if stratification is possible
        unique, counts = np.unique(coverage_classes, return_counts=True)
        min_count = counts.min()

        # Fall back to non-stratified if any class has < 2 samples
        stratify_param = coverage_classes if min_count >= 2 else None

        if stratify_param is None:
            print("Warning: Using non-stratified split due to small dataset size")

        # Split
        (
            train_images,
            val_images,
            train_masks,
            val_masks,
            train_depths,
            val_depths,
            train_ids,
            val_ids,
            train_coverage,
            val_coverage,
        ) = train_test_split(
            images,
            masks,
            depths,
            ids,
            coverage_classes,
            test_size=test_size,
            random_state=TRAINING_CONFIG["random_state"],
            stratify=stratify_param,
        )

        return {
            "train": {
                "images": train_images,
                "masks": train_masks,
                "depths": train_depths,
                "ids": train_ids,
                "coverage": train_coverage,
            },
            "val": {
                "images": val_images,
                "masks": val_masks,
                "depths": val_depths,
                "ids": val_ids,
                "coverage": val_coverage,
            },
        }

    def get_depth_statistics(self, depths: np.ndarray) -> Dict:
        """Compute depth statistics for analysis."""
        return {
            "min": depths.min(),
            "max": depths.max(),
            "mean": depths.mean(),
            "std": depths.std(),
            "median": np.median(depths),
            "q25": np.percentile(depths, 25),
            "q75": np.percentile(depths, 75),
        }

    def get_salt_statistics(self, masks: np.ndarray) -> Dict:
        """Compute salt coverage statistics."""
        coverages = np.array([m.mean() for m in masks])
        has_salt = coverages > 0

        return {
            "total_samples": len(masks),
            "samples_with_salt": has_salt.sum(),
            "samples_without_salt": (~has_salt).sum(),
            "mean_coverage": coverages.mean(),
            "std_coverage": coverages.std(),
            "max_coverage": coverages.max(),
        }


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()

    print("\n--- Loading Training Data ---")
    images, masks, depths, ids = loader.load_all_train_data()

    print("\n--- Depth Statistics ---")
    depth_stats = loader.get_depth_statistics(depths)
    for k, v in depth_stats.items():
        print(f"{k}: {v:.2f}")

    print("\n--- Salt Statistics ---")
    salt_stats = loader.get_salt_statistics(masks)
    for k, v in salt_stats.items():
        print(f"{k}: {v}")

    print("\n--- Creating Stratified Split ---")
    split_data = loader.get_stratified_split(images, masks, depths, ids)
    print(f"Training samples: {len(split_data['train']['images'])}")
    print(f"Validation samples: {len(split_data['val']['images'])}")
