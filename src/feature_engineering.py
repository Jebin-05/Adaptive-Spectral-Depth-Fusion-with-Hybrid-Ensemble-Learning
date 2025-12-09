"""
Feature Engineering Module
Extracts hand-crafted features for salt body segmentation
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import (
    sobel,
    laplace,
    gaussian_filter,
    uniform_filter,
    maximum_filter,
    minimum_filter,
)
from skimage.feature import local_binary_pattern
from skimage.filters import gabor, scharr_h, scharr_v
from typing import List, Tuple, Dict, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_CONFIG, IMG_HEIGHT, IMG_WIDTH


class FeatureEngineer:
    """
    Feature engineering for seismic salt segmentation.
    Extracts gradient, texture, statistical, and depth-encoded features.
    """

    def __init__(self, config: Dict = None):
        self.config = config or FEATURE_CONFIG
        self.patch_size = self.config["patch_size"]

    def compute_gradient_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute gradient-based features.

        Args:
            image: 2D image array (H, W)

        Returns:
            Dictionary of gradient features
        """
        features = {}

        # Sobel gradients
        sobel_x = sobel(image, axis=1)
        sobel_y = sobel(image, axis=0)
        features["sobel_x"] = sobel_x
        features["sobel_y"] = sobel_y
        features["sobel_magnitude"] = np.sqrt(sobel_x**2 + sobel_y**2)
        features["sobel_direction"] = np.arctan2(sobel_y, sobel_x)

        # Scharr gradients (more accurate than Sobel)
        features["scharr_h"] = scharr_h(image)
        features["scharr_v"] = scharr_v(image)

        # Laplacian (second derivative)
        features["laplacian"] = laplace(image)

        # Gradient at multiple scales
        for sigma in [1, 2, 4]:
            smoothed = gaussian_filter(image, sigma=sigma)
            features[f"gradient_mag_sigma{sigma}"] = np.sqrt(
                sobel(smoothed, axis=0) ** 2 + sobel(smoothed, axis=1) ** 2
            )

        return features

    def compute_texture_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute texture features using LBP and Gabor filters.

        Args:
            image: 2D image array (H, W)

        Returns:
            Dictionary of texture features
        """
        features = {}

        # Local Binary Pattern
        # Convert to uint8 for LBP
        img_uint8 = (image * 255).astype(np.uint8)

        for radius in [1, 2, 3]:
            n_points = 8 * radius
            lbp = local_binary_pattern(img_uint8, n_points, radius, method="uniform")
            features[f"lbp_r{radius}"] = lbp / lbp.max() if lbp.max() > 0 else lbp

        # Gabor filters at different orientations and frequencies
        orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        frequencies = [0.1, 0.2, 0.3]

        for freq in frequencies:
            for theta in orientations:
                try:
                    filt_real, filt_imag = gabor(image, frequency=freq, theta=theta)
                    angle_deg = int(np.degrees(theta))
                    features[f"gabor_f{freq:.1f}_t{angle_deg}_real"] = filt_real
                    features[f"gabor_f{freq:.1f}_t{angle_deg}_imag"] = filt_imag
                except Exception:
                    pass  # Skip if gabor fails

        return features

    def compute_statistical_features(
        self, image: np.ndarray, window_sizes: List[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute local statistical features.

        Args:
            image: 2D image array (H, W)
            window_sizes: List of window sizes for local statistics

        Returns:
            Dictionary of statistical features
        """
        if window_sizes is None:
            window_sizes = [3, 5, 7, 11]

        features = {}

        for win_size in window_sizes:
            # Local mean
            local_mean = uniform_filter(image, size=win_size)
            features[f"local_mean_w{win_size}"] = local_mean

            # Local standard deviation
            local_sq_mean = uniform_filter(image**2, size=win_size)
            local_var = local_sq_mean - local_mean**2
            local_var = np.maximum(local_var, 0)  # Ensure non-negative
            local_std = np.sqrt(local_var)
            features[f"local_std_w{win_size}"] = local_std

            # Local max - min (range)
            local_max = maximum_filter(image, size=win_size)
            local_min = minimum_filter(image, size=win_size)
            features[f"local_range_w{win_size}"] = local_max - local_min

            # Local skewness approximation
            local_skew = (image - local_mean) / (local_std + 1e-8)
            features[f"local_skew_w{win_size}"] = uniform_filter(
                local_skew**3, size=win_size
            )

        return features

    def compute_edge_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute edge detection features.

        Args:
            image: 2D image array (H, W)

        Returns:
            Dictionary of edge features
        """
        features = {}

        # Canny-like edge strength (using gradient magnitude)
        grad_mag = np.sqrt(sobel(image, axis=0) ** 2 + sobel(image, axis=1) ** 2)
        features["edge_strength"] = grad_mag

        # Multi-scale edges
        for sigma in [1, 2, 3]:
            smoothed = gaussian_filter(image, sigma=sigma)
            edge = np.sqrt(
                sobel(smoothed, axis=0) ** 2 + sobel(smoothed, axis=1) ** 2
            )
            features[f"edge_sigma{sigma}"] = edge

        # Difference of Gaussians (DoG) - blob detection
        for sigma in [1, 2]:
            dog = gaussian_filter(image, sigma=sigma) - gaussian_filter(
                image, sigma=sigma * 2
            )
            features[f"dog_sigma{sigma}"] = dog

        # Horizontal edge emphasis (important for salt boundaries)
        horizontal_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        features["horizontal_edge"] = ndimage.convolve(image, horizontal_kernel)

        # Vertical edge emphasis
        vertical_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        features["vertical_edge"] = ndimage.convolve(image, vertical_kernel)

        return features

    def compute_depth_encoding(
        self, image: np.ndarray, depth: float
    ) -> Dict[str, np.ndarray]:
        """
        Compute depth-encoded positional features.

        Args:
            image: 2D image array (H, W)
            depth: Depth value in meters

        Returns:
            Dictionary of depth-encoded features
        """
        h, w = image.shape
        features = {}

        # Normalized depth value broadcast to image size
        depth_normalized = depth / 1000.0  # Normalize to km
        features["depth_value"] = np.full((h, w), depth_normalized, dtype=np.float32)

        # Positional encoding within image (row position as proxy for local depth)
        row_positions = np.linspace(0, 1, h).reshape(-1, 1)
        row_encoding = np.tile(row_positions, (1, w))
        features["row_position"] = row_encoding.astype(np.float32)

        # Combined depth encoding
        combined_depth = depth_normalized + row_encoding * 0.1
        features["combined_depth_encoding"] = combined_depth.astype(np.float32)

        # Sinusoidal positional encoding (borrowed from transformers)
        for freq in [1, 2, 4]:
            sin_encoding = np.sin(2 * np.pi * freq * row_encoding)
            cos_encoding = np.cos(2 * np.pi * freq * row_encoding)
            features[f"pos_sin_f{freq}"] = sin_encoding.astype(np.float32)
            features[f"pos_cos_f{freq}"] = cos_encoding.astype(np.float32)

        # Depth-weighted image
        depth_weight = 1.0 / (1.0 + np.exp(-5 * (depth_normalized - 0.5)))
        features["depth_weighted_image"] = (image * depth_weight).astype(np.float32)

        return features

    def extract_all_features(
        self,
        image: np.ndarray,
        depth: float,
        spectral_features: np.ndarray = None,
    ) -> np.ndarray:
        """
        Extract all features for a single image.

        Args:
            image: 2D image array (H, W)
            depth: Depth value in meters
            spectral_features: Pre-computed spectral features (n_spectral, H, W)

        Returns:
            Feature array (n_features, H, W)
        """
        all_features = []
        feature_names = []

        # Original image
        all_features.append(image)
        feature_names.append("original")

        # Gradient features
        if self.config["use_gradient_features"]:
            grad_feats = self.compute_gradient_features(image)
            for name, feat in grad_feats.items():
                all_features.append(feat)
                feature_names.append(name)

        # Texture features
        if self.config["use_texture_features"]:
            tex_feats = self.compute_texture_features(image)
            for name, feat in tex_feats.items():
                all_features.append(feat)
                feature_names.append(name)

        # Statistical features
        if self.config["use_statistical_features"]:
            stat_feats = self.compute_statistical_features(image)
            for name, feat in stat_feats.items():
                all_features.append(feat)
                feature_names.append(name)

        # Edge features
        if self.config["use_edge_features"]:
            edge_feats = self.compute_edge_features(image)
            for name, feat in edge_feats.items():
                all_features.append(feat)
                feature_names.append(name)

        # Depth encoding
        if self.config["use_depth_encoding"]:
            depth_feats = self.compute_depth_encoding(image, depth)
            for name, feat in depth_feats.items():
                all_features.append(feat)
                feature_names.append(name)

        # Spectral features (if provided)
        if spectral_features is not None:
            for i in range(spectral_features.shape[0]):
                all_features.append(spectral_features[i])
                feature_names.append(f"spectral_{i}")

        self._feature_names = feature_names
        return np.array(all_features, dtype=np.float32)

    def extract_batch_features(
        self,
        images: np.ndarray,
        depths: np.ndarray,
        spectral_features: np.ndarray = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Extract features for a batch of images.

        Args:
            images: Array of shape (N, H, W)
            depths: Array of shape (N,)
            spectral_features: Pre-computed spectral features (N, n_spectral, H, W)
            verbose: Whether to print progress

        Returns:
            Feature array (N, n_features, H, W)
        """
        n_samples = images.shape[0]

        # Get feature count from first image
        spec_feat = spectral_features[0] if spectral_features is not None else None
        sample_features = self.extract_all_features(images[0], depths[0], spec_feat)
        n_features = sample_features.shape[0]

        all_features = np.zeros(
            (n_samples, n_features, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32
        )
        all_features[0] = sample_features

        for i in range(1, n_samples):
            if verbose and (i + 1) % 200 == 0:
                print(f"Feature extraction: {i + 1}/{n_samples}")

            spec_feat = spectral_features[i] if spectral_features is not None else None
            all_features[i] = self.extract_all_features(images[i], depths[i], spec_feat)

        if verbose:
            print(f"Total features shape: {all_features.shape}")
            print(f"Features per image: {n_features}")

        return all_features

    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        if hasattr(self, "_feature_names"):
            return self._feature_names
        return []

    def flatten_features(
        self, features: np.ndarray, masks: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flatten features for pixel-wise classification.

        Args:
            features: Array of shape (N, n_features, H, W)
            masks: Array of shape (N, H, W) - ground truth masks

        Returns:
            X: Flattened features (N * H * W, n_features)
            y: Flattened labels (N * H * W,) if masks provided, else None
        """
        n_samples, n_features, h, w = features.shape

        # Reshape to (N, n_features, H*W) then transpose
        X = features.reshape(n_samples, n_features, -1)
        X = X.transpose(0, 2, 1)  # (N, H*W, n_features)
        X = X.reshape(-1, n_features)  # (N*H*W, n_features)

        y = None
        if masks is not None:
            y = masks.reshape(-1)

        return X, y

    def sample_balanced_pixels(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 100000,
        salt_ratio: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample balanced pixels for training.

        Args:
            X: Feature array (N_pixels, n_features)
            y: Label array (N_pixels,)
            n_samples: Total number of samples to select
            salt_ratio: Ratio of salt pixels in the sample

        Returns:
            Sampled X and y arrays
        """
        salt_mask = y > 0.5
        non_salt_mask = ~salt_mask

        n_salt = int(n_samples * salt_ratio)
        n_non_salt = n_samples - n_salt

        # Get indices
        salt_indices = np.where(salt_mask)[0]
        non_salt_indices = np.where(non_salt_mask)[0]

        # Random sampling
        np.random.seed(42)
        if len(salt_indices) >= n_salt:
            salt_sample = np.random.choice(salt_indices, n_salt, replace=False)
        else:
            salt_sample = np.random.choice(salt_indices, n_salt, replace=True)

        if len(non_salt_indices) >= n_non_salt:
            non_salt_sample = np.random.choice(non_salt_indices, n_non_salt, replace=False)
        else:
            non_salt_sample = np.random.choice(non_salt_indices, n_non_salt, replace=True)

        # Combine
        all_indices = np.concatenate([salt_sample, non_salt_sample])
        np.random.shuffle(all_indices)

        return X[all_indices], y[all_indices]


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import DataLoader

    print("Loading test data...")
    loader = DataLoader()
    train_ids = loader.get_train_ids()
    test_image = loader.load_image(train_ids[0])
    test_depth = loader.get_depth(train_ids[0])
    test_mask = loader.load_mask(train_ids[0])

    print(f"Image shape: {test_image.shape}")
    print(f"Depth: {test_depth}")

    engineer = FeatureEngineer()

    print("\n--- Testing gradient features ---")
    grad_feats = engineer.compute_gradient_features(test_image)
    print(f"Number of gradient features: {len(grad_feats)}")

    print("\n--- Testing texture features ---")
    tex_feats = engineer.compute_texture_features(test_image)
    print(f"Number of texture features: {len(tex_feats)}")

    print("\n--- Testing statistical features ---")
    stat_feats = engineer.compute_statistical_features(test_image)
    print(f"Number of statistical features: {len(stat_feats)}")

    print("\n--- Testing edge features ---")
    edge_feats = engineer.compute_edge_features(test_image)
    print(f"Number of edge features: {len(edge_feats)}")

    print("\n--- Testing depth encoding ---")
    depth_feats = engineer.compute_depth_encoding(test_image, test_depth)
    print(f"Number of depth features: {len(depth_feats)}")

    print("\n--- Testing full feature extraction ---")
    all_features = engineer.extract_all_features(test_image, test_depth)
    print(f"Total features shape: {all_features.shape}")
    print(f"Feature names: {len(engineer.get_feature_names())}")
