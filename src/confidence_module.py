"""
Depth-Adaptive Confidence Module
Uncertainty quantification with depth-dependent confidence scoring
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.ndimage import gaussian_filter, uniform_filter
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIDENCE_CONFIG


class DepthAdaptiveConfidence:
    """
    Depth-adaptive confidence scoring for salt segmentation.
    Implements uncertainty quantification based on depth and prediction consistency.
    """

    def __init__(self, config: Dict = None):
        self.config = config or CONFIDENCE_CONFIG
        self.shallow_threshold = self.config["shallow_depth_threshold"]
        self.medium_threshold = self.config["medium_depth_threshold"]
        self.deep_threshold = self.config["deep_depth_threshold"]
        self.shallow_boost = self.config["shallow_confidence_boost"]
        self.deep_penalty = self.config["deep_confidence_penalty"]
        self.uncertainty_scale = self.config["uncertainty_scale_factor"]

    def compute_depth_weight(self, depth: float) -> float:
        """
        Compute confidence weight based on depth.
        Shallow regions have higher confidence, deep regions lower.

        Args:
            depth: Depth value in meters

        Returns:
            Confidence weight multiplier
        """
        if depth < self.shallow_threshold:
            # Shallow: high confidence
            return self.shallow_boost
        elif depth < self.medium_threshold:
            # Medium: gradual decrease
            ratio = (depth - self.shallow_threshold) / (
                self.medium_threshold - self.shallow_threshold
            )
            return self.shallow_boost - ratio * (self.shallow_boost - 1.0)
        elif depth < self.deep_threshold:
            # Deep: continued decrease
            ratio = (depth - self.medium_threshold) / (
                self.deep_threshold - self.medium_threshold
            )
            return 1.0 - ratio * (1.0 - self.deep_penalty)
        else:
            # Very deep: low confidence
            return self.deep_penalty

    def compute_prediction_uncertainty(
        self, proba_map: np.ndarray
    ) -> np.ndarray:
        """
        Compute uncertainty from prediction probabilities.
        Uncertainty is highest when probability is near 0.5.

        Args:
            proba_map: Probability map (H, W)

        Returns:
            Uncertainty map (H, W) - values in [0, 1]
        """
        # Shannon entropy-based uncertainty
        p = np.clip(proba_map, 1e-7, 1 - 1e-7)
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        # Normalize to [0, 1]
        uncertainty = entropy  # Already in [0, 1] for binary
        return uncertainty

    def compute_spatial_consistency(
        self, proba_map: np.ndarray, window_size: int = 5
    ) -> np.ndarray:
        """
        Compute spatial consistency of predictions.
        Inconsistent regions have higher uncertainty.

        Args:
            proba_map: Probability map (H, W)
            window_size: Window size for local consistency

        Returns:
            Consistency score (H, W) - higher means more consistent
        """
        # Local mean and std
        local_mean = uniform_filter(proba_map, size=window_size)
        local_sq_mean = uniform_filter(proba_map**2, size=window_size)
        local_var = np.maximum(local_sq_mean - local_mean**2, 0)
        local_std = np.sqrt(local_var)

        # Consistency is inverse of local variation
        consistency = 1.0 - np.clip(local_std * 4, 0, 1)  # Scale factor 4
        return consistency

    def compute_boundary_uncertainty(
        self, proba_map: np.ndarray
    ) -> np.ndarray:
        """
        Compute uncertainty at boundaries.
        Boundaries typically have higher uncertainty.

        Args:
            proba_map: Probability map (H, W)

        Returns:
            Boundary uncertainty map (H, W)
        """
        # Compute gradient magnitude of probability map
        grad_y = np.gradient(proba_map, axis=0)
        grad_x = np.gradient(proba_map, axis=1)
        grad_mag = np.sqrt(grad_y**2 + grad_x**2)

        # Normalize
        boundary_uncertainty = np.clip(grad_mag * 2, 0, 1)
        return boundary_uncertainty

    def compute_model_agreement(
        self, model_predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute agreement between different models.
        Higher agreement = higher confidence.

        Args:
            model_predictions: Dict with model name -> probability map

        Returns:
            Agreement score (H, W) - higher means more agreement
        """
        # Stack all predictions
        preds = np.stack(list(model_predictions.values()), axis=0)

        # Compute variance across models
        variance = np.var(preds, axis=0)

        # Agreement is inverse of variance
        agreement = 1.0 - np.clip(variance * 4, 0, 1)
        return agreement

    def compute_confidence_score(
        self,
        proba_map: np.ndarray,
        depth: float,
        model_predictions: Dict[str, np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute comprehensive confidence score.

        Args:
            proba_map: Probability map (H, W)
            depth: Depth value in meters
            model_predictions: Optional dict of individual model predictions

        Returns:
            confidence_map: Final confidence score (H, W) in [0, 1]
            components: Dict with individual confidence components
        """
        components = {}

        # 1. Depth-based confidence weight
        depth_weight = self.compute_depth_weight(depth)
        components["depth_weight"] = depth_weight

        # 2. Prediction uncertainty (entropy-based)
        pred_uncertainty = self.compute_prediction_uncertainty(proba_map)
        pred_confidence = 1.0 - pred_uncertainty
        components["prediction_confidence"] = pred_confidence

        # 3. Spatial consistency
        spatial_consistency = self.compute_spatial_consistency(proba_map)
        components["spatial_consistency"] = spatial_consistency

        # 4. Boundary uncertainty
        boundary_uncertainty = self.compute_boundary_uncertainty(proba_map)
        boundary_confidence = 1.0 - boundary_uncertainty
        components["boundary_confidence"] = boundary_confidence

        # 5. Model agreement (if multiple models provided)
        if model_predictions is not None and len(model_predictions) > 1:
            model_agreement = self.compute_model_agreement(model_predictions)
            components["model_agreement"] = model_agreement
        else:
            model_agreement = np.ones_like(proba_map)
            components["model_agreement"] = model_agreement

        # Combine components with weights
        confidence = (
            0.25 * pred_confidence
            + 0.25 * spatial_consistency
            + 0.20 * boundary_confidence
            + 0.30 * model_agreement
        )

        # Apply depth weighting
        confidence = confidence * depth_weight

        # Clip to valid range
        confidence = np.clip(confidence, 0, 1)

        return confidence, components

    def compute_uncertainty_bounds(
        self,
        proba_map: np.ndarray,
        confidence_map: np.ndarray,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute uncertainty bounds for predictions.

        Args:
            proba_map: Probability map (H, W)
            confidence_map: Confidence map (H, W)
            alpha: Uncertainty level (0.1 = 90% confidence interval)

        Returns:
            lower_bound: Lower probability bound (H, W)
            upper_bound: Upper probability bound (H, W)
        """
        # Uncertainty width inversely proportional to confidence
        uncertainty_width = (1 - confidence_map) * self.uncertainty_scale

        lower_bound = np.clip(proba_map - uncertainty_width, 0, 1)
        upper_bound = np.clip(proba_map + uncertainty_width, 0, 1)

        return lower_bound, upper_bound

    def get_confidence_mask(
        self,
        confidence_map: np.ndarray,
        threshold: float = 0.7,
    ) -> np.ndarray:
        """
        Get binary mask of high-confidence regions.

        Args:
            confidence_map: Confidence map (H, W)
            threshold: Confidence threshold

        Returns:
            Binary mask (H, W) - True for high-confidence regions
        """
        return confidence_map >= threshold

    def compute_batch_confidence(
        self,
        proba_maps: np.ndarray,
        depths: np.ndarray,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute confidence for a batch of predictions.

        Args:
            proba_maps: Probability maps (N, H, W)
            depths: Depth values (N,)
            verbose: Whether to print progress

        Returns:
            confidence_maps: Confidence maps (N, H, W)
            lower_bounds: Lower probability bounds (N, H, W)
            upper_bounds: Upper probability bounds (N, H, W)
        """
        n_samples = proba_maps.shape[0]
        h, w = proba_maps.shape[1], proba_maps.shape[2]

        confidence_maps = np.zeros_like(proba_maps)
        lower_bounds = np.zeros_like(proba_maps)
        upper_bounds = np.zeros_like(proba_maps)

        for i in range(n_samples):
            if verbose and (i + 1) % 200 == 0:
                print(f"Computing confidence: {i + 1}/{n_samples}")

            conf, _ = self.compute_confidence_score(proba_maps[i], depths[i])
            confidence_maps[i] = conf

            lower, upper = self.compute_uncertainty_bounds(proba_maps[i], conf)
            lower_bounds[i] = lower
            upper_bounds[i] = upper

        return confidence_maps, lower_bounds, upper_bounds

    def analyze_depth_confidence(
        self, depths: np.ndarray, confidence_maps: np.ndarray
    ) -> Dict:
        """
        Analyze confidence statistics across depth ranges.

        Args:
            depths: Depth values (N,)
            confidence_maps: Confidence maps (N, H, W)

        Returns:
            Statistics dictionary
        """
        mean_confidences = np.array([cm.mean() for cm in confidence_maps])

        stats = {
            "overall": {
                "mean": mean_confidences.mean(),
                "std": mean_confidences.std(),
            }
        }

        # Analyze by depth range
        depth_ranges = [
            ("shallow", 0, 200),
            ("medium", 200, 500),
            ("deep", 500, 800),
            ("very_deep", 800, 1000),
        ]

        for name, d_min, d_max in depth_ranges:
            mask = (depths >= d_min) & (depths < d_max)
            if mask.sum() > 0:
                stats[name] = {
                    "n_samples": mask.sum(),
                    "mean_confidence": mean_confidences[mask].mean(),
                    "std_confidence": mean_confidences[mask].std(),
                }

        return stats


if __name__ == "__main__":
    # Test confidence module
    print("Testing DepthAdaptiveConfidence...")

    # Create synthetic data
    np.random.seed(42)
    h, w = 101, 101

    # Create sample probability map
    proba_map = np.random.rand(h, w).astype(np.float32)
    # Add some structure
    proba_map[:50, :] = np.clip(proba_map[:50, :] + 0.3, 0, 1)

    conf_module = DepthAdaptiveConfidence()

    print("\n--- Testing depth weights ---")
    for depth in [100, 300, 600, 900]:
        weight = conf_module.compute_depth_weight(depth)
        print(f"Depth {depth}m: weight = {weight:.3f}")

    print("\n--- Testing uncertainty computation ---")
    uncertainty = conf_module.compute_prediction_uncertainty(proba_map)
    print(f"Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")

    print("\n--- Testing spatial consistency ---")
    consistency = conf_module.compute_spatial_consistency(proba_map)
    print(f"Consistency range: [{consistency.min():.3f}, {consistency.max():.3f}]")

    print("\n--- Testing full confidence computation ---")
    confidence, components = conf_module.compute_confidence_score(proba_map, depth=300)
    print(f"Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    print(f"Mean confidence: {confidence.mean():.3f}")

    print("\n--- Testing uncertainty bounds ---")
    lower, upper = conf_module.compute_uncertainty_bounds(proba_map, confidence)
    print(f"Lower bound range: [{lower.min():.3f}, {lower.max():.3f}]")
    print(f"Upper bound range: [{upper.min():.3f}, {upper.max():.3f}]")

    print("\n--- Testing batch processing ---")
    n_samples = 10
    proba_maps = np.random.rand(n_samples, h, w).astype(np.float32)
    depths = np.random.randint(50, 900, n_samples).astype(np.float32)

    conf_maps, lowers, uppers = conf_module.compute_batch_confidence(
        proba_maps, depths, verbose=False
    )
    print(f"Confidence maps shape: {conf_maps.shape}")

    print("\n--- Testing depth-confidence analysis ---")
    stats = conf_module.analyze_depth_confidence(depths, conf_maps)
    for key, value in stats.items():
        print(f"{key}: {value}")
