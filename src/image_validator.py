"""
Seismic Image Validator Module
Validates that input images are legitimate seismic survey images
and not random photographs or other non-seismic images.
"""

import numpy as np
from scipy import ndimage
from scipy.stats import kurtosis, skew
from typing import Tuple, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_HEIGHT, IMG_WIDTH


class SeismicImageValidator:
    """
    Validates input images to ensure they are legitimate seismic survey images.
    Uses a scoring system to rate how "seismic-like" an image appears.

    The validator computes a confidence score from 0-1:
    - 0.7+ : High confidence seismic image
    - 0.4-0.7: Medium confidence, may be seismic
    - 0.2-0.4: Low confidence, likely not seismic
    - <0.2: Very unlikely to be seismic
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize the validator.

        Args:
            strict_mode: If True, require higher confidence for validation
        """
        self.strict_mode = strict_mode
        self.min_confidence = 0.35 if strict_mode else 0.25

    def validate(self, image: np.ndarray) -> Tuple[bool, Dict, str]:
        """
        Validate if the input image is a legitimate seismic survey image.

        Args:
            image: 2D grayscale image array (H, W), values 0-1

        Returns:
            Tuple of (is_valid, metrics_dict, message)
        """
        if image is None or image.size == 0:
            return False, {}, "Empty or null image provided"

        # Ensure correct shape
        if len(image.shape) != 2:
            return False, {}, "Image must be 2D grayscale"

        # Normalize if needed
        if image.max() > 1.0:
            image = image / 255.0

        metrics = {}

        # Compute various seismic-like features
        # 1. Basic statistics
        metrics["mean"] = float(np.mean(image))
        metrics["std"] = float(np.std(image))

        # Check for completely uniform/blank images
        if metrics["std"] < 0.001:
            return False, metrics, "Image is uniform/blank - no variation detected"

        # 2. Horizontal correlation (seismic has laterally continuous reflectors)
        h_corr = self._compute_horizontal_correlation(image)
        metrics["horizontal_correlation"] = h_corr

        # 3. Trace correlation (adjacent seismic traces are correlated)
        t_corr = self._compute_trace_correlation(image)
        metrics["trace_correlation"] = t_corr

        # 4. Oscillation pattern (seismic has wave-like patterns)
        osc = self._compute_oscillation_score(image)
        metrics["oscillation_score"] = osc

        # 5. Block uniformity (photos have large uniform blocks, seismic doesn't)
        block_uni = self._compute_block_uniformity(image)
        metrics["block_uniformity"] = block_uni

        # 6. Edge pattern analysis
        edge_score = self._analyze_edge_patterns(image)
        metrics["edge_pattern_score"] = edge_score

        # Compute overall confidence score
        confidence = self._compute_confidence(metrics)
        metrics["validation_confidence"] = confidence

        # Determine result
        if confidence >= self.min_confidence:
            if confidence >= 0.7:
                return True, metrics, "High confidence seismic image"
            elif confidence >= 0.5:
                return True, metrics, "Medium confidence - likely seismic image"
            else:
                return True, metrics, "Low confidence - may be seismic (proceed with caution)"
        else:
            # Rejection - explain why
            reasons = []
            if h_corr < 0.15:
                reasons.append("lacks horizontal layering patterns")
            if t_corr < 0.1:
                reasons.append("lacks trace-to-trace correlation")
            if osc < 0.03:
                reasons.append("no wave-like oscillation patterns")
            if block_uni > 0.5:
                reasons.append("contains large uniform blocks (photo-like)")

            reason_str = ", ".join(reasons) if reasons else "does not match seismic characteristics"
            return False, metrics, f"Image rejected: {reason_str} (confidence: {confidence:.0%})"

    def _compute_horizontal_correlation(self, image: np.ndarray) -> float:
        """Compute horizontal correlation (seismic has laterally continuous reflectors)."""
        correlations = []
        for row in range(image.shape[0]):
            row_data = image[row, :]
            if len(row_data) > 1 and np.std(row_data) > 0.001:
                corr = np.corrcoef(row_data[:-1], row_data[1:])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        return np.mean(correlations) if correlations else 0.0

    def _compute_trace_correlation(self, image: np.ndarray) -> float:
        """Compute trace-to-trace correlation (adjacent columns should be similar)."""
        correlations = []
        for col in range(image.shape[1] - 1):
            t1 = image[:, col]
            t2 = image[:, col + 1]
            if np.std(t1) > 0.001 and np.std(t2) > 0.001:
                corr = np.corrcoef(t1, t2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        return np.mean(correlations) if correlations else 0.0

    def _compute_oscillation_score(self, image: np.ndarray) -> float:
        """Compute oscillation score (seismic oscillates around mean)."""
        scores = []
        for col in range(image.shape[1]):
            trace = image[:, col]
            mean_val = np.mean(trace)
            centered = trace - mean_val
            # Count zero crossings
            crossings = np.sum(np.abs(np.diff(np.sign(centered))) > 0)
            scores.append(crossings / len(trace))
        return np.mean(scores) if scores else 0.0

    def _compute_block_uniformity(self, image: np.ndarray) -> float:
        """Detect large uniform blocks (common in photos, not in seismic)."""
        h, w = image.shape
        block_size = max(h // 5, 10)

        uniform_count = 0
        total_count = 0

        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = image[i:i+block_size, j:j+block_size]
                if np.std(block) < 0.03:  # Very uniform block
                    uniform_count += 1
                total_count += 1

        return uniform_count / total_count if total_count > 0 else 0.0

    def _analyze_edge_patterns(self, image: np.ndarray) -> float:
        """Analyze edge patterns - seismic has continuous horizontal edges."""
        # Compute gradients
        grad_y = ndimage.sobel(image, axis=0)
        grad_x = ndimage.sobel(image, axis=1)

        # Edge magnitude
        edge_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Check for presence of edges
        has_edges = np.percentile(edge_mag, 90) > 0.05

        # Compute horizontal edge continuity
        h_continuity = 0
        if has_edges:
            # Strong horizontal edges should be continuous
            for row in range(1, image.shape[0] - 1):
                row_edges = edge_mag[row, :]
                if np.max(row_edges) > 0.1:
                    # Check how continuous this edge is
                    above_threshold = row_edges > np.max(row_edges) * 0.3
                    h_continuity += np.mean(above_threshold)

            h_continuity /= image.shape[0]

        return h_continuity

    def _compute_confidence(self, metrics: Dict) -> float:
        """Compute overall seismic confidence score."""
        scores = []

        # Horizontal correlation (weight: 0.25)
        h_corr = metrics.get("horizontal_correlation", 0)
        h_score = min(max(h_corr, 0) / 0.6, 1.0)
        scores.append(("h_corr", h_score, 0.25))

        # Trace correlation (weight: 0.25)
        t_corr = metrics.get("trace_correlation", 0)
        t_score = min(max(t_corr, 0) / 0.5, 1.0)
        scores.append(("t_corr", t_score, 0.25))

        # Oscillation score (weight: 0.2)
        osc = metrics.get("oscillation_score", 0)
        osc_score = min(osc / 0.2, 1.0)
        scores.append(("osc", osc_score, 0.2))

        # Block uniformity penalty (weight: 0.15)
        block_uni = metrics.get("block_uniformity", 0)
        block_score = max(0, 1.0 - block_uni * 2)  # Penalize uniform blocks
        scores.append(("block", block_score, 0.15))

        # Edge pattern score (weight: 0.15)
        edge_score = metrics.get("edge_pattern_score", 0)
        scores.append(("edge", min(edge_score / 0.3, 1.0), 0.15))

        # Weighted average
        total_weight = sum(w for _, _, w in scores)
        confidence = sum(s * w for _, s, w in scores) / total_weight

        return confidence

    def get_validation_summary(self, image: np.ndarray) -> str:
        """Get a human-readable validation summary."""
        is_valid, metrics, reason = self.validate(image)

        summary = []
        summary.append("=" * 50)
        summary.append("SEISMIC IMAGE VALIDATION REPORT")
        summary.append("=" * 50)
        summary.append(f"Result: {'VALID' if is_valid else 'INVALID'}")
        summary.append(f"Message: {reason}")
        summary.append("")
        summary.append("Key Metrics:")

        for key in ["horizontal_correlation", "trace_correlation",
                    "oscillation_score", "block_uniformity", "validation_confidence"]:
            if key in metrics:
                summary.append(f"  {key}: {metrics[key]:.3f}")

        summary.append("=" * 50)

        return "\n".join(summary)


def validate_seismic_image(image: np.ndarray, strict: bool = True) -> Tuple[bool, str]:
    """
    Convenience function to validate a seismic image.

    Args:
        image: 2D grayscale image array
        strict: Whether to use strict validation

    Returns:
        Tuple of (is_valid, message)
    """
    validator = SeismicImageValidator(strict_mode=strict)
    is_valid, metrics, reason = validator.validate(image)

    if is_valid:
        confidence = metrics.get("validation_confidence", 0)
        return True, f"Valid seismic image (confidence: {confidence:.0%})"
    else:
        return False, reason


if __name__ == "__main__":
    # Test the validator
    print("Testing SeismicImageValidator...")

    validator = SeismicImageValidator(strict_mode=True)

    # Test with synthetic seismic-like image
    print("\n--- Test 1: Synthetic seismic-like pattern ---")
    h, w = 101, 101
    y = np.linspace(0, 10*np.pi, h).reshape(-1, 1)
    x = np.linspace(0, 2*np.pi, w).reshape(1, -1)
    seismic_like = 0.5 + 0.3 * np.sin(y) * np.cos(x * 0.5) + 0.05 * np.random.randn(h, w)
    seismic_like = np.clip(seismic_like, 0, 1)

    is_valid, metrics, reason = validator.validate(seismic_like)
    print(f"Valid: {is_valid}")
    print(f"Message: {reason}")
    print(f"Confidence: {metrics.get('validation_confidence', 0):.3f}")

    # Test with random noise
    print("\n--- Test 2: Random noise ---")
    noise = np.random.rand(101, 101)

    is_valid, metrics, reason = validator.validate(noise)
    print(f"Valid: {is_valid}")
    print(f"Message: {reason}")

    # Test with uniform image
    print("\n--- Test 3: Uniform image ---")
    uniform = np.ones((101, 101)) * 0.5

    is_valid, metrics, reason = validator.validate(uniform)
    print(f"Valid: {is_valid}")
    print(f"Message: {reason}")

    print("\n--- Validator testing complete ---")
