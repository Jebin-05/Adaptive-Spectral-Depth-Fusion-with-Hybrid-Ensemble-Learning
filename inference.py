"""
Inference Pipeline for Depth-Adaptive Spectral Salt Segmentation
Run predictions on new seismic images
"""

import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODELS_DIR, OUTPUTS_DIR, TEST_IMAGES_DIR
from src.data_loader import DataLoader
from src.spectral_decomposition import SpectralDecomposer
from src.feature_engineering import FeatureEngineer
from src.ensemble_model import HybridEnsemble
from src.confidence_module import DepthAdaptiveConfidence
from src.visualization import Visualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained salt segmentation model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model (default: latest)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of test samples to process (None for all)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save prediction masks",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose output",
    )
    return parser.parse_args()


class SaltSegmentationPipeline:
    """
    Complete pipeline for salt body segmentation.
    Combines spectral decomposition, feature engineering, and ensemble prediction.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the pipeline.

        Args:
            model_path: Path to trained model. If None, loads latest.
        """
        self.model_path = model_path or os.path.join(MODELS_DIR, "ensemble_latest.joblib")

        # Initialize components
        self.loader = DataLoader()
        self.decomposer = SpectralDecomposer()
        self.engineer = FeatureEngineer()
        self.ensemble = HybridEnsemble()
        self.confidence = DepthAdaptiveConfidence()

        # Load model
        if os.path.exists(self.model_path):
            self.ensemble.load(self.model_path)
            print(f"Model loaded from: {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}")

    def predict_single(
        self, image: np.ndarray, depth: float, threshold: float = 0.5
    ) -> dict:
        """
        Predict salt mask for a single image.

        Args:
            image: Seismic image (H, W)
            depth: Depth value in meters
            threshold: Classification threshold

        Returns:
            Dictionary with mask, probabilities, confidence, and bounds
        """
        start = time.time()

        # Spectral decomposition
        spectral_features = self.decomposer.compute_multi_scale_features(image)

        # Feature engineering
        all_features = self.engineer.extract_all_features(
            image, depth, spectral_features
        )

        # Prediction
        mask, proba_map = self.ensemble.predict_image(
            all_features, image.shape, threshold
        )

        # Confidence
        conf_map, components = self.confidence.compute_confidence_score(
            proba_map, depth
        )
        lower, upper = self.confidence.compute_uncertainty_bounds(proba_map, conf_map)

        inference_time = time.time() - start

        return {
            "mask": mask,
            "probabilities": proba_map,
            "confidence": conf_map,
            "lower_bound": lower,
            "upper_bound": upper,
            "depth_weight": components["depth_weight"],
            "inference_time_ms": inference_time * 1000,
        }

    def predict_batch(
        self,
        images: np.ndarray,
        depths: np.ndarray,
        threshold: float = 0.5,
        verbose: bool = True,
    ) -> dict:
        """
        Predict salt masks for a batch of images.

        Args:
            images: Seismic images (N, H, W)
            depths: Depth values (N,)
            threshold: Classification threshold
            verbose: Print progress

        Returns:
            Dictionary with masks, probabilities, confidences, and timing info
        """
        n_samples = len(images)

        if verbose:
            print(f"Processing {n_samples} images...")

        total_start = time.time()

        # Spectral decomposition
        if verbose:
            print("  Spectral decomposition...")
        spectral_start = time.time()
        spectral_features = self.decomposer.decompose_batch(images, verbose=False)
        spectral_time = time.time() - spectral_start

        # Feature engineering
        if verbose:
            print("  Feature extraction...")
        feature_start = time.time()
        all_features = self.engineer.extract_batch_features(
            images, depths, spectral_features, verbose=False
        )
        feature_time = time.time() - feature_start

        # Prediction
        if verbose:
            print("  Ensemble prediction...")
        pred_start = time.time()
        masks, proba_maps = self.ensemble.predict_batch(
            all_features, threshold, verbose=False
        )
        pred_time = time.time() - pred_start

        # Confidence
        if verbose:
            print("  Confidence computation...")
        conf_start = time.time()
        conf_maps, lowers, uppers = self.confidence.compute_batch_confidence(
            proba_maps, depths, verbose=False
        )
        conf_time = time.time() - conf_start

        total_time = time.time() - total_start

        return {
            "masks": masks,
            "probabilities": proba_maps,
            "confidences": conf_maps,
            "lower_bounds": lowers,
            "upper_bounds": uppers,
            "timing": {
                "total_ms": total_time * 1000,
                "per_image_ms": total_time / n_samples * 1000,
                "spectral_ms": spectral_time * 1000,
                "feature_ms": feature_time * 1000,
                "prediction_ms": pred_time * 1000,
                "confidence_ms": conf_time * 1000,
            },
        }


def main():
    """Main inference pipeline."""
    args = parse_args()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("SALT SEGMENTATION INFERENCE")
    print("=" * 60)

    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = SaltSegmentationPipeline(args.model_path)

    # Load test data
    print("\nLoading test data...")
    loader = DataLoader()

    test_ids = loader.get_test_ids()
    if args.n_samples is not None:
        test_ids = test_ids[: args.n_samples]

    n_samples = len(test_ids)
    print(f"Processing {n_samples} test images")

    # Load images and depths
    images = np.zeros((n_samples, 101, 101), dtype=np.float32)
    depths = np.zeros(n_samples, dtype=np.float32)

    for i, img_id in enumerate(test_ids):
        images[i] = loader.load_image(img_id, is_train=False)
        depths[i] = loader.get_depth(img_id)

    print(f"Images loaded: {images.shape}")
    print(f"Depth range: {depths.min():.0f} - {depths.max():.0f} m")

    # Run inference
    print("\nRunning inference...")
    results = pipeline.predict_batch(
        images, depths, threshold=args.threshold, verbose=args.verbose
    )

    # Print timing
    print("\n--- Timing Statistics ---")
    timing = results["timing"]
    print(f"Total time:        {timing['total_ms']:.1f} ms")
    print(f"Per image:         {timing['per_image_ms']:.1f} ms")
    print(f"  Spectral:        {timing['spectral_ms']:.1f} ms")
    print(f"  Features:        {timing['feature_ms']:.1f} ms")
    print(f"  Prediction:      {timing['prediction_ms']:.1f} ms")
    print(f"  Confidence:      {timing['confidence_ms']:.1f} ms")

    # Save predictions
    if args.save_predictions:
        print("\nSaving predictions...")
        pred_path = os.path.join(OUTPUTS_DIR, f"predictions_{timestamp}.npz")
        np.savez_compressed(
            pred_path,
            ids=test_ids,
            masks=results["masks"],
            probabilities=results["probabilities"],
            confidences=results["confidences"],
            depths=depths,
        )
        print(f"Saved to: {pred_path}")

    # Create visualizations
    if args.visualize:
        print("\nCreating visualizations...")
        viz = Visualizer(OUTPUTS_DIR)

        # Sample predictions
        n_viz = min(9, n_samples)
        viz.create_summary_grid(
            images[:n_viz],
            np.zeros_like(results["masks"][:n_viz]),  # No ground truth for test
            results["masks"][:n_viz],
            results["confidences"][:n_viz],
            title="Test Predictions",
            save_path=os.path.join(OUTPUTS_DIR, f"test_predictions_{timestamp}.png"),
        )

        # Uncertainty maps for first few samples
        for i in range(min(3, n_samples)):
            viz.plot_uncertainty_map(
                images[i],
                results["probabilities"][i],
                results["lower_bounds"][i],
                results["upper_bounds"][i],
                title=f"Test Sample {i+1} - Depth: {depths[i]:.0f}m",
                save_path=os.path.join(
                    OUTPUTS_DIR, f"test_uncertainty_{i}_{timestamp}.png"
                ),
            )

    # Summary statistics
    print("\n--- Prediction Statistics ---")
    salt_coverage = np.array([m.mean() for m in results["masks"]])
    print(f"Mean salt coverage:  {salt_coverage.mean()*100:.2f}%")
    print(f"Images with salt:    {(salt_coverage > 0).sum()}/{n_samples}")
    print(f"Mean confidence:     {results['confidences'].mean():.3f}")

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = main()
