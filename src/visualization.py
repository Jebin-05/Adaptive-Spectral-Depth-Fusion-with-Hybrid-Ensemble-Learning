"""
Visualization Module
Plotting utilities for salt segmentation analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUTS_DIR


class Visualizer:
    """
    Visualization utilities for salt segmentation project.
    """

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or OUTPUTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8-whitegrid")

    def plot_sample(
        self,
        image: np.ndarray,
        mask: np.ndarray = None,
        pred: np.ndarray = None,
        confidence: np.ndarray = None,
        title: str = None,
        save_path: str = None,
    ) -> None:
        """
        Plot a single sample with image, mask, prediction, and confidence.

        Args:
            image: Seismic image (H, W)
            mask: Ground truth mask (H, W)
            pred: Predicted mask (H, W)
            confidence: Confidence map (H, W)
            title: Plot title
            save_path: Path to save figure
        """
        n_plots = 1 + sum([mask is not None, pred is not None, confidence is not None])
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))

        if n_plots == 1:
            axes = [axes]

        idx = 0

        # Original image
        axes[idx].imshow(image, cmap="seismic")
        axes[idx].set_title("Seismic Image")
        axes[idx].axis("off")
        idx += 1

        # Ground truth mask
        if mask is not None:
            axes[idx].imshow(mask, cmap="gray")
            axes[idx].set_title("Ground Truth")
            axes[idx].axis("off")
            idx += 1

        # Prediction
        if pred is not None:
            axes[idx].imshow(pred, cmap="gray")
            axes[idx].set_title("Prediction")
            axes[idx].axis("off")
            idx += 1

        # Confidence
        if confidence is not None:
            im = axes[idx].imshow(confidence, cmap="viridis", vmin=0, vmax=1)
            axes[idx].set_title("Confidence")
            axes[idx].axis("off")
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.close()

    def plot_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray = None,
        pred: np.ndarray = None,
        alpha: float = 0.5,
        title: str = None,
        save_path: str = None,
    ) -> None:
        """
        Plot image with mask/prediction overlay.

        Args:
            image: Seismic image (H, W)
            mask: Ground truth mask (H, W)
            pred: Predicted mask (H, W)
            alpha: Overlay transparency
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3 if pred is not None else 2, figsize=(12, 4))

        # Original with ground truth overlay
        axes[0].imshow(image, cmap="gray")
        if mask is not None:
            mask_overlay = np.ma.masked_where(mask < 0.5, mask)
            axes[0].imshow(mask_overlay, cmap="Reds", alpha=alpha, vmin=0, vmax=1)
        axes[0].set_title("Ground Truth Overlay")
        axes[0].axis("off")

        if pred is not None:
            # Original with prediction overlay
            axes[1].imshow(image, cmap="gray")
            pred_overlay = np.ma.masked_where(pred < 0.5, pred)
            axes[1].imshow(pred_overlay, cmap="Blues", alpha=alpha, vmin=0, vmax=1)
            axes[1].set_title("Prediction Overlay")
            axes[1].axis("off")

            # Comparison: TP (green), FP (red), FN (yellow)
            axes[2].imshow(image, cmap="gray")
            comparison = self._create_comparison_mask(mask, pred)
            axes[2].imshow(comparison, alpha=alpha)
            axes[2].set_title("Comparison (G:TP, R:FP, Y:FN)")
            axes[2].axis("off")
        else:
            axes[1].imshow(image, cmap="seismic")
            axes[1].set_title("Original")
            axes[1].axis("off")

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.close()

    def _create_comparison_mask(
        self, mask: np.ndarray, pred: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """Create RGB comparison mask showing TP, FP, FN."""
        mask_binary = mask > threshold
        pred_binary = pred > threshold

        tp = np.logical_and(mask_binary, pred_binary)
        fp = np.logical_and(~mask_binary, pred_binary)
        fn = np.logical_and(mask_binary, ~pred_binary)

        rgb = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgb[tp] = [0, 1, 0, 1]  # Green: True Positive
        rgb[fp] = [1, 0, 0, 1]  # Red: False Positive
        rgb[fn] = [1, 1, 0, 1]  # Yellow: False Negative

        return rgb

    def plot_spectral_features(
        self,
        features: Dict[str, np.ndarray],
        title: str = None,
        save_path: str = None,
    ) -> None:
        """
        Plot spectral decomposition features.

        Args:
            features: Dictionary of spectral features
            title: Plot title
            save_path: Path to save figure
        """
        n_features = len(features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, (name, feature) in enumerate(features.items()):
            if idx >= len(axes):
                break
            im = axes[idx].imshow(feature, cmap="seismic")
            axes[idx].set_title(name, fontsize=10)
            axes[idx].axis("off")
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        # Hide unused axes
        for idx in range(len(features), len(axes)):
            axes[idx].axis("off")

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.close()

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        top_k: int = 20,
        title: str = "Feature Importance",
        save_path: str = None,
    ) -> None:
        """
        Plot feature importance bar chart.

        Args:
            feature_names: List of feature names
            importances: Importance scores
            top_k: Number of top features to show
            title: Plot title
            save_path: Path to save figure
        """
        # Get top features
        indices = np.argsort(importances)[::-1][:top_k]
        top_names = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))

        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_importances, color="steelblue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.close()

    def plot_depth_analysis(
        self,
        depths: np.ndarray,
        metrics: Dict[str, Dict],
        title: str = "Depth-Stratified Analysis",
        save_path: str = None,
    ) -> None:
        """
        Plot metrics stratified by depth.

        Args:
            depths: Depth values
            metrics: Depth-stratified metrics dict
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Depth histogram
        axes[0].hist(depths, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Depth (m)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Depth Distribution")

        # IoU by depth
        depth_names = list(metrics.keys())
        ious = [metrics[k]["iou_mean"] for k in depth_names if "iou_mean" in metrics[k]]
        if ious:
            axes[1].bar(range(len(ious)), ious, color="forestgreen", alpha=0.7)
            axes[1].set_xticks(range(len(ious)))
            axes[1].set_xticklabels(depth_names, rotation=45, ha="right")
            axes[1].set_ylabel("IoU")
            axes[1].set_title("IoU by Depth Range")
            axes[1].set_ylim(0, 1)

        # Accuracy by depth
        accs = [metrics[k]["accuracy"] for k in depth_names if "accuracy" in metrics[k]]
        if accs:
            axes[2].bar(range(len(accs)), accs, color="coral", alpha=0.7)
            axes[2].set_xticks(range(len(accs)))
            axes[2].set_xticklabels(depth_names, rotation=45, ha="right")
            axes[2].set_ylabel("Accuracy")
            axes[2].set_title("Accuracy by Depth Range")
            axes[2].set_ylim(0, 1)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.close()

    def plot_confidence_analysis(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
        title: str = "Confidence Analysis",
        save_path: str = None,
    ) -> None:
        """
        Plot confidence calibration and distribution.

        Args:
            confidences: Confidence values (flattened or per-sample means)
            accuracies: Per-bin accuracies
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Confidence histogram
        axes[0].hist(confidences.flatten(), bins=50, color="steelblue", alpha=0.7)
        axes[0].set_xlabel("Confidence")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Confidence Distribution")

        # Calibration plot
        if len(accuracies) > 0:
            bins = np.linspace(0, 1, len(accuracies) + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            valid = ~np.isnan(accuracies)

            axes[1].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            axes[1].scatter(
                bin_centers[valid], accuracies[valid], color="coral", s=50, label="Actual"
            )
            axes[1].set_xlabel("Mean Confidence")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title("Calibration Plot")
            axes[1].legend()
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.close()

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_path: str = None,
    ) -> None:
        """
        Plot training history curves.

        Args:
            history: Dictionary with metric lists
            title: Plot title
            save_path: Path to save figure
        """
        n_metrics = len(history)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))

        if n_metrics == 1:
            axes = [axes]

        for idx, (name, values) in enumerate(history.items()):
            axes[idx].plot(values, color="steelblue", linewidth=2)
            axes[idx].set_xlabel("Iteration")
            axes[idx].set_ylabel(name)
            axes[idx].set_title(name)
            axes[idx].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.close()

    def plot_uncertainty_map(
        self,
        image: np.ndarray,
        pred: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        title: str = None,
        save_path: str = None,
    ) -> None:
        """
        Plot prediction with uncertainty bounds.

        Args:
            image: Seismic image (H, W)
            pred: Predicted probabilities (H, W)
            lower_bound: Lower probability bound (H, W)
            upper_bound: Upper probability bound (H, W)
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Original
        axes[0].imshow(image, cmap="seismic")
        axes[0].set_title("Seismic Image")
        axes[0].axis("off")

        # Prediction
        im1 = axes[1].imshow(pred, cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title("Prediction")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Uncertainty width
        uncertainty = upper_bound - lower_bound
        im2 = axes[2].imshow(uncertainty, cmap="hot", vmin=0, vmax=0.5)
        axes[2].set_title("Uncertainty Width")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Bounds visualization
        bounds_rgb = np.zeros((*image.shape, 3))
        bounds_rgb[:, :, 0] = upper_bound  # Red channel: upper
        bounds_rgb[:, :, 2] = lower_bound  # Blue channel: lower
        axes[3].imshow(bounds_rgb)
        axes[3].set_title("Bounds (R:Upper, B:Lower)")
        axes[3].axis("off")

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.close()

    def create_summary_grid(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        preds: np.ndarray,
        confidences: np.ndarray = None,
        n_samples: int = 9,
        title: str = "Sample Predictions",
        save_path: str = None,
    ) -> None:
        """
        Create a grid of sample predictions.

        Args:
            images: Seismic images (N, H, W)
            masks: Ground truth masks (N, H, W)
            preds: Predictions (N, H, W)
            confidences: Confidence maps (N, H, W)
            n_samples: Number of samples to show
            title: Plot title
            save_path: Path to save figure
        """
        n_samples = min(n_samples, len(images))
        n_cols = 4 if confidences is not None else 3
        n_rows = int(np.ceil(np.sqrt(n_samples)))

        fig, axes = plt.subplots(
            n_rows * n_cols, n_rows, figsize=(4 * n_rows, 4 * n_cols)
        )

        # Simpler grid approach
        fig, axes = plt.subplots(n_samples, n_cols, figsize=(3 * n_cols, 3 * n_samples))

        for i in range(n_samples):
            # Image
            axes[i, 0].imshow(images[i], cmap="seismic")
            axes[i, 0].axis("off")
            if i == 0:
                axes[i, 0].set_title("Image")

            # Ground truth
            axes[i, 1].imshow(masks[i], cmap="gray")
            axes[i, 1].axis("off")
            if i == 0:
                axes[i, 1].set_title("Ground Truth")

            # Prediction
            axes[i, 2].imshow(preds[i], cmap="gray")
            axes[i, 2].axis("off")
            if i == 0:
                axes[i, 2].set_title("Prediction")

            # Confidence
            if confidences is not None:
                axes[i, 3].imshow(confidences[i], cmap="viridis", vmin=0, vmax=1)
                axes[i, 3].axis("off")
                if i == 0:
                    axes[i, 3].set_title("Confidence")

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.close()


if __name__ == "__main__":
    # Test visualizer
    print("Testing Visualizer...")

    np.random.seed(42)
    h, w = 101, 101

    # Create synthetic data
    image = np.random.rand(h, w).astype(np.float32)
    mask = (np.random.rand(h, w) > 0.6).astype(np.float32)
    pred = np.clip(mask + np.random.randn(h, w) * 0.2, 0, 1)
    confidence = np.random.rand(h, w).astype(np.float32)

    viz = Visualizer()

    print("Creating sample plot...")
    viz.plot_sample(
        image, mask, pred, confidence,
        title="Test Sample",
        save_path=os.path.join(viz.output_dir, "test_sample.png")
    )

    print("Creating overlay plot...")
    viz.plot_overlay(
        image, mask, pred,
        title="Test Overlay",
        save_path=os.path.join(viz.output_dir, "test_overlay.png")
    )

    print("Visualization tests complete!")
