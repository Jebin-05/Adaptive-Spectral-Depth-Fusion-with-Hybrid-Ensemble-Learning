"""
Evaluation Module
Metrics computation and model evaluation for salt segmentation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EVAL_CONFIG


class Evaluator:
    """
    Evaluation metrics for salt body segmentation.
    Includes IoU, Dice, pixel-level metrics, and depth-stratified analysis.
    """

    def __init__(self, config: Dict = None):
        self.config = config or EVAL_CONFIG
        self.iou_threshold = self.config["iou_threshold"]

    def compute_iou(
        self, pred: np.ndarray, target: np.ndarray, threshold: float = 0.5
    ) -> float:
        """
        Compute Intersection over Union (IoU / Jaccard Index).

        Args:
            pred: Predicted mask or probabilities (H, W)
            target: Ground truth mask (H, W)
            threshold: Binarization threshold

        Returns:
            IoU score
        """
        pred_binary = (pred > threshold).astype(bool)
        target_binary = (target > threshold).astype(bool)

        intersection = np.logical_and(pred_binary, target_binary).sum()
        union = np.logical_or(pred_binary, target_binary).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return intersection / union

    def compute_dice(
        self, pred: np.ndarray, target: np.ndarray, threshold: float = 0.5
    ) -> float:
        """
        Compute Dice coefficient (F1 for segmentation).

        Args:
            pred: Predicted mask or probabilities (H, W)
            target: Ground truth mask (H, W)
            threshold: Binarization threshold

        Returns:
            Dice score
        """
        pred_binary = (pred > threshold).astype(bool)
        target_binary = (target > threshold).astype(bool)

        intersection = np.logical_and(pred_binary, target_binary).sum()
        total = pred_binary.sum() + target_binary.sum()

        if total == 0:
            return 1.0 if intersection == 0 else 0.0

        return 2 * intersection / total

    def compute_pixel_metrics(
        self, pred: np.ndarray, target: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute pixel-level classification metrics.

        Args:
            pred: Predicted mask or probabilities (H, W) or flattened
            target: Ground truth mask (H, W) or flattened
            threshold: Binarization threshold

        Returns:
            Dictionary of metrics
        """
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        pred_binary = (pred_flat > threshold).astype(int)
        target_binary = (target_flat > threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(target_binary, pred_binary),
            "precision": precision_score(target_binary, pred_binary, zero_division=0),
            "recall": recall_score(target_binary, pred_binary, zero_division=0),
            "f1": f1_score(target_binary, pred_binary, zero_division=0),
        }

        # AUC if we have probabilities
        if pred_flat.min() >= 0 and pred_flat.max() <= 1:
            try:
                metrics["auc"] = roc_auc_score(target_binary, pred_flat)
            except ValueError:
                metrics["auc"] = 0.5

        return metrics

    def compute_batch_metrics(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute metrics over a batch of images.

        Args:
            preds: Predicted masks (N, H, W)
            targets: Ground truth masks (N, H, W)
            threshold: Binarization threshold

        Returns:
            Dictionary of aggregated metrics
        """
        n_samples = preds.shape[0]

        ious = []
        dices = []

        for i in range(n_samples):
            ious.append(self.compute_iou(preds[i], targets[i], threshold))
            dices.append(self.compute_dice(preds[i], targets[i], threshold))

        # Pixel-level metrics on all data
        pixel_metrics = self.compute_pixel_metrics(preds, targets, threshold)

        return {
            "iou_mean": np.mean(ious),
            "iou_std": np.std(ious),
            "iou_median": np.median(ious),
            "dice_mean": np.mean(dices),
            "dice_std": np.std(dices),
            "dice_median": np.median(dices),
            **pixel_metrics,
        }

    def compute_depth_stratified_metrics(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        depths: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics stratified by depth ranges.

        Args:
            preds: Predicted masks (N, H, W)
            targets: Ground truth masks (N, H, W)
            depths: Depth values (N,)
            threshold: Binarization threshold

        Returns:
            Dictionary with metrics for each depth range
        """
        depth_ranges = [
            ("shallow_0-200m", 0, 200),
            ("medium_200-500m", 200, 500),
            ("deep_500-800m", 500, 800),
            ("very_deep_800m+", 800, 1000),
        ]

        results = {}

        for name, d_min, d_max in depth_ranges:
            mask = (depths >= d_min) & (depths < d_max)
            if mask.sum() < 5:
                continue

            subset_preds = preds[mask]
            subset_targets = targets[mask]

            metrics = self.compute_batch_metrics(subset_preds, subset_targets, threshold)
            metrics["n_samples"] = mask.sum()
            results[name] = metrics

        return results

    def compute_coverage_stratified_metrics(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics stratified by salt coverage.

        Args:
            preds: Predicted masks (N, H, W)
            targets: Ground truth masks (N, H, W)
            threshold: Binarization threshold

        Returns:
            Dictionary with metrics for each coverage range
        """
        coverages = np.array([t.mean() for t in targets])

        coverage_ranges = [
            ("no_salt_0%", 0.0, 0.001),
            ("low_0-10%", 0.001, 0.1),
            ("medium_10-50%", 0.1, 0.5),
            ("high_50-100%", 0.5, 1.01),
        ]

        results = {}

        for name, c_min, c_max in coverage_ranges:
            mask = (coverages >= c_min) & (coverages < c_max)
            if mask.sum() < 5:
                continue

            subset_preds = preds[mask]
            subset_targets = targets[mask]

            metrics = self.compute_batch_metrics(subset_preds, subset_targets, threshold)
            metrics["n_samples"] = mask.sum()
            results[name] = metrics

        return results

    def compute_confidence_calibration(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        confidences: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Compute confidence calibration metrics.
        Well-calibrated models should have accuracy ≈ confidence.

        Args:
            preds: Predicted masks (N, H, W)
            targets: Ground truth masks (N, H, W)
            confidences: Confidence maps (N, H, W)
            n_bins: Number of calibration bins

        Returns:
            Dictionary with calibration data
        """
        # Flatten all
        preds_flat = preds.flatten()
        targets_flat = targets.flatten()
        conf_flat = confidences.flatten()

        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        accuracies = []
        mean_confidences = []
        counts = []

        for i in range(n_bins):
            mask = (conf_flat >= bin_edges[i]) & (conf_flat < bin_edges[i + 1])
            if mask.sum() == 0:
                accuracies.append(np.nan)
                mean_confidences.append(np.nan)
                counts.append(0)
                continue

            bin_preds = (preds_flat[mask] > 0.5).astype(int)
            bin_targets = (targets_flat[mask] > 0.5).astype(int)
            bin_conf = conf_flat[mask]

            accuracies.append((bin_preds == bin_targets).mean())
            mean_confidences.append(bin_conf.mean())
            counts.append(mask.sum())

        # Expected Calibration Error
        accuracies = np.array(accuracies)
        mean_confidences = np.array(mean_confidences)
        counts = np.array(counts)

        valid = ~np.isnan(accuracies)
        if valid.sum() > 0:
            ece = np.sum(
                counts[valid] * np.abs(accuracies[valid] - mean_confidences[valid])
            ) / counts[valid].sum()
        else:
            ece = 0.0

        return {
            "bin_centers": bin_centers,
            "accuracies": accuracies,
            "mean_confidences": mean_confidences,
            "counts": counts,
            "expected_calibration_error": ece,
        }

    def threshold_search(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        thresholds: np.ndarray = None,
        metric: str = "iou",
    ) -> Tuple[float, float]:
        """
        Search for optimal threshold.

        Args:
            preds: Predicted probabilities (N, H, W)
            targets: Ground truth masks (N, H, W)
            thresholds: Thresholds to try
            metric: Metric to optimize ('iou' or 'dice')

        Returns:
            best_threshold, best_score
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05)

        best_threshold = 0.5
        best_score = 0.0

        for thresh in thresholds:
            if metric == "iou":
                scores = [
                    self.compute_iou(preds[i], targets[i], thresh)
                    for i in range(len(preds))
                ]
            else:
                scores = [
                    self.compute_dice(preds[i], targets[i], thresh)
                    for i in range(len(preds))
                ]

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_threshold = thresh

        return best_threshold, best_score

    def generate_report(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        depths: np.ndarray = None,
        confidences: np.ndarray = None,
        threshold: float = 0.5,
    ) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            preds: Predicted masks (N, H, W)
            targets: Ground truth masks (N, H, W)
            depths: Depth values (N,)
            confidences: Confidence maps (N, H, W)
            threshold: Classification threshold

        Returns:
            Report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("SALT SEGMENTATION EVALUATION REPORT")
        lines.append("=" * 60)

        # Overall metrics
        lines.append("\n--- OVERALL METRICS ---")
        metrics = self.compute_batch_metrics(preds, targets, threshold)
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")

        # Threshold search
        lines.append("\n--- OPTIMAL THRESHOLD SEARCH ---")
        best_thresh, best_iou = self.threshold_search(preds, targets, metric="iou")
        lines.append(f"Best threshold (IoU): {best_thresh:.2f} -> IoU: {best_iou:.4f}")

        # Depth-stratified metrics
        if depths is not None:
            lines.append("\n--- DEPTH-STRATIFIED METRICS ---")
            depth_metrics = self.compute_depth_stratified_metrics(
                preds, targets, depths, threshold
            )
            for depth_range, metrics in depth_metrics.items():
                lines.append(f"\n{depth_range}:")
                lines.append(f"  Samples: {metrics['n_samples']}")
                lines.append(f"  IoU: {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}")
                lines.append(f"  Dice: {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
                lines.append(f"  Accuracy: {metrics['accuracy']:.4f}")

        # Coverage-stratified metrics
        lines.append("\n--- COVERAGE-STRATIFIED METRICS ---")
        coverage_metrics = self.compute_coverage_stratified_metrics(
            preds, targets, threshold
        )
        for coverage_range, metrics in coverage_metrics.items():
            lines.append(f"\n{coverage_range}:")
            lines.append(f"  Samples: {metrics['n_samples']}")
            lines.append(f"  IoU: {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}")

        # Confidence calibration
        if confidences is not None:
            lines.append("\n--- CONFIDENCE CALIBRATION ---")
            calib = self.compute_confidence_calibration(preds, targets, confidences)
            lines.append(f"Expected Calibration Error: {calib['expected_calibration_error']:.4f}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    # Test evaluator
    print("Testing Evaluator...")

    np.random.seed(42)
    n_samples = 100
    h, w = 101, 101

    # Create synthetic data
    targets = (np.random.rand(n_samples, h, w) > 0.6).astype(np.float32)
    preds = targets + np.random.randn(n_samples, h, w) * 0.2
    preds = np.clip(preds, 0, 1)
    depths = np.random.randint(50, 900, n_samples).astype(np.float32)
    confidences = np.random.rand(n_samples, h, w).astype(np.float32)

    evaluator = Evaluator()

    print("\n--- Single image metrics ---")
    iou = evaluator.compute_iou(preds[0], targets[0])
    dice = evaluator.compute_dice(preds[0], targets[0])
    print(f"IoU: {iou:.4f}")
    print(f"Dice: {dice:.4f}")

    print("\n--- Batch metrics ---")
    batch_metrics = evaluator.compute_batch_metrics(preds, targets)
    for key, value in batch_metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    print("\n--- Full report ---")
    report = evaluator.generate_report(preds, targets, depths, confidences)
    print(report)
