"""
Training Pipeline for Depth-Adaptive Spectral Segmentation
Main entry point for training the salt body segmentation model
"""

import os
import sys
import time
import argparse
import numpy as np
import joblib
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    TRAINING_CONFIG,
    SPECTRAL_CONFIG,
    FEATURE_CONFIG,
    MODEL_CONFIG,
)
from src.data_loader import DataLoader
from src.spectral_decomposition import SpectralDecomposer
from src.feature_engineering import FeatureEngineer
from src.ensemble_model import HybridEnsemble
from src.confidence_module import DepthAdaptiveConfidence
from src.evaluation import Evaluator
from src.visualization import Visualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Depth-Adaptive Spectral Salt Segmentation Model"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples to use (None for all)",
    )
    parser.add_argument(
        "--n_pixels",
        type=int,
        default=200000,
        help="Number of pixels to sample for training",
    )
    parser.add_argument(
        "--skip_spectral",
        action="store_true",
        help="Skip spectral decomposition (use cached if available)",
    )
    parser.add_argument(
        "--save_features",
        action="store_true",
        help="Save extracted features to disk",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose output",
    )
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    # Create output directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("DEPTH-ADAPTIVE SPECTRAL SALT SEGMENTATION")
    print("Training Pipeline")
    print("=" * 60)

    total_start = time.time()

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)

    loader = DataLoader()
    images, masks, depths, ids = loader.load_all_train_data(verbose=args.verbose)

    if args.n_samples is not None:
        print(f"\nUsing subset of {args.n_samples} samples")
        indices = np.random.choice(len(images), args.n_samples, replace=False)
        images = images[indices]
        masks = masks[indices]
        depths = depths[indices]
        ids = [ids[i] for i in indices]

    # Print statistics
    print("\n--- Data Statistics ---")
    depth_stats = loader.get_depth_statistics(depths)
    print(f"Depth range: {depth_stats['min']:.0f} - {depth_stats['max']:.0f} m")
    print(f"Mean depth: {depth_stats['mean']:.0f} m")

    salt_stats = loader.get_salt_statistics(masks)
    print(f"Samples with salt: {salt_stats['samples_with_salt']}/{salt_stats['total_samples']}")
    print(f"Mean salt coverage: {salt_stats['mean_coverage']*100:.2f}%")

    # Create train/val split
    print("\nCreating train/validation split...")
    split_data = loader.get_stratified_split(images, masks, depths, ids)

    train_images = split_data["train"]["images"]
    train_masks = split_data["train"]["masks"]
    train_depths = split_data["train"]["depths"]

    val_images = split_data["val"]["images"]
    val_masks = split_data["val"]["masks"]
    val_depths = split_data["val"]["depths"]

    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")

    # =========================================================================
    # Step 2: Spectral Decomposition
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Spectral Decomposition")
    print("=" * 60)

    spectral_cache_path = os.path.join(MODELS_DIR, "spectral_features.npz")

    if args.skip_spectral and os.path.exists(spectral_cache_path):
        print("Loading cached spectral features...")
        cached = np.load(spectral_cache_path)
        train_spectral = cached["train_spectral"]
        val_spectral = cached["val_spectral"]
    else:
        decomposer = SpectralDecomposer()

        print("\nDecomposing training images...")
        start = time.time()
        train_spectral = decomposer.decompose_batch(train_images, verbose=args.verbose)
        print(f"Training decomposition time: {time.time() - start:.1f}s")

        print("\nDecomposing validation images...")
        start = time.time()
        val_spectral = decomposer.decompose_batch(val_images, verbose=args.verbose)
        print(f"Validation decomposition time: {time.time() - start:.1f}s")

        if args.save_features:
            print("\nSaving spectral features to cache...")
            np.savez_compressed(
                spectral_cache_path,
                train_spectral=train_spectral,
                val_spectral=val_spectral,
            )

    print(f"Spectral features shape: {train_spectral.shape}")

    # =========================================================================
    # Step 3: Feature Engineering
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Feature Engineering")
    print("=" * 60)

    engineer = FeatureEngineer()

    print("\nExtracting training features...")
    start = time.time()
    train_features = engineer.extract_batch_features(
        train_images, train_depths, train_spectral, verbose=args.verbose
    )
    print(f"Training feature extraction time: {time.time() - start:.1f}s")

    print("\nExtracting validation features...")
    start = time.time()
    val_features = engineer.extract_batch_features(
        val_images, val_depths, val_spectral, verbose=args.verbose
    )
    print(f"Validation feature extraction time: {time.time() - start:.1f}s")

    feature_names = engineer.get_feature_names()
    print(f"Total features per pixel: {len(feature_names)}")

    # =========================================================================
    # Step 4: Prepare Training Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Preparing Training Data")
    print("=" * 60)

    print("\nFlattening features for pixel-wise classification...")
    X_train_full, y_train_full = engineer.flatten_features(train_features, train_masks)
    print(f"Full training data: {X_train_full.shape}")

    print(f"\nSampling {args.n_pixels} balanced pixels...")
    X_train, y_train = engineer.sample_balanced_pixels(
        X_train_full, y_train_full, n_samples=args.n_pixels, salt_ratio=0.5
    )
    print(f"Sampled training data: {X_train.shape}")
    print(f"Salt ratio in sample: {y_train.mean():.2%}")

    # Clean up memory
    del X_train_full, y_train_full

    # =========================================================================
    # Step 5: Train Ensemble Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Training Ensemble Model")
    print("=" * 60)

    ensemble = HybridEnsemble()

    start = time.time()
    ensemble.fit(X_train, y_train, verbose=args.verbose)
    training_time = time.time() - start
    print(f"\nTotal training time: {training_time:.1f}s")

    # Save model
    model_path = ensemble.save(os.path.join(MODELS_DIR, f"ensemble_{timestamp}.joblib"))

    # Also save as latest
    ensemble.save(os.path.join(MODELS_DIR, "ensemble_latest.joblib"))

    # Print top features
    print("\n--- Top 15 Important Features ---")
    top_features = ensemble.get_top_features(feature_names, top_k=15)
    for name, importance in top_features:
        print(f"  {name}: {importance:.4f}")

    # =========================================================================
    # Step 6: Validation
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Validation")
    print("=" * 60)

    print("\nPredicting on validation set...")
    start = time.time()
    val_preds, val_proba = ensemble.predict_batch(val_features, verbose=args.verbose)
    inference_time = time.time() - start
    print(f"Validation inference time: {inference_time:.1f}s")
    print(f"Average time per image: {inference_time/len(val_images)*1000:.1f}ms")

    # =========================================================================
    # Step 7: Compute Confidence
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Computing Confidence Scores")
    print("=" * 60)

    confidence_module = DepthAdaptiveConfidence()

    print("\nComputing depth-adaptive confidence...")
    val_confidence, val_lower, val_upper = confidence_module.compute_batch_confidence(
        val_proba, val_depths, verbose=args.verbose
    )

    # Analyze confidence by depth
    conf_stats = confidence_module.analyze_depth_confidence(val_depths, val_confidence)
    print("\n--- Confidence by Depth ---")
    for key, value in conf_stats.items():
        if isinstance(value, dict) and 'mean_confidence' in value:
            print(f"{key}: mean={value['mean_confidence']:.3f}")

    # =========================================================================
    # Step 8: Evaluation
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 8: Evaluation")
    print("=" * 60)

    evaluator = Evaluator()

    # Generate full report
    report = evaluator.generate_report(
        val_preds, val_masks, val_depths, val_confidence
    )
    print(report)

    # Save report
    report_path = os.path.join(OUTPUTS_DIR, f"evaluation_report_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # =========================================================================
    # Step 9: Visualization
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 9: Creating Visualizations")
    print("=" * 60)

    viz = Visualizer(OUTPUTS_DIR)

    # Sample predictions
    print("\nCreating sample prediction visualizations...")
    n_viz = min(9, len(val_images))
    viz.create_summary_grid(
        val_images[:n_viz],
        val_masks[:n_viz],
        val_preds[:n_viz],
        val_confidence[:n_viz],
        title="Validation Samples",
        save_path=os.path.join(OUTPUTS_DIR, f"samples_{timestamp}.png"),
    )

    # Individual samples with uncertainty
    for i in range(min(3, len(val_images))):
        viz.plot_uncertainty_map(
            val_images[i],
            val_proba[i],
            val_lower[i],
            val_upper[i],
            title=f"Sample {i+1} - Depth: {val_depths[i]:.0f}m",
            save_path=os.path.join(OUTPUTS_DIR, f"uncertainty_sample_{i}_{timestamp}.png"),
        )

    # Feature importance
    print("\nCreating feature importance plot...")
    viz.plot_feature_importance(
        feature_names,
        ensemble.feature_importances_,
        top_k=20,
        title="Top 20 Feature Importances",
        save_path=os.path.join(OUTPUTS_DIR, f"feature_importance_{timestamp}.png"),
    )

    # Depth analysis
    print("\nCreating depth analysis plot...")
    depth_metrics = evaluator.compute_depth_stratified_metrics(
        val_preds, val_masks, val_depths
    )
    viz.plot_depth_analysis(
        val_depths,
        depth_metrics,
        title="Depth-Stratified Performance Analysis",
        save_path=os.path.join(OUTPUTS_DIR, f"depth_analysis_{timestamp}.png"),
    )

    # Confidence calibration
    print("\nCreating confidence analysis plot...")
    calib_data = evaluator.compute_confidence_calibration(
        val_preds, val_masks, val_confidence
    )
    viz.plot_confidence_analysis(
        val_confidence,
        calib_data["accuracies"],
        title="Confidence Analysis",
        save_path=os.path.join(OUTPUTS_DIR, f"confidence_analysis_{timestamp}.png"),
    )

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Final metrics
    final_metrics = evaluator.compute_batch_metrics(val_preds, val_masks)

    print(f"\n--- Final Results ---")
    print(f"IoU:       {final_metrics['iou_mean']:.4f} ± {final_metrics['iou_std']:.4f}")
    print(f"Dice:      {final_metrics['dice_mean']:.4f} ± {final_metrics['dice_std']:.4f}")
    print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall:    {final_metrics['recall']:.4f}")
    print(f"F1 Score:  {final_metrics['f1']:.4f}")

    print(f"\n--- Timing ---")
    print(f"Training time:  {training_time:.1f}s")
    print(f"Inference time: {inference_time:.1f}s ({inference_time/len(val_images)*1000:.1f}ms/image)")
    print(f"Total time:     {total_time:.1f}s ({total_time/60:.1f} minutes)")

    print(f"\n--- Output Files ---")
    print(f"Model:  {model_path}")
    print(f"Report: {report_path}")
    print(f"Plots:  {OUTPUTS_DIR}/")

    return final_metrics


if __name__ == "__main__":
    metrics = main()
