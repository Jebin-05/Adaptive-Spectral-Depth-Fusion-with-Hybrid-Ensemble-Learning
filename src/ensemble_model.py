"""
Hybrid Ensemble Model Module
XGBoost + Random Forest ensemble for salt segmentation
"""

import numpy as np
import joblib
from typing import Dict, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, MODELS_DIR


class HybridEnsemble:
    """
    Hybrid ensemble combining XGBoost and Random Forest for salt segmentation.
    Includes probability calibration and weighted voting.
    """

    def __init__(self, config: Dict = None):
        self.config = config or MODEL_CONFIG
        self.xgb_model = None
        self.rf_model = None
        self.is_fitted = False
        self.feature_importances_ = None

        # Initialize models
        self._init_models()

    def _init_models(self) -> None:
        """Initialize XGBoost and Random Forest models."""
        # XGBoost
        xgb_params = self.config["xgboost"].copy()
        self.xgb_model = XGBClassifier(**xgb_params)

        # Random Forest
        rf_params = self.config["random_forest"].copy()
        self.rf_model = RandomForestClassifier(**rf_params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> "HybridEnsemble":
        """
        Fit both models in the ensemble.

        Args:
            X: Feature array (N, n_features)
            y: Label array (N,)
            verbose: Whether to print progress

        Returns:
            self
        """
        if verbose:
            print(f"Training ensemble on {X.shape[0]} samples with {X.shape[1]} features")

        # Ensure binary labels
        y_binary = (y > 0.5).astype(int)

        # Train XGBoost
        if verbose:
            print("\n--- Training XGBoost ---")
        self.xgb_model.fit(X, y_binary)
        if verbose:
            print("XGBoost training complete")

        # Train Random Forest
        if verbose:
            print("\n--- Training Random Forest ---")
        self.rf_model.fit(X, y_binary)
        if verbose:
            print("Random Forest training complete")

        # Compute combined feature importances
        self._compute_feature_importances()

        self.is_fitted = True
        return self

    def _compute_feature_importances(self) -> None:
        """Compute weighted average of feature importances from both models."""
        xgb_weight = self.config["ensemble_weights"]["xgboost"]
        rf_weight = self.config["ensemble_weights"]["random_forest"]

        xgb_importance = self.xgb_model.feature_importances_
        rf_importance = self.rf_model.feature_importances_

        self.feature_importances_ = (
            xgb_weight * xgb_importance + rf_weight * rf_importance
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of salt for each pixel.

        Args:
            X: Feature array (N, n_features)

        Returns:
            Probability array (N,) - probability of being salt
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Get probabilities from both models
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        rf_proba = self.rf_model.predict_proba(X)[:, 1]

        # Weighted average
        xgb_weight = self.config["ensemble_weights"]["xgboost"]
        rf_weight = self.config["ensemble_weights"]["random_forest"]

        ensemble_proba = xgb_weight * xgb_proba + rf_weight * rf_proba
        return ensemble_proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary salt/non-salt labels.

        Args:
            X: Feature array (N, n_features)
            threshold: Classification threshold

        Returns:
            Binary prediction array (N,)
        """
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)

    def predict_image(
        self,
        features: np.ndarray,
        img_shape: Tuple[int, int],
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict salt mask for a single image.

        Args:
            features: Feature array (n_features, H, W)
            img_shape: Original image shape (H, W)
            threshold: Classification threshold

        Returns:
            Binary mask (H, W) and probability map (H, W)
        """
        n_features, h, w = features.shape

        # Flatten for prediction
        X = features.reshape(n_features, -1).T  # (H*W, n_features)

        # Predict probabilities
        proba = self.predict_proba(X)

        # Reshape to image
        proba_map = proba.reshape(h, w)
        mask = (proba_map > threshold).astype(np.float32)

        return mask, proba_map

    def predict_batch(
        self,
        features: np.ndarray,
        threshold: float = 0.5,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict salt masks for a batch of images.

        Args:
            features: Feature array (N, n_features, H, W)
            threshold: Classification threshold
            verbose: Whether to print progress

        Returns:
            Binary masks (N, H, W) and probability maps (N, H, W)
        """
        n_samples, n_features, h, w = features.shape

        masks = np.zeros((n_samples, h, w), dtype=np.float32)
        proba_maps = np.zeros((n_samples, h, w), dtype=np.float32)

        for i in range(n_samples):
            if verbose and (i + 1) % 200 == 0:
                print(f"Predicting: {i + 1}/{n_samples}")

            masks[i], proba_maps[i] = self.predict_image(features[i], (h, w), threshold)

        return masks, proba_maps

    def get_model_predictions(
        self, X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get individual model predictions for analysis.

        Args:
            X: Feature array (N, n_features)

        Returns:
            Dictionary with individual model probabilities
        """
        return {
            "xgboost": self.xgb_model.predict_proba(X)[:, 1],
            "random_forest": self.rf_model.predict_proba(X)[:, 1],
            "ensemble": self.predict_proba(X),
        }

    def get_top_features(
        self, feature_names: List[str], top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Get top important features.

        Args:
            feature_names: List of feature names
            top_k: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fitted first")

        indices = np.argsort(self.feature_importances_)[::-1][:top_k]
        return [(feature_names[i], self.feature_importances_[i]) for i in indices]

    def save(self, path: str = None) -> str:
        """
        Save the ensemble model to disk.

        Args:
            path: Save path (default: models/hybrid_ensemble.joblib)

        Returns:
            Path where model was saved
        """
        if path is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            path = os.path.join(MODELS_DIR, "hybrid_ensemble.joblib")

        model_data = {
            "xgb_model": self.xgb_model,
            "rf_model": self.rf_model,
            "config": self.config,
            "feature_importances": self.feature_importances_,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
        return path

    def load(self, path: str = None) -> "HybridEnsemble":
        """
        Load the ensemble model from disk.

        Args:
            path: Load path (default: models/hybrid_ensemble.joblib)

        Returns:
            self
        """
        if path is None:
            path = os.path.join(MODELS_DIR, "hybrid_ensemble.joblib")

        model_data = joblib.load(path)
        self.xgb_model = model_data["xgb_model"]
        self.rf_model = model_data["rf_model"]
        self.config = model_data["config"]
        self.feature_importances_ = model_data["feature_importances"]
        self.is_fitted = model_data["is_fitted"]

        print(f"Model loaded from {path}")
        return self


class DepthAwareEnsemble(HybridEnsemble):
    """
    Extended ensemble that incorporates depth information for
    depth-stratified training and prediction.
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.depth_models = {}  # Separate models for depth ranges
        self.depth_ranges = [
            (0, 200),     # Shallow
            (200, 500),   # Medium
            (500, 800),   # Deep
            (800, 1000),  # Very deep
        ]

    def fit_depth_stratified(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depths: np.ndarray,
        verbose: bool = True,
    ) -> "DepthAwareEnsemble":
        """
        Fit separate models for different depth ranges.

        Args:
            X: Feature array (N, n_features)
            y: Label array (N,)
            depths: Depth values (N,)
            verbose: Whether to print progress
        """
        # First fit global model
        self.fit(X, y, verbose=verbose)

        # Then fit depth-specific models
        for depth_min, depth_max in self.depth_ranges:
            mask = (depths >= depth_min) & (depths < depth_max)
            if mask.sum() < 1000:
                continue

            if verbose:
                print(f"\n--- Training model for depth {depth_min}-{depth_max}m ---")
                print(f"Samples: {mask.sum()}")

            model = HybridEnsemble(self.config)
            model.fit(X[mask], y[mask], verbose=False)
            self.depth_models[(depth_min, depth_max)] = model

        return self

    def predict_proba_depth_aware(
        self, X: np.ndarray, depth: float
    ) -> np.ndarray:
        """
        Predict with depth-aware model selection.

        Args:
            X: Feature array (N, n_features)
            depth: Depth value

        Returns:
            Probability array (N,)
        """
        # Find appropriate depth model
        depth_model = None
        for (depth_min, depth_max), model in self.depth_models.items():
            if depth_min <= depth < depth_max:
                depth_model = model
                break

        if depth_model is None:
            # Fall back to global model
            return self.predict_proba(X)

        # Blend global and depth-specific predictions
        global_proba = self.predict_proba(X)
        depth_proba = depth_model.predict_proba(X)

        # Weight depth model more for its specific range
        return 0.4 * global_proba + 0.6 * depth_proba


if __name__ == "__main__":
    # Test ensemble model
    print("Testing HybridEnsemble...")

    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 10000
    n_features = 50

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * 2 + np.random.randn(n_samples) * 0.5 > 0).astype(int)

    print(f"Test data: {X.shape}, {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Train ensemble
    ensemble = HybridEnsemble()
    ensemble.fit(X, y, verbose=True)

    # Predict
    proba = ensemble.predict_proba(X)
    preds = ensemble.predict(X)

    accuracy = (preds == y).mean()
    print(f"\nTraining accuracy: {accuracy:.4f}")

    # Test save/load
    path = ensemble.save()
    loaded_ensemble = HybridEnsemble()
    loaded_ensemble.load(path)

    loaded_proba = loaded_ensemble.predict_proba(X)
    print(f"Loaded model predictions match: {np.allclose(proba, loaded_proba)}")
