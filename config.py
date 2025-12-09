"""
Configuration file for Depth-Adaptive Spectral Segmentation
Salt Body Characterization Project
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Datasets")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_MASKS_DIR = os.path.join(DATA_DIR, "train", "masks")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test", "images")
DEPTHS_CSV = os.path.join(DATA_DIR, "depths.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

# Output paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Image parameters
IMG_HEIGHT = 101
IMG_WIDTH = 101
IMG_CHANNELS = 1  # Convert to grayscale

# Spectral decomposition parameters
SPECTRAL_CONFIG = {
    "wavelet": "morl",  # Morlet wavelet for CWT
    "frequencies": [25, 40, 50],  # Hz - frequency bands for decomposition
    "scales": [2, 4, 8, 16, 32],  # Multi-scale analysis
    "sampling_rate": 100,  # Assumed sampling rate
}

# Feature engineering parameters
FEATURE_CONFIG = {
    "use_gradient_features": True,
    "use_texture_features": True,
    "use_statistical_features": True,
    "use_edge_features": True,
    "use_depth_encoding": True,
    "patch_size": 5,  # For local feature extraction
}

# Model parameters
MODEL_CONFIG = {
    # XGBoost parameters
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",  # CPU optimized
    },
    # Random Forest parameters
    "random_forest": {
        "n_estimators": 150,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    # Ensemble weights
    "ensemble_weights": {
        "xgboost": 0.6,
        "random_forest": 0.4,
    },
}

# Depth-adaptive confidence parameters
CONFIDENCE_CONFIG = {
    "shallow_depth_threshold": 200,  # meters
    "medium_depth_threshold": 500,  # meters
    "deep_depth_threshold": 800,  # meters
    "shallow_confidence_boost": 1.15,  # Higher confidence in shallow regions
    "deep_confidence_penalty": 0.85,  # Lower confidence in deep regions
    "uncertainty_scale_factor": 0.1,
}

# Training parameters
TRAINING_CONFIG = {
    "test_size": 0.2,
    "validation_size": 0.1,
    "random_state": 42,
    "batch_size": 1000,  # For feature extraction batching
    "use_stratified_split": True,
}

# Evaluation parameters
EVAL_CONFIG = {
    "iou_threshold": 0.5,
    "confidence_thresholds": [0.3, 0.5, 0.7, 0.9],
}
