# Salt Exploration Analysis System

A machine learning system for detecting salt deposits in seismic survey images. The system automatically predicts optimal drilling depths and provides detailed analysis of salt body characteristics.

> **Dataset Required:** This project requires the [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge/data) dataset from Kaggle. Please download and place it in the `Datasets/` folder before training. See [Installation](#installation) for details.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Project Structure](#project-structure)
7. [Usage](#usage)
   - [Training the Model](#training-the-model)
   - [Running the Web Application](#running-the-web-application)
   - [Using the Inference Module](#using-the-inference-module)
8. [Configuration](#configuration)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)
11. [License](#license)

---

## Overview

This system analyzes seismic survey images to:

- Detect salt body presence and boundaries
- Predict optimal drilling depth automatically (no manual input required)
- Estimate salt quantity and coverage
- Provide confidence scores for predictions
- Generate AI-powered geological interpretations

The core technology combines Continuous Wavelet Transform (CWT) spectral decomposition with an XGBoost and Random Forest ensemble model.

---

## Features

| Feature | Description |
|---------|-------------|
| Automatic Depth Prediction | Scans 0-2000m range to find optimal drilling depth |
| Salt Detection | Pixel-level salt body segmentation |
| Quantity Estimation | Coverage percentage and volume index calculation |
| Confidence Scoring | Depth-adaptive confidence with uncertainty bounds |
| AI Interpretation | LLM-powered geological analysis of results |
| Web Interface | Interactive Streamlit dashboard |
| Batch Processing | Process multiple images via inference module |

---

## System Architecture

```
Input Image (101x101 PNG)
        |
        v
+-------------------+
| Spectral Decomp.  |  <-- Continuous Wavelet Transform (Morlet)
| (Multi-scale CWT) |
+-------------------+
        |
        v
+-------------------+
| Feature Engineer  |  <-- Gradient, Texture, Statistical, Edge features
| (100+ features)   |
+-------------------+
        |
        v
+-------------------+
| Ensemble Model    |  <-- XGBoost (60%) + Random Forest (40%)
| (Hybrid Voting)   |
+-------------------+
        |
        v
+-------------------+
| Confidence Module |  <-- Depth-adaptive scoring
+-------------------+
        |
        v
+-------------------+
| Salt Analysis     |  <-- Depth, thickness, coverage calculations
+-------------------+
        |
        v
Output: Mask, Probabilities, Metrics, Recommendations
```

---

## Requirements

### Software Requirements

- Python 3.8 or higher
- pip (Python package manager)

### Hardware Requirements

- Minimum 4GB RAM
- CPU with multiple cores (for parallel processing)
- GPU not required (model uses CPU-optimized algorithms)

### Python Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
joblib>=1.1.0
Pillow>=8.0.0
scikit-image>=0.19.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
streamlit>=1.20.0
groq>=0.4.0
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Jebin-05/Adaptive-Spectral-Depth-Fusion-with-Hybrid-Ensemble-Learning.git
cd Adaptive-Spectral-Depth-Fusion-with-Hybrid-Ensemble-Learning
```

### Step 2: Create a Virtual Environment (Recommended)

On Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Additionally, install Streamlit and Groq for the web interface:
```bash
pip install streamlit groq
```

### Step 4: Download the Dataset

**Important:** This project requires the **TGS Salt Identification Challenge** dataset from Kaggle.

1. Download the dataset from: https://www.kaggle.com/c/tgs-salt-identification-challenge/data
2. Extract and place the files in the `Datasets/` folder with the following structure:
   ```
   Datasets/
   ├── train/
   │   ├── images/      # Training seismic images
   │   └── masks/       # Ground truth salt masks
   ├── test/
   │   └── images/      # Test seismic images
   ├── depths.csv       # Depth values for each sample
   ├── train.csv        # Training metadata
   └── sample_submission.csv
   ```

### Step 5: Verify Installation

```bash
python -c "import numpy, scipy, sklearn, xgboost, streamlit; print('All dependencies installed successfully')"
```

---

## Quick Start (Using Pre-Trained Model)

**No training required!** This repository includes a pre-trained model ready for immediate use.

### Option 1: Run the Web Application

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser and upload a seismic image.

### Option 2: Run Inference from Command Line

```bash
# Process test images
python inference.py --n_samples 10 --visualize

# Process with custom threshold
python inference.py --threshold 0.4 --save_predictions
```

### Option 3: Use in Your Python Code

```python
from inference import SaltSegmentationPipeline
import numpy as np
from PIL import Image

# Load the pre-trained model
pipeline = SaltSegmentationPipeline()  # Uses models/ensemble_latest.joblib

# Load your seismic image
image = np.array(Image.open("your_image.png").convert("L")) / 255.0

# Run prediction
result = pipeline.predict_single(image, depth=300.0)

print(f"Salt coverage: {result['mask'].mean() * 100:.1f}%")
print(f"Mean confidence: {result['confidence'].mean():.2f}")
print(f"Inference time: {result['inference_time_ms']:.1f} ms")
```

---

## Training Your Own Model (Optional)

If you want to train from scratch on your own dataset:

### Step 1: Prepare the Dataset

Place your seismic images in the following structure:
```
Datasets/
├── train/
│   ├── images/      # Training seismic images (PNG, 101x101)
│   └── masks/       # Ground truth salt masks (PNG, 101x101)
├── test/
│   └── images/      # Test seismic images
├── depths.csv       # Depth values for each sample (optional)
└── train.csv        # Training metadata (optional)
```

---

## Project Structure

```
salt-exploration/
│
├── app.py                 # Streamlit web application
├── train.py               # Model training script
├── inference.py           # Batch inference module
├── config.py              # Configuration parameters
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
│
├── src/                   # Core modules
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── spectral_decomposition.py # CWT spectral analysis
│   ├── feature_engineering.py    # Feature extraction
│   ├── ensemble_model.py         # ML model (XGBoost + RF)
│   ├── confidence_module.py      # Confidence scoring
│   ├── evaluation.py             # Metrics and evaluation
│   └── visualization.py          # Plotting utilities
│
├── models/                # Trained models (included in repo)
│   └── ensemble_latest.joblib  # Pre-trained model (35MB)
│
├── outputs/               # Analysis outputs
│   └── (generated files)
│
├── Datasets/              # Training and test data
│   ├── train/
│   └── test/
│
└── notebooks/             # Jupyter notebooks (optional)
```

---

## Usage

### Running the Web Application

Start the Streamlit application:
```bash
streamlit run app.py
```

Open your browser and navigate to http://localhost:8501

Using the application:

1. Upload a seismic survey image (PNG or JPG, will be resized to 101x101)
2. Click "Analyze for Salt Deposits"
3. View results:
   - AI-predicted optimal survey depth
   - Recommended drilling depth
   - Salt coverage percentage
   - Salt layer thickness
   - Model confidence score
4. Click "Generate Expert Geological Interpretation" for AI analysis
5. Download results (salt map, probability map, report)

### Re-Training the Model (Optional)

If you want to train a new model on your own dataset:

**Step 1:** Ensure your training data is in the correct location:
- Images: `Datasets/train/images/`
- Masks: `Datasets/train/masks/`

**Step 2:** Run the training script:
```bash
python train.py
```

**Step 3:** Wait for training to complete. The script will:
- Load and preprocess all training images
- Extract spectral and hand-crafted features
- Train the XGBoost and Random Forest ensemble
- Save the model to `models/ensemble_latest.joblib`
- Generate evaluation metrics and visualizations in `outputs/`

Training time depends on dataset size. For 4000 images, expect 10-30 minutes.

---

## Configuration

All configuration parameters are in `config.py`. Key settings:

### Image Parameters
```python
IMG_HEIGHT = 101
IMG_WIDTH = 101
```

### Model Parameters
```python
MODEL_CONFIG = {
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
    },
    "random_forest": {
        "n_estimators": 150,
        "max_depth": 12,
    },
    "ensemble_weights": {
        "xgboost": 0.6,
        "random_forest": 0.4,
    },
}
```

### Depth Confidence Parameters
```python
CONFIDENCE_CONFIG = {
    "shallow_depth_threshold": 200,   # meters
    "medium_depth_threshold": 500,    # meters
    "deep_depth_threshold": 800,      # meters
}
```

---

## API Reference

### SaltInference Class

```python
class SaltInference:
    def __init__(self, model_path: str = None)
    def predict_single(self, image_path: str, depth: float = 300) -> dict
    def predict_batch(self, image_paths: list, depths: list = None) -> list
```

### HybridEnsemble Class

```python
class HybridEnsemble:
    def fit(self, X: np.ndarray, y: np.ndarray) -> self
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray
    def predict_proba(self, X: np.ndarray) -> np.ndarray
    def predict_image(self, features: np.ndarray, img_shape: tuple) -> tuple
    def save(self, path: str) -> str
    def load(self, path: str) -> self
```

### FeatureEngineer Class

```python
class FeatureEngineer:
    def extract_all_features(self, image: np.ndarray, depth: float, spectral: dict) -> np.ndarray
    def compute_gradient_features(self, image: np.ndarray) -> dict
    def compute_texture_features(self, image: np.ndarray) -> dict
    def compute_statistical_features(self, image: np.ndarray) -> dict
```

---

## Troubleshooting

### Problem: "Model not found" error when running app.py

**Cause:** The model has not been trained yet.

**Solution:** Run `python train.py` first to train and save the model.

---

### Problem: "ModuleNotFoundError: No module named 'xxx'"

**Cause:** Missing Python dependency.

**Solution:** Install the missing module:
```bash
pip install module_name
```

Or reinstall all dependencies:
```bash
pip install -r requirements.txt
```

---

### Problem: Application runs but predictions are always the same

**Cause:** Images may not be loading correctly or model cache issue.

**Solution:**
1. Ensure images are valid PNG/JPG files
2. Clear Streamlit cache: Delete `.streamlit/` folder and restart
3. Verify images are different (check "Image ID" in results)

---

### Problem: Streamlit app not accessible

**Cause:** Port 8501 may be in use or firewall blocking.

**Solution:**
```bash
# Run on different port
streamlit run app.py --server.port 8502

# Or kill existing process
pkill -f streamlit
streamlit run app.py
```

---

### Problem: Out of memory during training

**Cause:** Dataset too large for available RAM.

**Solution:**
1. Reduce batch size in `config.py`:
   ```python
   TRAINING_CONFIG = {
       "batch_size": 500,  # Reduce from 1000
   }
   ```
2. Train on a subset of data first

---

### Problem: AI Explanation not working

**Cause:** Groq API key issue or rate limiting.

**Solution:**
1. Check API key in `app.py` is valid
2. Wait a few minutes if rate limited
3. The main analysis still works without AI explanation

---

## License

This project is provided for educational and research purposes.

---

## Acknowledgments

- TGS Salt Identification Challenge for the original dataset format
- XGBoost and scikit-learn teams for the ML libraries
- Streamlit team for the web framework
- Groq for the LLM API

---

## Contact

For questions or issues, please open an issue on the GitHub repository.
