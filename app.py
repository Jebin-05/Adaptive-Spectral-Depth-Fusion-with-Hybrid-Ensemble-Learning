"""
Salt Exploration Analysis System
Practical tool for estimating salt depth and quantity from seismic images
With Explainable AI powered by Gemini 2.5 Flash
"""

import os
import sys
import time
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
from groq import Groq

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODELS_DIR
from src.data_loader import DataLoader
from src.spectral_decomposition import SpectralDecomposer
from src.feature_engineering import FeatureEngineer
from src.ensemble_model import HybridEnsemble
from src.confidence_module import DepthAdaptiveConfidence
from src.evaluation import Evaluator
from src.image_validator import SeismicImageValidator, validate_seismic_image

# Groq API Configuration
# Set your Groq API key as an environment variable: export GROQ_API_KEY="your-api-key"
# Or enter it in the Streamlit sidebar when running the app
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# System prompt for geological salt exploration AI - EXPLAINS MODEL OUTPUT ONLY
SYSTEM_PROMPT = """You are reporting the EXACT predictions from a trained CWT + XGBoost/RF salt detection model.

CRITICAL RULES:
1. Use ONLY the EXACT numbers provided - no approximations, no "approximately", no "around"
2. State depths as PRECISE values (e.g., "324m" not "approximately 320m")
3. This is a trained ML model's output - report it with confidence
4. Keep response under 250 words - be direct

OUTPUT FORMAT (use exact numbers from input):

## DETECTION RESULT
The model detected salt at **[exact first_salt_depth]m** depth with **[exact coverage]%** coverage.

## PRECISE DRILLING COORDINATES
• **Start Drilling**: [exact first_salt_depth]m
• **Target Zone**: [exact first_salt_depth]m to [exact bottom_salt_depth]m
• **Peak Concentration**: [exact max_salt_depth]m
• **Salt Thickness**: [exact thickness]m

## QUANTITY DETECTED
• **Coverage**: [exact]% of survey area
• **Volume Index**: [exact]/100
• **Salt Pixels**: [exact] detected

## MODEL CONFIDENCE
**[exact confidence]%** - [one sentence interpretation]

## ACTION
[One clear, specific recommendation using exact depths]

## EXTRACTION METHOD
For [exact depth]m: [specific technique and equipment]

---
REMEMBER: The trained model has analyzed this seismic image and produced these EXACT predictions. Report them precisely."""


def get_groq_client():
    """Initialize Groq client."""
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        return None


def generate_ai_explanation(analysis_results, survey_depth):
    """Generate AI explanation using Groq LLaMA 3.3."""

    client = get_groq_client()
    if client is None:
        return "Error: AI service unavailable. Please check the configuration."

    # Prepare the analysis data for the prompt - provide EXACT values
    if analysis_results["has_salt"]:
        # Calculate precise values
        first_salt = int(analysis_results['depth_to_first_salt'])
        max_salt = int(analysis_results['depth_to_max_salt'])
        bottom_salt = int(analysis_results['depth_to_bottom_salt'])
        thickness = int(analysis_results['salt_thickness'])
        coverage = round(analysis_results['salt_coverage_percent'], 1)
        volume = round(analysis_results['estimated_salt_volume'], 1)
        salt_pixels = int(analysis_results['salt_area_pixels'])
        confidence = round(analysis_results['confidence'] * 100, 1)

        analysis_text = f"""TRAINED MODEL OUTPUT (EXACT VALUES):

DEPTH PREDICTIONS:
- First Salt Depth: {first_salt}m
- Peak Salt Depth: {max_salt}m
- Bottom Salt Depth: {bottom_salt}m
- Salt Thickness: {thickness}m
- Survey Range: {survey_depth}m to {survey_depth + 200}m

QUANTITY PREDICTIONS:
- Coverage: {coverage}%
- Volume Index: {volume}/100
- Salt Pixels: {salt_pixels}

CONFIDENCE: {confidence}%

Use these EXACT numbers in your response. No approximations."""
    else:
        analysis_text = f"""TRAINED MODEL OUTPUT:

RESULT: NO SALT DETECTED
Survey Range: {survey_depth}m to {survey_depth + 200}m

The model analyzed the seismic image and found no significant salt deposits.

Explain this result briefly."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": analysis_text}
            ],
            max_tokens=1000,
            temperature=0.1  # Very low temperature for precise, exact output
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower():
            return "Error: Rate limit exceeded. Please wait a moment and try again."
        elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            return "Error: API authentication failed. Please check the API key configuration."
        else:
            return f"Error: Unable to generate AI explanation. Details: {error_msg}"

# Page configuration
st.set_page_config(
    page_title="Salt Exploration Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Purple/Black Theme with Light Yellow Text CSS
st.markdown("""
<style>
    /* Dark purple/black theme base */
    .stApp {
        background: linear-gradient(180deg, #0a0012 0%, #1a0a2e 50%, #0a0012 100%);
    }

    .main .block-container {
        background: transparent;
        padding-top: 2rem;
    }

    /* Global text color - Light Yellow */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label {
        color: #FFFACD !important;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #FFE066 !important;
        text-align: center;
        margin-bottom: 0.3rem;
        padding-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(255, 224, 102, 0.4);
    }

    .subtitle {
        font-size: 1.1rem;
        color: #FFFACD !important;
        text-align: center;
        margin-bottom: 1.5rem;
        letter-spacing: 0.5px;
    }

    .result-card {
        background: linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: #FFFACD !important;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255,250,205,0.1);
        margin: 0.5rem 0;
        border: 1px solid rgba(255,250,205,0.2);
    }

    .result-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
        color: #FFE066 !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }

    .result-unit {
        font-size: 1.2rem;
        font-weight: 400;
        color: #FFFACD !important;
    }

    .result-label {
        font-size: 0.95rem;
        margin-top: 0.5rem;
        color: #FFFACD !important;
    }

    .salt-card {
        background: linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: #FFFACD !important;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255,250,205,0.1);
        margin: 0.5rem 0;
        border: 1px solid rgba(255,250,205,0.2);
    }

    .depth-card {
        background: linear-gradient(135deg, #2d1b4e 0%, #3d2b5e 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: #FFFACD !important;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255,250,205,0.1);
        margin: 0.5rem 0;
        border: 1px solid rgba(255,250,205,0.2);
    }

    .confidence-card {
        background: linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: #FFFACD !important;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255,250,205,0.2);
    }

    .metric-small {
        background: linear-gradient(135deg, #0a0012 0%, #1a0a2e 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FFE066;
        margin: 0.3rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }

    .metric-small-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #FFE066 !important;
    }

    .metric-small-label {
        font-size: 0.8rem;
        color: #FFFACD !important;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FFE066 !important;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3d2b5e;
        letter-spacing: 0.5px;
    }

    .info-panel {
        background: linear-gradient(135deg, #0a0012 0%, #1a0a2e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        border: 1px solid #3d2b5e;
        color: #FFFACD !important;
    }

    .info-panel p, .info-panel li, .info-panel strong {
        color: #FFFACD !important;
    }

    .warning-text {
        color: #FFE066 !important;
        background: linear-gradient(135deg, #2d1b4e 0%, #3d2b5e 100%);
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.95rem;
        border: 1px solid #FFE066;
    }

    .success-text {
        color: #FFE066 !important;
        background: linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 100%);
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.95rem;
        border: 1px solid #FFE066;
    }

    /* AI Explanation Panel */
    .ai-panel {
        background: linear-gradient(135deg, #0a0012 0%, #1a0a2e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #FFE066;
        box-shadow: 0 8px 32px rgba(255, 224, 102, 0.1), inset 0 1px 0 rgba(255,250,205,0.05);
    }

    .ai-header {
        color: #FFE066 !important;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .ai-content {
        color: #FFFACD !important;
        font-size: 0.95rem;
        line-height: 1.7;
        white-space: pre-wrap;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1a0a2e 0%, #0a0012 100%) !important;
    }

    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #FFFACD !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFE066 !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3d2b5e 0%, #5d4b7e 100%);
        color: #FFE066 !important;
        border: 1px solid #FFE066;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(255, 224, 102, 0.2);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 224, 102, 0.3);
        background: linear-gradient(135deg, #5d4b7e 0%, #7d6b9e 100%);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: transparent !important;
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed #3d2b5e;
    }

    [data-testid="stFileUploader"] > div {
        background: transparent !important;
    }

    [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small {
        color: #FFFACD !important;
        background: transparent !important;
    }

    /* File uploader dropzone */
    [data-testid="stFileUploaderDropzone"] {
        background: #1a0a2e !important;
    }

    [data-testid="stFileUploaderDropzone"] div {
        background: transparent !important;
    }

    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] small {
        color: #FFFACD !important;
    }

    /* Browse files button */
    [data-testid="stFileUploaderDropzone"] button {
        background: #1a0a2e !important;
        color: #FFFACD !important;
        border: 1px solid #3d2b5e !important;
    }

    /* Input fields */
    .stNumberInput input, .stTextInput input, .stSelectbox select {
        background: #1a0a2e !important;
        color: #FFFACD !important;
        border: 1px solid #3d2b5e !important;
        border-radius: 8px;
    }

    .stNumberInput label, .stTextInput label {
        color: #FFFACD !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #1a0a2e !important;
        color: #FFFACD !important;
        border-radius: 8px;
    }

    [data-testid="stExpander"] {
        background: #1a0a2e;
        border: 1px solid #3d2b5e;
        border-radius: 8px;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #FFE066 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #FFFACD !important;
    }

    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 100%);
        color: #FFFACD !important;
        border: 1px solid #FFE066;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #2d1b4e 0%, #3d2b5e 100%);
        border-color: #FFE066;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: #FFE066 transparent transparent transparent !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3d2b5e, #FFE066) !important;
    }

    /* Markdown text */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #FFFACD !important;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #FFE066 !important;
    }

    .stMarkdown a {
        color: #FFE066 !important;
    }

    /* Alert boxes */
    .stAlert {
        background: #1a0a2e !important;
        border: 1px solid #3d2b5e;
        color: #FFFACD !important;
    }

    .stAlert p {
        color: #FFFACD !important;
    }

    /* Horizontal rule */
    hr {
        border-color: #3d2b5e !important;
    }

    /* Tables */
    .stDataFrame, table {
        background: #1a0a2e !important;
    }

    th, td {
        color: #FFFACD !important;
        background: #1a0a2e !important;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0a0012;
    }

    ::-webkit-scrollbar-thumb {
        background: #3d2b5e;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #5d4b7e;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load trained model."""
    model_path = os.path.join(MODELS_DIR, "ensemble_latest.joblib")
    if not os.path.exists(model_path):
        return None
    ensemble = HybridEnsemble()
    ensemble.load(model_path)
    return ensemble


@st.cache_resource
def load_components():
    """Load pipeline components."""
    return {
        "decomposer": SpectralDecomposer(),
        "engineer": FeatureEngineer(),
        "confidence": DepthAdaptiveConfidence(),
        "evaluator": Evaluator(),
        "validator": SeismicImageValidator(strict_mode=True),
    }


def analyze_salt_distribution(mask, proba_map, survey_depth, image_size=101):
    """
    Analyze salt distribution and estimate depth/quantity.

    Args:
        mask: Binary salt mask
        proba_map: Probability map
        survey_depth: Starting depth of the survey (meters)
        image_size: Size of the image in pixels

    Returns:
        Analysis dictionary with depth and quantity estimates
    """
    # Assume image represents a depth range (survey_depth to survey_depth + depth_range)
    depth_range = 200  # Each image covers ~200m depth range
    meters_per_pixel = depth_range / image_size

    # Find salt regions
    salt_pixels = mask > 0.5
    salt_coverage = salt_pixels.mean()

    if salt_coverage == 0:
        return {
            "has_salt": False,
            "salt_coverage_percent": 0,
            "estimated_salt_volume": 0,
            "depth_to_first_salt": None,
            "depth_to_max_salt": None,
            "salt_thickness": 0,
            "confidence": 0,
            "recommendation": "No significant salt deposits detected in this survey area."
        }

    # Find rows that contain salt (boolean array)
    salt_rows = np.any(salt_pixels, axis=1)

    # Get indices of all rows with salt
    salt_row_indices = np.where(salt_rows)[0]

    if len(salt_row_indices) == 0:
        first_salt_row = 0
        last_salt_row = 0
        max_salt_row = 0
    else:
        # First row with salt (topmost)
        first_salt_row = salt_row_indices[0]

        # Last row with salt (bottommost)
        last_salt_row = salt_row_indices[-1]

        # Row with maximum salt concentration
        salt_per_row = salt_pixels.sum(axis=1)
        max_salt_row = np.argmax(salt_per_row)

    # Calculate depths based on actual pixel positions
    depth_to_first_salt = survey_depth + (first_salt_row * meters_per_pixel)
    depth_to_max_salt = survey_depth + (max_salt_row * meters_per_pixel)
    depth_to_bottom_salt = survey_depth + (last_salt_row * meters_per_pixel)

    # Salt thickness - actual difference between first and last salt rows
    salt_thickness = (last_salt_row - first_salt_row + 1) * meters_per_pixel

    # Estimate salt volume (relative units based on coverage)
    # Assuming 1 pixel = 1 unit area, coverage gives relative volume
    salt_area_units = salt_pixels.sum()

    # Volume estimation (assuming lateral extent similar to depth extent)
    # This is a simplified model - actual volume depends on 3D survey
    estimated_volume_relative = salt_coverage * 100  # Relative scale 0-100

    # Confidence based on probability values in salt regions
    if salt_pixels.any():
        mean_salt_probability = proba_map[salt_pixels].mean()
        confidence = mean_salt_probability
    else:
        confidence = 0

    # Generate recommendation
    if salt_coverage > 0.5:
        recommendation = f"HIGH POTENTIAL: Large salt deposit detected. Recommended drilling depth: {depth_to_first_salt:.0f}m to {depth_to_bottom_salt:.0f}m"
    elif salt_coverage > 0.2:
        recommendation = f"MODERATE POTENTIAL: Significant salt presence. Target depth range: {depth_to_first_salt:.0f}m to {depth_to_bottom_salt:.0f}m"
    elif salt_coverage > 0.05:
        recommendation = f"LOW POTENTIAL: Minor salt traces detected around {depth_to_max_salt:.0f}m depth."
    else:
        recommendation = "MINIMAL: Very limited salt presence. Further survey recommended."

    return {
        "has_salt": True,
        "salt_coverage_percent": salt_coverage * 100,
        "estimated_salt_volume": estimated_volume_relative,
        "depth_to_first_salt": depth_to_first_salt,
        "depth_to_max_salt": depth_to_max_salt,
        "depth_to_bottom_salt": depth_to_bottom_salt,
        "salt_thickness": salt_thickness,
        "salt_area_pixels": salt_area_units,
        "confidence": confidence,
        "recommendation": recommendation,
        "first_salt_row": first_salt_row,
        "max_salt_row": max_salt_row,
        "last_salt_row": last_salt_row,
    }


def find_optimal_depth(image, model, components):
    """
    Scan multiple depth ranges to find the PRECISE optimal drilling depth.
    Uses fine-grained scanning for accurate depth prediction.

    Returns:
        dict with optimal_depth, all_results for each depth scanned
    """
    # Phase 1: Coarse scan (50m increments) to find general region
    coarse_depths = list(range(0, 2001, 50))
    best_coarse_depth = 0
    best_coarse_score = 0

    # Pre-compute spectral features (same for all depths)
    spectral = components["decomposer"].compute_multi_scale_features(image)

    coarse_scans = []
    for depth in coarse_depths:
        features = components["engineer"].extract_all_features(image, depth, spectral)
        mask, proba_map = model.predict_image(features, image.shape, threshold=0.5)

        salt_coverage = (mask > 0.5).mean()
        mean_probability = proba_map[mask > 0.5].mean() if salt_coverage > 0 else 0

        # Score based on coverage and confidence
        score = (salt_coverage * 0.4 + mean_probability * 0.6)

        coarse_scans.append({
            "depth": depth,
            "coverage": salt_coverage * 100,
            "probability": mean_probability * 100,
            "score": score
        })

        if score > best_coarse_score and salt_coverage > 0.005:
            best_coarse_score = score
            best_coarse_depth = depth

    # Phase 2: Fine scan (10m increments) around the best coarse depth
    fine_start = max(0, best_coarse_depth - 50)
    fine_end = min(2000, best_coarse_depth + 50)
    fine_depths = list(range(fine_start, fine_end + 1, 10))

    best_depth = best_coarse_depth
    best_score = best_coarse_score
    best_result = None

    for depth in fine_depths:
        features = components["engineer"].extract_all_features(image, depth, spectral)
        mask, proba_map = model.predict_image(features, image.shape, threshold=0.5)

        salt_coverage = (mask > 0.5).mean()
        mean_probability = proba_map[mask > 0.5].mean() if salt_coverage > 0 else 0

        score = (salt_coverage * 0.4 + mean_probability * 0.6)

        if score > best_score and salt_coverage > 0.005:
            best_score = score
            best_depth = depth
            best_result = {
                "mask": mask,
                "proba_map": proba_map,
                "coverage": salt_coverage,
                "probability": mean_probability
            }

    # If no salt found at all, use default and still get a result
    if best_result is None:
        features = components["engineer"].extract_all_features(image, best_depth, spectral)
        mask, proba_map = model.predict_image(features, image.shape, threshold=0.5)
        best_result = {
            "mask": mask,
            "proba_map": proba_map,
            "coverage": (mask > 0.5).mean(),
            "probability": 0
        }

    return {
        "optimal_depth": best_depth,
        "best_result": best_result,
        "all_scans": coarse_scans,
        "spectral": spectral
    }


def run_analysis(image, model, components):
    """Run complete salt analysis pipeline with automatic depth prediction."""

    progress = st.progress(0)
    status = st.empty()

    status.text("Scanning depth ranges to find optimal salt deposits...")
    progress.progress(10)

    # Find optimal depth by scanning multiple depths
    status.text("AI predicting optimal drilling depth...")
    progress.progress(30)
    depth_scan = find_optimal_depth(image, model, components)
    optimal_depth = depth_scan["optimal_depth"]

    # Now run full analysis at optimal depth
    status.text(f"Analyzing at predicted depth: {optimal_depth}m...")
    progress.progress(50)

    # Use pre-computed spectral features
    spectral = depth_scan["spectral"]

    # Feature extraction at optimal depth
    status.text("Extracting geological features...")
    progress.progress(60)
    features = components["engineer"].extract_all_features(image, optimal_depth, spectral)

    # Prediction
    status.text("Running salt detection model...")
    progress.progress(75)
    mask, proba_map = model.predict_image(features, image.shape, threshold=0.5)

    # Confidence
    status.text("Computing confidence scores...")
    progress.progress(85)
    conf_map, _ = components["confidence"].compute_confidence_score(proba_map, optimal_depth)

    # Analysis - use actual mask from this specific image
    status.text("Analyzing salt distribution...")
    progress.progress(95)
    analysis = analyze_salt_distribution(mask, proba_map, optimal_depth)

    # Add debug info to verify mask varies per image
    analysis["mask_stats"] = {
        "total_salt_pixels": int((mask > 0.5).sum()),
        "unique_proba_values": len(np.unique(proba_map.round(3))),
        "mean_proba": float(proba_map.mean()),
        "image_hash": int(image.sum() * 1000) % 100000  # Simple hash to verify different images
    }

    progress.progress(100)
    status.text("Analysis complete.")
    time.sleep(0.3)
    progress.empty()
    status.empty()

    return {
        "mask": mask,
        "probabilities": proba_map,
        "confidence": conf_map,
        "analysis": analysis,
        "predicted_depth": optimal_depth,
        "depth_scan": depth_scan["all_scans"],
    }


def create_depth_profile_plot(image, mask, proba_map, analysis, survey_depth):
    """Create depth profile visualization with proper layout and no overlapping."""

    # Use a larger figure with more spacing
    fig = plt.figure(figsize=(18, 14))

    # Create grid with better spacing - 3 rows, 3 columns
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35,
                          height_ratios=[1, 1, 0.8],
                          left=0.06, right=0.94, top=0.94, bottom=0.06)

    depth_range = 200

    # Row 1: Basic visualizations (3 plots)

    # 1. Seismic Image with depth scale
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='seismic', aspect='auto',
               extent=[0, 100, survey_depth + depth_range, survey_depth])
    ax1.set_ylabel('Depth (m)', fontsize=10)
    ax1.set_xlabel('Lateral Position (%)', fontsize=10)
    ax1.set_title('Seismic Survey Image', fontsize=12, fontweight='bold', pad=10)

    # 2. Salt Probability Map
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(proba_map, cmap='YlOrRd', aspect='auto',
                     extent=[0, 100, survey_depth + depth_range, survey_depth], vmin=0, vmax=1)
    ax2.set_ylabel('Depth (m)', fontsize=10)
    ax2.set_xlabel('Lateral Position (%)', fontsize=10)
    ax2.set_title('Salt Probability Map', fontsize=12, fontweight='bold', pad=10)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Probability', fontsize=9)

    # 3. Detected Salt Bodies
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(image, cmap='gray', aspect='auto', alpha=0.5,
               extent=[0, 100, survey_depth + depth_range, survey_depth])
    salt_overlay = np.ma.masked_where(mask < 0.5, mask)
    ax3.imshow(salt_overlay, cmap='Greens', aspect='auto', alpha=0.7,
               extent=[0, 100, survey_depth + depth_range, survey_depth], vmin=0, vmax=1)
    ax3.set_ylabel('Depth (m)', fontsize=10)
    ax3.set_xlabel('Lateral Position (%)', fontsize=10)
    ax3.set_title('Detected Salt Bodies', fontsize=12, fontweight='bold', pad=10)

    # Add depth markers if salt found
    if analysis["has_salt"]:
        ax3.axhline(y=analysis["depth_to_first_salt"], color='red', linestyle='--',
                    linewidth=2, label='First Salt')
        ax3.axhline(y=analysis["depth_to_max_salt"], color='blue', linestyle='--',
                    linewidth=2, label='Max Salt')
        ax3.legend(loc='lower right', fontsize=8, framealpha=0.9)

    # Row 2: Analysis plots (3 plots)

    # 4. Depth Profile (Salt concentration vs Depth)
    ax4 = fig.add_subplot(gs[1, 0])
    salt_per_row = (mask > 0.5).sum(axis=1) / mask.shape[1] * 100
    depths = np.linspace(survey_depth, survey_depth + depth_range, len(salt_per_row))
    ax4.fill_betweenx(depths, 0, salt_per_row, color='green', alpha=0.4)
    ax4.plot(salt_per_row, depths, color='darkgreen', linewidth=2)
    ax4.set_xlim(0, max(100, salt_per_row.max() * 1.1))
    ax4.set_ylim(survey_depth + depth_range, survey_depth)
    ax4.set_xlabel('Salt Coverage (%)', fontsize=10)
    ax4.set_ylabel('Depth (m)', fontsize=10)
    ax4.set_title('Salt vs Depth Profile', fontsize=12, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3)

    if analysis["has_salt"]:
        ax4.axhline(y=analysis["depth_to_max_salt"], color='blue', linestyle='--',
                    linewidth=1.5, label=f'Peak: {analysis["depth_to_max_salt"]:.0f}m')
        ax4.legend(loc='lower right', fontsize=8)

    # 5. Cross-section view with drilling zone
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(image, cmap='gray', aspect='auto', alpha=0.4,
               extent=[0, 100, survey_depth + depth_range, survey_depth])

    # Color-coded salt by confidence levels
    if mask.any():
        # High confidence (>0.7)
        high_conf = (proba_map >= 0.7) & (mask > 0.5)
        if high_conf.any():
            high_masked = np.ma.masked_where(~high_conf, np.ones_like(mask))
            ax5.imshow(high_masked, cmap='Greens', aspect='auto', alpha=0.8,
                       extent=[0, 100, survey_depth + depth_range, survey_depth])

        # Medium confidence (0.4-0.7)
        med_conf = (proba_map >= 0.4) & (proba_map < 0.7) & (mask > 0.5)
        if med_conf.any():
            med_masked = np.ma.masked_where(~med_conf, np.ones_like(mask))
            ax5.imshow(med_masked, cmap='YlOrBr', aspect='auto', alpha=0.7,
                       extent=[0, 100, survey_depth + depth_range, survey_depth])

        # Low confidence (<0.4)
        low_conf = (proba_map < 0.4) & (mask > 0.5)
        if low_conf.any():
            low_masked = np.ma.masked_where(~low_conf, np.ones_like(mask))
            ax5.imshow(low_masked, cmap='Reds', aspect='auto', alpha=0.6,
                       extent=[0, 100, survey_depth + depth_range, survey_depth])

    ax5.set_ylabel('Depth (m)', fontsize=10)
    ax5.set_xlabel('Lateral Position (%)', fontsize=10)
    ax5.set_title('Confidence-Coded Salt Bodies', fontsize=12, fontweight='bold', pad=10)

    # Add drilling zone recommendation
    if analysis["has_salt"] and analysis["salt_thickness"] > 0:
        rect = mpatches.Rectangle((25, analysis["depth_to_first_salt"]), 50,
                                   analysis["salt_thickness"],
                                   linewidth=3, edgecolor='red', facecolor='none',
                                   linestyle='--', label='Drilling Zone')
        ax5.add_patch(rect)
        ax5.legend(loc='lower right', fontsize=8, framealpha=0.9)

    # 6. Confidence histogram
    ax6 = fig.add_subplot(gs[1, 2])
    if mask.any() and (mask > 0.5).any():
        salt_probs = proba_map[mask > 0.5]
        ax6.hist(salt_probs, bins=20, color='teal', alpha=0.7, edgecolor='black')
        ax6.axvline(x=salt_probs.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {salt_probs.mean():.2f}')
        ax6.set_xlabel('Prediction Confidence', fontsize=10)
        ax6.set_ylabel('Pixel Count', fontsize=10)
        ax6.set_title('Confidence Distribution', fontsize=12, fontweight='bold', pad=10)
        ax6.legend(loc='upper left', fontsize=8)
        ax6.set_xlim(0, 1)
    else:
        ax6.text(0.5, 0.5, 'No salt detected', ha='center', va='center',
                 fontsize=12, transform=ax6.transAxes)
        ax6.set_title('Confidence Distribution', fontsize=12, fontweight='bold', pad=10)

    # Row 3: Summary panel (spans all columns)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')

    if analysis["has_salt"]:
        # Create a formatted summary box
        summary_lines = [
            "═" * 90,
            "                              SALT EXPLORATION ANALYSIS SUMMARY",
            "═" * 90,
            "",
            f"  DEPTH ANALYSIS                          │  QUANTITY ESTIMATION",
            f"  ────────────────────────────────────────┼────────────────────────────────────────",
            f"  • AI Predicted Survey Range: {survey_depth:.0f}m - {survey_depth + 200:.0f}m    │  • Salt Coverage:     {analysis['salt_coverage_percent']:.1f}%",
            f"  • First Salt Detected:       {analysis['depth_to_first_salt']:.0f}m           │  • Volume Index:      {analysis['estimated_salt_volume']:.1f}/100",
            f"  • Peak Salt Concentration:   {analysis['depth_to_max_salt']:.0f}m           │  • Salt Area:         {analysis['salt_area_pixels']:.0f} pixels",
            f"  • Salt Body Thickness:       {analysis['salt_thickness']:.0f}m            │  • Model Confidence:  {analysis['confidence']*100:.1f}%",
            "",
            "  " + "─" * 88,
            f"  RECOMMENDATION: {analysis['recommendation']}",
            "═" * 90,
        ]
        summary_text = "\n".join(summary_lines)
    else:
        summary_lines = [
            "═" * 90,
            "                              SALT EXPLORATION ANALYSIS SUMMARY",
            "═" * 90,
            "",
            f"  Survey Depth Range: {survey_depth:.0f}m - {survey_depth + 200:.0f}m",
            "",
            "  RESULT: No significant salt deposits detected in this survey area",
            "",
            f"  RECOMMENDATION: {analysis['recommendation']}",
            "═" * 90,
        ]
        summary_text = "\n".join(summary_lines)

    ax7.text(0.5, 0.5, summary_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5dc', alpha=0.9,
                       edgecolor='#8b7355', linewidth=2))

    return fig


def main():
    # Header
    st.markdown('<h1 class="main-title">Salt Exploration Analysis System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered depth prediction and salt quantity estimation from seismic survey data</p>', unsafe_allow_html=True)

    # Load model
    model = load_model()
    components = load_components()

    if model is None:
        st.error("Model not found. Please train the model first: python train.py")
        st.stop()

    # Sidebar
    st.sidebar.markdown("### AI-Powered Analysis")
    st.sidebar.markdown("---")

    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 100%);
                padding: 1rem; border-radius: 10px; border: 1px solid #FFE066; margin-bottom: 1rem;">
        <p style="color: #FFE066; font-weight: bold; margin-bottom: 0.5rem;">Automatic Depth Prediction</p>
        <p style="color: #FFFACD; font-size: 0.85rem; margin: 0;">
            The AI model automatically scans multiple depth ranges (0-2000m) to find the optimal drilling depth with maximum salt concentration.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This system analyzes seismic images to:
    - **Auto-detect** optimal drilling depth
    - Detect salt body presence
    - Calculate salt quantity
    - Provide drilling recommendations
    - AI-powered geological interpretation
    """)

    # Main layout
    col_input, col_results = st.columns([1, 1.8])

    with col_input:
        st.markdown('<p class="section-header">Upload Seismic Data</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Seismic Survey Image",
            type=["png", "jpg", "jpeg"],
            help="Upload a seismic survey image for salt analysis"
        )

        # Optional ground truth for validation
        with st.expander("Advanced: Upload Ground Truth (for accuracy validation)"):
            uploaded_gt = st.file_uploader(
                "Ground Truth Mask",
                type=["png", "jpg", "jpeg"],
                key="gt_upload"
            )

        image = None
        gt_mask = None

        if uploaded_file:
            pil_image = Image.open(uploaded_file)
            if pil_image.mode != 'L':
                pil_image = pil_image.convert('L')
            pil_image = pil_image.resize((101, 101), Image.Resampling.LANCZOS)
            image = np.array(pil_image, dtype=np.float32) / 255.0

            # Validate that this is a seismic image
            validator = components["validator"]
            is_valid, validation_metrics, validation_reason = validator.validate(image)
            confidence = validation_metrics.get("validation_confidence", 0)

            if not is_valid:
                # Completely rejected - likely blank or severely malformed
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4a1a1a 0%, #2d0a0a 100%);
                            padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                            border: 2px solid #ff4444; color: #ffcccc;">
                    <p style="font-weight: 700; margin-bottom: 0.5rem; color: #ff6666; font-size: 1.1rem;">
                        Invalid Image
                    </p>
                    <p style="font-size: 0.95rem; color: #ffcccc; margin-bottom: 1rem;">
                        {validation_reason}
                    </p>
                    <p style="font-size: 0.85rem; color: #ffaaaa;">
                        This system is designed specifically for seismic survey images from salt exploration.
                        Please upload a valid seismic image (grayscale, showing subsurface reflectors).
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Show the rejected image for reference
                fig_rejected, ax_rejected = plt.subplots(figsize=(4, 4))
                ax_rejected.imshow(image, cmap='gray')
                ax_rejected.set_title('Uploaded Image (Rejected)', fontweight='bold', color='red')
                ax_rejected.axis('off')
                st.pyplot(fig_rejected)
                plt.close()

                image = None  # Prevent further processing

            elif confidence < 0.5:
                # Low confidence - warn user but allow processing
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4a3a1a 0%, #2d2a0a 100%);
                            padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                            border: 2px solid #ffaa44; color: #ffeecc;">
                    <p style="font-weight: 700; margin-bottom: 0.5rem; color: #ffcc66; font-size: 1.1rem;">
                        Warning: Low Seismic Confidence ({confidence:.0%})
                    </p>
                    <p style="font-size: 0.95rem; color: #ffeecc; margin-bottom: 0.5rem;">
                        This image may not be a seismic survey image. Results may not be accurate.
                    </p>
                    <p style="font-size: 0.85rem; color: #ffddaa;">
                        This tool is designed for TGS Salt Identification seismic images.
                        For best results, use grayscale seismic survey images showing subsurface geological layers.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Display input with warning
                fig_in, ax_in = plt.subplots(figsize=(5, 5))
                ax_in.imshow(image, cmap='seismic')
                ax_in.set_title(f'Input Image (Low Confidence: {confidence:.0%})', fontweight='bold')
                ax_in.axis('off')
                st.pyplot(fig_in)
                plt.close()

            else:
                # Good confidence - valid seismic image
                conf_label = "High" if confidence >= 0.7 else "Medium"
                st.markdown(f"""
                <div class="info-panel">
                    <p><strong>{conf_label} confidence seismic image</strong> ({confidence:.0%})</p>
                    <p>AI will automatically predict optimal drilling depth</p>
                </div>
                """, unsafe_allow_html=True)

                # Display input
                fig_in, ax_in = plt.subplots(figsize=(5, 5))
                ax_in.imshow(image, cmap='seismic')
                ax_in.set_title('Seismic Survey Input', fontweight='bold')
                ax_in.axis('off')
                st.pyplot(fig_in)
                plt.close()

            if uploaded_gt:
                pil_gt = Image.open(uploaded_gt)
                if pil_gt.mode != 'L':
                    pil_gt = pil_gt.convert('L')
                pil_gt = pil_gt.resize((101, 101), Image.Resampling.LANCZOS)
                gt_mask = np.array(pil_gt, dtype=np.float32) / 255.0
                gt_mask = (gt_mask > 0.5).astype(np.float32)

        # Analyze button
        if image is not None:
            if st.button("Analyze for Salt Deposits", use_container_width=True):
                results = run_analysis(image, model, components)

                st.session_state["results"] = results
                st.session_state["image"] = image
                st.session_state["gt_mask"] = gt_mask
                st.session_state["survey_depth"] = results["predicted_depth"]

    with col_results:
        st.markdown('<p class="section-header">Analysis Results</p>', unsafe_allow_html=True)

        if "results" in st.session_state:
            results = st.session_state["results"]
            analysis = results["analysis"]
            survey_depth = st.session_state["survey_depth"]

            # Show AI-predicted depth prominently
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2d1b4e 0%, #3d2b5e 100%);
                        padding: 0.8rem 1rem; border-radius: 10px; border: 1px solid #FFE066;
                        margin-bottom: 1rem; text-align: center;">
                <span style="color: #FFE066; font-size: 0.9rem;">AI Predicted Survey Depth:</span>
                <span style="color: #FFFACD; font-size: 1.1rem; font-weight: bold; margin-left: 0.5rem;">
                    {survey_depth}m - {survey_depth + 200}m
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Debug: Show image-specific stats to verify model variation
            if "mask_stats" in analysis:
                stats = analysis["mask_stats"]
                st.markdown(f"""
                <div style="background: #0a0012; padding: 0.5rem; border-radius: 5px;
                            font-size: 0.75rem; color: #888; margin-bottom: 0.5rem;">
                    Image ID: {stats['image_hash']} | Salt Pixels: {stats['total_salt_pixels']} | Unique Values: {stats['unique_proba_values']}
                </div>
                """, unsafe_allow_html=True)

            if analysis["has_salt"]:
                # Main results - Depth and Quantity
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown(f"""
                    <div class="depth-card">
                        <p class="result-value">{analysis['depth_to_first_salt']:.0f}<span class="result-unit"> m</span></p>
                        <p class="result-label">Recommended Drilling Depth</p>
                    </div>
                    """, unsafe_allow_html=True)

                with c2:
                    st.markdown(f"""
                    <div class="salt-card">
                        <p class="result-value">{analysis['salt_coverage_percent']:.1f}<span class="result-unit">%</span></p>
                        <p class="result-label">Salt Content in Survey Area</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Secondary metrics
                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)

                with m1:
                    st.markdown(f"""
                    <div class="metric-small">
                        <p class="metric-small-value">{analysis['depth_to_max_salt']:.0f} m</p>
                        <p class="metric-small-label">Peak Salt Depth</p>
                    </div>
                    """, unsafe_allow_html=True)

                with m2:
                    st.markdown(f"""
                    <div class="metric-small">
                        <p class="metric-small-value">{analysis['salt_thickness']:.0f} m</p>
                        <p class="metric-small-label">Salt Layer Thickness</p>
                    </div>
                    """, unsafe_allow_html=True)

                with m3:
                    st.markdown(f"""
                    <div class="metric-small">
                        <p class="metric-small-value">{analysis['estimated_salt_volume']:.1f}</p>
                        <p class="metric-small-label">Volume Index (0-100)</p>
                    </div>
                    """, unsafe_allow_html=True)

                with m4:
                    st.markdown(f"""
                    <div class="metric-small">
                        <p class="metric-small-value">{analysis['confidence']*100:.0f}%</p>
                        <p class="metric-small-label">Model Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Recommendation
                st.markdown("<br>", unsafe_allow_html=True)
                if "HIGH" in analysis["recommendation"]:
                    st.markdown(f'<div class="success-text"><strong>Recommendation:</strong> {analysis["recommendation"]}</div>', unsafe_allow_html=True)
                elif "MODERATE" in analysis["recommendation"]:
                    st.markdown(f'<div class="warning-text"><strong>Recommendation:</strong> {analysis["recommendation"]}</div>', unsafe_allow_html=True)
                else:
                    st.info(f"**Recommendation:** {analysis['recommendation']}")

            else:
                st.markdown(f"""
                <div class="result-card">
                    <p class="result-value">No Salt Detected</p>
                    <p class="result-label">in survey depth {survey_depth}m - {survey_depth + 200}m</p>
                </div>
                """, unsafe_allow_html=True)
                st.info(analysis["recommendation"])

            # Accuracy validation if ground truth provided
            gt_mask = st.session_state.get("gt_mask")
            if gt_mask is not None:
                st.markdown('<p class="section-header">Accuracy Validation</p>', unsafe_allow_html=True)
                metrics = components["evaluator"].compute_pixel_metrics(results["mask"], gt_mask)

                acc_cols = st.columns(4)
                with acc_cols[0]:
                    st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                with acc_cols[1]:
                    iou = components["evaluator"].compute_iou(results["mask"], gt_mask)
                    st.metric("IoU Score", f"{iou:.3f}")
                with acc_cols[2]:
                    st.metric("Precision", f"{metrics['precision']*100:.1f}%")
                with acc_cols[3]:
                    st.metric("Recall", f"{metrics['recall']*100:.1f}%")

            # AI Explanation Section
            st.markdown("---")
            st.markdown('<p class="section-header">AI Expert Geological Analysis</p>', unsafe_allow_html=True)

            st.markdown("""
            <div class="info-panel" style="margin-bottom: 1rem;">
                <span style="color: #FFE066;">Powered by Groq LLaMA 3.3</span> - Get detailed drilling recommendations,
                similar geological locations worldwide, and efficient extraction techniques.
            </div>
            """, unsafe_allow_html=True)

            if st.button("Generate Expert Geological Interpretation", use_container_width=True, key="ai_explain"):
                with st.spinner("Consulting AI geological expert..."):
                    ai_explanation = generate_ai_explanation(analysis, survey_depth)
                    st.session_state["ai_explanation"] = ai_explanation

            if "ai_explanation" in st.session_state:
                explanation = st.session_state["ai_explanation"]

                # Check if it's an error message
                if "error" in explanation.lower()[:50]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 100%);
                                padding: 1.5rem; border-radius: 12px; margin-top: 1rem;
                                border: 1px solid #FFE066; color: #FFFACD;">
                        <p style="font-weight: 700; margin-bottom: 0.5rem; color: #FFE066;">AI Service Issue</p>
                        <p style="font-size: 0.9rem; color: #FFFACD;">{explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ai-panel">
                        <div class="ai-header">
                            Groq AI Geological Consultant
                        </div>
                        <div class="ai-content">{explanation}</div>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="info-panel">
                <p><strong>How to use:</strong></p>
                <ol>
                    <li>Upload a seismic survey image</li>
                    <li>Click "Analyze for Salt Deposits"</li>
                    <li>AI automatically predicts optimal depth</li>
                </ol>
                <p><strong>You will get:</strong></p>
                <ul>
                    <li>AI-predicted optimal drilling depth</li>
                    <li>Estimated salt quantity</li>
                    <li>Salt layer thickness</li>
                    <li>Confidence score</li>
                    <li>Expert geological interpretation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # Visualization section
    if "results" in st.session_state:
        st.markdown("---")
        st.markdown('<p class="section-header">Detailed Analysis Visualization</p>', unsafe_allow_html=True)

        results = st.session_state["results"]
        image = st.session_state["image"]
        survey_depth = st.session_state["survey_depth"]

        fig = create_depth_profile_plot(
            image,
            results["mask"],
            results["probabilities"],
            results["analysis"],
            survey_depth
        )
        st.pyplot(fig)
        plt.close()

        # Export options
        st.markdown("---")
        st.markdown('<p class="section-header">Export Results</p>', unsafe_allow_html=True)

        col_dl1, col_dl2, col_dl3 = st.columns(3)

        with col_dl1:
            mask_img = Image.fromarray((results["mask"] * 255).astype(np.uint8))
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            st.download_button(
                "Download Salt Map",
                data=buf.getvalue(),
                file_name="salt_detection_map.png",
                mime="image/png",
                use_container_width=True
            )

        with col_dl2:
            # Create report text
            analysis = results["analysis"]
            report = f"""SALT EXPLORATION REPORT
{'='*50}
Survey Depth: {survey_depth}m - {survey_depth + 200}m
Analysis Date: {time.strftime('%Y-%m-%d %H:%M')}

FINDINGS:
- Salt Detected: {'Yes' if analysis['has_salt'] else 'No'}
"""
            if analysis['has_salt']:
                report += f"""- First Salt Depth: {analysis['depth_to_first_salt']:.0f}m
- Maximum Salt Depth: {analysis['depth_to_max_salt']:.0f}m
- Salt Thickness: {analysis['salt_thickness']:.0f}m
- Salt Coverage: {analysis['salt_coverage_percent']:.1f}%
- Volume Index: {analysis['estimated_salt_volume']:.1f}/100
- Confidence: {analysis['confidence']*100:.0f}%

RECOMMENDATION:
{analysis['recommendation']}
"""
            st.download_button(
                "Download Report (TXT)",
                data=report,
                file_name="salt_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col_dl3:
            proba_img = Image.fromarray((results["probabilities"] * 255).astype(np.uint8))
            buf2 = io.BytesIO()
            proba_img.save(buf2, format="PNG")
            st.download_button(
                "Download Probability Map",
                data=buf2.getvalue(),
                file_name="salt_probability_map.png",
                mime="image/png",
                use_container_width=True
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem 0;'>
        <p style='color: #FFE066; font-size: 0.9rem; margin-bottom: 0.5rem;'>
            Salt Exploration Analysis System
        </p>
        <p style='color: #FFFACD; font-size: 0.8rem;'>
            Depth-Adaptive Spectral Segmentation | CWT + XGBoost/RF Ensemble | Groq AI Integration
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
