"""
Spectral Decomposition Module using Continuous Wavelet Transform (CWT)
Extracts multi-frequency features from seismic images
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
import pywt
from typing import List, Tuple, Dict, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPECTRAL_CONFIG, IMG_HEIGHT, IMG_WIDTH


class SpectralDecomposer:
    """
    Spectral decomposition of seismic images using CWT.
    Extracts frequency-dependent features at multiple scales.
    """

    def __init__(self, config: Dict = None):
        self.config = config or SPECTRAL_CONFIG
        self.wavelet = self.config["wavelet"]
        self.frequencies = self.config["frequencies"]
        self.scales = self.config["scales"]
        self.sampling_rate = self.config["sampling_rate"]

        # Pre-compute scale-frequency mapping
        self._compute_scale_mapping()

    def _compute_scale_mapping(self) -> None:
        """Compute scales corresponding to target frequencies."""
        # For Morlet wavelet, scale = sampling_rate / (2 * pi * frequency)
        self.frequency_scales = {}
        for freq in self.frequencies:
            scale = self.sampling_rate / (2 * np.pi * freq / 5)  # Morlet center freq ~5
            self.frequency_scales[freq] = scale

    def cwt_1d(self, trace: np.ndarray, scales: np.ndarray = None) -> np.ndarray:
        """
        Apply 1D CWT to a single trace using PyWavelets.

        Args:
            trace: 1D signal array
            scales: Wavelet scales to use

        Returns:
            CWT coefficients (n_scales, n_samples)
        """
        if scales is None:
            scales = np.array(self.scales)

        # Use PyWavelets for CWT with Morlet wavelet
        coefficients, _ = pywt.cwt(trace, scales, 'morl')
        return np.abs(coefficients)

    def decompose_image_1d(
        self, image: np.ndarray, direction: str = "vertical"
    ) -> np.ndarray:
        """
        Apply CWT along one direction of the image.

        Args:
            image: 2D image array (H, W)
            direction: 'vertical' or 'horizontal'

        Returns:
            Decomposed features (n_scales, H, W)
        """
        scales = np.array(self.scales)
        n_scales = len(scales)
        h, w = image.shape

        if direction == "vertical":
            result = np.zeros((n_scales, h, w), dtype=np.float32)
            for col in range(w):
                trace = image[:, col]
                result[:, :, col] = self.cwt_1d(trace, scales)
        else:  # horizontal
            result = np.zeros((n_scales, h, w), dtype=np.float32)
            for row in range(h):
                trace = image[row, :]
                coeffs = self.cwt_1d(trace, scales)
                result[:, row, :] = coeffs

        return result

    def decompose_image_2d(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Full 2D spectral decomposition combining vertical and horizontal.

        Args:
            image: 2D image array (H, W)

        Returns:
            Dictionary with spectral features
        """
        # Vertical decomposition (along depth axis - most important for seismic)
        vertical_coeffs = self.decompose_image_1d(image, direction="vertical")

        # Horizontal decomposition (along lateral axis)
        horizontal_coeffs = self.decompose_image_1d(image, direction="horizontal")

        # Combined features
        combined_coeffs = (vertical_coeffs + horizontal_coeffs) / 2

        return {
            "vertical": vertical_coeffs,
            "horizontal": horizontal_coeffs,
            "combined": combined_coeffs,
        }

    def extract_frequency_bands(
        self, image: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract specific frequency band features.

        Args:
            image: 2D image array (H, W)

        Returns:
            Dictionary with frequency band images
        """
        frequency_bands = {}

        # Use bandpass filtering for specific frequency bands
        for freq in self.frequencies:
            # Create bandpass filter
            low_freq = freq * 0.8
            high_freq = freq * 1.2

            # Normalize frequencies
            nyq = self.sampling_rate / 2
            low = max(low_freq / nyq, 0.01)
            high = min(high_freq / nyq, 0.99)

            # Design butterworth bandpass filter
            try:
                b, a = signal.butter(3, [low, high], btype="band")

                # Apply filter along vertical axis (depth)
                filtered_vertical = np.zeros_like(image)
                for col in range(image.shape[1]):
                    filtered_vertical[:, col] = signal.filtfilt(b, a, image[:, col])

                frequency_bands[f"{freq}Hz"] = np.abs(filtered_vertical)
            except ValueError:
                # Fallback to gaussian filter at appropriate scale
                sigma = self.sampling_rate / (2 * np.pi * freq)
                frequency_bands[f"{freq}Hz"] = gaussian_filter(image, sigma=sigma)

        return frequency_bands

    def compute_instantaneous_attributes(
        self, image: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute instantaneous seismic attributes using Hilbert transform.

        Args:
            image: 2D image array (H, W)

        Returns:
            Dictionary with instantaneous attributes
        """
        h, w = image.shape

        # Initialize output arrays
        inst_amplitude = np.zeros_like(image)
        inst_phase = np.zeros_like(image)
        inst_frequency = np.zeros_like(image)

        for col in range(w):
            trace = image[:, col]

            # Analytic signal via Hilbert transform
            analytic = signal.hilbert(trace)

            # Instantaneous amplitude (envelope)
            inst_amplitude[:, col] = np.abs(analytic)

            # Instantaneous phase
            inst_phase[:, col] = np.unwrap(np.angle(analytic))

            # Instantaneous frequency (derivative of phase)
            inst_frequency[1:-1, col] = np.diff(inst_phase[:, col], n=1)[:-1] / (
                2 * np.pi / self.sampling_rate
            )

        return {
            "instantaneous_amplitude": inst_amplitude,
            "instantaneous_phase": inst_phase,
            "instantaneous_frequency": np.clip(inst_frequency, -50, 50),  # Clip outliers
        }

    def compute_multi_scale_features(
        self, image: np.ndarray
    ) -> np.ndarray:
        """
        Compute multi-scale spectral features for the entire image.

        Args:
            image: 2D image array (H, W)

        Returns:
            Feature array (n_features, H, W)
        """
        features = []

        # 1. CWT decomposition at multiple scales
        cwt_features = self.decompose_image_2d(image)
        for key in ["vertical", "horizontal", "combined"]:
            for scale_idx in range(len(self.scales)):
                features.append(cwt_features[key][scale_idx])

        # 2. Frequency band features
        freq_bands = self.extract_frequency_bands(image)
        for freq in self.frequencies:
            features.append(freq_bands[f"{freq}Hz"])

        # 3. Instantaneous attributes
        inst_attrs = self.compute_instantaneous_attributes(image)
        features.append(inst_attrs["instantaneous_amplitude"])
        features.append(inst_attrs["instantaneous_phase"])
        features.append(inst_attrs["instantaneous_frequency"])

        return np.array(features, dtype=np.float32)

    def decompose_batch(
        self, images: np.ndarray, verbose: bool = True
    ) -> np.ndarray:
        """
        Apply spectral decomposition to a batch of images.

        Args:
            images: Array of shape (N, H, W)
            verbose: Whether to print progress

        Returns:
            Feature array (N, n_features, H, W)
        """
        n_samples = images.shape[0]

        # Get number of features from first image
        sample_features = self.compute_multi_scale_features(images[0])
        n_features = sample_features.shape[0]

        all_features = np.zeros(
            (n_samples, n_features, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32
        )
        all_features[0] = sample_features

        for i in range(1, n_samples):
            if verbose and (i + 1) % 200 == 0:
                print(f"Spectral decomposition: {i + 1}/{n_samples}")
            all_features[i] = self.compute_multi_scale_features(images[i])

        if verbose:
            print(f"Spectral features shape: {all_features.shape}")
            print(f"Features per image: {n_features}")

        return all_features

    def get_feature_names(self) -> List[str]:
        """Get names of all spectral features."""
        names = []

        # CWT features
        for direction in ["vertical", "horizontal", "combined"]:
            for scale in self.scales:
                names.append(f"cwt_{direction}_scale{scale}")

        # Frequency band features
        for freq in self.frequencies:
            names.append(f"freq_band_{freq}Hz")

        # Instantaneous attributes
        names.extend([
            "instantaneous_amplitude",
            "instantaneous_phase",
            "instantaneous_frequency",
        ])

        return names


if __name__ == "__main__":
    # Test the spectral decomposer
    from data_loader import DataLoader

    print("Loading test image...")
    loader = DataLoader()
    train_ids = loader.get_train_ids()
    test_image = loader.load_image(train_ids[0])

    print(f"Image shape: {test_image.shape}")

    decomposer = SpectralDecomposer()

    print("\n--- Testing CWT decomposition ---")
    cwt_features = decomposer.decompose_image_2d(test_image)
    print(f"Vertical CWT shape: {cwt_features['vertical'].shape}")
    print(f"Horizontal CWT shape: {cwt_features['horizontal'].shape}")

    print("\n--- Testing frequency band extraction ---")
    freq_bands = decomposer.extract_frequency_bands(test_image)
    for freq, band in freq_bands.items():
        print(f"{freq} band shape: {band.shape}")

    print("\n--- Testing instantaneous attributes ---")
    inst_attrs = decomposer.compute_instantaneous_attributes(test_image)
    for name, attr in inst_attrs.items():
        print(f"{name} shape: {attr.shape}")

    print("\n--- Testing full feature extraction ---")
    all_features = decomposer.compute_multi_scale_features(test_image)
    print(f"Total features: {all_features.shape}")

    print("\n--- Feature names ---")
    feature_names = decomposer.get_feature_names()
    print(f"Number of features: {len(feature_names)}")
    for name in feature_names[:5]:
        print(f"  - {name}")
    print("  ...")
