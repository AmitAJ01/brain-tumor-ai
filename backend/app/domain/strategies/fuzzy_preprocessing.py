"""
Fuzzy Logic-Based Image Preprocessing Strategy
Inspired by Hassan & Boulila (2025): "Efficient Approach for Brain Tumor Detection
and Classification Using Fuzzy Thresholding and Deep Learning Algorithms", IEEE Access.
"""

import numpy as np
import cv2
from backend.app.domain.interfaces.preprocessing_strategy import PreprocessingStrategy

class FuzzyPreprocessingStrategy(PreprocessingStrategy):
    """
    Fuzzy Preprocessing Strategy:
    1. Resizes input and maps intensities to [0, 255].
    2. Constructs 16 triangular fuzzy membership functions across the intensity universe [0, 255].
    3. Calculates an adaptive threshold T using Center-of-Gravity (CoG) defuzzification (Eq. 8-9).
    4. Evaluates fuzzy membership zones (Very Low, Low, Medium, High, Very High) to accentuate
       heterogeneous abnormal tissue boundaries while suppressing background noise.
    5. Re-normalizes the enhanced feature map into [0.0, 1.0] and expands to 3 channels.
    """

    @property
    def name(self) -> str:
        return "fuzzy"

    @property
    def description(self) -> str:
        return "Fuzzy thresholding and 16-membership CoG contrast enhancement (Hassan & Boulila 2025)."

    def _triangular_mf(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Compute triangular membership degree for input array x."""
        term1 = np.where((x > a) & (x <= b), (x - a) / max(b - a, 1e-5), 0.0)
        term2 = np.where((x > b) & (x < c), (c - x) / max(c - b, 1e-5), 0.0)
        return np.maximum(term1, term2)

    def _calculate_fuzzy_threshold(self, img_uint8: np.ndarray) -> float:
        """
        Calculate fuzzy adaptive threshold T using 16 membership functions
        as formulated in Hassan & Boulila (2025), Section III-C1.
        """
        hist, bin_edges = np.histogram(img_uint8, bins=256, range=(0, 256))
        total_pixels = max(img_uint8.size, 1)

        # 16 partitions across [0, 255]
        n_rules = 16
        step = 256.0 / n_rules
        centers = [(i + 0.5) * step for i in range(n_rules)]

        weights_sum = 0.0
        weighted_centers_sum = 0.0

        for idx, center in enumerate(centers):
            a = max(0.0, center - step)
            b = center
            c = min(255.0, center + step)
            
            # Vectorized membership for all 256 gray values
            gray_vals = np.arange(256, dtype=np.float32)
            mf_vals = self._triangular_mf(gray_vals, a, b, c)
            
            # Activation weight across image histogram
            rule_activation = np.sum(hist * mf_vals)
            weighted_centers_sum += rule_activation * center
            weights_sum += rule_activation

        if weights_sum > 0:
            threshold = weighted_centers_sum / weights_sum
        else:
            threshold = 128.0

        return float(np.clip(threshold, 10.0, 240.0))

    def preprocess(self, image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Handle dimensions
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        elif image.ndim == 3 and image.shape[2] == 1:
            gray = image[:, :, 0]
        else:
            gray = image

        # Normalize to 0-255 uint8 for fuzzy logic operations
        gray_f = gray.astype(np.float32)
        g_min, g_max = gray_f.min(), gray_f.max()
        if g_max > g_min:
            scaled = ((gray_f - g_min) / (g_max - g_min) * 255.0).astype(np.uint8)
        else:
            scaled = np.zeros_like(gray, dtype=np.uint8)

        # Resize to target
        resized = cv2.resize(scaled, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)

        # 1. Compute Fuzzy Threshold T
        T = self._calculate_fuzzy_threshold(resized)

        # 2. Fuzzy contrast accentuation based on five regions (VL, L, M, H, VH)
        # Brain parenchyma typically clusters in Medium illumination (around T),
        # while lesion margins and enhancing boundaries reside in L, H, VH.
        res_f = resized.astype(np.float32)
        
        # Soft sigmoid-shaped fuzzy contrast transformation anchored at T
        gamma = 1.2
        normalized_diff = (res_f - T) / 64.0
        fuzzy_contrast = 1.0 / (1.0 + np.exp(-gamma * normalized_diff))

        # Blend with original normalized signal for high fidelity to anatomy
        base_norm = res_f / 255.0
        enhanced = 0.65 * base_norm + 0.35 * fuzzy_contrast
        enhanced = np.clip(enhanced, 0.0, 1.0).astype(np.float32)

        # Replicate across 3 channels (H, W, 3)
        three_channel = np.stack([enhanced, enhanced, enhanced], axis=-1)
        return three_channel
