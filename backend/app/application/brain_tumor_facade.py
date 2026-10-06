"""
Brain Tumor Analysis Facade
Implements the Facade Pattern to provide a unified, simple entry point
for the complete brain MRI diagnostic pipeline.
Coordinates:
1. Preprocessing (Standard or Fuzzy via Strategy Pattern)
2. Classification (EfficientNet via Adapter Pattern)
3. Segmentation (EfficientNet-UNet via Adapter Pattern, if available)
4. Tumor Area Estimation (from segmentation mask)
5. Grad-CAM Explainability (heatmap generation & overlay)
6. Response Formatting and Visualization encoding
"""

import time
import io
import base64
from typing import Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

from backend.app.domain.interfaces.preprocessing_strategy import PreprocessingStrategy
from backend.app.domain.interfaces.model_adapter import ModelAdapter
from backend.app.domain.strategies.standard_preprocessing import StandardPreprocessingStrategy
from backend.app.domain.strategies.fuzzy_preprocessing import FuzzyPreprocessingStrategy
from backend.app.infrastructure.factories.model_factory import ModelFactory
from backend.app.infrastructure.repositories.model_repository import ModelRepository
from backend.app.infrastructure.explainability.gradcam import GradCAMExplainer

class BrainTumorAnalysisFacade:
    """
    Facade Pattern.
    Exposes high-level methods (analyze, predict, segment) while hiding
    the complexity of subsystem interactions, PyTorch tensors, Grad-CAM hooks,
    and image encoding.
    """

    def __init__(
        self,
        classifier: Optional[ModelAdapter] = None,
        segmentor: Optional[ModelAdapter] = None,
        preprocessing_strategy: Optional[PreprocessingStrategy] = None,
        model_repository: Optional[ModelRepository] = None,
        device: str = "cpu"
    ):
        self.device = device
        self.model_repo = model_repository or ModelRepository()

        # Preprocessing strategy (Default to standard)
        self.strategy: PreprocessingStrategy = preprocessing_strategy or StandardPreprocessingStrategy()
        self._strategy_cache = {
            "standard": StandardPreprocessingStrategy(),
            "fuzzy": FuzzyPreprocessingStrategy()
        }

        # Initialize or load classifier
        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = ModelFactory.create("efficientnet", device=self.device)
            # Auto-load weights if checkpoint exists
            default_weights = self.model_repo.get_path("brain_tumor_efficientnet.pth")
            if default_weights.exists():
                try:
                    self.classifier.load_model(str(default_weights), device=self.device)
                    print(f"[FACADE] Loaded classifier weights from {default_weights}")
                except Exception as e:
                    print(f"[FACADE WARNING] Could not load classifier: {e}")

        # Initialize or load segmentor
        if segmentor is not None:
            self.segmentor = segmentor
        else:
            self.segmentor = ModelFactory.create("efficient_unet", device=self.device)
            default_seg_weights = self.model_repo.get_path("brain_tumor_efficientunet.pth")
            if default_seg_weights.exists():
                try:
                    self.segmentor.load_model(str(default_seg_weights), device=self.device)
                    print(f"[FACADE] Loaded segmentation weights from {default_seg_weights}")
                except Exception as e:
                    print(f"[FACADE NOTICE] Segmentor weights not loaded: {e}")

    def set_preprocessing_strategy(self, strategy: Union[str, PreprocessingStrategy]):
        """Dynamically switch preprocessing strategy at runtime (Strategy Pattern)."""
        if isinstance(strategy, str):
            strat_key = strategy.lower().strip()
            if strat_key in self._strategy_cache:
                self.strategy = self._strategy_cache[strat_key]
            else:
                raise ValueError(f"Unknown strategy '{strategy}'. Options: {list(self._strategy_cache.keys())}")
        elif isinstance(strategy, PreprocessingStrategy):
            self.strategy = strategy
        else:
            raise TypeError("Strategy must be string name or PreprocessingStrategy instance")

    def _decode_image(self, image_input: Union[bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """Decode input payload into a 2D/3D uint8/float32 numpy array."""
        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))
            return np.array(image)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input)
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _encode_png_base64(self, arr: np.ndarray) -> str:
        """Helper to encode RGB/grayscale array into PNG Base64 data URL."""
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def analyze(
        self,
        image_input: Union[bytes, np.ndarray, Image.Image],
        strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Unified analysis workflow (Facade method):
        1. Decode & validate image
        2. Apply Preprocessing Strategy
        3. Run EfficientNet Classifier
        4. Generate Grad-CAM Explainability
        5. Run EfficientNet-UNet Segmentation (if weights available)
        6. Calculate Tumor Area & Mask Overlay
        7. Format consolidated academic response
        """
        start_time = time.time()

        # Switch strategy if explicitly passed
        if strategy_name is not None:
            self.set_preprocessing_strategy(strategy_name)

        # 1. Decode Image
        raw_np = self._decode_image(image_input)

        # Prepare base image for web display
        if raw_np.ndim == 2:
            display_base = cv2.cvtColor(
                (cv2.resize(raw_np, (224, 224)) / max(raw_np.max(), 1) * 255).astype(np.uint8),
                cv2.COLOR_GRAY2RGB
            )
        else:
            display_base = cv2.resize(raw_np, (224, 224))
            if display_base.ndim == 2 or display_base.shape[2] == 1:
                display_base = cv2.cvtColor(display_base, cv2.COLOR_GRAY2RGB)

        # 2. Preprocess via active strategy
        prep_img = self.strategy.preprocess(raw_np, target_size=(224, 224)) # (224, 224, 3) float32 [0, 1]

        # Convert to Tensor (B, C, H, W) and apply ImageNet normalization
        tensor = torch.from_numpy(prep_img).permute(2, 0, 1).unsqueeze(0).float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        norm_tensor = (tensor - mean) / std

        # 3. Classification
        classification_result = self.classifier.predict(norm_tensor)
        predicted_class = classification_result["predicted_class"]
        pred_idx = classification_result["class_index"]
        confidence = classification_result["confidence"]

        # 4. Grad-CAM Explainability
        gradcam_overlay_url = None
        try:
            # We use the raw model from adapter
            raw_model = self.classifier.model
            explainer = GradCAMExplainer(raw_model)
            _, _, gradcam_overlay_url = explainer.generate(
                input_tensor=norm_tensor,
                target_class_idx=pred_idx,
                original_image_uint8=display_base,
                alpha=0.45
            )
            explainer.remove_hooks()
        except Exception as e:
            print(f"[FACADE] Grad-CAM generation warning: {e}")

        # 5 & 6. Segmentation & Tumor Area Estimation
        segmentation_info = {
            "available": False,
            "tumor_detected": False,
            "tumor_area_pixels": 0,
            "tumor_area_percentage": 0.0,
            "mask_url": None,
            "overlay_url": None,
            "disclaimer": "Tumor area is a 2D pixel estimation from segmentation and NOT a clinical 3D volume."
        }

        if self.segmentor.is_loaded():
            try:
                seg_res = self.segmentor.predict(tensor)
                mask_np = seg_res["mask"] # (224, 224) uint8 [0, 255]
                
                # Build overlay: Red boundary & fill on MRI
                mask_color = np.zeros_like(display_base)
                mask_color[:, :, 0] = mask_np # Red channel for tumor highlight
                
                # Contours for outline
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                seg_overlay = display_base.copy()
                cv2.drawContours(seg_overlay, contours, -1, (255, 50, 50), 2)
                # Soft fill inside tumor
                seg_overlay = cv2.addWeighted(seg_overlay, 0.75, mask_color, 0.25, 0)

                segmentation_info = {
                    "available": True,
                    "tumor_detected": seg_res["tumor_detected"],
                    "tumor_area_pixels": seg_res["tumor_area_pixels"],
                    "total_brain_pixels": seg_res["total_brain_pixels"],
                    "tumor_area_percentage": seg_res["tumor_area_percentage"],
                    "mask_url": self._encode_png_base64(mask_np),
                    "overlay_url": self._encode_png_base64(seg_overlay),
                    "disclaimer": "Tumor area is a 2D pixel estimation from segmentation and NOT a clinical 3D volume."
                }
            except Exception as e:
                print(f"[FACADE] Segmentation inference error: {e}")

        elapsed_ms = round((time.time() - start_time) * 1000.0, 2)

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "confidence_percentage": classification_result["confidence_percentage"],
            "probabilities": classification_result["probabilities"],
            "processing_time_ms": elapsed_ms,
            "strategy_used": self.strategy.name,
            "original_image_url": self._encode_png_base64(display_base),
            "gradcam_url": gradcam_overlay_url,
            "segmentation": segmentation_info,
            "model_info": {
                "classifier": self.classifier.get_model_info(),
                "segmentor": self.segmentor.get_model_info(),
            },
            "disclaimer": "This system is an academic research prototype and is not intended for medical diagnosis."
        }

    def predict(self, image_input: Union[bytes, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Classification-only endpoint."""
        raw_np = self._decode_image(image_input)
        prep = self.strategy.preprocess(raw_np, target_size=(224, 224))
        tensor = torch.from_numpy(prep).permute(2, 0, 1).unsqueeze(0).float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        norm_tensor = (tensor - mean) / std
        return self.classifier.predict(norm_tensor)

    def segment(self, image_input: Union[bytes, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Segmentation-only endpoint."""
        raw_np = self._decode_image(image_input)
        prep = self.strategy.preprocess(raw_np, target_size=(224, 224))
        tensor = torch.from_numpy(prep).permute(2, 0, 1).unsqueeze(0).float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        norm_tensor = (tensor - mean) / std
        if not self.segmentor.is_loaded():
            return {"error": "Segmentation model checkpoint is not loaded."}
        return self.segmentor.predict(norm_tensor)
