"""
Pydantic Schemas for API Request and Response Models
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    app_name: str = Field(..., example="BrainTumorAI API")
    version: str = Field(..., example="1.0.0")
    device: str = Field(..., example="cpu")
    models_status: Dict[str, bool] = Field(...)

class ModelInfoResponse(BaseModel):
    classifier: Dict[str, Any]
    segmentor: Dict[str, Any]
    available_preprocessing_strategies: List[str]
    current_strategy: str

class SegmentationResult(BaseModel):
    available: bool
    tumor_detected: bool
    tumor_area_pixels: int
    total_brain_pixels: int
    tumor_area_percentage: float
    mask_url: Optional[str] = None
    overlay_url: Optional[str] = None
    disclaimer: str

class ClassificationResult(BaseModel):
    model_type: str
    class_index: int
    predicted_class: str
    confidence: float
    confidence_percentage: float
    probabilities: Dict[str, float]
    logits: Optional[List[float]] = None

class AnalysisResponse(BaseModel):
    prediction: str
    confidence: float
    confidence_percentage: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    strategy_used: str
    original_image_url: Optional[str] = None
    gradcam_url: Optional[str] = None
    segmentation: SegmentationResult
    model_info: Dict[str, Any]
    disclaimer: str
