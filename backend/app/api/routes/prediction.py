"""
Prediction and Analysis Routes
Delegates incoming MRI image analysis to BrainTumorAnalysisFacade.
Contains no direct ML or tensor manipulation logic (adheres to Layered Architecture).
"""

from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from typing import Optional
from backend.app.schemas.prediction import AnalysisResponse, ClassificationResult

router = APIRouter(tags=["Analysis"])

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_mri(
    request: Request,
    file: UploadFile = File(...),
    strategy: Optional[str] = Form("standard")
):
    """
    Main diagnostic endpoint:
    Processes MRI image via BrainTumorAnalysisFacade.
    Returns predicted tumor class, confidence, probabilities, Grad-CAM heatmap overlay,
    and segmentation mask with estimated tumor area.
    """
    if not file.content_type.startswith("image/") and not file.filename.endswith((".png", ".jpg", ".jpeg", ".mat", ".tif")):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image file.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    facade = request.app.state.facade

    try:
        result = facade.analyze(image_input=contents, strategy_name=strategy)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")

@router.post("/predict")
async def predict_tumor(
    request: Request,
    file: UploadFile = File(...),
    strategy: Optional[str] = Form("standard")
):
    """Classification-only endpoint."""
    contents = await file.read()
    facade = request.app.state.facade
    if strategy:
        facade.set_preprocessing_strategy(strategy)
    return facade.predict(image_input=contents)

@router.post("/segment")
async def segment_tumor(
    request: Request,
    file: UploadFile = File(...),
    strategy: Optional[str] = Form("standard")
):
    """Segmentation-only endpoint."""
    contents = await file.read()
    facade = request.app.state.facade
    if strategy:
        facade.set_preprocessing_strategy(strategy)
    res = facade.segment(image_input=contents)
    # Mask array cannot be directly serialized to json, convert to list or summary
    if "mask" in res and hasattr(res["mask"], "tolist"):
        res["mask_preview_shape"] = list(res["mask"].shape)
        del res["mask"]
    if "probability_map" in res:
        del res["probability_map"]
    return res
