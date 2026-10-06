"""
Model Info Route
Exposes metadata about loaded models and preprocessing strategies.
"""

from fastapi import APIRouter, Request
from backend.app.schemas.prediction import ModelInfoResponse

router = APIRouter(tags=["Model"])

@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(request: Request):
    facade = request.app.state.facade
    return ModelInfoResponse(
        classifier=facade.classifier.get_model_info(),
        segmentor=facade.segmentor.get_model_info(),
        available_preprocessing_strategies=["standard", "fuzzy"],
        current_strategy=facade.strategy.name
    )
