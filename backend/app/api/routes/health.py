"""
Health Route
Provides liveness and model readiness checks.
"""

from fastapi import APIRouter, Request
from backend.app.schemas.prediction import HealthResponse

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    facade = request.app.state.facade
    return HealthResponse(
        status="ok",
        app_name="BrainTumorAI API",
        version="1.0.0",
        device=facade.device,
        models_status={
            "classifier_loaded": facade.classifier.is_loaded(),
            "segmentor_loaded": facade.segmentor.is_loaded()
        }
    )
