"""
BrainTumorAI Backend Application Entrypoint
FastAPI application with Layered Architecture and Design Patterns.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.app.application.brain_tumor_facade import BrainTumorAnalysisFacade
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.model import router as model_router
from backend.app.api.routes.prediction import router as prediction_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Application Layer Facade
    print("[BACKEND] Initializing BrainTumorAnalysisFacade...")
    app.state.facade = BrainTumorAnalysisFacade()
    print("[BACKEND] Facade initialized. System ready for requests.")
    yield
    print("[BACKEND] Shutting down BrainTumorAI API...")

app = FastAPI(
    title="BrainTumorAI API",
    description="Academic prototype for brain tumor classification, segmentation, and Grad-CAM explainability.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow development frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health_router)
app.include_router(model_router)
app.include_router(prediction_router)

@app.get("/")
def read_root():
    return {
        "project": "BrainTumorAI",
        "title": "Deep Learning Based Brain Tumor Segmentation and Classification from MRI Images",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "disclaimer": "Academic research prototype. Not intended for clinical or medical diagnosis."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
