# BrainTumorAI - Backend

## Layered Architecture & Design Patterns

The backend follows a strict 5-layer Clean Architecture:
1. **Presentation Layer**: Consumed by Next.js client.
2. **API Layer** (`backend/app/api/`): FastAPI routers (`health`, `model`, `prediction`) using Pydantic schemas (`backend/app/schemas/`). Zero ML logic in routes.
3. **Application Layer** (`backend/app/application/`): `BrainTumorAnalysisFacade` coordinates image preprocessing, model classification, segmentation, Grad-CAM, and tumor area calculation.
4. **Domain Layer** (`backend/app/domain/`): Core interfaces (`PreprocessingStrategy`, `ModelAdapter`, `IDatasetRepository`, `IModelRepository`) and concrete strategies (`StandardPreprocessingStrategy`, `FuzzyPreprocessingStrategy`).
5. **Infrastructure Layer** (`backend/app/infrastructure/`): PyTorch models (`EfficientNet-B0`, `EfficientNet-UNet`), Model Adapters, Grad-CAM explainer, and repositories.

## API Endpoints
- `GET /`: API overview and status
- `GET /health`: System health and model availability
- `GET /model-info`: Architecture parameters, classes, and loaded weights
- `POST /analyze`: Main endpoint accepting MRI image; returns classification, confidence, probabilities, Grad-CAM overlay, and segmentation metrics
- `POST /predict`: Direct classification inference
- `POST /segment`: Direct segmentation inference

## Running Backend
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
