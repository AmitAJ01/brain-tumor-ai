# BrainTumorAI: System Architecture Documentation

**Course:** Design and Architectural Patterns  
**Semester:** 7th Semester B.Tech Computer Science and Engineering  
**Project:** BrainTumorAI  
**Title:** Deep Learning Based Brain Tumor Segmentation and Classification from MRI Images  

---

## 1. Architectural Overview

BrainTumorAI is designed using a **5-Layer Architecture (Clean/Onion-inspired Layered Architecture)**. This architecture strictly separates concerns, ensures high cohesion and loose coupling, isolates infrastructure (frameworks, storage, deep learning libraries) from business and domain contracts, and facilitates testability.

```mermaid
graph TD
    subgraph Presentation_Layer["1. Presentation Layer"]
        UI["Next.js / React 18 UI Dashboard"]
        Visualizer["Multi-Modal Visualizer (MRI, Grad-CAM, Mask, Overlay)"]
    end

    subgraph API_Layer["2. API Layer"]
        FastAPI["FastAPI App (main.py)"]
        Routes["Routes: /health, /model-info, /analyze, /predict, /segment"]
        Schemas["Pydantic Schemas (prediction.py)"]
    end

    subgraph Application_Layer["3. Application Layer"]
        Facade["BrainTumorAnalysisFacade (Facade Pattern)"]
    end

    subgraph Domain_Layer["4. Domain Layer"]
        subgraph Interfaces["Domain Interfaces"]
            IStrategy["PreprocessingStrategy (ABC)"]
            IAdapter["ModelAdapter (ABC)"]
            IDataRepo["IDatasetRepository (ABC)"]
            IModelRepo["IModelRepository (ABC)"]
        end
        subgraph Strategies["Concrete Strategies (Strategy Pattern)"]
            StdStrat["StandardPreprocessingStrategy"]
            FuzzyStrat["FuzzyPreprocessingStrategy (16-MF CoG)"]
        end
    end

    subgraph Infrastructure_Layer["5. Infrastructure Layer"]
        subgraph Adapters["Adapters (Adapter Pattern)"]
            EffNetAdapt["EfficientNetAdapter"]
            UNetAdapt["EfficientNetUNetAdapter"]
        end
        subgraph Models["PyTorch Neural Networks"]
            EffNetB0["EfficientNet-B0 Classifier (3-Class)"]
            EffUNet["EfficientNet-UNet Segmentor"]
        end
        subgraph Explainability["XAI Subsystem"]
            GradCAM["GradCAMExplainer (Features Backprop)"]
        end
        subgraph Repositories["Repositories (Repository Pattern)"]
            DataRepo["DatasetRepository (data/raw, data/processed)"]
            ModelRepo["ModelRepository (models/*.pth)"]
        end
        subgraph Factories["Factories (Factory Pattern)"]
            MFactory["ModelFactory"]
        end
    end

    Presentation_Layer -->|HTTP/REST /analyze| API_Layer
    API_Layer -->|Delegates request| Application_Layer
    Application_Layer -->|Selects & invokes| Domain_Layer
    Application_Layer -->|Orchestrates| Infrastructure_Layer
    Infrastructure_Layer -.->|Implements| Domain_Layer
```

---

## 2. Layer-by-Layer Breakdown

### Layer 1: Presentation Layer
- **Technology:** Next.js (React 18), TypeScript, Tailwind CSS, Lucide Icons.
- **Responsibilities:**
  - Medical image upload interface (drag-and-drop or single click).
  - Benchmark sample selection (allowing 1-click loading of verified test slices from test set).
  - Runtime Preprocessing Strategy selector (`Standard` vs `Fuzzy`).
  - Rendering real classification results (Predicted class, confidence %, probabilities breakdown).
  - Multi-modal diagnostic visualization (Original MRI, Grad-CAM attention heatmap, Predicted tumor mask, Tumor outline overlay).
  - Quantitative metrics display (2D estimated tumor area in pixels and percentage of brain tissue).
  - Prominent academic disclaimer.

### Layer 2: API Layer
- **Technology:** FastAPI, Uvicorn, Pydantic v2.
- **Responsibilities:**
  - Route declarations (`/health`, `/model-info`, `/analyze`, `/predict`, `/segment`).
  - Request validation and content-type checking.
  - Zero ML or business logic inside route functions: routes exclusively extract payloads and delegate to the Application Layer Facade.
  - CORS configuration enabling cross-origin requests from the web dashboard.

### Layer 3: Application Layer
- **Component:** `BrainTumorAnalysisFacade`.
- **Responsibilities:**
  - Orchestrates the sequential pipeline across domain and infrastructure components.
  - Coordinates image decoding, strategy selection, classifier forward pass, Grad-CAM gradient calculation, segmentation inference, tumor area measurement, and Base64 packaging.
  - Serves as the single point of entry for the API routes.

### Layer 4: Domain Layer
- **Interfaces:**
  - `PreprocessingStrategy`: Strategy interface specifying `preprocess(image, target_size)`.
  - `ModelAdapter`: Target interface defining `load_model()`, `predict()`, `get_model_info()`, `is_loaded()`.
  - `IDatasetRepository`: Contract for retrieving samples, partitions, and dataset metadata.
  - `IModelRepository`: Contract for saving/loading model weights and metrics.
- **Strategies:**
  - `StandardPreprocessingStrategy`: Min-Max normalization, bicubic interpolation, 3-channel replication.
  - `FuzzyPreprocessingStrategy`: Center-of-Gravity (CoG) adaptive threshold calculation using 16 triangular membership functions and 5 illumination zones (VL, L, M, H, VH) inspired by Hassan & Boulila (2025).

### Layer 5: Infrastructure Layer
- **Models:**
  - `EfficientNet-B0`: Modified transfer learning classification network for 3 tumor classes (`0 = glioma`, `1 = meningioma`, `2 = pituitary`).
  - `EfficientNetUNet`: Encoder-decoder architecture combining EfficientNet-B0 feature extractor with U-Net upsampling and skip connections.
- **Adapters:**
  - `EfficientNetAdapter`: Adapts raw PyTorch classifier outputs to standardized prediction dictionaries.
  - `EfficientNetUNetAdapter`: Adapts segmentation outputs, performs binarization, calculates 2D pixel area and percentage of brain parenchyma.
- **Factory:**
  - `ModelFactory`: Factory pattern implementation creating model adapters on demand.
- **Explainability:**
  - `GradCAMExplainer`: Computes gradients of the target class score with respect to the final convolutional feature maps, produces Jet colormap heatmap, and alpha-blends over MRI slice.
- **Repositories:**
  - `DatasetRepository`: Manages dataset access from disk.
  - `ModelRepository`: Manages checkpoint persistence and retrieval.

---

## 3. End-to-End Data Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as Doctor / Researcher
    participant UI as Next.js Dashboard
    participant API as FastAPI (/analyze)
    participant Facade as BrainTumorAnalysisFacade
    participant Strat as PreprocessingStrategy
    participant ClsAdapt as EfficientNetAdapter
    participant GradCAM as GradCAMExplainer
    participant SegAdapt as EfficientNetUNetAdapter

    User->>UI: Selects MRI & clicks "Analyze MRI"
    UI->>API: POST /analyze (multipart/form-data: image, strategy)
    API->>Facade: analyze(image_bytes, strategy_name)
    
    rect rgb(240, 245, 255)
        Note over Facade,Strat: Step 1: Preprocessing (Strategy Pattern)
        Facade->>Strat: preprocess(raw_image, target_size=(224,224))
        Strat-->>Facade: preprocessed_tensor (1, 3, 224, 224)
    end

    rect rgb(245, 250, 245)
        Note over Facade,ClsAdapt: Step 2: Classification (Adapter Pattern)
        Facade->>ClsAdapt: predict(tensor)
        ClsAdapt-->>Facade: {prediction: "glioma", confidence: 0.96, probabilities}
    end

    rect rgb(255, 250, 240)
        Note over Facade,GradCAM: Step 3: Explainability
        Facade->>GradCAM: generate(tensor, target_class="glioma")
        GradCAM-->>Facade: gradcam_overlay_base64
    end

    rect rgb(255, 245, 245)
        Note over Facade,SegAdapt: Step 4: Segmentation & Area (Adapter Pattern)
        Facade->>SegAdapt: predict(tensor)
        SegAdapt-->>Facade: {mask, tumor_area_pixels, tumor_area_percentage}
    end

    Facade-->>API: Consolidated AnalysisResponse dictionary
    API-->>UI: JSON AnalysisResponse
    UI-->>User: Displays Prediction, Probabilities, Grad-CAM, Mask & Tumor Area
```
