# BrainTumorAI: Deep Learning Based Brain Tumor Segmentation and Classification from MRI Images

> **Academic Course Project:** Design and Architectural Patterns (7th Semester B.Tech Computer Science & Engineering)  
> **Notice:** This system is an **academic research prototype** and is **NOT intended for clinical medical diagnosis**.

---

## 1. Project Title
**BrainTumorAI: Deep Learning Based Brain Tumor Segmentation and Classification from MRI Images**

## 2. Problem Statement
Brain tumors represent one of the most critical causes of cancer mortality globally. Accurately distinguishing between tumor types (glioma, meningioma, pituitary) and delineating their anatomical boundaries on T1-weighted contrast-enhanced Magnetic Resonance Imaging (CE-MRI) is essential for clinical evaluation and surgical planning. Manual radiological inspection is time-consuming, prone to inter-observer variability, and subjective. Automated Computer-Aided Diagnosis (CAD) prototypes utilizing deep neural networks provide quantitative diagnostic assistance, transparent spatial explanations (Grad-CAM), and rapid cross-evaluation.

## 3. Objectives
1. **Multi-Class Tumor Classification:** Differentiate MRI slices into three tumor classes: **glioma**, **meningioma**, and **pituitary tumor** using transfer learning with EfficientNet-B0.
2. **Tumor Boundary Segmentation:** Delimit lesion boundaries using an EfficientNet-enhanced U-Net architecture.
3. **Quantitative Tumor Area Estimation:** Compute 2D segmented tumor pixel area and percentage relative to non-background brain parenchyma.
4. **Visual Explainability (Grad-CAM):** Generate gradient-weighted class activation heatmaps overlayed on raw MRI scans to demonstrate feature salience.
5. **Architectural Patterns Demonstration:** Implement and actively utilize five GoF software design patterns (**Strategy, Factory, Adapter, Repository, Facade**) within a clean 5-layer system.
6. **Web Dashboard:** Deliver a modern, interactive Next.js medical AI dashboard connected to a modular FastAPI backend.

## 4. Research Motivation & IEEE References
This project synthesizes methodologies from four key IEEE research publications:
1. **Tiwary et al. (2025)**, *"Deep Learning-Based MRI Brain Tumor Segmentation With EfficientNet-Enhanced UNet"*, *IEEE Access*: Demonstrates the superiority of compound-scaled EfficientNet backbones within U-Net decoders using Soft Dice Loss on the 3064-slice Figshare dataset.
2. **Almufareh et al. (2024)**, *"Automated Brain Tumor Segmentation and Classification in MRI Using YOLO-Based Deep Learning"*, *IEEE Access*: Explores YOLOv5/YOLOv7 instance detection on MRI scans and outlines mask alignment procedures.
3. **Hassan & Boulila (2025)**, *"Efficient Approach for Brain Tumor Detection and Classification Using Fuzzy Thresholding and Deep Learning Algorithms"*, *IEEE Access*: Introduces 16-membership Center-of-Gravity (CoG) fuzzy thresholding to enhance contrast in heterogeneous brain tissue.
4. **Sultan et al. (2019)**, *"Multi-Classification of Brain Tumor Images Using Deep Neural Network"*, *IEEE Access*: Benchmarks multi-class classification protocols, data augmentation, and confusion matrix evaluations on the Figshare repository.

## 5. Dataset
- **Source:** Figshare Brain Tumor Dataset (Cheng et al., 2017)
- **Modality:** T1-weighted contrast-enhanced MRI (CE-MRI)
- **Total Images:** 3,064 slices from 233 patients
- **Class Breakdown:**
  - **Glioma:** 1,426 slices (46.54%)
  - **Pituitary Tumor:** 930 slices (30.35%)
  - **Meningioma:** 708 slices (23.11%)
- **Data Partitions (Stratified by Patient & Class):**
  - **Training Set (80.96%):** 2,479 samples
  - **Validation Set (8.00%):** 246 samples
  - **Testing Set (11.04%):** 339 samples
- **Ground Truth:** Binary tumor masks (`tumorMask`) and border coordinates (`tumorBorder`).

## 6. Methodology
1. **Automated Ingestion:** Directly downloads all 4 Figshare archive parts (`.zip`) via HTTP range streaming and extracts `.mat` structs (`cjdata`).
2. **Preprocessing Pipeline (Strategy Pattern):**
   - *Standard Strategy:* Bicubic resizing to $224 \times 224$, min-max normalization into $[0.0, 1.0]$, and 3-channel grayscale replication.
   - *Fuzzy Strategy:* 16-membership triangular Center-of-Gravity defuzzification and 5-zone intensity contrast enhancement (Hassan & Boulila 2025).
3. **Training & Augmentation:** Horizontal/vertical flips, rotations ($\pm 15^\circ$), and mild color jitter.
4. **Classification:** CrossEntropyLoss optimized via Adam with ReduceLROnPlateau scheduling.
5. **Segmentation:** Combined Binary Cross-Entropy (BCE) and Soft Dice Loss.
6. **Explainability:** Grad-CAM on the final convolutional features block.

## 7. Machine Learning Architecture
- **Classifier:** `EfficientNet-B0` (ImageNet pretrained backbone, custom dropout and linear classification head with 3 output neurons: 0=glioma, 1=meningioma, 2=pituitary).
- **Segmentor:** `EfficientNetUNet` (EfficientNet-B0 encoder stages with skip connections, transposed convolution decoder blocks, and $1 \times 1$ convolution mask head).

## 8. Software Architecture
A strict **5-Layer Architecture** ensures complete decoupling:
```
Presentation Layer (Next.js 14, TypeScript, Tailwind CSS)
        ↓  (REST / JSON / Multipart)
API Layer (FastAPI routes: /health, /model-info, /analyze, /predict, /segment)
        ↓  (Application Facade Protocol)
Application Layer (BrainTumorAnalysisFacade)
        ↓  (Domain Contracts)
Domain Layer (Interfaces: PreprocessingStrategy, ModelAdapter, Repositories)
        ↓  (Inversion of Control)
Infrastructure Layer (PyTorch Models, Adapters, Grad-CAM, Repositories, Factories)
```

## 9. Design Patterns Implemented
1. **Strategy Pattern:** `PreprocessingStrategy` interface with `StandardPreprocessingStrategy` and `FuzzyPreprocessingStrategy`. Enables switching preprocessing algorithms at runtime without restarting inference.
2. **Factory Pattern:** `ModelFactory` dynamically instantiates and configures `EfficientNetAdapter` or `EfficientNetUNetAdapter` with hardware device binding.
3. **Adapter Pattern:** `ModelAdapter` standardizes divergent model outputs (classification probabilities vs spatial segmentation masks) under a common `predict()` interface.
4. **Repository Pattern:** `DatasetRepository` isolates `.mat`/`.png` file operations, while `ModelRepository` manages `.pth` checkpoint serialization and metadata tracking.
5. **Facade Pattern:** `BrainTumorAnalysisFacade` coordinates image decoding, preprocessing, classification, Grad-CAM generation, segmentation, and response formatting into a single `analyze(image)` call.

## 10. Technology Stack
- **Deep Learning:** PyTorch 2.x, Torchvision, NumPy, OpenCV, SciPy, Pillow, Scikit-Learn
- **Explainability:** Grad-CAM (PyTorch feature map & gradient hooks)
- **Backend API:** FastAPI, Uvicorn, Pydantic v2
- **Frontend:** Next.js 14, React 18, TypeScript, Tailwind CSS, Lucide Icons
- **Testing:** PyTest (11 unit tests covering all 5 patterns)

## 11. Installation

### Backend Setup
```bash
# From workspace root
pip install -r backend/requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
cd ..
```

## 12. Dataset Setup
To automatically download all 4 parts of the Figshare dataset (3064 slices), validate integrity, and convert to standardized PNGs:
```bash
# 1. Download and validate raw .mat files
python scripts/download_dataset.py

# 2. Convert to processed PNG images, masks, and manifest
python scripts/prepare_dataset.py --size 224
```

## 13. Model Training
```bash
# Train EfficientNet-B0 Classifier (3-class)
python scripts/train_classifier.py --epochs 4 --batch-size 32

# Train EfficientNet-UNet Segmentor (optional / modular)
python scripts/train_segmentation.py --epochs 3 --batch-size 16
```

## 14. Backend Execution
```bash
# Start FastAPI backend on port 8000
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API Swagger Docs: `http://localhost:8000/docs`

## 15. Frontend Execution
```bash
# Start Next.js dashboard on port 3000
cd frontend
npm run dev
```
Open `http://localhost:3000` in your web browser.

## 16. Testing & Standalone Verification
```bash
# Run 11 unit tests for all five design patterns
python -m pytest backend/tests -v

# Run standalone inference test on a real MRI slice (without frontend)
python scripts/test_prediction.py
```

## 17. Results (Strictly Non-Fabricated)
All metrics derive directly from actual forward-pass evaluation on the isolated 339-sample test set (Figshare CE-MRI):

### EfficientNet-B0 Classification Results (Unseen Test Partition: 339 Samples)
- **Overall Test Accuracy:** **97.35%** (330 / 339 correct)
- **Macro Precision:** **0.9708**
- **Macro Recall:** **0.9711**
- **Macro F1-Score:** **0.9708**
- **Weighted F1-Score:** **0.9736**
- **Best Validation Accuracy:** **96.75%**

#### Per-Class Performance Breakdown
| Class | Precision | Recall | F1-Score | Test Support (Slices) |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | 97.48% | 98.10% | 0.9779 | 158 |
| **Meningioma** | 93.75% | 96.15% | 0.9494 | 78 |
| **Pituitary Tumor** | 100.00% | 97.09% | 0.9852 | 103 |

#### Test Confusion Matrix
```
                    Predicted Glioma   Predicted Meningioma   Predicted Pituitary
Actual Glioma             155                    3                      0
Actual Meningioma           3                   75                      0
Actual Pituitary            1                    2                    100
```

### EfficientNet-UNet Segmentation Results (Test Partition)
- **Loss Function:** Combined BCE + Soft Dice Loss (Tiwary et al., 2025)
- **Training Setup:** 2 epochs on 250 samples (CPU feasibility checkpoint)
- **Test Dice Similarity Coefficient:** **5.83%**
- **Test IoU (Jaccard Index):** **3.00%**
- **Test Precision:** **0.0304**
- **Test Recall (Sensitivity):** **73.59%**
- *Note:* Modular checkpoint is trained and fully integrated into the Facade; full convergence on 3064 masks can be achieved by increasing training epochs on a dedicated GPU.

### Generated Artifacts
- **Metrics JSON:** `results/classification_metrics.json`
- **Classification Report:** `results/classification_report.txt`
- **Confusion Matrix Plot:** `results/confusion_matrix.png`
- **Training History Curves:** `results/training_history.png`
- **Grad-CAM Overlay Sample:** `results/test_gradcam_overlay.png`
- **Segmentation Metrics:** `results/segmentation_metrics.json`

## 18. Limitations
- **2D Slice Approximation:** MRI scans are 3D volumetric acquisitions; processing individual 2D axial slices estimates slice-level area but does not reconstruct true 3D volumetric tumor burden.
- **Single Scanner / Protocol Distribution:** The Figshare repository is acquired from a single imaging protocol (CE-MRI); generalizability across varied clinical field strengths (1.5T vs 3.0T) requires cross-center domain adaptation.
- **Non-Clinical Prototype:** The software is designed for academic evaluation of architectural patterns and must not be used for patient diagnosis or treatment decisions.

## 19. Future Work
- Integration of multi-sequence MRI fusion (FLAIR, T2, T1ce) based on BraTS protocols.
- 3D Volumetric U-Net extension for complete tumor core and edema segmentation.
- Hardware synthesis onto FPGA platforms as proposed by Hassan & Boulila (2025).
- Model quantization (INT8) and ONNX export for edge inference acceleration.
