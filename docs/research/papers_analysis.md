# Research Papers Analysis & Synthesis

**Project:** BrainTumorAI  
**Course:** Design and Architectural Patterns  

This document synthesizes the four IEEE research papers provided for the BrainTumorAI project. It details the public dataset, clinical and imaging backgrounds, comparative methodologies, evaluation metrics, and how our architecture incorporates their findings.

---

## Paper 1: Deep Learning-Based MRI Brain Tumor Segmentation With EfficientNet-Enhanced UNet

- **Authors:** Pradeep Kumar Tiwary, Prashant Johri, Alok Katiyar, Mayur Kumar Chhipa
- **Publication:** *IEEE Access*, Vol. 13, pp. 54920–54937, March 2025
- **DOI:** `10.1109/ACCESS.2025.3554405`

### 1. Key Methodology
- Addresses the limitation of standard CNN receptive fields in traditional U-Net by adopting an **EfficientNet encoder** within a U-Net architecture.
- EfficientNet balances network depth, width, and resolution using compound scaling ($d = \alpha^\phi, w = \beta^\phi, r = \gamma^\phi$).
- Mobile Inverted Bottleneck Convolution (MBConv) with Squeeze-and-Excitation (SE) modules provides channel-wise recalibration.
- Skip connections concatenate encoder feature maps ($F_i$) with decoder upsampled layers ($U_i$), preserving spatial boundaries of brain tumors.
- **Loss Function:** Soft Dice Loss:
  $$\text{SoftDiceLoss} = 1 - \frac{2 \sum (P_i G_i)}{\sum P_i^2 + \sum G_i^2 + \epsilon}$$

### 2. Dataset Utilized
- Brain-Tumor dataset (Jun Cheng et al., Figshare):
  - **3064** T1-weighted contrast-enhanced MRI images
  - **233** patients
  - Classes: Meningioma (708 slices), Glioma (1426 slices), Pituitary tumor (930 slices)
  - Split: 80.96% train, 8% validation, 11.04% test (random seed = 42).
  - Format: MATLAB `.mat` structs containing `cjdata.label`, `cjdata.PID`, `cjdata.image`, `cjdata.tumorBorder`, `cjdata.tumorMask`.

### 3. Application to BrainTumorAI
- Defines our primary segmentation architecture: **EfficientNet-B0 + U-Net Decoder**.
- Adopted the exact dataset, splits, and Soft Dice loss formulation.

---

## Paper 2: Automated Brain Tumor Segmentation and Classification in MRI Using YOLO-Based Deep Learning

- **Authors:** Maram Fahaad Almufareh, Muhammad Imran, Abdullah Khan, Mamoona Humayun, Muhammad Asim
- **Publication:** *IEEE Access*, Vol. 12, pp. 16189–16207, January 2024
- **DOI:** `10.1109/ACCESS.2024.3359418`

### 1. Key Methodology
- Formulates tumor localization as an object detection and instance segmentation task using YOLOv5 and YOLOv7 frameworks.
- Introduces mask alignment algorithms (Algorithm 1 and 2 in the paper) to convert MATLAB `.mat` structs into standard `.png` images and polygons/bounding boxes for YOLO annotation `.txt` files.
- Evaluates box detection and mask segmentation using mean Average Precision (mAP@0.5 and mAP@0.5:0.95), precision, recall, and F1 curves.

### 2. Findings & Relevance to Our Project
- The authors show that while YOLO provides high frame rates (100–143 FPS), it exhibits higher false positives in glioma segmentation compared to meningioma and pituitary.
- Per project specifications, YOLO is cataloged as related/future work. Our primary implementation leverages EfficientNet-B0 and EfficientNet-UNet, which better capture fine-grained pixel boundaries.

---

## Paper 3: Efficient Approach for Brain Tumor Detection and Classification Using Fuzzy Thresholding and Deep Learning Algorithms

- **Authors:** Nashaat M. Hussain Hassan, Wadii Boulila
- **Publication:** *IEEE Access*, Vol. 13, pp. 78808–78832, May 2025
- **DOI:** `10.1109/ACCESS.2025.3566332`

### 1. Key Methodology
- Proposes a two-stage diagnostic framework integrating **Fuzzy Logic** with deep learning:
  1. **Fuzzy Thresholding (Stage 1):** Divides the image intensity histogram into 16 triangular membership functions and defuzzifies using Center-of-Gravity (CoG):
     $$T = \frac{1}{M} \sum_{i=1}^M \sum_{j=1}^R \alpha_{ij} c_{ij}$$
  2. **Multi-Region Segmentation:** Classifies brain tissue into 5 linguistic intensity partitions: Very Low (VL), Low (L), Medium (M), High (H), and Very High (VH). Normal brain parenchyma occupies the Medium band, whereas necrotic core or hyperintense contrast-enhanced lesions occupy the boundary bands.
  3. **Tumor Area Calculation (Stage 2):** Calculates the relative size of the tumor compared to the total non-background brain mass.
  4. **Deep Classification (Stage 3):** Classifies lesions into four types.

### 2. Application to BrainTumorAI
- Directly motivates our **Strategy Pattern**: `FuzzyPreprocessingStrategy` implements this 16-membership CoG calculation and contrast enhancement.
- Informs our tumor area calculation module, which computes 2D pixel area and percentage of brain parenchyma.

---

## Paper 4: Multi-Classification of Brain Tumor Images Using Deep Neural Network

- **Authors:** Hossam H. Sultan, Nancy M. Salem, Walid Al-Atabany
- **Publication:** *IEEE Access*, Vol. 7, pp. 69215–69225, May 2019
- **DOI:** `10.1109/ACCESS.2019.2919122`

### 1. Key Methodology
- Benchmark study on multi-class brain tumor classification across axial, coronal, and sagittal views using convolutional neural networks.
- Uses the Figshare dataset (Cheng et al., 3064 slices across 233 patients) for 3-class classification (meningioma, glioma, pituitary).
- Implements extensive data augmentations: horizontal/vertical flips, rotations, and noise injection.
- Employs Cross-Entropy loss and Adam / SGDM optimizers with early stopping.
- Detailed evaluation reporting: confusion matrix, precision, sensitivity (recall), specificity, and overall accuracy.

### 2. Application to BrainTumorAI
- Informs our classification baseline, class mapping (`0 = glioma`, `1 = meningioma`, `2 = pituitary`), augmentation pipeline, and evaluation reporting.

---

## Comparative Synthesis Table

| Feature / Dimension | Tiwary et al. (2025) | Almufareh et al. (2024) | Hassan & Boulila (2025) | Sultan et al. (2019) | **Our BrainTumorAI Implementation** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Task** | Segmentation | Detection & Seg | Detection & Classif | Classification | **Unified Classification + Segmentation + XAI** |
| **Backbone Architecture** | EfficientNet-UNet | YOLOv5 / YOLOv7 | Custom CNN + Fuzzy | Custom 16-layer CNN | **EfficientNet-B0 + EfficientNet-UNet** |
| **Dataset** | Figshare (3064 slices) | Figshare (3064 slices) | Multi-source (23k imgs) | Figshare (3064 slices) | **Figshare (3064 slices, 233 patients)** |
| **Loss Function** | Soft Dice Loss | YOLO BBox + Cls + Obj | MAE Loss | CrossEntropyLoss | **CrossEntropy (Cls) + BCE + Soft Dice (Seg)** |
| **Explainability** | Visual slices | Bounding boxes | Visual fuzzy masks | Feature maps | **Grad-CAM (Gradient-weighted Class Activations)** |
| **Software Patterns** | None documented | None documented | None documented | None documented | **Strategy, Factory, Adapter, Repository, Facade** |
| **Interface** | Offline Python scripts | Google Colab scripts | Offline MATLAB / Py | MATLAB / Python | **Next.js Dashboard + FastAPI REST API** |
