# DefectLens AI — System Architecture

## Overview

DefectLens AI is a multi-mode industrial visual inspection system. It integrates three complementary computer vision approaches: **image classification**, **anomaly detection**, and **pixel-level segmentation**, to address different inspection requirements without overfitting to a single method.

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Input Layer                                  │
│          Raw industrial image (JPEG/PNG, any resolution)             │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Preprocessing    │
                    │  - Resize 224×224 │
                    │  - Normalize      │
                    │  - Augment (train)│
                    └─────────┬─────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
  ┌───────▼────────┐  ┌───────▼────────┐  ┌──────▼──────────┐
  │  Classifier    │  │  Anomaly       │  │  Segmentor      │
  │  ResNet-18     │  │  Detector      │  │  U-Net          │
  │  Fine-tuned    │  │  PatchCore     │  │  (encoder-      │
  │  binary head   │  │  memory bank   │  │   decoder)      │
  └───────┬────────┘  └───────┬────────┘  └──────┬──────────┘
          │                   │                   │
  ┌───────▼────────┐  ┌───────▼────────┐  ┌──────▼──────────┐
  │  normal/defect │  │  anomaly score │  │  pixel mask     │
  │  probability   │  │  + heatmap     │  │  + defect area  │
  └───────┬────────┘  └───────┬────────┘  └──────┬──────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  DefectPredictor  │
                    │  Result fusion +  │
                    │  confidence score │
                    └─────────┬─────────┘
                              │
               ┌──────────────┼──────────────┐
               │              │              │
       ┌───────▼──────┐ ┌─────▼─────┐ ┌────▼──────────┐
       │ Visualizer   │ │ REST API  │ │ Streamlit App  │
       │ overlay maps │ │ FastAPI   │ │ interactive UI │
       └──────────────┘ └───────────┘ └───────────────┘
```

---

## Component Descriptions

| Component | Location | Responsibility |
|-----------|----------|---------------|
| `MVTecDataset` | `src/defectlens/data/dataset.py` | Loads MVTec AD images and ground-truth masks; supports train/test splits and per-category filtering |
| `get_train_transforms` | `src/defectlens/data/transforms.py` | Applies random crop, flip, colour jitter, and normalization for training augmentation |
| `validate_dataset` | `src/defectlens/data/validation.py` | Checks dataset directory structure; returns `DatasetReport` with errors and category list |
| `DefectClassifier` | `src/defectlens/models/classifier.py` | ResNet-18 backbone with replaceable classification head; supports `finetune` and `feature_extract` modes; returns dict `{logits, probs, features}` |
| `PatchCoreAnomalyDetector` | `src/defectlens/models/anomaly.py` | Builds a memory bank of normal-image patch features; scores test images by nearest-neighbour distance; produces anomaly heatmaps |
| `DefectSegmentor` | `src/defectlens/models/segmentor.py` | U-Net with configurable encoder; produces binary pixel masks marking defective regions |
| `DefectClassifierTrainer` | `src/defectlens/training/trainer.py` | Training loop with early stopping, LR scheduling, and checkpoint saving; logs loss and accuracy per epoch |
| `compute_classification_metrics` | `src/defectlens/evaluation/metrics.py` | Computes precision, recall, F1, ROC-AUC, confusion matrix from predictions and labels |
| `generate_error_report` | `src/defectlens/evaluation/analysis.py` | Identifies false positives, false negatives, borderline cases; outputs structured error report dict |
| `DefectPredictor` | `src/defectlens/inference/predictor.py` | Wraps classifier + anomaly detector + segmentor into a single `predict()` call; also exposes `benchmark()` for latency measurement |
| `DefectVisualizer` | `src/defectlens/visualization/visualizer.py` | Overlays segmentation masks, bounding boxes, and confidence scores on original images |
| `api.py` | `app/api.py` | FastAPI endpoints: `POST /inspect` (image file → JSON result), `GET /health` |
| `streamlit_app.py` | `app/streamlit_app.py` | Browser-based inspection dashboard with image upload, mode selection, and result display |

---

## Data Flow

```
MVTec AD dataset
       │
       ▼
MVTecDataset (loads image + mask pairs)
       │
       ▼
Transforms (resize → normalize → optional augment)
       │
       ├──► DefectClassifierTrainer ──► checkpoint.pth
       │
       ├──► PatchCoreAnomalyDetector.fit(normal_images) ──► memory_bank.pkl
       │
       └──► DefectSegmentor training ──► segmentor_checkpoint.pth
                                               │
                                               ▼
                                     DefectPredictor.predict(image)
                                               │
                                     {label, confidence, anomaly_score, mask}
                                               │
                                     DefectVisualizer / FastAPI / Streamlit
```

---

## Key Design Decisions

### 1. Three-mode inspection
A single image can be evaluated by classification, anomaly detection, or segmentation depending on what information is available and what fidelity is needed. The `DefectPredictor` can run any subset of the three modes.

### 2. PatchCore for anomaly detection
PatchCore does not require defect labels — it only needs normal examples to build a memory bank. This is realistic for industrial settings where defects are rare and difficult to annotate exhaustively.

### 3. U-Net for segmentation
U-Net's skip connections preserve fine spatial detail, which is critical for detecting small scratches and hairline cracks. The architecture scales to any input resolution through bilinear upsampling.

### 4. dict output from classifier
The classifier returns `{"logits", "probs", "features"}` rather than a tensor, allowing downstream components to access intermediate feature representations without a second forward pass.

### 5. CPU-first
All models run on CPU by default. CUDA is used automatically when available. This makes the system accessible without specialised hardware.

---

## Performance Characteristics

| Mode | Typical latency (CPU, 224×224) | Memory |
|------|-------------------------------|--------|
| Classifier (ResNet-18) | ~30–80 ms | ~46 MB |
| PatchCore (score only) | ~20–60 ms | depends on bank size |
| U-Net segmentor | ~80–150 ms | ~31 MB |
| Full pipeline | ~150–300 ms | ~120 MB |

> [!NOTE]
> Latency varies with hardware. GPU inference is typically 5–10× faster.

---

## Dataset

**MVTec AD** — the primary benchmark dataset for unsupervised anomaly detection.
- 15 industrial categories (metal, texture, object).
- 5,354 training images (defect-free).
- 1,725 test images with pixel-level ground truth masks.
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
- Download: https://www.mvtec.com/company/research/datasets/mvtec-ad

See `data/README.md` for detailed download and preparation instructions.
