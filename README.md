<div align="center">

# 🔍 DefectLens AI

### Industrial Visual Inspection & Anomaly Detection

Detect manufacturing defects — scratches, cracks, dents, and surface irregularities — using deep learning anomaly detection and image classification.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Overview

DefectLens AI is an industrial visual inspection system that detects manufacturing defects in product images. The system supports three complementary approaches:

1. **Binary Classification** — Classify products as normal or defective using a fine-tuned ResNet-18
2. **Anomaly Detection** — Detect anomalies by learning the distribution of normal products (no defective training data needed)
3. **Defect Segmentation** — Locate the exact defective regions using U-Net pixel-level segmentation

The system is trained and evaluated on the [MVTec Anomaly Detection (MVTec AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset, which contains 5,354 high-resolution images across 15 industrial product and texture categories.

---

## Key Features

- **Multi-approach defect detection**: Classification, anomaly detection, and segmentation
- **15 industrial categories**: Bottle, cable, capsule, carpet, grid, hazelnut, leather, metal nut, pill, screw, tile, toothbrush, transistor, wood, zipper
- **Visual inspection dashboard**: Interactive Streamlit application for real-time inspection
- **Anomaly heatmaps**: Spatial visualization of defect likelihood
- **Comprehensive evaluation**: AUROC, F1, Precision, Recall, Dice, IoU, confusion matrices
- **Error analysis**: Identifies failure cases and hardest examples
- **Reproducible experiments**: Configurable via YAML, seeded randomness

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DefectLens AI                        │
├──────────┬──────────────────┬───────────────────────────┤
│          │                  │                           │
│  Binary  │   Anomaly        │   Defect                  │
│ Classifier│  Detection       │  Segmentation             │
│          │                  │                           │
│ ResNet-18│  Feature Memory  │   U-Net                   │
│ Fine-tune│  Bank + Distance │   ResNet-18 Encoder       │
│          │  Scoring         │                           │
├──────────┴──────────────────┴───────────────────────────┤
│              Shared Preprocessing                       │
│         Resize → Normalize → Augment                    │
├─────────────────────────────────────────────────────────┤
│              MVTec AD Dataset                           │
│     15 categories · 5,354 images · Pixel masks          │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
defectlens-ai/
├── configs/
│   ├── base.yaml                  # Base configuration
│   ├── train_classifier.yaml      # Classification training
│   └── train_anomaly.yaml         # Anomaly detection config
├── data/
│   └── README.md                  # Dataset download instructions
├── docs/
│   └── architecture.md            # Technical documentation
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_baseline_experiments.ipynb
│   └── 03_results_analysis.ipynb
├── scripts/
│   ├── download_dataset.py        # Dataset preparation
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   └── inference.py               # Single image inference
├── src/defectlens/
│   ├── data/                      # Dataset loaders & transforms
│   ├── models/                    # Classifier, anomaly, segmentor
│   ├── training/                  # Training loop & callbacks
│   ├── evaluation/                # Metrics & error analysis
│   ├── inference/                 # Inference engine
│   └── visualization/            # Heatmaps, overlays, plots
├── app/
│   └── streamlit_app.py           # Interactive inspection dashboard
├── tests/                         # Unit & integration tests
├── requirements.txt
├── setup.py
├── Dockerfile
├── .gitignore
└── LICENSE
```

---

## Quick Start

### 1. Clone and Install

```powershell
git clone https://github.com/moise744/defectlens-ai.git
cd defectlens-ai
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare Dataset

Download MVTec AD from [mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract to `data/mvtec_ad/`.

See [data/README.md](data/README.md) for detailed instructions.

```powershell
# Verify dataset structure
python scripts/download_dataset.py --verify --data-dir data/mvtec_ad
```

### 3. Train a Model

```powershell
# Train binary classifier on a single category
python scripts/train.py --config configs/train_classifier.yaml --category bottle

# Train on all categories
python scripts/train.py --config configs/train_classifier.yaml --category all
```

### 4. Evaluate

```powershell
python scripts/evaluate.py --config configs/base.yaml --category bottle --checkpoint outputs/bottle/best_model.pth
```

### 5. Run Inference

```powershell
# Single image
python scripts/inference.py --image path/to/image.png --checkpoint outputs/bottle/best_model.pth

# Directory of images
python scripts/inference.py --image-dir path/to/images/ --checkpoint outputs/bottle/best_model.pth
```

### 6. Launch Dashboard

```powershell
streamlit run app/streamlit_app.py
```

---

## Dataset: MVTec AD

| Property | Value |
|----------|-------|
| **Name** | MVTec Anomaly Detection Dataset |
| **Source** | [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| **License** | CC BY-NC-SA 4.0 |
| **Total Images** | 5,354 |
| **Categories** | 15 (10 objects + 5 textures) |
| **Image Resolution** | 700×700 to 1024×1024 |
| **Annotations** | Pixel-level ground truth masks for defects |
| **Train Set** | Normal images only |
| **Test Set** | Normal + various defect types |

---

## Models

### Binary Classifier (ResNet-18)
- **Architecture**: ResNet-18 pretrained on ImageNet, fine-tuned for binary classification
- **Input**: 256×256 RGB image
- **Output**: Normal (0) or Defective (1) with confidence score
- **Training**: Transfer learning with optional backbone freezing

### Anomaly Detection (Feature Memory Bank)
- **Architecture**: ResNet-18 feature extractor + nearest-neighbor scoring
- **Approach**: Learn normal feature distribution, score anomalies by distance
- **Input**: 256×256 RGB image
- **Output**: Anomaly score + spatial anomaly heatmap
- **Advantage**: No defective training images required

### Defect Segmentation (U-Net)
- **Architecture**: U-Net with ResNet-18 encoder
- **Input**: 256×256 RGB image
- **Output**: Pixel-level defect probability mask
- **Training**: Supervised with ground truth defect masks

---

## Evaluation Metrics

### Image-Level
| Metric | Description |
|--------|-------------|
| AUROC | Area Under ROC Curve |
| F1 Score | Harmonic mean of precision and recall |
| Precision | Fraction of correct defect predictions |
| Recall | Fraction of actual defects detected |
| Accuracy | Overall classification accuracy |

### Pixel-Level (Segmentation)
| Metric | Description |
|--------|-------------|
| Pixel AUROC | Per-pixel area under ROC curve |
| IoU (Jaccard) | Intersection over Union of predicted masks |
| Dice Score | Dice coefficient (F1 for segmentation) |

### Performance
| Metric | Description |
|--------|-------------|
| Inference Latency | Milliseconds per image |
| FPS | Frames per second |
| Model Size | Parameters and disk size |

---

## Experiment Results

> [!NOTE]
> Results will be populated after training. Due to CPU-only hardware, initial experiments use a subset of categories with limited epochs. Full reproduction instructions with GPU configuration are provided.

Results for each category will include:
- Classification AUROC and F1
- Anomaly detection AUROC
- Segmentation Dice and IoU
- Inference speed benchmarks
- Failure case analysis

---

## Interactive Dashboard

The Streamlit dashboard provides:
- **Image upload** for inspection
- **Real-time prediction** with confidence score
- **Anomaly heatmap** showing defect likelihood regions
- **Segmentation mask** overlay
- **Inspection history** tracking
- **Performance metrics** display

---

## Limitations

- Training performed on CPU with limited epochs — GPU recommended for full training
- MVTec AD is licensed CC BY-NC-SA 4.0 (non-commercial use only)
- Anomaly detection performance varies significantly across categories
- Model is trained on specific product types and may not generalize without fine-tuning
- Real-time inference speed depends on hardware

---

## References

- Bergmann, P., et al. (2019). "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." CVPR.
- Roth, K., et al. (2022). "Towards Total Recall in Industrial Anomaly Detection." CVPR. (PatchCore)
- Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation."
- He, K., et al. (2016). "Deep Residual Learning for Image Recognition."

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The MVTec AD dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

## Author

**Habimana Moise** — Computer Vision & Machine Learning Engineer

- GitHub: [@moise744](https://github.com/moise744)
- LinkedIn: [habimana-moise](https://www.linkedin.com/in/habimana-moise/)
- Email: habimanamoise77@gmail.com
