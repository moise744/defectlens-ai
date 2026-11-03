# DefectLens AI — Dataset Guide

## Primary Dataset: MVTec AD

**MVTec Anomaly Detection (MVTec AD)** is the primary benchmark dataset for this project.

| Property | Value |
|----------|-------|
| Source | MVTec Software GmbH |
| License | CC BY-NC-SA 4.0 (non-commercial) |
| Size | ~4.9 GB |
| Images | 5,354 training + 1,725 test |
| Categories | 15 |
| Annotation | Pixel-level ground-truth masks |

### Categories

```
Textures:          carpet, grid, leather, tile, wood
Objects:           bottle, cable, capsule, hazelnut, metal_nut,
                   pill, screw, toothbrush, transistor, zipper
```

### Download Instructions

1. Register and download from the official site:
   ```
   https://www.mvtec.com/company/research/datasets/mvtec-ad
   ```

2. Extract to the `data/` directory:
   ```
   data/
   └── mvtec_ad/
       ├── bottle/
       │   ├── train/
       │   │   └── good/          ← defect-free training images
       │   └── test/
       │       ├── good/          ← defect-free test images
       │       ├── broken_large/  ← defect test images
       │       ├── broken_small/
       │       └── contamination/
       ├── cable/
       ├── carpet/
       ... (15 categories)
   ```

3. Verify dataset integrity:
   ```bash
   python scripts/download_dataset.py --verify --data_dir data/mvtec_ad
   ```

### Dataset Limitations

- **Non-commercial only.** Cannot be used for commercial products without separate licensing.
- Training images contain only normal (defect-free) samples. Defect samples appear only in the test split.
- Defect types vary significantly between categories — per-category models often outperform unified models.
- Small number of test defect images per category (typically 70–150) limits statistical reliability.

---

## Alternative / Supplementary Datasets

### MVTec AD 2

- Extended version with additional categories and defect types.
- Download: https://www.mvtec.com/company/research/datasets/mvtec-ad-2

### BTAD (BeanTech Anomaly Detection)

- 3 industrial objects with real-world variability.
- Download: http://www.eoarc.net/BTAD/

### KolektorSDD2

- Surface defect dataset for electrical commutators.
- Download: https://www.vicos.si/resources/kolektorsdd2/

---

## Expected Directory Structure

```
data/
├── mvtec_ad/           ← primary dataset
│   ├── bottle/
│   ├── cable/
│   └── ...
├── custom/             ← your own images (optional)
│   ├── train/good/
│   └── test/
│       ├── good/
│       └── defect/
└── README.md           ← this file
```

Do not commit dataset files to git. They are excluded by `.gitignore`.
