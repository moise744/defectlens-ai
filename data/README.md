# MVTec AD Dataset — Setup Instructions

## Overview

DefectLens AI uses the **MVTec Anomaly Detection (MVTec AD)** dataset for training and evaluation.

| Property | Value |
|----------|-------|
| **Full Name** | MVTec Anomaly Detection Dataset |
| **Source** | https://www.mvtec.com/company/research/datasets/mvtec-ad |
| **License** | CC BY-NC-SA 4.0 (Non-Commercial Research Use) |
| **Total Images** | 5,354 high-resolution images |
| **Image Sizes** | 700×700 to 1024×1024 pixels |
| **Categories** | 15 industrial product and texture categories |
| **Annotations** | Per-pixel ground truth masks for all defect types |

## Categories

The dataset includes 15 categories:

**Object Categories (10):**
bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper

**Texture Categories (5):**
carpet, grid, leather, tile, wood

## Download Instructions

> **Note:** MVTec AD requires registration before download for academic and research use.

1. Visit the official dataset page:
   ```
   https://www.mvtec.com/company/research/datasets/mvtec-ad
   ```

2. Complete the registration form with your name, institution, and intended use.

3. Download the full dataset archive (`mvtec_anomaly_detection.tar.xz`, ~4.9 GB) or individual category archives.

4. Extract the archive to `data/mvtec_ad/`:
   ```powershell
   # Windows (using 7-Zip or Windows built-in)
   Expand-Archive -Path mvtec_anomaly_detection.tar.xz -DestinationPath data/mvtec_ad/
   
   # Or with tar (if available)
   tar -xf mvtec_anomaly_detection.tar.xz -C data/mvtec_ad/
   ```

## Expected Directory Structure

After extraction, your `data/mvtec_ad/` directory should look like:

```
data/mvtec_ad/
├── bottle/
│   ├── train/
│   │   └── good/           # 209 normal training images
│   ├── test/
│   │   ├── good/           # 83 normal test images
│   │   ├── broken_large/   # Defective images
│   │   ├── broken_small/
│   │   └── contamination/
│   └── ground_truth/
│       ├── broken_large/   # Pixel-level defect masks (binary PNG)
│       ├── broken_small/
│       └── contamination/
├── cable/
│   └── ...
├── capsule/
│   └── ...
... (12 more categories)
```

## Verify Dataset

After extraction, verify the dataset structure:

```powershell
python scripts/download_dataset.py --verify --data-dir data/mvtec_ad
python scripts/download_dataset.py --stats --data-dir data/mvtec_ad
```

## Dataset Statistics

| Category | Train (Normal) | Test (Normal) | Test (Defective) | Defect Types |
|----------|----------------|---------------|------------------|--------------|
| bottle | 209 | 83 | 63 | 3 |
| cable | 224 | 58 | 92 | 8 |
| capsule | 219 | 23 | 109 | 5 |
| carpet | 280 | 28 | 89 | 5 |
| grid | 264 | 21 | 57 | 5 |
| hazelnut | 391 | 40 | 70 | 4 |
| leather | 245 | 32 | 92 | 5 |
| metal_nut | 220 | 22 | 93 | 4 |
| pill | 267 | 26 | 141 | 7 |
| screw | 320 | 41 | 119 | 5 |
| tile | 230 | 33 | 84 | 5 |
| toothbrush | 60 | 12 | 30 | 1 |
| transistor | 213 | 60 | 40 | 4 |
| wood | 247 | 19 | 60 | 5 |
| zipper | 240 | 32 | 119 | 7 |

## Citation

If you use MVTec AD in your research, please cite:

```bibtex
@inproceedings{bergmann2019mvtec,
  title={MVTec AD -- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author={Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

## License

The MVTec AD dataset is licensed under the
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.

- ✅ Academic and research use permitted
- ✅ Adaptation and redistribution permitted with same license
- ❌ Commercial use **not permitted**
- ❌ Do **not** include dataset images in this repository

See https://creativecommons.org/licenses/by-nc-sa/4.0/ for full license terms.
