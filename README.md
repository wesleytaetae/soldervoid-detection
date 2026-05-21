# Solder Void Detection for PQFN X-ray Images

An end-to-end Industrial Machine Learning & Computer Vision Metrology Pipeline that automates solder void detection on a semiconductor factory floor. The system processes raw X-ray images, segments microscopic solder voids, and computes defect ratios to determine pass/fail.

---

## Project Architecture Overview

**Pipeline**
```
[ Raw X-Rays ] ➔ [ Data Engineering & CLAHE ] ➔ [ U-Net ResNet34 Core ] ➔ [ OpenCV Metrology Overlay ]
```

| Layer | Purpose | Key Techniques |
|---|---|---|
| Data Engineering | Enhance and prepare raw X-rays | CLAHE contrast boosting, LabelMe vector rasterization, class stacking |
| Deep Learning Core | Segment solder voids | U-Net with ResNet34 encoder, weighted loss |
| Metrology Overlay | Compute defect ratio | Pixel-based measurement and thresholding |

---

## 1) Data Engineering Layer

**Contrast Optimization (CLAHE)**
Raw X-rays are noisy and low-contrast. CLAHE amplifies local gradients to reveal sub-pixel air bubbles for both humans and the model.

**Vector Rasterization**
LabelMe JSON vectors are compiled into integer PNG masks using the Painter's Algorithm with strict class order:
- 0: Background
- 1: Solid Solder
- 2: Void

**Synchronized Augmentation**
Albumentations applies synchronized transforms to ensure the same spatial matrix is applied to image and mask (rotations, flips, crops).

---

## 2) Deep Learning Training Engine (Core Focus)

**Model Architecture**
- U-Net semantic segmentation network
- ResNet34 encoder backbone (pretrained on ImageNet)
- Single-channel grayscale input
- 3 output classes: Background / Solder / Void

**Loss Engineering**
- Weighted Cross-Entropy Loss
- Void class heavily penalized with weight 5.0 to reduce false negatives

**Hardware Acceleration**
- AMP (Automatic Mixed Precision) via `autocast` + `GradScaler`
- Optimized for NVIDIA GPUs (e.g., RTX 3080)

---

### Training Parameters

| Parameter | Value |
|---|---|
| Batch size | 8 |
| Epochs | 500 |
| Patience | 15 |
| Learning rate | 1e-4 |
| Classes | 3 |
| Validation split | 0.2 |
| Optimizer | AdamW |
| Loss | Weighted Cross-Entropy |

---

### Data Augmentation (Training)

| Transform | Notes |
|---|---|
| PadIfNeeded | Ensures 512x512 |
| RandomCrop | 512x512 crop |
| RandomRotate90 | 90-degree rotations |
| HorizontalFlip | Random |
| VerticalFlip | Random |
| ToTensorV2 | Converts to tensor |

---

## 3) Production Inference & Metrology

The inference engine computes physical defect ratio using pixel counts:

$$
	ext{Void Ratio} = \left( \frac{\text{Void Pixels}}{\text{Solder Pixels} + \text{Void Pixels}} \right) \times 100
$$

This enables real-time pass/fail logic for microchip inspection.

---

## How to Train

1. Place raw X-ray images in `data/1.input/`.
2. Run the preprocessing pipeline:

```
python main.py resize
python main.py prepare
python main.py mask
```

3. Train the model:

```
python main.py train
```

The training command uses the parameters listed above and saves weights to `best_unet_model.pth`.

---

## How to Run the UI

Launch the desktop application:

```
python app.py
```

---

## Notes

- Designed for industrial datasets with strong class imbalance
- Tailored to PQFN device solder void segmentation
- Full pipeline integrates preprocessing, training, inference, and metrology
