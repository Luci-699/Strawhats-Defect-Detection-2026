# Strawhats Defect Detection 2026 🔍

**Team SafePath | RVCE Hackathon 2026**
*Problem Statement 5: Automated Inspection System for Cracks & Fractures in Finished Materials*

---

## 🎯 Problem Statement

Quality control in manufacturing is currently performed **manually by operators**, leading to:
- Human fatigue → missed defects → product failures
- High rejection rates → factory losses
- Inconsistent results across shifts

**Our Solution:** A fully automated, real-time multi-material crack and defect inspection system using deep learning + morphological analysis.

---

## 🏗️ System Architecture

```
                        ┌─────────────────────────────────────┐
  Camera / Image  ──►   │   Material Classifier (ResNet18)    │
                        │   Steel / Aluminum / Wood           │
                        └────────────┬────────────────────────┘
                                     │ Routes to material-specific model
              ┌──────────────────────┼───────────────────────┐
              ▼                      ▼                        ▼
    ┌──────────────┐      ┌──────────────────┐    ┌──────────────────┐
    │ YOLOv10m     │      │   YOLOv10m       │    │   YOLOv10m       │
    │ Steel Model  │      │ Aluminum Model   │    │   Wood Model     │
    │ 15 classes   │      │   10 classes     │    │   10 classes     │
    └──────┬───────┘      └────────┬─────────┘    └────────┬─────────┘
           │                       │                        │
           └───────────────────────┼────────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │   Morphology Fusion Layer    │
                    │  ┌─────────────────────────┐ │
                    │  │  11 Morphological       │ │
                    │  │  Descriptors (OpenCV)   │ │
                    │  └──────────┬──────────────┘ │
                    │             ▼                 │
                    │  MorphologyEncoder (MLP)      │
                    │  11D → 64D → 128D             │
                    │             ▼                 │
                    │  CrossAttentionFusion         │
                    │  Q=Visual, K=V=Morphology     │
                    │             ▼                 │
                    │  Classification Head          │
                    └──────────────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │   Real-Time HUD Overlay      │
                    │   Defect Class + Confidence  │
                    │   Material + FPS Counter     │
                    └──────────────────────────────┘
```

---

## 📦 Dataset

| Material | Source | Images | Classes |
|----------|--------|--------|---------|
| Steel | NEU-DET + Severstal (Kaggle) | ~5,200 | 15 |
| Aluminum | AMID Dataset (Kaggle) | ~4,700 | 10 |
| Wood | Wood Defect Dataset (Kaggle) | ~7,400 | 10 |
| Material Classifier | All 3 combined | ~4,700 | 3 |
| **Total** | | **~20,800** | **26+** |

### Steel Defect Classes (15)
`crazing` · `inclusion` · `patches` · `pitted_surface` · `rolled_in_scale` · `scratches` · `crease` · `crescent_gap` · `water_spot` · `welding_line` · `silk_spot` · `oil_spot` · `punching` · `rolling_pit` · `waist_fold`

### Aluminum Defect Classes (10)
`crease` · `crescent_gap` · `water_spot` · `welding_line` · `silk_spot` · `inclusion` · `oil_spot` · `punching` · `rolling_pit` · `waist_fold`

### Wood Defect Classes (10)
`live_knot` · `dead_knot` · `knot_with_crack` · `crack` · `resin` · `marrow` · `quartzite` · `missing_knot` · `blue_stain` · `overgrown`

---

## 🧠 Technical Innovation

### 1. Morphology-Aware Detection (Novel Contribution)
Unlike standard YOLO-only systems, we augment visual detection with **11 mathematical morphological descriptors** computed per defect region:

| # | Descriptor | Captures |
|---|-----------|---------|
| 1 | Area | Defect size |
| 2 | Perimeter | Boundary length |
| 3 | Aspect Ratio | Elongation (cracks vs. spots) |
| 4 | Circularity | Shape regularity |
| 5 | Solidity | Convexity (pitting vs. crazing) |
| 6 | Convex Hull Perimeter | Outer boundary |
| 7 | Compactness | Area/ellipse ratio |
| 8 | Eccentricity | Stretch direction |
| 9 | Edge Density | Surface roughness |
| 10 | Skeleton Orientation | Crack direction angle |
| 11 | Texture Roughness | RMS surface deviation |

### 2. Cross-Attention Fusion
```
Q = W_Q · f_visual   (from YOLOv10m ROI features)
K = W_K · f_morph    (from MorphologyEncoder)
V = W_V · f_morph

fused = LayerNorm( Concat(Attention(Q,K,V), f_visual) )
```

### 3. DSP Preprocessing Pipeline
Each image undergoes: `CLAHE → Gaussian Blur → Otsu Threshold → Attention Map`
This enhances defect visibility before feeding to the model.

---

## 🚀 Quick Start

### Prerequisites
```bash
conda create -n RVCE python=3.11
conda activate RVCE
pip install -r requirements.txt
```

### Run Live Demo (Camera)
```bash
python inference/realtime_demo.py
```

### Run Live Demo (Image/Video)
```bash
python inference/realtime_demo.py --source path/to/image.jpg
python inference/realtime_demo.py --source path/to/video.mp4
```

### Demo Controls
| Key | Action |
|-----|--------|
| `SPACE` | Freeze frame for judges |
| `S` | Save screenshot |
| `+` / `-` | Adjust confidence threshold |
| `P` | Pause/Resume |
| `Q` | Quit |

### Train All Models (Full Pipeline)
```bash
# Stage 1: Material Classifier (ResNet18)
# Stage 2-4: Material-Specific YOLO (Steel, Aluminum, Wood)
# Stage 5-7: Morphology Fusion (per material)
python train_all.py --stages 1 2 3 4 5 6 7
```

### Train Specific Stages
```bash
python train_all.py --stages 2 3 4     # YOLO only
python train_all.py --stages 5 6 7     # Fusion only
```

### Evaluate
```bash
python evaluation/evaluate.py
```

---

## 📁 Project Structure

```
Strawhats-Defect-Detection-2026/
│
├── data/                          # Datasets (Kaggle downloads)
│   ├── processed/steel_unified/   # Steel: 15 classes
│   ├── processed_aluminum/        # Aluminum: 10 classes
│   ├── processed/wood_10class/    # Wood: 10 classes
│   └── material_classifier/       # 3-class routing dataset
│
├── models/                        # Neural network modules
│   ├── yolo_backbone.py           # YOLOv10m feature extractor
│   ├── cross_attention.py         # CrossAttentionFusion module
│   └── classification_head.py     # Morphology classifier head
│
├── morphology/                    # Morphological analysis
│   ├── preprocessing.py           # CLAHE + Gaussian pipeline
│   ├── feature_extractor.py       # 11 morphological descriptors
│   ├── encoder.py                 # MLP: 11D → 128D embedding
│   ├── attention_map.py           # Binary mask → attention map
│   └── dsp_filters.py             # DSP preprocessing filters
│
├── training/                      # Training scripts
│   ├── config.yaml                # All hyperparameters
│   ├── train_yolo_baseline.py     # YOLOv10m training (Stages 2-4)
│   ├── train_morphology_fusion.py # Fusion training (Stages 5-7)
│   ├── train_material_classifier.py # ResNet18 (Stage 1)
│   └── losses.py                  # MorphologyAwareLoss
│
├── inference/
│   └── realtime_demo.py           # Live demo with HUD overlay
│
├── evaluation/
│   └── evaluate.py                # mAP50, accuracy, per-class metrics
│
├── scripts/                       # Data preparation utilities
│   ├── download_neu_det.py
│   ├── download_severstal.py
│   ├── convert_annotations.py
│   └── split_dataset.py
│
├── demo/
│   └── print_cards/               # Printed sample cards for demo day
│       ├── print_sheet.html       # A4 print-ready HTML
│       ├── Steel/                 # 6 steel defect cards
│       ├── Aluminum/              # 10 aluminum defect cards
│       └── Wood/                  # 9 wood defect cards
│
├── train_all.py                   # Master training pipeline
└── requirements.txt
```

---

## 📊 Training Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Model | YOLOv10m | Best speed/accuracy trade-off for 6GB GPU |
| Image Size | 800×800 | Higher resolution for fine defect detail |
| Batch Size | 4 | Safe within 6GB VRAM at imgsz=800 |
| Epochs | 150 (YOLO) / 100 (Fusion) | With patience=30 early stopping |
| Optimizer | AdamW | Better generalization than SGD |
| LR Schedule | Cosine Annealing | Smooth convergence |
| Augmentation | Mosaic + MixUp + Copy-Paste + HSV | Robust to real-world variation |
| AMP | FP16 Mixed Precision | 1.5-2× faster training |

---

## 📈 Results

> *Full results available after training completes. See `evaluation/evaluate.py` output.*

| Model | mAP50 | Val Accuracy |
|-------|-------|-------------|
| Material Classifier (ResNet18) | — | ~97% |
| Steel YOLO (baseline 50ep) | ~73% | — |
| Steel Morphology Fusion | — | ~73.7% |
| Steel YOLO v2 (150ep) | *training* | — |
| Aluminum YOLO v2 (150ep) | *training* | — |
| Wood YOLO v2 (150ep) | *training* | — |

---

## 👥 Team

**Team SafePath | RVCE Hackathon 2026**

| Member | Role |
|--------|------|
| [Member 1] | Model architecture + training pipeline |
| [Member 2] | Data preprocessing + augmentation |
| [Member 3] | Morphology module + fusion |
| [Member 4] | Demo + evaluation |

---

## 🏆 Judging Criteria Coverage

| Criterion | Our Implementation |
|-----------|-------------------|
| **Problem Understanding** | Addresses factory QC automation for 3 material types |
| **Technical Complexity** | YOLOv10m + Morphology Fusion + Cross-Attention — 3-stage pipeline |
| **Feasibility** | Runs real-time at 15-30 FPS on RTX 4050 |
| **Functionality** | Detects 26 defect classes across Steel, Aluminum, Wood |
| **Code Quality** | Modular architecture, docstrings, type hints, config-driven |
| **Teamwork** | Git history with tracked contributions |
| **Model Training** | 150-epoch training with augmentation + early stopping |
| **Output Response** | Live HUD: class label, confidence, FPS, defect count |
| **Demo** | Webcam demo + physical sample cards as fallback |
