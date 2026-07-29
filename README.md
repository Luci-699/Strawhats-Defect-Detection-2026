# Morphology-Aware Industrial Crack Inspection System

**Team:** Strawhat Pirates  
**Event:** India CV Hackathon 2026, RVCE  

---

## Overview

The **Morphology-Aware Industrial Crack Inspection System** is an end-to-end computer vision solution designed for automated surface defect detection, precise geometric morphology analysis (skeletonization, crack width, length, orientation, branching), and explainable real-time industrial deployment.

---

## System Architecture

```
+-----------------------------------------------------------------------------------+
|                                 INPUT SOURCE                                      |
|            [ NEU-DET / Severstal Datasets / Live Camera Stream ]                 |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                             MORPHOLOGY-AWARE BACKBONE                             |
|                                                                                   |
|  +---------------------------+                +--------------------------------+  |
|  | Multi-Scale Augmentation  | -------------> | YOLOv8 / DeepLabV3 / UNet     |  |
|  |  (Morphology-Preserving)  |                | Detection & Segmentation Model |  |
|  +---------------------------+                +--------------------------------+  |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                           GEOMETRIC MORPHOLOGY ENGINE                             |
|                                                                                   |
|  +-------------------+  +------------------+  +-------------------+  +----------+ |
|  |  Skeletonization  |  | Distance Transform|  | Width/Length Calc |  | Tortuosity| |
|  |   & Medial Axis   |  |   (Max Width)    |  |  & Orientation    |  | Analysis | |
|  +-------------------+  +------------------+  +-------------------+  +----------+ |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        EXPLAINABILITY & HARDWARE INFERENCE                        |
|                                                                                   |
|  +-------------------+  +------------------+  +-------------------+  +----------+ |
|  |   Grad-CAM / SHAP |  | FastAPI Real-time|  | PySerial Hardware |  | Dashboard| |
|  | Visual Inspection |  | Inference API    |  |  Trigger / Relay  |  |  Control | |
|  +-------------------+  +------------------+  +-------------------+  +----------+ |
+-----------------------------------------------------------------------------------+
```

---

## Project Structure

```
RVCE/
├── data/                       # Datasets and processed annotations
│   ├── NEU-DET/                # Raw NEU-DET steel surface defect dataset
│   ├── Severstal/              # Raw Severstal steel defect dataset
│   ├── processed/              # Formatted YOLO/Segmentation dataset
│   │   ├── images/             # train / val / test images
│   │   └── labels/             # train / val / test label text files
│   ├── real_samples/           # Real-world high-res camera samples
│   └── dataset.yaml            # Dataset YAML configuration file
├── morphology/                 # Morphology extraction algorithms
│   ├── __init__.py             # Module initialization
│   ├── skeleton.py             # Thinning, medial axis, skeletonization
│   └── metrics.py              # Geometric width, tortuosity, branching analysis
├── models/                     # Deep learning architectures & wrappers
│   ├── __init__.py             # Module initialization
│   └── detector.py             # YOLOv8 / custom model definitions
├── hardware/                   # Physical inspection rig integration
│   └── serial_controller.py    # PySerial trigger & alert system
├── training/                   # Model training and fine-tuning pipelines
│   ├── __init__.py             # Module initialization
│   └── train.py                # Training execution script
├── evaluation/                 # Metrics calculation & benchmark scripts
│   ├── __init__.py             # Module initialization
│   └── evaluate.py             # Precision, recall, mAP, morph accuracy
├── inference/                  # Production inference & API services
│   ├── __init__.py             # Module initialization
│   └── app.py                  # FastAPI real-time REST endpoint
├── dashboard/                  # Interactive UI and control panel
│   └── README.md               # Dashboard documentation (Akash's area)
├── docs/presentation/          # Slide decks, reports, and diagrams
├── scripts/                    # Utility and dataset processing scripts
│   ├── __init__.py             # Module initialization
│   └── preprocess.py           # Dataset conversion and formatting
├── tests/                      # Unit tests & validation suite
│   ├── __init__.py             # Module initialization
│   └── test_morphology.py      # Test suite for morphology functions
├── runs/                       # Output directory for training runs & artifacts
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore configuration
└── README.md                   # Project overview & documentation
```

---

## Quick Start

### 1. Environment Setup

Clone the repository and install dependencies in a virtual environment:

```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Dataset Preparation

Download NEU-DET or Severstal datasets and place them under `data/NEU-DET/` and `data/Severstal/` respectively. Run the preprocessing script:

```bash
python -m scripts.preprocess
```

### 3. Training

Train the defect detection model:

```bash
python -m training.train --config data/dataset.yaml --epochs 50
```

### 4. Running Inference & API

Launch the real-time inference FastAPI server:

```bash
uvicorn inference.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Defect Classes

The system detects and characterizes six primary industrial surface defect categories:

1. **Crazing (`crazing`)**
2. **Inclusion (`inclusion`)**
3. **Patches (`patches`)**
4. **Pitted Surface (`pitted_surface`)**
5. **Rolled-in Scale (`rolled_in_scale`)**
6. **Scratches (`scratches`)**

---

## Credits & References

Developed by **Team Strawhat Pirates** for **India CV Hackathon 2026** at RV College of Engineering (RVCE).

- NEU Surface Defect Database (Northeastern University, China)
- Severstal Steel Defect Detection (Kaggle)
- Ultralytics YOLOv8 Framework
- Scikit-Image & OpenCV Morphology Libraries
