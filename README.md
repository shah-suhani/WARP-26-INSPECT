# INSPECT

Intelligent Neural System for Product Evaluation and Condition Tracking

INSPECT is a deep learning pipeline for automated vehicle damage assessment. A single uploaded car photo is classified as damaged or undamaged, damage is localized and categorized, and a natural-language inspection report is generated — end to end, no human inspector required.

## Overview

The pipeline chains three model stages rather than one end-to-end model, so each stage can be trained, evaluated, and improved independently:

1. **Classify** whether the vehicle is damaged at all (ResNet-50)
2. **Detect and segment** the specific damage regions, if any (YOLOv26m)
3. **Report** location, severity, and repair complexity in natural language (Qwen3.5-VL)

## Key Features

- 🔍 Binary damage classification using a fine-tuned ResNet-50
- 🎯 Multi-class damage detection and instance segmentation across 7 damage categories using YOLOv26m
- 🧠 Natural-language damage reporting using Qwen3.5-VL
- 🖥️ Gradio interface for uploading an image and viewing results end to end
- ⚙️ Single config-driven setup for both training and inference paths
- 🧩 Early-exit design — undamaged images skip detection and reporting entirely

## Architecture

```
                  Uploaded car image
                          │
                          ▼
                ┌───────────────────┐
                │  ResNet-50        │
                │  binary classifier│
                └─────────┬─────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        "undamaged"               "damaged"
              │                       │
              ▼                       ▼
     Return "no damage"      ┌───────────────────┐
                              │  YOLOv26m detector │
                              │  detect + segment  │
                              └─────────┬───────────┘
                                        ▼
                              ┌───────────────────┐
                              │  Qwen3.5-VL         │
                              │  report generator   │
                              └─────────┬───────────┘
                                        ▼
                                 Gradio interface
                     (segmented image, class, damage list, report)
```

## Repository Structure

```
.
├── Backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── router.py              # Gradio interface definition
│   │   ├── models/
│   │   │   ├── Resnet50/
│   │   │   │   └── resnet50.py        # classifier: model, training, inference
│   │   │   └── YOLO26m/
│   │   │       ├── data_converter.py  # VIA JSON -> YOLO segmentation dataset
│   │   │       └── yolo26m.py         # detector: model, training, inference
│   │   ├── utils/
│   │   │   └── report.py              # Qwen3.5-VL report generation
│   │   ├── damage_pipeline.py         # loads models, chains the pipeline
│   │   └── config.py                  # shared config loader
│   ├── main.py                        # entry point — launches the Gradio app
│   ├── config.yaml.example
│   └── requirements.txt
│
├── assets/
│   ├── demo_undamaged.png
│   ├── demo_damaged.png
│   ├── yolov26_results.png
│   └── precision_recall_curves.png
│
├── .gitignore
└── README.md
```

## Core Modules

### 🔍 Classifier — `Backend/app/models/Resnet50/resnet50.py`

ResNet-50 with a custom head (`Dropout → Linear(2)`) fine-tuned to answer one binary question: is this vehicle damaged? If undamaged, the pipeline exits immediately and skips detection and reporting.

### 🎯 Detector — `Backend/app/models/YOLO26m/yolo26m.py`

Localizes and segments damage across 7 classes: broken glass, broken lamp, dent, hole, lost part, paint scratch, and torn. Returns bounding boxes, class labels, confidence scores, and an annotated segmentation.

### 🧠 Reporter — `Backend/app/utils/report.py`

Passes the segmented image and detected damage classes to Qwen3.5-VL, prompted as an expert insurance damage inspector, returning damage location and type, severity, estimated repair complexity, and a summary.

### 🧩 Pipeline — `Backend/app/damage_pipeline.py`

Loads all three models once and chains classification → detection → reporting, with the early-exit branch for undamaged vehicles.

### 🖥️ Interface — `Backend/app/api/router.py` + `Backend/main.py`

`router.py` builds the Gradio `Interface` around `damage_pipeline.analyze()`; `main.py` launches it.

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Classification | ResNet-50 (torchvision) |
| Detection & Segmentation | YOLOv26m (Ultralytics) |
| Vision-Language Reporting | Qwen3.5-VL (via Hugging Face Router) |
| Interface | Gradio |
| Config Management | YAML |
| HTTP Client | httpx |

## Performance & Results

### YOLOv26m detection and segmentation

| Class | Mask precision | Mask recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| All | 0.665 | 0.501 | 0.523 | 0.308 |
| Broken glass | 0.856 | 0.789 | 0.836 | 0.608 |
| Broken lamp | 0.669 | 0.564 | 0.561 | 0.290 |
| Dent | 0.618 | 0.339 | 0.361 | 0.159 |
| Hole | 0.663 | 0.481 | 0.518 | 0.284 |
| Lost part | 0.787 | 0.677 | 0.719 | 0.511 |
| Paint scratch | 0.490 | 0.289 | 0.282 | 0.119 |
| Torn | 0.574 | 0.368 | 0.384 | 0.186 |

Overall mask precision across all classes reaches 0.99 at a confidence threshold of 1.0, and overall mask recall reaches 0.70 at a confidence threshold of 0.0.


### Pipeline in action

**Undamaged vehicle** — classifier correctly identifies no damage and the pipeline exits before detection:
<img width="1600" height="611" alt="undamaged" src="https://github.com/user-attachments/assets/1901dd81-e530-43b9-bccb-588917d0c7da" />


[Undamaged classification result]

**Damaged vehicle** — front bumper lost parts and torn material detected, segmented, and described:
<img width="1600" height="779" alt="image" src="https://github.com/user-attachments/assets/c02c233a-2b1b-4ea6-bd3c-c905541953d5" />


[Damaged classification and report]

## Getting Started

### Install

```bash
cd Backend
pip install -r requirements.txt
```

### Configure

```bash
cp config.yaml.example config.yaml
```


### Run the app

```bash
python main.py
```

## Training

Training scripts live alongside the model code they belong to, and are run as modules from inside `Backend/`:

**1. Convert raw annotations into a YOLO segmentation dataset:**

```bash
python -m app.models.YOLO26m.data_converter
```

**2. Train the ResNet-50 classifier:**

```bash
python -m app.models.Resnet50.resnet50
```

Saves the best checkpoint (by test accuracy) to the path configured under `paths.output_model`.

**3. Train the YOLOv26m detector:**

```bash
python -m app.models.YOLO26m.yolo26m
```

Resumes automatically from the last checkpoint if a run with the same name already exists, otherwise starts from the pretrained YOLOv26m-seg weights.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Qwen](https://github.com/QwenLM/Qwen) via Hugging Face Router
- [Gradio](https://www.gradio.app/)
