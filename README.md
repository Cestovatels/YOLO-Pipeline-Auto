# 🧠 YOLO Auto Trainer Pipeline

A fully automated end-to-end pipeline for training, validating, and evaluating YOLOv8/v11 models for **object detection** and **instance segmentation** tasks. This framework is designed to streamline the entire machine learning workflow — from dataset preprocessing to model training and generating evaluation reports.

> Supports both detection and segmentation tasks using Ultralytics' YOLO models.

---

## 🚀 Features

- 🔄 **Automated Dataset Preprocessing** (Split into train/test/validation)
- 📁 YOLO-compatible directory structure creation
- 🧪 Integrated **model training**, **validation**, and **inference**
- 📊 Evaluation with:
  - ROC Curve
  - IoU, Dice, Jaccard Metrics
  - Prediction Visualization
  - Confusion Matrix
  - Comprehensive Report Generation
  - Overall Performance Metrics
  - Ground Truth & Prediction Overlay
- 🧬 Data Augmentation Support (HSV, flipping, mosaic, etc.)
- 🖼️ Works with both **YOLOv8** and **YOLOv11** (detection & segmentation variants)

---

## 📦 Installation

```bash
Python 3.10.11 or higher is required.
git clone https://github.com/your-username/yolo-auto-trainer.git
cd yolo-auto-trainer
pip install -r requirements.txt
```


## 🧰 Usage
```bash
python main.py \
  --image_dir Dataset/images \
  --label_dir Dataset/labels \
  --classes cat dog rabbit \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --model yolov8s-seg \
  --task segmentation \
  --training_name MyExperiment
```
## ✅ For all available arguments, check:

```bash
python main.py --help
```

## 📁 Directory Structure
```bash
Results-Yolo-Auto/
└── MyExperiment/
    ├── Dataset_MyExperiment/
    │   ├── train/
    │   ├── test/
    │   ├── valid/
    │   └── custom.yaml
    ├── Train-MyExperiment/
    ├── Test-Validation-MyExperiment/
    ├── Pred-MyExperiment/
    ├── report.docx
    ├── report_excel.xlsx 
    └── roc_curve.png
```


## 🤝 Contributions
Feel free to open an issue or submit a pull request. Feature requests and feedback are always welcome!
