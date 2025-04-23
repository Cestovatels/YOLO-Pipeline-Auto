import os
import numpy as np
from collections import defaultdict

def load_labels(file_path):
    """Function that reads a YOLO label file and returns bboxes grouped by class"""
    boxes = defaultdict(list)
    if not os.path.exists(file_path):  # Return empty if file does not exist
        return boxes
    
    with open(file_path, "r") as f:
        for line in f.readlines():
            data = line.strip().split()
            cls, x, y, w, h = int(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4])
            boxes[cls].append([x, y, w, h])
    return boxes

def iou(boxA, boxB):
    """Function to calculate IoU between two YOLO format bboxes"""
    xa, ya, wa, ha = boxA
    xb, yb, wb, hb = boxB

    # Convert YOLO format to pixel coordinates
    xa1, ya1, xa2, ya2 = xa - wa / 2, ya - ha / 2, xa + wa / 2, ya + ha / 2
    xb1, yb1, xb2, yb2 = xb - wb / 2, yb - hb / 2, xb + wb / 2, yb + hb / 2

    # Calculate intersection area
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union area
    boxA_area = wa * ha
    boxB_area = wb * hb
    union_area = boxA_area + boxB_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def match_boxes(gt_boxes, pred_boxes):
    """Matches bboxes per class and calculates IoU"""
    iou_scores_per_class = defaultdict(list)
    
    for cls in gt_boxes:
        if cls in pred_boxes:
            for gt in gt_boxes[cls]:
                best_iou = 0
                for pred in pred_boxes[cls]:
                    iou_val = iou(gt, pred)
                    best_iou = max(best_iou, iou_val)
                iou_scores_per_class[cls].append(best_iou)
    
    return iou_scores_per_class

def process_all_files(test_folder, predict_folder):
    """Compares all .txt files and calculates average IoU per class"""
    test_files = set(f for f in os.listdir(test_folder) if f.endswith(".txt"))
    predict_files = set(f for f in os.listdir(predict_folder) if f.endswith(".txt"))

    common_files = test_files.intersection(predict_files)  # Match files with the same name
    iou_per_class = defaultdict(list)

    for file in common_files:
        gt_boxes = load_labels(os.path.join(test_folder, file))
        pred_boxes = load_labels(os.path.join(predict_folder, file))
        iou_scores = match_boxes(gt_boxes, pred_boxes)

        for cls, scores in iou_scores.items():
            iou_per_class[cls].extend(scores)

    # Calculate average IoU per class
    class_iou_results = {cls: np.mean(scores) if scores else 0 for cls, scores in iou_per_class.items()}

    # Calculate overall average IoU
    overall_iou = np.mean([iou for scores in iou_per_class.values() for iou in scores]) if iou_per_class else 0
    IoU = ""
    print("\n📌 **Class-wise IoU Values:**")
    for cls, avg_iou in class_iou_results.items():
        print(f"  Class {cls}: IoU = {avg_iou:.4f}")
        IoU += f"  Class {cls}: IoU = {avg_iou:.4f}\n"

    print(f"\nOverall Average IoU: {overall_iou:.4f}**")
    
    return overall_iou, IoU
