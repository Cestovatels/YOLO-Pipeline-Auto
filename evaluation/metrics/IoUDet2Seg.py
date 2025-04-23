import os
import numpy as np
from collections import defaultdict

def convert_polygon_to_bbox(points):
    """Converts points in (x1, y1, ..., xn, yn) format to an axis-aligned bounding box"""
    xs = points[0::2]
    ys = points[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Return as center-x, center-y, width, height in YOLO format
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return [x_center, y_center, width, height]

def load_labels(file_path, is_segmentation=False):
    """
    Reads a YOLO format label file.
    - If is_segmentation=True, it converts segmentation to a bbox.
    - If is_segmentation=False, it directly reads the bbox.
    """
    boxes = defaultdict(list)
    if not os.path.exists(file_path):
        return boxes

    with open(file_path, "r") as f:
        for line in f.readlines():
            data = line.strip().split()
            cls = int(data[0])
            if is_segmentation:
                points = list(map(float, data[1:]))
                bbox = convert_polygon_to_bbox(points)
            else:
                bbox = list(map(float, data[1:5]))
            boxes[cls].append(bbox)
    return boxes

def iou(boxA, boxB):
    """Calculates IoU for two bounding boxes in YOLO format"""
    xa, ya, wa, ha = boxA
    xb, yb, wb, hb = boxB

    xa1, ya1, xa2, ya2 = xa - wa / 2, ya - ha / 2, xa + wa / 2, ya + ha / 2
    xb1, yb1, xb2, yb2 = xb - wb / 2, yb - hb / 2, xb + wb / 2, yb + hb / 2

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    boxA_area = wa * ha
    boxB_area = wb * hb
    union_area = boxA_area + boxB_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def match_boxes(gt_boxes, pred_boxes):
    """Matches bounding boxes class-wise and calculates IoU"""
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
    """Compares all files and calculates average IoU"""
    test_files = set(f for f in os.listdir(test_folder) if f.endswith(".txt"))
    predict_files = set(f for f in os.listdir(predict_folder) if f.endswith(".txt"))

    common_files = test_files.intersection(predict_files)
    iou_per_class = defaultdict(list)

    for file in common_files:
        gt_boxes = load_labels(os.path.join(test_folder, file), is_segmentation=True)
        pred_boxes = load_labels(os.path.join(predict_folder, file), is_segmentation=False)
        iou_scores = match_boxes(gt_boxes, pred_boxes)

        for cls, scores in iou_scores.items():
            iou_per_class[cls].extend(scores)

    class_iou_results = {cls: np.mean(scores) if scores else 0 for cls, scores in iou_per_class.items()}
    overall_iou = np.mean([iou for scores in iou_per_class.values() for iou in scores]) if iou_per_class else 0
    IoU = ""
    print("\n📌 **Class-wise IoU Scores:**")
    for cls, avg_iou in class_iou_results.items():
        print(f"  Class {cls}: IoU = {avg_iou:.4f}")
        IoU += f"  Class {cls}: IoU = {avg_iou:.4f}\n"

    print(f"\n🔍 Overall Average IoU: {overall_iou:.4f}")
    
    return overall_iou, IoU
