import os
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon

def load_segmentation_labels(file_path):
    """Reads segmentation data in YOLO format and returns a class-based polygon list"""
    polygons = defaultdict(list)
    if not os.path.exists(file_path):  # If there is no file, return empty
        return polygons
    
    with open(file_path, "r") as f:
        for line in f.readlines():
            data = list(map(float, line.strip().split()))
            cls = int(data[0])
            points = [(data[i], data[i+1]) for i in range(1, len(data), 2)]  # X,Y pairs
            polygons[cls].append(Polygon(points))  # Save as polygon with Shapely
    return polygons

def iou_polygon(polyA, polyB):
    """Calculates the IoU between two polygons"""
    if not polyA.is_valid or not polyB.is_valid:  # Invalid polygon check
        return 0
    
    intersection = polyA.intersection(polyB).area
    union = polyA.union(polyB).area
    return intersection / union if union > 0 else 0

def match_segmentation(gt_polygons, pred_polygons):
    """Calculates IoU by matching polygons by class"""
    iou_scores_per_class = defaultdict(list)
    
    for cls in gt_polygons:
        if cls in pred_polygons:
            for gt_poly in gt_polygons[cls]:
                best_iou = 0
                for pred_poly in pred_polygons[cls]:
                    iou_val = iou_polygon(gt_poly, pred_poly)
                    best_iou = max(best_iou, iou_val)
                iou_scores_per_class[cls].append(best_iou)
    
    return iou_scores_per_class

def process_all_files(test_folder, predict_folder):
    """Calculates the average IoU for each class by comparing all .txt files"""
    test_files = set(f for f in os.listdir(test_folder) if f.endswith(".txt"))
    predict_files = set(f for f in os.listdir(predict_folder) if f.endswith(".txt"))

    common_files = test_files.intersection(predict_files)  # Match files with the same name
    iou_per_class = defaultdict(list)

    for file in common_files:
        gt_polygons = load_segmentation_labels(os.path.join(test_folder, file))
        pred_polygons = load_segmentation_labels(os.path.join(predict_folder, file))
        iou_scores = match_segmentation(gt_polygons, pred_polygons)

        for cls, scores in iou_scores.items():
            iou_per_class[cls].extend(scores)

    # Calculate average IoU for each class
    class_iou_results = {cls: np.mean(scores) if scores else 0 for cls, scores in iou_per_class.items()}

    # Calculate the overall average IoU
    overall_iou = np.mean([iou for scores in iou_per_class.values() for iou in scores]) if iou_per_class else 0
    IoU = ""
    print("\n📌 **Class-wise Segmentation IoU Scores:**")
    for cls, avg_iou in class_iou_results.items():
        print(f"  Class {cls}: IoU = {avg_iou:.4f}")
        IoU += f"  Class {cls}: IoU = {avg_iou:.4f}\n"

    print(f"\n🎯 **Overall Average IoU: {overall_iou:.4f}**")
    
    return overall_iou, IoU
