import os
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon

def load_segmentation_labels(file_path):
    """Reads segmentation data in YOLO format and returns a class-wise list of polygons"""
    polygons = defaultdict(list)
    if not os.path.exists(file_path):  # Return empty if file doesn't exist
        return polygons
    
    with open(file_path, "r") as f:
        for line in f.readlines():
            data = list(map(float, line.strip().split()))
            cls = int(data[0])
            points = [(data[i], data[i+1]) for i in range(1, len(data), 2)]  # X,Y pairs
            polygons[cls].append(Polygon(points))  # Store as polygon using Shapely
    return polygons

def dice_polygon(polyA, polyB):
    """Calculates DICE (F1-score) between two polygons"""
    if not polyA.is_valid or not polyB.is_valid:
        return 0
    
    intersection = polyA.intersection(polyB).area
    total_area = polyA.area + polyB.area
    return (2 * intersection) / total_area if total_area > 0 else 0

def jaccard_polygon(polyA, polyB):
    """Calculates Jaccard (IoU) index between two polygons"""
    if not polyA.is_valid or not polyB.is_valid:
        return 0
    
    intersection = polyA.intersection(polyB).area
    union = polyA.union(polyB).area
    return intersection / union if union > 0 else 0

def match_segmentation(gt_polygons, pred_polygons):
    """Matches polygons class-wise and calculates DICE and Jaccard"""
    dice_scores_per_class = defaultdict(list)
    jaccard_scores_per_class = defaultdict(list)
    
    for cls in gt_polygons:
        if cls in pred_polygons:
            for gt_poly in gt_polygons[cls]:
                best_dice = 0
                best_jaccard = 0
                for pred_poly in pred_polygons[cls]:
                    dice_val = dice_polygon(gt_poly, pred_poly)
                    jaccard_val = jaccard_polygon(gt_poly, pred_poly)
                    best_dice = max(best_dice, dice_val)
                    best_jaccard = max(best_jaccard, jaccard_val)
                dice_scores_per_class[cls].append(best_dice)
                jaccard_scores_per_class[cls].append(best_jaccard)
    
    return dice_scores_per_class, jaccard_scores_per_class

def process_all_files(test_folder, predict_folder):
    """Compares all .txt files to calculate average DICE and Jaccard for each class"""
    test_files = set(f for f in os.listdir(test_folder) if f.endswith(".txt"))
    predict_files = set(f for f in os.listdir(predict_folder) if f.endswith(".txt"))

    common_files = test_files.intersection(predict_files)  # Match files with the same name
    dice_per_class = defaultdict(list)
    jaccard_per_class = defaultdict(list)

    for file in common_files:
        gt_polygons = load_segmentation_labels(os.path.join(test_folder, file))
        pred_polygons = load_segmentation_labels(os.path.join(predict_folder, file))
        dice_scores, jaccard_scores = match_segmentation(gt_polygons, pred_polygons)

        for cls, scores in dice_scores.items():
            dice_per_class[cls].extend(scores)
        for cls, scores in jaccard_scores.items():
            jaccard_per_class[cls].extend(scores)

    # Compute average DICE and Jaccard for each class
    class_dice_results = {cls: np.mean(scores) if scores else 0 for cls, scores in dice_per_class.items()}
    class_jaccard_results = {cls: np.mean(scores) if scores else 0 for cls, scores in jaccard_per_class.items()}

    # Compute overall average DICE and Jaccard
    overall_dice = np.mean([dice for scores in dice_per_class.values() for dice in scores]) if dice_per_class else 0
    overall_jaccard = np.mean([jaccard for scores in jaccard_per_class.values() for jaccard in scores]) if jaccard_per_class else 0

    print("\n📌 **Class-wise Segmentation Metrics:**")
    jaccards_dice = ""
    for cls in sorted(set(class_dice_results.keys()).union(class_jaccard_results.keys())):
        avg_dice = class_dice_results.get(cls, 0)
        avg_jaccard = class_jaccard_results.get(cls, 0)
        jaccards_dice += f"  Class {cls}: DICE = {avg_dice:.4f}, Jaccard = {avg_jaccard:.4f}\n"
    
    print(f"\n🎯 **Overall Average DICE: {overall_dice:.4f}**")
    print(f"🎯 **Overall Average Jaccard: {overall_jaccard:.4f}**")
    
    return jaccards_dice, overall_dice, overall_jaccard
