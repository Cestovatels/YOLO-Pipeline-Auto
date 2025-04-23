import os
import numpy as np
from collections import defaultdict
from shapely.geometry import box

def load_labels(file_path, is_segmentation=False):
    """Reads YOLO bbox or segmentation data and returns a class-based list of Shapely boxes"""
    shapes = defaultdict(list)
    if not os.path.exists(file_path):
        return shapes
    
    with open(file_path, "r") as f:
        for line in f.readlines():
            data = list(map(float, line.strip().split()))
            cls = int(data[0])

            if is_segmentation and len(data) > 5:
                # Convert segmentation polygons to bounding boxes
                xs = data[1::2]
                ys = data[2::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
            else:
                # Standard YOLO bbox format
                cx, cy, w, h = data[1:5]
                x_min, y_min = cx - w / 2, cy - h / 2
                x_max, y_max = cx + w / 2, cy + h / 2

            shapes[cls].append(box(x_min, y_min, x_max, y_max))  # Store as Shapely box
    return shapes

def dice_score(boxA, boxB):
    """Calculates the DICE (F1-score) between two boxes"""
    if not boxA.is_valid or not boxB.is_valid:
        return 0
    
    intersection = boxA.intersection(boxB).area
    total_area = boxA.area + boxB.area
    return (2 * intersection) / total_area if total_area > 0 else 0

def jaccard_score(boxA, boxB):
    """Calculates the Jaccard (IoU) between two boxes"""
    if not boxA.is_valid or not boxB.is_valid:
        return 0
    
    intersection = boxA.intersection(boxB).area
    union = boxA.union(boxB).area
    return intersection / union if union > 0 else 0

def match_boxes(gt_boxes, pred_boxes):
    """Matches boxes by class and calculates DICE and Jaccard metrics"""
    dice_scores_per_class = defaultdict(list)
    jaccard_scores_per_class = defaultdict(list)
    
    for cls in gt_boxes:
        if cls in pred_boxes:
            for gt_box in gt_boxes[cls]:
                best_dice = 0
                best_jaccard = 0
                for pred_box in pred_boxes[cls]:
                    dice_val = dice_score(gt_box, pred_box)
                    print("*****Dice******", dice_val)
                    jaccard_val = jaccard_score(gt_box, pred_box)
                    print("*****Jaccard******", jaccard_val)
                    best_dice = max(best_dice, dice_val)
                    best_jaccard = max(best_jaccard, jaccard_val)
                dice_scores_per_class[cls].append(best_dice)
                jaccard_scores_per_class[cls].append(best_jaccard)
    
    return dice_scores_per_class, jaccard_scores_per_class

def process_all_files(test_folder, predict_folder):
    """Compares all .txt files and computes per-class average DICE and Jaccard scores"""
    test_files = set(f for f in os.listdir(test_folder) if f.endswith(".txt"))
    predict_files = set(f for f in os.listdir(predict_folder) if f.endswith(".txt"))

    common_files = test_files.intersection(predict_files)  # Match files with same names
    dice_per_class = defaultdict(list)
    jaccard_per_class = defaultdict(list)

    for file in common_files:
        gt_boxes = load_labels(os.path.join(test_folder, file), is_segmentation=True)
        pred_boxes = load_labels(os.path.join(predict_folder, file), is_segmentation=False)
        dice_scores, jaccard_scores = match_boxes(gt_boxes, pred_boxes)

        for cls, scores in dice_scores.items():
            dice_per_class[cls].extend(scores)
        for cls, scores in jaccard_scores.items():
            jaccard_per_class[cls].extend(scores)

    # Calculate average DICE and Jaccard per class
    class_dice_results = {cls: np.mean(scores) if scores else 0 for cls, scores in dice_per_class.items()}
    class_jaccard_results = {cls: np.mean(scores) if scores else 0 for cls, scores in jaccard_per_class.items()}

    # Calculate overall average DICE and Jaccard
    overall_dice = np.mean([dice for scores in dice_per_class.values() for dice in scores]) if dice_per_class else 0
    overall_jaccard = np.mean([jaccard for scores in jaccard_per_class.values() for jaccard in scores]) if jaccard_per_class else 0
    jaccards_dice = ""
    print("\n📌 **Class-Based Metrics:**")
    for cls in sorted(set(class_dice_results.keys()).union(class_jaccard_results.keys())):
        avg_dice = class_dice_results.get(cls, 0)
        avg_jaccard = class_jaccard_results.get(cls, 0)
        jaccards_dice += f"  Class {cls}: DICE = {avg_dice:.4f}, Jaccard = {avg_jaccard:.4f}\n"

        print(f"  Class {cls}: DICE = {avg_dice:.4f}, Jaccard = {avg_jaccard:.4f}")
        
    print(f"\n🎯 **Overall Average DICE: {overall_dice:.4f}**")
    print(f"🎯 **Overall Average Jaccard: {overall_jaccard:.4f}**")

    return jaccards_dice, overall_dice, overall_jaccard
