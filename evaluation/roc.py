import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def create_roc_curve_from_yolo_metrics(metrics,training_name):
        metric_dict = {name: {} for name in metrics.names.values()}
        
        for i, name in enumerate(metrics.names.values()):
            metric_dict[name]["tp"] = metrics.confusion_matrix.matrix[i][i]
            metric_dict[name]["fp"] = np.sum(metrics.confusion_matrix.matrix[:, i]) - metric_dict[name]["tp"]
            metric_dict[name]["fn"] = np.sum(metrics.confusion_matrix.matrix[i, :]) - metric_dict[name]["tp"]
        
        total_samples = np.sum(metrics.confusion_matrix.matrix)
        
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        for i, (name, values) in enumerate(metric_dict.items()):
            values["tn"] = total_samples - values["tp"] - values["fp"] - values["fn"]
            
            tpr = values["tp"] / (values["tp"] + values["fn"]) if (values["tp"] + values["fn"]) > 0 else 0
            fpr = values["fp"] / (values["fp"] + values["tn"]) if (values["fp"] + values["tn"]) > 0 else 0
            
            fpr_points = [0, fpr, 1]
            tpr_points = [0, tpr, 1]
            
            roc_auc = auc(fpr_points, tpr_points)
            
            plt.plot(fpr_points, tpr_points, color=colors[i % len(colors)], lw=2, 
                     label=f'{name} (AUC = {roc_auc:.2f})')
            
            print(f"Class: {name}")
            print(f"TP: {values['tp']}, FP: {values['fp']}, FN: {values['fn']}, TN: {values['tn']}")
            print(f"TPR: {tpr:.4f}, FPR: {fpr:.4f}, AUC: {roc_auc:.4f}")
            print("---")
        
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-Class ROC Curve from YOLO Metrics')
        plt.legend(loc="lower right")
        
        plt.savefig(f"Results-Yolo-Auto/{training_name}/roc_curve.png")
        return metric_dict