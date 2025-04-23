from evaluation.metrics.DiceJaccardSeg import process_all_files as Seg_process_all_files_Dice_Jaccard
from evaluation.metrics.DiceJaccardDet import process_all_files as Det_process_all_files_Dice_Jaccard
from evaluation.metrics.DiceJaccardDet2Seg import process_all_files as Det2Seg_process_all_files_Dice_Jaccard

from evaluation.metrics.SegIoU import process_all_files as Seg_process_all_files_IoU
from evaluation.metrics.DetIoU import process_all_files as Det_process_all_file_IoU
from evaluation.metrics.IoUDet2Seg import process_all_files as Det2Seg_process_all_file_IoU



def IoU_Jaccard_Dice(model,training_name,task):
        is_seg_model = "-seg" in model.lower()
        is_seg_task = task == 'segmentation'
        test_folder = f"Results-Yolo-Auto/{training_name}/Dataset_{training_name}/test/labels"
        predict_folder = f"Results-Yolo-Auto/{training_name}/Pred-{training_name}/labels"

        if is_seg_task and is_seg_model:  # For direct segmentation model and segmentation operations.
            jaccards_dice, overall_dice, overall_jaccard = Seg_process_all_files_Dice_Jaccard(test_folder, predict_folder)
            overall_IoU, IoU = Seg_process_all_files_IoU(test_folder, predict_folder)

        elif is_seg_task and not is_seg_model: # If Detection operations are performed on segmentation data.
            jaccards_dice, overall_dice, overall_jaccard = Det2Seg_process_all_files_Dice_Jaccard(test_folder, predict_folder)
            overall_IoU, IoU = Det2Seg_process_all_file_IoU(test_folder, predict_folder)

        else: # For direct detection model and detection operations.
            jaccards_dice, overall_dice, overall_jaccard = Det_process_all_files_Dice_Jaccard(test_folder, predict_folder)
            overall_IoU, IoU = Det_process_all_file_IoU(test_folder, predict_folder)

        return jaccards_dice, overall_dice, overall_jaccard, IoU, overall_IoU 