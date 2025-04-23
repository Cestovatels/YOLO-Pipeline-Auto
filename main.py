import os
import cv2
import shutil
import random
import logging
import argparse
import numpy as np
from typing import List
from random import randint
from ultralytics import YOLO
from colorlog import ColoredFormatter
from evaluation.report import create_report
from evaluation.draw import draw_ground_truth
from evaluation.IoUJaccardDice import IoU_Jaccard_Dice
from evaluation.roc import create_roc_curve_from_yolo_metrics



class YoloArguments:
    """Handles argument parsing for YOLO training pipeline."""
    @staticmethod
    def parse() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Dataset preprocessing for YOLO format')
        parser.add_argument('--image_dir', type=str, default='Dataset/images',
                            help='Path to the image directory')
        parser.add_argument('--label_dir', type=str, default='Dataset/labels',
                            help='Path to the label directory')
        parser.add_argument('--classes', type=str, nargs='+', help='Enter class names separated by a space (e.g. cat dog rabbit)', required=True)
        parser.add_argument('--epochs', type=str, required=True)
        parser.add_argument('--batch', type=str, required=True)
        parser.add_argument('--imgsz', type=str, required=True)
        parser.add_argument('--patience', type=int, default=100)
        parser.add_argument('--device', default='')
        parser.add_argument('--model', type=str, required=True, choices=[
            'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
            'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x',
            'yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg',
            'yolo11n-seg', 'yolo11s-seg', 'yolo11m-seg', 'yolo11l-seg', 'yolo11x-seg'
        ])
        parser.add_argument('--conf', type=float, default=0.25)
        parser.add_argument('--iou', type=float, default=0.25)
        parser.add_argument('--agnostic_nms', action='store_true')
        parser.add_argument('--show_labels', default=True, action='store_true')
        parser.add_argument('--show_conf', default=True, action='store_true')
        parser.add_argument('--show_boxes', default=True, action='store_true')
        parser.add_argument('--retina_masks', action='store_true')
        parser.add_argument('--hsv_h', type=float, default=0.015)
        parser.add_argument('--hsv_s', type=float, default=0.7)
        parser.add_argument('--hsv_v', type=float, default=0.4)
        parser.add_argument('--degrees', type=float, default=0.0)
        parser.add_argument('--translate', type=float, default=0.1)
        parser.add_argument('--scale', type=float, default=0.5)
        parser.add_argument('--shear', type=float, default=0.0)
        parser.add_argument('--perspective', type=float, default=0.0)
        parser.add_argument('--flipud', type=float, default=0.0)
        parser.add_argument('--fliplr', type=float, default=0.5)
        parser.add_argument('--bgr', type=float, default=0.0)
        parser.add_argument('--mosaic', type=float, default=1.0)
        parser.add_argument('--mixup', type=float, default=0.0)
        parser.add_argument('--copy_paste', type=float, default=0.0)
        parser.add_argument('--copy_paste_mode', type=str, default='flip')
        parser.add_argument('--auto_augment', type=str, default='randaugment')
        parser.add_argument('--erasing', type=float, default=0.4)
        parser.add_argument('--crop_fraction', type=float, default=1.0)
        parser.add_argument('--overlap_mask', action='store_true')
        parser.add_argument('--task', type=str, required=True, choices=['segmentation', 'detection'])
        parser.add_argument('--training_name', type=str, required=True)
        parser.add_argument('--multi_to_single', action='store_true')
        return parser.parse_args()
        
def setup_logger():
    """Set up a color logger for console output."""
    formatter = ColoredFormatter(
        "%(log_color)s[%(levelname)s] %(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'blue',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger

logger = setup_logger()

class YoloPreprocessor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base_path = f"Results-Yolo-Auto/{args.training_name}/Dataset_{args.training_name}"

    def create_folders(self):
        logger.info("Creating folders...")
        os.makedirs("Results-Yolo-Auto", exist_ok=True)
        os.makedirs(f"Results-Yolo-Auto/{self.args.training_name}", exist_ok=True)
        os.makedirs(self.base_path, exist_ok=True)

    def create_yaml_file(self):
        logger.info("Creating YAML configuration...")
        current_path = os.getcwd()
        content = (
            f'train: {current_path}/{self.base_path}/train/images\n'
            f'val: {current_path}/{self.base_path}/valid/images\n'
            f'test: {current_path}/{self.base_path}/test/images\n'
            f'nc: {len(self.args.classes)}\n'
            f"names: {self.args.classes}"
        )
        yaml_path = os.path.join(self.base_path, 'custom.yaml')
        with open(yaml_path, 'w') as f:
            f.write(content)

    def create_yolo_dataset(self, image_dir, label_dir, train_ratio=0.8, test_ratio=0.1, valid_ratio=0.1):
        logger.info("Creating YOLO dataset structure...")
        for subset in ['train', 'test', 'valid']:
            os.makedirs(os.path.join(self.base_path, subset, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.base_path, subset, 'labels'), exist_ok=True)

        images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
        labels = [f.rsplit('.', 1)[0] + '.txt' for f in images]
        combined = list(zip(images, labels))
        random.shuffle(combined)

        total = len(combined)
        train_split = int(train_ratio * total)
        test_split = int(test_ratio * total)
        train_files = combined[:train_split]
        test_files = combined[train_split:train_split + test_split]
        valid_files = combined[train_split + test_split:]

        def move_files(file_list, subset):
            for image_file, label_file in file_list:
                src_img = os.path.join(image_dir, image_file)
                src_lbl = os.path.join(label_dir, label_file)
                dst_img = os.path.join(self.base_path, subset, 'images', image_file)
                dst_lbl = os.path.join(self.base_path, subset, 'labels', label_file)
                if os.path.exists(src_img) and os.path.exists(src_lbl):
                    shutil.copy(src_img, dst_img)
                    shutil.copy(src_lbl, dst_lbl)

        move_files(train_files, 'train')
        move_files(test_files, 'test')
        move_files(valid_files, 'valid')
        logger.info("Dataset split and copied successfully.")
    
    
    def start_training(self):
        
        logger.info("Training begins...")
        devices = [int(d) for d in self.args.device.split(",")] if self.args.device else self.args.device
        model = YOLO(self.args.model)
        dataset_path = f"Results-Yolo-Auto/{self.args.training_name}/Dataset_{self.args.training_name}/custom.yaml"

        
        model.train(
            data=dataset_path,
            epochs=int(self.args.epochs),
            imgsz=int(self.args.imgsz),
            project=f"Results-Yolo-Auto/{self.args.training_name}",
            name=f"Train-{self.args.training_name}",
            batch=int(self.args.batch),
            patience=self.args.patience,
            device=devices,
            overlap_mask=self.args.overlap_mask,
            fliplr=self.args.fliplr
            single_cls=self.args.multi_to_single
        )
        logger.info("Training Done...")
        logger.info("Validation On Test Data...")
        metrics = model.val(
            data=dataset_path,
            split="test",
            imgsz=int(self.args.imgsz),
            project=f"Results-Yolo-Auto/{self.args.training_name}",
            name=f"Test-Validation-{self.args.training_name}"
            single_cls=self.args.multi_to_single
        )
        logger.info("Completed validation on test data...")
        logger.info("Test data forecasting process begins...")
        results = model(
            f"Results-Yolo-Auto/{self.args.training_name}/Dataset_{self.args.training_name}/test/images",
            save=True,
            imgsz=int(self.args.imgsz),
            name=f"Pred-{self.args.training_name}",
            device=devices,
            conf=self.args.conf,
            iou=self.args.iou,
            agnostic_nms=self.args.agnostic_nms,
            show_labels=self.args.show_labels,
            show_conf=self.args.show_labels,
            show_boxes=self.args.show_boxes,
            retina_masks=self.args.retina_masks,
            stream=True,
            save_txt=True
        )
        
        logger.info("Test data forecasting process done...")
        for result in results:
            continue

        create_roc_curve_from_yolo_metrics(metrics, self.args.training_name)
        draw_ground_truth(self.args.training_name, self.args.task, self.args.classes, self.args.model)
        jaccards_dice, overall_dice, overall_jaccard, IoU, overall_IoU = IoU_Jaccard_Dice(self.args.model, self.args.training_name,self.args.task)
        create_report(metrics, self.args.task, self.args.training_name, self.args.model, int(self.args.epochs), int(self.args.imgsz), jaccards_dice, overall_dice, overall_jaccard, IoU, overall_IoU)


def main():
    args = YoloArguments.parse()
    logger.info("Preprocessing starts...")
    processor = YoloPreprocessor(args)
    processor.create_folders()
    processor.create_yaml_file()
    processor.create_yolo_dataset(args.image_dir, args.label_dir)
    logger.info("Preprocessing completed successfully.")
    processor.start_training()
    logger.info("Training and Testing completed successfully.")
if __name__ == "__main__":
    main()
