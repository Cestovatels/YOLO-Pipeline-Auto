import os
import cv2
import numpy as np
from tqdm import tqdm
from random import randint


def draw_ground_truth(training_name, task, classes, model):
    print('[INFO] Creating show folders...')
    color_dict = {}

    test_images_dir = f"Results-Yolo-Auto/{training_name}/Dataset_{training_name}/test/images"
    test_labels_dir = f"Results-Yolo-Auto/{training_name}/Dataset_{training_name}/test/labels"
    pred_dir = f"Results-Yolo-Auto/{training_name}/Pred-{training_name}"

    images = os.listdir(test_images_dir)
    line_width = 3

    for image in tqdm(images, desc="Processing Images"):
        # print(image)
        img = cv2.imread(os.path.join(test_images_dir, image))
        if img is None:
            print(f"[WARNING] Could not read image {image}, skipping.")
            continue

        h, w, _ = img.shape
        font_scale = max(w, h) / 1500
        thickness = int(font_scale * 2)
        lw = line_width or max(round(sum(img.shape) / 2 * 0.003), 2)
        tf = max(lw - 1, 1)

        label_path = os.path.join(test_labels_dir, os.path.splitext(image)[0] + ".txt")
        if not os.path.exists(label_path):
            print(f"[WARNING] Label not found for {image}")
            continue

        with open(label_path, 'r') as f:
            labels = f.readlines()

        for i, label in enumerate(labels):
            clss_id = int(label.split()[0])
            clss = classes[clss_id]
            bimg = np.zeros([h, w, 3], dtype=np.uint8)

            if clss not in color_dict:
                color_dict[clss] = (randint(0, 125), randint(0, 125), randint(0, 125))

            is_seg_model = "-seg" in model.lower()
            is_seg_task = task == 'segmentation'
            coords = list(map(float, label.split()[1:]))

            if is_seg_task and is_seg_model:
                coord_list = np.array([[int(coords[i] * w), int(coords[i + 1] * h)] for i in range(0, len(coords), 2)])
                bimg = cv2.fillPoly(bimg, [coord_list], color_dict[clss])
                img = cv2.addWeighted(bimg, 1, img, 1, 1)
            elif is_seg_task and not is_seg_model:
                x_coords = [coords[i] * w for i in range(0, len(coords), 2)]
                y_coords = [coords[i] * h for i in range(1, len(coords), 2)]
                xmin, xmax = int(min(x_coords)), int(max(x_coords))
                ymin, ymax = int(min(y_coords)), int(max(y_coords))
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_dict[clss], line_width, cv2.LINE_AA)
            else:
                if len(coords) >= 4:
                    xcenter, ycenter, width, height = coords[:4]
                    xmin = int((xcenter - width / 2) * w)
                    xmax = int((xcenter + width / 2) * w)
                    ymin = int((ycenter - height / 2) * h)
                    ymax = int((ycenter + height / 2) * h)

                    # Draw bounding box
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_dict[clss], line_width, cv2.LINE_AA)

                    # Draw label above box
                    cv2.putText(
                        img, clss,
                        (xmin, max(ymin - 10, 10)),  # To the top, don't overflow the border
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color_dict[clss],
                        thickness,
                        cv2.LINE_AA
                    )

        pred_image_path = os.path.join(pred_dir, image)
        if os.path.exists(pred_image_path):
            predict_img = cv2.imread(pred_image_path)
            if predict_img is not None:
                result_img = np.concatenate((img, predict_img), axis=0)
                cv2.imwrite(pred_image_path, result_img)
            else:
                print(f"[WARNING] Prediction image {image} not found or unreadable.")
        else:
            print(f"[WARNING] Prediction image {image} not found.")
