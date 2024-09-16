import tqdm

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file, get_checkpoint_url
import os, cv2, pdb, glob, csv, pandas as pd
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage, Keypoint
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


import warnings
warnings.filterwarnings("ignore")


cfg = get_cfg()
cfg.merge_from_file(get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train_keypoints",)
cfg.DATASETS.TEST = ("val_keypoints",)
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 12
cfg.SOLVER.BASE_LR = 0.00025

cfg.SOLVER.CHECKPOINT_PERIOD = 2000
cfg.SOLVER.MAX_ITER = 10000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of classes (excluding background)
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 13  # Number of keypoints

# Set input size to 480x640
cfg.INPUT.MIN_SIZE_TRAIN = (640,)
cfg.INPUT.MAX_SIZE_TRAIN = 480
cfg.INPUT.MIN_SIZE_TEST = 640
cfg.INPUT.MAX_SIZE_TEST = 480

cfg.OUTPUT_DIR = "./output"
cfg.MODEL.DEVICE = "cpu"

keypoint_names = [
    "Head", "LeftShoulder", "RightShoulder", "LeftChest", "RightChest",
    "LeftWaist", "RightWaist", "LeftHighHip", "RightHighHip",
    "LeftHip", "RightHip", "LeftFoot", "RightFoot"
]

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "front_pose_model", "model_final.pth")  # path to the model we just trained for front
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor_front_pose = DefaultPredictor(cfg)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "side_pose_model", "model_final.pth")  # path to the model we just trained for side
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor_side_pose = DefaultPredictor(cfg)



def write_to_csv(output_csv_path, distances):
    df = pd.DataFrame([distances])
    if not os.path.exists(output_csv_path):
        df.to_csv(output_csv_path, index=False, header=True)
    else:
        df.to_csv(output_csv_path, mode='a', index=False, header=False)



def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return round(np.sqrt((x2 - x1)**2 + (y2 - y1)**2), 2)

def calculate_distances_with_check_front_pose(keypoints_distance, filename="", is_aug=False):
    # Calculate distances
    shoulder_dist = calculate_distance(keypoints_distance['LeftShoulder'], keypoints_distance['RightShoulder'])
    chest_dist = calculate_distance(keypoints_distance['LeftChest'], keypoints_distance['RightChest'])
    waist_dist = calculate_distance(keypoints_distance['LeftWaist'], keypoints_distance['RightWaist'])
    hip_dist = calculate_distance(keypoints_distance['LeftHip'], keypoints_distance['RightHip'])
    height_in_pixels_dist = abs(keypoints_distance['Foot'][1] - keypoints_distance['Head'][1])

    # Store distances in a dictionary for easy checking
    distances = {
        f'{pose_name}_shoulder_dist': shoulder_dist,
        f'{pose_name}_chest_dist': chest_dist,
        f'{pose_name}_waist_dist': waist_dist,
        f'{pose_name}_hip_dist': hip_dist,
        f'{pose_name}_height_in_pixels_dist': height_in_pixels_dist
    }

    # Identify distances that are less than 50
    problematic_distances = {key: value for key, value in distances.items() if value < 50}

    # Handle cases based on the number of problematic distances
    if len(problematic_distances) == 1 and not is_aug:
        print("problematic_distances___ ", filename)
        # Use fallback distance for the single problematic one
        for key in problematic_distances:
            print(f"Anomalous distance detected in {key}: {distances[key]}. Using fallback value.")
            if key == f'{pose_name}_shoulder_dist':
                distances[key] = round(1.1*distances[f'{pose_name}_chest_dist'], 2)
            elif key == f'{pose_name}_chest_dist':
                distances[key] = round(0.9*distances[f'{pose_name}_shoulder_dist'],2) or round(distances[f'{pose_name}_hip_dist'],2)
            elif key == f'{pose_name}_waist_dist':
                distances[key] = round(.9*distances[f'{pose_name}_chest_dist'],2) or round(.9*distances[f'{pose_name}_hip_dist'], 2)
            elif key == f'{pose_name}_hip_dist':
                distances[key] = round(1.1*distances[f'{pose_name}_waist_dist'],2)
    return distances

def calculate_distances_with_check_side_pose(keypoints_distance, filename="", is_aug=False):

    # Calculate distances
    chest_dist = calculate_distance(keypoints_distance['LeftChest'], keypoints_distance['RightChest'])
    waist_dist = calculate_distance(keypoints_distance['LeftWaist'], keypoints_distance['RightWaist'])
    hip_dist = calculate_distance(keypoints_distance['LeftHip'], keypoints_distance['RightHip'])
    height_in_pixels_dist = abs(keypoints_distance['Foot'][1] - keypoints_distance['Head'][1])

    # Store distances in a dictionary for easy checking
    distances = {
        f'{pose_name}_chest_dist': chest_dist,
        f'{pose_name}_waist_dist': waist_dist,
        f'{pose_name}_hip_dist': hip_dist,
        f'{pose_name}_height_in_pixels_dist': height_in_pixels_dist
    }

    # Identify distances that are less than 50
    problematic_distances = {key: value for key, value in distances.items() if value < 50}

    # Handle cases based on the number of problematic distances
    if len(problematic_distances) == 1 and not is_aug:
        print("problematic_distances___ ", filename)
        # Use fallback distance for the single problematic one
        for key in problematic_distances:
            print(f"Anomalous distance detected in {key}: {distances[key]}. Using fallback value.")
            if key == f'{pose_name}_chest_dist':
                distances[key] = round(distances[f'{pose_name}_hip_dist'],2)
            elif key == f'{pose_name}_waist_dist':
                distances[key] = round(.9*distances[f'{pose_name}_chest_dist'],2) or round(.9*distances[f'{pose_name}_hip_dist'], 2)
            elif key == f'{pose_name}_hip_dist':
                distances[key] = round(1.1*distances[f'{pose_name}_chest_dist'],2)

    return distances


def add_padding_and_resize(image, target_width=720, target_height=1280):
    image = Image.fromarray(image)
    original_width, original_height = image.size

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height

    # Determine the new size while maintaining the aspect ratio
    if aspect_ratio > (target_width / target_height):  # Wider image
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:  # Taller image
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new image with the target size and white background
    new_image = Image.new("RGB", (target_width, target_height), (255, 255, 255))

    # Paste the resized image onto the new image, centered
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return np.array(new_image)


def scaleNrotate(im, keypoints, pred_boxes):

    # Assuming keypoints are in the format [num_keypoints, 3] where last dimension is (x, y, visibility)
    keypoints = keypoints[0, :, :2]  # Extract x, y coordinates

    # Convert keypoints to imgaug format
    keypoints_on_image = KeypointsOnImage([Keypoint(x=kp[0], y=kp[1]) for kp in keypoints], shape=im.shape)

    # Convert bounding box to imgaug format
    bounding_box = BoundingBox(x1=pred_boxes[0], y1=pred_boxes[1], x2=pred_boxes[2], y2=pred_boxes[3])
    bboxes_on_image = BoundingBoxesOnImage([bounding_box], shape=im.shape)

    # Define augmentation pipeline
    seq = iaa.Sequential([
        iaa.Affine(scale=(0.6, 1.2), cval=255),

        # Scale between 60% to 110% and rotate between -45 to 45 degrees
    ])

    # Apply augmentation to image, keypoints, and bounding boxes
    image_aug, keypoints_aug, bboxes_aug = seq(image=im, keypoints=keypoints_on_image, bounding_boxes=bboxes_on_image,)

    # Convert keypoints back to array format
    keypoints_aug_array = np.array([[[kp.x, kp.y] for kp in keypoints_aug.keypoints]])

    # Convert bounding boxes back to array format
    bboxes_aug_array = bboxes_aug.bounding_boxes[0].coords.astype(int).flatten()

    # Visualize augmented image with keypoints and bounding box (optional)
    # for kp in keypoints_aug.keypoints:
    #     cv2.circle(image_aug, (int(kp.x), int(kp.y)), 5, (255, 0, 0), -1)
    #
    # cv2.rectangle(image_aug, (bboxes_aug_array[0][0], bboxes_aug_array[0][1]),
    #               (bboxes_aug_array[1][0], bboxes_aug_array[1][1]), (0, 255, 0), 2)

    # cv2.imshow('Augmented Image with Keypoints and BBox', cv2.resize(image_aug, None, fx=0.7, fy=0.7))
    # cv2.waitKey(0)

    return image_aug, keypoints_aug_array, bboxes_aug_array


def keypoints2csv(im, keypoints, pred_boxes, filename, is_aug=False):
    if keypoints is not None:

        # Create a visualizer instance
        all_keypoints = keypoints[0]
        keypoints_distance = {}

        keypoints_distance['LeftShoulder'] = (int(all_keypoints[1][0]), int(all_keypoints[1][1]))
        keypoints_distance['RightShoulder'] = (int(all_keypoints[2][0]), int(all_keypoints[2][1]))
        keypoints_distance['LeftChest'] = (int(all_keypoints[3][0]), int(all_keypoints[3][1]))
        keypoints_distance['RightChest'] = (int(all_keypoints[4][0]), int(all_keypoints[4][1]))
        keypoints_distance['LeftWaist'] = (int(all_keypoints[5][0]), int(all_keypoints[5][1]))
        keypoints_distance['RightWaist'] = (int(all_keypoints[6][0]), int(all_keypoints[6][1]))
        keypoints_distance['LeftHip'] = (int(all_keypoints[7][0]), int(all_keypoints[7][1]))
        keypoints_distance['RightHip'] = (int(all_keypoints[8][0]), int(all_keypoints[8][1]))
        keypoints_distance['Head'] = pred_boxes[:2]
        keypoints_distance['Foot'] = pred_boxes[2:]

        if pose_name == "front":
            all_distances = calculate_distances_with_check_front_pose(keypoints_distance, filename=filename, is_aug=is_aug)
            shoulder_dist = all_distances[f'{pose_name}_shoulder_dist']

        else:
            all_distances = calculate_distances_with_check_side_pose(keypoints_distance, filename=filename, is_aug=is_aug)
            shoulder_dist = ""

        all_distances["person_name"] = filename.split("_")[0]
        write_to_csv(output_csv_path, all_distances)

        chest_dist = all_distances[f'{pose_name}_chest_dist']
        waist_dist = all_distances[f'{pose_name}_waist_dist']
        hip_dist = all_distances[f'{pose_name}_hip_dist']
        height_in_pixels_dist = all_distances[f'{pose_name}_height_in_pixels_dist']

        if vis_image:
            im_bgr = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2RGB)

            image_line = cv2.line(im_bgr, keypoints_distance['LeftShoulder'], keypoints_distance['RightShoulder'],
                                  (0, 255, 0), 9)
            cv2.putText(image_line, str(shoulder_dist), keypoints_distance['LeftShoulder'], cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)

            image_line = cv2.line(im_bgr, keypoints_distance['LeftChest'], keypoints_distance['RightChest'],
                                  (0, 255, 0), 9)
            cv2.putText(image_line, str(chest_dist), keypoints_distance['LeftChest'], cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)

            image_line = cv2.line(im_bgr, keypoints_distance['LeftWaist'], keypoints_distance['RightWaist'],
                                  (0, 255, 0), 9)
            cv2.putText(image_line, str(waist_dist), keypoints_distance['LeftWaist'], cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)

            image_line = cv2.line(im_bgr, keypoints_distance['LeftHip'], keypoints_distance['RightHip'], (0, 255, 0), 9)
            cv2.putText(image_line, str(hip_dist), keypoints_distance['LeftHip'], cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)

            image_line = cv2.line(im_bgr, keypoints_distance['Head'], keypoints_distance['Foot'], (0, 255, 0), 9)
            cv2.putText(image_line, str(height_in_pixels_dist), keypoints_distance['Head'], cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imwrite(os.path.join(dest_vis_folder, filename), image_line)
    else:
        print("No keypoints found in the output.")



def run_on_pose_images(imagePaths, num_augs=10):
    for index_, imagePath in enumerate(tqdm.tqdm(imagePaths[:])):

        filename = os.path.basename(imagePath)
        im = cv2.imread(imagePath)[:,:,::-1]
        im = add_padding_and_resize(im)

        # im = cv2.resize(im, (767, 1042))
        if pose_name == "front":
            outputs = predictor_front_pose(im)
        else:
            outputs = predictor_side_pose(im)

        # Assuming the inference output is stored in a variable named `outputs`
        instances = outputs["instances"]
        keypoints = instances.pred_keypoints if instances.has("pred_keypoints") else None
        pred_boxes = instances.pred_boxes.tensor.numpy().astype("int")[0]
        keypoints = np.asarray(keypoints)


        keypoints2csv(im, keypoints, pred_boxes, filename)
        for i in range(num_augs):
            im_aug_np, keypoints_aug_np, pred_boxes_aug_np = scaleNrotate(im, keypoints, pred_boxes)
            keypoints2csv(im_aug_np, keypoints_aug_np, pred_boxes_aug_np, filename, is_aug=True)



if __name__ == "__main__":
    pose_name = "side"         # pose_name: front/side
    vis_image = True

    imagePaths = glob.glob(f"training_datasets/datasets/Female Customers_data/{pose_name}/*")
    output_csv_path = f'training_datasets/datasets/Female Customers_data/{pose_name}_distances.csv'
    dest_vis_folder = f'training_datasets/datasets/Female Customers_data/{pose_name}_vis'
    os.makedirs(dest_vis_folder, exist_ok=True)

    run_on_pose_images(imagePaths)
