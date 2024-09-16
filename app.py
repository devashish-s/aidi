from PIL import Image
from sklearn.preprocessing import MinMaxScaler

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file, get_checkpoint_url
import os, cv2, pdb, glob, csv
import numpy as np
import joblib

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

model_path = './regression_model_height_augmented.pkl'
regression_model = joblib.load(model_path)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "front_pose_model", "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "side_pose_model","model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor_side_pose = DefaultPredictor(cfg)

show_on_image = True
dest_folder = './temp'
os.makedirs(dest_folder, exist_ok=True)

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return round(np.sqrt((x2 - x1)**2 + (y2 - y1)**2), 2)

def calculate_distances_with_check_front_pose(keypoints_distance):
    # Calculate distances
    shoulder_dist = calculate_distance(keypoints_distance['LeftShoulder'], keypoints_distance['RightShoulder'])
    chest_dist = calculate_distance(keypoints_distance['LeftChest'], keypoints_distance['RightChest'])
    waist_dist = calculate_distance(keypoints_distance['LeftWaist'], keypoints_distance['RightWaist'])
    hip_dist = calculate_distance(keypoints_distance['LeftHip'], keypoints_distance['RightHip'])
    height_in_pixels_dist = abs(keypoints_distance['Foot'][1] - keypoints_distance['Head'][1])

    # Store distances in a dictionary for easy checking
    distances = {
        'shoulder_dist': shoulder_dist,
        'chest_dist': chest_dist,
        'waist_dist': waist_dist,
        'hip_dist': hip_dist,
        'height_in_pixels_dist': height_in_pixels_dist
    }

    # Identify distances that are less than 50
    problematic_distances = {key: value for key, value in distances.items() if value < 50}

    # Handle cases based on the number of problematic distances
    if len(problematic_distances) == 1:
        # Use fallback distance for the single problematic one
        for key in problematic_distances:
            print(f"Anomalous distance detected in {key}: {distances[key]}. Using fallback value.")
            if key == 'shoulder_dist':
                distances[key] = round(1.1*distances['chest_dist'], 2)
            elif key == 'chest_dist':
                distances[key] = round(0.9*distances['shoulder_dist'],2) or round(distances['hip_dist'],2)
            elif key == 'waist_dist':
                distances[key] = round(.9*distances['chest_dist'],2) or round(.9*distances['hip_dist'], 2)
            elif key == 'hip_dist':
                distances[key] = round(1.1*distances['waist_dist'], 2)
    elif len(problematic_distances) > 1:
        return None

    return distances

def calculate_distances_with_check_side_pose(keypoints_distance):

    # Calculate distances
    chest_dist = calculate_distance(keypoints_distance['LeftChest'], keypoints_distance['RightChest'])
    waist_dist = calculate_distance(keypoints_distance['LeftWaist'], keypoints_distance['RightWaist'])
    hip_dist = calculate_distance(keypoints_distance['LeftHip'], keypoints_distance['RightHip'])
    height_in_pixels_dist = abs(keypoints_distance['Foot'][1] - keypoints_distance['Head'][1])

    # Store distances in a dictionary for easy checking
    distances = {
        'chest_dist': chest_dist,
        'waist_dist': waist_dist,
        'hip_dist': hip_dist,
        'height_in_pixels_dist': height_in_pixels_dist
    }

    # Identify distances that are less than 50
    problematic_distances = {key: value for key, value in distances.items() if value < 50}

    # Handle cases based on the number of problematic distances
    if len(problematic_distances) == 1:
        # Use fallback distance for the single problematic one
        for key in problematic_distances:
            print(f"Anomalous distance detected in {key}: {distances[key]}. Using fallback value.")
            if key == 'chest_dist':
                distances[key] = round(distances['hip_dist'],2)
            elif key == 'waist_dist':
                distances[key] = round(.9*distances['chest_dist'],2) or round(.9*distances['hip_dist'], 2)
            elif key == 'hip_dist':
                distances[key] = round(1.1*distances['chest_dist'],2)

    return distances



def run_on_front_pose_images(imagePath):

        filename = os.path.basename(imagePath)
        im = cv2.imread(imagePath)[:,:,::-1]
        im = add_padding_and_resize(im)
        # im = cv2.resize(im, (767, 1042))
        # cv2.imshow('as', im); cv2.waitKey(0)
        outputs = predictor(im)

        # Assuming the inference output is stored in a variable named `outputs`
        instances = outputs["instances"]
        keypoints = instances.pred_keypoints if instances.has("pred_keypoints") else None
        pred_boxes = instances.pred_boxes.tensor.numpy().astype("int")[0]
        keypoints = np.asarray(keypoints)

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

            # shoulder_dist = calculate_distance(keypoints_distance['LeftShoulder'], keypoints_distance['RightShoulder'])
            # chest_dist = calculate_distance(keypoints_distance['LeftChest'], keypoints_distance['RightChest'])
            # waist_dist = calculate_distance(keypoints_distance['LeftWaist'], keypoints_distance['RightWaist'])
            # hip_dist = calculate_distance(keypoints_distance['LeftHip'], keypoints_distance['RightHip'])

            all_distances = calculate_distances_with_check_front_pose(keypoints_distance)
            # write_to_csv(output_csv_path, all_distances, filename)


            if show_on_image:

                shoulder_dist = all_distances['shoulder_dist']
                chest_dist = all_distances['chest_dist']
                waist_dist = all_distances['waist_dist']
                hip_dist = all_distances['hip_dist']

                im_bgr = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2RGB)

                image_line = cv2.line(im_bgr, keypoints_distance['LeftShoulder'], keypoints_distance['RightShoulder'], (0, 255, 0), 9)
                cv2.putText(image_line, str(shoulder_dist), keypoints_distance['LeftShoulder'], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                image_line = cv2.line(im_bgr, keypoints_distance['LeftChest'], keypoints_distance['RightChest'], (0, 255, 0), 9)
                cv2.putText(image_line, str(chest_dist), keypoints_distance['LeftChest'], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                image_line = cv2.line(im_bgr, keypoints_distance['LeftWaist'], keypoints_distance['RightWaist'], (0, 255, 0), 9)
                cv2.putText(image_line, str(waist_dist), keypoints_distance['LeftWaist'], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                image_line = cv2.line(im_bgr, keypoints_distance['LeftHip'], keypoints_distance['RightHip'], (0, 255, 0), 9)
                cv2.putText(image_line, str(hip_dist), keypoints_distance['LeftHip'], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
                image_line = cv2.line(im_bgr, keypoints_distance['Head'], keypoints_distance['Foot'], (0, 255, 0), 9)
                cv2.putText(image_line, str(hip_dist), keypoints_distance['Head'], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                
                cv2.imwrite(os.path.join(dest_folder, filename), image_line)

            return all_distances.values()

        else:
            print("No keypoints found in the output.")

        return None

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


def run_on_side_pose_images(imagePath):
    filename = os.path.basename(imagePath)
    im = cv2.imread(imagePath)[:,:,::-1]
    im = add_padding_and_resize(im)
    # im = cv2.resize(im, (767, 1042))

    outputs = predictor_side_pose(im)

    # Assuming the inference output is stored in a variable named `outputs`
    instances = outputs["instances"]
    keypoints = instances.pred_keypoints if instances.has("pred_keypoints") else None
    pred_boxes = instances.pred_boxes.tensor.numpy().astype("int")[0]
    keypoints = np.asarray(keypoints)

    if keypoints is not None:
        # Create a visualizer instance

        all_keypoints = keypoints[0]
        keypoints_distance = {}

        keypoints_distance['LeftChest'] = (int(all_keypoints[3][0]), int(all_keypoints[3][1]))
        keypoints_distance['RightChest'] = (int(all_keypoints[4][0]), int(all_keypoints[4][1]))
        keypoints_distance['LeftWaist'] = (int(all_keypoints[5][0]), int(all_keypoints[5][1]))
        keypoints_distance['RightWaist'] = (int(all_keypoints[6][0]), int(all_keypoints[6][1]))
        keypoints_distance['LeftHip'] = (int(all_keypoints[7][0]), int(all_keypoints[7][1]))
        keypoints_distance['RightHip'] = (int(all_keypoints[8][0]), int(all_keypoints[8][1]))
        keypoints_distance['Head'] = pred_boxes[:2]
        keypoints_distance['Foot'] = pred_boxes[2:]

        # shoulder_dist = calculate_distance(keypoints_distance['LeftShoulder'], keypoints_distance['RightShoulder'])
        # chest_dist = calculate_distance(keypoints_distance['LeftChest'], keypoints_distance['RightChest'])
        # waist_dist = calculate_distance(keypoints_distance['LeftWaist'], keypoints_distance['RightWaist'])
        # hip_dist = calculate_distance(keypoints_distance['LeftHip'], keypoints_distance['RightHip'])

        all_distances = calculate_distances_with_check_side_pose(keypoints_distance)

        if show_on_image:
            # write_to_csv(output_csv_path, all_distances, filename)
            chest_dist = all_distances['chest_dist']
            waist_dist = all_distances['waist_dist']
            hip_dist = all_distances['hip_dist']

            im_bgr = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2RGB)

            image_line = cv2.line(im_bgr, keypoints_distance['LeftChest'], keypoints_distance['RightChest'], (0, 255, 0), 9)
            cv2.putText(image_line, str(chest_dist), keypoints_distance['LeftChest'], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            image_line = cv2.line(im_bgr, keypoints_distance['LeftWaist'], keypoints_distance['RightWaist'], (0, 255, 0), 9)
            cv2.putText(image_line, str(waist_dist), keypoints_distance['LeftWaist'], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            image_line = cv2.line(im_bgr, keypoints_distance['LeftHip'], keypoints_distance['RightHip'], (0, 255, 0), 9)
            cv2.putText(image_line, str(hip_dist), keypoints_distance['LeftHip'], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


            image_line = cv2.line(im_bgr, keypoints_distance['Head'], keypoints_distance['Foot'], (0, 255, 0), 9)
            cv2.putText(image_line, str(hip_dist), keypoints_distance['Head'], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


            cv2.imwrite(os.path.join(dest_folder, filename), image_line)

        return all_distances.values()
    else:
        print("No keypoints found in the output.")
    return None


def app(imagePathFront, imagePathSide):

    distance_front = run_on_front_pose_images(imagePathFront)
    distance_side = run_on_side_pose_images(imagePathSide)
    print(distance_front)
    values = list(distance_front) + list(distance_side)
    print(values)
    return values

def run_regression(values, height):
    values.append(height)
    values = np.array(values)
    print('input:', values)
    input = values.reshape(1, -1)

    y_test_pred = regression_model.predict(input)

    # print(y_test_pred[0])
    Shoulder, Chest, Waist, Hip = y_test_pred[0]
    return round(Shoulder, 2), round(Chest, 2), round(Waist, 2), round(Hip, 2)


front_im1= '/home/nithin/Downloads/measurment/Anurag_front_175cm.jpg'
side_im1= '/home/nithin/Downloads/measurment/Anurag_side_175cm.jpg'

height = 175/2.54
values = app(front_im1, side_im1)
Shoulder,Chest,Waist,Hip = run_regression(values, height)
print(Shoulder,Chest,Waist,Hip)
