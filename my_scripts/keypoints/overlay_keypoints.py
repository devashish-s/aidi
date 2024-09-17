import os
import cv2
import json
import numpy as np


def draw_keypoints(image, keypoints, color=(0, 255, 0), radius=3, thickness=2):
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        if v > 0:  # Visible keypoint
            cv2.circle(image, (int(x), int(y)), radius, color, thickness)
    return image


def visualize_keypoints(json_file, images_folder, output_folder):
    print("Start")
    with open(json_file) as f:
        keypoints_data = json.load(f)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_info in keypoints_data['images'][:50]:
        image_id = image_info['id']
        image_filename = image_info['file_name']
        image_path = os.path.join(images_folder, image_filename)

        image = cv2.imread(image_path)

        # Find corresponding annotation
        annotations = [annot for annot in keypoints_data['annotations'] if annot['image_id'] == image_id]

        for annot in annotations:
            keypoints = annot['keypoints']
            try:
                image_with_keypoints = draw_keypoints(image.copy(), keypoints)
            except:
                print("imagePath: ", image_path)

            output_image_path = os.path.join(output_folder, f"keypoints_{image_id}.jpg")
            cv2.imwrite(output_image_path, image_with_keypoints)
            print(f"Saved: {output_image_path}")


if __name__ == "__main__":
    # json_file = "/media/nithin/17b8fbe0-a0a3-46d6-bb0e-d8bebe7bcbd0/experimental/KeypointsRepos/keypoint-detection/dataset/augmented_side/train_augmented_side_keypoints.json"  # Replace with your JSON path
    # images_folder = "/media/nithin/17b8fbe0-a0a3-46d6-bb0e-d8bebe7bcbd0/experimental/KeypointsRepos/keypoint-detection/dataset/augmented_side/train_side_pose"  # Replace with your images folder path
    # output_folder = "output_keypoints_visualization"
    json_file = "/workspace/ai-tailer-detectron/input_data/data_kps.json"  # Replace with your JSON path
    images_folder = "/workspace/ai-tailer-detectron/input_data/image_processed/"  # Replace with your images folder path
    # json_file = "/media/nithin/17b8fbe0-a0a3-46d6-bb0e-d8bebe7bcbd0/experimental/KeypointsRepos/keypoint-detection/dataset/augmented_side/train_augmented_side_keypoints.json"  # Replace with your JSON path
    # images_folder = "/media/nithin/17b8fbe0-a0a3-46d6-bb0e-d8bebe7bcbd0/experimental/KeypointsRepos/keypoint-detection/dataset/augmented_side/train_side_pose"  # Replace with your images folder path
    output_folder = "/workspace/ai-tailer-detectron/output_keypoints_visualization"

    visualize_keypoints(json_file, images_folder, output_folder)
